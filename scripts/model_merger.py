# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.distributed._tensor import DTensor, Placement, Shard
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
)


def merge_by_placement(tensors: list[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def upload_model_to_huggingface(local_path: str, remote_path: str):
    # Push to hugging face
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=remote_path, private=False, exist_ok=True)
    api.upload_folder(repo_id=remote_path, folder_path=local_path, repo_type="model")


def _load_missing_keys_from_base(
    base_model_path: str,
    missing_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Stream the base model's safetensors shards and pick out only the
    parameters listed in ``missing_keys``. Used to backfill frozen modules
    (e.g. ``visual.*`` when freeze_vision_tower=True) that never end up in
    the FSDP per-rank state_dict."""
    from safetensors.torch import load_file as _safe_load

    if not missing_keys:
        return {}

    shard_paths = sorted(
        os.path.join(base_model_path, name)
        for name in os.listdir(base_model_path)
        if name.endswith(".safetensors")
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"--base_model_path={base_model_path} has no .safetensors files; "
            "cannot backfill frozen weights."
        )

    found: dict[str, torch.Tensor] = {}
    remaining = set(missing_keys)
    for path in shard_paths:
        if not remaining:
            break
        shard = _safe_load(path, device="cpu")
        hit = remaining & shard.keys()
        for k in hit:
            found[k] = shard[k].to(torch.bfloat16).contiguous()
        remaining -= hit
        del shard
    if remaining:
        print(
            f"[merger] WARNING: {len(remaining)} keys still missing after "
            f"scanning base model. First 5: {sorted(list(remaining))[:5]}"
        )
    print(f"[merger] backfilled {len(found)} keys from base model.")
    return found


def _copy_hf_metadata_from_base(base_model_path: str, hf_path: str) -> None:
    """Copy non-weight HF files needed to load config/tokenizer/processor.

    Some thinned FSDP checkpoints keep only ``model_world_size_*`` shards and may
    not preserve ``actor/huggingface/config.json``. The merger still needs those
    metadata files before it can instantiate the architecture and write the
    merged weights.
    """
    os.makedirs(hf_path, exist_ok=True)
    for name in os.listdir(base_model_path):
        # Do not copy base weights or stale shard indexes into the output dir.
        if (
            name.endswith((".safetensors", ".bin", ".pt"))
            or name.startswith("model-")
            or name == "model.safetensors.index.json"
        ):
            continue
        src = os.path.join(base_model_path, name)
        dst = os.path.join(hf_path, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", required=True, type=str, help="The path for your saved model")
    parser.add_argument("--hf_upload_path", default=False, type=str, help="The path of the huggingface repo to upload")
    parser.add_argument(
        "--base_model_path",
        default=None,
        type=str,
        help="Optional path to the original HF base model. If provided, any "
        "architecture parameters missing from the FSDP shards (e.g. the "
        "vision tower when freeze_vision_tower=True) are pulled from here. "
        "If omitted, the merger auto-discovers it from "
        "<local_dir>/huggingface/config.json's `_name_or_path`.",
    )
    args = parser.parse_args()
    local_dir: str = args.local_dir

    assert not local_dir.endswith("huggingface"), "The local_dir should not end with huggingface."

    # copy rank zero to find the shape of (dp, fsdp)
    rank = 0
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break

    assert world_size, "No model file with the proper format."

    rank0_weight_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    state_dict = torch.load(rank0_weight_path, map_location="cpu", weights_only=False)
    pivot_key = sorted(state_dict.keys())[0]
    weight = state_dict[pivot_key]
    if isinstance(weight, DTensor):
        # get sharding info
        device_mesh = weight.device_mesh
        mesh = device_mesh.mesh
        mesh_dim_names = device_mesh.mesh_dim_names
    else:
        # for non-DTensor
        mesh = np.array([int(world_size)], dtype=np.int64)
        mesh_dim_names = ("fsdp",)

    print(f"Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}")

    assert mesh_dim_names in (("fsdp",), ("ddp", "fsdp")), f"Unsupported mesh_dim_names {mesh_dim_names}."

    if "tp" in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f"Processing {total_shards} model shards in total.")
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank, model_state_dict_lst):
        model_path = os.path.join(local_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank, model_state_dict_lst)

    state_dict: dict[str, list[torch.Tensor]] = {}
    param_placements: dict[str, list[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except Exception:
                print(f"Cannot find key {key} in rank {rank}.")

            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # replicated placement at ddp dimension can be discarded
                if mesh_dim_names[0] == "ddp":
                    placements = placements[1:]

                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key].append(tensor.bfloat16())

    del model_state_dict_lst

    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue

        if key in param_placements:
            # merge shards
            placements: tuple[Shard] = param_placements[key]
            if len(mesh_shape) == 1:
                # 1-D list, FSDP without TP
                assert len(placements) == 1
                shards = state_dict[key]
                state_dict[key] = merge_by_placement(shards, placements[0])
            else:
                # 2-D list, FSDP + TP
                raise NotImplementedError("FSDP + TP is not supported yet.")
        else:
            state_dict[key] = torch.cat(state_dict[key], dim=0)

    print("Merge completed.")
    hf_path = os.path.join(local_dir, "huggingface")
    if not os.path.isfile(os.path.join(hf_path, "config.json")):
        if args.base_model_path and os.path.isdir(args.base_model_path):
            print(
                f"[merger] {hf_path}/config.json missing; copying HF metadata "
                f"from base model {args.base_model_path}"
            )
            _copy_hf_metadata_from_base(args.base_model_path, hf_path)
        else:
            raise FileNotFoundError(
                f"{hf_path}/config.json is missing. Pass "
                f"--base_model_path /path/to/base/hf/model so the merger can "
                f"copy config/tokenizer/processor metadata before loading."
            )
    config: PretrainedConfig = AutoConfig.from_pretrained(hf_path)
    architectures: list[str] = getattr(config, "architectures", ["Unknown"])

    if "ForTokenClassification" in architectures[0]:
        AutoClass = AutoModelForTokenClassification
    elif "ForConditionalGeneration" in architectures[0]:
        AutoClass = AutoModelForImageTextToText
    elif "ForCausalLM" in architectures[0]:
        AutoClass = AutoModelForCausalLM
    else:
        raise NotImplementedError(f"Unknown architecture {architectures}.")

    # Backfill from the base model any architecture keys that didn't make it
    # into the FSDP shards (typically the frozen vision tower when
    # freeze_vision_tower=True). We compute the architecture's expected key
    # set on a meta model (free), then pull only the diff from the base
    # safetensors. vLLM rejects checkpoints with uninitialized weights, so
    # this must run before save.
    with torch.device("meta"):
        meta_model: PreTrainedModel = AutoClass.from_config(config, torch_dtype=torch.bfloat16)
    arch_keys = set(meta_model.state_dict().keys())
    del meta_model
    missing_keys = arch_keys - set(state_dict.keys())
    if missing_keys:
        base_model_path = args.base_model_path
        if base_model_path is None:
            base_model_path = getattr(config, "_name_or_path", None)
        if not base_model_path or not os.path.isdir(base_model_path):
            raise RuntimeError(
                f"[merger] {len(missing_keys)} architecture keys are missing from "
                f"the FSDP shards (e.g. {sorted(missing_keys)[:3]}). "
                f"Pass --base_model_path /path/to/base/hf/model to backfill "
                f"them (or set _name_or_path in config.json to a valid base "
                f"model directory)."
            )
        print(
            f"[merger] {len(missing_keys)} keys missing from FSDP shards; "
            f"backfilling from base model at {base_model_path}"
        )
        backfill = _load_missing_keys_from_base(base_model_path, missing_keys)
        state_dict.update(backfill)
        del backfill

    # Sanity check: every key has a real (non-empty) tensor.
    extra_keys = set(state_dict.keys()) - arch_keys
    if extra_keys:
        print(
            f"[merger] dropping {len(extra_keys)} keys not declared by the "
            f"architecture (likely optimizer/training-only state). "
            f"Examples: {sorted(extra_keys)[:3]}"
        )
        for k in extra_keys:
            state_dict.pop(k, None)

    # Write the safetensors file ourselves to avoid HF `save_pretrained`'s
    # implicit key remapping. On nested VL architectures (Qwen3.5-VL etc.)
    # save_pretrained(state_dict=…) silently renames `model.visual.*` to
    # `model.language_model.visual.*`, producing a file that vLLM refuses to
    # load. Writing the merged dict directly preserves the exact same key
    # layout as the base HF model, which vLLM already knows how to map.
    from safetensors.torch import save_file as _safe_save

    # save_pretrained would also have written config.json + generation_config
    # + processor + tokenizer, but those files are already present in
    # hf_path (verl copies them at training-init time), so we only need to
    # (re)write the weight file and refresh the safetensors index if any.
    hf_weights_path = os.path.join(hf_path, "model.safetensors")
    print(f"Writing merged weights to {hf_weights_path} ({len(state_dict)} keys)")
    # safetensors requires contiguous tensors.
    save_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
        if t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        save_dict[k] = t.contiguous()
    _safe_save(save_dict, hf_weights_path, metadata={"format": "pt"})
    del state_dict, save_dict

    # Drop any stale sharded index that would mislead the HF loader; the
    # weights now live in a single model.safetensors.
    index_path = os.path.join(hf_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        print(f"Removing stale {index_path}")
        os.remove(index_path)
    for stale_shard in os.listdir(hf_path):
        if stale_shard.startswith("model-") and stale_shard.endswith(".safetensors"):
            os.remove(os.path.join(hf_path, stale_shard))

    # We no longer call save_pretrained, so generation_config.json /
    # tokenizer / processor configs are not touched and don't need a backup.
    if args.hf_upload_path:
        upload_model_to_huggingface(hf_path, args.hf_upload_path)
