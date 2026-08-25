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

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.video_utils import VideoMetadata
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image
from ...utils.multimodal_contract import load_video_tensors_and_metadata
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, list], repeats: int) -> Union[torch.Tensor, np.ndarray, list]:
    # repeat the elements, supports tensor, numpy array and list
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    elif isinstance(value, list):
        out = []
        for v in value:
            out.extend([v] * repeats)
        return out
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output vision special tokens (image/video placeholders)
    if processor is None:
        return None

    logit_bias = {}
    if hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        logit_bias[image_token_id] = -100
    if hasattr(processor, "video_token"):
        video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        logit_bias[video_token_id] = -100

    return logit_bias if logit_bias else None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    image_min_pixels: int,
    image_max_pixels: int,
    video_min_pixels: int,
    video_max_pixels: int,
    video_max_frames: int,
    video_fps: float,
    video_total_pixels: Optional[int],
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    images, videos = [], []
    mm_kwargs = None

    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, image_min_pixels, image_max_pixels))

    video_tensors, video_metadatas = load_video_tensors_and_metadata(
        multi_modal_data,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        video_max_frames=video_max_frames,
        video_fps=video_fps,
        video_total_pixels=video_total_pixels,
    )
    if video_tensors:
        if video_metadatas is None or len(video_metadatas) != len(video_tensors):
            metadata_count = 0 if video_metadatas is None else len(video_metadatas)
            raise ValueError(
                "Resolved video data is missing valid metadata entries. "
                f"Got {len(video_tensors)} video tensors and {metadata_count} metadata entries."
            )
        for tensor, metadata in zip(video_tensors, video_metadatas):
            videos.append((tensor, metadata))
        mm_kwargs = {"do_sample_frames": False, "do_resize": False}

    if len(images) != 0:
        return {"image": images}, None

    if len(videos) != 0:
        return {"video": videos}, mm_kwargs

    return None, None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)

        # Bypass mode: have vLLM emit the logprob of every sampled token so the
        # trainer can reuse it as old_log_probs, skipping one FSDP recompute and
        # keeping the first update's ratio away from a constant 1.
        self.calculate_log_probs = bool(getattr(config, "calculate_log_probs", False))
        if self.calculate_log_probs and self.rank == 0:
            print(
                "[rollout] calculate_log_probs=True: vLLM rollout logprobs will be reused as old_log_probs "
                "(bypass mode, equivalent to verl's algorithm.rollout_correction.bypass_mode=True)."
            )

        # Filter-only path: derive a per-sequence mean logprob from the vLLM
        # token logprobs. It never becomes old_log_probs, so it stays compatible
        # with oracle rows.
        self.collect_seq_logprob_for_filter = bool(
            getattr(config, "collect_seq_logprob_for_filter", False)
        )
        if self.collect_seq_logprob_for_filter and self.rank == 0:
            print(
                "[rollout] collect_seq_logprob_for_filter=True: per-seq mean logprob will be "
                "written into non_tensor_batch['seq_logprob_for_filter']."
            )
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images, "video": 1}

        kv_cache_dtype = str(getattr(config, "kv_cache_dtype", "auto") or "auto").lower()
        if kv_cache_dtype not in {"auto", "fp8", "fp8_e5m2", "fp8_e4m3"}:
            raise ValueError(
                f"rollout.kv_cache_dtype must be one of 'auto'/'fp8'/'fp8_e5m2'/'fp8_e4m3', "
                f"got {kv_cache_dtype!r}."
            )
        if kv_cache_dtype != "auto" and self.rank == 0:
            print(
                f"[rollout] kv_cache_dtype={kv_cache_dtype}: KV cache stored in 1 byte/elt "
                f"(vs bf16 2 bytes), expecting ~2x decode speed on Hopper."
            )

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            kv_cache_dtype=kv_cache_dtype,
            # Disable the vLLM v1 multimodal IPC preprocessing cache, which keeps
            # one LRU on the P0 sender and another on the P1 receiver. Repeated
            # sleep(level=1)/wake_up cycles drift the two eviction orders apart,
            # so P0 sends only None believing the receiver still holds mm_hash
            # while P1 has already evicted it, raising
            # "AssertionError: Expected a cached item for mm_hash=...", most
            # often around _validate() / save_checkpoint. RL rollout rarely
            # repeats a video, so disabling this cache costs no throughput.
            mm_processor_cache_gb=0,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            # ``config.seed`` initializes the vLLM engine RNG above. Passing the
            # same value again as SamplingParams.seed resets every request to an
            # identical per-request RNG stream, unlike Transformers generation
            # where each device's RNG advances continuously.
            if key == "seed":
                continue
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        # logprobs=1 returns the top-1 logprob plus the chosen token's logprob
        # even when the chosen token is not the top-1. Do not use logprobs=0:
        # vLLM v1 engines treat it inconsistently across versions and some return
        # output.logprobs=None, which would silently fall back to 0.0 and degrade
        # training into probability-weighted asymmetric REINFORCE (visible as
        # ppo_kl close to entropy) instead of a PPO bypass.
        if self.calculate_log_probs or self.collect_seq_logprob_for_filter:
            sampling_kwargs["logprobs"] = 1

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params = self.sampling_params
        if kwargs:
            # vLLM 0.17+: SamplingParams is a msgspec.Struct with read-only properties
            # Get the valid constructor fields
            struct_fields = getattr(self.sampling_params, "__struct_fields__", None)
            if struct_fields is not None:
                # msgspec.Struct: rebuild with current values + overrides
                current_args = {}
                for field in struct_fields:
                    if not field.startswith("_"):
                        current_args[field] = getattr(self.sampling_params, field)
                for key, value in kwargs.items():
                    if key in current_args:
                        current_args[key] = value
                self.sampling_params = SamplingParams(**current_args)
            else:
                # Older vLLM: use setattr directly
                for key, value in kwargs.items():
                    if hasattr(self.sampling_params, key):
                        setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        self.sampling_params = old_sampling_params

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                item = {"prompt_token_ids": list(raw_prompt_ids)}
                mm_data, mm_kwargs = _process_multi_modal_data(
                    multi_modal_data,
                    prompts.meta_info["image_min_pixels"],
                    prompts.meta_info["image_max_pixels"],
                    prompts.meta_info["video_min_pixels"],
                    prompts.meta_info["video_max_pixels"],
                    prompts.meta_info["video_max_frames"],
                    prompts.meta_info["video_fps"],
                    prompts.meta_info.get("video_total_pixels"),
                )
                if mm_data is not None:
                    if "video" in mm_data:
                        videos = []
                        for tensor, metadata in mm_data["video"]:
                            if isinstance(metadata, dict):
                                metadata_obj = VideoMetadata(
                                    total_num_frames=metadata.get("total_num_frames", tensor.shape[0] if hasattr(tensor, "shape") else len(tensor)),
                                    fps=metadata.get("fps"),
                                    frames_indices=metadata.get("frames_indices"),
                                    video_backend=metadata.get("video_backend"),
                                    width=metadata.get("width"),
                                    height=metadata.get("height"),
                                    duration=metadata.get("duration"),
                                )
                            else:
                                metadata_obj = metadata
                            videos.append((tensor, metadata_obj))
                        item["multi_modal_data"] = {"video": videos}
                    else:
                        item["multi_modal_data"] = mm_data
                    if mm_kwargs is not None:
                        item["mm_processor_kwargs"] = mm_kwargs
                vllm_inputs.append(item)
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=self.use_tqdm
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            # vLLM only attaches logprobs when SamplingParams.logprobs is set.
            # Validation goes through update_sampling_params, which leaves that
            # field alone, so training runs always see them and other runs skip
            # this whole block.
            collect_log_probs = (
                (self.calculate_log_probs or self.collect_seq_logprob_for_filter)
                and self.sampling_params.logprobs is not None
            )
            if collect_log_probs:
                response_log_probs: list[list[float]] = []
                missing_seqs = 0
                missing_chosen = 0
                total_tokens = 0
                for completion in completions:
                    for out in completion.outputs:
                        token_ids_seq = out.token_ids
                        lp_dicts = out.logprobs  # list[dict[token_id, Logprob]]
                        if lp_dicts is None or len(lp_dicts) == 0:
                            missing_seqs += 1
                            response_log_probs.append([])
                            continue
                        seq_lps: list[float] = []
                        for tok_id, lp_dict in zip(token_ids_seq, lp_dicts):
                            total_tokens += 1
                            entry = lp_dict.get(tok_id) if lp_dict else None
                            if entry is None:
                                missing_chosen += 1
                                if lp_dict:
                                    entry = next(iter(lp_dict.values()))
                            seq_lps.append(float(entry.logprob) if entry is not None else 0.0)
                        response_log_probs.append(seq_lps)

                # Fail loudly when vLLM returned no logprobs, usually because a
                # vLLM v1 engine silently ignored SamplingParams.logprobs. A
                # silent fallback to old_log_probs == 0 would degrade training
                # into probability-weighted asymmetric REINFORCE, visible only as
                # actor/ppo_kl close to actor/entropy.
                if missing_seqs > 0:
                    raise RuntimeError(
                        f"[rollout] calculate_log_probs=True but vLLM returned no logprobs for "
                        f"{missing_seqs} sequence(s). Check that "
                        f"SamplingParams.logprobs={self.sampling_params.logprobs} is supported by "
                        f"this vLLM version (use logprobs>=1, never 0)."
                    )
                if total_tokens > 0 and missing_chosen / total_tokens > 0.01:
                    raise RuntimeError(
                        f"[rollout] {missing_chosen}/{total_tokens} chosen tokens are absent from "
                        f"the vLLM logprob dicts (>1%); the values are not trustworthy."
                    )

                # Pad to response_length with 0.0; response_mask hides the
                # padding, so it never reaches the ratio.
                response_log_probs = VF.pad_2d_list_to_length(
                    response_log_probs, 0.0, max_length=self.config.response_length
                ).to(input_ids.device, dtype=torch.float32)

                # One-shot self check: print the mean/std of the vLLM logprobs on
                # the first rollout so the log shows real values rather than
                # silent zeros. A working bypass reports mean between -2.0 and
                # -0.5 and std above 0.5, depending on temperature and entropy.
                if self.rank == 0 and not getattr(self, "_logprob_self_check_done", False):
                    flat = response_log_probs.flatten()
                    nonzero = flat[flat != 0.0]
                    print(
                        f"[rollout][bypass self-check] vLLM rollout_log_probs "
                        f"shape={tuple(response_log_probs.shape)} "
                        f"all_mean={flat.mean().item():.4f} all_std={flat.std().item():.4f} "
                        f"nonzero_mean={nonzero.mean().item() if nonzero.numel() > 0 else float('nan'):.4f} "
                        f"nonzero_std={nonzero.std().item() if nonzero.numel() > 1 else float('nan'):.4f} "
                        f"nonzero_min={nonzero.min().item() if nonzero.numel() > 0 else float('nan'):.4f} "
                        f"nonzero_max={nonzero.max().item() if nonzero.numel() > 0 else float('nan'):.4f} "
                        f"({nonzero.numel()}/{flat.numel()} non-zero)"
                    )
                    self._logprob_self_check_done = True

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch_tensors = {
            "prompts": input_ids,
            "responses": response_ids,
            "input_ids": sequence_ids,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        }
        # Only bypass mode puts token logprobs into batch.rollout_log_probs. The
        # filter-only path needs the per-sequence mean alone, which avoids both
        # the oracle-row token mismatch and one (B, T) tensor of dispatch traffic.
        if collect_log_probs and self.calculate_log_probs:
            batch_tensors["rollout_log_probs"] = response_log_probs
        batch = TensorDict(batch_tensors, batch_size=batch_size)

        non_tensor_batch: dict = {}
        if batch_multi_modal_data is not None:
            non_tensor_batch["multi_modal_data"] = batch_multi_modal_data

        # Per-sequence mean logprob, never used in the gradient path. Masking
        # with response_mask is more robust than treating != 0 as padding.
        if collect_log_probs and self.collect_seq_logprob_for_filter:
            valid_lengths = response_mask.sum(dim=-1).clamp_min(1).to(response_log_probs.dtype)
            seq_logprob_sum = (response_log_probs * response_mask.to(response_log_probs.dtype)).sum(dim=-1)
            seq_logprob_mean = (seq_logprob_sum / valid_lengths).detach().cpu().numpy().astype(np.float32)
            non_tensor_batch["seq_logprob_for_filter"] = seq_logprob_mean

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
