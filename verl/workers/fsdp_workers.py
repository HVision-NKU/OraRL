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
"""
The main entry point to run the PPO algorithm
"""

import json
import os
import time
from datetime import timedelta
from typing import Literal, Optional, Union, cast

import numpy as np
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForTokenClassification,
    GenerationConfig,
    PreTrainedModel,
)
try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    from transformers.initialization import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, dispatch_one_to_all, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image
from ..utils.multimodal_contract import load_video_tensors_and_metadata
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import (
    AnyPrecisionAdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .config import ActorConfig, CriticConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


def _collect_hf_rollout_prompt_major(worker_group, outputs: list[DataProto]) -> DataProto:
    """Collect rank-major HF generations as prompt-major rollout groups."""
    if len(outputs) != worker_group.world_size:
        raise ValueError(
            f"Expected {worker_group.world_size} HF rollout outputs, got {len(outputs)}."
        )
    prompts_per_rank = len(outputs[0])
    if any(len(output) != prompts_per_rank for output in outputs):
        raise ValueError("HF rollout ranks returned different prompt counts.")
    merged = DataProto.concat(outputs)
    prompt_major_indices = np.asarray(
        [
            rank * prompts_per_rank + prompt
            for prompt in range(prompts_per_rank)
            for rank in range(worker_group.world_size)
        ],
        dtype=np.int64,
    )
    return merged.index_select(prompt_major_indices)


_HF_ROLLOUT_DISPATCH = {
    "dispatch_fn": dispatch_one_to_all,
    "collect_fn": _collect_hf_rollout_prompt_major,
}


class FSDPWorker(Worker):
    def __init__(
        self,
        config: WorkerConfig,
        role: Literal["actor", "critic", "rollout", "ref", "actor_rollout", "actor_rollout_ref"],
    ):
        super().__init__()
        self.config = config
        self.role = role
        self._cache = {}

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=10))

        # improve numerical stability
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_critic = self.role == "critic"
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        if self._has_actor and self._has_critic:
            raise ValueError("Actor and critic cannot be both initialized.")

        if self.config.actor.disable_kl:
            self._has_ref = False

        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")

        if self._has_critic:
            self._use_param_offload = self.config.critic.offload.offload_params
            self._use_optimizer_offload = self.config.critic.offload.offload_optimizer
            self._init_dist_mesh(self.config.critic, "critic")

        if self._has_ref:  # NOTE: it seems that manual offload is slower than FSDP offload
            self._use_ref_param_offload = self.config.ref.offload.offload_params

    def _init_dist_mesh(self, config: Union[ActorConfig, CriticConfig], role: Literal["actor", "critic"]):
        world_size = dist.get_world_size()
        # create main device mesh
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:  # hsdp
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )

        # create ulysses device mesh
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # validate and normalize config
        if self.config.rollout.n > 1:
            # `actor.global_batch_size` follows the convention of "unique prompts
            # per mini-batch", so the worker scales it up by the rollout fan-out
            # to get the actual row count. Under OraRL selection only
            # k=floor(n*(1-P)) rollouts per prompt survive, so the actor sees
            # k*rbs rows instead of n*rbs and must size its mini-batch
            # accordingly. Critic (GAE-only) always gets the full batch.
            effective_n = self.config.rollout.n
            prune_ratio = float(getattr(config, "selection_prune_ratio", 0.0))
            if role == "actor" and prune_ratio > 0.0:
                effective_n = max(
                    1,
                    int(self.config.rollout.n * (1.0 - prune_ratio)),
                )
                self.print_rank0(
                    f"{role} OraRL selection active (P={prune_ratio}): "
                    "scaling global_batch_size "
                    f"with k={effective_n} instead of n={self.config.rollout.n}."
                )
            config.global_batch_size *= effective_n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")

        config.global_batch_size_per_device = config.global_batch_size // (world_size // config.ulysses_size)
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")

        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")

        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "critic", "ref"],
    ) -> None:
        if role != "ref":  # ref model's tokenizer is same as actor
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.processor = get_processor(
                model_config.tokenizer_path,
                trust_remote_code=model_config.trust_remote_code,
                use_fast=True,
            )
            self.model_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_config.override_config,
            )

            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)

            self.print_rank0(f"Model config: {self.model_config}")

        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")

        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)

        if role == "critic":
            AutoClass = AutoModelForTokenClassification
        elif type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM

        if (not fsdp_config.enable_rank0_init) or self.device_mesh.get_local_rank("fsdp") == 0:
            model = AutoClass.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = AutoClass.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )

        model = cast(PreTrainedModel, model)  # lint
        model.tie_weights()  # avoid hanging
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        if role == "ref":
            model.requires_grad_(False)

        if model_config.freeze_vision_tower:
            visual = None
            if hasattr(model, "model") and hasattr(model.model, "visual"):  # transformers >= 4.52.0
                visual = model.model.visual
            elif hasattr(model, "visual"):  # transformers < 4.52.0
                visual = model.visual
            else:
                self.print_rank0("No vision tower found.")
                if model_config.train_vision_merger and role == "actor":
                    raise RuntimeError(
                        "train_vision_merger=True, but the model has no visual tower."
                    )
            if visual is not None:
                visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision backbone is set to not trainable.")
                if model_config.train_vision_merger and role == "actor":
                    merger_modules = [
                        (name, module)
                        for name, module in visual.named_modules()
                        if name == "merger" or name.endswith(".merger")
                    ]
                    if not merger_modules:
                        raise RuntimeError(
                            "train_vision_merger=True, but no merger module was "
                            "found under the model's visual tower."
                        )
                    for _, merger in merger_modules:
                        merger.requires_grad_(True)
                    merger_params = {
                        id(parameter): parameter
                        for _, merger in merger_modules
                        for parameter in merger.parameters()
                        if parameter.requires_grad
                    }
                    unexpected_visual_params = [
                        name
                        for name, parameter in visual.named_parameters()
                        if parameter.requires_grad and id(parameter) not in merger_params
                    ]
                    if unexpected_visual_params:
                        raise RuntimeError(
                            "Vision freeze invariant failed; non-merger visual "
                            f"parameters remain trainable: {unexpected_visual_params[:10]}"
                        )
                    merger_param_count = sum(
                        parameter.numel() for parameter in merger_params.values()
                    )
                    if merger_param_count <= 0:
                        raise RuntimeError(
                            "Vision merger was located but has no trainable parameters."
                        )
                    merger_names = ", ".join(name for name, _ in merger_modules)
                    self.print_rank0(
                        "Vision backbone frozen; merger trainable: "
                        f"modules=[{merger_names}], parameters={merger_param_count:,}, "
                        "FSDP use_orig_params=True."
                    )

        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")

        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.FULL_SHARD
            else:
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None

        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None

        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")

        if role in ["actor", "critic"]:
            self.fsdp_module = fsdp_module
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")

            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)

            if optim_config.lr_scheduler_type == "constant":
                self.lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
                )
            elif optim_config.lr_scheduler_type == "cosine":
                total_steps = optim_config.training_steps
                min_lr_ratio = optim_config.min_lr_ratio
                num_cycles = 0.5
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {optim_config.lr_scheduler_type} is not supported")
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    def _build_rollout(self) -> None:
        rollout_backend = str(self.config.rollout.name).lower()
        if rollout_backend in {"hf", "transformers"}:
            if not bool(self.config.actor.fsdp.use_orig_params):
                raise ValueError(
                    "HF rollout with FSDP requires actor.fsdp.use_orig_params=true "
                    "so transformers.generate can access unsharded parameters."
                )
            from .rollout.hf_rollout import HFRollout

            self.rollout = HFRollout(
                actor_module=self.fsdp_module,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
            )
            self.rollout_sharding_manager = None
            self.print_rank0(
                "[rollout] backend=hf: using the FSDP actor's transformers.generate "
                "(no vLLM engine)."
            )
            print_gpu_memory_usage("After HF rollout init")
            return
        if rollout_backend != "vllm":
            raise ValueError(
                f"Unsupported rollout backend {self.config.rollout.name!r}; "
                "expected 'vllm' or 'hf'."
            )

        from .rollout.vllm_rollout_spmd import vLLMRollout
        from .sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")

        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
            rollout_seed=self.config.rollout.seed,
        )
        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._has_critic:
            self._build_model_optimizer(
                model_config=self.config.critic.model,
                fsdp_config=self.config.critic.fsdp,
                optim_config=self.config.critic.optim,
                padding_free=self.config.critic.padding_free,
                role="critic",
            )

        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
            )

        if self._has_ref:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                padding_free=self.config.ref.padding_free,
                role="ref",
            )

        if self._has_actor:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
            )

        if self._has_critic:
            from .critic.dp_critic import DataParallelPPOCritic  # lazy import

            self.critic = DataParallelPPOCritic(
                config=self.config,
                critic_module=self.fsdp_module,
                critic_optimizer=self.optimizer,
            )

        if self._has_rollout:  # must after actor
            self._build_rollout()

        if self._has_ref:
            from .actor.dp_actor import DataParallelPPOActor  # lazy import

            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref,
                actor_module=self.ref_fsdp_module,
            )

        if self._has_actor or self._has_critic:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.save_checkpoint(path, save_model_only)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor or self._has_critic
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        self.checkpoint_manager.load_checkpoint(path)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:  # avoid OOM in resuming
            offload_fsdp_optimizer(self.optimizer)

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        if "uid" in self._cache:
            cached_uid = self._cache["uid"]
            new_uid = data.non_tensor_batch["uid"]
            if cached_uid.shape != new_uid.shape or not np.all(new_uid == cached_uid):
                self._cache.clear()

        if "multi_modal_inputs" not in self._cache:
            # Get pixel config from meta_info
            image_min_pixels = data.meta_info["image_min_pixels"]
            image_max_pixels = data.meta_info["image_max_pixels"]
            video_min_pixels = data.meta_info["video_min_pixels"]
            video_max_pixels = data.meta_info["video_max_pixels"]
            video_total_pixels = data.meta_info.get("video_total_pixels")
            video_fps = data.meta_info["video_fps"]
            video_max_frames = data.meta_info["video_max_frames"]

            batch_multi_modal_inputs = []
            multi_modal_inputs_cache = {}  # avoid repeated processing for n > 1 samples

            for index, multi_modal_data in zip(
                data.non_tensor_batch["uid"], data.non_tensor_batch["multi_modal_data"]
            ):
                if index not in multi_modal_inputs_cache:
                    images, videos = [], []
                    video_metadatas = None

                    if "images" in multi_modal_data:
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, image_min_pixels, image_max_pixels))
                    else:
                        videos, video_metadatas = load_video_tensors_and_metadata(
                            multi_modal_data,
                            video_min_pixels=video_min_pixels,
                            video_max_pixels=video_max_pixels,
                            video_max_frames=video_max_frames,
                            video_fps=video_fps,
                            video_total_pixels=video_total_pixels,
                        )

                    # Generate multi_modal_inputs using processor
                    if len(images) != 0:
                        multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                    elif len(videos) != 0:
                        processor_kwargs = {
                            "videos": videos,
                            "return_tensors": "pt",
                            "do_resize": False,
                            "do_sample_frames": False,
                        }
                        if video_metadatas is not None and len(video_metadatas) > 0:
                            processor_kwargs["video_metadata"] = video_metadatas

                        if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
                            multi_modal_inputs = dict(self.processor.video_processor(**processor_kwargs))
                        else:
                            processor_kwargs["images"] = None
                            multi_modal_inputs = dict(self.processor.image_processor(**processor_kwargs))
                    else:
                        multi_modal_inputs = {}

                    multi_modal_inputs_cache[index] = multi_modal_inputs

                batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])

            self._cache["uid"] = data.non_tensor_batch["uid"]
            self._cache["multi_modal_inputs"] = np.array(batch_multi_modal_inputs, dtype=object)

        data.non_tensor_batch["multi_modal_inputs"] = self._cache["multi_modal_inputs"]

        self._diagnose_video_alignment(data)

    def _diagnose_video_alignment(self, data: DataProto) -> None:
        """Per-sample video token/feature alignment audit (no-cache RL diagnostic).

        The FSDP forward crashes on the *aggregate* padding-free micro-batch
        (``sum(video tokens)`` vs ``sum(video features)`` in ``_get_input_embeds``),
        which hides *which* sample diverged. This runs where the per-sample
        ``video_grid_thw`` is freshly computed, so it names the exact offending
        sample: uid / problem_id / source_type / path / inline frame shape /
        grid_thw / the decode budget actually used / whether it is an oracle row.

        Toggle with env ``VERL_DIAGNOSE_VIDEO_MISMATCH`` (default "1"). Optional
        JSONL sink via ``VERL_DIAGNOSE_VIDEO_MISMATCH_LOG``. Cheap (integer
        counts over the mini-batch) and defensive (never raises).
        """
        if os.environ.get("VERL_DIAGNOSE_VIDEO_MISMATCH", "1") != "1":
            return
        try:
            mm_inputs = data.non_tensor_batch.get("multi_modal_inputs")
            if mm_inputs is None or "input_ids" not in data.batch:
                return
            processor = self.processor
            video_token_id = getattr(processor, "video_token_id", None)
            if processor is None or video_token_id is None:
                return
            merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2) or 2
            merge_length = int(merge_size) ** 2

            input_ids = data.batch["input_ids"]
            ntb = data.non_tensor_batch
            mm_data = ntb.get("multi_modal_data")
            uids = ntb.get("uid")
            problem_ids = ntb.get("problem_id")
            is_oracle = ntb.get("is_oracle_row")
            budget = {
                "video_min_pixels": data.meta_info.get("video_min_pixels"),
                "video_max_pixels": data.meta_info.get("video_max_pixels"),
                "video_total_pixels": data.meta_info.get("video_total_pixels"),
                "video_fps": data.meta_info.get("video_fps"),
                "video_max_frames": data.meta_info.get("video_max_frames"),
            }

            agg_tokens = 0
            agg_feats = 0
            culprits: list[dict] = []
            for i in range(len(mm_inputs)):
                mmi = mm_inputs[i]
                grid = None if mmi is None else mmi.get("video_grid_thw")
                if grid is None:
                    continue
                grid_t = grid if torch.is_tensor(grid) else torch.as_tensor(grid)
                if grid_t.ndim == 1:
                    grid_t = grid_t.unsqueeze(0)
                f_i = int((grid_t.prod(dim=-1).sum() // merge_length).item())
                n_i = int((input_ids[i] == video_token_id).sum().item())
                agg_tokens += n_i
                agg_feats += f_i
                if n_i == f_i:
                    continue
                info: dict = {
                    "idx": int(i),
                    "n_tokens": n_i,
                    "n_features": f_i,
                    "delta": f_i - n_i,
                    "grid_thw": grid_t.tolist(),
                    "uid": None if uids is None else str(uids[i]),
                    "problem_id": None if problem_ids is None else str(problem_ids[i]),
                    "is_oracle_row": None if is_oracle is None else bool(is_oracle[i]),
                }
                md = None if mm_data is None else mm_data[i]
                if isinstance(md, dict):
                    info["source_type"] = md.get("source_type")
                    info["paths"] = md.get("paths") or md.get("video")
                    frames = md.get("frames")
                    if frames is not None and len(frames) > 0:
                        info["inline_frames_shape"] = list(getattr(frames[0], "shape", []) or [])
                        info["inline_num_clips"] = len(frames)
                    metas = md.get("metadatas")
                    if metas and isinstance(metas[0], dict):
                        info["metadata0"] = {
                            k: metas[0].get(k) for k in ("total_num_frames", "fps", "duration")
                        }
                culprits.append(info)

            if not culprits and agg_tokens == agg_feats:
                return

            record = {
                "rank": getattr(self, "rank", None),
                "n_samples": int(len(mm_inputs)),
                "agg_tokens": agg_tokens,
                "agg_features": agg_feats,
                "agg_delta": agg_feats - agg_tokens,
                "budget": budget,
                "culprits": culprits,
            }
            log_path = os.environ.get("VERL_DIAGNOSE_VIDEO_MISMATCH_LOG")
            if log_path:
                # Per-rank file: many FSDP ranks (across nodes) run this
                # concurrently and shared-file appends corrupt lines.
                rank_path = f"{log_path}.rank{getattr(self, 'rank', 0)}"
                try:
                    os.makedirs(os.path.dirname(rank_path) or ".", exist_ok=True)
                    with open(rank_path, "a") as fh:
                        fh.write(json.dumps(record, default=str) + "\n")
                except Exception:
                    pass

            now = time.time()
            if now - getattr(self, "_video_diag_last_log", 0.0) >= 10.0:
                self._video_diag_last_log = now
                print(
                    f"[VIDEO-ALIGN][rank={record['rank']}] MISMATCH agg tokens={agg_tokens} "
                    f"features={agg_feats} delta={agg_feats - agg_tokens} | budget={budget} | "
                    f"{len(culprits)} culprit(s): {json.dumps(culprits, default=str)[:2000]}",
                    flush=True,
                )
        except Exception as exc:  # diagnostics must never take down training
            if os.environ.get("VERL_DIAGNOSE_VIDEO_MISMATCH_VERBOSE") == "1":
                print(f"[VIDEO-ALIGN] diagnostic error: {exc!r}", flush=True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            rollout_freed_bytes = (
                0
                if self.rollout_sharding_manager is None
                else self.rollout_sharding_manager.freed_bytes
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - rollout_freed_bytes
            ) / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - rollout_freed_bytes
            ) / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        if self.rollout_sharding_manager is None:
            if self._use_param_offload:
                load_fsdp_model(self.fsdp_module)
            self.rollout.prepare()
            return
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        if self.rollout_sharding_manager is None:
            self.rollout.release()
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
            torch.cuda.empty_cache()
            return
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        if self.rollout_sharding_manager is None:
            local_prompt_count = torch.tensor(
                [len(prompts)],
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            gathered_prompt_counts = [
                torch.zeros_like(local_prompt_count) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_prompt_counts, local_prompt_count)
            prompt_counts = [int(count.item()) for count in gathered_prompt_counts]
            if len(set(prompt_counts)) != 1:
                raise ValueError(
                    "HF rollout requires the same prompt count on every FSDP rank "
                    f"because generate(synced_gpus=True) is collective; got {prompt_counts}."
                )
            # The actor forward path normally receives uid from the repeated RL
            # batch. Generation batches contain only prompt fields, so provide a
            # per-call cache key for multimodal preprocessing.
            rollout_call = getattr(self, "_hf_rollout_call", 0)
            self._hf_rollout_call = rollout_call + 1
            if "uid" not in prompts.non_tensor_batch:
                rollout_rank = dist.get_rank() if dist.is_initialized() else 0
                prompts.non_tensor_batch["uid"] = np.asarray(
                    [
                        f"hf-rollout-r{rollout_rank}-c{rollout_call}-i{i}"
                        for i in range(len(prompts))
                    ],
                    dtype=object,
                )
            self._process_multi_modal_inputs(prompts)
            prompts = prompts.to(torch.cuda.current_device())
            output = self.rollout.generate_sequences(prompts=prompts)
        else:
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=_HF_ROLLOUT_DISPATCH)
    def generate_sequences_hf_official(self, prompts: DataProto):
        """Broadcast each prompt to every rank and sample once per rank.

        The custom collector transposes rank-major outputs to
        ``prompt0 x world_size, prompt1 x world_size, ...``. With 8 ranks this
        reproduces the reference recipe's eight device-specific HF generations
        per prompt instead of sampling all eight on one vLLM/HF worker.
        """
        assert self._has_rollout
        if self.rollout_sharding_manager is not None:
            raise RuntimeError("generate_sequences_hf_official requires rollout.name=hf.")

        prompts.meta_info.update(
            {
                "eos_token_id": self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id,
                # One completion from each rank; the collector forms G=world_size.
                "n": 1,
                "_hf_return_multi_modal_data": False,
            }
        )

        rollout_call = getattr(self, "_hf_rollout_call", 0)
        self._hf_rollout_call = rollout_call + 1
        if "uid" not in prompts.non_tensor_batch:
            rollout_rank = dist.get_rank() if dist.is_initialized() else 0
            prompts.non_tensor_batch["uid"] = np.asarray(
                [
                    f"hf-official-r{rollout_rank}-c{rollout_call}-i{i}"
                    for i in range(len(prompts))
                ],
                dtype=object,
            )
        self._process_multi_modal_inputs(prompts)
        prompts = prompts.to(torch.cuda.current_device())
        output = self.rollout.generate_sequences(prompts=prompts)
        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        # Add barrier before reshard to ensure all ranks are ready
        if self.world_size > 1:
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            self.fsdp_module._handle.reshard(True)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_probs(self, data: DataProto):
        assert self._has_ref

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_ref_param_offload:
            load_fsdp_model(self.ref_fsdp_module)

        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        # Add barrier before reshard to ensure all ranks are ready
        if self.world_size > 1:
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            self.ref_fsdp_module._handle.reshard(True)

        if self._use_ref_param_offload:
            offload_fsdp_model(self.ref_fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={"values": values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        assert self._has_critic

        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_critic"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )

            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["critic/lr"] = lr

            # Metrics should be in non_tensor_batch instead of meta_info, as DataProto not concat meta_info
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
            # Metrics do not need post processing since their batch size is 1

        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)

        output = output.to("cpu")
        return output
