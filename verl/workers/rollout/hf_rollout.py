# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils import torch_functional as VF
from .base import BaseRollout
from .config import RolloutConfig


def _normalize_eos_token_id(value: Any) -> int | list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten().tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        token_ids = [int(token_id) for token_id in value]
        if not token_ids:
            raise ValueError("eos_token_id must not be empty.")
        return token_ids
    return int(value)


def _repeat_interleave(value: Any, repeats: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    if isinstance(value, list):
        return [item for item in value for _ in range(repeats)]
    return np.repeat(value, repeats, axis=0)


class HFRollout(BaseRollout):
    """Generate on the actor itself with ``transformers.generate``.

    This mirrors the reference training recipe, which does not enable vLLM.
    FSDP parameters are materialized on every rank only for the no-grad
    generation window, then re-sharded before actor training.
    """

    def __init__(
        self,
        actor_module: nn.Module,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        if config.tensor_parallel_size != 1:
            raise ValueError("HF rollout requires rollout.tensor_parallel_size=1.")
        self.actor_module = actor_module
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._prepared = False

        # Keep generation randomness independent from any actor-side stochastic
        # ops while retaining the device-specific seed convention.
        training_rng_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(int(config.seed) + self.rank)
        self._generation_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(training_rng_state)
        self._training_rng_state = None

    def prepare(self) -> None:
        if self._prepared:
            raise RuntimeError("HF rollout is already prepared.")
        self._training_rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self._generation_rng_state)
        self.actor_module.eval()
        self._prepared = True

    def release(self) -> None:
        if not self._prepared:
            raise RuntimeError("HF rollout is not prepared.")
        self._generation_rng_state = torch.cuda.get_rng_state()
        if self._training_rng_state is not None:
            torch.cuda.set_rng_state(self._training_rng_state)
        self._training_rng_state = None
        self.actor_module.train()
        self._prepared = False

    def _full_params_context(self):
        if isinstance(self.actor_module, FSDP):
            return FSDP.summon_full_params(
                self.actor_module,
                recurse=True,
                writeback=False,
                rank0_only=False,
                offload_to_cpu=False,
            )
        return nullcontext()

    def _generation_model(self) -> nn.Module:
        if isinstance(self.actor_module, FSDP):
            return self.actor_module.module
        return self.actor_module

    @staticmethod
    def _move_multimodal_inputs(inputs: Any, device: torch.device) -> dict[str, Any]:
        if inputs is None:
            return {}
        moved = {}
        for key, value in dict(inputs).items():
            moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        return moved

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if not self._prepared:
            raise RuntimeError("Call prepare() before HF rollout generation.")

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = input_ids.shape[0]
        n = int(prompts.meta_info.get("n", self.config.n))
        temperature = float(prompts.meta_info.get("temperature", self.config.temperature))
        top_p = float(prompts.meta_info.get("top_p", self.config.top_p))
        top_k = int(prompts.meta_info.get("top_k", self.config.top_k))
        if n < 1:
            raise ValueError(f"HF rollout requires n >= 1, got {n}.")
        if n > 1 and temperature <= 0:
            raise ValueError("HF rollout with n > 1 requires temperature > 0.")
        response_length = int(self.config.response_length)
        eos_token_id = _normalize_eos_token_id(prompts.meta_info["eos_token_id"])
        pad_token_id = self.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id

        batch_mm_inputs = prompts.non_tensor_batch.get("multi_modal_inputs")
        if batch_mm_inputs is None:
            batch_mm_inputs = np.asarray([{} for _ in range(batch_size)], dtype=object)
        if len(batch_mm_inputs) != batch_size:
            raise ValueError(
                "HF rollout multimodal batch does not align with prompts: "
                f"{len(batch_mm_inputs)} != {batch_size}."
            )

        response_rows: list[torch.Tensor] = []
        device = torch.device("cuda", torch.cuda.current_device())
        generation_kwargs: dict[str, Any] = {
            "do_sample": temperature > 0,
            "max_new_tokens": response_length,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "use_cache": True,
            "synced_gpus": dist.is_initialized() and dist.get_world_size() > 1,
            "return_dict_in_generate": False,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            if top_k > 0:
                generation_kwargs["top_k"] = top_k

        with self._full_params_context():
            model = self._generation_model()
            for row in range(batch_size):
                row_mask = attention_mask[row].bool()
                valid_positions = torch.nonzero(row_mask, as_tuple=False)
                if valid_positions.numel() == 0:
                    raise ValueError(f"HF rollout prompt row {row} has no valid tokens.")
                start = int(valid_positions[0].item())
                row_input_ids = input_ids[row : row + 1, start:].to(device, non_blocking=True)
                row_attention_mask = attention_mask[row : row + 1, start:].to(
                    device, non_blocking=True
                )
                mm_inputs = self._move_multimodal_inputs(batch_mm_inputs[row], device)

                # Generate sequentially to avoid expanding a 448-frame visual
                # tensor n times on one GPU. The released trainer also performs
                # one completion per model.generate call/device.
                for _ in range(n):
                    output_ids = model.generate(
                        input_ids=row_input_ids,
                        attention_mask=row_attention_mask,
                        **mm_inputs,
                        **generation_kwargs,
                    )
                    generated = output_ids[0, row_input_ids.shape[-1] :]
                    response_rows.append(generated.detach())

        responses = torch.full(
            (batch_size * n, response_length),
            fill_value=pad_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        for row, generated in enumerate(response_rows):
            copy_length = min(response_length, int(generated.numel()))
            if copy_length > 0:
                responses[row, :copy_length] = generated[:copy_length]

        prompt_ids = _repeat_interleave(input_ids.to(device), n)
        prompt_attention_mask = _repeat_interleave(attention_mask.to(device), n)
        repeated_position_ids = _repeat_interleave(position_ids.to(device), n)
        sequence_ids = torch.cat([prompt_ids, responses], dim=-1)

        delta_position_id = torch.arange(1, response_length + 1, device=device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size * n, -1)
        if repeated_position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(batch_size * n, 1, -1).expand(
                batch_size * n,
                repeated_position_ids.size(1),
                -1,
            )
        response_position_ids = repeated_position_ids[..., -1:] + delta_position_id
        full_position_ids = torch.cat([repeated_position_ids, response_position_ids], dim=-1)

        response_mask = VF.get_response_mask(
            response_ids=responses,
            eos_token_id=eos_token_id,
            dtype=prompt_attention_mask.dtype,
        )
        full_attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=-1)
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": responses,
                "input_ids": sequence_ids,
                "attention_mask": full_attention_mask,
                "response_mask": response_mask,
                "position_ids": full_position_ids,
            },
            batch_size=batch_size * n,
        )

        non_tensor_batch = {}
        multi_modal_data = prompts.non_tensor_batch.get("multi_modal_data")
        if multi_modal_data is not None and bool(
            prompts.meta_info.get("_hf_return_multi_modal_data", True)
        ):
            non_tensor_batch["multi_modal_data"] = _repeat_interleave(multi_modal_data, n)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)
