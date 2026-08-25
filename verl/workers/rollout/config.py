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
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 8192
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)

    # vLLM KV cache dtype. Decode is HBM-bandwidth-bound on long-prompt RL
    # rollouts; switching from "auto" (matches model dtype, bf16=2 bytes/elt)
    # to "fp8" (1 byte/elt) halves KV traffic ⇒ ~2× decode speedup on Hopper.
    # Quality impact is typically < 0.5 IoU on temporal-grounding tasks
    # because attention is a low-rank op; H20 has native FP8 path so no
    # software emulation overhead. Choices: "auto" / "fp8" / "fp8_e5m2" /
    # "fp8_e4m3". Use "fp8" (vLLM picks the best variant on Hopper).
    kv_cache_dtype: str = "auto"

    # Return the per-token logprob of every sampled token in
    # ``rollout_log_probs`` and let the trainer reuse it as ``old_log_probs``
    # instead of recomputing under FSDP, so the first PPO mini-batch update does
    # not start from ratio == 1. Incompatible with oracle rows, whose tokens
    # differ from what the rollout engine sampled.
    calculate_log_probs: bool = False

    # Emit the per-sequence mean logprob into
    # ``non_tensor_batch["seq_logprob_for_filter"]`` only. It never becomes
    # ``old_log_probs``, so it stays out of the PPO ratio path and remains
    # compatible with oracle rows.
    collect_seq_logprob_for_filter: bool = False

    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
