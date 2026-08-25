# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 The OraRL Authors
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
"""Shared tensor and duck-typed batch utilities.

Parts of the masking and grouping behavior are derived from the Apache-2.0
EasyR1/verl implementation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import torch


def group_rows(
    group_ids: Sequence[Any] | np.ndarray | torch.Tensor,
    size: int,
) -> dict[Any, list[int]]:
    """Return row indices grouped in first-seen order."""

    if isinstance(group_ids, torch.Tensor):
        if group_ids.ndim != 1:
            raise ValueError("group_ids must be one-dimensional.")
        values = group_ids.detach().cpu().tolist()
    else:
        array = np.asarray(group_ids, dtype=object)
        if array.ndim != 1:
            raise ValueError("group_ids must be one-dimensional.")
        values = array.tolist()
    if len(values) != size:
        raise ValueError(f"group_ids has {len(values)} rows, expected {size}.")

    grouped: dict[Any, list[int]] = defaultdict(list)
    for row, value in enumerate(values):
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, Hashable):
            raise TypeError(f"group id at row {row} is not hashable.")
        grouped[value].append(row)
    return dict(grouped)


def boolean_mask(
    values: Sequence[bool] | np.ndarray | torch.Tensor,
    size: int,
    *,
    device: torch.device,
    name: str = "mask",
) -> torch.Tensor:
    """Materialize a one-dimensional boolean mask on ``device``."""

    if isinstance(values, torch.Tensor):
        result = values.detach().to(device=device, dtype=torch.bool)
    else:
        array = np.asarray(values, dtype=bool)
        result = torch.as_tensor(array, dtype=torch.bool, device=device)
    if result.ndim != 1 or result.numel() != size:
        raise ValueError(f"{name} must be one-dimensional with {size} rows.")
    return result


def validate_floating_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating dtype.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values.")


def validate_response_mask(response_mask: torch.Tensor) -> None:
    """Accept boolean, integer, or floating response masks."""

    if not isinstance(response_mask, torch.Tensor):
        raise TypeError("response_mask must be a torch.Tensor.")
    if response_mask.is_complex():
        raise TypeError("response_mask must have a real-valued dtype.")
    if not bool(torch.isfinite(response_mask).all()):
        raise ValueError("response_mask contains non-finite values.")


def sequence_rewards(
    rewards: torch.Tensor,
    response_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, bool]:
    """Collapse token rewards, returning ``(scores, token_input)``."""

    validate_floating_tensor("rewards", rewards)
    if rewards.ndim == 1:
        if response_mask is not None:
            raise ValueError("response_mask is only valid with token-level rewards.")
        return rewards.detach(), False
    if rewards.ndim != 2:
        raise ValueError("rewards must be one- or two-dimensional.")
    if response_mask is None or response_mask.shape != rewards.shape:
        raise ValueError("token-level rewards require a matching response_mask.")
    validate_response_mask(response_mask)
    mask = response_mask.detach().to(dtype=rewards.dtype)
    return (rewards.detach() * mask).sum(dim=-1), True


def sequence_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Collapse token advantages with a response-length-neutral masked mean."""

    validate_floating_tensor("advantages", advantages)
    validate_response_mask(response_mask)
    if advantages.ndim != 2 or advantages.shape != response_mask.shape:
        raise ValueError("advantages and response_mask must be matching matrices.")
    mask = response_mask.detach().to(dtype=advantages.dtype)
    lengths = mask.sum(dim=-1)
    if bool((lengths <= 0).any()):
        raise ValueError("every row must contain at least one valid response token.")
    return (advantages.detach() * mask).sum(dim=-1) / lengths


def broadcast_sequence_values(
    values: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Broadcast one scalar per row over valid response tokens."""

    if values.ndim != 1 or values.shape[0] != response_mask.shape[0]:
        raise ValueError("values must contain one scalar per response-mask row.")
    target_dtype = response_mask.dtype if dtype is None else dtype
    return values.to(dtype=target_dtype).unsqueeze(-1) * response_mask.to(dtype=target_dtype)


def sample_std(values: torch.Tensor) -> torch.Tensor:
    """Sample standard deviation, with zero for fewer than two rows."""

    if values.numel() < 2:
        return values.new_zeros(())
    return values.std(unbiased=True)


def batch_reward_tokens(data: Any, preferred_key: str | None = None) -> torch.Tensor:
    """Read token rewards from a DataProto-like object."""

    if preferred_key is not None:
        if preferred_key not in data.batch:
            raise ValueError(f"batch is missing reward key {preferred_key!r}.")
        return data.batch[preferred_key]
    for key in ("token_level_scores", "token_level_rewards"):
        if key in data.batch:
            return data.batch[key]
    raise ValueError("batch requires token_level_scores or token_level_rewards.")


def copy_and_refresh_meta(data: Any) -> None:
    """Detach selected metadata and refresh token counts when available."""

    if not hasattr(data, "meta_info"):
        return
    meta = getattr(data, "meta_info")
    data.meta_info = dict(meta) if meta is not None else {}
    if hasattr(data, "batch") and "attention_mask" in data.batch:
        data.meta_info["global_token_num"] = (
            data.batch["attention_mask"].sum(dim=-1).detach().cpu().tolist()
        )
