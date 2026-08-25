"""OraRL's strict sign-balanced rollout selection."""

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from ..protocol import DataProto
from ..utils import torch_functional as VF


def _sequence_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if advantages.shape != response_mask.shape:
        raise ValueError(
            "advantages and response_mask must have identical shapes, got "
            f"{tuple(advantages.shape)} and {tuple(response_mask.shape)}."
        )
    return VF.masked_mean(advantages, response_mask, dim=-1)


def _rank_by_magnitude(rows: list[int], scores: torch.Tensor) -> list[int]:
    return sorted(
        rows,
        key=lambda row: (-abs(float(scores[row].item())), row),
    )


@torch.no_grad()
def select_orarl_rollouts(
    data: DataProto,
    *,
    n_rollouts: int,
    prune_ratio: float,
    world_size: int,
    positive_quota: int,
    negative_quota: int,
    oracle_key: str = "is_oracle_row",
) -> tuple[DataProto, dict[str, float]]:
    """Retain one oracle row and sign-balanced policy rows in every group."""

    if not 0.0 < prune_ratio < 1.0:
        raise ValueError("OraRL prune_ratio must be in (0, 1).")
    if n_rollouts <= 1:
        raise ValueError("OraRL n_rollouts must be greater than one.")
    if world_size <= 0:
        raise ValueError("OraRL world_size must be positive.")
    keep_per_group = max(1, int(n_rollouts * (1.0 - prune_ratio)))
    if positive_quota < 0 or negative_quota < 0:
        raise ValueError("OraRL sign quotas must be non-negative.")
    if positive_quota + negative_quota + 1 != keep_per_group:
        raise ValueError(
            "OraRL sign quotas plus one oracle must equal the keep budget."
        )

    required_batch = ("advantages", "response_mask", "attention_mask")
    missing_batch = sorted(
        key for key in required_batch if key not in data.batch
    )
    if missing_batch:
        raise ValueError(
            "OraRL selection is missing batch fields: "
            + ", ".join(missing_batch)
        )
    if "uid" not in data.non_tensor_batch:
        raise ValueError("OraRL selection requires non_tensor_batch['uid'].")
    if oracle_key not in data.non_tensor_batch:
        raise ValueError(
            f"OraRL selection requires non_tensor_batch[{oracle_key!r}]."
        )

    signed_scores = _sequence_advantages(
        data.batch["advantages"],
        data.batch["response_mask"],
    )
    total_rows = int(signed_scores.numel())
    uids = np.asarray(data.non_tensor_batch["uid"], dtype=object)
    oracle_flags = np.asarray(
        data.non_tensor_batch[oracle_key],
        dtype=bool,
    )
    if len(uids) != total_rows or len(oracle_flags) != total_rows:
        raise ValueError(
            "OraRL uid and oracle flags must align with batch rows."
        )

    grouped: dict[Any, list[int]] = defaultdict(list)
    for row, uid in enumerate(uids):
        grouped[uid].append(row)

    selected_rows: list[int] = []
    positive_kept = 0
    negative_kept = 0
    zero_kept = 0
    cross_sign_fallback = 0
    zero_fallback = 0

    for uid, rows in grouped.items():
        oracle_rows = [row for row in rows if bool(oracle_flags[row])]
        if len(oracle_rows) != 1:
            raise ValueError(
                f"OraRL group {uid!r} requires exactly one oracle row, "
                f"got {len(oracle_rows)}."
            )
        candidates = [row for row in rows if row != oracle_rows[0]]
        if len(candidates) != n_rollouts:
            raise ValueError(
                f"OraRL group {uid!r} requires {n_rollouts} policy rows, "
                f"got {len(candidates)}."
            )

        positive = _rank_by_magnitude(
            [
                row
                for row in candidates
                if float(signed_scores[row].item()) > 0.0
            ],
            signed_scores,
        )
        negative = _rank_by_magnitude(
            [
                row
                for row in candidates
                if float(signed_scores[row].item()) < 0.0
            ],
            signed_scores,
        )
        zeros = [
            row
            for row in candidates
            if float(signed_scores[row].item()) == 0.0
        ]

        chosen = positive[:positive_quota] + negative[:negative_quota]
        chosen_set = set(chosen)
        remaining = keep_per_group - 1 - len(chosen)
        if remaining > 0:
            surplus = _rank_by_magnitude(
                [
                    row
                    for row in positive + negative
                    if row not in chosen_set
                ],
                signed_scores,
            )
            cross_fill = surplus[:remaining]
            chosen.extend(cross_fill)
            chosen_set.update(cross_fill)
            cross_sign_fallback += len(cross_fill)
            remaining -= len(cross_fill)
        if remaining > 0:
            zero_fill = [
                row for row in zeros if row not in chosen_set
            ][:remaining]
            chosen.extend(zero_fill)
            chosen_set.update(zero_fill)
            zero_fallback += len(zero_fill)
            remaining -= len(zero_fill)
        if remaining:
            raise RuntimeError(
                f"OraRL group {uid!r} could not satisfy its keep budget."
            )

        for row in chosen:
            score = float(signed_scores[row].item())
            positive_kept += int(score > 0.0)
            negative_kept += int(score < 0.0)
            zero_kept += int(score == 0.0)
        selected_rows.extend([oracle_rows[0], *chosen])

    selected_rows.sort()
    if len(selected_rows) % world_size:
        raise RuntimeError(
            f"OraRL selected batch size {len(selected_rows)} is not "
            f"divisible by world_size {world_size}."
        )

    selected = data[selected_rows]
    selected.meta_info = dict(selected.meta_info)
    selected.meta_info["global_token_num"] = (
        torch.sum(selected.batch["attention_mask"], dim=-1).tolist()
    )

    selected_index = torch.tensor(
        selected_rows,
        dtype=torch.long,
        device=signed_scores.device,
    )
    selected_set = set(selected_rows)
    dropped_rows = [
        row for row in range(total_rows) if row not in selected_set
    ]
    metrics = {
        "orarl/selection/groups": float(len(grouped)),
        "orarl/selection/keep_per_group": float(keep_per_group),
        "orarl/selection/kept_rows": float(len(selected_rows)),
        "orarl/selection/dropped_rows": float(
            total_rows - len(selected_rows)
        ),
        "orarl/selection/effective_keep_ratio": (
            len(selected_rows) / float(total_rows)
        ),
        "orarl/selection/oracle_rows_forced": float(len(grouped)),
        "orarl/selection/positive_policy_rows_kept": float(positive_kept),
        "orarl/selection/negative_policy_rows_kept": float(negative_kept),
        "orarl/selection/zero_policy_rows_kept": float(zero_kept),
        "orarl/selection/cross_sign_fallback_rows": float(
            cross_sign_fallback
        ),
        "orarl/selection/zero_fallback_rows": float(zero_fallback),
        "orarl/selection/abs_advantage_kept_mean": float(
            signed_scores.index_select(0, selected_index).abs().mean().item()
        ),
    }
    if dropped_rows:
        dropped_index = torch.tensor(
            dropped_rows,
            dtype=torch.long,
            device=signed_scores.device,
        )
        metrics["orarl/selection/abs_advantage_dropped_mean"] = float(
            signed_scores.index_select(0, dropped_index).abs().mean().item()
        )
    return selected, metrics
