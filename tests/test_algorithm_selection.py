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

import numpy as np
import pytest
import torch

from orarl.algorithm import select_sign_balanced_rollouts


class FakeDataProto:
    def __init__(self, batch, non_tensor_batch, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = {} if meta_info is None else meta_info

    def __len__(self):
        return next(iter(self.batch.values())).shape[0]

    def __getitem__(self, rows):
        return FakeDataProto(
            {key: value[rows] for key, value in self.batch.items()},
            {key: np.asarray(value)[rows] for key, value in self.non_tensor_batch.items()},
            self.meta_info,
        )


def _data(scores, groups, oracle):
    scalar = torch.tensor(scores, dtype=torch.float32)
    advantages = scalar.unsqueeze(-1).repeat(1, 2)
    response_mask = torch.ones_like(advantages)
    return FakeDataProto(
        batch={
            "advantages": advantages,
            "response_mask": response_mask,
            "attention_mask": torch.ones(len(scores), 3),
        },
        non_tensor_batch={
            "uid": np.asarray(groups, dtype=object),
            "is_oracle_row": np.asarray(oracle, dtype=bool),
            "row_id": np.arange(len(scores)),
        },
        meta_info={"original": True},
    )


def test_strict_selection_meets_budget_with_all_fallback_paths():
    scores = [
        0.8,
        0.4,
        -0.9,
        -0.3,
        0.0,
        0.5,
        -0.9,
        -0.8,
        -0.7,
        0.0,
        0.0,
        0.5,
        0.7,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
    ]
    groups = ["a"] * 6 + ["b"] * 6 + ["c"] * 6
    oracle = [False] * 5 + [True]
    data = _data(scores, groups, oracle * 3)

    selected, metrics = select_sign_balanced_rollouts(
        data,
        keep_per_group=4,
        positive_quota=1,
        negative_quota=2,
        world_size=4,
    )

    assert selected.non_tensor_batch["row_id"].tolist() == [
        0,
        2,
        3,
        5,
        6,
        7,
        8,
        11,
        12,
        13,
        14,
        17,
    ]
    assert len(selected) == 12
    assert selected.non_tensor_batch["is_oracle_row"].sum() == 3
    assert metrics["orarl/kept_rows"] == 12.0
    assert metrics["orarl/oracle_rows_forced"] == 3.0
    assert metrics["orarl/positive_policy_rows_kept"] == 2.0
    assert metrics["orarl/negative_policy_rows_kept"] == 5.0
    assert metrics["orarl/zero_policy_rows_kept"] == 2.0
    assert metrics["orarl/cross_sign_fallback_rows"] == 1.0
    assert metrics["orarl/zero_fallback_rows"] == 2.0
    assert selected.meta_info["global_token_num"] == [3.0] * 12
    assert data.meta_info == {"original": True}
    assert all(key.startswith("orarl/") for key in metrics)


def test_selection_uses_sequence_mean_instead_of_response_length():
    data = _data(
        [0.3, 0.2, -0.4, -0.1, 0.0, 0.5],
        ["group"] * 6,
        [False, False, False, False, False, True],
    )
    data.batch["response_mask"][0, 1] = 0.0

    selected, _ = select_sign_balanced_rollouts(
        data,
        positive_quota=1,
        negative_quota=1,
        world_size=1,
        n_rollouts=6,
        prune_ratio=0.5,
    )

    assert selected.non_tensor_batch["row_id"].tolist() == [0, 2, 5]


def test_selection_checks_world_size_before_slicing():
    data = _data(
        [0.3, 0.2, -0.4, -0.1, 0.0, 0.5],
        ["group"] * 6,
        [False, False, False, False, False, True],
    )

    with pytest.raises(RuntimeError, match="not divisible"):
        select_sign_balanced_rollouts(
            data,
            keep_per_group=4,
            positive_quota=1,
            negative_quota=2,
            world_size=3,
        )


def test_selection_requires_exactly_one_oracle_per_group():
    data = _data(
        [0.3, 0.2, -0.4, -0.1],
        ["group"] * 4,
        [False, False, False, False],
    )

    with pytest.raises(ValueError, match="exactly one oracle"):
        select_sign_balanced_rollouts(
            data,
            keep_per_group=3,
            positive_quota=1,
            negative_quota=1,
        )
