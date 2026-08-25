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

import math

import numpy as np
import pytest
import torch

from orarl.algorithm import (
    CorrectionConfig,
    PostSelectionReference,
    apply_post_selection_correction,
    capture_pre_selection_references,
    correct_post_selection_group,
)


class FakeDataProto:
    def __init__(self, batch, non_tensor_batch, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch
        self.meta_info = {} if meta_info is None else meta_info

    def __getitem__(self, rows):
        return FakeDataProto(
            {key: value[rows] for key, value in self.batch.items()},
            {key: np.asarray(value)[rows] for key, value in self.non_tensor_batch.items()},
            self.meta_info,
        )


def test_oracle_projection_preserves_zero_mean_and_nonnegative_anchor():
    active = torch.tensor([0.8, 0.2, -0.1, 0.1])
    oracle = torch.tensor([False, False, False, True])

    corrected, metrics = correct_post_selection_group(
        active,
        oracle,
        reference=PostSelectionReference(
            policy_rms=0.5,
            sigma_policy=0.2,
            policy_rows=3,
        ),
        config=CorrectionConfig(rms_match=False),
    )

    assert torch.allclose(
        corrected,
        torch.tensor([0.5, -0.1, -0.4, 0.0]),
        atol=1e-6,
    )
    assert abs(float(corrected.sum())) < 1e-6
    assert corrected[oracle].item() >= 0.0
    assert metrics["oracle_sign_projection"] == 1.0


def test_rms_matching_only_downscales():
    active = torch.tensor([0.5, -0.3, -0.2, 0.4])
    oracle = torch.tensor([False, False, False, True])

    corrected, metrics = correct_post_selection_group(
        active,
        oracle,
        reference=PostSelectionReference(
            policy_rms=0.2,
            sigma_policy=0.1,
            policy_rows=3,
        ),
    )

    assert abs(float(corrected.mean())) < 1e-6
    assert math.isclose(
        float(torch.sqrt(torch.mean(corrected.square()))),
        0.2,
        abs_tol=2e-6,
    )
    assert 0.25 <= metrics["rms_scale"] <= 1.0

    small, small_metrics = correct_post_selection_group(
        torch.tensor([-0.1, 0.1, 0.05]),
        torch.tensor([False, False, True]),
        reference=PostSelectionReference(
            policy_rms=10.0,
            sigma_policy=1.0,
            policy_rows=2,
        ),
    )
    assert small_metrics["rms_scale"] == 1.0
    assert torch.sqrt(torch.mean(small.square())) <= torch.tensor(0.1)


def test_small_reward_spread_skips_rms_scaling():
    active = torch.tensor([0.0, 0.0, 0.0, 0.4])
    oracle = torch.tensor([False, False, False, True])

    corrected, metrics = correct_post_selection_group(
        active,
        oracle,
        reference=PostSelectionReference(
            policy_rms=0.0,
            sigma_policy=0.0,
            policy_rows=3,
        ),
    )

    assert torch.allclose(corrected, torch.tensor([-0.1, -0.1, -0.1, 0.3]))
    assert metrics["small_sigma_fallback"] == 1.0
    assert metrics["rms_scale"] == 1.0


def test_reference_capture_and_batch_correction_use_preselection_policy_rows():
    advantages = torch.tensor(
        [
            [-1.0, -1.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.2, 0.2],
        ]
    )
    data = FakeDataProto(
        batch={
            "advantages": advantages,
            "response_mask": torch.ones_like(advantages),
            "token_level_scores": torch.tensor(
                [
                    [0.0, 0.1],
                    [0.0, 0.2],
                    [0.0, 0.3],
                    [0.0, 0.4],
                    [0.0, 1.0],
                ]
            ),
        },
        non_tensor_batch={
            "uid": np.asarray(["group"] * 5, dtype=object),
            "is_oracle_row": np.asarray(
                [False, False, False, False, True],
                dtype=bool,
            ),
            "row_id": np.arange(5),
        },
    )

    references = capture_pre_selection_references(data)
    expected_rms = math.sqrt((1.0 + 0.01 + 0.04 + 0.09) / 4.0)
    assert references["group"].policy_rows == 4
    assert references["group"].policy_rms == pytest.approx(expected_rms)

    selected = data[[0, 3, 4]]
    metrics = apply_post_selection_correction(selected, references)
    active = selected.batch["advantages"][:, 0]
    assert abs(float(active.mean())) < 1e-6
    assert active[-1].item() >= 0.0
    assert metrics["orarl/post_selection_groups"] == 1.0
    assert metrics["orarl/post_selection_rms_scale"] <= 1.0
