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

from orarl.algorithm import (
    DetachedOracleConfig,
    DirectionalGainConfig,
    OraRLConfig,
    compute_directional_gain,
    compute_grpo_advantages,
    compute_grpo_outcome_advantage,
    compute_orarl_advantages,
    compute_orarl_policy_advantages,
    detached_oracle_advantage,
    normalized_reward_gap,
)


@pytest.mark.parametrize(
    "rewards",
    [
        [0.0, 0.5, 1.0, 0.5],
        [-2.0, -0.5, 0.25, 3.0],
        [0.3, 0.3, 0.3, 0.3],
    ],
)
def test_vanilla_grpo_matches_reference_formula(rewards):
    values = torch.tensor(rewards, dtype=torch.float64)
    expected = (values - values.mean()) / (values.std(unbiased=True) + 1e-6)

    actual = compute_grpo_advantages(
        values,
        np.asarray(["fixture"] * len(rewards), dtype=object),
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_grouped_grpo_wrapper_matches_token_fixture():
    rewards = torch.tensor([[0.0, 1.0, 9.0], [0.0, 3.0, 9.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    mask = torch.tensor(
        [[1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0]],
        dtype=torch.bool,
    )
    groups = np.asarray(["a", "a", "b", "b"], dtype=object)

    actual, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=mask,
        index=groups,
    )

    expected_scalars = torch.tensor(
        [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)] * 2,
        dtype=torch.float32,
    )
    expected = expected_scalars.unsqueeze(-1) * mask
    assert torch.allclose(actual, expected, atol=1e-5)
    assert torch.equal(actual, returns)


def test_directional_gain_is_positive_only_and_policy_recentered():
    rewards = torch.tensor([0.0, 0.2, 0.4, 0.6, 1.0])
    oracle = torch.tensor([False, False, False, False, True])
    config = DirectionalGainConfig(gamma=0.25)
    metrics = {}

    actual = compute_orarl_policy_advantages(
        rewards,
        ["group"] * len(rewards),
        oracle,
        config=config,
        metrics_out=metrics,
    )

    policy_rewards = rewards[~oracle]
    base = policy_rewards - policy_rewards.mean()
    gain = compute_directional_gain(
        rewards.std(unbiased=True),
        policy_rewards.std(unbiased=True),
        config=config,
    )
    transformed = torch.where(base > 0, base * gain, base)
    expected = transformed - transformed.mean()
    assert torch.allclose(actual[~oracle], expected, atol=1e-6)
    assert actual[oracle].item() == 0.0
    assert abs(float(actual[~oracle].mean())) < 1e-7
    assert metrics["orarl/directional_gain_mean"] == pytest.approx(float(gain))


def test_directional_gain_caps_degenerate_policy_spread():
    gain = compute_directional_gain(
        torch.tensor(0.5),
        torch.tensor(0.0),
        config=DirectionalGainConfig(gamma=0.25),
    )

    assert gain.item() == 4.0


def test_detached_oracle_uses_gap_and_best_positive_cap():
    policy_rewards = torch.tensor([0.2, 0.4, 0.6], requires_grad=True)
    policy_advantages = torch.tensor([-0.2, 0.0, 0.2], requires_grad=True)
    config = DetachedOracleConfig(
        scale=4.0,
        match_best_ratio=1.2,
        match_best_min=0.05,
        match_best_max=1.0,
    )

    actual = detached_oracle_advantage(
        torch.tensor(1.0, requires_grad=True),
        policy_rewards,
        policy_advantages,
        config=config,
    )

    assert actual.item() == pytest.approx(0.24, abs=1e-6)
    assert actual.requires_grad is False


def test_reward_gap_uses_the_paper_squared_decay():
    oracle_reward = torch.tensor(1.0)
    policy_rewards = torch.tensor([0.25, 0.5, 0.75])

    actual = normalized_reward_gap(oracle_reward, policy_rewards)
    expected = ((1.0 - 0.5) / (1.0 + 1e-6)) ** 2

    assert actual.item() == pytest.approx(expected, abs=1e-7)


def test_complete_orarl_keeps_policy_zero_mean_and_sets_oracle_anchor():
    rewards = torch.tensor([0.2, 0.4, 0.6, 1.0])
    oracle = np.asarray([False, False, False, True])
    config = OraRLConfig(
        directional_gain=DirectionalGainConfig(gamma=0.0),
        detached_oracle=DetachedOracleConfig(
            scale=4.0,
            match_best_ratio=1.2,
            match_best_min=0.05,
            match_best_max=1.0,
        ),
    )
    metrics = {}

    advantages = compute_orarl_advantages(
        rewards,
        ["group"] * len(rewards),
        oracle,
        config=config,
        metrics_out=metrics,
    )

    assert abs(float(advantages[:3].mean())) < 1e-7
    assert torch.allclose(advantages[:3], torch.tensor([-0.2, 0.0, 0.2]))
    assert advantages[3].item() == pytest.approx(0.24, abs=1e-6)
    assert metrics["orarl/oracle_cap_applied_fraction"] == 1.0


def test_orarl_rejects_groups_without_exactly_one_oracle():
    with pytest.raises(ValueError, match="one oracle row"):
        compute_orarl_policy_advantages(
            torch.tensor([0.0, 1.0]),
            ["group", "group"],
            [False, False],
        )
