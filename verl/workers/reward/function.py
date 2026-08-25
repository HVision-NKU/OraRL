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

import importlib
import importlib.util
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Callable, Optional, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardInput(TypedDict):
    prompt: Optional[str]
    response: str
    response_length: int
    ground_truth: str
    data_type: Optional[str]
    problem_type: Optional[str]
    problem: Optional[str]
    problem_id: Optional[int]


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


def _float_env(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _apply_oracle_reward_control(
    score: RewardScore,
    is_oracle_row: bool,
    reward_metrics: defaultdict,
) -> RewardScore:
    """Optionally reduce the scalar reward of oracle rows only.

    This lives in the reward manager rather than in a task reward file because
    only the manager sees the `is_oracle_row` flag. Comparing response against
    ground_truth inside a task reward would also catch the rare on-policy exact
    match, which must keep its full reward.

    Env knobs (disabled by default):
      ORARL_ORACLE_REWARD_SCALE  multiply the oracle overall by this value
      ORARL_ORACLE_REWARD_CAP    cap the oracle overall after scaling

    Example for tracking: an exact annotation normally scores overall=1.5.
    ORARL_ORACLE_REWARD_CAP=1.25 keeps it a positive anchor while preventing it
    from dominating the group standard deviation and the selection step under
    aggressive reward shaping.
    """
    if not is_oracle_row:
        return score

    if not isinstance(score, dict) or "overall" not in score:
        return score

    scale = _float_env("ORARL_ORACLE_REWARD_SCALE", 1.0)
    cap = _float_env("ORARL_ORACLE_REWARD_CAP", None)
    if scale is None:
        scale = 1.0

    old_overall = float(score.get("overall", 0.0))
    new_overall = old_overall * float(scale)
    if cap is not None:
        new_overall = min(new_overall, float(cap))

    if new_overall != old_overall:
        score = dict(score)
        score["overall"] = float(new_overall)
        reward_metrics["oracle_overall_before_control"].append(old_overall)
        reward_metrics["oracle_overall_after_control"].append(float(new_overall))
        reward_metrics["oracle_reward_scale"].append(float(scale))
        if cap is not None:
            reward_metrics["oracle_reward_cap"].append(float(cap))

    return score


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[list[RewardInput]], list[RewardScore]]


def _build_reward_input(
    data: DataProto,
    response_str: str,
    response_length: int,
    index: int,
    prompt_str: Optional[str] = None,
) -> RewardInput:
    response_prefix = os.getenv("MULTITASK_REWARD_RESPONSE_PREFIX", "")
    if response_prefix and not response_str.startswith(response_prefix):
        response_str = response_prefix + response_str
    non_tensor = data.non_tensor_batch
    data_type = non_tensor["data_type"][index] if "data_type" in non_tensor else None
    problem_type = non_tensor["problem_type"][index] if "problem_type" in non_tensor else None
    problem = non_tensor["problem_reserved_text"][index] if "problem_reserved_text" in non_tensor else None
    problem_id = non_tensor["problem_id"][index] if "problem_id" in non_tensor else None
    item = {
        "prompt": prompt_str,
        "response": response_str,
        "response_length": response_length,
        "ground_truth": non_tensor["ground_truth"][index],
        "data_type": data_type,
        "problem_type": problem_type,
        "problem": problem,
        "problem_id": problem_id,
    }
    # Task rewards may need side-channel metadata that is not part of the text
    # prompt. Mask-aware segmentation reward, for instance, needs the annotated
    # mask payload and video metadata to score the predicted prompts.
    for key in (
        "segmentation_output",
        "fps",
        "video_second",
        "resolution",
        "path",
        "data_source",
        "meta",
    ):
        if key in non_tensor:
            item[key] = non_tensor[key][index]
    return item


class SequentialFunctionRewardManagerMixin:
    reward_fn: SequentialRewardFunction

    def compute_reward_sequential(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        prompt_ids = data.batch.get("prompts", None)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            prompt_str = None
            if prompt_ids is not None:
                prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=self.config.skip_special_tokens)
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(_build_reward_input(data, response_str, cur_response_length, i, prompt_str))
            is_oracle = bool(data.non_tensor_batch.get("is_oracle_row", [False] * len(data))[i])
            score = _apply_oracle_reward_control(score, is_oracle, reward_metrics)
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManagerMixin:
    reward_fn: BatchRewardFunction

    def compute_reward_batch(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        reward_inputs = []
        prompt_ids = data.batch.get("prompts", None)
        response_ids = data.batch["responses"]
        response_length = torch.sum(data.batch["response_mask"], dim=-1)
        for i in range(len(data)):
            prompt_str = None
            if prompt_ids is not None:
                prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=self.config.skip_special_tokens)
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            valid_response_ids = response_ids[i][:cur_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(_build_reward_input(data, response_str, cur_response_length, i, prompt_str))

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        is_oracle_rows = data.non_tensor_batch.get("is_oracle_row", [False] * len(data))
        for i, score in enumerate(scores):
            cur_response_length = int(response_length[i].item())  # avoid tensor indexing error
            score = _apply_oracle_reward_control(score, bool(is_oracle_rows[i]), reward_metrics)
            reward_tensor[i, cur_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class AutoRewardManager(BatchFunctionRewardManagerMixin, SequentialFunctionRewardManagerMixin):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if config.reward_function_is_module:
            try:
                module = importlib.import_module(config.reward_function)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to import reward module {config.reward_function!r}: {e}"
                ) from e
        else:
            if not os.path.exists(config.reward_function):
                raise FileNotFoundError(
                    f"Reward function file {config.reward_function} not found."
                )
            spec = importlib.util.spec_from_file_location(
                "custom_reward_fn",
                config.reward_function,
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(
                    f"Could not create an import spec for {config.reward_function!r}."
                )
            module = importlib.util.module_from_spec(spec)
            try:
                sys.modules["custom_reward_fn"] = module
                spec.loader.exec_module(module)
            except Exception as e:
                raise RuntimeError(f"Failed to load reward function: {e}") from e

        if not config.reward_function_name or not hasattr(
            module,
            config.reward_function_name,
        ):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        reward_name = getattr(module, "REWARD_NAME", "unknown")
        reward_type = getattr(module, "REWARD_TYPE", "batch")
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        print(f"Reward name: {reward_name}, reward type: {reward_type}.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.reward_type = reward_type
        self.config = config
        self.tokenizer = tokenizer

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, dict[str, list[float]]]:
        """Compute reward for a batch of data."""
        if self.reward_type == "batch":
            return self.compute_reward_batch(data)
        elif self.reward_type == "sequential":
            return self.compute_reward_sequential(data)
        else:
            raise ValueError(f"Unsupported reward type: {self.reward_type}.")
