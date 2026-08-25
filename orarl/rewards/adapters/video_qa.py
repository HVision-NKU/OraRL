"""Built-in exact A-through-H Video-QA adapter."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..types import RewardContractError
from ._common import answer_payload, canonical_answer, exact_answer_payload, ground_truth

REWARD_NAME = "video_qa"
REWARD_TYPE = "batch"
DEFAULT_OPTIONS = "ABCDEFGH"
REQUIRE_ANSWER_TAGS = True


class VideoQAAdapter:
    """Configurable exact-option scorer used by the public router."""

    def __init__(
        self,
        options: str = DEFAULT_OPTIONS,
        require_answer_tags: bool = REQUIRE_ANSWER_TAGS,
    ) -> None:
        normalized = "".join(dict.fromkeys(str(options).upper()))
        if not normalized or any(not ("A" <= option <= "Z") for option in normalized):
            raise ValueError("Video-QA options must contain ASCII option letters.")
        self.options = normalized
        self.require_answer_tags = bool(require_answer_tags)
        choices = re.escape(normalized)
        self._option_re = re.compile(
            rf"\A\s*([{choices}])\s*\Z",
            flags=re.IGNORECASE,
        )

    def _option(self, value: Any, *, response: bool) -> str:
        payload = exact_answer_payload(value)
        if payload is None and (not response or not self.require_answer_tags):
            payload = answer_payload(value)
        match = self._option_re.fullmatch(str(payload or ""))
        return match.group(1).upper() if match is not None else ""

    def compute_score(
        self,
        batch: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        del kwargs
        results: list[dict[str, float]] = []
        for item in batch:
            prediction = self._option(item.get("response"), response=True)
            target = self._option(ground_truth(item), response=False)
            format_score = float(bool(prediction))
            accuracy = float(bool(prediction) and bool(target) and prediction == target)
            results.append(
                {
                    "overall": float(accuracy * format_score),
                    "accuracy": float(accuracy),
                    "format": float(format_score),
                }
            )
        return results

    def build_oracle_response_from_ground_truth(
        self,
        ground_truth: Any,
        extra: Mapping[str, Any] | None = None,
    ) -> str:
        del extra
        option = self._option(ground_truth, response=False)
        if not option:
            raise RewardContractError(
                "Video-QA ground truth must be exactly one configured option."
            )
        return canonical_answer(option)


_DEFAULT_ADAPTER = VideoQAAdapter()


def compute_score(
    batch: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, float]]:
    options = kwargs.pop("options", DEFAULT_OPTIONS)
    require_tags = kwargs.pop("require_answer_tags", REQUIRE_ANSWER_TAGS)
    if options == DEFAULT_OPTIONS and require_tags is REQUIRE_ANSWER_TAGS:
        return _DEFAULT_ADAPTER.compute_score(batch, **kwargs)
    return VideoQAAdapter(options, require_tags).compute_score(batch, **kwargs)


def build_oracle_response_from_ground_truth(
    ground_truth: Any,
    extra: Mapping[str, Any] | None = None,
) -> str:
    return _DEFAULT_ADAPTER.build_oracle_response_from_ground_truth(
        ground_truth,
        extra,
    )
