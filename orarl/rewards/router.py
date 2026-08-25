"""Lightweight task router for the seven OraRL reward families."""

from __future__ import annotations

import importlib
import inspect
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple, Union

from .types import (
    RewardAdapter,
    RewardContractError,
    RewardResult,
    TaskFamily,
    UnknownTaskError,
)

REWARD_NAME = "orarl"
REWARD_TYPE = "batch"

DEFAULT_MODULE_PATHS: Mapping[TaskFamily, str] = MappingProxyType(
    {
        TaskFamily.TEMPORAL_GROUNDING: ("orarl.rewards.adapters.temporal_grounding"),
        TaskFamily.TRACKING: "orarl.rewards.adapters.tracking",
        TaskFamily.SEGMENTATION: "orarl.rewards.adapters.segmentation",
        TaskFamily.SPATIAL_GROUNDING: ("orarl.rewards.adapters.spatial_grounding"),
        TaskFamily.SPATIAL_TEMPORAL_GROUNDING: (
            "orarl.rewards.adapters.spatial_temporal_grounding"
        ),
        TaskFamily.SPATIAL_INTELLIGENCE: ("orarl.rewards.adapters.spatial_intelligence"),
        TaskFamily.VIDEO_QA: "orarl.rewards.adapters.video_qa",
    }
)


def _default_module_paths() -> Dict[TaskFamily, str]:
    return dict(DEFAULT_MODULE_PATHS)


_BUILTIN_ALIASES: Dict[str, TaskFamily] = {}


def _slug(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _register_aliases(family: TaskFamily, *names: str) -> None:
    for name in names:
        _BUILTIN_ALIASES[_slug(name)] = family


_register_aliases(
    TaskFamily.TEMPORAL_GROUNDING,
    TaskFamily.TEMPORAL_GROUNDING.value,
    "temporal grounding",
    "temporal",
    "timelens",
    "time lens",
    "moment retrieval",
    "temporal localization",
    "video temporal grounding",
)
_register_aliases(
    TaskFamily.TRACKING,
    TaskFamily.TRACKING.value,
    "video tracking",
    "object tracking",
    "single object tracking",
    "sot",
)
_register_aliases(
    TaskFamily.SEGMENTATION,
    TaskFamily.SEGMENTATION.value,
    "image segmentation",
    "video segmentation",
    "referring segmentation",
    "mask segmentation",
)
_register_aliases(
    TaskFamily.SPATIAL_GROUNDING,
    TaskFamily.SPATIAL_GROUNDING.value,
    "spatial grounding",
    "spatial",
    "visual grounding",
    "image grounding",
    "bbox grounding",
    "refcoco",
    "referring expression comprehension",
    "rec",
)
_register_aliases(
    TaskFamily.SPATIAL_TEMPORAL_GROUNDING,
    TaskFamily.SPATIAL_TEMPORAL_GROUNDING.value,
    "spatial temporal grounding",
    "spatial-temporal grounding",
    "spatio temporal grounding",
    "spatiotemporal grounding",
    "video spatial temporal grounding",
    "video grounding",
    "stvg",
)
_register_aliases(
    TaskFamily.SPATIAL_INTELLIGENCE,
    TaskFamily.SPATIAL_INTELLIGENCE.value,
    "spatial intelligence",
    "video spatial intelligence",
    "spatial reasoning",
    "vsi",
    "vsi bench",
    "object abs distance",
    "object counting",
    "object size estimation",
    "room size estimation",
    "object rel distance",
    "route planning",
    "obj appearance order",
)
_register_aliases(
    TaskFamily.VIDEO_QA,
    TaskFamily.VIDEO_QA.value,
    "video qa",
    "video question answering",
    "video multiple choice",
    "video mc",
    "video_qa_mc",
    "multiple choice",
    "multiple_choice",
    "mc",
    "image_sequence_mc_answer_only",
)


_SPATIAL_INTELLIGENCE_PREFIXES: Tuple[str, ...] = ("object_rel_direction",)


def _normalize_with_aliases(
    value: Any,
    aliases: Mapping[str, TaskFamily],
) -> TaskFamily:
    if isinstance(value, TaskFamily):
        return value
    normalized = _slug(value)
    family = aliases.get(normalized)
    if family is not None:
        return family
    if normalized.startswith(_SPATIAL_INTELLIGENCE_PREFIXES):
        return TaskFamily.SPATIAL_INTELLIGENCE
    supported = ", ".join(family.value for family in TaskFamily)
    raise UnknownTaskError(f"Unknown reward task {value!r}. Supported families: {supported}.")


def normalize_task_name(value: Any) -> TaskFamily:
    """Normalize a public task name or a dataset ``problem_type``."""

    return _normalize_with_aliases(value, _BUILTIN_ALIASES)


@dataclass(frozen=True)
class RouterSettings:
    """Explicit router configuration.

    Packaged reward modules are imported only when their family is first scored
    or asked to build an oracle response. Adapter and module-path overrides can
    replace any family without environment configuration.
    """

    module_paths: Mapping[Union[TaskFamily, str], str] = field(
        default_factory=_default_module_paths
    )
    adapters: Mapping[Union[TaskFamily, str], RewardAdapter] = field(default_factory=dict)
    task_aliases: Mapping[str, Union[TaskFamily, str]] = field(default_factory=dict)
    score_kwargs: Mapping[Union[TaskFamily, str], Mapping[str, Any]] = field(default_factory=dict)
    task_fields: Tuple[str, ...] = (
        "task",
        "task_name",
        "problem_type",
        "task_source",
        "scoring_family",
        "question_type",
    )
    ground_truth_fields: Tuple[str, ...] = ("ground_truth", "answer")
    video_qa_options: str = "ABCDEFGH"
    video_qa_require_answer_tags: bool = True


class RewardRouter:
    """Route typed samples to lazy task-specific reward adapters."""

    def __init__(
        self,
        settings: Optional[RouterSettings] = None,
        *,
        importer: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.settings = settings or RouterSettings()
        if not self.settings.task_fields:
            raise ValueError("RouterSettings.task_fields cannot be empty.")
        if not self.settings.ground_truth_fields:
            raise ValueError("RouterSettings.ground_truth_fields cannot be empty.")

        self._aliases = dict(_BUILTIN_ALIASES)
        for alias, target in self.settings.task_aliases.items():
            family = _normalize_with_aliases(target, self._aliases)
            self._aliases[_slug(alias)] = family

        self._module_paths = _default_module_paths()
        for key, module_path in self.settings.module_paths.items():
            family = _normalize_with_aliases(key, self._aliases)
            if not isinstance(module_path, str) or not module_path.strip():
                raise ValueError(f"Module path for {family.value!r} must be non-empty.")
            self._module_paths[family] = module_path.strip()

        self._adapter_overrides: Dict[TaskFamily, RewardAdapter] = {}
        for key, adapter in self.settings.adapters.items():
            family = _normalize_with_aliases(key, self._aliases)
            self._adapter_overrides[family] = adapter

        self._score_kwargs: Dict[TaskFamily, Dict[str, Any]] = {}
        for key, task_kwargs in self.settings.score_kwargs.items():
            family = _normalize_with_aliases(key, self._aliases)
            self._score_kwargs[family] = dict(task_kwargs)

        self._importer = importer
        self._adapter_cache: Dict[TaskFamily, RewardAdapter] = {}

    def clear_cache(self) -> None:
        """Forget lazily imported modules, primarily for worker reconfiguration."""

        self._adapter_cache.clear()

    def normalize_task(self, value: Any) -> TaskFamily:
        """Normalize a task using built-in and configured aliases."""

        return _normalize_with_aliases(value, self._aliases)

    def resolve_task(self, sample: Mapping[str, Any]) -> TaskFamily:
        """Resolve the first recognized task field in a sample."""

        if not isinstance(sample, Mapping):
            raise TypeError("Reward samples must be mappings.")

        examined: List[str] = []
        for field_name in self.settings.task_fields:
            if field_name not in sample:
                continue
            value = sample.get(field_name)
            if value is None or not str(value).strip():
                continue
            examined.append(f"{field_name}={value!r}")
            try:
                return self.normalize_task(value)
            except UnknownTaskError:
                continue

        details = ", ".join(examined) if examined else "no task fields"
        supported = ", ".join(family.value for family in TaskFamily)
        raise UnknownTaskError(
            f"Could not route reward sample ({details}). Supported families: {supported}."
        )

    def _load_adapter(self, family: TaskFamily) -> RewardAdapter:
        cached = self._adapter_cache.get(family)
        if cached is not None:
            return cached

        adapter = self._adapter_overrides.get(family)
        if adapter is None:
            module_path = self._module_paths.get(family)
            if module_path is None:
                raise RewardContractError(f"No adapter module is configured for {family.value!r}.")
            importer = self._importer or importlib.import_module
            adapter = importer(module_path)
            if (
                family is TaskFamily.VIDEO_QA
                and module_path == DEFAULT_MODULE_PATHS[TaskFamily.VIDEO_QA]
            ):
                factory = getattr(adapter, "VideoQAAdapter", None)
                if callable(factory):
                    adapter = factory(
                        self.settings.video_qa_options,
                        self.settings.video_qa_require_answer_tags,
                    )

        scorer = getattr(adapter, "compute_score", None)
        if not callable(scorer):
            raise RewardContractError(
                f"Adapter for {family.value!r} has no callable compute_score."
            )
        self._adapter_cache[family] = adapter
        return adapter

    @staticmethod
    def _normalize_metrics(
        family: TaskFamily,
        raw_metrics: Any,
    ) -> RewardResult:
        if not isinstance(raw_metrics, Mapping):
            raise RewardContractError(f"{family.value} adapter returned a non-mapping score.")
        if "overall" not in raw_metrics:
            raise RewardContractError(f"{family.value} adapter score is missing 'overall'.")

        metrics: RewardResult = {}
        for key, value in raw_metrics.items():
            if not isinstance(key, str):
                raise RewardContractError(
                    f"{family.value} adapter returned a non-string metric key."
                )
            if isinstance(value, (str, bytes)) or value is None:
                raise RewardContractError(f"{family.value} metric {key!r} must be numeric.")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise RewardContractError(
                    f"{family.value} metric {key!r} must be numeric."
                ) from error
            if not math.isfinite(number):
                raise RewardContractError(f"{family.value} metric {key!r} must be finite.")
            metrics[key] = number
        return metrics

    def _score_group(
        self,
        family: TaskFamily,
        rows: List[Dict[str, Any]],
        kwargs: Mapping[str, Any],
    ) -> List[RewardResult]:
        adapter = self._load_adapter(family)
        call_kwargs = dict(self._score_kwargs.get(family, {}))
        call_kwargs.update(kwargs)
        raw_results = adapter.compute_score(rows, **call_kwargs)
        if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
            raise RewardContractError(f"{family.value} adapter returned a non-sequence batch.")
        if len(raw_results) != len(rows):
            raise RewardContractError(
                f"{family.value} adapter returned {len(raw_results)} scores "
                f"for {len(rows)} samples."
            )
        return [self._normalize_metrics(family, metrics) for metrics in raw_results]

    def compute_reward(
        self,
        sample: Mapping[str, Any],
        response: str,
        **kwargs: Any,
    ) -> RewardResult:
        """Compute one task-specific reward while preserving its telemetry."""

        if not isinstance(sample, Mapping):
            raise TypeError("Reward samples must be mappings.")
        if not isinstance(response, str):
            raise TypeError("Reward responses must be strings.")
        family = self.resolve_task(sample)
        row = dict(sample)
        row["response"] = response
        return self._score_group(family, [row], kwargs)[0]

    def compute_score(
        self,
        batch: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> List[RewardResult]:
        """Score a mixed-task batch and restore its original row order."""

        if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
            raise ValueError("compute_score expects a sequence of samples.")

        grouped: Dict[TaskFamily, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
        for index, sample in enumerate(batch):
            if not isinstance(sample, Mapping):
                raise TypeError(f"Reward sample at index {index} must be a mapping.")
            family = self.resolve_task(sample)
            grouped[family].append((index, dict(sample)))

        ordered: List[Optional[RewardResult]] = [None] * len(batch)
        for family, indexed_rows in grouped.items():
            rows = [row for _, row in indexed_rows]
            scores = self._score_group(family, rows, kwargs)
            for (index, _), score in zip(indexed_rows, scores):
                ordered[index] = score

        if any(score is None for score in ordered):
            raise RewardContractError("A routed reward row was not scored.")
        return [score for score in ordered if score is not None]

    def _ground_truth(self, sample: Mapping[str, Any]) -> Any:
        for field_name in self.settings.ground_truth_fields:
            if field_name in sample and sample[field_name] is not None:
                return sample[field_name]
        names = ", ".join(self.settings.ground_truth_fields)
        raise RewardContractError(f"Reward sample is missing ground truth; checked: {names}.")

    @staticmethod
    def _builder_for(
        family: TaskFamily,
        adapter: RewardAdapter,
    ) -> Callable[..., Any]:
        builder = getattr(
            adapter,
            "build_oracle_response_from_ground_truth",
            None,
        )
        if callable(builder):
            return builder
        raise RewardContractError(f"Adapter for {family.value!r} has no oracle response builder.")

    @staticmethod
    def _call_builder(
        family: TaskFamily,
        builder: Callable[..., Any],
        ground_truth: Any,
        extra: Any,
    ) -> Optional[str]:
        try:
            signature = inspect.signature(builder)
            signature.bind(ground_truth, extra)
        except (TypeError, ValueError):
            result = builder(ground_truth)
        else:
            result = builder(ground_truth, extra)
        if result is not None and not isinstance(result, str):
            raise RewardContractError(
                f"{family.value} oracle builder returned "
                f"{type(result).__name__}, expected str or None."
            )
        return result

    def _build_for_family(
        self,
        family: TaskFamily,
        ground_truth: Any,
        extra: Any,
    ) -> Optional[str]:
        adapter = self._load_adapter(family)
        builder = self._builder_for(family, adapter)
        return self._call_builder(family, builder, ground_truth, extra)

    def build_oracle_response(
        self,
        sample: Mapping[str, Any],
    ) -> Optional[str]:
        """Build an oracle response using the sample's resolved task."""

        if not isinstance(sample, Mapping):
            raise TypeError("Reward samples must be mappings.")
        family = self.resolve_task(sample)
        ground_truth = self._ground_truth(sample)
        return self._build_for_family(family, ground_truth, dict(sample))

    def build_oracle_response_from_ground_truth(
        self,
        ground_truth: Any,
        extra: Any = None,
    ) -> Optional[str]:
        """Build a trainer-compatible oracle response.

        When ``extra`` carries task metadata it is authoritative. Otherwise the
        family is inferred conservatively from the ground-truth structure.
        """

        family: Optional[TaskFamily] = None
        if isinstance(extra, Mapping):
            has_task_metadata = any(
                field_name in extra
                and extra[field_name] is not None
                and str(extra[field_name]).strip()
                for field_name in self.settings.task_fields
            )
            if has_task_metadata:
                family = self.resolve_task(extra)
        elif isinstance(extra, (str, TaskFamily)) and str(extra).strip():
            family = self.normalize_task(extra)

        if family is None:
            family = infer_task_from_ground_truth(ground_truth)
        return self._build_for_family(family, ground_truth, extra)


_ANSWER_CONTENT_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)
_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"


def _answer_content(value: Any) -> str:
    text = str(value or "").strip()
    matches = _ANSWER_CONTENT_RE.findall(text)
    return matches[-1].strip() if matches else text


def _json_payload(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple)):
        return value
    text = _answer_content(value)
    fenced = re.fullmatch(
        r"\s*```(?:json)?\s*(.*?)\s*```\s*",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced is not None:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass

    for opening, closing in (("{", "}"), ("[", "]")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start < 0 or end <= start:
            continue
        try:
            return json.loads(text[start : end + 1])
        except (TypeError, ValueError):
            continue
    return None


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_box(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(_is_number(coordinate) for coordinate in value)
    )


def _infer_from_json(payload: Any) -> Optional[TaskFamily]:
    if isinstance(payload, (list, tuple)):
        if _is_box(payload):
            return TaskFamily.SPATIAL_GROUNDING
        if len(payload) == 2 and all(_is_number(value) for value in payload):
            return TaskFamily.TEMPORAL_GROUNDING
        if payload and all(isinstance(item, Mapping) for item in payload):
            if any(any(key in item for key in ("bbox_2d", "bbox", "box")) for item in payload):
                return TaskFamily.SPATIAL_GROUNDING
        return None

    if not isinstance(payload, Mapping):
        return None
    keys = {_slug(key): key for key in payload}
    if "positive_points" in keys or "negative_points" in keys:
        return TaskFamily.SEGMENTATION

    boxes_key = keys.get("boxes")
    time_key = keys.get("time")
    if boxes_key is not None and time_key is not None:
        return TaskFamily.SPATIAL_TEMPORAL_GROUNDING
    if boxes_key is not None:
        boxes = payload[boxes_key]
        if isinstance(boxes, Mapping):
            return TaskFamily.TRACKING
        if _is_box(boxes):
            return TaskFamily.SPATIAL_GROUNDING

    for key_name in ("bbox_2d", "bbox", "box"):
        source_key = keys.get(key_name)
        if source_key is not None and _is_box(payload[source_key]):
            return TaskFamily.SPATIAL_GROUNDING

    if time_key is not None:
        time_value = payload[time_key]
        if (
            isinstance(time_value, (list, tuple))
            and len(time_value) == 2
            and all(_is_number(value) for value in time_value)
        ):
            return TaskFamily.TEMPORAL_GROUNDING
    if "start" in keys and "end" in keys:
        return TaskFamily.TEMPORAL_GROUNDING
    return None


def infer_task_from_ground_truth(ground_truth: Any) -> TaskFamily:
    """Infer a builder family when the trainer provides no task metadata."""

    payload_family = _infer_from_json(_json_payload(ground_truth))
    if payload_family is not None:
        return payload_family

    text = _answer_content(ground_truth)
    if re.fullmatch(r"[A-Ha-h]", text):
        return TaskFamily.VIDEO_QA
    if re.fullmatch(_NUMBER_RE, text):
        return TaskFamily.SPATIAL_INTELLIGENCE
    if re.search(
        rf"{_NUMBER_RE}\s*(?:to|and|[-–—])\s*{_NUMBER_RE}",
        text,
        flags=re.IGNORECASE,
    ):
        return TaskFamily.TEMPORAL_GROUNDING
    if re.search(
        rf"\(\s*{_NUMBER_RE}\s*,\s*{_NUMBER_RE}\s*\)"
        rf"\s*,\s*\(\s*{_NUMBER_RE}\s*,\s*{_NUMBER_RE}\s*\)",
        text,
    ):
        return TaskFamily.SPATIAL_GROUNDING

    raise UnknownTaskError(
        "Could not infer an oracle task from ground truth. Pass task metadata "
        "through extra, for example {'problem_type': 'tracking'}."
    )


_DEFAULT_ROUTER = RewardRouter()


def _router_for(
    settings: Optional[RouterSettings],
    router: Optional[RewardRouter],
) -> RewardRouter:
    if settings is not None and router is not None:
        raise ValueError("Pass either settings or router, not both.")
    if router is not None:
        return router
    return RewardRouter(settings) if settings is not None else _DEFAULT_ROUTER


def compute_reward(
    sample: Mapping[str, Any],
    response: str,
    *,
    settings: Optional[RouterSettings] = None,
    router: Optional[RewardRouter] = None,
    **kwargs: Any,
) -> RewardResult:
    """Compute one reward through the default or supplied router."""

    return _router_for(settings, router).compute_reward(
        sample,
        response,
        **kwargs,
    )


def compute_score(
    batch: Sequence[Mapping[str, Any]],
    **kwargs: Any,
) -> List[RewardResult]:
    """Trainer-compatible mixed-task batch entry point."""

    settings = kwargs.pop("settings", None)
    router = kwargs.pop("router", None)
    return _router_for(settings, router).compute_score(batch, **kwargs)


def build_oracle_response(
    sample: Mapping[str, Any],
    *,
    settings: Optional[RouterSettings] = None,
    router: Optional[RewardRouter] = None,
) -> Optional[str]:
    """Build one oracle response from a typed sample."""

    return _router_for(settings, router).build_oracle_response(sample)


def build_oracle_response_from_ground_truth(
    ground_truth: Any,
    extra: Any = None,
    *,
    settings: Optional[RouterSettings] = None,
    router: Optional[RewardRouter] = None,
) -> Optional[str]:
    """Trainer-compatible oracle builder entry point."""

    return _router_for(
        settings,
        router,
    ).build_oracle_response_from_ground_truth(ground_truth, extra)
