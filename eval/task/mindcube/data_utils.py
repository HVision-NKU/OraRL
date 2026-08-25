"""Official MindCube-Tiny data loading and record normalization."""

from __future__ import annotations

import io
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

_TASK_DIR = Path(__file__).resolve().parents[1]
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from canonical_data import load_json_records  # noqa: E402

OFFICIAL_DATA_SOURCE = (
    "hf://datasets/oscarqjh/MindCube_lmmseval"
    "@7dd2725d9bd4149f2aad00a9843f72a3824da003/tiny/combined"
)
OFFICIAL_EXPECTED_SAMPLES = 1050
OFFICIAL_GROUP_COUNTS = {
    "rotation": 200,
    "among": 600,
    "around": 250,
}
HF_DATASET_PREFIX = "hf://datasets/"
CHOICES = list("ABCDEFGH")


def parse_hf_dataset_source(
    source: str,
) -> tuple[str, str | None, str, str]:
    """Parse hf://datasets/<org>/<name>[@revision]/<config>/<split>."""
    if not source.startswith(HF_DATASET_PREFIX):
        raise ValueError(f"Not a Hugging Face dataset source: {source}")
    payload = source[len(HF_DATASET_PREFIX) :]
    try:
        dataset_spec, config, split = payload.rsplit("/", 2)
    except ValueError as error:
        raise ValueError(
            "MindCube Hugging Face source must use "
            "hf://datasets/<org>/<name>/<config>/<split>"
        ) from error
    dataset_name, separator, revision = dataset_spec.rpartition("@")
    if not separator:
        dataset_name = dataset_spec
        revision = None
    if not dataset_name or "/" not in dataset_name or not config or not split:
        raise ValueError(f"Invalid MindCube Hugging Face source: {source}")
    return dataset_name, revision, config, split


def mindcube_group(record: Dict[str, Any]) -> str:
    """Return one of the official Rotation/Among/Around settings."""
    for key in ("task", "setting"):
        value = str(record.get(key) or "").strip().lower()
        if value in OFFICIAL_GROUP_COUNTS:
            return value

    sample_id = str(
        record.get("id")
        or record.get("index")
        or record.get("sample_id")
        or ""
    ).strip().lower()
    match = re.match(r"^(rotation|among|around)(?:_|$)", sample_id)
    return match.group(1) if match else "unknown"


def _validate_records(
    records: Any,
    expected_samples: int,
) -> None:
    if expected_samples <= 0:
        return
    actual = len(records)
    if actual != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} official MindCube-Tiny samples, "
            f"got {actual}. Refusing to evaluate a reduced or different split."
        )
    if expected_samples != OFFICIAL_EXPECTED_SAMPLES:
        return

    if hasattr(records, "column_names") and "id" in records.column_names:
        groups = Counter(
            mindcube_group({"id": sample_id}) for sample_id in records["id"]
        )
    else:
        groups = Counter(mindcube_group(record) for record in records)
    actual_groups = {
        group: groups.get(group, 0) for group in OFFICIAL_GROUP_COUNTS
    }
    if actual_groups != OFFICIAL_GROUP_COUNTS or groups.get("unknown", 0):
        raise RuntimeError(
            "Official MindCube-Tiny must contain "
            f"{OFFICIAL_GROUP_COUNTS}, got {dict(groups)}."
        )


def load_mindcube_records(
    source: str,
    *,
    chunk: int = 1,
    index: int = 0,
    expected_samples: int = OFFICIAL_EXPECTED_SAMPLES,
    cache_dir: str | None = None,
) -> List[Dict[str, Any]]:
    """Load and shard the official split before decoding its images."""
    if chunk <= 0 or index < 0 or index >= chunk:
        raise ValueError(f"Invalid shard index {index}/{chunk}")

    if source.startswith(HF_DATASET_PREFIX):
        from datasets import load_dataset

        dataset_name, revision, config, split = parse_hf_dataset_source(
            source
        )
        records = load_dataset(
            dataset_name,
            config,
            split=split,
            revision=revision,
            cache_dir=cache_dir or None,
        )
        _validate_records(records, expected_samples)
        if chunk > 1:
            records = records.shard(
                num_shards=chunk,
                index=index,
                contiguous=False,
            )
        return [records[row_index] for row_index in range(len(records))]

    if source.endswith((".jsonl", ".json")):
        records = load_json_records(source)
        _validate_records(records, expected_samples)
        return records[index::chunk]

    import pandas as pd

    if not source.endswith(".parquet"):
        raise ValueError(
            "MindCube data must be the official Hugging Face source or a "
            f"local official parquet, got: {source}"
        )
    records = pd.read_parquet(source).to_dict("records")
    _validate_records(records, expected_samples)
    return records[index::chunk]


def decode_hf_image(
    value: Any,
    min_image_bytes: int,
) -> Image.Image | None:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    raw = None
    path = None
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = value
    elif isinstance(value, str):
        path = value

    if raw is None and path:
        return Image.open(path).convert("RGB")
    if raw is None:
        return None
    raw = bytes(raw)
    if min_image_bytes > 0 and len(raw) < min_image_bytes:
        return None
    return Image.open(io.BytesIO(raw)).convert("RGB")


def decode_mindcube_images(
    record: Dict[str, Any],
    max_images: int,
    min_image_bytes: int,
) -> List[Image.Image]:
    values = record.get("images")
    if values is not None:
        if hasattr(values, "tolist"):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            values = [values]
    else:
        values = [
            record.get(f"image_{idx}") for idx in range(max_images)
        ]

    images = []
    for value in values[:max_images]:
        image = decode_hf_image(value, min_image_bytes)
        if image is not None:
            images.append(image)
    return images


def normalize_choices(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def labels_from_question(question: str) -> List[str]:
    labels = []
    for label in re.findall(
        r"(?:^|[\s,])([A-H])\s*[:\).]",
        question or "",
    ):
        if label not in labels:
            labels.append(label)
    return labels or list("ABCD")


def mindcube_prompt(
    record: Dict[str, Any],
    prompt_tail: str,
) -> tuple[str, List[str]]:
    question = str(record.get("question") or "").strip()
    choices = normalize_choices(record.get("choices"))
    labels = CHOICES[: len(choices)] if choices else labels_from_question(
        question
    )

    official_prompt = str(record.get("input_prompt") or "").strip()
    if official_prompt:
        return official_prompt, labels
    if choices:
        options = "\n".join(
            f"{label}. {text}"
            for label, text in zip(labels, choices)
        )
        return f"{question}\nOptions:\n{options}\n{prompt_tail}", labels
    return f"{question}\n{prompt_tail}", labels


def mindcube_answer_value(record: Dict[str, Any]) -> Any:
    answer = record.get("gt_answer")
    return record.get("answer") if answer is None else answer
