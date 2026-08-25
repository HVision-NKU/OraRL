"""Shared helpers for RefCOCO / RefCOCO+ / RefCOCOg evaluation.

Pure-Python (no torch / vLLM / HF dependencies) so it can be imported by both
the inference script (``eval_refcoco_vllm.py``) and the offline rescorer
(``rescore_results.py``).
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Prompts — single source of truth in eval/task/eval_prompt.py
# (qwen_native matches data/joint/sft_joint_all.jsonl spatial grounding samples).
# ---------------------------------------------------------------------------
_EVAL_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_TASK_DIR not in sys.path:
    sys.path.insert(0, _EVAL_TASK_DIR)

from canonical_data import load_json_records  # noqa: E402
from eval_prompt import QWEN_NATIVE_PROMPT_SG  # noqa: E402


def _with_terminal_period(expression: str) -> str:
    expression = (expression or "").strip()
    if expression and expression[-1] not in ".?!":
        expression += "."
    return expression


def build_qwen_native_prompt(expression: str) -> str:
    # The 13k iou<0.95 training set uses the qwen-native JSON bbox format,
    # wrapped in <answer> tags for consistency with the joint SFT mixture:
    #   <image>Locate "expr." in the image. Output its bounding box in JSON
    #   format within <answer>...</answer> tags. Example: ...
    # Keep eval byte-for-byte aligned on the user text after the <image> token.
    return QWEN_NATIVE_PROMPT_SG.format(_with_terminal_period(expression))


# Alternate grounding prompt styles below are NOT used by the eval pipeline
# (eval.sh always runs qwen_native). They are retained only for the offline
# analysis / data-selection tools in this folder (compute_difficulty.py,
# select_train_data.py), so they live here rather than in eval_prompt.py.
_QWEN_OFFICIAL_PROMPT_SG = (
    'Locate every object that matches the description "{}" in the image. '
    "Report bbox coordinates in JSON format."
)
_EVAL_BENCH_SG_PROMPT = (
    'Locate "{}" in the image. Output its bounding box in JSON format '
    'within <answer>...</answer> tags. '
    'Example: <answer>[{{"bbox_2d": [123, 30, 404, 846]}}]</answer>'
)


def build_qwen_official_prompt(expression: str) -> str:
    return _QWEN_OFFICIAL_PROMPT_SG.format(expression or "")


def build_eval_bench_prompt(expression: str) -> str:
    return _EVAL_BENCH_SG_PROMPT.format(expression or "")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATASET_CONFIGS = {
    "refcoco-val": {"anno": "refcoco_val.json"},
    "refcoco-testA": {"anno": "refcoco_testA.json"},
    "refcoco-testB": {"anno": "refcoco_testB.json"},
    "refcoco-train": {"anno": "refcoco_train.json"},
    "refcoco+-val": {"anno": "refcocop_val.json"},
    "refcoco+-testA": {"anno": "refcocop_testA.json"},
    "refcoco+-testB": {"anno": "refcocop_testB.json"},
    "refcoco+-train": {"anno": "refcocop_train.json"},
    "refcocop-val": {"anno": "refcocop_val.json"},
    "refcocop-testA": {"anno": "refcocop_testA.json"},
    "refcocop-testB": {"anno": "refcocop_testB.json"},
    "refcocop-train": {"anno": "refcocop_train.json"},
    "refcocog-val": {"anno": "refcocog_val.json"},
    "refcocog-test": {"anno": "refcocog_test.json"},
    "refcocog-train": {"anno": "refcocog_train.json"},
}
CANONICAL_DATASET_NAMES = {
    "refcoco-val": "refcoco_val",
    "refcoco-testA": "refcoco_test_a",
    "refcoco-testB": "refcoco_test_b",
    "refcoco+-val": "refcocop_val",
    "refcoco+-testA": "refcocop_test_a",
    "refcoco+-testB": "refcocop_test_b",
    "refcocog-val": "refcocog_val",
    "refcocog-test": "refcocog_test",
}
for _canonical_name in CANONICAL_DATASET_NAMES.values():
    DATASET_CONFIGS.setdefault(
        _canonical_name,
        {"anno": f"{_canonical_name}.jsonl"},
    )


def _first_existing_path(paths: List[str]) -> Optional[str]:
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return None


def _resolve_annotation_path(bench_dir: str, anno_name: str, dataset_name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    canonical_name = CANONICAL_DATASET_NAMES.get(dataset_name, dataset_name)
    candidates = [
        os.path.join(bench_dir, f"{canonical_name}.jsonl"),
        os.path.join(bench_dir, "annotations", f"{canonical_name}.jsonl"),
        os.path.join(bench_dir, anno_name),
        os.path.join(bench_dir, "rec_jsons_processed", anno_name),
        os.path.join(bench_dir, "annotations", anno_name),
        os.path.join(bench_dir, "annotations", "rec_jsons_processed", anno_name),
        os.path.join(bench_dir, f"{dataset_name}.json"),
        os.path.join(bench_dir, "rec_jsons_processed", f"{dataset_name}.json"),
        os.path.join(here, "rec_jsons_processed", anno_name),
    ]
    anno_path = _first_existing_path(candidates)
    if anno_path is None:
        raise FileNotFoundError(
            f"Could not find annotations for {dataset_name!r}. Tried: "
            + ", ".join(candidates)
        )
    return anno_path


def _resolve_image_root(bench_dir: str, anno_path: str, sample_image_ref: str) -> str:
    if not sample_image_ref or os.path.isabs(sample_image_ref):
        return ""

    override = os.getenv("SPATIAL_GROUNDING_IMAGE_ROOT", "").strip()
    if override:
        return override

    anno_dir = os.path.dirname(os.path.abspath(anno_path))
    bench_parent = os.path.dirname(os.path.abspath(bench_dir))
    rel_dir = os.path.dirname(sample_image_ref)
    candidates = [
        bench_dir,
        os.path.join(bench_dir, "images"),
        os.path.join(bench_dir, "Spatial-Grounding"),
        os.path.join(bench_parent, "Spatial-Grounding"),
        anno_dir,
        os.path.dirname(anno_dir),
    ]
    for root in candidates:
        probe = os.path.join(root, rel_dir) if rel_dir else root
        if os.path.isdir(probe):
            return root
    return bench_dir


def _resolve_image_path(image_ref: str, image_root: str) -> str:
    if not image_ref:
        return ""
    if os.path.isabs(image_ref):
        return image_ref
    direct = os.path.join(image_root, image_ref)
    if os.path.exists(direct):
        return direct
    basename_path = os.path.join(image_root, os.path.basename(image_ref))
    if os.path.exists(basename_path):
        return basename_path
    return direct


def _as_box(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def load_annotations(bench_dir: str, dataset_name: str) -> List[Dict[str, Any]]:
    cfg = DATASET_CONFIGS.get(
        dataset_name,
        {"anno": f"{dataset_name}.jsonl"},
    )
    anno_path = _resolve_annotation_path(bench_dir, cfg["anno"], dataset_name)
    if anno_path.endswith(".jsonl"):
        raw = load_json_records(anno_path)
    else:
        with open(anno_path, "r") as f:
            raw = json.load(f)

    rows = raw if isinstance(raw, list) else list(raw.values())
    first_image_ref = ""
    for item in rows:
        first_image_ref = item.get("image_path") or item.get("image") or item.get("path") or ""
        if first_image_ref:
            break
    image_root = _resolve_image_root(bench_dir, anno_path, first_image_ref)

    annos: List[Dict[str, Any]] = []
    for idx, item in enumerate(rows):
        expression = (
            item.get("expression")
            or item.get("normal_caption")
            or item.get("caption")
            or item.get("query")
            or item.get("problem")
            or ""
        )
        bbox = (
            _as_box(item.get("bbox"))
            or _as_box(item.get("normalized_solution"))
            or _as_box(item.get("gt_bbox"))
            or _as_box(item.get("solution"))
        )
        image_ref = item.get("image_path") or item.get("image") or item.get("path") or ""
        if not expression or bbox is None or not image_ref:
            continue

        annos.append({
            "problem_id": item.get("problem_id", item.get("id", idx)),
            "image_path": _resolve_image_path(image_ref, image_root),
            "expression": re.sub(r"\s+", " ", str(expression)).strip().strip("."),
            "bbox": bbox,
            "width": item.get("width"),
            "height": item.get("height"),
        })
    return annos


# ---------------------------------------------------------------------------
# Bounding-box extraction / scoring
# ---------------------------------------------------------------------------

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = r"-?\d+(?:\.\d+)?"


def _clean_jsonish(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _iter_json_values(text: str) -> List[Any]:
    decoder = json.JSONDecoder()
    values: List[Any] = []
    for i, ch in enumerate(text):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def _append_candidate(
    out: List[Dict[str, Any]],
    box: Any,
    label: Optional[str] = None,
) -> None:
    parsed = _as_box(box)
    if parsed is not None:
        out.append({"bbox": parsed, "label": label})


def _extract_from_json_value(value: Any, out: List[Dict[str, Any]],
                             label: Optional[str] = None) -> None:
    if isinstance(value, dict):
        local_label = value.get("label") or value.get("name") or value.get("category") or label
        for key in ("bbox", "bbox_2d", "box", "boxes", "bboxes"):
            if key not in value:
                continue
            box_value = value[key]
            if _as_box(box_value) is not None:
                _append_candidate(out, box_value, local_label)
            elif isinstance(box_value, list):
                for entry in box_value:
                    if _as_box(entry) is not None:
                        _append_candidate(out, entry, local_label)
                    else:
                        _extract_from_json_value(entry, out, local_label)
        for key in ("objects", "items", "regions", "detections", "predictions"):
            if key in value:
                _extract_from_json_value(value[key], out, local_label)
    elif isinstance(value, list):
        if _as_box(value) is not None:
            _append_candidate(out, value, label)
        else:
            for entry in value:
                _extract_from_json_value(entry, out, label)


def extract_bboxes(text: str) -> List[Dict[str, Any]]:
    """Extract candidate bboxes from Qwen JSON and legacy R1 answer formats."""
    if not text:
        return []

    candidates: List[Dict[str, Any]] = []
    chunks = [m.group(1) for m in ANSWER_RE.finditer(text)] or [text]
    for chunk in chunks:
        cleaned = _clean_jsonish(chunk)
        for value in _iter_json_values(cleaned):
            _extract_from_json_value(value, candidates)

        pair_pat = re.compile(
            rf"\(({NUMBER_RE})\s*,\s*({NUMBER_RE})\)\s*,?\s*"
            rf"\(({NUMBER_RE})\s*,\s*({NUMBER_RE})\)"
        )
        for match in pair_pat.finditer(cleaned):
            _append_candidate(candidates, [float(x) for x in match.groups()])

        bracket_pat = re.compile(
            rf"\[\s*({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*,\s*"
            rf"({NUMBER_RE})\s*,\s*({NUMBER_RE})\s*\]"
        )
        for match in bracket_pat.finditer(cleaned):
            _append_candidate(candidates, [float(x) for x in match.groups()])

    if candidates:
        return _dedupe_candidates(candidates)

    nums = re.findall(NUMBER_RE, ANSWER_RE.sub(r"\1", text))
    if len(nums) >= 4:
        _append_candidate(candidates, [float(x) for x in nums[:4]])
    return _dedupe_candidates(candidates)


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique: List[Dict[str, Any]] = []
    for cand in candidates:
        bbox = cand.get("bbox")
        if bbox is None:
            continue
        key = tuple(round(float(x), 6) for x in bbox)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def compute_iou_2d(box1: Any, box2: Any) -> float:
    b1 = _as_box(box1)
    b2 = _as_box(box2)
    if b1 is None or b2 is None:
        return 0.0

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 1e-12 else 0.0


def select_pred_bbox(candidates: List[Dict[str, Any]], gt_box: Any,
                     mode: str = "first") -> Tuple[Optional[List[float]], Optional[str]]:
    if not candidates:
        return None, None
    if mode == "best_iou":
        best = max(candidates, key=lambda c: compute_iou_2d(gt_box, c.get("bbox")))
    else:
        best = candidates[0]
    return best.get("bbox"), best.get("label")


def sanitize_video_kwargs(video_kwargs: Any, has_video: bool = False) -> Dict[str, Any]:
    if not has_video:
        return {}
    if not isinstance(video_kwargs, dict):
        return {}
    return {k: v for k, v in video_kwargs.items() if v is not None}
