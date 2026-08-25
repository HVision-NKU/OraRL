"""ReVSI evaluation with vLLM data-parallel sharding.

This keeps the existing OraRL VSI execution style (one vLLM process per GPU,
JSONL shards, centralized merge) while using ReVSI's official prompt,
question-type grouping, and macro-averaged metrics.

Metrics:
    * numerical tasks: MRA (mean relative accuracy)
    * multiple-choice tasks: exact-match ACC on the first token/letter

Outputs one JSONL shard per process and a summary JSON after launcher merge.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from canonical_data import load_json_records  # noqa: E402


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


NUMERICAL_TASKS = {
    "object_abs_distance",
    "object_counting_single",
    "object_counting_multiple",
    "object_size_estimation",
    "room_size_estimation_single",
    "room_size_estimation_multiple",
}

MC_TASKS = {
    "object_rel_direction_forward_easy",
    "object_rel_direction_backward_easy",
    "object_rel_direction_forward_hard",
    "object_rel_direction_backward_hard",
    "object_rel_distance_closest",
    "object_rel_distance_farthest",
    "route_planning",
}

DIRECTION_PREFIXES = ("object_rel_direction",)

REPORT_ORDER = [
    ("Abs. Dist", "object_abs_distance", "MRA"),
    ("Obj. Count", "object_counting", "MRA"),
    ("Rel. Dir", "object_rel_direction", "ACC"),
    ("Rel. Dis", "object_rel_distance", "ACC"),
    ("Obj. Size", "object_size_estimation", "MRA"),
    ("Room Size", "room_size_estimation", "MRA"),
    ("Route Plan", "route_planning", "ACC"),
]

COMPOSITE_METRICS = {
    "object_counting": (
        "object_counting_single",
        "object_counting_multiple",
    ),
    "object_rel_direction": (
        "object_rel_direction_forward_easy",
        "object_rel_direction_backward_easy",
        "object_rel_direction_forward_hard",
        "object_rel_direction_backward_hard",
    ),
    "object_rel_distance": (
        "object_rel_distance_closest",
        "object_rel_distance_farthest",
    ),
    "room_size_estimation": (
        "room_size_estimation_single",
        "room_size_estimation_multiple",
    ),
}

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)
FINAL_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+answer|answer|答案)\s*(?:is|=|是|为)?\s*[:：]?\s*"
    r"([-+]?\d+(?:\.\d+)?|[A-Da-d])\b",
    flags=re.IGNORECASE,
)
LOOSE_NUMERIC_INSTRUCTION_RE = re.compile(
    r"\s*Please answer the question using a single word or phrase\.?\s*",
    flags=re.IGNORECASE,
)
NUMERIC_STRICT_INSTRUCTION = (
    "Output only a number. Do not output option letters, units, explanations, "
    "or punctuation."
)
MC_STRICT_INSTRUCTION = "Answer with the option letter only."


def _strip_answer_tags(text: str) -> str:
    text = text or ""
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def strip_think_block(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # Prefer content after the last closing tag. This handles common Qwen
    # no-think outputs such as "</think> B" and complete think blocks.
    parts = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        text = parts[-1]
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def fuzzy_matching(pred: str) -> str:
    pred = _strip_answer_tags(strip_think_block(pred))
    pred = pred.strip()
    if not pred:
        return ""
    m = FINAL_ANSWER_RE.search(pred)
    if m:
        return m.group(1).rstrip(".").strip()
    # Official VSI code uses the first whitespace-separated token and strips
    # trailing punctuation.  After stripping dangling think tags, this recovers
    # outputs like "</think> B" -> "B" and "</think> 2" -> "2".
    return pred.split()[0].rstrip(".。,:：").strip()


def extract_numeric_prediction(pred: str) -> str:
    """Robust numeric extraction for VSI numerical tasks.

    Some RL checkpoints emit option-like prefixes on numeric answers, e.g.
    "A.2" or "A\n\n4".  For numerical tasks, recover the number while keeping
    MC parsing unchanged.
    """
    text = _strip_answer_tags(strip_think_block(pred)).strip()
    if not text:
        return ""
    m = FINAL_ANSWER_RE.search(text)
    if m and re.search(r"\d", m.group(1)):
        return m.group(1).rstrip(".").strip()
    m = re.match(r"^\s*[A-Da-d]\s*[\.\):：\-]?\s*([-+]?\d+(?:\.\d+)?)\b", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return m.group(0).strip() if m else fuzzy_matching(pred)


def clean_prediction_for_task(question_type: str, pred: str) -> str:
    if question_type in NUMERICAL_TASKS:
        return extract_numeric_prediction(pred)
    return fuzzy_matching(pred)


def to_float(pred: Any) -> Optional[float]:
    try:
        return float(str(pred).strip())
    except (ValueError, TypeError):
        return None


def exact_match(pred: str, target: Any) -> float:
    return 1.0 if str(pred).strip().lower() == str(target).strip().lower() else 0.0


def abs_dist_norm(pred: float, target: float) -> float:
    if target == 0:
        return float("inf")
    return abs(pred - target) / abs(target)


def mean_relative_accuracy(pred: float, target: float,
                           start: float = 0.5,
                           end: float = 0.95,
                           interval: float = 0.05) -> float:
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return float(accuracy.mean())


def is_direction_task(question_type: str) -> bool:
    return any(question_type.startswith(p) for p in DIRECTION_PREFIXES)


def report_key(question_type: str) -> str:
    for key, members in COMPOSITE_METRICS.items():
        if question_type in members:
            return key
    return question_type


def strengthen_prompt(prompt: str, question_type: str) -> str:
    prompt = str(prompt or "").strip()
    if question_type in NUMERICAL_TASKS:
        if NUMERIC_STRICT_INSTRUCTION in prompt:
            return prompt
        prompt = LOOSE_NUMERIC_INSTRUCTION_RE.sub("\n", prompt).strip()
        return prompt.rstrip() + "\n" + NUMERIC_STRICT_INSTRUCTION
    if question_type in MC_TASKS or is_direction_task(question_type):
        lower = prompt.lower()
        if "option" in lower and "letter" in lower:
            return prompt
        return prompt.rstrip() + "\n" + MC_STRICT_INSTRUCTION
    return prompt


def compute_sample_score(question_type: str, prediction: str,
                         ground_truth: Any) -> Tuple[str, float]:
    cleaned = clean_prediction_for_task(question_type, prediction)
    if question_type in NUMERICAL_TASKS:
        pred_val = to_float(cleaned)
        gt_val = to_float(ground_truth)
        if pred_val is None or gt_val is None or gt_val == 0:
            return "MRA", 0.0
        return "MRA", mean_relative_accuracy(pred_val, gt_val)
    if question_type in MC_TASKS or is_direction_task(question_type):
        return "ACC", exact_match(cleaned, ground_truth)
    return "UNK", 0.0


def load_rows(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    return load_json_records(path)


def get_image_list(item: Dict[str, Any]) -> List[str]:
    imgs = item.get("image_list") or item.get("images") or item.get("image") or []
    if isinstance(imgs, str):
        imgs = [imgs]
    return [str(p) for p in imgs if p]


def get_video_list(item: Dict[str, Any]) -> List[str]:
    vids = item.get("videos") or item.get("video") or []
    if isinstance(vids, str):
        vids = [vids]
    return [str(p) for p in vids if p]


def infer_video_path(item: Dict[str, Any], video_root: str = "") -> str:
    """Resolve ``ReVSI/<budget>_frame/<scene_id>.mp4``."""
    vids = get_video_list(item)
    if vids:
        p = vids[0]
        return os.path.join(video_root, p) if video_root and not os.path.isabs(p) else p

    raw = item.get("raw") if isinstance(item.get("raw"), dict) else item
    scene = str(
        item.get("scene_id")
        or item.get("scene_name")
        or raw.get("scene_id")
        or raw.get("scene_name")
        or ""
    ).strip()
    if not scene:
        return ""
    frame_budget = str(item.get("num_frames") or raw.get("num_frames") or "").strip()
    frame_budget = re.sub(r"[_-]?frame$", "", frame_budget, flags=re.IGNORECASE)
    if not frame_budget:
        return ""
    filename = scene if scene.endswith(".mp4") else f"{scene}.mp4"
    relative = os.path.join(f"{frame_budget}_frame", filename)
    return os.path.join(video_root, relative) if video_root else relative


def get_item_id(item: Dict[str, Any], idx: int) -> str:
    for key in ("id", "item_id"):
        if item.get(key) is not None:
            return str(item[key])
    return f"idx_{idx}"


# ---------------------------------------------------------------------------
# Train / Eval prompt alignment.
#
# We rewrite the eval prompts so they match the *new training* prompts
# byte-for-byte. The training prompts were rewritten by
# scripts/normalize_sft_prompts.py to:
#
#  - drop the verbose "Please answer the question using a single word or
#    phrase." instruction
#  - use a concise, format-anchored sentence on numeric tasks:
#      abs_distance : "...? Answer with a number in meters (e.g. 2.3)."
#      counting     : "...? Answer with an integer (e.g. 3)."
#      size         : "...? Answer with a number (e.g. 120)."
#      room_size    : "...? Answer with a number (e.g. 25.5)."
#  - simplify size: "the length of the longest dimension (length, width, or
#    height) of X, measured in centimeters" -> "the longest dimension
#    (length, width, or height) of X in centimeters"
#  - normalise rel_distance to the "(a, b, c, d)" + "If there are multiple
#    instances ..." form (matches VSI-Bench eval).
#  - keep route_planning steps on separate lines.
#
# All rewrites are idempotent.
# ---------------------------------------------------------------------------

# ----- numeric tasks --------------------------------------------------------

_NUM_TAIL_OLD = re.compile(
    r"\s*Please answer the question using a single word or phrase\.\s*$",
    flags=re.IGNORECASE,
)

# abs_distance
_ABS_DIST_NEW_TAIL = (
    "? Answer with a number in meters within <answer>...</answer> tags. "
    "e.g. <answer>2.3</answer>"
)
_ABS_DIST_RE = re.compile(
    r"(Measuring from the closest point of each object, what is the )"
    r"(?:direct )?(distance between the [^?]+? and the [^?]+?)"
    r"(?:\s*\(in meters\))?\?",
    flags=re.IGNORECASE,
)

# counting
_COUNTING_NEW_TAIL = (
    "? Answer with an integer within <answer>...</answer> tags. "
    "e.g. <answer>3</answer>"
)
_COUNTING_RE = re.compile(
    r"(How many [^?]+ are in this room)\?",
    flags=re.IGNORECASE,
)

# size
_SIZE_LONG_RE = re.compile(
    r"What is the length of the longest dimension \(length, width, or height\) "
    r"of (?P<obj>the [^,?]+?), measured in centimeters\?",
    flags=re.IGNORECASE,
)
# Short form already used in training-rewritten samples.
_SIZE_SHORT_RE = re.compile(
    r"What is the longest dimension \(length, width, or height\) "
    r"of (?P<obj>the [^?]+?) in centimeters\?",
    flags=re.IGNORECASE,
)
_SIZE_NEW_TAIL = (
    " Answer with a number in centimeters within <answer>...</answer> tags. "
    "e.g. <answer>120</answer>"
)

# room_size
_ROOM_SIZE_NEW = (
    "What is the size of this room in square meters? "
    "If multiple rooms are shown, estimate the combined size. "
    "Answer with a number in square meters within <answer>...</answer> tags. "
    "e.g. <answer>25.5</answer>"
)
_ROOM_SIZE_OLD_RE = re.compile(
    r"What is the size of this room \(in square meters\)\?"
    r"[ \t]*\n?[ \t]*"
    r"(?:If multiple rooms are shown, estimate the size of the combined space\.)?",
    flags=re.IGNORECASE,
)

# Generic detector for "already aligned with new tail" so all 4 numeric
# rewriters short-circuit.
_HAS_NEW_ANSWER_TAG_HINT_RE = re.compile(
    r"<answer>\.\.\.</answer>\s*tags",
    flags=re.IGNORECASE,
)

# MC: rewrite legacy direct-answer tails to the joint-SFT answer-only format.
_MC_OLD_TAIL_RE = re.compile(
    r"\s*(?:"
    r"Answer with the option'?s letter from the given choices directly\.?|"
    r"Answer with the option letter within <answer>\.\.\.</answer> tags\.?\s*(?:e\.g\.|Example:)?\s*<answer>A</answer>|"
    r"Output only the option letter inside <answer>\.\.\.</answer>\.?\s*Do not explain\.?|"
    r"Choose the best answer from the options\. Put exactly one uppercase option letter inside <answer>\.\.\.</answer>\s*Do not explain\. Example: <answer>A</answer>"
    r")\s*$",
    flags=re.IGNORECASE,
)
_MC_NEW_TAIL = (
    "Choose the best answer from the options. "
    "Put exactly one uppercase option letter inside <answer>...</answer> "
    "Do not explain. Example: <answer>A</answer>"
)

# ----- rel_distance / route_planning ----------------------------------------

_REL_DIST_TRIGGER_RE = re.compile(
    r"which\s+of\s+these\s+objects\s*\([^)]*\)\s+is\s+the\s+closest\s+to\s+",
    flags=re.IGNORECASE,
)
_REL_DIST_INSERT_RE = re.compile(
    r"(\?)(\s*\n?\s*)(Options\s*:)", flags=re.IGNORECASE
)
_REL_DIST_MULTI_INSTANCE_LINE = (
    "If there are multiple instances of an object category, measure to the closest."
)

_ROUTE_BLOCK_RE = re.compile(
    r"(turn right\.'\)\s*:)\s*(.*?)(\s*You have reached the final destination\.)",
    flags=re.IGNORECASE | re.DOTALL,
)
_ROUTE_STEP_FINDALL_RE = re.compile(r"\d+\.\s+\S")


def _fix_route_planning(prompt: str) -> str:
    def repl(m):
        head, body, tail = m.group(1), m.group(2), m.group(3)
        positions = [mt.start() for mt in _ROUTE_STEP_FINDALL_RE.finditer(body)]
        if not positions:
            return m.group(0)
        positions.append(len(body))
        parts = [body[positions[i]:positions[i + 1]].strip()
                 for i in range(len(positions) - 1)]
        parts = [p for p in parts if p]
        if not parts:
            return m.group(0)
        return head + "\n" + "\n".join(parts) + "\n" + tail.lstrip()

    return _ROUTE_BLOCK_RE.sub(repl, prompt, count=1)


def _rewrite_abs_distance(prompt: str) -> str:
    """Rewrite to "...what is the direct distance between X and Y?
    Answer with a number in meters within <answer>...</answer> tags. ..."."""
    if "Answer with a number in meters within <answer>" in prompt:
        return prompt
    # Strip any older tails (legacy "Please answer ..." or the previous
    # format-anchored "e.g. 2.3" form).
    prompt = _NUM_TAIL_OLD.sub("", prompt)
    prompt = re.sub(r"\s*Answer with a number in meters \(e\.g\.[^)]*\)\.\s*$",
                    "", prompt, flags=re.IGNORECASE)
    new_prompt = _ABS_DIST_RE.sub(
        lambda m: m.group(1) + "direct " + m.group(2) + _ABS_DIST_NEW_TAIL,
        prompt,
        count=1,
    )
    return new_prompt


def _rewrite_counting(prompt: str) -> str:
    if "Answer with an integer within <answer>" in prompt:
        return prompt
    prompt = _NUM_TAIL_OLD.sub("", prompt)
    prompt = re.sub(r"\s*Answer with an integer \(e\.g\.[^)]*\)\.\s*$",
                    "", prompt, flags=re.IGNORECASE)
    return _COUNTING_RE.sub(lambda m: m.group(1) + _COUNTING_NEW_TAIL,
                            prompt, count=1)


def _rewrite_size(prompt: str) -> str:
    if "Answer with a number in centimeters within <answer>" in prompt:
        return prompt
    prompt = _NUM_TAIL_OLD.sub("", prompt)
    prompt = re.sub(r"\s*Answer with a number \(e\.g\.[^)]*\)\.\s*$",
                    "", prompt, flags=re.IGNORECASE)
    # Long form -> new short form with anchor.
    new_prompt, n = _SIZE_LONG_RE.subn(
        lambda m: ("What is the longest dimension (length, width, or height) "
                   "of " + m.group("obj") + " in centimeters?" + _SIZE_NEW_TAIL),
        prompt,
        count=1,
    )
    if n:
        return new_prompt
    # Short form (no "measured in centimeters") -> just add the anchor.
    return _SIZE_SHORT_RE.sub(
        lambda m: ("What is the longest dimension (length, width, or height) "
                   "of " + m.group("obj") + " in centimeters?" + _SIZE_NEW_TAIL),
        prompt,
        count=1,
    )


def _rewrite_room_size(prompt: str) -> str:
    if "Answer with a number in square meters within <answer>" in prompt:
        return prompt
    prompt = _NUM_TAIL_OLD.sub("", prompt)
    prompt = re.sub(r"\s*Answer with a number \(e\.g\.[^)]*\)\.\s*$",
                    "", prompt, flags=re.IGNORECASE)
    return _ROOM_SIZE_OLD_RE.sub(_ROOM_SIZE_NEW, prompt, count=1)


def _rewrite_mc_tail(prompt: str) -> str:
    """Replace "Answer with the option's letter from the given choices
    directly." -> the new <answer>-tagged version."""
    if _MC_NEW_TAIL in prompt:
        return prompt
    if _MC_OLD_TAIL_RE.search(prompt):
        return _MC_OLD_TAIL_RE.sub("\n" + _MC_NEW_TAIL, prompt, count=1)
    return prompt


def align_prompt_to_training(question_type: str, prompt: str) -> str:
    """Rewrite eval prompts to match the (new) training phrasing.

    Idempotent: applying twice yields the same result.
    """
    if not prompt:
        return prompt

    if question_type == "object_abs_distance":
        return _rewrite_abs_distance(prompt)

    if question_type == "object_counting":
        return _rewrite_counting(prompt)

    if question_type == "object_size_estimation":
        return _rewrite_size(prompt)

    if question_type == "room_size_estimation":
        return _rewrite_room_size(prompt)

    if question_type == "object_rel_distance":
        if _REL_DIST_TRIGGER_RE.search(prompt) and (
            _REL_DIST_MULTI_INSTANCE_LINE not in prompt
        ):
            prompt = _REL_DIST_INSERT_RE.sub(
                lambda m: m.group(1) + "\n" + _REL_DIST_MULTI_INSTANCE_LINE
                + "\n" + m.group(3),
                prompt,
                count=1,
            )
        return _rewrite_mc_tail(prompt)

    if question_type == "route_planning":
        if "turn right.')" in prompt.lower() and "\n1." not in prompt:
            prompt = _fix_route_planning(prompt)
        return _rewrite_mc_tail(prompt)

    # All other MC tasks (rel_direction_*, obj_appearance_order) only need
    # the tail rewrite.
    return _rewrite_mc_tail(prompt)


# ---------------------------------------------------------------------------
# Optional strict-numeric instruction (path A diagnostic).
#
# Background: a long SFT on the 87k mixture (numeric ~17k vs MC/letter ~70k)
# pushes the model to emit single letters (mostly "B") even on numeric tasks,
# despite the training prompt's "Please answer ... single word or phrase."
# This helper appends an *additional* sentence that explicitly forbids
# letters, so we can A/B test whether the model still has numeric capability
# but is just being triggered into MC-mode.
#
# Toggled via the --strict_numeric_prompt CLI flag (default OFF).
# ---------------------------------------------------------------------------
STRICT_NUMERIC_SUFFIX = (
    "Output ONLY a single integer or decimal number (no units, no letters, no "
    "punctuation, no explanation)."
)


def apply_strict_numeric_suffix(prompt: str) -> str:
    if not prompt:
        return prompt
    if STRICT_NUMERIC_SUFFIX in prompt:
        return prompt
    return prompt.rstrip() + "\n" + STRICT_NUMERIC_SUFFIX


def normalise_item(item: Dict[str, Any], idx: int,
                   strict_numeric: bool = False,
                   video_root: str = "") -> Dict[str, Any]:
    question = str(item.get("question") or "").strip()
    gt = item.get("ground_truth")
    qtype = str(item.get("question_type") or "unknown")
    raw_options = item.get("options")
    if raw_options is None:
        options = []
    elif isinstance(raw_options, np.ndarray):
        options = raw_options.tolist()
    elif isinstance(raw_options, (list, tuple)):
        options = list(raw_options)
    else:
        options = [raw_options]
    options = [str(option) for option in options if option is not None]

    prompt_parts = ["These are frames of a video.", question]
    if qtype in MC_TASKS:
        prompt_parts.append("Options:\n" + "\n".join(options))
        prompt_parts.append(
            "Answer with the option's letter from the given choices directly."
        )
    elif qtype in NUMERICAL_TASKS:
        prompt_parts.append(
            "Answer the question using a single integer or decimal number."
        )
    prompt = "\n".join(part for part in prompt_parts if part).strip()
    if strict_numeric and qtype in NUMERICAL_TASKS:
        prompt = apply_strict_numeric_suffix(prompt)

    scene_id = str(item.get("scene_id") or "").strip()
    out = {
        "id": get_item_id(item, idx),
        "dataset": item.get("dataset", "revsi"),
        "scene_id": scene_id,
        "scene_name": scene_id,
        "num_frames": str(item.get("num_frames") or "").strip(),
        "question_type": qtype,
        "question": question,
        "prompt": prompt,
        "ground_truth": "" if gt is None else str(gt),
        "options": options,
        "image_list": get_image_list(item),
        "videos": get_video_list(item),
        "raw": item,
    }
    out["video_path"] = infer_video_path(out, video_root)
    return out


def build_conversation(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = []
    for img_path in item["image_list"]:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"file://{img_path}"},
        })
    content.append({"type": "text", "text": item["prompt"]})
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]


@lru_cache(maxsize=4096)
def get_video_frame_count(path: str) -> int:
    """Read frame-count metadata without decoding the full video."""

    from decord import VideoReader, cpu

    return len(VideoReader(path, ctx=cpu(0), num_threads=1))


def build_video_messages(item: Dict[str, Any], args) -> List[Dict[str, Any]]:
    video_item: Dict[str, Any] = {
        "type": "video",
        "video": item["video_path"],
        "total_pixels": args.video_total_pixels,
    }
    if args.exact_nframes:
        total_frames = get_video_frame_count(item["video_path"])
        # qwen_vl_utils requires nframes <= total_frames and a multiple of two.
        # ReVSI all-frame videos are not uniformly long enough for a literal
        # 128-frame request, so short videos use every available even frame.
        nframes = min(args.max_frames, total_frames)
        nframes -= nframes % 2
        if nframes < 2:
            raise ValueError(
                f"Video has fewer than two usable frames: {item['video_path']}"
            )
        video_item["nframes"] = nframes
    else:
        video_item["max_frames"] = args.max_frames
        video_item["fps"] = args.fps
    if args.video_min_pixels is not None:
        video_item["min_pixels"] = args.video_min_pixels
    if args.video_max_pixels is not None:
        video_item["max_pixels"] = args.video_max_pixels
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [video_item, {"type": "text", "text": item["prompt"]}]},
    ]


def prepare_video_for_vllm(messages: List[Dict[str, Any]], processor, patch_size: int,
                           enable_thinking: bool = False) -> Dict[str, Any]:
    """Client-side video decoding path for vLLM.

    Mirrors existing STVG/Timelens eval: apply chat template, decode video via
    qwen_vl_utils, then pass multi_modal_data to vLLM.generate.
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    _images, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_kwargs = video_kwargs or {}
    video_kwargs["do_resize"] = False

    llm_input: Dict[str, Any] = {"prompt": text}
    if video_inputs:
        llm_input["multi_modal_data"] = {"video": video_inputs}
        llm_input["mm_processor_kwargs"] = video_kwargs
    return llm_input


def summarise(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    qtype_scores = defaultdict(list)
    for r in records:
        qtype = r.get("question_type", "unknown")
        score = float(r.get("score", 0.0) or 0.0)
        qtype_scores[qtype].append(score)

    task_scores = {}
    task_means = {}
    for display, key, metric in REPORT_ORDER:
        members = COMPOSITE_METRICS.get(key, (key,))
        member_scores = [
            float(np.mean(qtype_scores[member]))
            for member in members
            if qtype_scores.get(member)
        ]
        if member_scores:
            task_mean = float(np.mean(member_scores))
            task_means[key] = task_mean
            task_scores[key] = {
                "display": display,
                "metric": metric,
                # ReVSI first averages each fine-grained question type, then
                # macro-averages related types into the reported category.
                "score": round(task_mean * 100, 2),
                "count": sum(len(qtype_scores.get(member, [])) for member in members),
            }

    num_keys = {"object_counting", "object_abs_distance", "object_size_estimation", "room_size_estimation"}
    mc_keys = {"object_rel_distance", "object_rel_direction", "route_planning"}
    num_vals = [value for key, value in task_means.items() if key in num_keys]
    mc_vals = [value for key, value in task_means.items() if key in mc_keys]
    all_vals = list(task_means.values())

    out = {
        "total": sum(len(v) for v in qtype_scores.values()),
        "task_scores": task_scores,
        "question_type_scores": {
            k: {"score": round(float(np.mean(v) * 100), 2), "count": len(v)}
            for k, v in sorted(qtype_scores.items())
        },
    }
    if num_vals:
        out["numerical_avg"] = round(float(np.mean(num_vals) * 100), 2)
    if mc_vals:
        out["mc_avg"] = round(float(np.mean(mc_vals) * 100), 2)
    if all_vals:
        out["overall_avg"] = round(float(np.mean(all_vals) * 100), 2)
    return out


def print_summary(summary: Dict[str, Any], prefix: str = "") -> None:
    print(prefix + "=" * 58, flush=True)
    print(prefix + f"ReVSI Results  (samples={summary.get('total', 0)})", flush=True)
    print(prefix + "=" * 58, flush=True)
    print(prefix + f"{'Task':<16} {'Metric':<6} {'Score':>8} {'Count':>7}", flush=True)
    print(prefix + "-" * 42, flush=True)
    for _display, key, _metric in REPORT_ORDER:
        if key not in summary.get("task_scores", {}):
            continue
        s = summary["task_scores"][key]
        print(prefix + f"{s['display']:<16} {s['metric']:<6} {s['score']:>7.2f}% {s['count']:>7}", flush=True)
    print(prefix + "-" * 42, flush=True)
    if "numerical_avg" in summary:
        print(prefix + f"{'Numerical Avg':<16} {'MRA':<6} {summary['numerical_avg']:>7.2f}%", flush=True)
    if "mc_avg" in summary:
        print(prefix + f"{'MC Avg':<16} {'ACC':<6} {summary['mc_avg']:>7.2f}%", flush=True)
    if "overall_avg" in summary:
        print(prefix + f"{'Overall Avg':<16} {'---':<6} {summary['overall_avg']:>7.2f}%", flush=True)
    print(prefix + "=" * 58, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="ReVSI evaluation via vLLM")
    p.add_argument("--model_path", required=True)
    p.add_argument("--qa_file", required=True)
    p.add_argument("--output_json_path", required=True)
    p.add_argument("--resume_dir", default="", help="Existing output dir; skip all ids already present in results_shard*.jsonl before re-sharding remaining samples.")
    p.add_argument("--task_filter", default="", help="Comma-separated question_type filter.")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--score_log_interval", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--media_mode", choices=["image", "video"], default="video",
                   help="image: use image_list; video: reconstruct/read mp4 and sample frames.")
    p.add_argument("--video_root", default="",
                   help="Root directory for relative video paths, e.g. VSI-590K root.")
    p.add_argument("--max_frames", type=int, default=128)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument(
        "--exact_nframes",
        action="store_true",
        help="Decode exactly max_frames frames; use for fixed ReVSI subsets.",
    )
    p.add_argument("--video_total_pixels", type=int, default=16777216)
    p.add_argument("--video_min_pixels", type=int, default=65536)
    p.add_argument("--video_max_pixels", type=int, default=None)
    p.add_argument(
        "--strict_numeric_prompt",
        action="store_true",
        help="Append a strict 'output only a number' suffix to numeric-task "
             "prompts. Useful when a long SFT has biased the model toward "
             "single-letter outputs on numeric questions.",
    )
    p.add_argument(
        "--enable_thinking",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable Qwen3 <think> block. Default: false (no_think). "
             "When false, vLLM is told via chat_template_kwargs to skip the "
             "<think>...</think> reasoning prefix so the model emits the "
             "<answer>...</answer> answer directly.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_rows(args.qa_file)
    items = [normalise_item(r, i, strict_numeric=args.strict_numeric_prompt,
                            video_root=args.video_root)
             for i, r in enumerate(rows)]
    if args.task_filter:
        keep = {x.strip() for x in args.task_filter.split(",") if x.strip()}
        items = [x for x in items if x["question_type"] in keep]
        print(f"Task filter: {sorted(keep)} -> {len(items)} samples", flush=True)
    if args.max_samples and len(items) > args.max_samples:
        items = items[:args.max_samples]

    # Global resume mode: remove all IDs already written by any previous shard,
    # then redistribute the remaining samples across the current world_size.
    if args.resume_dir:
        done_global = set()
        for name in os.listdir(args.resume_dir) if os.path.isdir(args.resume_dir) else []:
            if not (name.startswith("results_shard") and name.endswith(".jsonl")):
                continue
            p = os.path.join(args.resume_dir, name)
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        done_global.add(str(json.loads(line).get("id")))
                    except Exception:
                        pass
        if done_global:
            before = len(items)
            items = [x for x in items if str(x["id"]) not in done_global]
            print(f"Global resume: skip {before - len(items)} done ids from {args.resume_dir}; remaining={len(items)}", flush=True)

    # Contiguous sharding: matches official VSI script.
    total = len(items)
    chunk_size = total // args.world_size
    remainder = total % args.world_size
    start = args.rank * chunk_size + min(args.rank, remainder)
    end = start + chunk_size + (1 if args.rank < remainder else 0)
    shard_items = items[start:end]

    # Resume: skip IDs already present in the shard output.
    done = set()
    if os.path.exists(args.output_json_path):
        with open(args.output_json_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    shard_items = [x for x in shard_items if x["id"] not in done]

    if not shard_items:
        print(f"rank={args.rank}: no remaining samples", flush=True)
        return

    if args.media_mode == "video":
        missing = [x for x in shard_items if not x.get("video_path") or not os.path.isfile(x["video_path"])]
        if missing:
            ex = missing[0]
            raise FileNotFoundError(
                f"video_mode missing {len(missing)} videos; first id={ex['id']} "
                f"dataset={ex['dataset']} scene={ex['scene_name']} path={ex.get('video_path')!r}"
            )

    os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)

    # vLLM local-media permission root for image-url mode.
    all_img_paths = [p for item in shard_items for p in item["image_list"]]
    common_prefix = os.path.commonpath(all_img_paths) if all_img_paths else "/"
    if not os.path.isdir(common_prefix):
        common_prefix = os.path.dirname(common_prefix)
    num_images = max((len(x["image_list"]) for x in shard_items), default=0)

    from vllm import LLM, SamplingParams

    print("Initializing vLLM engine...", flush=True)
    print(f"Model: {args.model_path}", flush=True)
    print(f"QA: {args.qa_file}", flush=True)
    print(f"Rank: {args.rank}/{args.world_size}, samples={len(shard_items)} / total={total}", flush=True)
    print(f"Media mode: {args.media_mode}", flush=True)
    if args.media_mode == "video":
        print(f"Video: root={args.video_root} max_frames={args.max_frames} "
              f"exact_nframes={args.exact_nframes} fps={args.fps} "
              f"total_pixels={args.video_total_pixels}", flush=True)
    else:
        print(f"Images per prompt max: {num_images}, allowed_media={common_prefix}", flush=True)
    enable_thinking = (str(args.enable_thinking).lower() == "true")
    print(f"Thinking: {enable_thinking}", flush=True)

    processor = None
    patch_size = None
    if args.media_mode == "video":
        from transformers import AutoProcessor, AutoTokenizer
        processor = AutoProcessor.from_pretrained(
            args.model_path, padding_side="left", do_resize=False,
            trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        processor.tokenizer = tokenizer
        patch_size = processor.image_processor.patch_size
        print(f"Patch size: {patch_size}", flush=True)

    mm_limit = {"video": 1, "image": 1} if args.media_mode == "video" else {"image": num_images}
    llm_kwargs = dict(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        limit_mm_per_prompt=mm_limit,
    )
    if args.media_mode == "image":
        llm_kwargs["allowed_local_media_path"] = common_prefix

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # Passed through to the tokenizer's apply_chat_template inside vLLM.
    # Qwen3 chat templates expose `enable_thinking` to skip the
    # `<think>...</think>` reasoning prefix when set to False.
    chat_template_kwargs = {"enable_thinking": enable_thinking}

    t0 = time.time()
    local_records: List[Dict[str, Any]] = []
    with open(args.output_json_path, "a", encoding="utf-8") as f:
        for batch_start in tqdm(range(0, len(shard_items), args.batch_size),
                                desc=f"rank{args.rank}"):
            batch = shard_items[batch_start:batch_start + args.batch_size]
            if args.media_mode == "video":
                conversations = [build_video_messages(x, args) for x in batch]
                llm_inputs = [prepare_video_for_vllm(
                    m, processor, patch_size, enable_thinking=enable_thinking)
                    for m in conversations]
                outputs = llm.generate(llm_inputs, sampling_params=sampling)
            else:
                conversations = [build_conversation(x) for x in batch]
                outputs = llm.chat(
                    conversations,
                    sampling_params=sampling,
                    chat_template_kwargs=chat_template_kwargs,
                )
            for item, output in zip(batch, outputs):
                pred = output.outputs[0].text
                metric, score = compute_sample_score(
                    item["question_type"], pred, item["ground_truth"])
                record = {
                    "id": item["id"],
                    "dataset": item["dataset"],
                    "scene_id": item["scene_id"],
                    "scene_name": item["scene_name"],
                    "num_frames": item["num_frames"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "prompt": item["prompt"],
                    "pred": pred,
                    "pred_clean": clean_prediction_for_task(item["question_type"], pred),
                    "ground_truth": item["ground_truth"],
                    "options": item.get("options"),
                    "image_list": item["image_list"],
                    "video_path": item.get("video_path"),
                    "media_mode": args.media_mode,
                    "score": score,
                    "metric": metric,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                local_records.append(record)
            f.flush()

            done_n = min(batch_start + args.batch_size, len(shard_items))
            if done_n % args.score_log_interval == 0 or done_n == len(shard_items):
                elapsed = max(time.time() - t0, 1e-6)
                print(f"[{done_n}/{len(shard_items)}] {elapsed:.1f}s "
                      f"({done_n / elapsed:.2f} items/s)", flush=True)
                print_summary(summarise(local_records), prefix="  ")

    print("FINAL SHARD SUMMARY", flush=True)
    print_summary(summarise(local_records))


if __name__ == "__main__":
    main()
