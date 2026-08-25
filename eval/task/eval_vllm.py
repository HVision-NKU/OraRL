#!/usr/bin/env python3
"""Single-file vLLM evaluator for VideoMME, VideoMME-V2, VideoMMMU, MVBench, VSI, MMSI, MindCube, grounding, Tracking, and STVG.

This is the only evaluation entry point. It owns:
  - task defaults
  - output directory layout
  - GPU sharding
  - worker execution
  - result merging

Usage:
  python eval/task/eval_vllm.py --tasks videomme,vsi --model_path /path/to/model
  TASKS=mmsi bash eval/task/eval.sh
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[2]
THIS_FILE = Path(__file__).resolve()
THIS_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from canonical_data import load_json_records  # noqa: E402

# Centralized prompts (single source of truth) — see eval/task/eval_prompt.py.
from eval_prompt import (  # noqa: E402
    PROMPT_TAIL,
    VIDEO_QA_MC_TAIL,
    VSI_CENTIMETERS_TAIL,
    VSI_INTEGER_TAIL,
    VSI_MC_TAIL,
    VSI_METERS_TAIL,
    VSI_SQUARE_METERS_TAIL,
)
from mindcube.data_utils import (  # noqa: E402
    decode_mindcube_images,
    load_mindcube_records,
    mindcube_answer_value,
    mindcube_group,
    mindcube_prompt,
)


TASKS = ("videomme", "videommev2", "videommmu", "mmvu", "mvbench", "videoholmes", "longvideobench", "lvbench", "mlvu", "vsi", "mmsi", "mindcube", "spatial_grounding", "tracking", "stvg")
OUTPUT_TASK_GROUPS = {
    "videomme": "video_qa",
    "videommev2": "video_qa",
    "videommmu": "video_qa",
    "mmvu": "video_qa",
    "mvbench": "video_qa",
    "videoholmes": "video_qa",
    "longvideobench": "video_qa",
    "lvbench": "video_qa",
    "mlvu": "video_qa",
    "vsi": "spatial_intelligence",
    "mmsi": "spatial_intelligence",
    "mindcube": "spatial_intelligence",
    "stvg": "spatial_temporal_grounding",
}
CHOICES = list("ABCDEFGH")
VIDEOMMMU_CHOICES = list("ABCDEFGHIJ")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
LEGACY_MC_TAIL_RE = re.compile(
    r"\s*(?:"
    r"Answer with the option'?s letter from the given choices directly\.?|"
    r"Answer with the option letter within <answer>\.\.\.</answer> tags\.?\s*(?:e\.g\.|Example:)?\s*<answer>A</answer>|"
    r"Output only the option letter inside <answer>\.\.\.</answer>\.?\s*Do not explain\.?|"
    r"Choose the best answer from the options\. Put exactly one uppercase option letter inside <answer>\.\.\.</answer>\s*Do not explain\. Example: <answer>A</answer>"
    r")\s*$",
    re.IGNORECASE,
)
ANSWER_PHRASES = [
    "the answer is", "answer is", "the correct answer is", "correct answer is",
    "the best answer is", "best answer is", "the correct option is",
    "correct option is", "i choose", "i select", "my answer is", "答案是", "答案为",
]
FORMAT_PRIORITY = {
    "start": 10, "end": 9, "phrase": 7, "parentheses": 6, "period": 5,
    "colon": 4, "right_paren": 3, "space": 2, "fallback": 0,
}


# ---------------------------------------------------------------------------
# Common parsing / scoring helpers
# ---------------------------------------------------------------------------


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    return load_json_records(path)


def strip_think_block(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    parts = re.split(r"</think>", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        text = parts[-1]
    else:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return TAG_RE.sub("", text).strip()


def strip_answer_tags(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else (text or "").strip()


def extract_mcq_answer(response: str, choices: Optional[List[str]] = None) -> str:
    if not response or not response.strip():
        return ""
    all_choices = choices or CHOICES
    text = strip_answer_tags(strip_think_block(response)).strip()
    if not text:
        return ""
    for char in [",", ".", "!", "?", ";", ":", "'", '"', "。", "："]:
        text = text.strip(char)
    padded = " " + text + " "
    candidates = []

    for ch in all_choices:
        for token, fmt in (
            (f"({ch})", "parentheses"),
            (f"{ch}.", "period"),
            (f"{ch}:", "colon"),
            (f"{ch})", "right_paren"),
            (f"{ch} ", "space"),
        ):
            pos = padded.rfind(token)
            if pos != -1:
                candidates.append((ch, pos, fmt))

    lower = padded.lower()
    for phrase in ANSWER_PHRASES:
        idx = lower.rfind(phrase.lower())
        if idx != -1:
            after = idx + len(phrase)
            for ch in all_choices:
                m = re.search(rf"\b{re.escape(ch)}\b", padded[after:], flags=re.IGNORECASE)
                if m:
                    candidates.append((ch, after + m.start(), "phrase"))

    stripped = padded.strip()
    for ch in all_choices:
        if stripped.upper() == ch:
            candidates.append((ch, 0, "start"))
        elif stripped.startswith(ch) and (len(stripped) == 1 or not stripped[1].isalpha()):
            candidates.append((ch, 0, "start"))
        elif stripped.endswith(ch) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            candidates.append((ch, len(padded) - 1, "end"))

    if not candidates:
        for ch in all_choices:
            m = re.search(rf"\b{re.escape(ch)}\b", padded)
            if m:
                candidates.append((ch, m.start(), "fallback"))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: (FORMAT_PRIORITY.get(x[2], 0), x[1]), reverse=True)
    return candidates[0][0]


def _normalized_option_text(value: Any, label: str = "") -> str:
    text = strip_answer_tags(strip_think_block(str(value or ""))).strip()
    if label:
        text = re.sub(
            rf"^\s*{re.escape(label)}\s*[\.\):：]\s*",
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )
    return " ".join(text.split()).casefold()


def mcq_ground_truth(record: Dict[str, Any], labels: List[str]) -> str:
    raw = next(
        (
            record[key]
            for key in ("solution", "answer", "ground_truth")
            if record.get(key) not in (None, "")
        ),
        "",
    )
    normalized_raw = _normalized_option_text(raw)
    options = record.get("options") or record.get("choices") or []
    if isinstance(options, list):
        for index, option in enumerate(options):
            if index >= len(labels):
                break
            label = str(labels[index])
            if normalized_raw in {
                _normalized_option_text(option),
                _normalized_option_text(option, label),
            }:
                return label
    return extract_mcq_answer(str(raw), labels)


def require_mcq_ground_truth(
    record: Dict[str, Any],
    labels: List[str],
    *,
    task: str,
) -> str:
    answer = mcq_ground_truth(record, labels)
    if answer:
        return answer
    identifier = record.get("problem_id") or record.get("question_id") or record.get("id")
    raise ValueError(f"{task} sample {identifier!r} has no mappable ground-truth option")


def pct(correct: int, total: int) -> float:
    return round(100.0 * correct / total, 2) if total else 0.0


def render_chat_prompt(messages: List[Dict[str, Any]], processor, enable_thinking: bool = False) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def normalize_video_metadata(metadata: Any, frames: Any) -> Dict[str, Any]:
    if isinstance(metadata, dict):
        raw = dict(metadata)
    elif hasattr(metadata, "keys") and hasattr(metadata, "__getitem__"):
        raw = {k: metadata[k] for k in metadata.keys()}
    else:
        total = frames.shape[0] if hasattr(frames, "shape") else len(frames)
        raw = {"fps": 2.0, "frames_indices": list(range(total)), "total_num_frames": total}
    return {
        k: raw[k]
        for k in ("fps", "frames_indices", "total_num_frames", "video_backend")
        if k in raw
    }


_VIDEO_CACHE: "OrderedDict[str, Tuple[Any, Dict[str, Any]]]" = OrderedDict()


def load_preprocessed_video(path: str):
    import torch

    cache_size = int(os.environ.get("VIDEO_CACHE_SIZE", "1"))
    if cache_size > 0 and path in _VIDEO_CACHE:
        value = _VIDEO_CACHE.pop(path)
        _VIDEO_CACHE[path] = value
        return value

    data = torch.load(path, map_location="cpu", weights_only=False)
    frames = data["frames"]
    value = (frames, normalize_video_metadata(data.get("metadata"), frames))
    if cache_size > 0:
        _VIDEO_CACHE[path] = value
        while len(_VIDEO_CACHE) > cache_size:
            _VIDEO_CACHE.popitem(last=False)
    return value


def prepare_preprocessed_video(messages: List[Dict[str, Any]], processor, path: str) -> Dict[str, Any]:
    frames, metadata = load_preprocessed_video(path)
    return {
        "prompt": render_chat_prompt(messages, processor, False),
        "multi_modal_data": {"video": [(frames, metadata)]},
        "mm_processor_kwargs": {"do_sample_frames": False, "do_resize": False},
    }


def prepare_raw_video(messages: List[Dict[str, Any]], processor, patch_size: int) -> Dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    text = render_chat_prompt(messages, processor, False)
    _images, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_kwargs = video_kwargs or {}
    video_kwargs.setdefault("do_resize", False)
    return {
        "prompt": text,
        "multi_modal_data": {"video": video_inputs},
        "mm_processor_kwargs": video_kwargs,
    }


# ---------------------------------------------------------------------------
# Task defaults and output layout
# ---------------------------------------------------------------------------


def model_tags(model_path: str) -> Tuple[str, str]:
    path = Path(model_path.rstrip("/"))
    if path.name == "huggingface" and path.parent.name == "actor" and path.parent.parent.name.startswith("global_step_"):
        return path.parent.parent.parent.name, path.parent.parent.name
    if path.name.startswith("checkpoint-"):
        return path.parent.parent.name, path.name
    return path.name, "base"


def make_output_dir(task: str, model_path: str, setting: str) -> Path:
    family, ckpt = model_tags(model_path)
    group = OUTPUT_TASK_GROUPS.get(task)
    task_directory = Path(group) / task if group is not None else Path(task)
    out = (
        PROJECT_DIR
        / "outputs"
        / family
        / task_directory
        / ckpt
        / setting
        / time.strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def split_gpus(gpus: str, tp_size: int) -> List[str]:
    values = [x.strip() for x in gpus.split(",") if x.strip()]
    if not values:
        raise ValueError("empty GPU list")
    duplicates = sorted({x for x in values if values.count(x) > 1})
    if duplicates:
        raise ValueError(f"Duplicate GPU ids in GPUS={gpus}: {duplicates}")
    if len(values) % tp_size != 0:
        raise ValueError(f"NUM_GPUS={len(values)} must be divisible by TP_SIZE={tp_size}")
    return [",".join(values[i:i + tp_size]) for i in range(0, len(values), tp_size)]


def shard_records(records: List[Dict[str, Any]], chunk: int, index: int) -> List[Dict[str, Any]]:
    if chunk <= 1:
        return records
    mode = os.environ.get("SHARD_MODE", "contiguous").strip().lower()
    if mode in {"round_robin", "stride", "strided"}:
        return records[index::chunk]
    n = len(records)
    start = n * index // chunk
    end = n * (index + 1) // chunk
    return records[start:end]


def iter_prefetched_batches(
    records: List[Dict[str, Any]],
    batch_size: int,
    prepare_batch: Callable[[int, List[Dict[str, Any]]], Any],
    prefetch_batches: int = 1,
) -> Iterator[Any]:
    """Prepare upcoming eval batches in a background thread while vLLM runs."""
    starts = list(range(0, len(records), batch_size))
    if not starts:
        return
    prefetch_batches = max(0, int(prefetch_batches or 0))
    if prefetch_batches <= 0:
        for start in starts:
            yield prepare_batch(start, records[start:start + batch_size])
        return

    with ThreadPoolExecutor(max_workers=1) as ex:
        pending: Dict[int, Future] = {}
        submit_pos = 0

        def submit_until_full() -> None:
            nonlocal submit_pos
            while submit_pos < len(starts) and len(pending) < prefetch_batches + 1:
                start = starts[submit_pos]
                pending[start] = ex.submit(prepare_batch, start, records[start:start + batch_size])
                submit_pos += 1

        submit_until_full()
        for start in starts:
            fut = pending.pop(start)
            yield fut.result()
            submit_until_full()


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing required env var {name}; set it in eval/task/eval.sh or the shell.")
    return value


def env_value(name: str, default: Any) -> Any:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def env_int(name: str, default: int) -> int:
    return int(env_value(name, default))


def env_float(name: str, default: float) -> float:
    return float(env_value(name, default))


def task_defaults(task: str) -> Dict[str, Any]:
    if task == "videomme":
        return {
            "setting": env_required("VIDEOMME_SETTING"),
            "data_file": env_required("VIDEOMME_DATA_FILE"),
            "video_base": env_required("VIDEOMME_VIDEO_BASE"),
            "video_dir": env_required("VIDEOMME_VIDEO_DIR"),
            "preprocessed_video_dir": env_required("VIDEOMME_PREPROCESSED_VIDEO_DIR"),
            "batch_size": env_int("VIDEOMME_BATCH_SIZE", 4),
            "max_model_len": env_int("VIDEOMME_MAX_MODEL_LEN", 32768),
            "max_new_tokens": env_int("VIDEOMME_MAX_NEW_TOKENS", 16),
            "max_num_batched_tokens": env_int("VIDEOMME_MAX_NUM_BATCHED_TOKENS", 32768),
            "gpu_memory_utilization": env_float("VIDEOMME_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("VIDEOMME_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("VIDEOMME_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("VIDEOMME_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("VIDEOMME_MAX_FRAMES", 384),
            "fps": env_int("VIDEOMME_FPS", 2),
            "max_samples": env_int("VIDEOMME_MAX_SAMPLES", 0),
        }
    if task == "videommev2":
        return {
            "setting": env_required("VIDEOMMEV2_SETTING"),
            "data_file": env_required("VIDEOMMEV2_DATA_FILE"),
            "video_base": env_required("VIDEOMMEV2_VIDEO_BASE"),
            "video_dir": env_required("VIDEOMMEV2_VIDEO_DIR"),
            "preprocessed_video_dir": env_required("VIDEOMMEV2_PREPROCESSED_VIDEO_DIR"),
            "batch_size": env_int("VIDEOMMEV2_BATCH_SIZE", 1),
            "max_model_len": env_int("VIDEOMMEV2_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("VIDEOMMEV2_MAX_NEW_TOKENS", 128),
            "max_num_batched_tokens": env_int("VIDEOMMEV2_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("VIDEOMMEV2_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("VIDEOMMEV2_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("VIDEOMMEV2_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("VIDEOMMEV2_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("VIDEOMMEV2_MAX_FRAMES", 384),
            "fps": env_int("VIDEOMMEV2_FPS", 2),
            "prompt_mode": str(env_value("VIDEOMMEV2_PROMPT_MODE", "default")),
            "answer_filter": str(env_value("VIDEOMMEV2_ANSWER_FILTER", "")),
            "max_samples": env_int("VIDEOMMEV2_MAX_SAMPLES", 0),
        }
    if task == "videommmu":
        return {
            "setting": env_required("VIDEOMMMU_SETTING"),
            "data_file": env_required("VIDEOMMMU_DATA_FILE"),
            "video_root": env_required("VIDEOMMMU_VIDEO_ROOT"),
            "batch_size": env_int("VIDEOMMMU_BATCH_SIZE", 4),
            "max_model_len": env_int("VIDEOMMMU_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("VIDEOMMMU_MAX_NEW_TOKENS", 64),
            "max_num_batched_tokens": env_int("VIDEOMMMU_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("VIDEOMMMU_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("VIDEOMMMU_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("VIDEOMMMU_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("VIDEOMMMU_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("VIDEOMMMU_MAX_FRAMES", 128),
            "fps": env_int("VIDEOMMMU_FPS", 2),
            "prompt_mode": str(env_value("VIDEOMMMU_PROMPT_MODE", "default")),
            "image_max_pixels": env_int("VIDEOMMMU_IMAGE_MAX_PIXELS", 1048576),
            "enable_thinking": env_value("VIDEOMMMU_ENABLE_THINKING", "false").lower() == "true",
            "temperature": env_float("VIDEOMMMU_TEMPERATURE", 0.0),
            "top_p": env_float("VIDEOMMMU_TOP_P", 1.0),
            "top_k": env_int("VIDEOMMMU_TOP_K", -1),
            "presence_penalty": env_float("VIDEOMMMU_PRESENCE_PENALTY", 0.0),
            "min_p": env_float("VIDEOMMMU_MIN_P", 0.0),
        }
    if task == "mmvu":
        return {
            "setting": env_required("MMVU_SETTING"),
            "data_file": env_required("MMVU_DATA_FILE"),
            "video_root": env_value("MMVU_VIDEO_ROOT", ""),
            "batch_size": env_int("MMVU_BATCH_SIZE", 4),
            "max_model_len": env_int("MMVU_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("MMVU_MAX_NEW_TOKENS", 64),
            "max_num_batched_tokens": env_int("MMVU_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("MMVU_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("MMVU_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("MMVU_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("MMVU_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("MMVU_MAX_FRAMES", 128),
            "fps": env_int("MMVU_FPS", 2),
            "prompt_mode": str(env_value("MMVU_PROMPT_MODE", "default")),
            "enable_thinking": env_value("MMVU_ENABLE_THINKING", "false").lower() == "true",
            "temperature": env_float("MMVU_TEMPERATURE", 0.0),
            "top_p": env_float("MMVU_TOP_P", 1.0),
            "top_k": env_int("MMVU_TOP_K", -1),
            "presence_penalty": env_float("MMVU_PRESENCE_PENALTY", 0.0),
            "min_p": env_float("MMVU_MIN_P", 0.0),
            "max_samples": env_int("MMVU_MAX_SAMPLES", 0),
        }
    if task == "mvbench":
        return {
            "setting": env_required("MVBENCH_SETTING"),
            "data_file": env_required("MVBENCH_DATA_FILE"),
            "video_root": env_value("MVBENCH_VIDEO_ROOT", ""),
            "batch_size": env_int("MVBENCH_BATCH_SIZE", 4),
            "max_model_len": env_int("MVBENCH_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("MVBENCH_MAX_NEW_TOKENS", 64),
            "max_num_batched_tokens": env_int("MVBENCH_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("MVBENCH_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("MVBENCH_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("MVBENCH_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("MVBENCH_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("MVBENCH_MAX_FRAMES", 128),
            "fps": env_int("MVBENCH_FPS", 2),
            "prompt_mode": str(env_value("MVBENCH_PROMPT_MODE", "default")),
            "enable_thinking": env_value("MVBENCH_ENABLE_THINKING", "false").lower() == "true",
            "temperature": env_float("MVBENCH_TEMPERATURE", 0.0),
            "top_p": env_float("MVBENCH_TOP_P", 1.0),
            "top_k": env_int("MVBENCH_TOP_K", -1),
            "presence_penalty": env_float("MVBENCH_PRESENCE_PENALTY", 0.0),
            "min_p": env_float("MVBENCH_MIN_P", 0.0),
            "max_samples": env_int("MVBENCH_MAX_SAMPLES", 0),
        }
    if task == "videoholmes":
        return {
            "setting": env_required("VIDEOHOLMES_SETTING"),
            "data_file": env_required("VIDEOHOLMES_DATA_FILE"),
            "video_root": env_value("VIDEOHOLMES_VIDEO_ROOT", ""),
            "batch_size": env_int("VIDEOHOLMES_BATCH_SIZE", 4),
            "max_model_len": env_int("VIDEOHOLMES_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("VIDEOHOLMES_MAX_NEW_TOKENS", 64),
            "max_num_batched_tokens": env_int("VIDEOHOLMES_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("VIDEOHOLMES_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("VIDEOHOLMES_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("VIDEOHOLMES_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("VIDEOHOLMES_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("VIDEOHOLMES_MAX_FRAMES", 128),
            "fps": env_int("VIDEOHOLMES_FPS", 2),
            "prompt_mode": str(env_value("VIDEOHOLMES_PROMPT_MODE", "default")),
            "enable_thinking": env_value("VIDEOHOLMES_ENABLE_THINKING", "false").lower() == "true",
            "temperature": env_float("VIDEOHOLMES_TEMPERATURE", 0.0),
            "top_p": env_float("VIDEOHOLMES_TOP_P", 1.0),
            "top_k": env_int("VIDEOHOLMES_TOP_K", -1),
            "presence_penalty": env_float("VIDEOHOLMES_PRESENCE_PENALTY", 0.0),
            "min_p": env_float("VIDEOHOLMES_MIN_P", 0.0),
            "max_samples": env_int("VIDEOHOLMES_MAX_SAMPLES", 0),
        }
    if task == "longvideobench":
        return {
            "setting": env_required("LONGVIDEOBENCH_SETTING"),
            "data_file": env_required("LONGVIDEOBENCH_DATA_FILE"),
            "video_root": env_value("LONGVIDEOBENCH_VIDEO_ROOT", ""),
            "subtitle_root": env_value("LONGVIDEOBENCH_SUBTITLE_ROOT", ""),
            "use_subtitles": env_value("LONGVIDEOBENCH_USE_SUBTITLES", "true").lower() == "true",
            "batch_size": env_int("LONGVIDEOBENCH_BATCH_SIZE", 4),
            "max_model_len": env_int("LONGVIDEOBENCH_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("LONGVIDEOBENCH_MAX_NEW_TOKENS", 64),
            "max_num_batched_tokens": env_int("LONGVIDEOBENCH_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("LONGVIDEOBENCH_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("LONGVIDEOBENCH_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("LONGVIDEOBENCH_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("LONGVIDEOBENCH_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("LONGVIDEOBENCH_MAX_FRAMES", 128),
            "fps": env_int("LONGVIDEOBENCH_FPS", 2),
            "prompt_mode": str(env_value("LONGVIDEOBENCH_PROMPT_MODE", "default")),
            "enable_thinking": env_value("LONGVIDEOBENCH_ENABLE_THINKING", "false").lower() == "true",
            "temperature": env_float("LONGVIDEOBENCH_TEMPERATURE", 0.0),
            "top_p": env_float("LONGVIDEOBENCH_TOP_P", 1.0),
            "top_k": env_int("LONGVIDEOBENCH_TOP_K", -1),
            "presence_penalty": env_float("LONGVIDEOBENCH_PRESENCE_PENALTY", 0.0),
            "min_p": env_float("LONGVIDEOBENCH_MIN_P", 0.0),
            "max_samples": env_int("LONGVIDEOBENCH_MAX_SAMPLES", 0),
        }
    if task == "lvbench":
        return {
            "setting": env_required("LVBENCH_SETTING"),
            "data_file": env_required("LVBENCH_DATA_FILE"),
            "video_root": env_value("LVBENCH_VIDEO_ROOT", ""),
            "batch_size": env_int("LVBENCH_BATCH_SIZE", 4),
            "max_model_len": env_int("LVBENCH_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("LVBENCH_MAX_NEW_TOKENS", 64),
            "max_num_batched_tokens": env_int("LVBENCH_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("LVBENCH_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("LVBENCH_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("LVBENCH_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("LVBENCH_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("LVBENCH_MAX_FRAMES", 128),
            "fps": env_int("LVBENCH_FPS", 2),
            "prompt_mode": str(env_value("LVBENCH_PROMPT_MODE", "default")),
            "enable_thinking": env_value("LVBENCH_ENABLE_THINKING", "false").lower() == "true",
            "temperature": env_float("LVBENCH_TEMPERATURE", 0.0),
            "top_p": env_float("LVBENCH_TOP_P", 1.0),
            "top_k": env_int("LVBENCH_TOP_K", -1),
            "presence_penalty": env_float("LVBENCH_PRESENCE_PENALTY", 0.0),
            "min_p": env_float("LVBENCH_MIN_P", 0.0),
        }
    if task == "mlvu":
        return {
            "setting": env_required("MLVU_SETTING"),
            "data_file": env_required("MLVU_DATA_FILE"),
            "video_root": env_value("MLVU_VIDEO_ROOT", ""),
            "batch_size": env_int("MLVU_BATCH_SIZE", 4),
            "max_model_len": env_int("MLVU_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("MLVU_MAX_NEW_TOKENS", 64),
            "max_num_batched_tokens": env_int("MLVU_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("MLVU_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("MLVU_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("MLVU_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("MLVU_VIDEO_TOTAL_PIXELS", 0),
            "max_frames": env_int("MLVU_MAX_FRAMES", 128),
            "fps": env_int("MLVU_FPS", 2),
            "prompt_mode": str(env_value("MLVU_PROMPT_MODE", "default")),
            "enable_thinking": env_value("MLVU_ENABLE_THINKING", "false").lower() == "true",
            "temperature": env_float("MLVU_TEMPERATURE", 0.0),
            "top_p": env_float("MLVU_TOP_P", 1.0),
            "top_k": env_int("MLVU_TOP_K", -1),
            "presence_penalty": env_float("MLVU_PRESENCE_PENALTY", 0.0),
            "min_p": env_float("MLVU_MIN_P", 0.0),
            "max_samples": env_int("MLVU_MAX_SAMPLES", 0),
        }
    if task == "vsi":
        return {
            "setting": env_required("VSI_SETTING"),
            "data_file": env_required("VSI_DATA_FILE"),
            "preprocessed_video_dir": env_required("VSI_PREPROCESSED_VIDEO_DIR"),
            "batch_size": env_int("VSI_BATCH_SIZE", 16),
            "max_model_len": env_int("VSI_MAX_MODEL_LEN", 32768),
            "max_new_tokens": env_int("VSI_MAX_NEW_TOKENS", 1024),
            "gpu_memory_utilization": env_float("VSI_GPU_MEMORY_UTILIZATION", 0.90),
            "video_min_pixels": env_int("VSI_VIDEO_MIN_PIXELS", 65536),
            "video_max_pixels": env_int("VSI_VIDEO_MAX_PIXELS", 262144),
            "video_total_pixels": env_int("VSI_VIDEO_TOTAL_PIXELS", 16777216),
            "max_frames": env_int("VSI_MAX_FRAMES", 128),
            "fps": env_int("VSI_FPS", 2),
        }
    if task == "mmsi":
        return {
            "setting": env_required("MMSI_SETTING"),
            "data_file": env_required("MMSI_DATA_FILE"),
            "max_images": env_int("MMSI_MAX_IMAGES", 8),
            "batch_size": env_int("MMSI_BATCH_SIZE", 16),
            "max_model_len": env_int("MMSI_MAX_MODEL_LEN", 32768),
            "max_new_tokens": env_int("MMSI_MAX_NEW_TOKENS", 1024),
            "max_num_batched_tokens": env_int("MMSI_MAX_NUM_BATCHED_TOKENS", 32768),
            "gpu_memory_utilization": env_float("MMSI_GPU_MEMORY_UTILIZATION", 0.90),
        }
    if task == "mindcube":
        return {
            "setting": env_required("MINDCUBE_SETTING"),
            "data_file": env_required("MINDCUBE_DATA_FILE"),
            "max_images": env_int("MINDCUBE_MAX_IMAGES", 3),
            "expected_samples": env_int(
                "MINDCUBE_EXPECTED_SAMPLES", 1050
            ),
            "batch_size": env_int("MINDCUBE_BATCH_SIZE", 16),
            "max_model_len": env_int("MINDCUBE_MAX_MODEL_LEN", 32768),
            "max_new_tokens": env_int("MINDCUBE_MAX_NEW_TOKENS", 1024),
            "max_num_batched_tokens": env_int("MINDCUBE_MAX_NUM_BATCHED_TOKENS", 32768),
            "gpu_memory_utilization": env_float("MINDCUBE_GPU_MEMORY_UTILIZATION", 0.90),
        }
    if task == "spatial_grounding":
        return {
            "setting": env_required("SPATIAL_GROUNDING_SETTING"),
            "data_file": "",
            "processor_path": env_value("SPATIAL_GROUNDING_PROCESSOR_PATH", os.environ.get("MODEL_PATH", "")),
            "bench_dir": env_required("SPATIAL_GROUNDING_BENCH_DIR"),
            "datasets": env_value(
                "SPATIAL_GROUNDING_DATASETS",
                "refcoco-val,refcoco-testA,refcoco-testB,refcoco+-val,refcoco+-testA,refcoco+-testB,refcocog-val,refcocog-test",
            ),
            "prompt_style": env_value("SPATIAL_GROUNDING_PROMPT_STYLE", "qwen_native"),
            "coord_system": env_value("SPATIAL_GROUNDING_COORD_SYSTEM", "norm1000"),
            "bbox_select": env_value("SPATIAL_GROUNDING_BBOX_SELECT", "first"),
            "enable_thinking": env_value("SPATIAL_GROUNDING_ENABLE_THINKING", "false").lower() == "true",
            "min_tokens": env_int("SPATIAL_GROUNDING_MIN_TOKENS", 64),
            "total_tokens": env_int("SPATIAL_GROUNDING_TOTAL_TOKENS", 1024),
            "batch_size": env_int("SPATIAL_GROUNDING_BATCH_SIZE", 64),
            "max_model_len": env_int("SPATIAL_GROUNDING_MAX_MODEL_LEN", 32768),
            "max_new_tokens": env_int("SPATIAL_GROUNDING_MAX_NEW_TOKENS", 1024),
            "max_num_batched_tokens": env_int("SPATIAL_GROUNDING_MAX_NUM_BATCHED_TOKENS", 32768),
            "gpu_memory_utilization": env_float("SPATIAL_GROUNDING_GPU_MEMORY_UTILIZATION", 0.85),
            "max_samples": env_int("SPATIAL_GROUNDING_MAX_SAMPLES", 0),
        }
    if task == "tracking":
        return {
            "setting": env_required("TRACKING_SETTING"),
            "data_file": "",
            "processor_path": env_value("TRACKING_PROCESSOR_PATH", os.environ.get("MODEL_PATH", "")),
            "bench_dir": env_required("TRACKING_BENCH_DIR"),
            "base_prefix": env_value("TRACKING_BASE_PREFIX", env_required("TRACKING_BENCH_DIR")),
            "datasets": env_value("TRACKING_DATASETS", "eval_got10k"),
            "video_min_pixels": env_int("TRACKING_VIDEO_MIN_PIXELS", 4096),
            "video_max_pixels": env_int("TRACKING_VIDEO_MAX_PIXELS", 786432),
            "video_total_pixels": env_int("TRACKING_VIDEO_TOTAL_PIXELS", 8388608),
            "max_frames": env_int("TRACKING_MAX_FRAMES", 32),
            "fps": env_int("TRACKING_FPS", 1),
            "batch_size": env_int("TRACKING_BATCH_SIZE", 64),
            "max_model_len": env_int("TRACKING_MAX_MODEL_LEN", 32768),
            "max_new_tokens": env_int("TRACKING_MAX_NEW_TOKENS", 8192),
            "max_num_batched_tokens": env_int("TRACKING_MAX_NUM_BATCHED_TOKENS", 32768),
            "gpu_memory_utilization": env_float("TRACKING_GPU_MEMORY_UTILIZATION", 0.95),
            "enable_thinking": env_value("TRACKING_ENABLE_THINKING", "false").lower() == "true",
            "prompt_mode": env_value("TRACKING_PROMPT_MODE", "default"),
            "chunked_reprompt": env_int("TRACKING_CHUNKED_REPROMPT", 0),
            "reprompt_use_gt_first_box": env_value("TRACKING_REPROMPT_USE_GT_FIRST_BOX", "0") in {"1", "true", "True"},
            "max_samples": env_int("TRACKING_MAX_SAMPLES", 0),
        }
    if task == "stvg":
        return {
            "setting": env_required("STVG_SETTING"),
            "data_file": "",
            "processor_path": env_value("STVG_PROCESSOR_PATH", os.environ.get("MODEL_PATH", "")),
            "bench_dir": env_required("STVG_BENCH_DIR"),
            "base_prefix": env_value("STVG_BASE_PREFIX", env_required("STVG_BENCH_DIR")),
            "datasets": env_value("STVG_DATASETS", "eval_stvg"),
            "video_min_pixels": env_int("STVG_VIDEO_MIN_PIXELS", 65536),
            "video_max_pixels": env_int("STVG_VIDEO_MAX_PIXELS", 393216),
            "video_total_pixels": env_int("STVG_VIDEO_TOTAL_PIXELS", 10485760),
            "max_frames": env_int("STVG_MAX_FRAMES", 128),
            "fps": env_int("STVG_FPS", 2),
            "batch_size": env_int("STVG_BATCH_SIZE", 64),
            "max_model_len": env_int("STVG_MAX_MODEL_LEN", 65536),
            "max_new_tokens": env_int("STVG_MAX_NEW_TOKENS", 2048),
            "max_num_batched_tokens": env_int("STVG_MAX_NUM_BATCHED_TOKENS", 65536),
            "gpu_memory_utilization": env_float("STVG_GPU_MEMORY_UTILIZATION", 0.85),
            "enable_thinking": env_value("STVG_ENABLE_THINKING", "false").lower() == "true",
            "prompt_mode": env_value("STVG_PROMPT_MODE", "train_stvg"),
            "max_samples": env_int("STVG_MAX_SAMPLES", 0),
        }
    raise ValueError(task)


# ---------------------------------------------------------------------------
# VideoMME
# ---------------------------------------------------------------------------


def videomme_prompt(rec: Dict[str, Any], prompt_mode: str = "default") -> str:
    question = str(rec.get("question") or rec.get("problem") or "").strip()
    options = rec.get("options") or rec.get("choices") or []
    if isinstance(options, dict):
        options = [str(options[k]) for k in sorted(options)]
    elif isinstance(options, str):
        options = [x.strip() for x in options.split("\n") if x.strip()]
    else:
        options = [str(x) for x in options]
    tail = VIDEO_QA_MC_TAIL
    if prompt_mode == "explicit_ah":
        tail = (
            "There are 8 options, A through H. Answer with exactly one option "
            "letter from A, B, C, D, E, F, G, H within <answer>...</answer> tags. "
            "Example: <answer>H</answer>"
        )
    elif prompt_mode == "careful_ah":
        tail = (
            "Carefully watch the video and compare all options. The correct answer "
            "may be any option from A, B, C, D, E, F, G, or H. Do not favor earlier "
            "options. Answer with exactly one option letter within <answer>...</answer> "
            "tags. Example: <answer>G</answer>"
        )
    return (
        f"{question}\n"
        f"Options:\n" + "\n".join(options) + "\n"
        f"{tail}"
    )


def videomme_preproc_path(rec: Dict[str, Any], root: str) -> str:
    value = rec.get("preprocessed_video") or rec.get("preprocessed_video_path")
    if not value:
        return ""
    value = str(value)
    if os.path.isabs(value):
        return value
    return os.path.join(root, value)


def videomme_video_path(rec: Dict[str, Any], args) -> str:
    raw = rec.get("path") or rec.get("video_path") or rec.get("file_name")
    videos = rec.get("videos")
    if not raw and isinstance(videos, list) and videos:
        raw = videos[0]
    video_obj = rec.get("video")
    if not raw and isinstance(video_obj, dict):
        raw = video_obj.get("path") or video_obj.get("filename")
    elif not raw and isinstance(video_obj, str):
        raw = video_obj

    candidates: List[str] = []
    for base in [getattr(args, "video_dir", ""), getattr(args, "video_base", "")]:
        if not base:
            continue
        if raw:
            rel = str(raw).lstrip("./").lstrip("/")
            candidates.extend([os.path.join(base, rel), os.path.join(base, os.path.basename(rel))])
        video_id = str(rec.get("videoID") or rec.get("video_id") or "").strip()
        if video_id:
            stem, ext = os.path.splitext(video_id)
            names = [video_id] if ext else [video_id + e for e in (".mp4", ".MP4", ".mkv", ".avi", ".mov", ".webm")]
            for name in names:
                candidates.extend([os.path.join(base, name), os.path.join(base, "videos", name), os.path.join(base, "data", name)])
    if raw and os.path.isabs(str(raw)):
        candidates.insert(0, str(raw))

    seen = set()
    for path in candidates:
        if path and path not in seen:
            seen.add(path)
            if os.path.isfile(path):
                return path
    return ""


def videomme_video_content(path: str, args) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "type": "video",
        "video": path,
        "max_pixels": args.video_max_pixels,
        "max_frames": args.max_frames,
        "fps": args.fps,
    }
    if args.video_min_pixels > 0:
        item["min_pixels"] = args.video_min_pixels
    if args.video_total_pixels > 0:
        item["total_pixels"] = args.video_total_pixels
    return item


def worker_videomme(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: (str(r.get("videoID") or r.get("video_id") or ""), str(r.get("question_id") or r.get("id") or "")))
    answer_filter = {c for c in str(getattr(args, "answer_filter", "") or "").upper() if c in CHOICES}
    if answer_filter:
        records = [
            r for r in records
            if extract_mcq_answer(str(r.get("answer") or r.get("ground_truth") or ""), CHOICES) in answer_filter
        ]
    if args.max_samples > 0:
        records = records[: args.max_samples]
    records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", do_resize=False, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    patch_size = getattr(processor.image_processor, "patch_size", 16)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1)
    results = []

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs = []
        keep = []
        for rec in batch:
            preprocessed_path = videomme_preproc_path(
                rec,
                args.preprocessed_video_dir,
            )
            use_preprocessed = bool(
                args.preprocessed_video_dir
                and args.preprocessed_video_dir != "__none__"
                and preprocessed_path
                and os.path.isfile(preprocessed_path)
            )
            path = (
                preprocessed_path
                if use_preprocessed
                else videomme_video_path(rec, args)
            )
            if not path:
                print(f"[warn] VideoMME video not found: video_id={rec.get('videoID') or rec.get('video_id')}", flush=True)
                continue
            messages = [{"role": "user", "content": [
                {"type": "video", "video": path} if use_preprocessed else videomme_video_content(path, args),
                {"type": "text", "text": videomme_prompt(rec, getattr(args, "prompt_mode", "default"))},
            ]}]
            if use_preprocessed:
                inputs.append(prepare_preprocessed_video(messages, processor, path))
            else:
                inputs.append(prepare_raw_video(messages, processor, patch_size))
            keep.append(rec)
        return keep, inputs

    for keep, inputs in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for rec, out in zip(keep, outs):
            raw = out.outputs[0].text
            options = rec.get("options") or rec.get("choices") or []
            labels = CHOICES[:len(options)] if isinstance(options, list) and options else list("ABCD")
            pred = extract_mcq_answer(raw, labels)
            gt = require_mcq_ground_truth(rec, labels, task=args.task)
            results.append({
                "question_id": rec.get("question_id") or rec.get("id"),
                "videoID": rec.get("videoID") or rec.get("video_id"),
                "duration": rec.get("duration", "unknown"),
                "category": rec.get("domain") or rec.get("category") or "unknown",
                "sub_category": rec.get("sub_category", "unknown"),
                "task_category": rec.get("task_type") or rec.get("task_category") or "unknown",
                "question": rec.get("question") or rec.get("problem"),
                "answer": gt,
                "pred_answer": pred,
                "score": 1.0 if pred and gt and pred == gt else 0.0,
                "raw_prediction": raw,
            })
    write_json(args.output_dir, f"results_{args.task}_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_videomme(results))


def aggregate_videomme(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    def group(key: str, order: Optional[List[str]] = None):
        d = defaultdict(lambda: {"correct": 0, "total": 0})
        for s in samples:
            name = str(s.get(key) or "unknown")
            d[name]["total"] += 1
            d[name]["correct"] += int(s.get("score", 0))
        names = order or sorted(d)
        return {k: {"accuracy": pct(v["correct"], v["total"]), "correct": v["correct"], "total": v["total"]} for k, v in ((k, d[k]) for k in names if k in d)}

    correct = sum(int(s.get("score", 0)) for s in samples)
    parsed = sum(1 for s in samples if s.get("pred_answer"))
    labeled = sum(1 for s in samples if s.get("answer"))
    return {
        "num_samples": len(samples),
        "correct": correct,
        "accuracy": pct(correct, len(samples)),
        "parse_rate": pct(parsed, len(samples)),
        "ground_truth_rate": pct(labeled, len(samples)),
        "by_duration": group("duration", ["short", "medium", "long"]),
        "by_category": group("category"),
        "by_sub_category": group("sub_category"),
        "by_task_category": group("task_category"),
    }


# ---------------------------------------------------------------------------
# VideoMMMU
# ---------------------------------------------------------------------------


def videommmu_use_separate_images(root: str) -> bool:
    base = Path(root)
    images_dir = base / "images"
    if images_dir.is_dir() and any(images_dir.glob("*.png")):
        return True
    ot_dir = base / "Adaptation" / "test-00000-of-00001"
    return ot_dir.is_dir() and any(ot_dir.glob("*.png"))


def videommmu_index(root: str, use_separate_images: bool = False) -> Dict[str, str]:
    base = Path(root)
    index: Dict[str, str] = {}
    suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".png", ".jpg", ".jpeg", ".webp"}
    for path in base.glob("*/*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if use_separate_images and path.parent.name == "question_only":
            continue
        if use_separate_images:
            if path.name not in index:
                index[path.name] = str(path)
            continue
        # Legacy fallback: prefer question_only/*_image.mp4 when separate PNGs are unavailable.
        if path.parent.name == "question_only" or path.name not in index:
            index[path.name] = str(path)
        if path.parent.name == "question_only" and path.stem.endswith("_image"):
            original_name = path.stem[: -len("_image")] + path.suffix
            index[original_name] = str(path)
    for path in base.glob("images/*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index[path.name] = str(path)
    for path in (base / "Adaptation" / "test-00000-of-00001").glob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index[path.name] = str(path)
    for path in base.glob("*/images/*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            index.setdefault(path.name, str(path))
    return index


def videommmu_resolve(path_value: Any, root: str, index: Dict[str, str]) -> str:
    raw = str(path_value or "").strip()
    raw = raw[2:] if raw.startswith("./") else raw
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute() and path.is_file():
        return str(path)
    direct = Path(root) / raw
    if direct.is_file():
        return str(direct)
    return index.get(path.name, str(direct))


def videommmu_resolve_image(rec: Dict[str, Any], root: str, index: Dict[str, str]) -> str:
    # Only Adaptation questions carry a separate image. Comprehension/Perception
    # records have an empty image_path and must stay pure-video; do not fall back
    # to same-video figures, or we would staple the Adaptation image onto them.
    declared = str(rec.get("image_path") or rec.get("additional_path") or "").strip()
    if not declared:
        return ""

    for key in ("image_path", "additional_path"):
        resolved = videommmu_resolve(rec.get(key), root, index)
        if resolved and os.path.isfile(resolved):
            return resolved

    video_stem = Path(str(rec.get("path") or "")).stem
    if video_stem:
        by_stem = Path(root) / "images" / f"{video_stem}.png"
        if by_stem.is_file():
            return str(by_stem)

    problem_id = rec.get("problem_id")
    if problem_id is not None and str(problem_id).strip():
        by_id = Path(root) / "Adaptation" / "test-00000-of-00001" / f"{int(problem_id)}.png"
        if by_id.is_file():
            return str(by_id)
    return ""


def videommmu_safe_total_frames(video_path: str, cache: Dict[str, int]) -> int:
    if video_path in cache:
        return cache[video_path]
    counts: List[int] = []
    try:
        from decord import VideoReader, cpu

        counts.append(len(VideoReader(video_path, ctx=cpu(0))))
    except Exception:
        pass
    try:
        import torchvision.io as io

        video, _, _ = io.read_video(video_path, output_format="TCHW")
        counts.append(int(video.size(0)))
    except Exception:
        pass
    cache[video_path] = min(counts) if counts else 0
    return cache[video_path]


def videommmu_uses_appended_image(
    rec: Dict[str, Any],
    video_path: str,
    has_separate_image: bool = False,
) -> bool:
    if has_separate_image:
        return False
    if rec.get("image_path") or rec.get("additional_path"):
        return True
    return "/question_only/" in video_path.replace("\\", "/")


def videommmu_video_item(
    rec: Dict[str, Any],
    video_path: str,
    args,
    frame_count_cache: Dict[str, int],
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "min_pixels": args.video_min_pixels,
        "max_pixels": args.video_max_pixels,
    }
    if args.video_total_pixels > 0:
        item["total_pixels"] = args.video_total_pixels

    # Adaptation questions either use a separate PNG image or, as a legacy
    # fallback, question_only/*_image.mp4 with the figure appended. Use
    # fps+max_frames for both so we do not scan full source videos just to cap
    # nframes.
    if (
        rec.get("_has_separate_image")
        or videommmu_uses_appended_image(rec, video_path, bool(rec.get("_has_separate_image")))
    ):
        item["fps"] = args.fps
        item["max_frames"] = args.max_frames
        return item

    if args.max_frames > 0:
        total = videommmu_safe_total_frames(video_path, frame_count_cache)
        if total > 0:
            nframes = min(args.max_frames, total)
            nframes = max(2, nframes - (nframes % 2))
            item["nframes"] = nframes
        else:
            item["nframes"] = args.max_frames
    else:
        item["fps"] = args.fps
    return item


# OneThinker (Evaluation/Eval/eval_bench.py) answer-format tails, ported but
# WITHOUT the <think> reasoning instruction (no-think / direct-answer mode).
ONETHINK_MC_TAIL = (
    "Please answer this question based on the visual content.\n"
    "Answer directly without any explanation or reasoning. "
    "Output only the single option letter (e.g., A, B, C, D, etc.) "
    "within the <answer>...</answer> tags.\n"
    "Example:\n<answer>A</answer>"
)
ONETHINK_NUM_TAIL = (
    "Please answer this question based on the visual content.\n"
    "Answer directly without any explanation or reasoning. "
    "Output only the numerical value within the <answer>...</answer> tags.\n"
    "Example:\n<answer>3.14</answer>"
)


# OneThinker's verbatim CoT wrapper (Evaluation/Eval/eval_bench.py QUESTION_TEMPLATE
# + TYPE_TEMPLATE). Use prompt_mode=onethink_cot with a large max_new_tokens (8192).
ONETHINK_COT_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)
ONETHINK_COT_MC_TAIL = (
    "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
    "within the <answer>...</answer> tags.\n"
    "Example:\n<answer>A</answer>"
)
ONETHINK_COT_NUM_TAIL = (
    "Please provide only the numerical value within the <answer>...</answer> tags.\n"
    "Example:\n<answer>3.14</answer>"
)


def videommmu_prompt_onethink(rec: Dict[str, Any], cot: bool = False) -> Tuple[str, List[str]]:
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = normalize_choices(rec.get("options"))
    labels = VIDEOMMMU_CHOICES[:len(options)] if options else list("ABCD")
    is_numerical = str(rec.get("problem_type") or "").lower() == "numerical"
    if options and not is_numerical and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    if cot:
        tail = ONETHINK_COT_NUM_TAIL if is_numerical else ONETHINK_COT_MC_TAIL
        prompt = ONETHINK_COT_TEMPLATE.format(Question=question) + tail
    else:
        tail = ONETHINK_NUM_TAIL if is_numerical else ONETHINK_MC_TAIL
        prompt = f"{question}\n{tail}"
    return prompt, ([] if is_numerical else labels)


# Official Qwen3.5 benchmarking format (model card "Best Practices"): the model
# reasons in its native thinking mode (enable_thinking=True) and we only fix the
# final-answer format. Do NOT forbid reasoning here.
def videommmu_prompt_qwen(rec: Dict[str, Any]) -> Tuple[str, List[str]]:
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = normalize_choices(rec.get("options"))
    labels = VIDEOMMMU_CHOICES[:len(options)] if options else list("ABCD")
    is_numerical = str(rec.get("problem_type") or "").lower() == "numerical"
    if options and not is_numerical and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    if is_numerical:
        tail = (
            "Please answer the question based on the visual content. "
            "Put your final numerical answer inside <answer></answer> tags, "
            "e.g., <answer>3.14</answer>."
        )
        return f"{question}\n{tail}", []
    tail = (
        "Please answer the question based on the visual content. "
        "Put only the choice letter inside <answer></answer> tags, "
        "e.g., <answer>C</answer>."
    )
    return f"{question}\n{tail}", labels


def videommmu_prompt(
    rec: Dict[str, Any],
    has_separate_image: bool = False,
    prompt_mode: str = "default",
) -> Tuple[str, List[str]]:
    if prompt_mode == "onethink":
        return videommmu_prompt_onethink(rec, cot=False)
    if prompt_mode == "onethink_cot":
        return videommmu_prompt_onethink(rec, cot=True)
    if prompt_mode == "qwen":
        return videommmu_prompt_qwen(rec)
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = normalize_choices(rec.get("options"))
    labels = VIDEOMMMU_CHOICES[:len(options)] if options else list("ABCD")
    if options and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    has_appended_image = bool(rec.get("image_path") or rec.get("additional_path")) and not has_separate_image
    if has_appended_image:
        question = re.sub(r"<image\s*\d*>", "the image at the end of the video", question, flags=re.IGNORECASE)
        question = "The image for this question is at the end of the video.\n" + question
    else:
        question = re.sub(r"<image\s*\d*>", "the image", question, flags=re.IGNORECASE)
    if str(rec.get("problem_type") or "").lower() == "numerical":
        return (
            f"{question}\n"
            "Answer with the final number in the format <answer>NUMBER</answer>. Do not explain.",
            [],
        )
    return (
        f"{question}\n{VIDEO_QA_MC_TAIL}",
        labels,
    )


def videommmu_extract_number(text: str) -> str:
    text = strip_answer_tags(strip_think_block(text))
    match = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    return match.group(0).replace(",", "") if match else ""


def videommmu_score(rec: Dict[str, Any], raw: str, labels: List[str]) -> Tuple[str, str, float]:
    gt_text = str(rec.get("solution") or rec.get("answer") or rec.get("ground_truth") or "")
    if str(rec.get("problem_type") or "").lower() == "numerical":
        pred = videommmu_extract_number(raw)
        gt = videommmu_extract_number(gt_text)
        if not pred or not gt:
            return pred, gt, 0.0
        try:
            pv, gv = float(pred), float(gt)
        except Exception:
            return pred, gt, 0.0
        return pred, gt, 1.0 if round(pv, 2) == round(gv, 2) else 0.0
    pred = extract_mcq_answer(raw, labels)
    gt = require_mcq_ground_truth(rec, labels, task="videommmu")
    return pred, gt, 1.0 if pred and gt and pred == gt else 0.0


def prepare_video_image(
    messages: List[Dict[str, Any]],
    processor,
    media_cache_key: Optional[Tuple[Any, ...]] = None,
    media_cache: Optional[Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    text = render_chat_prompt(messages, processor, enable_thinking)
    cached = media_cache.get(media_cache_key) if media_cache is not None and media_cache_key is not None else None
    if cached is None:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        multi_modal_data: Dict[str, Any] = {}
        if image_inputs:
            multi_modal_data["image"] = image_inputs
        if video_inputs:
            multi_modal_data["video"] = video_inputs
        if media_cache is not None and media_cache_key is not None:
            # Keep only the most recent media item. Records are path-sorted, so
            # this avoids repeated decoding for multiple questions on one video
            # without growing CPU memory across the whole shard.
            media_cache.clear()
            media_cache[media_cache_key] = (multi_modal_data, video_kwargs)
    else:
        multi_modal_data, video_kwargs = cached
    return {
        "prompt": text,
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": video_kwargs,
    }


def worker_videommmu(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: (str(r.get("path") or ""), str(r.get("problem_id") or "")))
    records = shard_records(records, args.chunk, args.index)
    use_separate_images = videommmu_use_separate_images(args.preprocessed_video_dir)
    media_index = videommmu_index(args.preprocessed_video_dir, use_separate_images=use_separate_images)
    if use_separate_images:
        print("[videommmu] using subject videos + separate PNG images", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
    )
    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.presence_penalty:
        sampling_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p:
        sampling_kwargs["min_p"] = args.min_p
    sampling = SamplingParams(**sampling_kwargs)
    print(f"[videommmu] sampling={sampling_kwargs} enable_thinking={args.enable_thinking} prompt_mode={args.prompt_mode}", flush=True)
    results = []
    media_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    frame_count_cache: Dict[str, int] = {}

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs, kept, prompts, labels = [], [], {}, {}
        for j, rec in enumerate(batch):
            try:
                video_path = videommmu_resolve(rec.get("path"), args.preprocessed_video_dir, media_index)
                if not os.path.isfile(video_path):
                    print(f"[warn] skip {rec.get('problem_id')}: video not found: {video_path}", flush=True)
                    continue
                image_path = videommmu_resolve_image(rec, args.preprocessed_video_dir, media_index)
                has_separate_image = bool(image_path)
                rec = {**rec, "_has_separate_image": has_separate_image}
                prompt, labs = videommmu_prompt(rec, has_separate_image=has_separate_image, prompt_mode=args.prompt_mode)
                video_item = videommmu_video_item(rec, video_path, args, frame_count_cache)
                content: List[Dict[str, Any]] = [video_item]
                if has_separate_image:
                    image_item: Dict[str, Any] = {"type": "image", "image": image_path}
                    if args.image_max_pixels > 0:
                        image_item["max_pixels"] = args.image_max_pixels
                    content.append(image_item)
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content}]
                media_key = (
                    video_path,
                    image_path if image_path and os.path.isfile(image_path) else "",
                    args.video_min_pixels,
                    args.video_max_pixels,
                    args.video_total_pixels,
                    args.image_max_pixels,
                    video_item.get("nframes", 0),
                    video_item.get("fps", 0),
                    video_item.get("max_frames", 0),
                )
                inputs.append(prepare_video_image(messages, processor, media_key, media_cache, enable_thinking=args.enable_thinking))
                kept.append(j)
                prompts[j] = prompt
                labels[j] = labs
            except Exception as e:
                print(f"[warn] skip {start + j}: {type(e).__name__}: {e}", flush=True)
        return start, batch, inputs, kept, prompts, labels

    for start, batch, inputs, kept, prompts, labels in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for j, out in zip(kept, outs):
            rec = batch[j]
            raw = out.outputs[0].text
            pred, gt, score = videommmu_score(rec, raw, labels[j])
            video_name = Path(str(rec.get("path") or "")).name
            group = video_name.split("_")[1] if "_" in video_name else "unknown"
            results.append({
                "id": str(rec.get("problem_id") or start + j),
                "bench": "videommmu",
                "group": group,
                "question": rec.get("problem") or rec.get("question"),
                "prompt": prompts[j],
                "answer": gt,
                "pred_answer": pred,
                "raw_prediction": raw,
                "score": score,
            })
    write_json(args.output_dir, f"results_videommmu_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_image_mc("videommmu", results))


# ---------------------------------------------------------------------------
# MMVU (multiple-choice, expert-level multi-discipline video understanding)
# ---------------------------------------------------------------------------


def mmvu_labels(rec: Dict[str, Any]) -> List[str]:
    labs = rec.get("labels")
    if isinstance(labs, list) and labs:
        return [str(x) for x in labs]
    options = rec.get("options") or []
    return CHOICES[: len(options)] if options else list("ABCDE")


def mmvu_prompt(rec: Dict[str, Any], prompt_mode: str = "default") -> Tuple[str, List[str]]:
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = rec.get("options") or []
    labels = mmvu_labels(rec)
    if options and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    if prompt_mode == "qwen":
        tail = (
            "Please answer the question based on the visual content. "
            "Put only the choice letter inside <answer></answer> tags, e.g., <answer>C</answer>."
        )
    else:
        tail = VIDEO_QA_MC_TAIL
    return f"{question}\n{tail}", labels


def mmvu_video_item(rec: Dict[str, Any], video_path: str, args) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "min_pixels": args.video_min_pixels,
        "max_pixels": args.video_max_pixels,
        "fps": args.fps,
        "max_frames": args.max_frames,
    }
    if args.video_total_pixels > 0:
        item["total_pixels"] = args.video_total_pixels
    return item


def worker_mmvu(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: (str(r.get("path") or ""), str(r.get("id") or "")))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
    )
    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.presence_penalty:
        sampling_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p:
        sampling_kwargs["min_p"] = args.min_p
    sampling = SamplingParams(**sampling_kwargs)
    print(f"[mmvu] sampling={sampling_kwargs} enable_thinking={args.enable_thinking} prompt_mode={args.prompt_mode}", flush=True)
    results = []
    media_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs, kept, prompts, labels = [], [], {}, {}
        for j, rec in enumerate(batch):
            try:
                video_path = str(rec.get("path") or "")
                if not os.path.isfile(video_path):
                    print(f"[warn] skip {rec.get('id')}: video not found: {video_path}", flush=True)
                    continue
                prompt, labs = mmvu_prompt(rec, prompt_mode=args.prompt_mode)
                video_item = mmvu_video_item(rec, video_path, args)
                messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]
                media_key = (video_path, args.video_min_pixels, args.video_max_pixels, args.video_total_pixels, args.fps, args.max_frames)
                inputs.append(prepare_video_image(messages, processor, media_key, media_cache, enable_thinking=args.enable_thinking))
                kept.append(j)
                prompts[j] = prompt
                labels[j] = labs
            except Exception as e:
                print(f"[warn] skip {start + j}: {type(e).__name__}: {e}", flush=True)
        return start, batch, inputs, kept, prompts, labels

    for start, batch, inputs, kept, prompts, labels in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for j, out in zip(kept, outs):
            rec = batch[j]
            raw = out.outputs[0].text
            pred = extract_mcq_answer(raw, labels[j])
            gt = require_mcq_ground_truth(rec, labels[j], task="mmvu")
            results.append({
                "id": str(rec.get("id") or start + j),
                "bench": "mmvu",
                "group": rec.get("subject") or "unknown",
                "question": rec.get("problem") or rec.get("question"),
                "prompt": prompts[j],
                "answer": gt,
                "pred_answer": pred,
                "raw_prediction": raw,
                "score": 1.0 if pred and gt and pred == gt else 0.0,
            })
    write_json(args.output_dir, f"results_mmvu_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_image_mc("mmvu", results))


# ---------------------------------------------------------------------------
# MVBench (multi-task short-video multiple-choice)
# ---------------------------------------------------------------------------


def resolve_relative_video(path_value: Any, data_file: str, video_root: str) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    rel = raw[2:] if raw.startswith("./") else raw
    candidates = []
    if video_root:
        candidates.append(os.path.join(video_root, rel))
    candidates.append(os.path.join(os.path.dirname(data_file), rel))
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[0] if candidates else rel


def mvbench_prompt(rec: Dict[str, Any], prompt_mode: str = "default") -> Tuple[str, List[str]]:
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = rec.get("options") or []
    labels = CHOICES[: len(options)] if options else list("ABCD")
    if options and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    if prompt_mode == "qwen":
        tail = (
            "Please answer the question based on the video. "
            "Put only the choice letter inside <answer></answer> tags, e.g., <answer>C</answer>."
        )
    else:
        tail = VIDEO_QA_MC_TAIL
    return f"{question}\n{tail}", labels


def worker_mvbench(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: (str(r.get("path") or ""), int(r.get("problem_id") or r.get("id") or 0)))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
    )
    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.presence_penalty:
        sampling_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p:
        sampling_kwargs["min_p"] = args.min_p
    sampling = SamplingParams(**sampling_kwargs)
    print(f"[mvbench] sampling={sampling_kwargs} enable_thinking={args.enable_thinking} prompt_mode={args.prompt_mode}", flush=True)
    results = []
    media_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs, kept, prompts, labels = [], [], {}, {}
        for j, rec in enumerate(batch):
            try:
                video_path = resolve_relative_video(rec.get("path") or rec.get("video"), args.data_file, args.preprocessed_video_dir)
                if not os.path.isfile(video_path):
                    print(f"[warn] skip {rec.get('problem_id') or rec.get('id')}: video not found: {video_path}", flush=True)
                    continue
                prompt, labs = mvbench_prompt(rec, prompt_mode=args.prompt_mode)
                video_item = mmvu_video_item(rec, video_path, args)
                messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]
                media_key = (video_path, args.video_min_pixels, args.video_max_pixels, args.video_total_pixels, args.fps, args.max_frames)
                inputs.append(prepare_video_image(messages, processor, media_key, media_cache, enable_thinking=args.enable_thinking))
                kept.append(j)
                prompts[j] = prompt
                labels[j] = labs
            except Exception as e:
                print(f"[warn] skip {start + j}: {type(e).__name__}: {e}", flush=True)
        return start, batch, inputs, kept, prompts, labels

    for start, batch, inputs, kept, prompts, labels in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for j, out in zip(kept, outs):
            rec = batch[j]
            raw = out.outputs[0].text
            pred = extract_mcq_answer(raw, labels[j])
            gt = require_mcq_ground_truth(rec, labels[j], task="mvbench")
            results.append({
                "id": str(rec.get("problem_id") or rec.get("id") or start + j),
                "bench": "mvbench",
                "group": rec.get("original_question_type") or rec.get("problem_type") or "multiple choice",
                "question": rec.get("problem") or rec.get("question"),
                "prompt": prompts[j],
                "answer": gt,
                "pred_answer": pred,
                "raw_prediction": raw,
                "score": 1.0 if pred and gt and pred == gt else 0.0,
            })
    write_json(args.output_dir, f"results_mvbench_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_image_mc("mvbench", results))


# ---------------------------------------------------------------------------
# Video-Holmes (complex multi-clue video reasoning, all multiple-choice A-F)
# ---------------------------------------------------------------------------

# Official Video-Holmes prompt (evaluate.py) is a CoT reasoning prompt.
VIDEOHOLMES_TASK_TYPES = ["SR", "IMC", "TCI", "TA", "MHR", "PAR", "CTI"]


def videoholmes_labels(rec: Dict[str, Any]) -> List[str]:
    labs = rec.get("labels")
    if isinstance(labs, list) and labs:
        return [str(x) for x in labs]
    options = rec.get("options") or []
    return CHOICES[: len(options)] if options else list("ABCDEF")


def videoholmes_prompt(rec: Dict[str, Any], prompt_mode: str = "default") -> Tuple[str, List[str]]:
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = rec.get("options") or []
    labels = videoholmes_labels(rec)
    opts_inline = ", ".join(str(x).replace(". ", ": ", 1) for x in options)
    if prompt_mode == "holmes":
        # Verbatim official CoT prompt (TencentARC/Video-Holmes evaluate.py).
        prompt = (
            "Based on the given video, reason and answer the single-choice question. "
            "Provide your reasoning between the <think> and </think> tags, and then give "
            "your final answer between the <answer> and </answer> tags. "
            f"The question is: {question}. The options are: {opts_inline}. Your answer:"
        )
        return prompt, labels
    # Direct-answer (no-think) default, consistent with our other MC evals.
    if options and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    tail = VIDEO_QA_MC_TAIL
    return f"{question}\n{tail}", labels


def worker_videoholmes(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: (str(r.get("path") or ""), str(r.get("id") or "")))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
    )
    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.presence_penalty:
        sampling_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p:
        sampling_kwargs["min_p"] = args.min_p
    sampling = SamplingParams(**sampling_kwargs)
    print(f"[videoholmes] sampling={sampling_kwargs} enable_thinking={args.enable_thinking} prompt_mode={args.prompt_mode}", flush=True)
    results = []
    media_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs, kept, prompts, labels = [], [], {}, {}
        for j, rec in enumerate(batch):
            try:
                video_path = str(rec.get("path") or "")
                if not os.path.isfile(video_path):
                    print(f"[warn] skip {rec.get('id')}: video not found: {video_path}", flush=True)
                    continue
                prompt, labs = videoholmes_prompt(rec, prompt_mode=args.prompt_mode)
                video_item = mmvu_video_item(rec, video_path, args)
                messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]
                media_key = (video_path, args.video_min_pixels, args.video_max_pixels, args.video_total_pixels, args.fps, args.max_frames)
                inputs.append(prepare_video_image(messages, processor, media_key, media_cache, enable_thinking=args.enable_thinking))
                kept.append(j)
                prompts[j] = prompt
                labels[j] = labs
            except Exception as e:
                print(f"[warn] skip {start + j}: {type(e).__name__}: {e}", flush=True)
        return start, batch, inputs, kept, prompts, labels

    for start, batch, inputs, kept, prompts, labels in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for j, out in zip(kept, outs):
            rec = batch[j]
            raw = out.outputs[0].text
            pred = extract_mcq_answer(raw, labels[j])
            gt = require_mcq_ground_truth(rec, labels[j], task="videoholmes")
            results.append({
                "id": str(rec.get("id") or start + j),
                "bench": "videoholmes",
                "group": rec.get("q_type") or "unknown",
                "question": rec.get("problem") or rec.get("question"),
                "prompt": prompts[j],
                "answer": gt,
                "pred_answer": pred,
                "raw_prediction": raw,
                "score": 1.0 if pred and gt and pred == gt else 0.0,
            })
    write_json(args.output_dir, f"results_videoholmes_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_image_mc("videoholmes", results))


# ---------------------------------------------------------------------------
# LongVideoBench (long-form video MC, optional subtitles)
# ---------------------------------------------------------------------------


def longvideobench_prompt(
    rec: Dict[str, Any],
    prompt_mode: str = "default",
    use_subtitles: bool = True,
) -> Tuple[str, List[str]]:
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = rec.get("options") or []
    labels = rec.get("labels") if isinstance(rec.get("labels"), list) else CHOICES[: len(options)]
    subtitle = str(rec.get("subtitle_text") or "").strip() if use_subtitles else ""
    if use_subtitles and not subtitle:
        subtitle_path = str(rec.get("subtitle_path") or "").strip()
        if subtitle_path and os.path.isfile(subtitle_path):
            subtitle = Path(subtitle_path).read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
    if subtitle:
        question = f"Subtitles/transcript snippets:\n{subtitle}\n\nQuestion: {question}"
    if options and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    if prompt_mode == "qwen":
        tail = (
            "Please answer the question based on the video and any provided subtitles. "
            "Put only the choice letter inside <answer></answer> tags, e.g., <answer>C</answer>."
        )
    else:
        tail = VIDEO_QA_MC_TAIL
    return f"{question}\n{tail}", [str(x) for x in labels] if labels else list("ABCDE")


def worker_longvideobench(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: (str(r.get("path") or ""), str(r.get("id") or "")))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
    )
    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.presence_penalty:
        sampling_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p:
        sampling_kwargs["min_p"] = args.min_p
    sampling = SamplingParams(**sampling_kwargs)
    print(f"[longvideobench] sampling={sampling_kwargs} enable_thinking={args.enable_thinking} prompt_mode={args.prompt_mode}", flush=True)
    results = []
    media_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs, kept, prompts, labels = [], [], {}, {}
        for j, rec in enumerate(batch):
            try:
                video_path = str(rec.get("path") or "")
                if not os.path.isfile(video_path):
                    print(f"[warn] skip {rec.get('id')}: video not found: {video_path}", flush=True)
                    continue
                prompt, labs = longvideobench_prompt(
                    rec,
                    prompt_mode=args.prompt_mode,
                    use_subtitles=args.use_subtitles,
                )
                video_item = mmvu_video_item(rec, video_path, args)
                messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]
                media_key = (video_path, args.video_min_pixels, args.video_max_pixels, args.video_total_pixels, args.fps, args.max_frames)
                inputs.append(prepare_video_image(messages, processor, media_key, media_cache, enable_thinking=args.enable_thinking))
                kept.append(j)
                prompts[j] = prompt
                labels[j] = labs
            except Exception as e:
                print(f"[warn] skip {start + j}: {type(e).__name__}: {e}", flush=True)
        return start, batch, inputs, kept, prompts, labels

    for start, batch, inputs, kept, prompts, labels in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for j, out in zip(kept, outs):
            rec = batch[j]
            raw = out.outputs[0].text
            pred = extract_mcq_answer(raw, labels[j])
            gt = require_mcq_ground_truth(rec, labels[j], task="longvideobench")
            results.append({
                "id": str(rec.get("id") or start + j),
                "bench": "longvideobench",
                "group": rec.get("question_category") or "unknown",
                "topic_category": rec.get("topic_category") or "unknown",
                "duration_group": rec.get("duration_group"),
                "question": rec.get("problem") or rec.get("question"),
                "prompt": prompts[j],
                "answer": gt,
                "pred_answer": pred,
                "raw_prediction": raw,
                "score": 1.0 if pred and gt and pred == gt else 0.0,
            })
    write_json(args.output_dir, f"results_longvideobench_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_image_mc("longvideobench", results))


# ---------------------------------------------------------------------------
# LVBench (long-video multiple-choice)
# ---------------------------------------------------------------------------


def lvbench_prompt(rec: Dict[str, Any], prompt_mode: str = "default") -> Tuple[str, List[str]]:
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = rec.get("options") or []
    labels = CHOICES[: len(options)] if options else list("ABCD")
    if options and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    if prompt_mode == "qwen":
        tail = (
            "Please answer the question based on the video. "
            "Put only the choice letter inside <answer></answer> tags, e.g., <answer>C</answer>."
        )
    else:
        tail = VIDEO_QA_MC_TAIL
    return f"{question}\n{tail}", labels


def resolve_lvbench_video(path_value: Any, data_file: str, video_root: str) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    if os.path.isabs(raw):
        return raw
    rel = raw[2:] if raw.startswith("./") else raw
    candidates = []
    if video_root:
        candidates.append(os.path.join(video_root, rel))
    candidates.append(os.path.join(os.path.dirname(data_file), rel))
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return candidates[0] if candidates else rel


def worker_lvbench(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: (str(r.get("path") or ""), int(r.get("problem_id") or 0)))
    records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
    )
    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.presence_penalty:
        sampling_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p:
        sampling_kwargs["min_p"] = args.min_p
    sampling = SamplingParams(**sampling_kwargs)
    print(f"[lvbench] sampling={sampling_kwargs} enable_thinking={args.enable_thinking} prompt_mode={args.prompt_mode}", flush=True)
    results = []
    media_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs, kept, prompts, labels = [], [], {}, {}
        for j, rec in enumerate(batch):
            try:
                video_path = resolve_lvbench_video(rec.get("path"), args.data_file, args.preprocessed_video_dir)
                if not os.path.isfile(video_path):
                    print(f"[warn] skip {rec.get('problem_id')}: video not found: {video_path}", flush=True)
                    continue
                prompt, labs = lvbench_prompt(rec, prompt_mode=args.prompt_mode)
                video_item = mmvu_video_item(rec, video_path, args)
                messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]
                media_key = (video_path, args.video_min_pixels, args.video_max_pixels, args.video_total_pixels, args.fps, args.max_frames)
                inputs.append(prepare_video_image(messages, processor, media_key, media_cache, enable_thinking=args.enable_thinking))
                kept.append(j)
                prompts[j] = prompt
                labels[j] = labs
            except Exception as e:
                print(f"[warn] skip {start + j}: {type(e).__name__}: {e}", flush=True)
        return start, batch, inputs, kept, prompts, labels

    for start, batch, inputs, kept, prompts, labels in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for j, out in zip(kept, outs):
            rec = batch[j]
            raw = out.outputs[0].text
            pred = extract_mcq_answer(raw, labels[j])
            gt = require_mcq_ground_truth(rec, labels[j], task="lvbench")
            results.append({
                "id": str(rec.get("problem_id") or rec.get("id") or start + j),
                "bench": "lvbench",
                "group": rec.get("problem_type") or "multiple choice",
                "data_source": rec.get("data_source") or "LVBench",
                "question": rec.get("problem") or rec.get("question"),
                "prompt": prompts[j],
                "answer": gt,
                "pred_answer": pred,
                "raw_prediction": raw,
                "score": 1.0 if pred and gt and pred == gt else 0.0,
            })
    write_json(args.output_dir, f"results_lvbench_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_image_mc("lvbench", results))


# ---------------------------------------------------------------------------
# MLVU (dev, multi-task long video understanding, multiple-choice M-Avg)
# ---------------------------------------------------------------------------


def mlvu_prompt(rec: Dict[str, Any], prompt_mode: str = "default") -> Tuple[str, List[str]]:
    question = str(rec.get("problem") or rec.get("question") or "").strip()
    options = rec.get("options") or []
    labels = rec.get("labels") if isinstance(rec.get("labels"), list) else CHOICES[: len(options)]
    if options and "Options:" not in question:
        question += "\nOptions:\n" + "\n".join(str(x) for x in options)
    if prompt_mode == "qwen":
        tail = (
            "Please answer the question based on the visual content. "
            "Put only the choice letter inside <answer></answer> tags, e.g., <answer>C</answer>."
        )
    else:
        tail = VIDEO_QA_MC_TAIL
    return f"{question}\n{tail}", [str(x) for x in labels] if labels else list("ABCD")


def worker_mlvu(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: (str(r.get("path") or ""), str(r.get("id") or "")))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
    )
    sampling_kwargs: Dict[str, Any] = {
        "max_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.presence_penalty:
        sampling_kwargs["presence_penalty"] = args.presence_penalty
    if args.min_p:
        sampling_kwargs["min_p"] = args.min_p
    sampling = SamplingParams(**sampling_kwargs)
    print(f"[mlvu] sampling={sampling_kwargs} enable_thinking={args.enable_thinking} prompt_mode={args.prompt_mode}", flush=True)
    results = []
    media_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs, kept, prompts, labels = [], [], {}, {}
        for j, rec in enumerate(batch):
            try:
                video_path = str(rec.get("path") or "")
                if not os.path.isfile(video_path):
                    print(f"[warn] skip {rec.get('id')}: video not found: {video_path}", flush=True)
                    continue
                prompt, labs = mlvu_prompt(rec, prompt_mode=args.prompt_mode)
                video_item = mmvu_video_item(rec, video_path, args)
                messages = [{"role": "user", "content": [video_item, {"type": "text", "text": prompt}]}]
                media_key = (video_path, args.video_min_pixels, args.video_max_pixels, args.video_total_pixels, args.fps, args.max_frames)
                inputs.append(prepare_video_image(messages, processor, media_key, media_cache, enable_thinking=args.enable_thinking))
                kept.append(j)
                prompts[j] = prompt
                labels[j] = labs
            except Exception as e:
                print(f"[warn] skip {start + j}: {type(e).__name__}: {e}", flush=True)
        return start, batch, inputs, kept, prompts, labels

    for start, batch, inputs, kept, prompts, labels in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for j, out in zip(kept, outs):
            rec = batch[j]
            raw = out.outputs[0].text
            pred = extract_mcq_answer(raw, labels[j])
            gt = require_mcq_ground_truth(rec, labels[j], task="mlvu")
            results.append({
                "id": str(rec.get("id") or start + j),
                "bench": "mlvu",
                "group": rec.get("task_type") or "unknown",
                "question": rec.get("problem") or rec.get("question"),
                "prompt": prompts[j],
                "answer": gt,
                "pred_answer": pred,
                "raw_prediction": raw,
                "score": 1.0 if pred and gt and pred == gt else 0.0,
            })
    write_json(args.output_dir, f"results_mlvu_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_image_mc("mlvu", results))


# ---------------------------------------------------------------------------
# VSI
# ---------------------------------------------------------------------------


NUMERICAL_TASKS = {"object_abs_distance", "object_counting", "object_size_estimation", "room_size_estimation"}
VSI_REPORT = [
    ("Obj. Count", "object_counting", "MRA"),
    ("Abs. Dist", "object_abs_distance", "MRA"),
    ("Obj. Size", "object_size_estimation", "MRA"),
    ("Room Size", "room_size_estimation", "MRA"),
    ("Rel. Dis", "object_rel_distance", "ACC"),
    ("Rel. Dir", "object_rel_direction", "ACC"),
    ("Route Plan", "route_planning", "ACC"),
    ("Appr. Order", "obj_appearance_order", "ACC"),
]


def vsi_qtype(rec: Dict[str, Any]) -> str:
    return str(
        rec.get("original_question_type")
        or rec.get("question_type")
        or rec.get("problem_type")
        or "unknown"
    )


def vsi_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, dict):
        values = [str(options[k]) for k in sorted(options)]
    elif isinstance(options, (list, tuple)):
        values = [str(x) for x in options]
    else:
        values = [str(options)]
    return "\n".join(v.strip() for v in values if v.strip())


def vsi_prompt(rec: Dict[str, Any]) -> str:
    qtype = vsi_qtype(rec)
    prompt = str(rec.get("prompt") or rec.get("question") or rec.get("problem") or "").strip()
    prompt = re.sub(r"^(?:\s*<video>\s*|\s*<image>\s*)+", "", prompt).strip()
    options = vsi_options(rec.get("options"))
    if options and "Options:" not in prompt:
        prompt += "\nOptions:\n" + options
    if qtype == "object_counting" and "Answer with an integer within <answer>" not in prompt:
        prompt += " " + VSI_INTEGER_TAIL
    elif qtype == "object_abs_distance" and "Answer with a number in meters within <answer>" not in prompt:
        prompt += " " + VSI_METERS_TAIL
    elif qtype == "object_size_estimation":
        prompt = re.sub(r"What is the length of the longest dimension \(length, width, or height\) of ([^,?]+), measured in centimeters\?",
                        r"What is the longest dimension (length, width, or height) of \1 in centimeters?", prompt)
        if "Answer with a number in centimeters within <answer>" not in prompt:
            prompt += " " + VSI_CENTIMETERS_TAIL
    elif qtype == "room_size_estimation":
        prompt = "What is the size of this room in square meters? If multiple rooms are shown, estimate the combined size. " + VSI_SQUARE_METERS_TAIL
    else:
        if qtype == "object_rel_distance" and "If there are multiple instances" not in prompt:
            prompt = prompt.replace("\nOptions:", "\nIf there are multiple instances of an object category, measure to the closest.\nOptions:")
        prompt = LEGACY_MC_TAIL_RE.sub("", prompt).rstrip()
        # spatial_intelligence MC uses "Answer with the option letter within ..."
        # (no "only"), matching data/joint/sft_joint_all.jsonl.
        if "Answer with the option letter within <answer>" not in prompt:
            prompt += "\n" + VSI_MC_TAIL
    if not prompt.startswith("These are frames of a video."):
        prompt = "These are frames of a video.\n" + prompt
    return prompt


def vsi_preproc_path(rec: Dict[str, Any], root: str) -> str:
    value = rec.get("preprocessed_video") or rec.get("preprocessed_video_path")
    if not value:
        return ""
    return value if os.path.isabs(str(value)) else os.path.join(root, str(value))


def to_float(value: Any) -> Optional[float]:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def mean_relative_accuracy(pred: float, target: float) -> float:
    if target == 0:
        return 0.0
    confs = np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05 + 2))
    return float((abs(pred - target) / abs(target) <= 1 - confs).mean())


def extract_number(text: str) -> str:
    text = strip_answer_tags(strip_think_block(text))
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return m.group(0) if m else ""


def vsi_score(qtype: str, pred: str, gt: str) -> Tuple[str, str, float]:
    if qtype in NUMERICAL_TASKS:
        pv, gv = to_float(extract_number(pred)), to_float(strip_answer_tags(gt))
        if pv is None or gv is None:
            return "MRA", "", 0.0
        return "MRA", str(pv), mean_relative_accuracy(pv, gv)
    p = extract_mcq_answer(pred, list("ABCD"))
    g = extract_mcq_answer(gt, list("ABCD"))
    return "ACC", p, 1.0 if p and g and p == g else 0.0


def worker_vsi(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = read_json_or_jsonl(args.data_file)
    records.sort(key=lambda r: str(r.get("preprocessed_video") or r.get("preprocessed_video_path") or r.get("path") or r.get("scene_name") or ""))
    records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", do_resize=False, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    patch_size = getattr(processor.image_processor, "patch_size", 16)
    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp_size, gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len, trust_remote_code=True, limit_mm_per_prompt={"video": 1, "image": 1})
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0)
    out_path = Path(args.output_dir) / f"results_shard{args.index}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        def prepare_batch(start: int, batch: List[Dict[str, Any]]):
            inputs = []
            for rec in batch:
                preprocessed_path = vsi_preproc_path(
                    rec,
                    args.preprocessed_video_dir,
                )
                use_preprocessed = bool(
                    preprocessed_path and os.path.isfile(preprocessed_path)
                )
                path = (
                    preprocessed_path
                    if use_preprocessed
                    else str(
                        rec.get("video_path")
                        or rec.get("video")
                        or rec.get("path")
                        or ""
                    )
                )
                video_item: Dict[str, Any] = {
                    "type": "video",
                    "video": path,
                    "min_pixels": args.video_min_pixels,
                    "max_pixels": args.video_max_pixels,
                    "max_frames": args.max_frames,
                    "fps": args.fps,
                }
                if args.video_total_pixels > 0:
                    video_item["total_pixels"] = args.video_total_pixels
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            video_item,
                            {"type": "text", "text": vsi_prompt(rec)},
                        ],
                    },
                ]
                inputs.append(
                    prepare_preprocessed_video(
                        messages,
                        processor,
                        path,
                    )
                    if use_preprocessed
                    else prepare_raw_video(messages, processor, patch_size)
                )
            return start, batch, inputs

        for start, batch, inputs in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
            outs = llm.generate(inputs, sampling_params=sampling)
            for rec, out in zip(batch, outs):
                raw = out.outputs[0].text
                qtype = vsi_qtype(rec)
                metric, clean, score = vsi_score(qtype, raw, str(rec.get("solution") or rec.get("ground_truth") or rec.get("answer") or ""))
                row = {
                    "id": str(rec.get("problem_id") or rec.get("id") or start),
                    "dataset": rec.get("dataset") or rec.get("data_source") or "vsi",
                    "scene_name": rec.get("path") or rec.get("scene_name"),
                    "question_type": qtype,
                    "question": rec.get("question") or rec.get("problem"),
                    "prompt": vsi_prompt(rec),
                    "pred": raw,
                    "pred_clean": clean,
                    "ground_truth": strip_answer_tags(str(rec.get("solution") or rec.get("ground_truth") or rec.get("answer") or "")),
                    "options": rec.get("options"),
                    "preprocessed_video_path": vsi_preproc_path(rec, args.preprocessed_video_dir),
                    "media_mode": "video",
                    "score": score,
                    "metric": metric,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def vsi_report_key(qtype: str) -> str:
    return "object_rel_direction" if str(qtype).startswith("object_rel_direction") else str(qtype)


def aggregate_vsi(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = defaultdict(list)
    qtype_scores = defaultdict(list)
    for r in records:
        q = r.get("question_type", "unknown")
        s = float(r.get("score", 0.0) or 0.0)
        scores[vsi_report_key(q)].append(s)
        qtype_scores[q].append(s)
    task_scores = {}
    for display, key, metric in VSI_REPORT:
        vals = scores.get(key, [])
        if vals:
            task_scores[key] = {"display": display, "metric": metric, "score": round(float(np.mean(vals) * 100), 2), "count": len(vals)}
    num_vals = [v["score"] for k, v in task_scores.items() if k in NUMERICAL_TASKS]
    mc_keys = {"object_rel_distance", "object_rel_direction", "route_planning", "obj_appearance_order"}
    mc_vals = [v["score"] for k, v in task_scores.items() if k in mc_keys]
    all_vals = [v["score"] for v in task_scores.values()]
    summary = {"total": len(records), "task_scores": task_scores, "question_type_scores": {k: {"score": round(float(np.mean(v) * 100), 2), "count": len(v)} for k, v in sorted(qtype_scores.items())}}
    if num_vals:
        summary["numerical_avg"] = round(float(np.mean(num_vals)), 2)
    if mc_vals:
        summary["mc_avg"] = round(float(np.mean(mc_vals)), 2)
    if all_vals:
        summary["overall_avg"] = round(float(np.mean(all_vals)), 2)
    return summary


# ---------------------------------------------------------------------------
# Image-sequence MC: MMSI + MindCube
# ---------------------------------------------------------------------------


def load_image_records(args) -> List[Dict[str, Any]]:
    if args.task == "mmsi":
        if args.data_file.endswith((".jsonl", ".json")):
            return read_json_or_jsonl(args.data_file)
        csv.field_size_limit(sys.maxsize)
        with open(args.data_file, encoding="utf-8", errors="replace", newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
    return load_mindcube_records(
        args.data_file,
        chunk=args.chunk,
        index=args.index,
        expected_samples=args.expected_samples,
        cache_dir=os.environ.get("MINDCUBE_HF_CACHE_DIR") or None,
    )


def decode_mmsi_images(value: Any, max_images: int) -> List[Image.Image]:
    images_b64 = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(images_b64, (list, tuple)):
        images_b64 = [images_b64]
    images = []
    for item in images_b64[:max_images]:
        if not item:
            continue
        path = ""
        if isinstance(item, dict):
            path = str(
                item.get("path")
                or item.get("image")
                or item.get("image_path")
                or ""
            )
        elif isinstance(item, str) and os.path.isfile(item):
            path = item
        if path:
            images.append(Image.open(path).convert("RGB"))
            continue
        raw = item if isinstance(item, bytes) else base64.b64decode(str(item))
        images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return images


def normalize_choices(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def labels_for(choices: List[Any]) -> List[str]:
    return CHOICES[:len(choices)] if choices else list("ABCD")


def image_prompt(args, rec: Dict[str, Any]) -> Tuple[str, List[str]]:
    if args.task == "mmsi":
        q = str(rec.get("question") or "").strip()
        labels = []
        for ch in re.findall(r"(?:^|[\s,])([A-H])\s*[:\).]", q):
            if ch not in labels:
                labels.append(ch)
        return f"{q}\n{PROMPT_TAIL}", labels or list("ABCD")
    return mindcube_prompt(rec, PROMPT_TAIL)


def image_answer(args, rec: Dict[str, Any]) -> str:
    answer = (
        mindcube_answer_value(rec)
        if args.task == "mindcube"
        else rec.get("answer")
    )
    if args.task == "mindcube" and isinstance(answer, (int, float)) and not isinstance(answer, bool):
        labels = labels_for(normalize_choices(rec.get("choices")))
        idx = int(answer)
        return labels[idx] if 0 <= idx < len(labels) else ""
    return extract_mcq_answer(str(answer), CHOICES) or str(answer).strip().upper()[:1]


def prepare_images(images: List[Image.Image], prompt: str, processor, args) -> Dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    content = []
    for img in images:
        item = {"type": "image", "image": img, "max_pixels": 262144}
        item["min_pixels"] = 4096
        content.append(item)
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = render_chat_prompt(messages, processor, False)
    image_inputs, _ = process_vision_info(messages, image_patch_size=16)
    return {"prompt": text, "multi_modal_data": {"image": image_inputs}, "mm_processor_kwargs": {"do_resize": False}}


def worker_image_mc(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    records = load_image_records(args)
    if args.task == "mmsi":
        records = shard_records(records, args.chunk, args.index)
    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True, min_pixels=4096, max_pixels=262144)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": args.max_images},
    )
    sampling = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0, top_p=1.0, top_k=-1)
    results = []

    def prepare_batch(start: int, batch: List[Dict[str, Any]]):
        inputs, kept, prompts, labels = [], [], {}, {}
        for j, rec in enumerate(batch):
            try:
                images = (
                    decode_mmsi_images(
                        rec.get("images") or rec.get("image"),
                        args.max_images,
                    )
                    if args.task == "mmsi"
                    else decode_mindcube_images(rec, args.max_images, 1024)
                )
                if not images:
                    continue
                prompt, labs = image_prompt(args, rec)
                inputs.append(prepare_images(images, prompt, processor, args))
                kept.append(j)
                prompts[j] = prompt
                labels[j] = labs
            except Exception as e:
                print(f"[warn] skip {start + j}: {type(e).__name__}: {e}", flush=True)
        return start, batch, inputs, kept, prompts, labels

    for start, batch, inputs, kept, prompts, labels in iter_prefetched_batches(records, args.batch_size, prepare_batch, args.prefetch_batches):
        outs = llm.generate(inputs, sampling_params=sampling) if inputs else []
        for j, out in zip(kept, outs):
            rec = batch[j]
            raw = out.outputs[0].text
            pred = extract_mcq_answer(raw, labels[j])
            gt = image_answer(args, rec)
            group = (
                rec.get("category")
                if args.task == "mmsi"
                else mindcube_group(rec)
            )
            results.append({"id": str(rec.get("index") or rec.get("id") or start + j), "bench": args.task, "group": group or "unknown", "question": rec.get("question"), "prompt": prompts[j], "answer": gt, "pred_answer": pred, "raw_prediction": raw, "score": 1.0 if pred and gt and pred == gt else 0.0})
    write_json(args.output_dir, f"results_{args.task}_shard{args.index}.json", results)
    write_json(args.output_dir, f"summary_shard{args.index}.json", aggregate_image_mc(args.task, results))


def aggregate_image_mc(task: str, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for s in samples:
        groups[str(s.get("group") or "unknown")]["total"] += 1
        groups[str(s.get("group") or "unknown")]["correct"] += int(s.get("score", 0))
    correct = sum(int(s.get("score", 0)) for s in samples)
    parsed = sum(1 for s in samples if s.get("pred_answer"))
    labeled = sum(1 for s in samples if s.get("answer"))
    key = "by_category" if task == "mmsi" else (
        "by_subject" if task == "mmvu" else (
            "by_task_type" if task == "mlvu" else (
                "by_question_type" if task in {"videoholmes", "longvideobench"} else "by_task"
            )
        )
    )
    by_group = {k: {"accuracy": pct(v["correct"], v["total"]), "correct": v["correct"], "total": v["total"]} for k, v in sorted(groups.items())}
    out = {
        "num_samples": len(samples),
        "correct": correct,
        "accuracy": pct(correct, len(samples)),
        "parse_rate": pct(parsed, len(samples)),
        "ground_truth_rate": pct(labeled, len(samples)),
        key: by_group,
    }
    if task == "mlvu" and by_group:
        # Official MLVU M-Avg is the macro mean over the per-task accuracies.
        out["M_avg"] = round(sum(g["accuracy"] for g in by_group.values()) / len(by_group), 2)
    return out


# ---------------------------------------------------------------------------
# Spatial grounding
# ---------------------------------------------------------------------------


def aggregate_spatial_grounding(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(samples)
    ious = [float(s.get("iou", 0.0) or 0.0) for s in samples]
    parsed = sum(1 for s in samples if s.get("pred_bbox") is not None)
    metrics = {
        "num_samples": n,
        "mIoU": round(sum(ious) / max(n, 1) * 100, 2),
        "parse_rate": round(parsed / max(n, 1) * 100, 2),
    }
    for threshold in (0.5, 0.7, 0.9):
        metrics[f"acc@{threshold}"] = round(sum(1 for v in ious if v >= threshold) / max(n, 1) * 100, 2)
    return metrics


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def worker_tracking(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    from eval.task.tracking.eval_tracking_vllm import evaluate_one

    if not args.processor_path:
        args.processor_path = args.model_path

    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.processor_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    args.patch_size = processor.image_processor.patch_size

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seed=args.seed,
    )
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop_token_ids=[],
    )

    summaries = {}
    for dataset in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        metrics = evaluate_one(llm, sampling, processor, dataset, args)
        if metrics:
            summaries[dataset] = metrics
    suffix = f"_shard{args.index}" if args.chunk > 1 else ""
    write_json(args.output_dir, f"summary{suffix}.json", summaries)


def aggregate_tracking(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(samples)
    mious = [float(s.get("miou", 0.0) or 0.0) for s in samples]
    all_frame_ious = []
    for sample in samples:
        values = sample.get("frame_ious")
        if isinstance(values, list):
            all_frame_ious.extend(float(v) for v in values)
    nf = len(all_frame_ious)
    parsed = sum(1 for s in samples if int(s.get("pred_boxes_n", 0) or 0) > 0)
    return {
        "num_samples": n,
        "num_frames": nf,
        "AO": round(sum(all_frame_ious) / max(nf, 1) * 100, 2),
        "R@0.3": round(sum(1 for v in all_frame_ious if v >= 0.3) / max(nf, 1) * 100, 2),
        "R@0.5": round(sum(1 for v in all_frame_ious if v >= 0.5) / max(nf, 1) * 100, 2),
        "R@0.7": round(sum(1 for v in all_frame_ious if v >= 0.7) / max(nf, 1) * 100, 2),
        "mIoU": round(sum(mious) / max(n, 1) * 100, 2),
        "sample_R@0.3": round(sum(1 for v in mious if v >= 0.3) / max(n, 1) * 100, 2),
        "sample_R@0.5": round(sum(1 for v in mious if v >= 0.5) / max(n, 1) * 100, 2),
        "sample_R@0.7": round(sum(1 for v in mious if v >= 0.7) / max(n, 1) * 100, 2),
        "parse_rate": round(parsed / max(n, 1) * 100, 2),
    }


def worker_stvg(args) -> None:
    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    from eval.task.spatial_temporal_grounding.eval_stvg_vllm import evaluate_one

    if not args.processor_path:
        args.processor_path = args.model_path

    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.processor_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    args.patch_size = processor.image_processor.patch_size

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1, "image": 1},
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        seed=args.seed,
    )
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop_token_ids=[],
    )

    summaries = {}
    for dataset in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        metrics = evaluate_one(llm, sampling, processor, dataset, args)
        if metrics:
            summaries[dataset] = metrics
    suffix = f"_shard{args.index}" if args.chunk > 1 else ""
    write_json(args.output_dir, f"summary{suffix}.json", summaries)


def aggregate_stvg(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(samples)
    accs = [float(s.get("accuracy", 0.0) or 0.0) for s in samples]
    tious = [float(s.get("tiou", 0.0) or 0.0) for s in samples]
    sious = [float(s.get("siou", 0.0) or 0.0) for s in samples]
    mious_inter = [float(s.get("miou_inter", 0.0) or 0.0) for s in samples]
    parsed = sum(1 for s in samples if s.get("pred_time") is not None or int(s.get("pred_boxes_n", 0) or 0) > 0)
    metrics = {
        "num_samples": n,
        "tIoU": round(sum(tious) / max(n, 1) * 100, 2),
        "tIoU@0.5": round(sum(1 for v in tious if v >= 0.5) / max(n, 1) * 100, 2),
        "sIoU": round(sum(sious) / max(n, 1) * 100, 2),
        "sIoU@0.5": round(sum(1 for v in sious if v >= 0.5) / max(n, 1) * 100, 2),
        "mIoU_inter": round(sum(mious_inter) / max(n, 1) * 100, 2),
        "mean_accuracy": round(sum(accs) / max(n, 1) * 100, 2),
        "parse_rate": round(parsed / max(n, 1) * 100, 2),
    }
    for threshold in (0.3, 0.7):
        metrics[f"tIoU@{threshold}"] = round(sum(1 for v in tious if v >= threshold) / max(n, 1) * 100, 2)
        metrics[f"sIoU@{threshold}"] = round(sum(1 for v in sious if v >= threshold) / max(n, 1) * 100, 2)
    return metrics


# ---------------------------------------------------------------------------
# Parent orchestration / merge
# ---------------------------------------------------------------------------


def write_json(output_dir: str | Path, name: str, obj: Any) -> None:
    path = Path(output_dir) / name
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json_files(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    rows = []
    for p in paths:
        if p.is_file():
            rows.extend(json.loads(p.read_text(encoding="utf-8")))
    return rows


def dataset_stem(name: str) -> str:
    base = name[:-5] if name.endswith(".json") else name
    return base[:-6] if base.endswith(".jsonl") else base


def merge_task(task: str, output_dir: Path, n: int) -> None:
    if task in {"videomme", "videommev2"}:
        rows = load_json_files(output_dir / f"results_{task}_shard{i}.json" for i in range(n))
        write_json(output_dir, f"results_{task}.json", rows)
        write_json(output_dir, "summary.json", aggregate_videomme(rows))
    elif task == "vsi":
        seen = {}
        for i in range(n):
            p = output_dir / f"results_shard{i}.jsonl"
            if p.is_file():
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        r = json.loads(line)
                        seen[str(r.get("id"))] = r
        rows = list(seen.values())
        expected_samples = env_int("VSI_EXPECTED_SAMPLES", 5130)
        if expected_samples > 0 and len(rows) != expected_samples:
            raise RuntimeError(
                f"Expected {expected_samples} VSI-Bench results, got {len(rows)}."
            )
        (output_dir / "merged_results.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        summary = aggregate_vsi(rows)
        if expected_samples > 0:
            summary["expected_samples"] = expected_samples
            summary["coverage"] = round(100.0 * len(rows) / expected_samples, 2)
        write_json(output_dir, "summary.json", summary)
    elif task == "spatial_grounding":
        summary = {}
        default_datasets = "refcoco-val,refcoco-testA,refcoco-testB,refcoco+-val,refcoco+-testA,refcoco+-testB,refcocog-val,refcocog-test"
        datasets = [x.strip() for x in os.environ.get("SPATIAL_GROUNDING_DATASETS", default_datasets).split(",") if x.strip()]
        for dataset in datasets:
            name = dataset_stem(dataset)
            rows = load_json_files(output_dir / f"results_{name}_shard{i}.json" for i in range(n))
            if rows:
                write_json(output_dir, f"results_{name}.json", rows)
                summary[name] = aggregate_spatial_grounding(rows)
            else:
                single = output_dir / f"results_{name}.json"
                if single.is_file():
                    rows = json.loads(single.read_text(encoding="utf-8"))
                    summary[name] = aggregate_spatial_grounding(rows)
        write_json(output_dir, "summary.json", summary)
    elif task == "tracking":
        summary = {}
        datasets = [x.strip() for x in os.environ.get("TRACKING_DATASETS", "eval_got10k").split(",") if x.strip()]
        for dataset in datasets:
            name = dataset_stem(dataset)
            rows = load_json_files(output_dir / f"results_{name}_shard{i}.json" for i in range(n))
            if rows:
                write_json(output_dir, f"results_{name}.json", rows)
                summary[name] = aggregate_tracking(rows)
            else:
                single = output_dir / f"results_{name}.json"
                if single.is_file():
                    rows = json.loads(single.read_text(encoding="utf-8"))
                    summary[name] = aggregate_tracking(rows)
        write_json(output_dir, "summary.json", summary)
    elif task == "stvg":
        summary = {}
        datasets = [x.strip() for x in os.environ.get("STVG_DATASETS", "eval_stvg").split(",") if x.strip()]
        for dataset in datasets:
            name = dataset_stem(dataset)
            rows = load_json_files(output_dir / f"results_{name}_shard{i}.json" for i in range(n))
            if rows:
                write_json(output_dir, f"results_{name}.json", rows)
                summary[name] = aggregate_stvg(rows)
            else:
                single = output_dir / f"results_{name}.json"
                if single.is_file():
                    rows = json.loads(single.read_text(encoding="utf-8"))
                    summary[name] = aggregate_stvg(rows)
        write_json(output_dir, "summary.json", summary)
    else:
        rows = load_json_files(output_dir / f"results_{task}_shard{i}.json" for i in range(n))
        expected_samples = (
            env_int("MINDCUBE_EXPECTED_SAMPLES", 1050)
            if task == "mindcube"
            else 0
        )
        if expected_samples > 0 and len(rows) != expected_samples:
            raise RuntimeError(
                f"Expected {expected_samples} MindCube-Tiny results, "
                f"got {len(rows)}."
            )
        write_json(output_dir, f"results_{task}.json", rows)
        summary = aggregate_image_mc(task, rows)
        if expected_samples > 0:
            summary["expected_samples"] = expected_samples
            summary["coverage"] = round(
                100.0 * len(rows) / expected_samples, 2
            )
        if task == "mindcube":
            summary["data_source"] = os.environ.get(
                "MINDCUBE_DATA_FILE", ""
            )
        write_json(output_dir, "summary.json", summary)
    print(f"Summary -> {output_dir / 'summary.json'}", flush=True)


def launch_task(task: str, model: str, gpu_groups: List[str], args) -> None:
    d = task_defaults(task)
    out = make_output_dir(task, model, d["setting"])
    cmds = []
    for i in range(len(gpu_groups)):
        cmd = [PYTHON, str(THIS_FILE), "--worker", "--task", task, "--model_path", model, "--output_dir", str(out), "--chunk", str(len(gpu_groups)), "--index", str(i), "--tp_size", str(args.tp_size), "--data_file", d["data_file"], "--batch_size", str(d["batch_size"]), "--prefetch_batches", str(args.prefetch_batches)]
        if task == "spatial_grounding":
            cmd = [
                PYTHON,
                str(PROJECT_DIR / "eval/task/spatial_grounding/eval_refcoco_vllm.py"),
                "--model_path", model,
                "--processor_path", str(d["processor_path"] or model),
                "--bench_dir", str(d["bench_dir"]),
                "--datasets", str(d["datasets"]),
                "--output_dir", str(out),
                "--prompt_style", str(d["prompt_style"]),
                "--coord_system", str(d["coord_system"]),
                "--bbox_select", str(d["bbox_select"]),
                "--enable_thinking", "true" if d["enable_thinking"] else "false",
                "--min_tokens", str(d["min_tokens"]),
                "--total_tokens", str(d["total_tokens"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--batch_size", str(d["batch_size"]),
                "--tensor_parallel_size", str(args.tp_size),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--max_model_len", str(d["max_model_len"]),
                "--max_samples", str(d["max_samples"]),
                "--chunk", str(len(gpu_groups)),
                "--index", str(i),
                "--seed", str(args.seed),
            ]
        if task in {"mmsi", "mindcube"}:
            cmd += [
                "--max_images", str(d["max_images"]),
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
            ]
            if task == "mindcube":
                cmd += [
                    "--expected_samples",
                    str(d["expected_samples"]),
                ]
        if task in {"videomme", "videommev2"}:
            cmd += [
                "--video_base", d["video_base"],
                "--video_dir", d["video_dir"],
                "--preprocessed_video_dir", d["preprocessed_video_dir"],
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--prompt_mode", str(d.get("prompt_mode", "default")),
                "--answer_filter", str(d.get("answer_filter", "")),
            ]
            cmd += ["--max_samples", str(d["max_samples"])]
        if task == "videommmu":
            cmd += [
                "--preprocessed_video_dir", d["video_root"],
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--image_max_pixels", str(d["image_max_pixels"]),
                "--temperature", str(d["temperature"]),
                "--top_p", str(d["top_p"]),
                "--top_k", str(d["top_k"]),
                "--presence_penalty", str(d["presence_penalty"]),
                "--min_p", str(d["min_p"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
        if task == "mmvu":
            cmd += [
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--temperature", str(d["temperature"]),
                "--top_p", str(d["top_p"]),
                "--top_k", str(d["top_k"]),
                "--presence_penalty", str(d["presence_penalty"]),
                "--min_p", str(d["min_p"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
        if task == "mvbench":
            cmd += [
                "--preprocessed_video_dir", str(d["video_root"]),
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--temperature", str(d["temperature"]),
                "--top_p", str(d["top_p"]),
                "--top_k", str(d["top_k"]),
                "--presence_penalty", str(d["presence_penalty"]),
                "--min_p", str(d["min_p"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
        if task == "videoholmes":
            cmd += [
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--temperature", str(d["temperature"]),
                "--top_p", str(d["top_p"]),
                "--top_k", str(d["top_k"]),
                "--presence_penalty", str(d["presence_penalty"]),
                "--min_p", str(d["min_p"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
        if task == "longvideobench":
            cmd += [
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--use_subtitles",
                "true" if d["use_subtitles"] else "false",
                "--temperature", str(d["temperature"]),
                "--top_p", str(d["top_p"]),
                "--top_k", str(d["top_k"]),
                "--presence_penalty", str(d["presence_penalty"]),
                "--min_p", str(d["min_p"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
        if task == "lvbench":
            cmd += [
                "--preprocessed_video_dir", str(d["video_root"]),
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--temperature", str(d["temperature"]),
                "--top_p", str(d["top_p"]),
                "--top_k", str(d["top_k"]),
                "--presence_penalty", str(d["presence_penalty"]),
                "--min_p", str(d["min_p"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
        if task == "mlvu":
            cmd += [
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--temperature", str(d["temperature"]),
                "--top_p", str(d["top_p"]),
                "--top_k", str(d["top_k"]),
                "--presence_penalty", str(d["presence_penalty"]),
                "--min_p", str(d["min_p"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
        if task == "vsi":
            cmd += [
                "--preprocessed_video_dir",
                d["preprocessed_video_dir"],
                "--max_model_len",
                str(d["max_model_len"]),
                "--max_new_tokens",
                str(d["max_new_tokens"]),
                "--gpu_memory_utilization",
                str(d["gpu_memory_utilization"]),
                "--video_min_pixels",
                str(d["video_min_pixels"]),
                "--video_max_pixels",
                str(d["video_max_pixels"]),
                "--video_total_pixels",
                str(d["video_total_pixels"]),
                "--max_frames",
                str(d["max_frames"]),
                "--fps",
                str(d["fps"]),
            ]
        if task == "tracking":
            cmd += [
                "--processor_path", str(d["processor_path"]),
                "--bench_dir", str(d["bench_dir"]),
                "--base_prefix", str(d["base_prefix"]),
                "--datasets", str(d["datasets"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--chunked_reprompt", str(d["chunked_reprompt"]),
                "--max_samples", str(d["max_samples"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
            if d["reprompt_use_gt_first_box"]:
                cmd.append("--reprompt_use_gt_first_box")
        if task == "stvg":
            cmd += [
                "--processor_path", str(d["processor_path"]),
                "--bench_dir", str(d["bench_dir"]),
                "--base_prefix", str(d["base_prefix"]),
                "--datasets", str(d["datasets"]),
                "--video_min_pixels", str(d["video_min_pixels"]),
                "--video_max_pixels", str(d["video_max_pixels"]),
                "--video_total_pixels", str(d["video_total_pixels"]),
                "--max_frames", str(d["max_frames"]),
                "--fps", str(d["fps"]),
                "--max_model_len", str(d["max_model_len"]),
                "--max_new_tokens", str(d["max_new_tokens"]),
                "--max_num_batched_tokens", str(d["max_num_batched_tokens"]),
                "--gpu_memory_utilization", str(d["gpu_memory_utilization"]),
                "--prompt_mode", str(d["prompt_mode"]),
                "--max_samples", str(d["max_samples"]),
            ]
            if d["enable_thinking"]:
                cmd.append("--enable_thinking")
        if task in {
            "mmvu",
            "mvbench",
            "videoholmes",
            "longvideobench",
            "mlvu",
        }:
            cmd += ["--max_samples", str(d["max_samples"])]
        cmds.append(cmd)

    run_config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "model_path": model,
        "model_family": model_tags(model)[0],
        "checkpoint": model_tags(model)[1],
        "output_dir": str(out),
        "setting": d.get("setting"),
        "data_file": d.get("data_file"),
        "gpu_groups": gpu_groups,
        "tp_size": args.tp_size,
        "task_defaults": json_safe(d),
        "commands": [" ".join(shlex.quote(str(x)) for x in cmd) for cmd in cmds],
    }
    write_json(out, "run_config.json", run_config)
    if task in {"videomme", "videommev2"}:
        try:
            records = read_json_or_jsonl(str(d["data_file"]))
            if records:
                records.sort(key=lambda r: (str(r.get("videoID") or r.get("video_id") or ""), str(r.get("question_id") or r.get("id") or "")))
                preview = videomme_prompt(records[0], str(d.get("prompt_mode", "default")))
                (out / "prompt_preview.txt").write_text(preview, encoding="utf-8")
        except Exception as exc:
            (out / "prompt_preview.txt").write_text(f"[failed to render preview] {exc}\n", encoding="utf-8")

    print("\n" + "=" * 60, flush=True)
    print(f"Running {task}", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"Output: {out}", flush=True)
    print(f"Run config: {out / 'run_config.json'}", flush=True)
    print("=" * 60, flush=True)
    procs = []
    for i, (cmd, gpus) in enumerate(zip(cmds, gpu_groups)):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["PYTHONUNBUFFERED"] = "1"
        log = (out / f"worker_{i}.log").open("w", encoding="utf-8")
        log.write("=" * 60 + "\n")
        log.write("OraRL Eval Worker\n")
        log.write(f"created_at: {run_config['created_at']}\n")
        log.write(f"task: {task}\n")
        log.write(f"model_path: {model}\n")
        log.write(f"output_dir: {out}\n")
        log.write(f"setting: {d.get('setting')}\n")
        log.write(f"data_file: {d.get('data_file')}\n")
        if d.get("preprocessed_video_dir"):
            log.write(f"preprocessed_video_dir: {d.get('preprocessed_video_dir')}\n")
        if d.get("video_root"):
            log.write(f"video_root: {d.get('video_root')}\n")
        log.write(f"gpu_group: {gpus}\n")
        log.write(f"tp_size: {args.tp_size}\n")
        log.write(f"command: {' '.join(shlex.quote(str(x)) for x in cmd)}\n")
        log.write(f"task_defaults: {json.dumps(json_safe(d), ensure_ascii=False, sort_keys=True)}\n")
        log.write("=" * 60 + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs.append((i, proc, log))
        print(f"Launched shard {i} on GPU {gpus} (PID {proc.pid})", flush=True)
        if i < len(cmds) - 1:
            time.sleep(float(os.environ.get("LAUNCH_DELAY", "3")))

    failed = False
    remaining = set(range(len(procs)))
    while remaining:
        for i, proc, log in procs:
            if i not in remaining:
                continue
            rc = proc.poll()
            if rc is None:
                continue
            remaining.remove(i)
            log.close()
            print(("[DONE]" if rc == 0 else "[FAIL]") + f" shard {i} exit={rc}", flush=True)
            if rc != 0:
                failed = True
                for j, other, other_log in procs:
                    if j in remaining and other.poll() is None:
                        print(f"[KILL] shard {j} after shard {i} failed", flush=True)
                        other.terminate()
                for j, other, other_log in procs:
                    if j in remaining:
                        try:
                            other.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            other.kill()
                            other.wait()
                        other_log.close()
                remaining.clear()
                break
        if remaining:
            time.sleep(2)

    if failed:
        raise RuntimeError(f"{task} failed; logs under {out}")
    merge_task(task, out, len(gpu_groups))


def parse_tasks(value: str) -> List[str]:
    if value == "all":
        return list(TASKS)
    tasks = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in tasks if x not in TASKS]
    if bad:
        raise ValueError(f"Unknown task(s): {bad}; available={TASKS}")
    return tasks


def worker_main(args) -> None:
    print(
        f"[worker] task={args.task} shard={args.index}/{args.chunk} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
        flush=True,
    )
    if args.task in {"videomme", "videommev2"}:
        worker_videomme(args)
    elif args.task == "videommmu":
        worker_videommmu(args)
    elif args.task == "mmvu":
        worker_mmvu(args)
    elif args.task == "mvbench":
        worker_mvbench(args)
    elif args.task == "videoholmes":
        worker_videoholmes(args)
    elif args.task == "longvideobench":
        worker_longvideobench(args)
    elif args.task == "lvbench":
        worker_lvbench(args)
    elif args.task == "mlvu":
        worker_mlvu(args)
    elif args.task == "vsi":
        worker_vsi(args)
    elif args.task in {"mmsi", "mindcube"}:
        worker_image_mc(args)
    elif args.task == "tracking":
        worker_tracking(args)
    elif args.task == "stvg":
        worker_stvg(args)
    else:
        raise ValueError(args.task)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=os.environ.get("MODEL_PATH", ""))
    p.add_argument("--tasks", default=os.environ.get("TASKS", "all"))
    p.add_argument("--gpus", default=os.environ.get("GPUS", "0,1,2,3,4,5,6,7"))
    p.add_argument("--tp_size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    p.add_argument("--worker", action="store_true")
    p.add_argument("--task", choices=TASKS)
    p.add_argument("--output_dir", default="")
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--data_file", default="")
    p.add_argument("--video_base", default="")
    p.add_argument("--video_dir", default="")
    p.add_argument("--preprocessed_video_dir", default="")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--prefetch_batches", type=int, default=int(os.environ.get("PREFETCH_BATCHES", "1")))
    p.add_argument("--max_images", type=int, default=8)
    p.add_argument("--expected_samples", type=int, default=0)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--max_num_batched_tokens", type=int, default=32768)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--processor_path", default="")
    p.add_argument("--bench_dir", default="")
    p.add_argument("--base_prefix", default="")
    p.add_argument("--datasets", default="eval_got10k")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--image_max_pixels", type=int, default=0)
    p.add_argument("--video_min_pixels", type=int, default=65536)
    p.add_argument("--video_max_pixels", type=int, default=8388608)
    p.add_argument("--video_total_pixels", type=int, default=8388608)
    p.add_argument("--max_frames", type=int, default=32)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--enable_thinking", action="store_true")
    p.add_argument("--prompt_mode", choices=["default", "bare", "onethink", "onethink_cot", "qwen", "holmes", "train_stvg", "explicit_ah", "careful_ah"], default="default")
    p.add_argument("--answer_filter", default="")
    p.add_argument("--use_subtitles", choices=["true", "false"], default="true")
    p.add_argument("--chunked_reprompt", type=int, default=0)
    p.add_argument("--reprompt_use_gt_first_box", action="store_true")
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--presence_penalty", type=float, default=0.0)
    p.add_argument("--min_p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.use_subtitles = args.use_subtitles == "true"
    if args.max_samples < 0:
        p.error("--max_samples must be non-negative")

    if not args.model_path:
        raise ValueError("Missing model path; set MODEL_PATH in eval/task/eval.sh or pass --model_path.")

    if args.worker:
        worker_main(args)
        return

    gpu_groups = split_gpus(args.gpus, args.tp_size)
    for task in parse_tasks(args.tasks):
        launch_task(task, args.model_path, gpu_groups, args)


if __name__ == "__main__":
    main()
