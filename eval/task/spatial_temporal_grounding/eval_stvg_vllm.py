#!/usr/bin/env python3
"""
Spatial-Temporal Video Grounding (STVG) evaluation — vLLM data-parallel.

Inputs: OneThinker-eval style JSON/JSONL under `<bench_dir>/<name>.json`.
Each record has:
    data_type       = "video"
    problem_type    = "spatial-temporal grounding"
    problem         = "<video>...question..."
    path            = "./<rel>.mp4"
    solution/answer = '<answer>{"time":[t0,t1],"boxes":{"9":[x1,y1,x2,y2],...}}</answer>'

`boxes` can also be a list-of-list (one box per integer second starting at
round(t0)); we normalize both shapes into a dict keyed by int-second.

Metrics (paper-standard STVG, mirrors Table 6 of OneThinker / LLaVA-ST):
    tIoU         — 1D IoU of `time`
    tIoU@0.5     — fraction of samples with tIoU >= 0.5
    sIoU         — 2D IoU averaged over EVERY GT frame (missing pred frame = 0)
    sIoU@0.5     — fraction of samples with sIoU >= 0.5
Extras:
    mIoU_inter   — 2D IoU averaged over pred ∩ gt frames (lenient, OneThinker
                   training-reward semantics; for training-time comparison)
    accuracy     — 0.5*tIoU + 0.5*mIoU_inter   (per-sample legacy combined)
    parse_rate   — frac of samples with a valid JSON answer

Coord system: norm1000 (same domain for gt and pred, no rescaling).

Inference mode: SERVER-SIDE video decoding.
    - The message only carries the local video path; vLLM workers load and
      decode the video themselves using qwen_vl_utils, controlled by
      `mm_processor_kwargs` and `media_io_kwargs` configured on the LLM.
    - `allowed_local_media_path` is set to `--base_prefix` (or `/` if empty)
      so vLLM is allowed to read the video files from disk.

Usage:
    python eval_stvg_vllm.py \\
        --model_path  /path/to/ckpt  \\
        --bench_dir   /path/to/OneThinker-eval  \\
        --datasets    eval_stvg  \\
        --output_dir  outputs/temporal_grounding/...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time as _time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")


# ---------------------------------------------------------------------------
# Prompt — single source of truth in eval/task/eval_prompt.py
# (matches data/joint/sft_joint_all.jsonl STVG samples).
# ---------------------------------------------------------------------------

_EVAL_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_TASK_DIR not in sys.path:
    sys.path.insert(0, _EVAL_TASK_DIR)

from canonical_data import load_json_records  # noqa: E402
from eval_prompt import (  # noqa: E402
    GROUNDING_QUESTION_TEMPLATE_NO_THINK as QUESTION_TEMPLATE_NO_THINK,
    STVG_TAIL,
    TRAIN_STVG_QUESTION_PREFIX,
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _load_json_relaxed(s: str) -> Optional[dict]:
    """Try json.loads, then first balanced `{...}` substring."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _is_list_of_numbers(x, n: Optional[int] = None) -> bool:
    if not isinstance(x, list):
        return False
    if n is not None and len(x) != n:
        return False
    try:
        for v in x:
            float(v)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# IoU helpers — identical to eval_bench.py
# ---------------------------------------------------------------------------

def iou_1d(pred, gt) -> float:
    if not _is_list_of_numbers(pred, 2) or not _is_list_of_numbers(gt, 2):
        return 0.0
    p0, p1 = float(pred[0]), float(pred[1])
    g0, g1 = float(gt[0]), float(gt[1])
    if p0 > p1: p0, p1 = p1, p0
    if g0 > g1: g0, g1 = g1, g0
    inter = max(0.0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    return inter / union if union > 1e-12 else 0.0


def iou_2d(b1, b2) -> float:
    if not _is_list_of_numbers(b1, 4) or not _is_list_of_numbers(b2, 4):
        return 0.0
    x1 = max(float(b1[0]), float(b2[0])); y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2])); y2 = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    u = a1 + a2 - inter
    return inter / u if u > 1e-12 else 0.0


def mean_iou_over_intersection(pred_boxes: Dict[str, List[float]],
                               gt_boxes: Dict[str, List[float]]) -> float:
    """Mean IoU over keys present in BOTH pred and gt (OneThinker / training
    reward semantics; lenient — does not penalize missing pred frames)."""
    if not isinstance(pred_boxes, dict) or not isinstance(gt_boxes, dict):
        return 0.0
    common = set(pred_boxes.keys()) & set(gt_boxes.keys())
    if not common:
        return 0.0
    vals = [iou_2d(pred_boxes[k], gt_boxes[k]) for k in common]
    return sum(vals) / len(vals) if vals else 0.0


def mean_iou_over_gt_frames(pred_boxes: Dict[str, List[float]],
                            gt_boxes: Dict[str, List[float]]) -> float:
    """Mean IoU over EVERY gt frame; missing pred frames score 0.

    This is the standard "sIoU" reported in STVG papers
    (Table 6 of OneThinker / LLaVA-ST), and it strictly penalizes
    pred frame coverage gaps."""
    if not isinstance(gt_boxes, dict) or not gt_boxes:
        return 0.0
    pred_dict = pred_boxes if isinstance(pred_boxes, dict) else {}
    total, n = 0.0, 0
    for k, gbox in gt_boxes.items():
        total += iou_2d(pred_dict.get(k, []), gbox)
        n += 1
    return total / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Box-dict normalization (supports dict AND list-of-list)
# ---------------------------------------------------------------------------

def _normalize_boxes(boxes: Any,
                     time_span: Optional[List[float]] = None
                     ) -> Dict[str, List[float]]:
    """Coerce boxes into {str(int_sec): [x1,y1,x2,y2]}.

    * dict: int/str-int keys preserved (as str(int)); bad entries dropped.
    * list: aligned with integer seconds starting at round(time_span[0]);
            requires a valid 2-number time_span.
    Other inputs → empty dict.
    """
    if isinstance(boxes, dict):
        out: Dict[str, List[float]] = {}
        for k, v in boxes.items():
            try:
                ki = int(float(k))
            except Exception:
                continue
            if _is_list_of_numbers(v, 4):
                out[str(ki)] = [float(x) for x in v]
        return out
    if isinstance(boxes, list) and _is_list_of_numbers(time_span, 2):
        t0 = int(round(float(time_span[0])))
        out = {}
        for i, b in enumerate(boxes):
            if _is_list_of_numbers(b, 4):
                out[str(t0 + i)] = [float(x) for x in b]
        return out
    return {}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    return load_json_records(path)


_CANONICAL_DATASET_ALIASES = {
    "eval_stvg": "stvg",
}


def load_dataset(bench_dir: str, name: str) -> List[Dict[str, Any]]:
    base = name[:-5] if name.endswith(".json") else name
    base = base[:-6] if base.endswith(".jsonl") else base
    candidates = (base, _CANONICAL_DATASET_ALIASES.get(base, base))
    for candidate in dict.fromkeys(candidates):
        for ext in (".json", ".jsonl"):
            p = os.path.join(bench_dir, candidate + ext)
            if os.path.isfile(p):
                return _read_json_or_jsonl(p)
    raise FileNotFoundError(
        f"Dataset not found for {name!r} under {bench_dir}"
    )


def _strip_leading_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<(?:video|image)>\s*", "", text, count=1).strip()


def _extract_raw_query(problem_text: str) -> str:
    """Extract the raw query from an eval-style STVG problem string.

    Eval format examples:
      "Who is watching the ball? Please find the corresponding time period ..."
      "The person in red shirt? Please find ... When and where ..."

    We take everything before the first occurrence of "Please find" or
    "When and where" as the raw query.
    """
    for sep in ("? Please find the corresponding time period",
                "? Please describe the location",
                "? When and where does",
                "When and where does"):
        idx = problem_text.find(sep)
        if idx > 0:
            return problem_text[:idx].strip()
    return problem_text.split("?")[0].strip() if "?" in problem_text else problem_text.strip()


def build_prompt_text(example: Dict[str, Any], enable_thinking: bool = False,
                      prompt_mode: str = "default") -> str:
    question = _strip_leading_tags(
        example.get("problem") or example.get("question") or "")
    raw_query = _extract_raw_query(question)
    q = TRAIN_STVG_QUESTION_PREFIX.format(query=raw_query)
    return QUESTION_TEMPLATE_NO_THINK.format(Question=q) + STVG_TAIL


def _resolve_video_path(rec: Dict[str, Any], base_prefix: str) -> Optional[str]:
    """Resolve a video path. Tries (in order):
        1. absolute path as-is
        2. <base>/<full_relative_path>
        3. <base>/<basename>
        4. drop common prefix dirs (e.g. anno records `./Evaluation/<sub>/...`
           but disk has only `<base>/<sub>/...` without the `Evaluation/` layer)
        5. last-resort: 1-level recursive search by basename under <base>
    Returns absolute path or None (caller should treat None as 'skip').
    """
    raw = (rec.get("path") or rec.get("video") or rec.get("video_path")
           or rec.get("file_name"))
    if not raw:
        return None
    if os.path.isabs(raw) and os.path.isfile(raw):
        return raw
    rel = raw.lstrip("./").lstrip("/")
    base = (base_prefix or "").rstrip("/")
    if not base:
        return raw if os.path.isfile(raw) else None

    candidates: List[str] = []
    candidates.append(os.path.join(base, rel))
    candidates.append(os.path.join(base, os.path.basename(rel)))

    # Drop one or two leading directory components to handle anno layouts
    # like "./Evaluation/<dataset>/<file>" living at "<base>/<dataset>/<file>".
    parts = rel.split("/")
    for k in (1, 2):
        if len(parts) > k:
            candidates.append(os.path.join(base, *parts[k:]))

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    # 1-level recursive lookup by basename under <base> (cheap on small dirs).
    base_name = os.path.basename(rel)
    if base_name:
        for root, _dirs, files in os.walk(base):
            if base_name in files:
                return os.path.join(root, base_name)
            # don't go too deep — STVG benches usually have only 1-2 levels.
            depth = root[len(base):].count(os.sep)
            if depth >= 3:
                _dirs.clear()
    return None


def _extract_gt(rec: Dict[str, Any]):
    blob = rec.get("solution") or rec.get("answer") or ""
    ans = extract_answer(blob) if isinstance(blob, str) else ""
    obj = _load_json_relaxed(ans)
    if obj is None and isinstance(blob, dict):
        obj = blob
    if not isinstance(obj, dict):
        return None, {}
    t = obj.get("time")
    if not _is_list_of_numbers(t, 2):
        t = None
    return t, _normalize_boxes(obj.get("boxes"), t)


def _extract_pred(text: str):
    ans = extract_answer(text)
    obj = _load_json_relaxed(ans)
    if not isinstance(obj, dict):
        return None, {}
    t = obj.get("time")
    if not _is_list_of_numbers(t, 2):
        t = None
    return t, _normalize_boxes(obj.get("boxes"), t)


# ---------------------------------------------------------------------------
# vLLM packing — client-side video decoding (same as verl / eval_timelens_vllm)
# ---------------------------------------------------------------------------

def _build_video_content(video_path: str, args) -> List[Dict[str, Any]]:
    """Build message content with video sampling parameters (client-side decode).

    `process_vision_info` will read the video from disk using these kwargs
    (fps, total_pixels, max_frames, min/max pixels per-frame) to produce
    decoded tensors. **All of these are forwarded so eval matches training.**
    """
    item: Dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "total_pixels": args.video_total_pixels,
        "max_frames": args.max_frames,
        "fps": args.fps,
    }
    if getattr(args, "video_min_pixels", None) is not None:
        item["min_pixels"] = args.video_min_pixels
    if getattr(args, "video_max_pixels", None) is not None:
        item["max_pixels"] = args.video_max_pixels
    return [item]


def _prepare_for_vllm(messages, processor, patch_size: int,
                      enable_thinking: bool = False):
    """Build vLLM input dict with client-side decoded video tensors.

    Mirrors verl's rollout: apply_chat_template → process_vision_info →
    pass multi_modal_data + mm_processor_kwargs to vLLM.
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
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


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_one(llm, sampling_params, processor,
                 dataset_name: str, args) -> Dict[str, Any]:
    print(f"\n>>> Evaluating {dataset_name}", flush=True)
    records = load_dataset(args.bench_dir, dataset_name)

    # Deterministic order for shard slicing.
    records.sort(key=lambda r: (r.get("problem_id", 0),
                                str(r.get("path", ""))))

    max_samples = int(getattr(args, "max_samples", 0) or 0)
    if max_samples > 0:
        records = records[:max_samples]

    if args.chunk > 1:
        records = records[args.index::args.chunk]
        print(f"  Shard {args.index}/{args.chunk}: {len(records)} samples",
              flush=True)
    else:
        print(f"  Loaded {len(records)} samples", flush=True)

    # Filter to STVG only (defensive — the file *should* be pure STVG).
    records = [
        r for r in records
        if str(r.get("problem_type", "")).strip().lower()
           == "spatial-temporal grounding"
        or not r.get("problem_type")  # accept untyped records too
    ]
    if not records:
        print("  [warn] no STVG records after filtering; skipping.", flush=True)
        return {}

    # ------------------------------------------------------------------
    # Pre-resolve all video paths up front so we fail loudly (and quickly)
    # if --base_prefix is wrong, instead of silently scoring 0 for hours.
    # ------------------------------------------------------------------
    print(f"  resolving video paths under base_prefix={args.base_prefix} ...",
          flush=True)
    resolved: List[Optional[str]] = []
    n_missing = 0
    for rec in records:
        vp = _resolve_video_path(rec, args.base_prefix)
        resolved.append(vp)
        if vp is None:
            n_missing += 1
    if n_missing:
        # Show first 3 examples to help the user diagnose.
        examples = [r.get("path") for r, vp in zip(records, resolved) if vp is None][:3]
        print(f"  [warn] {n_missing}/{len(records)} videos missing on disk. "
              f"First few unresolved paths:", flush=True)
        for ex in examples:
            print(f"    - {ex!r}", flush=True)
        if n_missing == len(records):
            raise SystemExit(
                "[fatal] 0/{n} videos resolved. Set --base_prefix correctly. "
                "Got base_prefix={base!r}, anno records reference paths like "
                "{ex!r}".format(n=len(records), base=args.base_prefix, ex=examples[0])
            )
    else:
        print(f"  all {len(records)} videos resolved OK.", flush=True)

    # Per-sample buffers. We track all four standard STVG metrics:
    #   tIoU         : 1D temporal IoU (paper "tIoU")
    #   sIoU         : mean spatial IoU averaged over EVERY GT frame  (paper "sIoU")
    #   mIoU_inter   : mean spatial IoU averaged over pred∩gt frames
    #                  (lenient, OneThinker training-reward semantics; kept for
    #                  reference when comparing against training-time numbers)
    #   accuracy     : 0.5*tIoU + 0.5*mIoU_inter  (legacy combined score)
    tious, sious, mious_inter, accs = [], [], [], []
    n_parsed = 0
    per_sample = []

    bsz = args.batch_size
    t0 = _time.time()
    for start in range(0, len(records), bsz):
        batch = records[start:start + bsz]
        batch_paths = resolved[start:start + bsz]
        inputs_for_vllm = []
        keep_idx: List[int] = []
        for j, (rec, vp) in enumerate(zip(batch, batch_paths)):
            if not vp:
                continue
            content = _build_video_content(vp, args)
            content.append({"type": "text", "text": build_prompt_text(
                rec, enable_thinking=args.enable_thinking,
                prompt_mode=args.prompt_mode)})
            messages = [{"role": "user", "content": content}]
            try:
                packed = _prepare_for_vllm(
                    messages, processor, args.patch_size,
                    enable_thinking=args.enable_thinking)
            except Exception as e:
                print(f"  [warn] prepare failed (pid={rec.get('problem_id')}): {e}",
                      flush=True)
                continue
            inputs_for_vllm.append(packed)
            keep_idx.append(j)

        texts = [""] * len(batch)
        if inputs_for_vllm:
            try:
                outs = llm.generate(inputs_for_vllm,
                                    sampling_params=sampling_params)
                for j, out in zip(keep_idx, outs):
                    texts[j] = out.outputs[0].text
            except Exception as e:
                print(f"  [error] vLLM generate failed @batch {start}: {e}",
                      flush=True)

        for rec, answer in zip(batch, texts):
            gt_t, gt_b = _extract_gt(rec)
            pr_t, pr_b = _extract_pred(answer)
            parsed = (pr_t is not None) or bool(pr_b)
            if parsed:
                n_parsed += 1
            tiou = iou_1d(pr_t, gt_t) if (pr_t and gt_t) else 0.0
            siou = mean_iou_over_gt_frames(pr_b, gt_b) if gt_b else 0.0
            miou_inter = mean_iou_over_intersection(pr_b, gt_b) if gt_b else 0.0
            acc = 0.5 * tiou + 0.5 * miou_inter
            tious.append(tiou)
            sious.append(siou)
            mious_inter.append(miou_inter)
            accs.append(acc)

            per_sample.append({
                "problem_id": rec.get("problem_id"),
                "path": rec.get("path"),
                "gt_time": gt_t,
                "gt_boxes_n": len(gt_b),
                "pred_time": pr_t,
                "pred_boxes_n": len(pr_b),
                "tiou": round(tiou, 4),
                "siou": round(siou, 4),
                "miou_inter": round(miou_inter, 4),
                "accuracy": round(acc, 4),
                "answer": answer,
            })

        if (start // bsz) % 10 == 0:
            done = start + len(batch)
            el = _time.time() - t0
            print(f"  [{done}/{len(records)}] {el:.1f}s  "
                  f"tIoU={sum(tious)/max(len(tious),1):.4f}  "
                  f"sIoU={sum(sious)/max(len(sious),1):.4f}  "
                  f"parse={n_parsed}/{len(accs)}", flush=True)

    n = len(accs)
    metrics = {
        "num_samples": n,
        # ---- paper-standard STVG metrics ----
        "tIoU":     round(sum(tious) / max(n, 1) * 100, 2),
        "tIoU@0.5": round(sum(1 for v in tious if v >= 0.5) / max(n, 1) * 100, 2),
        "sIoU":     round(sum(sious) / max(n, 1) * 100, 2),
        "sIoU@0.5": round(sum(1 for v in sious if v >= 0.5) / max(n, 1) * 100, 2),
        # ---- extras for OneThinker-style comparison ----
        "mIoU_inter":    round(sum(mious_inter) / max(n, 1) * 100, 2),
        "mean_accuracy": round(sum(accs) / max(n, 1) * 100, 2),
        "parse_rate":    round(n_parsed / max(n, 1) * 100, 2),
    }
    # extra recall thresholds (R@0.3 / 0.7) for both tIoU and sIoU
    for t in (0.3, 0.7):
        metrics[f"tIoU@{t}"] = round(
            sum(1 for v in tious if v >= t) / max(n, 1) * 100, 2)
        metrics[f"sIoU@{t}"] = round(
            sum(1 for v in sious if v >= t) / max(n, 1) * 100, 2)

    # Persist per-sample results.
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_shard{args.index}" if args.chunk > 1 else ""
    out_path = os.path.join(
        args.output_dir, f"results_{dataset_name}{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, ensure_ascii=False, indent=2)

    print(f"  [{dataset_name}] "
          f"tIoU={metrics['tIoU']:.2f}%  tIoU@0.5={metrics['tIoU@0.5']:.2f}%  "
          f"sIoU={metrics['sIoU']:.2f}%  sIoU@0.5={metrics['sIoU@0.5']:.2f}%  "
          f"mIoU_inter={metrics['mIoU_inter']:.2f}%  "
          f"parse={metrics['parse_rate']:.2f}%  n={n}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="STVG evaluation via vLLM (OneThinker prompt).")
    p.add_argument("--model_path", required=True)
    p.add_argument("--processor_path", default=None,
                   help="Defaults to --model_path.")
    p.add_argument("--bench_dir", required=True,
                   help="Dir containing eval_stvg*.json / .jsonl.")
    p.add_argument("--datasets", default="eval_stvg",
                   help="Comma-separated names (with or without .json).")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Maximum samples before sharding; 0 evaluates all.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_prefix", default="",
                   help="Prefix for relative video paths inside records.")
    # video sampling (client-side decoding via qwen_vl_utils / process_vision_info)
    p.add_argument("--video_total_pixels", type=int,
                   default=8 * 1024 * 1024,
                   help="Total pixel budget across all sampled frames "
                        "(default: 8M; matches the STVG training config).")
    p.add_argument("--video_min_pixels", type=int, default=None,
                   help="Minimum per-frame pixel count. Should match training "
                        "(STVG yaml uses 65536 = 256x256).")
    p.add_argument("--video_max_pixels", type=int, default=None,
                   help="Maximum per-frame pixel count. Should match training "
                        "(STVG yaml uses 8388608 = 8M).")
    p.add_argument("--max_pixels_video", type=int, default=None,
                   help="[DEPRECATED] alias for --video_total_pixels.")
    p.add_argument("--max_frames", type=int, default=128,
                   help="Max sampled frames; matches training default (128).")
    p.add_argument("--fps", type=int, default=2)
    # vLLM engine
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=65536)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--max_num_batched_tokens", type=int, default=65536)
    p.add_argument("--enforce_eager", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    # thinking mode
    p.add_argument("--enable_thinking", action="store_true", default=False,
                   help="Enable Qwen3 <think> block. Default: disabled.")
    p.add_argument("--prompt_mode", type=str, default="default",
                   choices=["default", "bare", "onethink", "train_stvg"],
                   help="'default': full NO_THINK+STVG_TAIL template. "
                        "'bare': raw question only (for SFT ckpts trained on raw prompts). "
                        "'onethink': OneThinker official format — reasoning "
                        "instruction in SYSTEM message, task tail in USER message. "
                        "'train_stvg': rewrite question to training format "
                        "'Given the query \"...\", when and where...' + template.")
    # sharding
    p.add_argument("--chunk", type=int, default=1,
                   help="Total shards (for data-parallel launcher).")
    p.add_argument("--index", type=int, default=0,
                   help="This shard's index in [0, chunk).")
    args = p.parse_args()
    if args.max_samples < 0:
        p.error("--max_samples must be non-negative")
    return args


def main():
    args = parse_args()
    if args.processor_path is None:
        args.processor_path = args.model_path

    if args.max_pixels_video is not None:
        args.video_total_pixels = args.max_pixels_video

    os.makedirs(args.output_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor, AutoTokenizer

    print(f"Model:       {args.model_path}", flush=True)
    print(f"Processor:   {args.processor_path}", flush=True)
    print(f"Bench dir:   {args.bench_dir}", flush=True)
    print(f"Datasets:    {args.datasets}", flush=True)
    print(f"Base prefix: {args.base_prefix}", flush=True)
    print(f"Video:       total_pixels={args.video_total_pixels} "
          f"max_frames={args.max_frames} fps={args.fps}", flush=True)
    print(f"Thinking:    {args.enable_thinking}", flush=True)
    print(f"Prompt mode: {args.prompt_mode}", flush=True)

    # Processor + tokenizer — client-side video decoding (like verl rollout)
    processor = AutoProcessor.from_pretrained(
        args.processor_path, padding_side="left",
        do_resize=False, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.processor_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    patch_size = processor.image_processor.patch_size
    args.patch_size = patch_size
    print(f"Patch size:  {patch_size}", flush=True)

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
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
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop_token_ids=[],
    )

    all_metrics: Dict[str, Dict[str, Any]] = {}
    for ds in [s.strip() for s in args.datasets.split(",") if s.strip()]:
        metrics = evaluate_one(llm, sampling_params, processor, ds, args)
        if metrics:
            all_metrics[ds] = metrics

    # Write summary (per-shard file if sharded).
    suffix = f"_shard{args.index}" if args.chunk > 1 else ""
    summary_path = os.path.join(args.output_dir, f"summary{suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSummary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
