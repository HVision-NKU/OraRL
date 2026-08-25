#!/usr/bin/env python3
"""
Tracking evaluation — vLLM data-parallel.

Inputs: OneThinker-eval style JSON/JSONL under `<bench_dir>/<name>.json`.
Each record has:
    data_type       = "video"
    problem_type    = "tracking"
    problem         = "<video>...question..."
    path            = "./<rel>.mp4"
    solution/answer = '<answer>{"boxes":{"1":[x1,y1,x2,y2],"2":[...],...,"32":[...]}}</answer>'

Schema notes (mirrors OneThinker's `eval_bench.py` 'tracking'):
    - Answer is a JSON dict with key `boxes` only (NO `time` field).
    - `boxes` is a dict keyed by integer second 1..32 (max 32 seconds).
    - Each value is a 4-number bbox [x1, y1, x2, y2] in norm1000 coords.

Metrics (GOT-10k standard, mirrors Table 7 of OneThinker / VideoChat-R1 papers):
    AO      — Average Overlap = mean per-frame IoU across ALL (sample, gt-frame)
              pairs (missing pred frame counts as 0). Equivalent to taking
              the per-sample mIoU and averaging — when every sample has the
              same # of GT frames (got10k always = 32) — but reported as
              FRAME-LEVEL mean.
    R@0.3   — Success Rate @ 0.3 = fraction of (sample, gt-frame) pairs with
              IoU >= 0.3.
    R@0.5   — same, threshold 0.5.
    R@0.7   — same, threshold 0.7.
Extras (per-sample, useful when GT frame counts vary across samples):
    mIoU              — per-sample mean IoU averaged over samples
    sample_R@{.3,.5,.7} — per-sample success rate (sample mIoU >= τ)
    parse_rate        — fraction of samples with a valid JSON answer

Inference: client-side video decoding (verl-style)
    - apply_chat_template → process_vision_info → multi_modal_data tensors
    - Same engine setup as eval_stvg_vllm.py.

Usage:
    python eval_tracking_vllm.py \\
        --model_path  /path/to/ckpt  \\
        --bench_dir   /path/to/OneThinker-eval  \\
        --datasets    eval_got10k  \\
        --output_dir  outputs/tracking/...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time as _time
from typing import Any, Dict, List, Optional

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")


# ---------------------------------------------------------------------------
# Prompt — single source of truth in eval/task/eval_prompt.py
# (matches data/joint/sft_joint_all.jsonl tracking samples).
# ---------------------------------------------------------------------------

_EVAL_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_TASK_DIR not in sys.path:
    sys.path.insert(0, _EVAL_TASK_DIR)

from canonical_data import load_json_records  # noqa: E402
from eval_prompt import (  # noqa: E402
    GROUNDING_QUESTION_TEMPLATE_NO_THINK as QUESTION_TEMPLATE_NO_THINK,
    TRACKING_TAIL,
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
# IoU
# ---------------------------------------------------------------------------

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


def mean_iou_over_gt_frames(pred_boxes: Dict[str, List[float]],
                            gt_boxes: Dict[str, List[float]]) -> float:
    """Mean IoU over EVERY gt frame; missing pred frames score 0.

    This is the standard "tracking" metric used in OneThinker / LLaVA-ST
    (eval_bench.py: tracking branch returns 'miou_gt' = this value)."""
    if not isinstance(gt_boxes, dict) or not gt_boxes:
        return 0.0
    pred_dict = pred_boxes if isinstance(pred_boxes, dict) else {}
    total, n = 0.0, 0
    for k, gbox in gt_boxes.items():
        total += iou_2d(pred_dict.get(k, []), gbox)
        n += 1
    return total / n if n > 0 else 0.0


def per_frame_ious(pred_boxes: Dict[str, List[float]],
                   gt_boxes: Dict[str, List[float]]) -> List[float]:
    """Return one IoU per GT frame (missing pred = 0).

    Used for GOT-10k AO / R@τ which are FRAME-LEVEL metrics, NOT sample-level.
    """
    if not isinstance(gt_boxes, dict) or not gt_boxes:
        return []
    pred_dict = pred_boxes if isinstance(pred_boxes, dict) else {}
    out: List[float] = []
    for k, gbox in gt_boxes.items():
        out.append(iou_2d(pred_dict.get(k, []), gbox))
    return out


# ---------------------------------------------------------------------------
# Box-dict normalization
# ---------------------------------------------------------------------------

def _normalize_boxes(boxes: Any) -> Dict[str, List[float]]:
    """Coerce boxes into {str(int_sec): [x1,y1,x2,y2]}.

    Tracking schema only allows dict (no list-of-list, since there's no time
    span anchor). Bad entries are dropped.
    """
    if not isinstance(boxes, dict):
        return {}
    out: Dict[str, List[float]] = {}
    for k, v in boxes.items():
        try:
            ki = int(float(k))
        except Exception:
            continue
        if _is_list_of_numbers(v, 4):
            out[str(ki)] = [float(x) for x in v]
    return out


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    return load_json_records(path)


_CANONICAL_DATASET_ALIASES = {
    "eval_got10k": "got10k",
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


def build_prompt_text(example: Dict[str, Any], enable_thinking: bool = False,
                      prompt_mode: str = "default") -> str:
    question = _strip_leading_tags(
        example.get("problem") or example.get("question") or "")
    return QUESTION_TEMPLATE_NO_THINK.format(Question=question) + TRACKING_TAIL


def _resolve_video_path(rec: Dict[str, Any], base_prefix: str) -> Optional[str]:
    """Resolve a video path with multi-tier fallback (same logic as STVG)."""
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

    parts = rel.split("/")
    for k in (1, 2):
        if len(parts) > k:
            candidates.append(os.path.join(base, *parts[k:]))

    for cand in candidates:
        if os.path.isfile(cand):
            return cand

    base_name = os.path.basename(rel)
    if base_name:
        for root, _dirs, files in os.walk(base):
            if base_name in files:
                return os.path.join(root, base_name)
            depth = root[len(base):].count(os.sep)
            if depth >= 3:
                _dirs.clear()
    return None


def _extract_gt(rec: Dict[str, Any]) -> Dict[str, List[float]]:
    blob = rec.get("solution") or rec.get("answer") or ""
    ans = extract_answer(blob) if isinstance(blob, str) else ""
    obj = _load_json_relaxed(ans)
    if obj is None and isinstance(blob, dict):
        obj = blob
    if not isinstance(obj, dict):
        return {}
    return _normalize_boxes(obj.get("boxes"))


def _extract_pred(text: str) -> Dict[str, List[float]]:
    ans = extract_answer(text)
    obj = _load_json_relaxed(ans)
    if not isinstance(obj, dict):
        return {}
    return _normalize_boxes(obj.get("boxes"))


# ---------------------------------------------------------------------------
# vLLM packing — client-side video decoding (verl-style)
# ---------------------------------------------------------------------------

def _build_video_content(video_path: str, args,
                         video_start: Optional[float] = None,
                         video_end: Optional[float] = None) -> List[Dict[str, Any]]:
    """Build the video content dict for the message.

    `video_start` / `video_end` (in seconds) are optional time-window cuts
    used by chunked re-prompt mode; pass None for the full video (default).

    Env knobs:
      EVAL_VIDEO_ITEM_ONETHINKER=1
        Build the video item with ONLY {video, max_pixels, max_frames, fps},
        omitting `min_pixels` and `total_pixels`. OneThinker's eval_bench.py
        constructs the item this way; the extra keys can subtly shift the
        per-frame pixel layout in qwen_vl_utils.
    """
    import os as _os
    onethinker_style = _os.getenv("EVAL_VIDEO_ITEM_ONETHINKER", "0") == "1"
    item: Dict[str, Any] = {
        "type": "video",
        "video": video_path,
        "max_pixels": args.video_max_pixels,
        "max_frames": args.max_frames,
        "fps": args.fps,
    }
    if not onethinker_style:
        item["min_pixels"] = args.video_min_pixels
        item["total_pixels"] = args.video_total_pixels
    if video_start is not None:
        item["video_start"] = float(video_start)
    if video_end is not None:
        item["video_end"] = float(video_end)
    return [item]


# ---------------------------------------------------------------------------
# Chunked re-prompt helpers (DIAGNOSTIC; only used when --chunked_reprompt > 0)
# ---------------------------------------------------------------------------

# Match an [x1,y1,x2,y2] bbox literal — the canonical pattern in GOT-10k
# prompts: 'Given the bounding box [537,403,768,703] of the target object'.
# We capture the FIRST 4-int bracket group in the prompt; tolerate spaces.
_BBOX_4INT_RE = re.compile(
    r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")


def _replace_first_bbox_literal(text: str, new_box: List[float]) -> str:
    """Replace the FIRST `[x1,y1,x2,y2]` integer-bbox literal in `text`
    with `new_box`. Coords are integerised (round). Returns text unchanged
    if no bbox literal is found (caller should warn)."""
    if not _BBOX_4INT_RE.search(text):
        return text
    repl = "[{},{},{},{}]".format(*[int(round(float(v))) for v in new_box])
    return _BBOX_4INT_RE.sub(repl, text, count=1)


def _build_chunked_prompt_text(example: Dict[str, Any],
                               window_first_box: List[float],
                               window_first_sec: int,
                               window_last_sec: int,
                               enable_thinking: bool,
                               prompt_mode: str) -> str:
    """For chunked re-prompt: build a prompt that asks for boxes in
    [window_first_sec, window_last_sec] with `window_first_box` as anchor.

    Uses the same template family as the default path; we substitute
    (a) the first-frame bbox in the question,
    (b) the "ONLY up to 32 seconds" temporal range hint,
    (c) the example key sequence in TRACKING_TAIL (`"1", "2", "32"`)
        with concrete keys from the window. Models tend to mimic the
        example's key set rather than follow instructions, so (c) is
        the most important substitution — without it, the model often
        outputs `"1", "2", "32"` keys regardless of instruction.
    """
    base = build_prompt_text(example, enable_thinking=enable_thinking,
                             prompt_mode=prompt_mode)
    # 1) replace first-frame bbox in question
    base = _replace_first_bbox_literal(base, window_first_box)
    # 2) tweak temporal range hint in the TRACKING_TAIL example.
    #    Original: 'ONLY up to 32 seconds'. Rewrite to the actual window.
    base = re.sub(
        r"ONLY up to \d+ seconds",
        f"from second {window_first_sec} to second {window_last_sec} (inclusive)",
        base, count=1,
    )
    base = re.sub(
        r"correspond to a second \(1, 2, 3, \.\.\., \d+\)",
        f"correspond to a second ({window_first_sec}, {window_first_sec+1}, "
        f"..., {window_last_sec})",
        base, count=1,
    )
    # 3) Critically: rewrite the example's key sequence to match the window.
    #    Original example uses keys "1", "2", "32"; in-context bias makes the
    #    model copy this exact key set. We swap to (first, first+1, last).
    f, l = window_first_sec, window_last_sec
    mid = f + 1 if l > f else f
    base = re.sub(
        r'\{"boxes": \{"1": \[[\d, ]+\], "2": \[[\d, ]+\], "32": \[[\d, ]+\]\}\}',
        f'{{"boxes": {{"{f}": [405, 230, 654, 463], '
        f'"{mid}": [435, 223, 678, 446], '
        f'"{l}": [415, 203, 691, 487]}}}}',
        base, count=1,
    )
    return base


def _prepare_for_vllm(messages, processor, patch_size: int,
                      enable_thinking: bool = False):
    """Prepare a vLLM-ready dict from a chat-style message list.

    Env knobs (debugging / OneThinker reproduction):
      EVAL_OMIT_ENABLE_THINKING_KW=1
          Do NOT pass `enable_thinking=...` to apply_chat_template.
          OneThinker's eval_bench.py omits this kwarg entirely; passing it
          may inject a `<think>\\n` generation prefix that mismatches the
          model's expected format. Enable this when reproducing
          OneThinker-8B paper numbers.
      EVAL_OMIT_DO_RESIZE_OVERRIDE=1
          Do NOT force `video_kwargs["do_resize"] = False`.
          OneThinker's eval_bench.py leaves do_resize at its default;
          forcing False can subtly change the per-frame pixel layout.
          Enable this together with the above for full OneThinker parity.
    """
    from qwen_vl_utils import process_vision_info
    import os as _os

    omit_thinking_kw = _os.getenv("EVAL_OMIT_ENABLE_THINKING_KW", "0") == "1"
    omit_do_resize = _os.getenv("EVAL_OMIT_DO_RESIZE_OVERRIDE", "0") == "1"

    if omit_thinking_kw:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
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
    if not omit_do_resize:
        video_kwargs["do_resize"] = False

    llm_input: Dict[str, Any] = {"prompt": text}
    if video_inputs:
        llm_input["multi_modal_data"] = {"video": video_inputs}
        llm_input["mm_processor_kwargs"] = video_kwargs

    return llm_input


# ---------------------------------------------------------------------------
# Per-dataset evaluation
# ---------------------------------------------------------------------------

def _evaluate_chunked_reprompt(records, resolved, llm, sampling_params,
                               processor, dataset_name, args):
    """Diagnostic chunked re-prompt evaluation. Each video is split into
    contiguous windows of `args.chunked_reprompt` seconds (over the 32 GT
    seconds). Window 0 uses the prompt's original first-frame bbox; for
    window k>=1 we substitute (a) the model's last predicted bbox of
    window k-1 (or, if --reprompt_use_gt_first_box, the GT bbox at second
    `window_first_sec`) and (b) only feed video frames within the window's
    time range. Predicted boxes from each window are merged into a single
    32-key dict and fed to the standard metric.

    Returns the same metrics dict as evaluate_one's no-reprompt path.
    """
    win = max(1, int(args.chunked_reprompt))
    fps = float(args.fps or 2.0)
    print(f"\n  [CHUNKED RE-PROMPT] window={win}s, "
          f"reprompt_use_gt_first_box={args.reprompt_use_gt_first_box}",
          flush=True)
    print(f"  [WARNING] chunked re-prompt mode is EXPERIMENTAL. The temporal "
          f"window (video_start/video_end) assumes GT 'second N' labels "
          f"correspond to REAL seconds in the source video. Verify this on "
          f"your dataset before trusting the numbers.", flush=True)

    # ---- per-record loop (videos are not batched across chunks because
    # different videos may be at different windows; we DO batch within a
    # single window across videos) ----
    mious: List[float] = []
    all_frame_ious: List[float] = []
    n_parsed = 0
    per_sample: List[Dict[str, Any]] = []

    n_videos = len(records)
    # Number of windows for the canonical 32-second tracking task.
    n_windows = (32 + win - 1) // win

    # State across windows: each video's "current first-frame box" used to
    # build the prompt of the next window, and the accumulated predictions.
    cur_first_box: List[Optional[List[float]]] = []
    accum_pred: List[Dict[str, List[float]]] = []
    gt_per_video: List[Dict[str, List[float]]] = []

    # Initialise from each record's first-frame GT bbox (which is what the
    # original prompt already contains; we recover it via regex).
    for rec in records:
        gt_b = _extract_gt(rec)
        gt_per_video.append(gt_b)
        # Try to read first-frame box from prompt; fallback to gt[1].
        prompt = (rec.get("problem") or rec.get("question") or "")
        m = _BBOX_4INT_RE.search(prompt)
        if m:
            init = [float(x) for x in m.groups()]
        else:
            init = gt_b.get("1", []) or gt_b.get(min(gt_b.keys(), default="1"), [])
        cur_first_box.append(init if _is_list_of_numbers(init, 4) else None)
        accum_pred.append({})

    t0 = _time.time()
    bsz = args.batch_size

    for w in range(n_windows):
        win_first = w * win + 1
        win_last = min((w + 1) * win, 32)
        win_seconds = list(range(win_first, win_last + 1))
        # IMPORTANT: GOT-10k GT-second labels (1, 2, ..., 32) are REAL seconds
        # in the source video, NOT sampled-frame indices. So the temporal cut
        # passed to qwen_vl_utils must be in real seconds too. Add a small
        # cushion (+0.5s on each side) so the model sees the boundary frames
        # cleanly even after FPS rounding.
        v_start = max(0.0, float(win_first) - 1.0 - 0.5)  # in seconds
        v_end = float(win_last) + 0.5

        print(f"    [window {w+1}/{n_windows}] secs {win_first}..{win_last} "
              f"(video {v_start:.2f}-{v_end:.2f}s)",
              flush=True)

        # Build inputs for ALL videos at this window
        for start in range(0, n_videos, bsz):
            batch_idx = list(range(start, min(start + bsz, n_videos)))
            inputs_for_vllm = []
            keep_idx: List[int] = []
            for j in batch_idx:
                rec, vp = records[j], resolved[j]
                if not vp:
                    continue
                # Decide first-box for this window
                if w == 0:
                    fbox = cur_first_box[j]
                elif args.reprompt_use_gt_first_box:
                    # Use GT box at win_first
                    fbox = gt_per_video[j].get(str(win_first))
                    if not _is_list_of_numbers(fbox, 4):
                        fbox = cur_first_box[j]  # fallback
                else:
                    fbox = cur_first_box[j]

                if not _is_list_of_numbers(fbox, 4):
                    # No anchor: skip this window for this video
                    continue

                # Build prompt for this window
                prompt_text = _build_chunked_prompt_text(
                    rec, fbox, win_first, win_last,
                    enable_thinking=args.enable_thinking,
                    prompt_mode=args.prompt_mode)

                content = _build_video_content(
                    vp, args, video_start=v_start, video_end=v_end)
                content.append({"type": "text", "text": prompt_text})
                messages = [{"role": "user", "content": content}]
                try:
                    packed = _prepare_for_vllm(
                        messages, processor, args.patch_size,
                        enable_thinking=args.enable_thinking)
                except Exception as e:
                    print(f"      [warn] prepare failed "
                          f"(pid={rec.get('problem_id')}, w={w}): {e}",
                          flush=True)
                    continue
                inputs_for_vllm.append(packed)
                keep_idx.append(j)

            if not inputs_for_vllm:
                continue

            try:
                outs = llm.generate(inputs_for_vllm,
                                    sampling_params=sampling_params)
            except Exception as e:
                print(f"      [error] vLLM generate failed: {e}", flush=True)
                continue

            # Merge predictions into accum_pred and update cur_first_box
            for j, out in zip(keep_idx, outs):
                ans = out.outputs[0].text
                pr = _extract_pred(ans)
                # Filter to keys within this window
                for sec in win_seconds:
                    box = pr.get(str(sec))
                    if _is_list_of_numbers(box, 4):
                        accum_pred[j][str(sec)] = [float(x) for x in box]
                # Update cur_first_box for NEXT window using last predicted
                # second of THIS window (preferred), else fall back.
                last_box: Optional[List[float]] = None
                for sec in reversed(win_seconds):
                    if str(sec) in accum_pred[j]:
                        last_box = accum_pred[j][str(sec)]
                        break
                if last_box is not None:
                    cur_first_box[j] = last_box
                # else: keep cur_first_box[j] unchanged

        el = _time.time() - t0
        print(f"      [window {w+1}/{n_windows} done] elapsed={el:.1f}s",
              flush=True)

    # Compute metrics from accum_pred
    for j, rec in enumerate(records):
        gt_b = gt_per_video[j]
        pr_b = accum_pred[j]
        parsed = bool(pr_b)
        if parsed:
            n_parsed += 1
        frame_ious = per_frame_ious(pr_b, gt_b) if gt_b else []
        miou = (sum(frame_ious) / len(frame_ious)) if frame_ious else 0.0
        mious.append(miou)
        all_frame_ious.extend(frame_ious)
        per_sample.append({
            "problem_id": rec.get("problem_id"),
            "path": rec.get("path"),
            "gt_boxes_n": len(gt_b),
            "pred_boxes_n": len(pr_b),
            "miou": round(miou, 4),
            "frame_ious": [round(v, 4) for v in frame_ious],
            "answer": json.dumps({"boxes": pr_b}, ensure_ascii=False),
            "_chunked": True,
        })

    n = len(mious)
    nf = len(all_frame_ious)
    metrics = {
        "num_samples": n,
        "num_frames":  nf,
        "AO":     round(sum(all_frame_ious) / max(nf, 1) * 100, 2),
        "R@0.3":  round(sum(1 for v in all_frame_ious if v >= 0.3) / max(nf, 1) * 100, 2),
        "R@0.5":  round(sum(1 for v in all_frame_ious if v >= 0.5) / max(nf, 1) * 100, 2),
        "R@0.7":  round(sum(1 for v in all_frame_ious if v >= 0.7) / max(nf, 1) * 100, 2),
        "mIoU":           round(sum(mious) / max(n, 1) * 100, 2),
        "sample_R@0.3":   round(sum(1 for v in mious if v >= 0.3) / max(n, 1) * 100, 2),
        "sample_R@0.5":   round(sum(1 for v in mious if v >= 0.5) / max(n, 1) * 100, 2),
        "sample_R@0.7":   round(sum(1 for v in mious if v >= 0.7) / max(n, 1) * 100, 2),
        "parse_rate":     round(n_parsed / max(n, 1) * 100, 2),
        "_chunked_reprompt": int(args.chunked_reprompt),
        "_reprompt_use_gt_first_box": bool(args.reprompt_use_gt_first_box),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_shard{args.index}" if args.chunk > 1 else ""
    suffix += f"_chunked{args.chunked_reprompt}"
    if args.reprompt_use_gt_first_box:
        suffix += "_gtfirst"
    out_path = os.path.join(
        args.output_dir, f"results_{dataset_name}{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, ensure_ascii=False, indent=2)

    print(f"  [{dataset_name}] [CHUNKED w={win} "
          f"gt_first={args.reprompt_use_gt_first_box}] "
          f"AO={metrics['AO']:.2f}%  R@0.3={metrics['R@0.3']:.2f}%  "
          f"R@0.5={metrics['R@0.5']:.2f}%  R@0.7={metrics['R@0.7']:.2f}%  "
          f"parse={metrics['parse_rate']:.2f}%  n={n} ({nf} frames)",
          flush=True)
    return metrics


def evaluate_one(llm, sampling_params, processor,
                 dataset_name: str, args) -> Dict[str, Any]:
    print(f"\n>>> Evaluating {dataset_name}", flush=True)
    records = load_dataset(args.bench_dir, dataset_name)

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

    # Filter to tracking only.
    records = [
        r for r in records
        if str(r.get("problem_type", "")).strip().lower() == "tracking"
        or not r.get("problem_type")
    ]
    if not records:
        print("  [warn] no tracking records after filtering; skipping.",
              flush=True)
        return {}

    # Pre-resolve all video paths.
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
        examples = [r.get("path") for r, vp in zip(records, resolved)
                    if vp is None][:3]
        print(f"  [warn] {n_missing}/{len(records)} videos missing on disk. "
              f"First few unresolved paths:", flush=True)
        for ex in examples:
            print(f"    - {ex!r}", flush=True)
        if n_missing == len(records):
            raise SystemExit(
                "[fatal] 0/{n} videos resolved. Set --base_prefix correctly. "
                "Got base_prefix={base!r}, anno records reference paths like "
                "{ex!r}".format(n=len(records), base=args.base_prefix,
                                ex=examples[0])
            )
    else:
        print(f"  all {len(records)} videos resolved OK.", flush=True)

    # ----- Optional diagnostic path: chunked re-prompt -----
    if int(getattr(args, "chunked_reprompt", 0) or 0) > 0:
        return _evaluate_chunked_reprompt(
            records, resolved, llm, sampling_params, processor,
            dataset_name, args)

    # Per-sample mIoU (kept for backwards compatibility / debugging)
    mious: List[float] = []
    # Frame-level IoU bag — used for GOT-10k AO / R@τ
    all_frame_ious: List[float] = []
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
                print(f"  [warn] prepare failed (pid={rec.get('problem_id')}): "
                      f"{e}", flush=True)
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
            gt_b = _extract_gt(rec)
            pr_b = _extract_pred(answer)
            parsed = bool(pr_b)
            if parsed:
                n_parsed += 1
            frame_ious = per_frame_ious(pr_b, gt_b) if gt_b else []
            miou = (sum(frame_ious) / len(frame_ious)) if frame_ious else 0.0
            mious.append(miou)
            all_frame_ious.extend(frame_ious)

            per_sample.append({
                "problem_id": rec.get("problem_id"),
                "path": rec.get("path"),
                "gt_boxes_n": len(gt_b),
                "pred_boxes_n": len(pr_b),
                "miou": round(miou, 4),
                "frame_ious": [round(v, 4) for v in frame_ious],
                "answer": answer,
            })

        if (start // bsz) % 10 == 0:
            done = start + len(batch)
            el = _time.time() - t0
            ao_cur = (sum(all_frame_ious) / len(all_frame_ious)
                      if all_frame_ious else 0.0)
            print(f"  [{done}/{len(records)}] {el:.1f}s  "
                  f"AO={ao_cur:.4f}  "
                  f"mIoU={sum(mious)/max(len(mious),1):.4f}  "
                  f"parse={n_parsed}/{len(mious)}", flush=True)

    n = len(mious)
    nf = len(all_frame_ious)
    metrics = {
        "num_samples": n,
        "num_frames":  nf,
        # ---- GOT-10k paper-standard (frame-level) ----
        "AO":     round(sum(all_frame_ious) / max(nf, 1) * 100, 2),
        "R@0.3":  round(sum(1 for v in all_frame_ious if v >= 0.3) / max(nf, 1) * 100, 2),
        "R@0.5":  round(sum(1 for v in all_frame_ious if v >= 0.5) / max(nf, 1) * 100, 2),
        "R@0.7":  round(sum(1 for v in all_frame_ious if v >= 0.7) / max(nf, 1) * 100, 2),
        # ---- per-sample (legacy / OneThinker training-reward style) ----
        "mIoU":           round(sum(mious) / max(n, 1) * 100, 2),
        "sample_R@0.3":   round(sum(1 for v in mious if v >= 0.3) / max(n, 1) * 100, 2),
        "sample_R@0.5":   round(sum(1 for v in mious if v >= 0.5) / max(n, 1) * 100, 2),
        "sample_R@0.7":   round(sum(1 for v in mious if v >= 0.7) / max(n, 1) * 100, 2),
        "parse_rate":     round(n_parsed / max(n, 1) * 100, 2),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_shard{args.index}" if args.chunk > 1 else ""
    out_path = os.path.join(
        args.output_dir, f"results_{dataset_name}{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_sample, f, ensure_ascii=False, indent=2)

    print(f"  [{dataset_name}] "
          f"AO={metrics['AO']:.2f}%  R@0.3={metrics['R@0.3']:.2f}%  "
          f"R@0.5={metrics['R@0.5']:.2f}%  R@0.7={metrics['R@0.7']:.2f}%  "
          f"mIoU={metrics['mIoU']:.2f}%  "
          f"parse={metrics['parse_rate']:.2f}%  n={n} ({nf} frames)",
          flush=True)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Tracking evaluation via vLLM (OneThinker prompt).")
    p.add_argument("--model_path", required=True)
    p.add_argument("--processor_path", default=None,
                   help="Defaults to --model_path.")
    p.add_argument("--bench_dir", required=True,
                   help="Dir containing the tracking anno JSON / JSONL "
                        "(default name 'eval_got10k.json').")
    p.add_argument("--datasets", default="eval_got10k",
                   help="Comma-separated names (with or without .json).")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Maximum samples before sharding; 0 evaluates all.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--base_prefix", default="",
                   help="Prefix for relative video paths inside records.")

    # video sampling — client-side qwen_vl_utils (mirrors verl rollout).
    # Defaults aligned with verl: min=4*32*32=4096, max=64*32*32=65536.
    p.add_argument("--video_min_pixels", type=int, default=4 * 32 * 32)
    p.add_argument("--video_max_pixels", type=int, default=64 * 32 * 32)
    p.add_argument("--video_total_pixels", type=int,
                   default=256 * 64 * 32 * 32,
                   help="Total pixel budget across all sampled frames "
                        "(default: 256 frames * 64 tokens/frame * 32 * 32).")
    # Tracking spec is at most 32 seconds @ 1 fps == 32 frames; but the
    # real video may be longer/shorter — keep generous defaults like STVG
    # so the model sees enough temporal context.
    p.add_argument("--max_frames", type=int, default=64)
    p.add_argument("--fps", type=int, default=2)

    # vLLM engine
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_num_batched_tokens", type=int, default=32768)
    p.add_argument("--enforce_eager", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    # thinking mode
    p.add_argument("--enable_thinking", action="store_true", default=False,
                   help="Enable Qwen3 <think> block. Default: disabled.")
    p.add_argument("--prompt_mode", type=str, default="default",
                   choices=["default", "bare"],
                   help="'default': joint-SFT-aligned tracking template. "
                        "'bare': raw question only (for SFT ckpts trained on raw prompts).")
    # sharding
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--index", type=int, default=0)

    # ----- DIAGNOSTIC: chunked re-prompt (test-time tracking refresh) -----
    # Default 0 (disabled). When > 0, the video is split into windows of N
    # GT-seconds; for window k>=1 we replace the "first-frame bbox" in the
    # prompt with the LAST predicted bbox of window k-1, and feed only the
    # frames for that window. Predictions are stitched into a 32-key dict
    # and metrics are computed identically to the no-reprompt path.
    # USE ONLY for diagnostics: this is NOT the default GOT-10k protocol.
    p.add_argument("--chunked_reprompt", type=int, default=0,
                   help="Split each video into windows of N GT-seconds "
                        "and re-prompt the model with the previous "
                        "window's last predicted box. 0 = disabled "
                        "(default). Suggested values: 8 or 16.")
    p.add_argument("--reprompt_use_gt_first_box", action="store_true",
                   default=False,
                   help="(diagnostic upper bound) When --chunked_reprompt>0, "
                        "use GT first-box of each window instead of the "
                        "model's previous prediction. This isolates 'visual "
                        "context refresh' benefit from 'good initial box' "
                        "benefit. Default: False.")
    args = p.parse_args()
    if args.max_samples < 0:
        p.error("--max_samples must be non-negative")
    return args


def main():
    args = parse_args()
    if args.processor_path is None:
        args.processor_path = args.model_path

    os.makedirs(args.output_dir, exist_ok=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor, AutoTokenizer

    print(f"Model:       {args.model_path}", flush=True)
    print(f"Processor:   {args.processor_path}", flush=True)
    print(f"Bench dir:   {args.bench_dir}", flush=True)
    print(f"Datasets:    {args.datasets}", flush=True)
    print(f"Base prefix: {args.base_prefix}", flush=True)
    print(f"Video:       min_px={args.video_min_pixels} "
          f"max_px={args.video_max_pixels} "
          f"total_px={args.video_total_pixels} "
          f"max_frames={args.max_frames} fps={args.fps}", flush=True)
    print(f"Thinking:    {args.enable_thinking}", flush=True)
    print(f"Prompt mode: {args.prompt_mode}", flush=True)

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

    suffix = f"_shard{args.index}" if args.chunk > 1 else ""
    summary_path = os.path.join(args.output_dir, f"summary{suffix}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSummary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
