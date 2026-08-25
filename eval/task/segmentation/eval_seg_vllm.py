#!/usr/bin/env python3
"""
Segmentation prompt evaluation with vLLM.

This is the inference stage only.  It reads OneThinker-eval style records,
asks the model for SAM2 prompts (`boxes`, `positive_points`, `negative_points`,
and optional `time`), and writes `predicted_answer_norm` for the SAM2
post-processing stage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

_PATCH_DIRS = (
    Path(__file__).resolve().parents[1],
    Path(__file__).resolve().parents[2] / "Eval-onethink",
)
if os.environ.get("FORCE_QWENVL_VIDEO_READER") == "decord":
    for _patch_dir in _PATCH_DIRS:
        if not (_patch_dir / "qwenvl_decord_patch.py").is_file():
            continue
        sys.path.insert(0, str(_patch_dir))
        try:
            import qwenvl_decord_patch  # noqa: F401
        except Exception as exc:
            print(f"[warn] failed to import qwenvl_decord_patch: {exc}", file=sys.stderr)
        break

import torch  # noqa: E402
from qwen_vl_utils import process_vision_info  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoProcessor  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


# Prompts — single source of truth in eval/task/eval_prompt.py
# (training tails match data/joint/sft_joint_all.jsonl seg_image / seg_video).
_EVAL_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_TASK_DIR not in sys.path:
    sys.path.insert(0, _EVAL_TASK_DIR)

from canonical_data import (  # noqa: E402
    load_json_records,
    repository_relative_output_path,
)
from eval_prompt import (  # noqa: E402
    TRAIN_SEG_IMAGE_TAIL,
    TRAIN_SEG_VIDEO_TAIL,
)


ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    match = ANSWER_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_json_records(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported input JSON format: {path}")


def resolve_dataset_path(bench_dir: str, dataset: str) -> Path:
    raw = Path(dataset)
    if raw.is_file():
        return raw
    base = Path(bench_dir)
    candidates = [base / dataset, base / f"{dataset}.json", base / f"{dataset}.jsonl"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find dataset {dataset!r} under {bench_dir}")


def clean_question(problem: str) -> str:
    return re.sub(r"^\s*<(image|video)>\s*", "", problem or "", flags=re.IGNORECASE).strip()


def build_prompt(example: Dict[str, Any], prompt_mode: str) -> List[Dict[str, Any]]:
    # Match the joint SFT training prompt verbatim: the question is followed
    # immediately by the training tail.
    question = clean_question(str(example.get("problem") or example.get("question") or ""))
    data_type = str(example.get("data_type") or "").strip().lower()
    tail = TRAIN_SEG_VIDEO_TAIL if data_type == "video" else TRAIN_SEG_IMAGE_TAIL
    text = f"{question}\n{tail}"
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def resolve_media_path(base_prefix: str, raw_path: str) -> tuple[str, str]:
    """Return `(absolute_path_for_inference, relative_path_for_output)`.

    OneThinker JSON often stores paths like `./Evaluation/Refcoco/...`, while
    local mirrors may place media directly under `Refcoco/...`.
    """
    if os.path.isabs(raw_path):
        return raw_path, repository_relative_output_path(raw_path, base_prefix)

    rel = raw_path.lstrip("./")
    candidates = [rel]
    if rel.startswith("Evaluation/"):
        candidates.append(rel[len("Evaluation/"):])

    for cand in candidates:
        full = os.path.join(base_prefix, cand)
        if os.path.exists(full):
            return full, f"./{cand}"

    return os.path.join(base_prefix, rel), f"./{rel}"


def canonical_output_path(base_prefix: str, raw_path: str) -> str:
    return resolve_media_path(base_prefix, raw_path)[1]


def build_content(example: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    data_type = str(example.get("data_type") or "").strip().lower()
    media_path, _ = resolve_media_path(args.base_prefix, str(example.get("path") or ""))
    if data_type == "image":
        return [{
            "type": "image",
            "image": media_path,
            "max_pixels": args.max_pixels_image,
            "min_pixels": args.min_pixels_image,
        }]
    if data_type == "video":
        item = {
            "type": "video",
            "video": media_path,
            "max_pixels": args.max_pixels_video,
            "max_frames": args.max_frames,
            "fps": args.fps,
        }
        if os.getenv("EVAL_VIDEO_ITEM_ONETHINKER", "0") != "1":
            item["min_pixels"] = args.min_pixels_video
            item["total_pixels"] = args.total_pixels_video
        return [item]
    raise ValueError(f"Unsupported data_type for segmentation: {data_type!r}")


def prepare_vllm_input(
    messages: List[Dict[str, Any]],
    processor: Any,
    patch_size: Optional[int],
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    omit_thinking_kw = os.getenv("EVAL_OMIT_ENABLE_THINKING_KW", "0") == "1"
    force_do_resize_false = os.getenv("EVAL_FORCE_DO_RESIZE_FALSE", "0") == "1"

    if omit_thinking_kw:
        # OneThinker eval_bench.py omits this kwarg entirely. Keep this
        # switch for paper-parity runs with OneThinker checkpoints.
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    kwargs: Dict[str, Any] = {"return_video_kwargs": True, "return_video_metadata": True}
    if patch_size is None:
        patch_size = getattr(getattr(processor, "image_processor", None), "patch_size", None)
    if patch_size is not None:
        kwargs["image_patch_size"] = patch_size
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, **kwargs)
    video_kwargs = video_kwargs or {}
    if force_do_resize_false:
        # Diagnostic only. For Qwen3.5 video segmentation this can trigger
        # processor shape mismatches when decoded frames are not patch-aligned.
        video_kwargs["do_resize"] = False
    mm_data: Dict[str, Any] = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def parse_json_answer(answer: str) -> Optional[Dict[str, Any]]:
    if not answer:
        return None
    try:
        obj = json.loads(answer)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = answer.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(answer)):
        if answer[idx] == "{":
            depth += 1
        elif answer[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(answer[start:idx + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _is_number_list(value: Any, length: int) -> bool:
    if not isinstance(value, list) or len(value) != length:
        return False
    try:
        for item in value:
            float(item)
        return True
    except Exception:
        return False


def _is_point_list(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 1:
        return False
    return all(_is_number_list(point, 2) for point in value)


def normalize_seg_prompt(obj: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize common model variants to the SAM2 prompt schema.

    The prompt asks for one bounding box but uses the plural key "boxes". Some
    models naturally return a list-of-boxes (`[[x1, y1, x2, y2]]`), while the
    downstream OneThinker SAM2 script expects the single box as a flat list.
    """
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    boxes = out.get("boxes")
    if (
        isinstance(boxes, list)
        and len(boxes) == 1
        and _is_number_list(boxes[0], 4)
    ):
        out["boxes"] = boxes[0]
    return out


def valid_seg_prompt(obj: Optional[Dict[str, Any]], data_type: str) -> bool:
    if not isinstance(obj, dict):
        return False
    boxes = obj.get("boxes")
    if not _is_number_list(boxes, 4):
        return False
    for key in ("positive_points", "negative_points"):
        if not _is_point_list(obj.get(key)):
            return False
    if data_type == "video":
        try:
            float(obj.get("time"))
        except Exception:
            return False
    return True


def iter_filtered(data: Iterable[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows = []
    missing_media = 0
    for rec in data:
        if str(rec.get("problem_type", "")).strip().lower() != "segmentation":
            continue
        data_type = str(rec.get("data_type") or "").strip().lower()
        if data_type not in {"image", "video"}:
            continue
        if args.data_type != "all" and data_type != args.data_type:
            continue
        media_path, _ = resolve_media_path(args.base_prefix, str(rec.get("path") or ""))
        if args.skip_missing_media and not os.path.exists(media_path):
            missing_media += 1
            continue
        rows.append(rec)
    if missing_media:
        print(f"[warn] skipped {missing_media} samples with missing media", file=sys.stderr)
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    if args.chunk > 1:
        rows = [r for i, r in enumerate(rows) if i % args.chunk == args.index]
    return rows


def evaluate_dataset(dataset: str, args: argparse.Namespace, llm: LLM, processor: Any, sampling_params: SamplingParams) -> Dict[str, Any]:
    data_path = resolve_dataset_path(args.bench_dir, dataset)
    rows = iter_filtered(load_json_or_jsonl(data_path), args)

    outputs: List[Dict[str, Any]] = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"{dataset}"):
        batch = rows[start:start + args.batch_size]
        inputs = []
        for example in batch:
            messages = build_prompt(example, args.prompt_mode)
            user_msg = messages[-1]
            user_msg = dict(user_msg)
            user_msg["content"] = build_content(example, args) + user_msg["content"]
            run_messages = messages[:-1] + [user_msg]
            inputs.append(prepare_vllm_input(run_messages, processor, args.patch_size, args.enable_thinking))

        generated = llm.generate(inputs, sampling_params=sampling_params)
        texts = [out.outputs[0].text for out in generated]

        for example, text in zip(batch, texts):
            answer = extract_answer(text)
            parsed = parse_json_answer(answer)
            parsed = normalize_seg_prompt(parsed)
            data_type = str(example.get("data_type") or "").strip().lower()
            sample = dict(example)
            sample["path"] = canonical_output_path(args.base_prefix, str(example.get("path") or ""))
            sample["output"] = text
            sample["prediction"] = answer
            sample["predicted_answer_norm"] = (
                json.dumps(parsed, ensure_ascii=False) if isinstance(parsed, dict) else answer
            )
            sample["parsed_prediction"] = parsed
            sample["parse_ok"] = valid_seg_prompt(parsed, data_type)
            outputs.append(sample)

    total = len(outputs)
    parsed = sum(1 for row in outputs if row.get("parse_ok"))
    by_type: Dict[str, Dict[str, Any]] = {}
    for data_type in ("image", "video"):
        part = [r for r in outputs if r.get("data_type") == data_type]
        if part:
            ok = sum(1 for r in part if r.get("parse_ok"))
            by_type[data_type] = {
                "num_samples": len(part),
                "parse_rate": round(ok / len(part) * 100.0, 2),
            }
    return {
        "dataset": dataset,
        "input_file": str(data_path),
        "results": outputs,
        "metrics": {
            "num_samples": total,
            "parse_ok": parsed,
            "parse_rate": round(parsed / total * 100.0, 2) if total else 0.0,
            "by_data_type": by_type,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--bench_dir", required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names or JSON/JSONL paths.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_prefix", default="")
    parser.add_argument("--data_type", choices=["all", "image", "video"], default="all")
    parser.add_argument("--prompt_mode", choices=["think", "no_think", "bare", "onethink_system", "train_seg"], default="no_think")
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_missing_media", action="store_true")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pixels_image", type=int, default=1024 * 32 * 32)
    parser.add_argument("--min_pixels_image", type=int, default=4 * 32 * 32)
    parser.add_argument("--max_pixels_video", type=int, default=256 * 32 * 32)
    parser.add_argument("--min_pixels_video", type=int, default=4 * 32 * 32)
    parser.add_argument("--total_pixels_video", type=int, default=256 * 64 * 32 * 32)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=None)
    args = parser.parse_args()

    if args.chunk < 1 or not (0 <= args.index < args.chunk):
        raise ValueError("--chunk must be >=1 and --index must be in [0, chunk)")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.processor_path or args.model_path)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        mm_encoder_tp_mode="data",
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_k=args.top_k,
        stop_token_ids=[],
    )

    summary: Dict[str, Any] = {}
    for dataset in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        payload = evaluate_dataset(dataset, args, llm, processor, sampling_params)
        dataset_key = Path(dataset).stem if Path(dataset).suffix else dataset
        suffix = f"_shard{args.index}" if args.chunk > 1 else ""
        out_path = Path(args.output_dir) / f"results_{dataset_key}{suffix}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"results": payload["results"], "metrics": payload["metrics"]}, f, ensure_ascii=False, indent=2)
        summary[dataset_key] = payload["metrics"]
        print(
            f"{dataset_key}: n={payload['metrics']['num_samples']} "
            f"parse={payload['metrics']['parse_rate']:.2f}% -> {out_path}"
        )

    summary_name = f"summary_shard{args.index}.json" if args.chunk > 1 else "summary.json"
    with (Path(args.output_dir) / summary_name).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
