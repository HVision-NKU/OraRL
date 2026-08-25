#!/usr/bin/env python3
"""Unified vLLM evaluator for image-sequence multiple-choice benchmarks.

Supported benchmarks:
  - MMSI-Bench: TSV/parquet with base64 image list in `image`.
  - MindCube-Tiny: official 1,050-sample Hugging Face split.
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
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

from canonical_data import load_json_records
from mindcube.data_utils import (
    OFFICIAL_DATA_SOURCE,
    decode_mindcube_images,
    load_mindcube_records,
    mindcube_answer_value,
    mindcube_group,
    mindcube_prompt,
)
from PIL import Image

CHOICES = list("ABCDEFGH")
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
FORMAT_PRIORITY = {
    "start": 10,
    "end": 9,
    "phrase": 7,
    "parentheses": 6,
    "period": 5,
    "colon": 4,
    "right_paren": 3,
    "space": 2,
    "fallback": 0,
}
ANSWER_PHRASES = [
    "the answer is",
    "answer is",
    "the correct answer is",
    "correct answer is",
    "the best answer is",
    "best answer is",
    "the correct option is",
    "correct option is",
    "i choose",
    "i select",
    "my answer is",
    "答案是",
    "答案为",
]
PROMPT_TAIL = (
    "Choose the best answer from the options. "
    "Put exactly one uppercase option letter inside <answer>...</answer> "
    "Do not explain. Example: <answer>A</answer>"
)


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
    if matches:
        return matches[-1].strip()
    return (text or "").strip()


def extract_mcq_answer(response: str, choices: List[str] | None = None) -> str:
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


def load_records(args) -> List[Dict[str, Any]]:
    if args.bench == "mindcube":
        return load_mindcube_records(
            args.data_file,
            chunk=args.chunk,
            index=args.index,
            expected_samples=args.expected_samples,
            cache_dir=args.hf_cache_dir or None,
        )
    if args.data_file.endswith((".jsonl", ".json")):
        return load_json_records(args.data_file)
    if args.data_file.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(args.data_file).to_dict("records")

    csv.field_size_limit(sys.maxsize)
    with open(args.data_file, encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def shard(records: List[Dict[str, Any]], chunk: int, index: int) -> List[Dict[str, Any]]:
    if chunk <= 1:
        return records
    return records[index::chunk]


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


def answer_choices(choices: List[Any] | None) -> List[str]:
    n = len(choices) if choices is not None else 0
    return CHOICES[:n] if n > 0 else list("ABCD")


def choices_from_question(question: str) -> List[str]:
    found = re.findall(r"(?:^|[\s,])([A-H])\s*[:\).]", question or "")
    out = []
    for ch in found:
        if ch not in out:
            out.append(ch)
    return out or list("ABCD")


def build_prompt(args, rec: Dict[str, Any]) -> tuple[str, List[str]]:
    if args.bench == "mmsi":
        question = str(rec.get("question") or "").strip()
        labels = choices_from_question(question)
        return f"{question}\n{PROMPT_TAIL}", labels

    return mindcube_prompt(rec, PROMPT_TAIL)


def get_answer(args, rec: Dict[str, Any]) -> str:
    answer = (
        mindcube_answer_value(rec)
        if args.bench == "mindcube"
        else rec.get("answer")
    )
    if answer is None:
        return ""
    if args.bench == "mindcube" and isinstance(answer, (int, float)) and not isinstance(answer, bool):
        idx = int(answer)
        labels = answer_choices(normalize_choices(rec.get("choices")))
        if 0 <= idx < len(labels):
            return labels[idx]
    return extract_mcq_answer(str(answer), CHOICES) or str(answer).strip().upper()[:1]


def get_group_key(args, rec: Dict[str, Any]) -> str:
    if args.bench == "mmsi":
        return str(rec.get("category") or "unknown")
    return mindcube_group(rec)


def get_filter_key(args) -> str:
    return "category" if args.bench == "mmsi" else "task"


def render_chat_prompt(messages, processor, enable_thinking: bool = False) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def prepare_for_vllm(images: List[Image.Image], prompt: str, processor, args):
    from qwen_vl_utils import process_vision_info

    content: List[Dict[str, Any]] = []
    for image in images:
        item: Dict[str, Any] = {
            "type": "image",
            "image": image,
            "max_pixels": args.image_max_pixels,
        }
        if args.image_min_pixels > 0:
            item["min_pixels"] = args.image_min_pixels
        content.append(item)
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = render_chat_prompt(messages, processor, args.enable_thinking)
    image_inputs, _video_inputs = process_vision_info(messages, image_patch_size=args.patch_size)
    return {
        "prompt": text,
        "multi_modal_data": {"image": image_inputs},
        "mm_processor_kwargs": {"do_resize": False},
    }


def pct(correct: int, total: int) -> float:
    return round(100.0 * correct / total, 2) if total else 0.0


def aggregate(args, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = sum(int(r.get("score", 0)) for r in results)
    parsed = sum(1 for r in results if r.get("pred_answer"))
    groups = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        groups[str(r.get("group") or "unknown")]["total"] += 1
        groups[str(r.get("group") or "unknown")]["correct"] += int(r.get("score", 0))
    group_name = "by_category" if args.bench == "mmsi" else "by_task"
    return {
        "num_samples": len(results),
        "correct": correct,
        "accuracy": pct(correct, len(results)),
        "parse_rate": pct(parsed, len(results)),
        group_name: {
            k: {"accuracy": pct(v["correct"], v["total"]), "correct": v["correct"], "total": v["total"]}
            for k, v in sorted(groups.items())
        },
    }


def decode_images_for_record(args, rec: Dict[str, Any]) -> List[Image.Image]:
    if args.bench == "mmsi":
        return decode_mmsi_images(
            rec.get("images") or rec.get("image"),
            args.max_images,
        )
    return decode_mindcube_images(rec, args.max_images, args.min_image_bytes)


def evaluate(llm, sampling_params, processor, args) -> Dict[str, Any]:
    records = load_records(args)
    filter_value = args.category if args.bench == "mmsi" else args.task
    if filter_value:
        if args.bench == "mindcube":
            records = [
                record
                for record in records
                if mindcube_group(record) == filter_value
            ]
        else:
            key = get_filter_key(args)
            records = [
                record
                for record in records
                if str(record.get(key) or "") == filter_value
            ]
    if args.max_samples and args.max_samples > 0:
        records = records[: args.max_samples]
    if args.bench == "mmsi":
        records = shard(records, args.chunk, args.index)
    print(f"Loaded {len(records)} {args.bench} samples for shard {args.index}/{args.chunk}", flush=True)

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        inputs = []
        keep = []
        prompts: Dict[int, str] = {}
        labels_by_j: Dict[int, List[str]] = {}
        for j, rec in enumerate(batch):
            try:
                images = decode_images_for_record(args, rec)
                if not images:
                    raise ValueError("no decoded images")
                prompt, labels = build_prompt(args, rec)
                inputs.append(prepare_for_vllm(images, prompt, processor, args))
                keep.append(j)
                prompts[j] = prompt
                labels_by_j[j] = labels
            except Exception as e:
                print(f"[warn] skip sample {start + j}: {type(e).__name__}: {e}", flush=True)

        outputs = llm.generate(inputs, sampling_params=sampling_params) if inputs else []
        out_by_j = {j: out for j, out in zip(keep, outputs)}

        for j, rec in enumerate(batch):
            if j not in out_by_j:
                continue
            raw = out_by_j[j].outputs[0].text
            labels = labels_by_j[j]
            pred = extract_mcq_answer(raw, labels)
            gt = get_answer(args, rec)
            score = 1.0 if pred and gt and pred.upper() == gt.upper() else 0.0
            results.append({
                "id": str(rec.get("index") or rec.get("id") or rec.get("sample_id") or start + j),
                "bench": args.bench,
                "group": get_group_key(args, rec),
                "question": rec.get("question"),
                "prompt": prompts.get(j, ""),
                "answer": gt,
                "pred_answer": pred,
                "raw_prediction": raw,
                "score": score,
            })

        done = min(start + args.batch_size, len(records))
        if done % 200 == 0 or done == len(records):
            elapsed = max(time.time() - t0, 1e-6)
            summary = aggregate(args, results)
            print(
                f"[{done}/{len(records)}] {elapsed:.1f}s "
                f"acc={summary['accuracy']:.2f}% parse={summary['parse_rate']:.2f}%",
                flush=True,
            )

    full_mindcube_eval = (
        args.bench == "mindcube"
        and args.chunk == 1
        and not args.task
        and not args.max_samples
    )
    if (
        full_mindcube_eval
        and args.expected_samples > 0
        and len(results) != args.expected_samples
    ):
        raise RuntimeError(
            f"Expected {args.expected_samples} official MindCube-Tiny "
            f"results, got {len(results)}."
        )

    out_name = f"results_{args.bench}"
    if args.chunk > 1:
        out_name += f"_shard{args.index}"
    out_path = os.path.join(args.output_dir, out_name + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = aggregate(args, results)
    if args.bench == "mindcube":
        summary["data_source"] = args.data_file
    if full_mindcube_eval and args.expected_samples > 0:
        summary["expected_samples"] = args.expected_samples
        summary["coverage"] = round(
            100.0 * len(results) / args.expected_samples, 2
        )
    summary_name = f"summary_shard{args.index}.json" if args.chunk > 1 else "summary.json"
    with open(os.path.join(args.output_dir, summary_name), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Unified image-sequence MC evaluation via vLLM.")
    p.add_argument("--bench", required=True, choices=["mmsi", "mindcube"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--processor_path", default="")
    p.add_argument("--data_file", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--image_min_pixels", type=int, default=4096)
    p.add_argument("--image_max_pixels", type=int, default=262144)
    p.add_argument("--max_images", type=int, default=8)
    p.add_argument("--min_image_bytes", type=int, default=1024)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    p.add_argument("--max_model_len", type=int, default=65536)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--max_num_batched_tokens", type=int, default=65536)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--expected_samples", type=int, default=1050)
    p.add_argument("--hf_cache_dir", default="")
    p.add_argument("--category", default="")
    p.add_argument("--task", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--enable_thinking", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.bench == "mindcube" and not args.data_file:
        args.data_file = OFFICIAL_DATA_SOURCE
    if not args.data_file:
        raise ValueError("--data_file is required for MMSI-Bench")
    args.processor_path = args.processor_path or args.model_path
    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        padding_side="left",
        trust_remote_code=True,
        min_pixels=args.image_min_pixels,
        max_pixels=args.image_max_pixels,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.processor_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": args.max_images},
    )
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    evaluate(llm, sampling, processor, args)


if __name__ == "__main__":
    main()
