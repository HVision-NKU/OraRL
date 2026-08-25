#!/usr/bin/env python3
"""MMSI-Bench evaluation with Hugging Face Transformers.

Unlike the vLLM evaluator, this backend can pass every image in a sample
without configuring a fixed multimodal-item limit.
"""

from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))

from eval_image_mc_vllm import (  # noqa: E402
    aggregate,
    build_prompt,
    extract_mcq_answer,
    get_answer,
    get_group_key,
    load_records,
    shard,
)


warnings.filterwarnings("ignore", message=".*pad_token_id.*")
logging.getLogger("transformers").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MMSI-Bench evaluation via Hugging Face Transformers."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--processor_path", default="")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_min_pixels", type=int, default=4096)
    parser.add_argument("--image_max_pixels", type=int, default=262144)
    parser.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Maximum images per sample; <=0 means use every image.",
    )
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--category", default="")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    args = parser.parse_args()
    args.bench = "mmsi"
    args.processor_path = args.processor_path or args.model_path
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_all_images(value: Any, max_images: int) -> List[Image.Image]:
    values = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(values, (list, tuple)):
        values = [values]
    if max_images > 0:
        values = values[:max_images]

    images: List[Image.Image] = []
    for item in values:
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
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
            continue
        raw = item if isinstance(item, bytes) else base64.b64decode(str(item))
        with Image.open(io.BytesIO(raw)) as image:
            images.append(image.convert("RGB"))
    return images


def prepare_inputs(
    images: List[Image.Image],
    prompt: str,
    processor,
    args: argparse.Namespace,
):
    content: List[Dict[str, Any]] = []
    for image in images:
        content.append(
            {
                "type": "image",
                "image": image,
                "min_pixels": args.image_min_pixels,
                "max_pixels": args.image_max_pixels,
            }
        )
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    image_inputs, _ = process_vision_info(
        messages,
        image_patch_size=args.patch_size,
    )
    return processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )


def sample_id(record: Dict[str, Any], fallback: int) -> str:
    for key in ("index", "id", "sample_id"):
        value = record.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(fallback)


def write_progress(
    output_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
    results: List[Dict[str, Any]],
) -> None:
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(aggregate(args, results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MMSI Transformers evaluation.")
    if args.chunk <= 0 or not 0 <= args.index < args.chunk:
        raise ValueError(
            f"Invalid shard index: index={args.index}, chunk={args.chunk}"
        )

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"results_mmsi_shard{args.index}.json"
    summary_path = output_dir / f"summary_shard{args.index}.json"

    print(f"Model:       {args.model_path}", flush=True)
    print(f"Processor:   {args.processor_path}", flush=True)
    print(f"Data:        {args.data_file}", flush=True)
    print(f"Shard:       {args.index}/{args.chunk}", flush=True)
    print(
        "Images:      "
        f"{'all' if args.max_images <= 0 else args.max_images} "
        f"(min_px={args.image_min_pixels}, max_px={args.image_max_pixels})",
        flush=True,
    )

    print("Loading Transformers model ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )

    records = load_records(args)
    if args.category:
        records = [
            record
            for record in records
            if str(record.get("category") or "") == args.category
        ]
    if args.max_samples > 0:
        records = records[: args.max_samples]
    indexed_records = list(enumerate(records))
    indexed_records = shard(indexed_records, args.chunk, args.index)
    print(f"Loaded {len(indexed_records)} samples for this shard.", flush=True)

    results: List[Dict[str, Any]] = []
    started = time.time()
    for local_index, (source_index, record) in enumerate(indexed_records, 1):
        images: List[Image.Image] = []
        raw_prediction = ""
        error = ""
        prompt, labels = build_prompt(args, record)
        ground_truth = get_answer(args, record)
        try:
            images = decode_all_images(
                record.get("images") or record.get("image"),
                args.max_images,
            )
            if not images:
                raise ValueError("sample contains no decodable images")
            inputs = prepare_inputs(images, prompt, processor, args).to(
                "cuda", non_blocking=True
            )
            input_length = inputs.input_ids.shape[-1]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )
            generated = generated[:, input_length:]
            raw_prediction = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            del inputs, generated
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
            print(
                f"[warn] sample={source_index} images={len(images)}: {error}",
                flush=True,
            )
        finally:
            for image in images:
                image.close()

        prediction = extract_mcq_answer(raw_prediction, labels)
        score = float(
            bool(prediction)
            and bool(ground_truth)
            and prediction.upper() == ground_truth.upper()
        )
        results.append(
            {
                "id": sample_id(record, source_index),
                "bench": "mmsi",
                "group": get_group_key(args, record),
                "question": record.get("question"),
                "prompt": prompt,
                "answer": ground_truth,
                "pred_answer": prediction,
                "raw_prediction": raw_prediction,
                "num_images": len(images),
                "score": score,
                "error": error,
            }
        )

        if (
            local_index % args.save_interval == 0
            or local_index == len(indexed_records)
        ):
            write_progress(result_path, summary_path, args, results)
            summary = aggregate(args, results)
            print(
                f"[{local_index}/{len(indexed_records)}] "
                f"{time.time() - started:.1f}s "
                f"acc={summary['accuracy']:.2f}% "
                f"parse={summary['parse_rate']:.2f}%",
                flush=True,
            )

    print(json.dumps(aggregate(args, results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
