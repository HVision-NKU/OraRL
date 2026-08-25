"""
Evaluate temporal grounding on TimeLens-Bench with HuggingFace transformers.

Self-contained: no dependency on nncore or the timelens package.
Supports Qwen3-VL / Qwen3.5-VL model families.

Usage (single GPU):
    python eval/eval_timelens_hf.py \
        --model_path /path/to/model \
        --bench_dir /path/to/TimeLens-Bench \
        --dataset charades-timelens \
        --output_dir outputs/eval_run

Multi-GPU (launched by run_eval.sh):
    CUDA_VISIBLE_DEVICES=0 python eval/eval_timelens_hf.py \
        --model_path /path/to/model ... --chunk 8 --index 0 &
    ...
"""

import argparse
import copy
import json
import logging
import os
import random
import re
import time
import warnings
from pathlib import Path

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList

warnings.filterwarnings("ignore", message=".*pad_token_id.*")
warnings.filterwarnings(
    "ignore",
    message="The video decoding and encoding capabilities of torchvision are deprecated.*",
)
logging.getLogger("transformers").setLevel(logging.ERROR)


class StopOnDecodedText(StoppingCriteria):
    def __init__(self, tokenizer, start_len: int, stop_text: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.start_len = start_len
        self.stop_text = stop_text

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids[:, self.start_len:]
        if generated.numel() == 0:
            return False
        stopped = []
        for seq in generated:
            text = self.tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            stopped.append(self.stop_text in text)
        return all(stopped)


def truncate_after_first_answer(text: str) -> str:
    end = text.find("</answer>")
    if end < 0:
        return text
    return text[: end + len("</answer>")]


# ---------------------------------------------------------------------------
# Prompts (aligned with TimeLens evaluation)
# ---------------------------------------------------------------------------

# Temporal grounding prompts — single source of truth in
# eval/task/eval_prompt.py (PROMPT_WO_THINK matches sft_joint_all.jsonl).
import sys as _sys
_EVAL_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EVAL_TASK_DIR not in _sys.path:
    _sys.path.insert(0, _EVAL_TASK_DIR)

from canonical_data import canonical_dataset_record, load_json_records  # noqa: E402
from eval_prompt import PROMPT_WO_THINK, TIMELENS_OFFICIAL_PROMPT  # noqa: E402

GROUNDER_PROMPT = TIMELENS_OFFICIAL_PROMPT


# ---------------------------------------------------------------------------
# Timestamp extraction (from TimeLens)
# ---------------------------------------------------------------------------

def extract_time(paragraph):
    paragraph = paragraph.lower()

    # Isolate the final answer. Prefer the LAST <answer>...</answer> block; if the
    # closing tag is missing (generation truncated, or stripped by a stop string),
    # fall back to everything after the last opening <answer>. This prevents
    # parsing timestamps out of the CoT reasoning that precedes the answer.
    answer_blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", paragraph)
    if answer_blocks:
        paragraph = answer_blocks[-1]
    else:
        open_idx = paragraph.rfind("<answer>")
        if open_idx >= 0:
            paragraph = paragraph[open_idx + len("<answer>"):]

    timestamps = []

    time_regex = re.compile(
        r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b"
    )
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]

    if time_matches:
        time_matches_converted = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:
                h, m = map(int, parts[:2])
                s = float(parts[2])
                time_in_sec = h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                time_in_sec = m * 60 + s
            time_matches_converted.append(float(time_in_sec))
        timestamps = [
            (time_matches_converted[i], time_matches_converted[i + 1])
            for i in range(0, len(time_matches_converted), 2)
        ]

    if len(timestamps) == 0:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
        ]
        for time_pattern in patterns:
            time_matches = re.findall(time_pattern, paragraph)
            if time_matches:
                timestamps = [(float(s), float(e)) for s, e in time_matches]
                break

    if len(timestamps) == 0:
        time_regex = re.compile(r"\b(\d+\.\d+|\d+)\b")
        time_matches = re.findall(time_regex, paragraph)
        time_matches = time_matches[: len(time_matches) // 2 * 2]
        timestamps = [
            (float(time_matches[i]), float(time_matches[i + 1]))
            for i in range(0, len(time_matches), 2)
        ]

    return timestamps


def compute_iou(a, b):
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    return max(min1 - max0, 0) / (max1 - min0) if (max1 - min0) > 0 else 0.0


# ---------------------------------------------------------------------------
# Data loading (self-contained, no timelens package needed)
# ---------------------------------------------------------------------------

def parse_query(query):
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


DATASET_CONFIGS = {
    "charades-timelens": {
        "anno": "charades-timelens.json",
        "video_subdir": "video_shards/charades",
    },
    "activitynet-timelens": {
        "anno": "activitynet-timelens.json",
        "video_subdir": "video_shards/activitynet",
    },
    "qvhighlights-timelens": {
        "anno": "qvhighlights-timelens.json",
        "video_subdir": "video_shards/qvhighlights",
    },
}


def load_annotations(bench_dir, dataset_name):
    canonical_path = Path(bench_dir) / f"{dataset_name}.jsonl"
    if canonical_path.is_file():
        annos = []
        for row in load_json_records(canonical_path):
            span = row.get("span", row.get("answer"))
            if (
                isinstance(span, (list, tuple))
                and len(span) == 2
                and not isinstance(span[0], (list, tuple))
            ):
                span = [list(span)]
            annos.append(
                {
                    "video_path": row.get("video_path") or row.get("path"),
                    "duration": row.get("duration", 0),
                    "query": parse_query(row.get("problem", "")),
                    "span": span,
                }
            )
        return annos

    cfg = DATASET_CONFIGS[dataset_name]
    anno_path = os.path.join(bench_dir, cfg["anno"])
    video_root = os.path.join(bench_dir, cfg["video_subdir"])

    with open(anno_path, "r") as f:
        raw = json.load(f)

    annos = []
    for vid, info in raw.items():
        video_path = os.path.join(video_root, vid + ".mp4")
        duration = info.get("duration", 0)
        queries = info.get("queries", info.get("sentences", []))
        spans = info.get("spans", info.get("timestamps", []))
        for span, query in zip(spans, queries):
            annos.append(dict(
                video_path=video_path,
                duration=duration,
                query=parse_query(query),
                span=[span] if not isinstance(span[0], (list, tuple)) else span,
            ))
    return annos


# ---------------------------------------------------------------------------
# Dataset (aligned with TimeLens evaluation/utils.py)
# ---------------------------------------------------------------------------

class GroundingDataset(Dataset):
    def __init__(self, annos, processor, args):
        super().__init__()
        self.annos = annos
        self.processor = processor
        self.args = args

        model_lower = args.model_path.lower()
        if getattr(args, "prompt_mode", "same") == "timelens_official":
            self.prompt = TIMELENS_OFFICIAL_PROMPT
        else:
            # Default remains aligned with data/joint/sft_joint_all.jsonl.
            self.prompt = PROMPT_WO_THINK

        if "qwen3" in model_lower or "timelens-8b" in model_lower:
            self.downsample_rate = 32
            self.model_family = "qwen3"
        elif "qwen2" in model_lower or "timelens-7b" in model_lower:
            self.downsample_rate = 28
            self.model_family = "qwen2"
        else:
            self.downsample_rate = 32
            self.model_family = "qwen3"

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])
        video_path = anno["video_path"]
        query = anno["query"]
        dr = self.downsample_rate

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": self.args.min_tokens * dr * dr,
                    "max_pixels": self.args.max_pixels,
                    "total_pixels": self.args.total_tokens * dr * dr,
                    "max_frames": self.args.max_frames,
                    "fps": self.args.fps,
                },
                {"type": "text", "text": self.prompt.format(query)},
            ],
        }]

        chat_kwargs = dict(tokenize=False, add_generation_prompt=True,
                           enable_thinking=self.args.enable_thinking)
        text = self.processor.apply_chat_template(messages, **chat_kwargs)

        if self.model_family == "qwen3":
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        else:
            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )

        return {
            "inputs": inputs,
            "anno": anno,
            "prompt": text,
            "prompt_template": self.prompt,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="TimeLens-Bench eval with HuggingFace")
    p.add_argument("--model_path", required=True)
    p.add_argument("--bench_dir", required=True)
    p.add_argument(
        "--dataset",
        required=True,
        help="Legacy TimeLens name or a canonical snake_case split.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--enable_thinking", default="false", choices=["true", "false"])
    p.add_argument(
        "--prompt_mode",
        default="same",
        choices=["same", "timelens_official"],
        help="same: training-aligned <answer> prompt; "
        "timelens_official: exact official TimeLens prompt",
    )
    p.add_argument("--min_tokens", type=int, default=1)
    p.add_argument("--total_tokens", type=int, default=128000)
    p.add_argument("--max_pixels", type=int, default=409600)
    p.add_argument("--max_frames", type=int, default=2048)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--num_return_sequences", type=int, default=1)
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum prompts in this shard; 0 evaluates the full shard.",
    )
    p.add_argument(
        "--stop_after_answer",
        default="true",
        choices=["true", "false"],
        help="Stop decoding immediately after the first generated </answer>.",
    )
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--compile", action="store_true", help="Use torch.compile for faster inference")
    p.add_argument("--processor_path", default=None, help="Path to load processor from (defaults to model_path)")
    args = p.parse_args()
    args.enable_thinking = args.enable_thinking == "true"
    args.stop_after_answer = args.stop_after_answer == "true"
    if args.num_return_sequences < 1:
        p.error("--num_return_sequences must be >= 1")
    if args.num_return_sequences > 1 and args.temperature <= 0:
        p.error("--num_return_sequences > 1 requires --temperature > 0")
    if args.processor_path is None:
        args.processor_path = args.model_path
    return args


def main():
    args = parse_args()
    canonical_profile = canonical_dataset_record("temporal_grounding", args.dataset)
    if canonical_profile is not None:
        preprocessing = canonical_profile.get("preprocessing", {})
        if isinstance(preprocessing, dict):
            for name in (
                "fps",
                "min_tokens",
                "max_frames",
                "max_pixels",
                "total_tokens",
            ):
                if name in preprocessing:
                    setattr(args, name, preprocessing[name])
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(
        args.output_dir,
        f"results_{args.dataset}_shard{args.index}.json",
    )

    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset} | Chunk {args.index}/{args.chunk}")
    print(
        f"Thinking: {args.enable_thinking} | Prompt mode: {args.prompt_mode} "
        f"| FPS: {args.fps}"
    )
    print(f"Tokens: min={args.min_tokens}, total={args.total_tokens}")
    print(f"Video: max_pixels={args.max_pixels}, max_frames={args.max_frames}")
    print(f"Output: {result_path}")

    # Load model
    print("\nLoading model ...")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for TimeLens HF evaluation because inputs are moved "
            "to CUDA and FlashAttention2 is enabled."
        )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": "cuda:0"},
    ).eval()

    if args.compile:
        print("Compiling model with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")

    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    # Load data
    annos = load_annotations(args.bench_dir, args.dataset)
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]
    if args.max_samples > 0:
        annos = annos[: args.max_samples]
    print(f"Loaded {len(annos)} samples (shard {args.index}/{args.chunk})")

    dataset = GroundingDataset(annos, processor, args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        pin_memory=True,
        collate_fn=lambda x: x[0],
    )

    # Inference
    results = []
    ious = []
    recall = {0.3: 0, 0.5: 0, 0.7: 0}
    t0 = time.time()

    data_wait = 0.0
    gen_total = 0.0
    sample_idx = 0
    iter_start = time.time()

    for data in tqdm(loader, desc=f"shard-{args.index}"):
        t_data = time.time()
        data_elapsed = t_data - iter_start
        data_wait += data_elapsed

        inputs = data["inputs"].to("cuda", non_blocking=True)
        anno = data["anno"]
        prompt = data["prompt"]
        prompt_template = data["prompt_template"]
        duration = anno["duration"]
        span = anno["span"]
        if isinstance(span[0], (list, tuple)):
            span = span[0]

        input_len = inputs.input_ids.shape[-1]
        stopping_criteria = None
        if args.stop_after_answer:
            stopping_criteria = StoppingCriteriaList([
                StopOnDecodedText(processor.tokenizer, input_len, "</answer>")
            ])

        t_gen = time.time()
        do_sample = args.temperature > 0
        output_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            top_k=None,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stopping_criteria=stopping_criteria,
        )
        gen_elapsed = time.time() - t_gen
        gen_total += gen_elapsed

        trimmed = [out[input_len:] for out in output_ids]
        answers = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if args.stop_after_answer:
            answers = [truncate_after_first_answer(answer) for answer in answers]

        sample_idx += 1
        if sample_idx <= 30 or sample_idx % 50 == 0:
            print(f"  [{sample_idx}] data={data_elapsed:.1f}s gen={gen_elapsed:.1f}s "
                  f"in={input_len} n={len(answers)} out={trimmed[0].shape[0]} | {answers[0][:80]}",
                  flush=True)

        for rollout_index, answer in enumerate(answers):
            timestamps = extract_time(answer)
            parsed = bool(timestamps)
            if not timestamps:
                timestamps = [(duration + 10, duration + 20)]
            timestamps = [(round(s), round(e)) for s, e in timestamps]
            pred = timestamps[0]

            iou_val = compute_iou(span, pred)
            ious.append(iou_val)
            for t in recall:
                if iou_val >= t:
                    recall[t] += 1

            results.append({
                "video": os.path.basename(anno["video_path"]),
                "query": anno["query"],
                "gt_span": span,
                "pred_span": list(pred),
                "iou": round(iou_val, 4),
                "answer": answer,
                "parsed": parsed,
                "rollout_index": rollout_index,
                "prompt": prompt,
                "prompt_template": prompt_template,
                "duration": duration,
            })

        if sample_idx % 50 == 0 or sample_idx == len(dataset):
            n_cur = len(ious)
            cur_metrics = {
                "num_samples": n_cur,
                "mIoU": round(sum(ious) / n_cur * 100, 2),
            }
            for t in [0.3, 0.5, 0.7]:
                cur_metrics[f"R@{t}"] = round(recall[t] / n_cur * 100, 2)
            print(f"  [checkpoint {sample_idx}/{len(dataset)}] "
                  f"mIoU={cur_metrics['mIoU']:.2f}%  R@0.3={cur_metrics['R@0.3']:.2f}%  "
                  f"R@0.5={cur_metrics['R@0.5']:.2f}%  R@0.7={cur_metrics['R@0.7']:.2f}%",
                  flush=True)
            with open(result_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        iter_start = time.time()

    elapsed = time.time() - t0
    n = len(ious)
    print(f"\nInference done: {n} samples in {elapsed:.1f}s ({n / elapsed:.1f} samples/s)")
    print(f"  data_wait={data_wait:.1f}s  gen={gen_total:.1f}s  "
          f"avg_data={data_wait/max(1, sample_idx):.2f}s  "
          f"avg_gen={gen_total/max(1, sample_idx):.2f}s")

    # Metrics
    metrics = {
        "num_samples": n,
        "mIoU": round(sum(ious) / n * 100, 2) if n else 0,
    }
    for t in [0.3, 0.5, 0.7]:
        metrics[f"R@{t}"] = round(recall[t] / n * 100, 2) if n else 0
    n_parsed = sum(1 for r in results if r.get("parsed", False))
    metrics["parse_rate"] = round(n_parsed / n * 100, 2) if n else 0

    print(f"  mIoU={metrics['mIoU']:.2f}%  R@0.3={metrics['R@0.3']:.2f}%  "
          f"R@0.5={metrics['R@0.5']:.2f}%  R@0.7={metrics['R@0.7']:.2f}%  "
          f"Parse={metrics['parse_rate']:.2f}%")

    # Save
    with open(result_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(
        args.output_dir,
        f"summary_shard{args.index}.json",
    )
    with open(summary_path, "w") as f:
        json.dump({args.dataset: metrics}, f, ensure_ascii=False, indent=2)

    print(f"Results -> {result_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
