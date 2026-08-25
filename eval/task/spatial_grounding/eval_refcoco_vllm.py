"""RefCOCO / RefCOCO+ / RefCOCOg evaluation with the vLLM backend.

Defaults use the joint-SFT answer-only RefCOCO prompt plus ``first``-bbox
scoring.

Usage:
    python eval/task/spatial_grounding/eval_refcoco_vllm.py \
        --model_path /path/to/model \
        --bench_dir  /path/to/OneThinker-eval \
        --datasets   refcoco-val,refcoco-testA,refcoco-testB,\\
                     refcoco+-val,refcoco+-testA,refcoco+-testB,\\
                     refcocog-val,refcocog-test \
        --output_dir outputs/eval_refcoco_vllm
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from _grounding_utils import (
    DATASET_CONFIGS,
    build_qwen_native_prompt,
    compute_iou_2d,
    extract_bboxes,
    load_annotations,
    sanitize_video_kwargs,
    select_pred_bbox,
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RefCOCO eval with vLLM")
    p.add_argument("--model_path", required=True)
    p.add_argument("--processor_path", default=None,
                   help="Defaults to --model_path.")
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--datasets", required=True,
                   help="Comma-separated dataset names. Available: "
                        + ",".join(DATASET_CONFIGS.keys()))
    p.add_argument("--output_dir", required=True)

    p.add_argument("--prompt_style", default="qwen_native",
                   choices=["qwen_official", "qwen_native", "legacy_r1", "eval_bench"],
                   help="qwen_official (default): Qwen team's RefCOCO eval "
                        "prompt — 'Locate every object that matches the "
                        "description \"...\" in the image. Report bbox "
                        "coordinates in JSON format.' "
                        "qwen_native: simpler 2D-grounding cookbook prompt. "
                        "legacy_r1: OurPO / ms-swift prompt — forces the "
                        "model to emit '<answer> (x1,y1),(x2,y2) </answer>' "
                        "with norm1000 coords (matches "
                        "grounding_prompt_wo_think.txt). Pair with "
                        "--coord_system norm1000 (default when chosen).")
    p.add_argument("--coord_system", default="auto",
                   choices=["auto", "pixel", "norm1000"],
                   help="What coordinate system the model emits in. Affects "
                        "ONLY pre-IoU rescaling; the OneThinker-eval RefCOCO "
                        "GT is in norm1000, so both branches end up comparing "
                        "in norm1000 space. 'auto' (default): norm1000 if "
                        "--prompt_style=legacy_r1, pixel otherwise. "
                        "'pixel': rescale [0,1] normalised → norm1000 by "
                        "(img_w, img_h); keep larger values as-is (original "
                        "behaviour, works because vanilla Qwen3.5-4B emits "
                        "norm1000 too). 'norm1000': never rescale (the model "
                        "is known to emit norm1000 — no [0,1] heuristic).")
    p.add_argument("--bbox_select", default="first",
                   choices=["first", "best_iou"],
                   help="When the model returns multiple candidate bboxes: "
                        "'first' (standard RefCOCO protocol, default) or "
                        "'best_iou' (ORACLE — peeks at GT, ablation only).")
    p.add_argument("--enable_thinking", default="false",
                   choices=["true", "false"],
                   help="Whether to render the chat template in thinking mode.")

    p.add_argument("--min_tokens", type=int, default=64,
                   help="min visual tokens (* downsample^2 = min image pixels)")
    p.add_argument("--total_tokens", type=int, default=1024,
                   help="max visual tokens (* downsample^2 = max image pixels). "
                        "1024 ≈ 1024x1024 pixels, already covers any COCO image "
                        "at native resolution; bumping higher rarely helps.")
    p.add_argument("--max_new_tokens", type=int, default=1024)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples per dataset before sharding; 0 evaluates all.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--tensor_parallel_size", type=int,
                   default=max(1, torch.cuda.device_count()))
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=32768)

    # Data-parallel sharding (one vLLM process per GPU group). When --chunk > 1,
    # this process handles samples whose global index % chunk == index, and
    # writes `results_<dataset>_shard{index}.json` for the launcher to merge.
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--index", type=int, default=0)

    args = p.parse_args()
    if args.max_samples < 0:
        p.error("--max_samples must be non-negative")
    args.enable_thinking = args.enable_thinking == "true"
    if args.processor_path is None:
        args.processor_path = args.model_path
    if args.coord_system == "auto":
        args.coord_system = (
            "norm1000" if args.prompt_style == "legacy_r1" else "pixel"
        )
    return args


# ---------------------------------------------------------------------------
# Prompt / vLLM input helpers
# ---------------------------------------------------------------------------

def _detect_downsample_rate(model_path: str) -> int:
    ml = (model_path or "").lower()
    if "qwen2" in ml and "qwen3" not in ml:
        return 28
    return 32  # qwen3 / qwen3.5 / unknown -> 32


_prompt_logged = False

def build_messages(anno: Dict[str, Any], args, dr: int):
    global _prompt_logged
    expression = anno["expression"]
    prompt_text = build_qwen_native_prompt(expression)

    if not _prompt_logged:
        print(f"\n{'='*60}")
        print(f"[PROMPT SAMPLE] style={args.prompt_style}, expr={expression!r}")
        print(f"{'='*60}")
        print(prompt_text)
        print(f"{'='*60}\n", flush=True)
        _prompt_logged = True

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": anno["image_path"],
                "min_pixels": args.min_tokens * dr * dr,
                "max_pixels": args.total_tokens * dr * dr,
            },
            {"type": "text", "text": prompt_text},
        ],
    }]
    return messages, bool(args.enable_thinking)


def prepare_vllm_input(messages, processor, chat_thinking, model_family,
                       process_vision_info):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=chat_thinking,
    )
    extra = {}
    if model_family == "qwen3":
        extra["image_patch_size"] = getattr(
            processor.image_processor, "patch_size", 16)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True, return_video_metadata=True, **extra,
    )
    video_kwargs = sanitize_video_kwargs(video_kwargs, has_video=bool(video_inputs))

    mm_data: Dict[str, Any] = {}
    if image_inputs:
        mm_data["image"] = image_inputs
    if video_inputs:
        mm_data["video"] = video_inputs
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }


# ---------------------------------------------------------------------------
# Per-split eval loop
# ---------------------------------------------------------------------------

IOU_THRESHOLDS = (0.5, 0.7, 0.9)


def evaluate_dataset(llm, sampling_params, processor, process_vision_info,
                     dataset_name: str, args, model_family: str, dr: int):
    print(f"\n>>> Evaluating: {dataset_name}")
    annos = load_annotations(args.bench_dir, dataset_name)
    annos.sort(key=lambda x: (x["image_path"], x["expression"]))
    if args.max_samples:
        annos = annos[: args.max_samples]
    n_total = len(annos)
    if args.chunk > 1:
        annos = [a for i, a in enumerate(annos) if i % args.chunk == args.index]
        print(f"  Shard {args.index}/{args.chunk}: {len(annos)}/{n_total} samples")
    else:
        print(f"  Loaded {n_total} samples")

    # ---- Resume from append-only partial JSONL ----------------------------
    suffix = (f"_shard{args.index}" if args.chunk > 1 else "")
    partial_path = os.path.join(
        args.output_dir, f"results_{dataset_name}{suffix}.partial.jsonl")

    recall = {t: 0 for t in IOU_THRESHOLDS}
    ious: List[float] = []
    n_parsed = 0
    results: List[Dict[str, Any]] = []
    done_keys: set = set()

    if os.path.isfile(partial_path):
        with open(partial_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                key = (rec.get("image"), rec.get("expression"))
                if key in done_keys:
                    continue
                done_keys.add(key)
                results.append(rec)
                iou_val = float(rec.get("iou") or 0.0)
                ious.append(iou_val)
                for t in recall:
                    if iou_val >= t:
                        recall[t] += 1
                if rec.get("pred_bbox") is not None:
                    n_parsed += 1
        if results:
            print(f"  [resume] loaded {len(results)} cached samples "
                  f"from {os.path.basename(partial_path)}")

    annos = [a for a in annos
             if (os.path.basename(a["image_path"]), a["expression"])
             not in done_keys]
    n = len(annos)
    if n == 0 and not results:
        return None, []

    t0 = time.time()
    bsz = max(1, int(args.batch_size))
    pf = open(partial_path, "a", buffering=1) if n > 0 else None

    for start in tqdm(range(0, n, bsz), desc=dataset_name):
        batch = annos[start:start + bsz]
        vllm_inputs = []
        for anno in batch:
            messages, chat_thinking = build_messages(anno, args, dr)
            try:
                vllm_inputs.append(prepare_vllm_input(
                    messages, processor, chat_thinking, model_family,
                    process_vision_info,
                ))
            except Exception as e:
                print(f"  [warn] prepare failed for {anno.get('image_path')}: {e}",
                      flush=True)
                vllm_inputs.append(None)

        keep_idx = [i for i, x in enumerate(vllm_inputs) if x is not None]
        valid_inputs = [vllm_inputs[i] for i in keep_idx]
        texts: List[str] = [""] * len(batch)
        if valid_inputs:
            try:
                outputs = llm.generate(valid_inputs, sampling_params=sampling_params)
                for j, out in zip(keep_idx, outputs):
                    texts[j] = out.outputs[0].text
            except Exception as e:
                print(f"  [error] vLLM generate failed at batch {start}: {e}",
                      flush=True)

        for anno, answer in zip(batch, texts):
            gt_box = anno["bbox"]
            candidates = extract_bboxes(answer)
            n_cands = len(candidates)
            if n_cands:
                n_parsed += 1
                w, h = anno.get("width"), anno.get("height")
                if not (w and h):
                    try:
                        from PIL import Image as _PILImage
                        with _PILImage.open(anno["image_path"]) as _im:
                            w, h = _im.size
                        anno["width"], anno["height"] = w, h
                    except Exception:
                        w, h = None, None
                if w and h:
                    for c in candidates:
                        mx = max(c["bbox"])
                        if args.coord_system == "norm1000":
                            # OneThinker-eval RefCOCO ships GT in norm1000
                            # already (e.g. [725, 632, 998, 1000] for a corner-
                            # crop), so predicted bbox in norm1000 is in the
                            # SAME space as GT — no rescale. We keep this
                            # branch (instead of falling through to the
                            # default no-op) for clarity and as an explicit
                            # contract: "I know the model emits norm1000;
                            # don't second-guess via [0,1] heuristics."
                            pass
                        else:  # pixel mode (current default behaviour)
                            if mx <= 1.0:
                                # Model emitted [0,1]-normalised; rescale.
                                c["bbox"] = [
                                    c["bbox"][0] * w, c["bbox"][1] * h,
                                    c["bbox"][2] * w, c["bbox"][3] * h,
                                ]

            pred_box, pred_label = select_pred_bbox(
                candidates, gt_box, mode=args.bbox_select)
            iou_val = compute_iou_2d(gt_box, pred_box) if pred_box is not None else 0.0
            ious.append(iou_val)
            for t in recall:
                if iou_val >= t:
                    recall[t] += 1

            rec = {
                "problem_id": anno.get("problem_id"),
                "image": os.path.basename(anno["image_path"]),
                "expression": anno["expression"],
                "gt_bbox": [round(float(x), 2) for x in gt_box],
                "pred_bbox": ([round(float(x), 2) for x in pred_box]
                              if pred_box else None),
                "pred_label": pred_label,
                "n_candidates": n_cands,
                "iou": round(iou_val, 4),
                "answer": answer,
            }
            results.append(rec)
            if pf is not None:
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if pf is not None:
            pf.flush()
            try:
                os.fsync(pf.fileno())
            except OSError:
                pass

    if pf is not None:
        pf.close()

    elapsed = time.time() - t0
    nn = len(ious)
    metrics = {
        "num_samples": nn,
        "mIoU": round(sum(ious) / nn * 100, 2) if nn else 0,
    }
    for t in IOU_THRESHOLDS:
        metrics[f"acc@{t}"] = round(recall[t] / nn * 100, 2) if nn else 0
    metrics["parse_rate"] = round(n_parsed / nn * 100, 2) if nn else 0

    print(f"  {dataset_name}: n={nn}  "
          f"mIoU={metrics['mIoU']:.2f}%  "
          f"acc@0.5={metrics['acc@0.5']:.2f}%  "
          f"acc@0.7={metrics['acc@0.7']:.2f}%  "
          f"acc@0.9={metrics['acc@0.9']:.2f}%  "
          f"Parse={metrics['parse_rate']:.2f}%  ({elapsed:.1f}s)")
    return metrics, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info

    torch.manual_seed(args.seed)
    print(f"Model:    {args.model_path}")
    print(f"Bench:    {args.bench_dir}")
    print(f"Datasets: {args.datasets}")
    print(f"TP size:  {args.tensor_parallel_size}")
    print(f"Tokens:   total={args.total_tokens}, max_new={args.max_new_tokens}")
    print(f"Output:   {args.output_dir}")

    processor = AutoProcessor.from_pretrained(args.processor_path,
                                              trust_remote_code=True)
    llm_kwargs = dict(
        model=args.model_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
    )
    # Newer vLLM exposes "data" mode for the multimodal encoder; older versions
    # don't have the kwarg, so fall back gracefully.
    try:
        llm = LLM(mm_encoder_tp_mode="data", **llm_kwargs)
    except TypeError:
        llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[],
    )

    model_family = "qwen2" if ("qwen2" in args.model_path.lower()
                               and "qwen3" not in args.model_path.lower()) else "qwen3"
    dr = _detect_downsample_rate(args.model_path)

    summary_path = os.path.join(args.output_dir, "summary.json")
    is_shard = args.chunk > 1
    summary: Dict[str, Any] = {}
    if not is_shard and os.path.isfile(summary_path):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for ds in datasets:
        if ds not in DATASET_CONFIGS:
            print(f"[skip] unknown dataset: {ds}")
            continue
        metrics, results = evaluate_dataset(
            llm, sampling_params, processor, process_vision_info,
            ds, args, model_family, dr,
        )
        if metrics is None:
            continue

        if is_shard:
            shard_path = os.path.join(
                args.output_dir, f"results_{ds}_shard{args.index}.json")
            with open(shard_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            partial_path = os.path.join(
                args.output_dir, f"results_{ds}_shard{args.index}.partial.jsonl")
            if os.path.isfile(partial_path):
                try: os.remove(partial_path)
                except OSError: pass
            print(f"  -> {shard_path}")
        else:
            result_path = os.path.join(args.output_dir, f"results_{ds}.json")
            with open(result_path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            partial_path = os.path.join(
                args.output_dir, f"results_{ds}.partial.jsonl")
            if os.path.isfile(partial_path):
                try: os.remove(partial_path)
                except OSError: pass
            summary[ds] = metrics
            with open(summary_path, "w") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"  -> {result_path}")

    if not is_shard:
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
