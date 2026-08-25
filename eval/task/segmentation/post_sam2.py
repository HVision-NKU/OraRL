#!/usr/bin/env python3
"""
Thin project-local wrapper around OneThinker's SAM2 segmentation post-process.

The OneThinker script contains the full image/video mask generation and metric
logic.  This wrapper only injects paths from CLI args so evaluation can be run
from this repository without editing the third-party checkout.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing.spawn as mp_spawn
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_ONETHINKER_SCRIPT = str(
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "OneThinker"
    / "Evaluation"
    / "Eval"
    / "seg_post_sam2.py"
)


def sam2_config_name(
    value: str,
    package_root: str | Path | None = None,
) -> str:
    """Convert an installed SAM2 YAML path to Hydra's package-relative name."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    if package_root is None:
        try:
            import sam2
        except ImportError:
            return str(path)
        package_root = Path(sam2.__file__).resolve().parent
    try:
        return path.resolve().relative_to(Path(package_root).resolve()).as_posix()
    except ValueError:
        return str(path)


def load_module(script_path: str):
    path = Path(script_path)
    if not path.is_file():
        raise FileNotFoundError(f"OneThinker SAM2 script not found: {path}")
    # Use the real filename as the module name so multiprocessing "spawn"
    # workers can import it from path.parent when unpickling worker_run.
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    # multiprocessing pickles functions by module name; dynamic modules must be
    # registered so child processes can resolve worker_run.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _results(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError("SAM2 input must be a list or an object containing `results`.")


def _segmentation_output(sample: dict[str, Any]) -> dict[str, Any] | None:
    output = sample.get("segmentation_output")
    if isinstance(output, dict):
        return output
    task_payload = sample.get("task_payload")
    if isinstance(task_payload, dict):
        output = task_payload.get("segmentation_output")
        if isinstance(output, dict):
            sample["segmentation_output"] = output
            return output
    return None


def _compressed_rle(
    size: Any,
    counts: list[Any],
    cocomask: Any,
) -> dict[str, Any] | None:
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or not all(isinstance(value, (int, float)) for value in size)
    ):
        return None
    height, width = (int(size[0]), int(size[1]))
    if height <= 0 or width <= 0:
        return None
    encoded = cocomask.frPyObjects(
        {"size": [height, width], "counts": counts},
        height,
        width,
    )
    if isinstance(encoded, list):
        encoded = cocomask.merge(encoded)
    encoded_counts = encoded["counts"]
    if isinstance(encoded_counts, bytes):
        encoded_counts = encoded_counts.decode("utf-8")
    return {"size": [height, width], "counts": encoded_counts}


def normalize_missing_rle_counts(payload: Any, cocomask: Any) -> int:
    """Repair mask annotations that encode empty masks or polygons without RLE counts."""

    repaired = 0
    for sample in _results(payload):
        segmentation = _segmentation_output(sample)
        if segmentation is None:
            continue
        rle = segmentation.get("segmentation_rle")

        if sample.get("data_type") == "video":
            if not isinstance(rle, dict):
                continue
            for frame, frame_rle in list(rle.items()):
                if not isinstance(frame_rle, dict):
                    continue
                counts = frame_rle.get("counts")
                if counts is not None and not isinstance(counts, list):
                    continue
                size = frame_rle.get("size")
                if (
                    counts is None
                    and (
                        not isinstance(size, (list, tuple))
                        or len(size) != 2
                    )
                ):
                    continue
                replacement = _compressed_rle(
                    size,
                    counts
                    if isinstance(counts, list)
                    else [int(size[0]) * int(size[1])],
                    cocomask,
                )
                if replacement is not None:
                    rle[frame] = replacement
                    repaired += 1
            continue

        if not isinstance(rle, dict):
            continue

        size = rle.get("size")
        counts = rle.get("counts")
        if isinstance(counts, list):
            replacement = _compressed_rle(size, counts, cocomask)
            if replacement is not None:
                segmentation["segmentation_rle"] = replacement
                repaired += 1
            continue
        if counts is not None:
            continue

        polygons = segmentation.get("segmentation_polygon")
        if polygons:
            if isinstance(polygons, list) and polygons and isinstance(
                polygons[0], (int, float)
            ):
                polygons = [polygons]
            if (
                isinstance(size, (list, tuple))
                and len(size) == 2
                and polygons
            ):
                height, width = int(size[0]), int(size[1])
                encoded = cocomask.frPyObjects(polygons, height, width)
                if isinstance(encoded, list):
                    encoded = cocomask.merge(encoded)
                counts = encoded["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("utf-8")
                segmentation["segmentation_rle"] = {
                    "size": [height, width],
                    "counts": counts,
                }
                repaired += 1
                continue

        replacement = None
        if isinstance(size, (list, tuple)) and len(size) == 2:
            replacement = _compressed_rle(
                size,
                [int(size[0]) * int(size[1])],
                cocomask,
            )
        if replacement is not None:
            segmentation["segmentation_rle"] = replacement
            repaired += 1

    return repaired


def _sam2_output_path(input_json: str | Path) -> Path:
    path = Path(input_json)
    return path.with_name(f"{path.stem}_sam2.json")


def _aggregate_rewards(payload: dict[str, Any]) -> None:
    ok_items = [
        row
        for row in _results(payload)
        if row.get("status") == "ok" and isinstance(row.get("reward"), dict)
    ]
    video_j: list[float] = []
    video_f: list[float] = []
    video_jf: list[float] = []
    image_iou: list[float] = []
    total_inter = 0
    total_union = 0

    for row in ok_items:
        reward = row["reward"]
        if all(key in reward for key in ("J", "F", "J&F")):
            video_j.append(float(reward["J"]))
            video_f.append(float(reward["F"]))
            video_jf.append(float(reward["J&F"]))
        if "IoU" in reward:
            image_iou.append(float(reward["IoU"]))
            total_inter += int(reward.get("inter", 0))
            total_union += int(reward.get("union", 0))

    averages: dict[str, float] = {}
    if video_j:
        averages.update(
            {
                "video/J": sum(video_j) / len(video_j),
                "video/F": sum(video_f) / len(video_f),
                "video/J&F": sum(video_jf) / len(video_jf),
            }
        )
    if image_iou:
        giou = sum(image_iou) / len(image_iou)
        averages.update(
            {
                "image/IoU": giou,
                "image/gIoU": giou,
                "image/cIoU": (
                    float(total_inter) / float(total_union) if total_union else 0.0
                ),
            }
        )
    payload["avg_rewards"] = averages


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _run_upstream(
    module: Any,
    *,
    input_json: Path,
    viz_ratio: float,
    normalize: bool,
) -> Path:
    original_load = module.json.load

    if normalize:
        def load_and_normalize(*args: Any, **kwargs: Any) -> Any:
            payload = original_load(*args, **kwargs)
            repaired = normalize_missing_rle_counts(payload, module.cocomask)
            if repaired:
                print(f"[INFO] normalized {repaired} mask RLE entries without counts")
            return payload

        module.json.load = load_and_normalize

    try:
        sys.argv = [
            str(Path(module.__file__).name),
            "--input_json",
            str(input_json),
            "--viz_ratio",
            str(viz_ratio),
        ]
        module.main()
    finally:
        module.json.load = original_load
    return _sam2_output_path(input_json)


def _retry_errors(
    module: Any,
    *,
    output_path: Path,
    attempts: int,
    viz_ratio: float,
) -> tuple[dict[str, Any], int]:
    with output_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)

    retried = 0
    result_fields = (
        "reward",
        "status",
        "viz_key_pred",
        "viz_key_gt",
        "viz_pred_video",
        "viz_gt_video",
    )
    module.WORKERS_PER_GPU = 1

    for attempt in range(1, attempts + 1):
        failed = [
            row
            for row in _results(payload)
            if str(row.get("status", "")).startswith("error:")
        ]
        if not failed:
            break
        retried += len(failed)
        retry_payload = {"results": failed}
        repaired = normalize_missing_rle_counts(retry_payload, module.cocomask)
        if repaired:
            print(f"[INFO] normalized {repaired} mask RLE entries without counts")
        print(
            f"[INFO] retrying {len(failed)} SAM2 errors "
            f"(attempt {attempt}/{attempts}, workers_per_gpu=1)"
        )

        with tempfile.TemporaryDirectory(
            dir=output_path.parent,
            prefix=".sam2-retry-",
        ) as temporary_dir:
            retry_input = Path(temporary_dir) / "retry.json"
            with retry_input.open("w", encoding="utf-8") as stream:
                json.dump(retry_payload, stream, ensure_ascii=False)
            retry_output = _run_upstream(
                module,
                input_json=retry_input,
                viz_ratio=viz_ratio,
                normalize=False,
            )
            with retry_output.open(encoding="utf-8") as stream:
                retry_result = json.load(stream)

        retried_by_id = {
            row.get("problem_id"): row for row in _results(retry_result)
        }
        for row in _results(payload):
            retry_row = retried_by_id.get(row.get("problem_id"))
            if retry_row is not None:
                for field in result_fields:
                    row[field] = retry_row.get(field)

    _aggregate_rewards(payload)
    _write_json_atomic(output_path, payload)
    return payload, retried


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Prediction JSON from eval_seg_vllm.py.")
    parser.add_argument("--data_root", required=True, help="Root joined with each record's relative `path`.")
    parser.add_argument("--sam2_ckpt", required=True)
    parser.add_argument("--sam2_cfg", required=True)
    parser.add_argument("--onethinker_script", default=DEFAULT_ONETHINKER_SCRIPT)
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--workers_per_gpu", type=int, default=None)
    parser.add_argument("--pre_extract_threads", type=int, default=None)
    parser.add_argument("--epoch_size", type=int, default=None)
    parser.add_argument("--viz_ratio", type=float, default=0.0)
    parser.add_argument(
        "--retry_attempts",
        type=int,
        default=1,
        help="Retry per-sample SAM2 errors with one worker per GPU.",
    )
    parser.add_argument(
        "--retry_existing_errors",
        action="store_true",
        help="Retry errors in an existing sibling *_sam2.json without rerunning all samples.",
    )
    args = parser.parse_args()

    # Python's spawn start method calls os.getcwd() for every new worker pool.
    # If the command was launched from a transient directory, later epochs can
    # fail with FileNotFoundError. Pin cwd to the project root.
    project_dir = Path(__file__).resolve().parents[3]
    os.chdir(project_dir)
    original_get_preparation_data = mp_spawn.get_preparation_data

    def get_preparation_data(name):
        try:
            return original_get_preparation_data(name)
        except FileNotFoundError:
            os.chdir(project_dir)
            return original_get_preparation_data(name)

    mp_spawn.get_preparation_data = get_preparation_data

    config_name = sam2_config_name(args.sam2_cfg)
    os.environ["SAM2_DATA_ROOT"] = args.data_root
    os.environ["SAM2_CKPT"] = args.sam2_ckpt
    os.environ["SAM2_CFG"] = config_name
    module = load_module(args.onethinker_script)
    module.DATA_ROOT = args.data_root
    module.SAM2_CKPT = args.sam2_ckpt
    module.SAM2_CFG = config_name
    if args.num_gpus is not None:
        module.NUM_GPUS = args.num_gpus
    if args.workers_per_gpu is not None:
        module.WORKERS_PER_GPU = args.workers_per_gpu
    if args.pre_extract_threads is not None:
        module.PRE_EXTRACT_THREADS = args.pre_extract_threads
    if args.epoch_size is not None:
        module.EPOCH_SIZE = args.epoch_size

    sys.argv = [
        str(Path(args.onethinker_script).name),
        "--input_json",
        args.input_json,
        "--viz_ratio",
        str(args.viz_ratio),
    ]
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.8",
    )
    module.mp.set_start_method("spawn", force=True)

    input_path = Path(args.input_json).resolve()
    output_path = _sam2_output_path(input_path)
    if not args.retry_existing_errors:
        output_path = _run_upstream(
            module,
            input_json=input_path,
            viz_ratio=args.viz_ratio,
            normalize=True,
        )
    elif not output_path.is_file():
        raise FileNotFoundError(f"Existing SAM2 result not found: {output_path}")

    payload, retried = _retry_errors(
        module,
        output_path=output_path,
        attempts=max(0, args.retry_attempts),
        viz_ratio=args.viz_ratio,
    )
    failed = [row for row in _results(payload) if row.get("status") != "ok"]
    print(
        f"[INFO] final SAM2 status: ok={len(_results(payload)) - len(failed)} "
        f"/ total={len(_results(payload))}; retried={retried}"
    )
    if failed:
        examples = ", ".join(
            f"{row.get('problem_id')}={row.get('status')}" for row in failed[:5]
        )
        raise RuntimeError(
            f"SAM2 post-processing left {len(failed)} failed samples: {examples}"
        )


if __name__ == "__main__":
    main()
