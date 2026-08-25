#!/usr/bin/env python3
"""Merge, validate, and officially summarize ReVSI vLLM shards."""

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_revsi_vllm import print_summary, summarise  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, default=0)
    args = parser.parse_args()

    missing = [
        shard
        for shard in range(args.num_shards)
        if not (args.output_dir / f"results_shard{shard}.jsonl").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)}/{args.num_shards} ReVSI shards: {missing}"
        )

    records_by_id = {}
    for shard in range(args.num_shards):
        path = args.output_dir / f"results_shard{shard}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                sample_id = record.get("id")
                if sample_id is None:
                    raise RuntimeError(f"{path}:{line_number}: record has no id")
                records_by_id[str(sample_id)] = record

    records = list(records_by_id.values())
    records.sort(
        key=lambda row: (
            0,
            int(row["id"]),
        )
        if str(row["id"]).isdigit()
        else (1, str(row["id"]))
    )
    if args.expected_samples > 0 and len(records) != args.expected_samples:
        raise RuntimeError(
            f"Expected {args.expected_samples} unique ReVSI samples, "
            f"got {len(records)}"
        )

    merged_path = args.output_dir / "merged_results.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarise(records)
    summary["num_shards"] = args.num_shards
    summary["frame_budgets"] = sorted(
        {str(record.get("num_frames") or "") for record in records}
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print_summary(summary)
    print(f"Merged: {merged_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
