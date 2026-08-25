#!/usr/bin/env python3
"""Finalize an annotation-only OraRL evaluation index for publication."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from orarl.evaluation.card import render_index_card
from orarl.evaluation.manifest import load_dataset_manifest

ALLOWED_ROOT_ENTRIES = {
    ".gitattributes",
    "README.md",
    "annotations",
    "datasets.jsonl",
}


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _set_segmentation_reader(record: dict[str, Any], reader: str) -> None:
    if record.get("task") != "segmentation":
        return
    legacy = dict(record.get("legacy_environment", {}))
    legacy["SEGMENTATION_VIDEO_READER"] = reader
    setting = str(legacy.get("SEGMENTATION_SETTING", ""))
    marker = f"reader{reader}"
    if setting and marker not in setting:
        new_marker = setting.rfind("-new")
        setting = (
            f"{setting[:new_marker]}-{marker}{setting[new_marker:]}"
            if new_marker >= 0
            else f"{setting}-{marker}"
        )
        legacy["SEGMENTATION_SETTING"] = setting
    record["legacy_environment"] = legacy
    preprocessing = dict(record.get("preprocessing", {}))
    preprocessing["video_reader"] = reader
    record["preprocessing"] = preprocessing


def _annotation_assets(root: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for record in records:
        relative = str(record["annotation_path"])
        annotation = root / relative
        if annotation.is_symlink() or not annotation.is_file():
            raise FileNotFoundError(f"annotation is missing: {annotation}")
        with annotation.open("r", encoding="utf-8") as stream:
            row_count = sum(1 for line in stream if line.strip())
        expected = int(record["expected_count"])
        if row_count != expected:
            raise ValueError(
                f"{relative}: expected {expected} rows, found {row_count}"
            )
        assets.append(
            {
                "benchmark": str(record["benchmark"]),
                "bytes": annotation.stat().st_size,
                "kind": "annotations",
                "path": relative,
            }
        )
    return assets


def finalize_index(
    root_path: str | os.PathLike[str],
    *,
    repo_id: str,
    segmentation_video_reader: str,
) -> dict[str, int]:
    root = Path(root_path).expanduser().resolve()
    unexpected = sorted(
        path.name
        for path in root.iterdir()
        if path.name not in ALLOWED_ROOT_ENTRIES
    )
    if unexpected:
        raise ValueError(
            "metadata-only index contains unexpected root entries: "
            + ", ".join(unexpected)
        )

    records = [dict(record) for record in load_dataset_manifest(root)]
    for record in records:
        _set_segmentation_reader(record, segmentation_video_reader)
    manifest = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    _atomic_write(root / "datasets.jsonl", manifest)

    validated = [dict(record) for record in load_dataset_manifest(root)]
    assets = _annotation_assets(root, validated)
    _atomic_write(
        root / "README.md",
        render_index_card(validated, assets, repo_id=repo_id),
    )
    return {
        "annotations": len(assets),
        "bytes": sum(int(asset["bytes"]) for asset in assets),
        "rows": sum(int(record["expected_count"]) for record in validated),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-id", default="OraRL/OraRL-Data")
    parser.add_argument("--segmentation-video-reader", default="decord")
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_index(
                args.root,
                repo_id=args.repo_id,
                segmentation_video_reader=args.segmentation_video_reader,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
