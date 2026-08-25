#!/usr/bin/env python3
"""Move legacy evaluation outputs into paper-level task families."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

TASK_FAMILIES = {
    "video_qa": (
        "videomme",
        "videommev2",
        "videommmu",
        "mmvu",
        "mvbench",
        "videoholmes",
        "longvideobench",
        "lvbench",
        "mlvu",
    ),
    "spatial_intelligence": ("vsi", "mmsi", "mindcube", "revsi"),
    "spatial_temporal_grounding": ("stvg",),
}


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def planned_moves(root: Path) -> list[tuple[Path, Path]]:
    moves = []
    for family, tasks in TASK_FAMILIES.items():
        for task in tasks:
            source = root / task
            if source.is_dir():
                moves.append((source, root / family / task))
    return moves


def _preflight_merge(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        target = destination / child.name
        if not _exists(target):
            continue
        if child.is_dir() and not child.is_symlink():
            if not target.is_dir() or target.is_symlink():
                raise FileExistsError(f"cannot merge directory into {target}")
            _preflight_merge(child, target)
            continue
        raise FileExistsError(f"refusing to overwrite existing output: {target}")


def _merge(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir() and not child.is_symlink() and target.is_dir():
            _merge(child, target)
        else:
            child.rename(target)
    source.rmdir()


def organize(root: Path, moves: Iterable[tuple[Path, Path]]) -> None:
    moves = list(moves)
    for source, destination in moves:
        if _exists(destination):
            _preflight_merge(source, destination)

    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _exists(destination):
            _merge(source, destination)
        else:
            source.rename(destination)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="One model's output root, for example outputs/Video-ORA-9B.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the moves. Without this flag, only print the plan.",
    )
    return parser


def main() -> int:
    args = create_parser().parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"output root is not a directory: {root}")

    moves = planned_moves(root)
    if not moves:
        print(f"Already organized: {root}")
        return 0

    action = "MOVE" if args.apply else "PLAN"
    for source, destination in moves:
        print(f"{action} {source.relative_to(root)} -> {destination.relative_to(root)}")

    if args.apply:
        organize(root, moves)
        print(f"Organized {len(moves)} task director{'y' if len(moves) == 1 else 'ies'}.")
    else:
        print("Dry run only; pass --apply to move these directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
