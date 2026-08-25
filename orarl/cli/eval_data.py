"""Inventory, build, and validate canonical OraRL evaluation data."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from orarl.evaluation.card import (
    DEFAULT_DATASET_REPO_ID,
    write_huggingface_metadata,
)
from orarl.evaluation.hub import upload_evaluation_repository
from orarl.evaluation.staging import (
    build_evaluation_repository,
    export_evaluation_index,
    export_public_evaluation_repository,
    inventory_and_write_locked_manifest,
    inventory_evaluation_sources,
    merge_evaluation_repository,
    validate_staged_repository,
)


def _add_checksum_option(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checksums",
        dest="checksums",
        action="store_true",
        help="Opt in to SHA-256 generation or verification.",
    )
    group.add_argument(
        "--skip-checksums",
        dest="checksums",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(checksums=False)


def create_parser() -> argparse.ArgumentParser:
    """Create the dependency-light ``orarl-eval-data`` command parser."""

    parser = argparse.ArgumentParser(prog="orarl-eval-data", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser(
        "inventory",
        help="Read source rows and referenced asset sizes without writing output.",
    )
    inventory.add_argument("manifest_path", nargs="?")
    inventory.add_argument(
        "--manifest",
        "--source-manifest",
        dest="manifest",
        help="Local JSONL source manifest.",
    )
    inventory.add_argument(
        "--write-locked-manifest",
        metavar="FILE",
        help="Write a private deterministic manifest with actual converted row counts.",
    )
    inventory.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Inventory only one eval task; repeat to select multiple tasks.",
    )
    inventory.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel source-conversion workers (default: 8).",
    )
    _add_checksum_option(inventory)

    build = commands.add_parser(
        "build",
        help="Build an atomic canonical evaluation repository.",
    )
    build.add_argument("manifest_path", nargs="?")
    build.add_argument("output_path", nargs="?")
    build.add_argument(
        "--manifest",
        "--source-manifest",
        dest="manifest",
        help="Local JSONL source manifest.",
    )
    build.add_argument("--output", dest="output", help="Target repository directory.")
    build.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="copy",
        help="Stage files by copying or hard-linking with copy fallback.",
    )
    build.add_argument("--overwrite", action="store_true")
    build.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel source-conversion workers (default: 8).",
    )
    _add_checksum_option(build)

    merge = commands.add_parser(
        "merge",
        help="Add a task repository to an existing canonical repository.",
    )
    merge.add_argument("--source", required=True, help="Validated task repository.")
    merge.add_argument(
        "--target",
        "--output",
        dest="target",
        required=True,
        help="Existing cumulative evaluation repository.",
    )
    merge.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help="Add files by hard-linking with copy fallback, or always copy.",
    )
    merge.add_argument(
        "--repair-missing-asset-manifest",
        action="store_true",
        help=(
            "Reconstruct a deleted target assets.jsonl from staged paths and "
            "byte sizes without hashing payloads."
        ),
    )
    merge.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel file metadata and hardlink workers (default: 8).",
    )
    merge.add_argument(
        "--replace-benchmark",
        action="append",
        default=[],
        help=(
            "Replace every existing split and asset for this benchmark; repeat "
            "for multiple benchmarks."
        ),
    )

    export_index = commands.add_parser(
        "export-index",
        help="Export portable JSONL metadata while keeping large assets external.",
    )
    export_index.add_argument(
        "--root",
        required=True,
        help="Complete validated evaluation repository.",
    )
    export_index.add_argument(
        "--output",
        required=True,
        help="Target JSONL metadata directory.",
    )
    export_index.add_argument("--overwrite", action="store_true")
    _add_checksum_option(export_index)

    export_public = commands.add_parser(
        "export-public",
        help="Export annotations and raw media while excluding processed artifacts.",
    )
    export_public.add_argument(
        "--root",
        required=True,
        help="Complete validated evaluation repository.",
    )
    export_public.add_argument(
        "--output",
        required=True,
        help="Target public evaluation repository.",
    )
    export_public.add_argument(
        "--copy-mode",
        choices=("copy", "hardlink"),
        default="hardlink",
        help="Stage raw media by hard-linking with copy fallback, or always copy.",
    )
    export_public.add_argument("--overwrite", action="store_true")
    export_public.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel raw-media staging workers (default: 8).",
    )
    _add_checksum_option(export_public)

    metadata = commands.add_parser(
        "metadata",
        help="Regenerate and validate Hugging Face README and LFS metadata.",
    )
    metadata.add_argument("root_path", nargs="?")
    metadata.add_argument("--root", help="Staged evaluation repository.")
    metadata.add_argument(
        "--repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Hugging Face repository id used in the generated card.",
    )
    _add_checksum_option(metadata)

    validate = commands.add_parser(
        "validate",
        help="Validate manifests, rows, references, and byte sizes.",
    )
    validate.add_argument("root_path", nargs="?")
    validate.add_argument("--root", help="Staged evaluation repository.")
    validate.add_argument(
        "--allow-unauthorized",
        action="store_true",
        help="Validate local structure without requiring redistribution authorization.",
    )
    validate.add_argument(
        "--repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help="Hugging Face repository id used by generated metadata.",
    )
    _add_checksum_option(validate)

    upload = commands.add_parser(
        "upload",
        help="Validate and resumably upload a complete Hugging Face dataset repository.",
    )
    upload.add_argument("--root", required=True, help="Staged evaluation repository.")
    upload.add_argument("--repo-id", required=True, help="Hugging Face dataset repository id.")
    upload.add_argument(
        "--revision",
        default="main",
        help="Target Hugging Face branch or revision (default: main).",
    )
    _add_checksum_option(upload)
    upload.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repository as private if it does not exist.",
    )
    upload.add_argument(
        "--num-workers",
        type=int,
        help="Optional upload worker count.",
    )
    return parser


def _required(option: str | None, positional: str | None, label: str) -> str:
    if option and positional and option != positional:
        raise ValueError(f"conflicting {label} values")
    value = option or positional
    if not value:
        raise ValueError(f"{label} is required")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run one evaluation-data staging command."""

    args = create_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            manifest = _required(args.manifest, args.manifest_path, "source manifest")
            if args.write_locked_manifest:
                result = inventory_and_write_locked_manifest(
                    manifest,
                    args.write_locked_manifest,
                    tasks=args.tasks,
                    workers=args.workers,
                    checksums=args.checksums,
                )
            else:
                result = inventory_evaluation_sources(
                    manifest,
                    tasks=args.tasks,
                    workers=args.workers,
                    checksums=args.checksums,
                )
        elif args.command == "build":
            manifest = _required(args.manifest, args.manifest_path, "source manifest")
            output = _required(args.output, args.output_path, "output")
            result = build_evaluation_repository(
                manifest,
                output,
                copy_mode=args.copy_mode,
                overwrite=args.overwrite,
                workers=args.workers,
                checksums=args.checksums,
            )
        elif args.command == "merge":
            result = merge_evaluation_repository(
                args.source,
                args.target,
                copy_mode=args.copy_mode,
                repair_missing_asset_manifest=args.repair_missing_asset_manifest,
                workers=args.workers,
                replace_benchmarks=args.replace_benchmark,
            )
        elif args.command == "metadata":
            root = _required(args.root, args.root_path, "repository root")
            metadata_result = write_huggingface_metadata(
                root,
                repo_id=args.repo_id,
            )
            result = {
                **validate_staged_repository(
                    root,
                    checksums=args.checksums,
                    repo_id=args.repo_id,
                ),
                **metadata_result,
            }
        elif args.command == "validate":
            root = _required(args.root, args.root_path, "repository root")
            result = validate_staged_repository(
                root,
                require_redistribution_authorized=not args.allow_unauthorized,
                checksums=args.checksums,
                repo_id=args.repo_id,
            )
        elif args.command == "export-index":
            result = export_evaluation_index(
                args.root,
                args.output,
                overwrite=args.overwrite,
                checksums=args.checksums,
            )
        elif args.command == "export-public":
            result = export_public_evaluation_repository(
                args.root,
                args.output,
                copy_mode=args.copy_mode,
                overwrite=args.overwrite,
                workers=args.workers,
                checksums=args.checksums,
            )
        else:
            result = upload_evaluation_repository(
                args.root,
                args.repo_id,
                revision=args.revision,
                private=args.private,
                num_workers=args.num_workers,
                checksums=args.checksums,
            )
    except (FileExistsError, OSError, ValueError) as error:
        print(f"[FATAL] {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
