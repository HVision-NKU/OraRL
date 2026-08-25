#!/usr/bin/env python3
"""Fail closed when the OraRL public tree contains release hygiene hazards."""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_MAX_BYTES = 5 * 1024 * 1024
EVALUATION_JSONL_MAX_BYTES = 64 * 1024 * 1024
_METHOD_TERMS = (
    "".join(("c", "p", "p", "o")),
    "".join(("g", "t", "p", "o")),
    "".join(("build_", "g", "t")),
    "".join(("lu", "ff", "y")),
)
# Benchmark evaluators name upstream annotation fields after ground truth, so
# this prefix is only a tripwire outside the vendored evaluation runtime.
_LEGACY_ORACLE_TERMS = ("".join(("g", "t", "_")),)
_EXCLUDED_TERMS = _METHOD_TERMS + _LEGACY_ORACLE_TERMS
_IGNORED_LOCAL_DIRECTORIES = {".git", "checkpoint"}
_GENERATED_DIRECTORIES = {
    ".nox",
    ".tox",
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "cache",
    "checkpoints",
    "dist",
    "htmlcov",
    "logs",
    "outputs",
    "runs",
    "venv",
}
_GENERATED_FILES = {".coverage", "orarl-eval-summary.json"}
_ALLOWED_RELEASE_BINARIES = {
    Path("assets/orarl-data-scaling.gif"),
    Path("assets/orarl-hero.gif"),
    Path("assets/orarl-method.gif"),
    Path("assets/orarl-model-scaling.gif"),
    Path("assets/orarl-teaser.mp4"),
    Path("assets/paper-data-scaling.png"),
    Path("assets/paper-framework.png"),
    Path("assets/paper-model-scaling.png"),
    Path("assets/paper-results.png"),
    Path("orarl.pdf"),
}
_ARTIFACT_SUFFIXES = {
    ".arrow",
    ".avi",
    ".bin",
    ".ckpt",
    ".jsonl",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".parquet",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".so",
    ".webm",
}
_RELEASE_BINARY_FILES = {
    Path("assets/orarl-data-scaling.gif"),
    Path("assets/orarl-hero.gif"),
    Path("assets/orarl-method.gif"),
    Path("assets/orarl-model-scaling.gif"),
    Path("assets/orarl-teaser.mp4"),
    Path("assets/paper-data-scaling.png"),
    Path("assets/paper-framework.png"),
    Path("assets/paper-model-scaling.png"),
    Path("assets/paper-results.png"),
    Path("orarl.pdf"),
}
_PRIVATE_PATH_PATTERNS = (
    re.compile("/(?:" + "|".join(("mnt", "home", "Users")) + r")/[A-Za-z0-9._-]+(?:/[^\s'\"`]+)?"),
    re.compile(
        "/" + "data" + "/(?:" + "|".join(("home", "user", "users")) + r")/"
        r"[A-Za-z0-9._-]+(?:/[^\s'\"`]+)?"
    ),
    re.compile(
        "/(?:" + "|".join(("apd" + "cephfs" + r"[^/\s]*", "jizhi" + "cfs")) + r")(?:/[^\s'\"`]+)?"
    ),
    re.compile("/" + "root" + r"(?:/[^\s'\"`]+)?"),
)
_SECRET_PATTERNS = (
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"\b\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
    ),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"://[^/\s:@]+:[^@\s/]+@"),
)
_PRIVATE_KEY_MARKER = "".join(("-" * 5, "BEGIN ", "PRIVATE", " KEY", "-" * 5))


@dataclass(frozen=True, slots=True)
class Finding:
    """One release gate failure."""

    path: Path
    reason: str


def _line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _text_findings(
    path: Path,
    text: str,
    *,
    check_excluded_terms: bool = True,
    allow_ground_truth_names: bool = False,
) -> Iterable[Finding]:
    if check_excluded_terms:
        lowered = text.casefold()
        terms = _METHOD_TERMS if allow_ground_truth_names else _EXCLUDED_TERMS
        for term in terms:
            index = lowered.find(term)
            if index >= 0:
                yield Finding(
                    path,
                    f"excluded method term at line {_line_number(text, index)}",
                )

    for pattern in _PRIVATE_PATH_PATTERNS:
        match = pattern.search(text)
        if match:
            yield Finding(
                path,
                f"private absolute path at line {_line_number(text, match.start())}",
            )

    if _PRIVATE_KEY_MARKER in text:
        yield Finding(path, "private key material")
    for pattern in _SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            yield Finding(
                path,
                f"credential-like value at line {_line_number(text, match.start())}",
            )


def _ignored_generated_directory(root: Path, generated_root: Path) -> bool:
    """Return whether root-level gitignore rules exclude generated state."""

    ignore_file = root / ".gitignore"
    if not ignore_file.is_file():
        return False
    try:
        patterns = {
            line.strip()
            for line in ignore_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "!"))
        }
    except OSError:
        return False
    name = generated_root.name
    if name.endswith(".egg-info") and "*.egg-info/" in patterns:
        return True
    return f"{name}/" in patterns or f"/{name}/" in patterns


def check_release(root: Path, max_bytes: int = DEFAULT_MAX_BYTES) -> list[Finding]:
    """Return every release hygiene problem under ``root``."""

    root = root.expanduser().resolve()
    findings: list[Finding] = []
    reported_generated: set[Path] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        lowered_parts = tuple(part.casefold() for part in relative.parts)
        release_binary = relative in _RELEASE_BINARY_FILES
        release_eval_data = lowered_parts[:2] == ("data", "eval")
        release_eval_runtime = lowered_parts[:2] == ("eval", "task")

        if any(part in _IGNORED_LOCAL_DIRECTORIES for part in lowered_parts):
            continue

        generated_index = next(
            (
                index
                for index, part in enumerate(lowered_parts)
                if part in _GENERATED_DIRECTORIES or part.endswith(".egg-info")
            ),
            None,
        )
        if generated_index is not None:
            generated_root = Path(*relative.parts[: generated_index + 1])
            if _ignored_generated_directory(root, generated_root):
                continue
            if generated_root not in reported_generated:
                findings.append(Finding(generated_root, "generated directory"))
                reported_generated.add(generated_root)
            continue

        path_terms = _METHOD_TERMS if release_eval_runtime else _EXCLUDED_TERMS
        if any(term in relative_text.casefold() for term in path_terms):
            findings.append(Finding(relative, "excluded method term in path"))

        if path.is_symlink():
            target = os.readlink(path)
            if not path.exists():
                findings.append(Finding(relative, "broken symbolic link"))
            else:
                resolved_target = path.resolve()
                try:
                    resolved_target.relative_to(root)
                except ValueError:
                    findings.append(Finding(relative, "symbolic link escapes release tree"))
            for pattern in _PRIVATE_PATH_PATTERNS:
                if pattern.search(target):
                    findings.append(Finding(relative, "symbolic link contains a private path"))
                    break
            continue

        if path.is_dir():
            if (
                lowered_parts
                and lowered_parts[0] == "data"
                and len(lowered_parts) > 1
                and not release_eval_data
            ):
                findings.append(Finding(relative, "generated root data directory"))
            continue
        if not path.is_file():
            findings.append(Finding(relative, "unsupported filesystem entry"))
            continue
        if relative in _ALLOWED_RELEASE_BINARIES:
            continue

        if path.name.casefold() in _GENERATED_FILES:
            findings.append(Finding(relative, "generated output file"))
        if (
            path.suffix.casefold() in _ARTIFACT_SUFFIXES
            and not release_binary
            and not (
                release_eval_data
                and path.suffix.casefold() == ".jsonl"
            )
        ):
            findings.append(Finding(relative, "generated or binary artifact"))

        size = path.stat().st_size
        effective_max_bytes = (
            max(max_bytes, EVALUATION_JSONL_MAX_BYTES)
            if release_eval_data and path.suffix.casefold() == ".jsonl"
            else max_bytes
        )
        if size > effective_max_bytes and not release_binary:
            findings.append(
                Finding(
                    relative,
                    f"file is too large ({size} bytes; limit {effective_max_bytes})",
                )
            )
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if not release_binary:
                findings.append(Finding(relative, "non-text file"))
            continue
        findings.extend(
            _text_findings(
                relative,
                text,
                check_excluded_terms=not release_eval_data,
                allow_ground_truth_names=release_eval_runtime,
            )
        )
    return findings


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1]),
        help="Release tree to inspect (default: OraRL source root).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Maximum permitted size for one source file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = create_parser().parse_args(argv)
    root = Path(namespace.root).expanduser()
    if not root.is_dir():
        print(f"ERROR: release root is not a directory: {root}")
        return 2
    if namespace.max_bytes <= 0:
        print("ERROR: --max-bytes must be positive")
        return 2

    findings = check_release(root, namespace.max_bytes)
    if findings:
        for finding in findings:
            print(f"ERROR: {finding.path}: {finding.reason}")
        print(f"release hygiene failed with {len(findings)} finding(s)")
        return 1
    file_count = sum(1 for path in root.rglob("*") if path.is_file())
    print(f"release hygiene passed ({file_count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
