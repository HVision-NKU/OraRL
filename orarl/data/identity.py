"""Stable prompt and media identities used for sampling and leakage checks."""

from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .schema import media_paths, normalize_text


def normalized_media_anchor(path: str) -> str:
    """Return a case-normalized lexical identity for one local media path."""

    normalized = os.path.normpath(str(path)).replace("\\", "/")
    return unicodedata.normalize("NFKC", normalized).casefold()


def media_anchors(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return exact normalized media anchors for cap enforcement."""

    return tuple(sorted({normalized_media_anchor(path) for path in media_paths(record)}))


def media_leakage_tokens(record: Mapping[str, Any]) -> frozenset[str]:
    """Return exact and basename tokens for fail-closed media comparisons."""

    tokens: set[str] = set()
    for anchor in media_anchors(record):
        path = Path(anchor)
        name = normalize_text(path.name)
        stem = normalize_text(path.stem)
        tokens.add(f"path:{anchor}")
        if name:
            tokens.add(f"name:{name}")
        if stem:
            tokens.add(f"stem:{stem}")
    return frozenset(tokens)


def prompt_identity(record: Mapping[str, Any]) -> str:
    """Hash what is asked and its media, intentionally excluding the answer."""

    payload = {
        "media": media_anchors(record),
        "problem": normalize_text(record.get("problem")),
        "problem_type": normalize_text(record.get("problem_type")),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_rank(seed: int, namespace: str, identity: str) -> int:
    """Create a process-independent deterministic rank."""

    payload = f"{seed}|{namespace}|{identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")
