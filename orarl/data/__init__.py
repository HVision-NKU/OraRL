"""Portable OraRL data preparation utilities."""

from typing import Any

from .identity import media_anchors, media_leakage_tokens, prompt_identity, stable_rank
from .schema import (
    CANONICAL_FIELDS,
    SchemaError,
    canonicalize_record,
    media_paths,
    normalize_text,
    validate_record,
    validation_errors,
)

__all__ = [
    "CANONICAL_FIELDS",
    "SchemaError",
    "build_dataset",
    "canonicalize_record",
    "media_anchors",
    "media_leakage_tokens",
    "media_paths",
    "normalize_text",
    "prompt_identity",
    "stable_rank",
    "validate_record",
    "validation_errors",
]


def build_dataset(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load and call the builder without eagerly importing its CLI module."""

    from .build import build_dataset as _build_dataset

    return _build_dataset(*args, **kwargs)
