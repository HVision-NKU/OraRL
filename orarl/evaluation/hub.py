"""Explicit, validated Hugging Face publication for evaluation repositories."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .staging import validate_staged_repository


class UploadError(ValueError):
    """Raised when an evaluation repository cannot be uploaded safely."""


def upload_evaluation_repository(
    repository_root: str | os.PathLike[str],
    repo_id: str,
    *,
    revision: str = "main",
    private: bool = False,
    num_workers: int | None = None,
    checksums: bool = False,
) -> dict[str, Any]:
    """Validate and resumably upload a complete dataset repository.

    Authentication is intentionally delegated to the Hugging Face environment
    or persisted CLI login. This API has no token parameter.
    """

    root = Path(repository_root).expanduser().resolve()
    validated = validate_staged_repository(root, checksums=checksums)
    if not isinstance(repo_id, str) or not repo_id.strip() or repo_id != repo_id.strip():
        raise UploadError("repo_id must be a nonempty Hugging Face repository id")
    if any(character.isspace() for character in repo_id):
        raise UploadError("repo_id must not contain whitespace")
    if not isinstance(revision, str) or not revision.strip() or revision != revision.strip():
        raise UploadError("revision must be a nonempty branch or revision name")
    if not isinstance(private, bool):
        raise UploadError("private must be a boolean")
    if num_workers is not None and (
        isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers <= 0
    ):
        raise UploadError("num_workers must be a positive integer")

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise UploadError(
            "upload requires huggingface_hub; install the 'hf' optional dependency"
        ) from error

    arguments: dict[str, Any] = {
        "repo_id": repo_id,
        "folder_path": str(root),
        "repo_type": "dataset",
        "revision": revision,
        "private": private,
    }
    if num_workers is not None:
        arguments["num_workers"] = num_workers

    try:
        HfApi().upload_large_folder(**arguments)
    except Exception as error:
        raise UploadError(f"Hugging Face upload failed ({type(error).__name__})") from error
    return {
        **validated,
        "repo_id": repo_id,
        "revision": revision,
        "private": private,
    }


upload_repository = upload_evaluation_repository


__all__ = ["UploadError", "upload_evaluation_repository", "upload_repository"]
