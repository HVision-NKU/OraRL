"""Locate OraRL configuration, documentation, and launcher resources."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path, PurePosixPath

_RESOURCE_KINDS = frozenset({"configs", "docs", "scripts"})


def _safe_name(name: str) -> str:
    candidate = str(name).strip()
    if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:
        raise ValueError(f"resource name must be a file name, got {name!r}")
    return candidate


def _source_resource(kind: str, name: str) -> Path | None:
    candidate = Path(__file__).resolve().parent.parent / kind / name
    return candidate if candidate.is_file() else None


def _installed_resource(kind: str, name: str) -> Path | None:
    try:
        package_distribution = distribution("orarl")
    except PackageNotFoundError:
        return None

    suffix = ("share", "orarl", kind, name)
    for entry in package_distribution.files or ():
        parts = PurePosixPath(str(entry)).parts
        if len(parts) >= len(suffix) and tuple(parts[-len(suffix) :]) == suffix:
            candidate = Path(package_distribution.locate_file(entry)).resolve()
            if candidate.is_file():
                return candidate
    return None


def resource_path(kind: str, name: str) -> Path:
    """Return one packaged resource from a source or wheel installation."""

    normalized_kind = str(kind).strip().casefold()
    if normalized_kind not in _RESOURCE_KINDS:
        choices = ", ".join(sorted(_RESOURCE_KINDS))
        raise ValueError(f"resource kind must be one of {{{choices}}}, got {kind!r}")
    normalized_name = _safe_name(name)
    candidate = _source_resource(normalized_kind, normalized_name)
    if candidate is None:
        candidate = _installed_resource(normalized_kind, normalized_name)
    if candidate is None:
        raise FileNotFoundError(
            f"OraRL {normalized_kind} resource was not found: {normalized_name}"
        )
    return candidate


def config_path(name: str) -> Path:
    """Return an included public YAML configuration."""

    return resource_path("configs", name)


def documentation_path(name: str) -> Path:
    """Return an included release document."""

    return resource_path("docs", name)


def script_path(name: str) -> Path:
    """Return an included shell or Python launcher."""

    return resource_path("scripts", name)


def available_configs() -> tuple[str, ...]:
    """Return the stable names of the four public training recipes."""

    return (
        "grpo_4b.yaml",
        "grpo_9b.yaml",
        "orarl_4b.yaml",
        "orarl_9b.yaml",
    )
