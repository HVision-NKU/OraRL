from __future__ import annotations

from pathlib import Path

import pytest

from orarl import (
    available_configs,
    config_path,
    documentation_path,
    script_path,
)

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_source_resources_are_discoverable() -> None:
    assert available_configs() == (
        "grpo_4b.yaml",
        "grpo_9b.yaml",
        "orarl_4b.yaml",
        "orarl_9b.yaml",
    )
    assert config_path("orarl_4b.yaml") == (RELEASE_ROOT / "configs" / "orarl_4b.yaml")
    assert documentation_path("environment.md") == (RELEASE_ROOT / "docs" / "environment.md")
    assert documentation_path("training.md") == (RELEASE_ROOT / "docs" / "training.md")
    assert documentation_path("evaluation.md") == (RELEASE_ROOT / "docs" / "evaluation.md")
    assert script_path("train_orarl.sh") == (RELEASE_ROOT / "scripts" / "train_orarl.sh")


@pytest.mark.parametrize("name", ("", ".", "..", "../orarl_4b.yaml"))
def test_resource_names_reject_path_traversal(name: str) -> None:
    with pytest.raises(ValueError, match="file name"):
        config_path(name)


def test_unknown_resource_has_clear_error() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        config_path("unknown.yaml")
