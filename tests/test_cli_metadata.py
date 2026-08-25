from __future__ import annotations

from pathlib import Path

import tomllib

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_install_metadata_and_console_scripts() -> None:
    with (RELEASE_ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)

    project = metadata["project"]
    assert project["name"] == "orarl"
    assert project["requires-python"] == ">=3.10"
    # The trainer ships inside this distribution, so no external runtime pin.
    assert not any(dependency.startswith("verl") for dependency in project["dependencies"])
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE", "NOTICE"]
    assert project["optional-dependencies"]["hf"] == ["huggingface_hub"]
    assert project["scripts"] == {
        "orarl-train": "orarl.cli.train:main",
        "orarl-eval": "orarl.cli.evaluate:main",
        "orarl-eval-data": "orarl.cli.eval_data:main",
        "orarl-prepare": "orarl.cli.prepare:main",
    }
    data_files = metadata["tool"]["setuptools"]["data-files"]
    assert data_files["share/orarl"] == ["environment.yml", "requirements-cu129.txt"]
    assert data_files["share/orarl/configs"] == ["configs/*.yaml"]
    assert data_files["share/orarl/docs"] == ["docs/*.md"]
    assert data_files["share/orarl/scripts"] == ["scripts/*.py", "scripts/*.sh"]


def test_install_ships_every_evaluator() -> None:
    with (RELEASE_ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)

    installed: set[Path] = set()
    for target, patterns in metadata["tool"]["setuptools"]["data-files"].items():
        if not target.startswith("share/orarl/eval"):
            continue
        for pattern in patterns:
            installed.update(RELEASE_ROOT.glob(pattern))

    # Importing an evaluator during the suite drops bytecode next to the source.
    shipped = {
        path
        for path in (RELEASE_ROOT / "eval").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }

    assert shipped
    assert sorted(shipped - installed) == []


def test_apache_attribution_files_are_present() -> None:
    license_text = (RELEASE_ROOT / "LICENSE").read_text(encoding="utf-8")
    notice_text = (RELEASE_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert "ByteDance Ltd." in notice_text
