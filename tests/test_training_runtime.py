"""Guard the bundled trainer against packaging drift and legacy naming."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

import tomllib
import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = RELEASE_ROOT / "verl"

# Internal vocabulary that must not resurface in the public runtime.
FORBIDDEN_RUNTIME_TERMS = (
    "".join(("c", "p", "p", "o")),
    "".join(("g", "t", "p", "o")),
    "".join(("g", "t", "_")),
    "".join(("lu", "ff", "y")),
    "easy_r1",
)


def _runtime_sources() -> Iterator[Path]:
    return (path for path in sorted(RUNTIME_ROOT.rglob("*.py")))


def test_runtime_ships_with_the_release() -> None:
    assert (RUNTIME_ROOT / "trainer" / "main.py").is_file()
    assert (RUNTIME_ROOT / "trainer" / "ray_trainer.py").is_file()
    assert (RUNTIME_ROOT / "workers" / "fsdp_workers.py").is_file()

    pyproject = tomllib.loads((RELEASE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "verl*" in pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    manifest = (RELEASE_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include verl *.py" in manifest


def test_orarl_recipes_declare_every_oracle_stage() -> None:
    algorithm_config = ast.parse(
        (RUNTIME_ROOT / "trainer" / "config.py").read_text(encoding="utf-8")
    )
    declared = {
        statement.target.id
        for node in ast.walk(algorithm_config)
        if isinstance(node, ast.ClassDef) and node.name == "AlgorithmConfig"
        for statement in node.body
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
    }
    for name in ("orarl_4b.yaml", "orarl_9b.yaml"):
        payload = yaml.safe_load((RELEASE_ROOT / "configs" / name).read_text(encoding="utf-8"))
        algorithm = payload["algorithm"]
        assert algorithm["oracle_builder"].startswith("orarl.rewards:")
        assert payload["worker"]["reward"]["reward_function"].startswith("orarl.rewards:")
        for stage in (
            "oracle_injection",
            "directional_gain",
            "detached_oracle_advantage",
            "selection_prune_ratio",
            "post_selection_recenter",
        ):
            assert stage in algorithm
            assert stage in declared


def test_runtime_uses_the_public_vocabulary() -> None:
    offenders: list[str] = []
    for path in _runtime_sources():
        lowered = path.read_text(encoding="utf-8").casefold()
        for term in FORBIDDEN_RUNTIME_TERMS:
            if term in lowered:
                offenders.append(f"{path.relative_to(RELEASE_ROOT)}: {term}")
    assert not offenders, offenders


def test_trainer_entry_point_is_the_bundled_module() -> None:
    launcher = (RELEASE_ROOT / "orarl" / "cli" / "train.py").read_text(encoding="utf-8")
    assert '"verl.trainer.main"' in launcher

    main_source = (RUNTIME_ROOT / "trainer" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(main_source)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "main" in functions
