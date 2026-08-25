"""Validated, temp-file-free launcher for OraRL training."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import yaml

from orarl.resources import config_path

_PROTECTED_OVERRIDES = {
    "config",
    "data.train_files",
    "data.val_files",
    "trainer.n_gpus_per_node",
    "trainer.nnodes",
    "trainer.save_checkpoint_path",
    "worker.actor.model.model_path",
    "worker.actor.model.tokenizer_path",
}
_SENSITIVE_OVERRIDE = re.compile(
    r"(?:^|[._-])(?:api[_-]?key|auth[_-]?token|access[_-]?token|"
    r"credential|password|private[_-]?key|secret)(?:$|[._-])",
    flags=re.IGNORECASE,
)
_URI_USERINFO = re.compile(r"://[^/\s:@]+:[^@\s/]+@")
_OC_ENV = re.compile(r"^\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,(.*))?\}$")


class CliError(ValueError):
    """Raised when a launch request is incomplete or unsafe."""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orarl-train",
        description=(
            "Validate an OraRL config and launch the local-compatible verl runtime. "
            "The default is a non-mutating dry run."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to a complete YAML config, or one of the bundled names "
            "(for example, orarl_4b.yaml)."
        ),
    )
    parser.add_argument("--model", required=True, help="Local model or checkpoint path.")
    parser.add_argument("--train-data", required=True, help="Training JSONL path.")
    parser.add_argument("--val-data", required=True, help="Validation/canary JSONL path.")
    parser.add_argument("--output", required=True, help="Checkpoint output directory.")
    parser.add_argument("--nodes", type=int, default=1, help="Number of training nodes.")
    parser.add_argument(
        "--gpus",
        "--gpus-per-node",
        dest="gpus_per_node",
        type=int,
        default=8,
        help="GPUs allocated on each node.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for the trainer (default: this interpreter).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional OmegaConf override; repeat as needed.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Validate and print the command without executing it (default).",
    )
    mode.add_argument(
        "--run",
        dest="dry_run",
        action="store_false",
        help="Execute the validated command.",
    )
    parser.set_defaults(dry_run=True)
    return parser


def _existing_path(value: str, label: str, *, kind: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise CliError(f"{label} does not exist: {path}")
    if kind == "file" and not path.is_file():
        raise CliError(f"{label} must be a file: {path}")
    if kind == "directory" and not path.is_dir():
        raise CliError(f"{label} must be a directory: {path}")
    return path


def _config_file(value: str) -> Path:
    path = Path(value).expanduser()
    if path.exists() or path.name != value:
        return _existing_path(value, "config", kind="file")
    try:
        return config_path(value)
    except (FileNotFoundError, ValueError) as error:
        raise CliError(f"config does not exist and is not a bundled recipe: {value}") from error


def _output_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise CliError(f"output must be a directory path: {path}")
    ancestor = path
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise CliError(f"output has no usable parent directory: {path}")
    if not os.access(ancestor, os.W_OK):
        raise CliError(f"output parent is not writable: {ancestor}")
    return path


def _python_executable(value: str) -> str:
    candidate = shutil.which(value)
    if candidate is None:
        path = Path(value).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            candidate = str(path.resolve())
    if candidate is None:
        raise CliError(f"Python executable is unavailable: {value}")
    return candidate


def _load_config(config: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise CliError(f"cannot load config {config}: {error}") from error
    if not isinstance(payload, dict):
        raise CliError(f"config must contain a YAML mapping: {config}")
    return payload


def _algorithm_name(payload: Mapping[str, Any]) -> str:
    algorithm = payload.get("algorithm")
    if not isinstance(algorithm, dict):
        raise CliError("config must define an algorithm mapping")
    name = str(algorithm.get("name", "")).strip().casefold()
    if name not in {"grpo", "orarl"}:
        raise CliError("algorithm.name must be either 'grpo' or 'orarl'")
    return name


def _nested_value(payload: Mapping[str, Any], key: str) -> Any:
    current: Any = payload
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise CliError(f"config must define {key}")
        current = current[part]
    return current


def _resolved_scalar(value: Any, key: str) -> Any:
    if not isinstance(value, str):
        return value
    match = _OC_ENV.fullmatch(value.strip())
    if match is None:
        return value
    environment_name, fallback = match.groups()
    resolved = os.environ.get(environment_name, fallback)
    if resolved is None:
        raise CliError(f"{key} requires environment variable {environment_name}")
    return resolved


def _config_number(
    payload: Mapping[str, Any],
    overrides: Mapping[str, str],
    key: str,
    cast: type[int] | type[float],
) -> int | float:
    raw_value: Any = overrides[key] if key in overrides else _nested_value(payload, key)
    raw_value = _resolved_scalar(raw_value, key)
    if isinstance(raw_value, bool):
        raise CliError(f"{key} must be a {cast.__name__}")
    try:
        return cast(raw_value)
    except (TypeError, ValueError) as error:
        raise CliError(f"{key} must resolve to a {cast.__name__}, got {raw_value!r}") from error


def _validate_orarl_batch_layout(
    payload: Mapping[str, Any],
    overrides: Mapping[str, str],
    *,
    nodes: int,
    gpus_per_node: int,
) -> None:
    rollout_batch_size = int(
        _config_number(
            payload,
            overrides,
            "data.rollout_batch_size",
            int,
        )
    )
    n_rollouts = int(_config_number(payload, overrides, "worker.rollout.n", int))
    prune_ratio = float(
        _config_number(
            payload,
            overrides,
            "algorithm.selection_prune_ratio",
            float,
        )
    )
    positive_quota = int(
        _config_number(
            payload,
            overrides,
            "algorithm.selection_positive_quota",
            int,
        )
    )
    negative_quota = int(
        _config_number(
            payload,
            overrides,
            "algorithm.selection_negative_quota",
            int,
        )
    )

    if rollout_batch_size <= 0:
        raise CliError("data.rollout_batch_size must be positive")
    if n_rollouts <= 1:
        raise CliError("OraRL requires worker.rollout.n > 1")
    if not 0.0 < prune_ratio < 1.0:
        raise CliError("algorithm.selection_prune_ratio must be in (0, 1)")

    keep_rows = max(1, int(n_rollouts * (1.0 - prune_ratio)))
    if positive_quota < 0 or negative_quota < 0:
        raise CliError("OraRL selection quotas must be non-negative")
    if positive_quota + negative_quota + 1 != keep_rows:
        raise CliError(
            "OraRL selection quotas must fill the keep budget: "
            f"positive({positive_quota}) + negative({negative_quota}) + "
            f"oracle(1) != keep_rows({keep_rows})"
        )

    world_size = nodes * gpus_per_node
    selected_batch = rollout_batch_size * keep_rows
    if selected_batch % world_size:
        raise CliError(
            "OraRL selected batch must divide the actor world size: "
            f"{selected_batch} rows for world_size={world_size}"
        )
    preselection_batch = rollout_batch_size * (n_rollouts + 1)
    if preselection_batch % world_size:
        raise CliError(
            "OraRL append-oracle batch must divide the actor world size: "
            f"{preselection_batch} rows for world_size={world_size}"
        )


def _validated_overrides(values: Sequence[str]) -> list[str]:
    overrides: list[str] = []
    for value in values:
        key, separator, raw_value = value.partition("=")
        key = key.strip()
        if not separator or not key or not raw_value.strip():
            raise CliError(f"--set expects KEY=VALUE, got: {value!r}")
        if key in _PROTECTED_OVERRIDES:
            raise CliError(f"use the dedicated option instead of --set for {key}")
        if key == "algorithm" or key.startswith("algorithm."):
            raise CliError("algorithm settings must be declared in the public config")
        if any(character.isspace() for character in key):
            raise CliError(f"override keys cannot contain whitespace: {key!r}")
        if _SENSITIVE_OVERRIDE.search(key) or _URI_USERINFO.search(raw_value):
            raise CliError("credential-bearing overrides are not accepted")
        overrides.append(f"{key}={raw_value}")
    return overrides


def _override_mapping(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, _, raw_value = value.partition("=")
        result[key] = raw_value
    return result


def build_command(namespace: argparse.Namespace) -> tuple[list[str], str]:
    """Validate ``namespace`` and return the trainer command and method name."""

    config = _config_file(namespace.config)
    model = _existing_path(namespace.model, "model", kind="directory")
    train_data = _existing_path(namespace.train_data, "training data", kind="file")
    val_data = _existing_path(namespace.val_data, "validation data", kind="file")
    output = _output_path(namespace.output)
    python = _python_executable(namespace.python)
    payload = _load_config(config)
    method = _algorithm_name(payload)
    validated_overrides = _validated_overrides(namespace.overrides)

    if namespace.nodes <= 0:
        raise CliError("--nodes must be positive")
    if namespace.gpus_per_node <= 0:
        raise CliError("--gpus-per-node must be positive")
    if method == "orarl":
        _validate_orarl_batch_layout(
            payload,
            _override_mapping(validated_overrides),
            nodes=namespace.nodes,
            gpus_per_node=namespace.gpus_per_node,
        )

    command = [
        python,
        "-m",
        "verl.trainer.main",
        f"config={config}",
        *validated_overrides,
        f"worker.actor.model.model_path={model}",
        f"worker.actor.model.tokenizer_path={model}",
        f"data.train_files={train_data}",
        f"data.val_files={val_data}",
        f"trainer.save_checkpoint_path={output}",
        f"trainer.nnodes={namespace.nodes}",
        f"trainer.n_gpus_per_node={namespace.gpus_per_node}",
    ]
    return command, method


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    namespace = parser.parse_args(argv)
    try:
        command, method = build_command(namespace)
    except CliError as error:
        parser.error(str(error))

    print(f"method={method} mode={'dry-run' if namespace.dry_run else 'run'}")
    print(shlex.join(command))
    if namespace.dry_run:
        return 0
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
