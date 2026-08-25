from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT))

from orarl.cli import train  # noqa: E402


def _inputs(tmp_path: Path, method: str = "orarl") -> list[str]:
    config = tmp_path / "config.yaml"
    if method == "orarl":
        config_text = (
            "data:\n"
            "  rollout_batch_size: 64\n"
            "algorithm:\n"
            "  name: orarl\n"
            "  selection_prune_ratio: 0.5\n"
            "  selection_positive_quota: 1\n"
            "  selection_negative_quota: 2\n"
            "worker:\n"
            "  rollout:\n"
            "    n: 8\n"
        )
    else:
        config_text = f"algorithm:\n  name: {method}\n"
    config.write_text(config_text, encoding="utf-8")
    model = tmp_path / "model"
    model.mkdir()
    train_data = tmp_path / "train.jsonl"
    train_data.write_text("{}\n", encoding="utf-8")
    val_data = tmp_path / "val.jsonl"
    val_data.write_text("{}\n", encoding="utf-8")
    return [
        "--config",
        str(config),
        "--model",
        str(model),
        "--train-data",
        str(train_data),
        "--val-data",
        str(val_data),
        "--output",
        str(tmp_path / "output"),
        "--nodes",
        "2",
        "--gpus",
        "4",
        "--python",
        sys.executable,
    ]


def test_train_command_uses_full_config_and_explicit_overrides(tmp_path: Path) -> None:
    parser = train.create_parser()
    namespace = parser.parse_args(_inputs(tmp_path))
    command, method = train.build_command(namespace)

    assert method == "orarl"
    assert namespace.dry_run is True
    assert command[:3] == [sys.executable, "-m", "verl.trainer.main"]
    assert f"config={(tmp_path / 'config.yaml').resolve()}" in command
    assert f"worker.actor.model.model_path={(tmp_path / 'model').resolve()}" in command
    assert f"data.train_files={(tmp_path / 'train.jsonl').resolve()}" in command
    assert f"data.val_files={(tmp_path / 'val.jsonl').resolve()}" in command
    assert f"trainer.save_checkpoint_path={(tmp_path / 'output').resolve()}" in command
    assert "trainer.nnodes=2" in command
    assert "trainer.n_gpus_per_node=4" in command


def test_train_resolves_bundled_config_name(tmp_path: Path) -> None:
    arguments = _inputs(tmp_path)
    arguments[arguments.index("--config") + 1] = "orarl_4b.yaml"
    namespace = train.create_parser().parse_args(arguments)

    command, method = train.build_command(namespace)

    assert method == "orarl"
    assert f"config={train.config_path('orarl_4b.yaml')}" in command


def test_orarl_dry_run_rejects_incompatible_world_size(tmp_path: Path) -> None:
    arguments = _inputs(tmp_path)
    arguments[arguments.index("--nodes") + 1] = "1"
    arguments[arguments.index("--gpus") + 1] = "6"
    namespace = train.create_parser().parse_args(arguments)

    with pytest.raises(train.CliError, match="selected batch must divide"):
        train.build_command(namespace)


def test_orarl_dry_run_validates_batch_override(tmp_path: Path) -> None:
    arguments = [
        *_inputs(tmp_path),
        "--set",
        "data.rollout_batch_size=66",
    ]
    namespace = train.create_parser().parse_args(arguments)

    with pytest.raises(train.CliError, match="append-oracle batch must divide"):
        train.build_command(namespace)


def test_train_rejects_missing_required_path(tmp_path: Path) -> None:
    arguments = _inputs(tmp_path)
    arguments[arguments.index("--model") + 1] = str(tmp_path / "missing")
    namespace = train.create_parser().parse_args(arguments)
    with pytest.raises(train.CliError, match="model does not exist"):
        train.build_command(namespace)


def test_protected_overrides_use_dedicated_options(tmp_path: Path) -> None:
    arguments = _inputs(tmp_path) + ["--set", "data.train_files=other.jsonl"]
    namespace = train.create_parser().parse_args(arguments)
    with pytest.raises(train.CliError, match="dedicated option"):
        train.build_command(namespace)


def test_public_configs_expose_final_recipe() -> None:
    grpo_4b = yaml.safe_load((RELEASE_ROOT / "configs" / "grpo_4b.yaml").read_text())
    grpo_9b = yaml.safe_load((RELEASE_ROOT / "configs" / "grpo_9b.yaml").read_text())
    recipe_4b = yaml.safe_load((RELEASE_ROOT / "configs" / "orarl_4b.yaml").read_text())
    recipe_9b = yaml.safe_load((RELEASE_ROOT / "configs" / "orarl_9b.yaml").read_text())

    for config in (grpo_4b, grpo_9b):
        assert config["algorithm"]["name"] == "grpo"
        assert config["data"]["group_by_task"] is True
        assert config["data"]["rollout_batch_size"] == ("${oc.env:ORARL_ROLLOUT_BATCH_SIZE,64}")
        assert config["worker"]["actor"]["global_batch_size"] == (
            "${oc.env:ORARL_GLOBAL_BATCH_SIZE,64}"
        )
        assert config["worker"]["rollout"]["n"] == 8
        assert config["trainer"]["logger"] == ["console"]
    assert grpo_9b["worker"]["actor"]["micro_batch_size_per_device_for_update"] == (
        "${oc.env:ORARL_UPDATE_MICRO_BATCH,1}"
    )
    assert grpo_9b["worker"]["actor"]["optim"]["lr"] == ("${oc.env:ORARL_LEARNING_RATE,1.0e-6}")
    for config in (recipe_4b, recipe_9b):
        algorithm = config["algorithm"]
        assert algorithm["name"] == "orarl"
        assert algorithm["oracle_injection_mode"] == "append"
        assert algorithm["scale_rewards"] is False
        assert algorithm["directional_gain"] is True
        assert algorithm["directional_gain_gamma"] == 0.25
        assert algorithm["directional_gain_positive_only"] is True
        assert algorithm["directional_gain_recenter"] is True
        assert algorithm["detached_oracle_advantage_scale"] == 2.0
        assert algorithm["detached_oracle_use_directional_gain"] is False
        assert algorithm["detached_oracle_match_best_ratio"] == 1.2
        assert algorithm["detached_oracle_match_best_min"] == 0.05
        assert algorithm["detached_oracle_match_best_max"] == 1.0
        assert algorithm["oracle_reward_gate_beta"] == 2.0
        assert algorithm["selection_prune_ratio"] == 0.5
        assert algorithm["selection_positive_quota"] == 1
        assert algorithm["selection_negative_quota"] == 2
        assert algorithm["selection_strict_sign_balance"] is True
        assert algorithm["post_selection_recenter"] is True
        assert algorithm["post_selection_rms_match"] is True
        assert algorithm["post_selection_rms_min_scale"] == 0.25
        assert algorithm["disable_kl"] is True
        assert config["data"]["group_by_task"] is True
        assert config["data"]["rollout_batch_size"] == ("${oc.env:ORARL_ROLLOUT_BATCH_SIZE,64}")
        assert config["worker"]["actor"]["global_batch_size"] == (
            "${oc.env:ORARL_GLOBAL_BATCH_SIZE,64}"
        )
        assert config["worker"]["rollout"]["n"] == 8
        assert config["trainer"]["logger"] == ["console"]
