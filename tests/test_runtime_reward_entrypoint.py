from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import fields, is_dataclass
from pathlib import Path

import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = RELEASE_ROOT / "verl"
sys.path.insert(0, str(RELEASE_ROOT))


def _package(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    return module


def _load_module(
    name: str,
    path: Path,
    monkeypatch,
) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_runtime_loads_public_reward_module(monkeypatch) -> None:
    """The public module:function config must survive runtime initialization."""

    verl_root = RUNTIME_ROOT
    reward_root = verl_root / "workers" / "reward"
    monkeypatch.setitem(sys.modules, "verl", _package("verl", verl_root))
    monkeypatch.setitem(
        sys.modules,
        "verl.workers",
        _package("verl.workers", verl_root / "workers"),
    )
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.reward",
        _package("verl.workers.reward", reward_root),
    )

    functional = types.ModuleType("verl.utils.py_functional")
    functional.get_abs_path = lambda value, **_: str(Path(value).resolve())
    monkeypatch.setitem(
        sys.modules,
        "verl.utils",
        _package("verl.utils", verl_root / "utils"),
    )
    monkeypatch.setitem(sys.modules, "verl.utils.py_functional", functional)

    protocol = types.ModuleType("verl.protocol")
    protocol.DataProto = object
    monkeypatch.setitem(sys.modules, "verl.protocol", protocol)

    torch = types.ModuleType("torch")
    torch.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", torch)

    transformers = types.ModuleType("transformers")
    transformers.PreTrainedTokenizer = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    config_module = _load_module(
        "verl.workers.reward.config",
        reward_root / "config.py",
        monkeypatch,
    )
    function_module = _load_module(
        "verl.workers.reward.function",
        reward_root / "function.py",
        monkeypatch,
    )

    config = config_module.RewardConfig(
        reward_function="orarl.rewards:compute_score",
    )
    config.post_init()
    assert config.reward_function == "orarl.rewards"
    assert config.reward_function_name == "compute_score"
    assert config.reward_function_is_module is True

    manager = function_module.AutoRewardManager(config, tokenizer=object())
    scores = manager.reward_fn(
        [
            {
                "problem_type": "video_qa_mc",
                "ground_truth": "<answer>A</answer>",
                "response": "<answer>A</answer>",
            }
        ]
    )
    assert scores[0]["overall"] == 1.0


def _assert_config_keys(config: object, payload: dict, path: str = "") -> None:
    known = {item.name for item in fields(config)}
    unknown = sorted(set(payload) - known)
    assert not unknown, f"{path or '<root>'} has unknown fields: {unknown}"
    for key, value in payload.items():
        child = getattr(config, key)
        if isinstance(value, dict) and is_dataclass(child):
            _assert_config_keys(child, value, f"{path}.{key}".strip("."))


def test_public_recipes_match_runtime_dataclasses(monkeypatch) -> None:
    """Every public YAML field must be accepted by the bundled runtime."""

    verl_root = RUNTIME_ROOT
    reward_root = verl_root / "workers" / "reward"
    monkeypatch.setitem(sys.modules, "verl", _package("verl", verl_root))
    monkeypatch.setitem(
        sys.modules,
        "verl.trainer",
        _package("verl.trainer", verl_root / "trainer"),
    )
    monkeypatch.setitem(
        sys.modules,
        "verl.workers",
        _package("verl.workers", verl_root / "workers"),
    )
    monkeypatch.setitem(
        sys.modules,
        "verl.workers.reward",
        _package("verl.workers.reward", reward_root),
    )
    monkeypatch.setitem(
        sys.modules,
        "verl.utils",
        _package("verl.utils", verl_root / "utils"),
    )

    functional = types.ModuleType("verl.utils.py_functional")
    functional.get_abs_path = lambda value, **_: str(Path(value).resolve())
    monkeypatch.setitem(sys.modules, "verl.utils.py_functional", functional)

    multimodal_contract = types.ModuleType("verl.utils.multimodal_contract")
    multimodal_contract.normalize_video_source_mode = lambda value, **_: value
    monkeypatch.setitem(
        sys.modules,
        "verl.utils.multimodal_contract",
        multimodal_contract,
    )

    reward_config_module = _load_module(
        "verl.workers.reward.config",
        reward_root / "config.py",
        monkeypatch,
    )
    reward_package = sys.modules["verl.workers.reward"]
    reward_package.RewardConfig = reward_config_module.RewardConfig

    trainer_config_module = _load_module(
        "verl.trainer.config",
        verl_root / "trainer" / "config.py",
        monkeypatch,
    )
    runtime_config = trainer_config_module.PPOConfig()
    for name in (
        "grpo_4b.yaml",
        "grpo_9b.yaml",
        "orarl_4b.yaml",
        "orarl_9b.yaml",
    ):
        payload = yaml.safe_load((RELEASE_ROOT / "configs" / name).read_text(encoding="utf-8"))
        _assert_config_keys(runtime_config, payload)
