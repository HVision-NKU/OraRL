from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ORARL_ROOT = Path(__file__).resolve().parents[1]
if str(ORARL_ROOT) not in sys.path:
    sys.path.insert(0, str(ORARL_ROOT))

from orarl.data.build import ConfigError, ShortfallError, build_dataset  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _fixture_config(tmp_path: Path, quota: int = 2, canary_size: int = 1) -> Path:
    media = tmp_path / "media"
    media.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4", "benchmark.mp4"):
        (media / name).write_bytes(name.encode("utf-8"))

    _write_jsonl(
        tmp_path / "source.jsonl",
        [
            {"problem": "Prompt one", "answer": "A", "videos": ["a.mp4"]},
            {"problem": "  PROMPT   ONE ", "answer": "A2", "videos": ["./a.mp4"]},
            {"problem": "Prompt two", "answer": "B", "videos": ["b.mp4"]},
            {"problem": "Prompt three", "answer": "C", "videos": ["c.mp4"]},
            {
                "problem": "Training prompt on benchmark media",
                "answer": "D",
                "videos": ["benchmark.mp4"],
            },
        ],
    )
    _write_jsonl(
        tmp_path / "benchmarks.jsonl",
        [
            {
                "problem": "Different benchmark prompt",
                "path": str(media / "benchmark.mp4"),
                "problem_type": "video_qa_mc",
            }
        ],
    )
    config = {
        "seed": 17,
        "target": quota,
        "canary_size": canary_size,
        "max_prompts_per_media": 1,
        "require_media": True,
        "benchmark_excludes": ["benchmarks.jsonl"],
        "sources": [
            {
                "name": "video_source",
                "input": "source.jsonl",
                "task": "video_qa_mc",
                "family": "video_qa",
                "quota": quota,
                "media_root": "media",
                "license": "test-only",
            }
        ],
    }
    config_path = tmp_path / f"config-{quota}-{canary_size}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class DataBuildTests(unittest.TestCase):
    def test_example_manifest_matches_materialized_paper_mixture(self) -> None:
        payload = yaml.safe_load(
            (ORARL_ROOT / "configs" / "data_sources.example.yaml").read_text(encoding="utf-8")
        )
        quotas = {source["name"]: source["quota"] for source in payload["sources"]}

        self.assertEqual(payload["target"], 100_032)
        self.assertEqual(
            quotas,
            {
                "temporal_grounding_train": 20_096,
                "tracking_train": 13_952,
                "segmentation_train": 12_032,
                "spatial_grounding_train": 7_040,
                "spatial_temporal_grounding_train": 9_536,
                "video_qa_train": 20_288,
                "spatial_intelligence_train": 17_088,
            },
        )
        self.assertEqual(sum(quotas.values()), payload["target"])

    def test_build_is_deterministic_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            config = _fixture_config(tmp_path)
            first_train = tmp_path / "first.jsonl"
            first_canary = tmp_path / "first.canary.jsonl"
            first_manifest_path = tmp_path / "first.manifest.json"
            second_train = tmp_path / "second.jsonl"
            second_canary = tmp_path / "second.canary.jsonl"

            manifest = build_dataset(
                config,
                first_train,
                canary_output=first_canary,
                manifest_output=first_manifest_path,
            )
            build_dataset(
                config,
                second_train,
                canary_output=second_canary,
                manifest_output=tmp_path / "second.manifest.json",
            )

            self.assertEqual(first_train.read_bytes(), second_train.read_bytes())
            self.assertEqual(first_canary.read_bytes(), second_canary.read_bytes())
            self.assertEqual(manifest["actual_train_rows"], 2)
            self.assertEqual(manifest["actual_canary_rows"], 1)
            self.assertEqual(manifest["counts"]["train"]["by_task"], {"video_qa_mc": 2})
            self.assertEqual(manifest["counts"]["train"]["by_source"], {"video_source": 2})
            self.assertEqual(manifest["duplicates"]["candidate_prompt_rows"], 1)
            self.assertEqual(manifest["leakage"]["candidate_rows_excluded"], 1)
            self.assertTrue(all(value == 0 for value in manifest["leakage"]["output"].values()))
            self.assertEqual(
                manifest["checksums"]["train"],
                hashlib.sha256(first_train.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["checksums"]["canary"],
                hashlib.sha256(first_canary.read_bytes()).hexdigest(),
            )

    def test_short_quota_fails_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            config = _fixture_config(tmp_path, quota=4, canary_size=0)
            train = tmp_path / "short.jsonl"

            with self.assertRaisesRegex(ShortfallError, "video_source short by 1"):
                build_dataset(config, train)

            self.assertFalse(train.exists())
            self.assertFalse((tmp_path / "short.canary.jsonl").exists())
            self.assertFalse((tmp_path / "short.manifest.json").exists())

    def test_short_quota_can_be_explicitly_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            config = _fixture_config(tmp_path, quota=4, canary_size=0)
            train = tmp_path / "allowed.jsonl"

            manifest = build_dataset(config, train, allow_shortfall=True)

            self.assertEqual(manifest["actual_train_rows"], 3)
            self.assertEqual(
                manifest["selection"]["source_shortfalls"],
                {"video_source": 1},
            )

    def test_preserves_declared_scoring_subtypes_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            media = tmp_path / "media"
            media.mkdir()
            for name in ("one.mp4", "two.mp4"):
                (media / name).write_bytes(b"media")
            _write_jsonl(
                tmp_path / "spatial.jsonl",
                [
                    {
                        "problem": "How many objects?",
                        "answer": "2",
                        "videos": ["one.mp4"],
                        "problem_type": "object_counting",
                    },
                    {
                        "problem": "Which direction?",
                        "answer": "A",
                        "videos": ["two.mp4"],
                        "problem_type": "object_rel_direction",
                    },
                ],
            )
            config = {
                "target": 2,
                "canary_size": 0,
                "require_media": True,
                "sources": [
                    {
                        "name": "spatial_intelligence",
                        "input": "spatial.jsonl",
                        "task": "spatial intelligence",
                        "family": "spatial_intelligence",
                        "preserve_problem_type": True,
                        "quota": 2,
                        "media_root": "media",
                    }
                ],
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = tmp_path / "train.jsonl"

            build_dataset(config_path, output)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                {row["problem_type"] for row in rows},
                {"object_counting", "object_rel_direction"},
            )

    def test_output_cannot_replace_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            config = _fixture_config(tmp_path)
            with self.assertRaisesRegex(ConfigError, "cannot replace"):
                build_dataset(
                    config,
                    tmp_path / "source.jsonl",
                    overwrite=True,
                )


if __name__ == "__main__":
    unittest.main()
