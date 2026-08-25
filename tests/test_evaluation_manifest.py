from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

ORARL_ROOT = Path(__file__).resolve().parents[1]
if str(ORARL_ROOT) not in sys.path:
    sys.path.insert(0, str(ORARL_ROOT))

from orarl.evaluation import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    ManifestError,
    load_dataset_manifest,
    validate_dataset_manifest,
    validate_dataset_manifest_record,
    validate_evaluation_repository,
)


def _manifest(expected_count: int = 1) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "benchmark": "videomme",
        "split": "test",
        "task": "videomme",
        "family": "video_qa",
        "annotation_path": "annotations/video_qa/videomme/test.jsonl",
        "media_paths": [
            "media/video_qa/videomme/videos",
            "media/video_qa/videomme/subtitles",
        ],
        "artifact_paths": [],
        "expected_count": expected_count,
        "license": "CC-BY-4.0",
        "source_url": "https://example.org/videomme",
        "redistribution_authorized": True,
        "evaluation": {
            "prompt_profile": "video_qa",
            "parser_profile": "multiple_choice",
            "metric_profile": "accuracy",
        },
        "legacy_environment": {
            "VIDEOMME_DATA_FILE": "annotations/video_qa/videomme/test.jsonl",
            "VIDEOMME_VIDEO_ROOT": "media/video_qa/videomme/videos",
            "VIDEOMME_EXPECTED_SAMPLES": 1,
        },
    }


def _evaluation_row() -> dict[str, object]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "eval_task": "videomme",
        "sample_id": "q-1",
        "benchmark": "videomme",
        "split": "test",
        "problem": "What happens next?",
        "answer": "B",
        "images": [],
        "videos": ["media/video_qa/videomme/videos/clip.mp4"],
        "problem_type": "video_qa_mc",
        "source": "Video-MME",
        "family": "video_qa",
        "subtitles": ["media/video_qa/videomme/subtitles/clip.srt"],
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


class EvaluationManifestTests(unittest.TestCase):
    def test_accepts_complete_manifest_record(self) -> None:
        record = _manifest()
        self.assertIs(validate_dataset_manifest_record(record), record)

    def test_rejects_duplicate_datasets_and_noncanonical_layout(self) -> None:
        with self.assertRaisesRegex(ManifestError, "duplicate dataset"):
            validate_dataset_manifest([_manifest(), deepcopy(_manifest())])

        record = _manifest()
        record["annotation_path"] = "annotations/VideoMME/test.jsonl"
        with self.assertRaisesRegex(ManifestError, "annotation_path must be"):
            validate_dataset_manifest_record(record)

    def test_rejects_unsafe_legacy_paths_and_credentialed_source_urls(self) -> None:
        record = _manifest()
        record["source_url"] = "https://" + "user:" + "secret@" + "example.org/videomme"
        record["legacy_environment"] = {"VIDEOMME_DATA_FILE": "/tmp/test.jsonl"}
        record["preprocessing"] = {"artifact_path": "https://example.org/cache.pt"}
        with self.assertRaises(ManifestError) as raised:
            validate_dataset_manifest_record(record)
        self.assertIn("without embedded credentials", str(raised.exception))
        self.assertIn("repository-relative", str(raised.exception))
        self.assertIn("absolute path or URL", str(raised.exception))

    def test_loads_and_validates_a_complete_repository(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            video = root / "media/video_qa/videomme/videos/clip.mp4"
            subtitle = root / "media/video_qa/videomme/subtitles/clip.srt"
            video.parent.mkdir(parents=True)
            subtitle.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            subtitle.write_bytes(b"subtitle")

            annotation = root / "annotations/video_qa/videomme/test.jsonl"
            _write_jsonl(annotation, [_evaluation_row()])
            record = _manifest()
            record["checksums"] = {
                "annotations/video_qa/videomme/test.jsonl": hashlib.sha256(
                    annotation.read_bytes()
                ).hexdigest(),
                "media/video_qa/videomme/videos/clip.mp4": hashlib.sha256(
                    video.read_bytes()
                ).hexdigest(),
            }
            manifest_path = root / MANIFEST_FILENAME
            _write_jsonl(manifest_path, [record])

            loaded = load_dataset_manifest(root)
            self.assertEqual(loaded[0]["benchmark"], "videomme")
            datasets = validate_evaluation_repository(root)
            self.assertEqual(len(datasets[("videomme", "test")]), 1)

            record["expected_count"] = 2
            _write_jsonl(manifest_path, [record])
            with self.assertRaisesRegex(ManifestError, "expected 2 rows"):
                validate_evaluation_repository(root)

    def test_checksum_and_redistribution_checks_are_switchable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            video = root / "media/video_qa/videomme/videos/clip.mp4"
            subtitle = root / "media/video_qa/videomme/subtitles/clip.srt"
            video.parent.mkdir(parents=True)
            subtitle.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            subtitle.write_bytes(b"subtitle")
            _write_jsonl(
                root / "annotations/video_qa/videomme/test.jsonl",
                [_evaluation_row()],
            )

            record = _manifest()
            record["redistribution_authorized"] = False
            record["checksums"] = {
                "media/video_qa/videomme/videos/clip.mp4": "0" * 64
            }
            _write_jsonl(root / MANIFEST_FILENAME, [record])

            load_dataset_manifest(root)
            with self.assertRaisesRegex(ManifestError, "checksum mismatch"):
                load_dataset_manifest(root, repository_root=root)

            record.pop("checksums")
            _write_jsonl(root / MANIFEST_FILENAME, [record])
            with self.assertRaisesRegex(ManifestError, "redistribution is not authorized"):
                validate_evaluation_repository(root)
            validate_evaluation_repository(
                root,
                require_redistribution_authorized=False,
            )

    def test_manifest_filename_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "manifest.jsonl"
            _write_jsonl(path, [_manifest()])
            with self.assertRaisesRegex(ManifestError, MANIFEST_FILENAME):
                load_dataset_manifest(path)


if __name__ == "__main__":
    unittest.main()
