from __future__ import annotations

import hashlib
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
    EvaluationSchemaError,
    annotation_path,
    artifact_directory,
    media_directory,
    validate_dataset_id,
    validate_evaluation_row,
    validate_evaluation_rows,
    validate_repository_path,
)


def _row(
    sample_id: str = "sample-001",
    video: str = "media/video_qa/videomme/videos/clip.mp4",
):
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "eval_task": "videomme",
        "sample_id": sample_id,
        "benchmark": "videomme",
        "split": "test",
        "problem": "What happens next?",
        "answer": "B",
        "images": [],
        "videos": [video],
        "problem_type": "video_qa_mc",
        "source": "Video-MME",
        "family": "video_qa",
        "choices": ["A. Nothing", "B. A person enters"],
        "subtitles": ["media/video_qa/videomme/subtitles/clip.srt"],
        "preprocessed": {
            "video_path": "artifacts/video_qa/videomme/preprocessed/clip.npz",
            "settings": {"frames": 128},
        },
        "task_payload": {"duration": 12.5},
        "metadata": {"question_id": "q-1"},
        "evaluation": {"group": "short_video"},
    }


class EvaluationSchemaTests(unittest.TestCase):
    def test_accepts_training_core_and_eval_envelope(self) -> None:
        row = _row()

        self.assertIs(validate_evaluation_row(row), row)
        self.assertEqual(
            annotation_path("videomme", "test"),
            "annotations/video_qa/videomme/test.jsonl",
        )

    def test_accepts_mask_backed_segmentation_without_text_answer(self) -> None:
        row = _row(video="")
        row.update(
            {
                "eval_task": "segmentation",
                "benchmark": "segmentation",
                "split": "mevis",
                "answer": None,
                "images": [],
                "videos": ["media/segmentation/videos/mevis/clip.mp4"],
                "problem_type": "segmentation",
                "source": "mevis",
                "family": "segmentation",
                "task_payload": {
                    "segmentation_output": {
                        "frames": ["00000"],
                        "segmentation_rle": {
                            "00000": {"size": [2, 2], "counts": "13"}
                        },
                    }
                },
            }
        )
        row.pop("subtitles")
        row.pop("preprocessed")

        self.assertIs(validate_evaluation_row(row), row)
        row["task_payload"] = {}
        with self.assertRaisesRegex(
            EvaluationSchemaError,
            "answer must contain a nonempty oracle label",
        ):
            validate_evaluation_row(row)

    def test_video_qa_benchmarks_share_one_physical_family(self) -> None:
        for benchmark in (
            "videomme",
            "videommev2",
            "mvbench",
            "mmvu",
            "videoholmes",
            "longvideobench",
            "mlvu",
        ):
            with self.subTest(benchmark=benchmark):
                self.assertEqual(
                    annotation_path(benchmark, "test"),
                    f"annotations/video_qa/{benchmark}/test.jsonl",
                )
                self.assertEqual(
                    media_directory(benchmark, "videos"),
                    f"media/video_qa/{benchmark}/videos",
                )
                self.assertEqual(
                    artifact_directory(benchmark),
                    f"artifacts/video_qa/{benchmark}",
                )

        self.assertEqual(
            annotation_path("spatial_grounding", "refcoco_val"),
            "annotations/spatial_grounding/refcoco_val.jsonl",
        )

    def test_spatial_intelligence_benchmarks_share_one_physical_family(self) -> None:
        for benchmark in ("vsi", "mmsi", "mindcube", "revsi"):
            with self.subTest(benchmark=benchmark):
                self.assertEqual(
                    annotation_path(benchmark, "test"),
                    f"annotations/spatial_intelligence/{benchmark}/test.jsonl",
                )
                self.assertEqual(
                    media_directory(benchmark, "videos"),
                    f"media/spatial_intelligence/{benchmark}/videos",
                )
                self.assertEqual(
                    artifact_directory(benchmark),
                    f"artifacts/spatial_intelligence/{benchmark}",
                )

    def test_repository_paths_are_strict_posix_relative_paths(self) -> None:
        invalid = (
            "/tmp/clip.mp4",
            "https://example.invalid/clip.mp4",
            r"media\videomme\videos\clip.mp4",
            "media/videomme/videos/../clip.mp4",
            "media/videomme/videos/./clip.mp4",
        )
        for path in invalid:
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_repository_path(path)

        for path in invalid:
            with self.subTest(row_path=path):
                row = _row(video=path)
                with self.assertRaises(EvaluationSchemaError):
                    validate_evaluation_row(row)

    def test_requires_snake_case_dataset_identifiers_and_canonical_scope(self) -> None:
        for identifier in ("VideoMME", "video-mme", "video__mme", "_videomme"):
            with self.subTest(identifier=identifier):
                with self.assertRaises(ValueError):
                    validate_dataset_id(identifier)

        row = _row(video="media/other_benchmark/videos/clip.mp4")
        with self.assertRaisesRegex(
            EvaluationSchemaError,
            "media/video_qa/videomme/videos",
        ):
            validate_evaluation_row(row)

    def test_rejects_duplicate_rows_and_asset_case_collisions(self) -> None:
        with self.assertRaisesRegex(EvaluationSchemaError, "duplicate sample_id"):
            validate_evaluation_rows([_row(), deepcopy(_row())])

        second = _row(
            sample_id="sample-002",
            video="media/video_qa/videomme/videos/Clip.mp4",
        )
        with self.assertRaisesRegex(EvaluationSchemaError, "case collision"):
            validate_evaluation_rows([_row(), second])

    def test_can_verify_asset_existence_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            assets = {
                "media/video_qa/videomme/videos/clip.mp4": b"video",
                "media/video_qa/videomme/subtitles/clip.srt": b"subtitle",
                "artifacts/video_qa/videomme/preprocessed/clip.npz": b"artifact",
            }
            for relative, content in assets.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            video_path = "media/video_qa/videomme/videos/clip.mp4"
            checksum = hashlib.sha256(assets[video_path]).hexdigest()
            validate_evaluation_row(
                _row(),
                repository_root=root,
                checksums={video_path: checksum},
            )

            with self.assertRaisesRegex(EvaluationSchemaError, "checksum mismatch"):
                validate_evaluation_row(
                    _row(),
                    repository_root=root,
                    checksums={video_path: "0" * 64},
                )

            (root / video_path).unlink()
            with self.assertRaisesRegex(EvaluationSchemaError, "does not exist"):
                validate_evaluation_row(_row(), repository_root=root)


if __name__ == "__main__":
    unittest.main()
