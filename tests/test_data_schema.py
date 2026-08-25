from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ORARL_ROOT = Path(__file__).resolve().parents[1]
if str(ORARL_ROOT) not in sys.path:
    sys.path.insert(0, str(ORARL_ROOT))

from orarl.data import (  # noqa: E402
    SchemaError,
    canonicalize_record,
    prompt_identity,
    validate_record,
)


class DataSchemaTests(unittest.TestCase):
    def test_canonicalizes_aliases_and_relative_media(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            media_root = tmp_path / "media"
            image = media_root / "frames" / "one.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")

            record = canonicalize_record(
                {
                    "question": "  Which object is red?  ",
                    "ground_truth": {"label": "ball"},
                    "image": "frames/one.jpg",
                    "metadata": {"split": "train"},
                },
                problem_type="spatial grounding",
                source="unit_source",
                family="spatial",
                media_root=media_root,
                require_media=True,
            )

            self.assertEqual(record["problem"], "  Which object is red?  ")
            self.assertEqual(record["answer"], {"label": "ball"})
            self.assertEqual(record["images"], [str(image)])
            self.assertEqual(record["videos"], [])
            self.assertEqual(record["problem_type"], "spatial grounding")
            self.assertEqual(record["source"], "unit_source")
            self.assertEqual(record["metadata"], {"split": "train"})

    def test_rejects_empty_oracle_labels(self) -> None:
        answers = [None, "", "   ", [], {}, {"label": ""}, float("nan")]
        for answer in answers:
            with self.subTest(answer=answer):
                record = {
                    "problem": "Question",
                    "answer": answer,
                    "images": ["image.jpg"],
                    "videos": [],
                    "problem_type": "spatial grounding",
                    "source": "unit_source",
                }
                with self.assertRaisesRegex(SchemaError, "oracle label"):
                    validate_record(record)

    def test_media_existence_is_switchable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            record = {
                "problem": "Question",
                "answer": "Answer",
                "images": [str(Path(raw_tmp) / "missing.jpg")],
                "videos": [],
                "problem_type": "spatial grounding",
                "source": "unit_source",
            }
            validate_record(record, require_media=False)
            with self.assertRaisesRegex(SchemaError, "does not exist"):
                validate_record(record, require_media=True)

    def test_prompt_identity_normalizes_text_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            first = {
                "problem": "<image> Which   OBJECT is red?",
                "answer": "ball",
                "images": [str(tmp_path / "frames" / ".." / "one.jpg")],
                "videos": [],
                "problem_type": "Spatial Grounding",
                "source": "one",
            }
            second = {
                **first,
                "problem": "which object is red?",
                "images": [str(tmp_path / "one.jpg")],
                "source": "two",
            }
            self.assertEqual(prompt_identity(first), prompt_identity(second))

    def test_rejects_remote_media_reference(self) -> None:
        with self.assertRaisesRegex(SchemaError, "remote media"):
            canonicalize_record(
                {
                    "problem": "Question",
                    "answer": "Answer",
                    "video": "https://example.invalid/clip.mp4",
                },
                problem_type="video_qa_mc",
                source="unit_source",
            )


if __name__ == "__main__":
    unittest.main()
