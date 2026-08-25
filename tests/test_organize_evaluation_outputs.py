from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "organize_evaluation_outputs.py"


def _run(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_output_organizer_previews_then_groups_legacy_tasks(tmp_path: Path) -> None:
    root = tmp_path / "Video-ORA-9B"
    legacy_video_qa = root / "mvbench" / "base" / "run"
    legacy_spatial = root / "vsi" / "base" / "run"
    legacy_revsi = root / "revsi" / "base" / "run"
    legacy_tracking = root / "tracking" / "base" / "run"
    for directory in (
        legacy_video_qa,
        legacy_spatial,
        legacy_revsi,
        legacy_tracking,
    ):
        directory.mkdir(parents=True)
        (directory / "summary.json").write_text("{}\n", encoding="utf-8")

    preview = _run(root)

    assert preview.returncode == 0
    assert "mvbench -> video_qa/mvbench" in preview.stdout
    assert legacy_video_qa.is_dir()

    applied = _run(root, "--apply")

    assert applied.returncode == 0
    assert (root / "video_qa/mvbench/base/run/summary.json").is_file()
    assert (root / "spatial_intelligence/vsi/base/run/summary.json").is_file()
    assert (root / "spatial_intelligence/revsi/base/run/summary.json").is_file()
    assert (legacy_tracking / "summary.json").is_file()
    assert not (root / "mvbench").exists()
    assert not (root / "vsi").exists()
    assert not (root / "revsi").exists()


def test_output_organizer_refuses_to_overwrite_results(tmp_path: Path) -> None:
    root = tmp_path / "Video-ORA-9B"
    source = root / "mvbench" / "base" / "run"
    destination = root / "video_qa/mvbench/base/run"
    source.mkdir(parents=True)
    destination.mkdir(parents=True)
    (source / "summary.json").write_text('{"source": true}\n', encoding="utf-8")
    (destination / "summary.json").write_text(
        '{"destination": true}\n',
        encoding="utf-8",
    )

    completed = _run(root, "--apply")

    assert completed.returncode != 0
    assert "refusing to overwrite existing output" in completed.stderr
    assert (source / "summary.json").is_file()
    assert (destination / "summary.json").is_file()
