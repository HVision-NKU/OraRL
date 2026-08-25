from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

import orarl.evaluation.staging as evaluation_staging
from orarl.cli import eval_data
from orarl.evaluation import (
    StagingError,
    UploadError,
    build_evaluation_repository,
    dataset_card_metadata,
    dataset_card_subsets,
    export_evaluation_index,
    export_public_evaluation_repository,
    inventory_evaluation_sources,
    upload_evaluation_repository,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def _source_fixture(
    tmp_path: Path,
    *,
    expected_count: int | None = 1,
    with_artifact: bool = False,
) -> Path:
    video_root = tmp_path / "source-videos"
    video_root.mkdir()
    (video_root / "clip.mp4").write_bytes(b"release-video")
    artifact_root = tmp_path / "source-artifacts"
    if with_artifact:
        artifact_root.mkdir()
        (artifact_root / "clip.npz").write_bytes(b"processed-video")
    annotation = tmp_path / "annotations.jsonl"
    row: dict[str, object] = {
        "id": "sample-1",
        "question": "What happens?",
        "answer": "A",
        "video": "clip.mp4",
    }
    if with_artifact:
        row["preprocessed_video"] = "clip.npz"
    _write_jsonl(
        annotation,
        [row],
    )
    manifest = tmp_path / "private-sources.jsonl"
    source: dict[str, object] = {
        "benchmark": "videomme",
        "eval_task": "videomme",
        "split": "test",
        "family": "video_qa",
        "adapter": "generic",
        "annotation_input": str(annotation),
        "video_root": str(video_root),
        "expected_count": expected_count,
        "license": "fixture-only",
        "source_url": "https://example.org/videomme",
        "redistribution_authorized": True,
        "evaluation": {
            "prompt_profile": "fixture",
            "parser_profile": "multiple_choice",
            "metric_profile": "accuracy",
        },
    }
    if with_artifact:
        source["preprocessed_root"] = str(artifact_root)
    _write_jsonl(
        manifest,
        [source],
    )
    return manifest


def _card_header(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _prefix, header, _body = text.split("---\n", 2)
    value = yaml.safe_load(header)
    assert isinstance(value, dict)
    return value


def test_build_generates_deterministic_hf_card_and_attributes(tmp_path: Path) -> None:
    manifest = _source_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_evaluation_repository(manifest, first)
    build_evaluation_repository(manifest, second)

    assert (first / "README.md").read_bytes() == (second / "README.md").read_bytes()
    assert (first / ".gitattributes").read_bytes() == (second / ".gitattributes").read_bytes()
    metadata = _card_header(first / "README.md")
    assert metadata["configs"] == [
        {
            "config_name": "videomme",
            "data_files": [
                {
                    "split": "test",
                    "path": "annotations/video_qa/videomme/test.jsonl",
                }
            ],
        }
    ]
    assert metadata["orarl"]["totals"]["row_count"] == 1
    assert metadata["orarl"]["totals"]["asset_count"] == 2
    assert metadata["orarl"]["benchmarks"][0]["task"] == "videomme"
    assert metadata["orarl"]["benchmarks"][0]["family"] == "video_qa"
    assert metadata["size_categories"] == ["n<1K"]
    assert "evaluation" in metadata["tags"]
    card = (first / "README.md").read_text(encoding="utf-8")
    assert "# OraRL-Eval" in card
    assert "> **Evaluation only.**" in card
    assert "## Benchmark configs and provenance" in card
    assert "## Canonical row schema" in card
    assert "## Intended use and limitations" in card
    assert "arXiv preprint arXiv:2608.20492" in card
    assert "load_dataset(" in card
    assert "orarl-eval \\" in card
    assert "--dataset OraRL/OraRL-Eval" in card
    attributes = (first / ".gitattributes").read_text(encoding="utf-8")
    assert "annotations/video_qa/videomme/test.jsonl filter=lfs" not in attributes
    assert "media/video_qa/videomme/videos/** filter=lfs" in attributes


def test_metadata_cli_refreshes_and_validates_existing_repository(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    build_evaluation_repository(_source_fixture(tmp_path), root)
    expected_readme = (root / "README.md").read_bytes()
    (root / "README.md").write_text("stale card\n", encoding="utf-8")

    assert eval_data.main(["metadata", "--root", str(root)]) == 0

    assert (root / "README.md").read_bytes() == expected_readme


def test_export_index_keeps_only_portable_jsonl_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = tmp_path / "complete"
    index = tmp_path / "index"
    fast_index = tmp_path / "fast-index"
    build_evaluation_repository(_source_fixture(tmp_path), complete)

    summary = export_evaluation_index(complete, index)

    def _fail_hash(_path: Path) -> str:
        pytest.fail("checksum-free index export must not calculate SHA-256")

    monkeypatch.setattr(evaluation_staging, "sha256_file", _fail_hash)
    fast_summary = export_evaluation_index(
        complete,
        fast_index,
        checksums=False,
    )

    assert summary["annotation_assets"] == 1
    assert summary["external_assets"] == 1
    assert summary["asset_checksums"] is False
    assert fast_summary["asset_checksums"] is False
    assert (fast_index / "datasets.jsonl").read_bytes() == (
        index / "datasets.jsonl"
    ).read_bytes()
    assert (index / "datasets.jsonl").is_file()
    assert not (index / "assets.jsonl").exists()
    assert (index / "annotations/video_qa/videomme/test.jsonl").is_file()
    assert not (index / "media").exists()
    assert not (index / "artifacts").exists()
    index_card = (index / "README.md").read_text(encoding="utf-8")
    assert "pretty_name: OraRL-Data" in index_card
    assert "No media is redistributed" in index_card
    assert "--asset-root" in index_card
    assert index.stat().st_mode & 0o777 == 0o755
    assert (index / "datasets.jsonl").stat().st_mode & 0o777 == 0o644
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_evaluation_index(complete, index)


def test_export_public_keeps_raw_media_and_removes_processed_artifacts(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "complete"
    public = tmp_path / "public"
    build_evaluation_repository(
        _source_fixture(tmp_path, with_artifact=True),
        complete,
    )
    generated_frame = (
        complete
        / "media/video_qa/videomme/videos/test/sam2-cache/000000.jpg"
    )
    generated_frame.parent.mkdir(parents=True)
    generated_frame.write_bytes(b"unmanifested-generated-frame")

    summary = export_public_evaluation_repository(complete, public)

    assert summary["excluded_artifacts"] == 1
    assert (public / "media/video_qa/videomme/videos/test/clip.mp4").is_file()
    assert not (public / generated_frame.relative_to(complete)).exists()
    assert not (public / "artifacts").exists()
    assets = [
        json.loads(line)
        for line in (public / "assets.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(record["kind"] != "artifacts" for record in assets)
    datasets = [
        json.loads(line)
        for line in (public / "datasets.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert datasets[0]["artifact_paths"] == []
    rows = [
        json.loads(line)
        for line in (
            public / "annotations/video_qa/videomme/test.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert "preprocessed" not in rows[0]
    card = (public / "README.md").read_text(encoding="utf-8")
    assert "--dataset ./OraRL-Data/OraRL-eval-data" in card
    assert "# OraRL-Data" in card
    assert "referenced raw media" in card
    assert "Preprocessed tensors" in card
    assert "No media is redistributed" not in card
    assert "path: OraRL-eval-data/annotations/video_qa/videomme/test.jsonl" in card
    assert "OraRL-eval-data/media/video_qa/videomme/videos/**" in (
        public / ".gitattributes"
    ).read_text(encoding="utf-8")

    (public / "README.md").write_text("stale card\n", encoding="utf-8")
    assert (
        eval_data.main(
            [
                "metadata",
                "--root",
                str(public),
                "--repo-id",
                "OraRL/OraRL-Data",
            ]
        )
        == 0
    )
    assert "referenced raw media" in (public / "README.md").read_text(
        encoding="utf-8"
    )


def test_card_subset_and_metadata_apis_group_splits() -> None:
    records = [
        {
            "schema_version": 1,
            "benchmark": "spatial_grounding",
            "split": split,
            "task": "spatial_grounding",
            "family": "spatial_grounding",
            "annotation_path": f"annotations/spatial_grounding/{split}.jsonl",
            "media_paths": ["media/spatial_grounding/images"],
            "artifact_paths": [],
            "expected_count": count,
            "license": "fixture-only",
            "source_url": "https://example.org/spatial",
            "redistribution_authorized": True,
            "evaluation": {"metric_profile": "iou"},
        }
        for split, count in (("refcoco_val", 2), ("refcocog_test", 3))
    ]
    assets = [
        {
            "path": f"annotations/spatial_grounding/{split}.jsonl",
            "bytes": count * 10,
            "kind": "annotations",
            "benchmark": "spatial_grounding",
        }
        for split, count in (("refcoco_val", 2), ("refcocog_test", 3))
    ]

    assert dataset_card_subsets(records) == [
        {
            "config_name": "spatial_grounding",
            "data_files": [
                {
                    "split": "refcoco_val",
                    "path": "annotations/spatial_grounding/refcoco_val.jsonl",
                },
                {
                    "split": "refcocog_test",
                    "path": "annotations/spatial_grounding/refcocog_test.jsonl",
                },
            ],
        }
    ]
    metadata = dataset_card_metadata(records, assets)
    assert metadata["orarl"]["totals"] == {
        "benchmark_count": 1,
        "split_count": 2,
        "row_count": 5,
        "asset_count": 2,
        "byte_count": 50,
    }
    assert metadata["size_categories"] == ["n<1K"]
    assert metadata["tags"][:3] == ["video", "multimodal", "evaluation"]
    fields = {item["name"]: item for item in metadata["orarl"]["schema_fields"]}
    assert fields["sample_id"]["required"] is True
    assert fields["task_payload"]["type"] == "object"


def test_inventory_can_lock_null_expected_counts_for_build(tmp_path: Path) -> None:
    manifest = _source_fixture(tmp_path, expected_count=None)
    inventory = inventory_evaluation_sources(manifest)

    assert inventory["sources"][0]["expected_count"] is None
    assert inventory["sources"][0]["count_matches"] is None
    assert inventory["totals"]["unlocked_sources"] == 1
    with pytest.raises(StagingError, match="expected_count is not locked"):
        build_evaluation_repository(manifest, tmp_path / "unlocked-output")

    locked = tmp_path / "locked-sources.jsonl"
    assert (
        eval_data.main(
            [
                "inventory",
                "--manifest",
                str(manifest),
                "--write-locked-manifest",
                str(locked),
                "--task",
                "videomme",
            ]
        )
        == 0
    )
    locked_record = json.loads(locked.read_text(encoding="utf-8"))
    assert locked_record["expected_count"] == 1
    assert Path(locked_record["annotation_input"]).is_absolute()
    build_evaluation_repository(locked, tmp_path / "locked-output")
    with pytest.raises(StagingError, match="no records for task"):
        inventory_evaluation_sources(manifest, tasks=["temporal_grounding"])


def test_upload_validates_then_passes_only_noncredential_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    build_evaluation_repository(_source_fixture(tmp_path), root)
    calls: list[dict[str, object]] = []
    constructors: list[tuple[object, ...]] = []
    module = types.ModuleType("huggingface_hub")

    class FakeApi:
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructors.append((*args, kwargs))

        def upload_large_folder(self, **kwargs: object) -> None:
            calls.append(dict(kwargs))

    module.HfApi = FakeApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("HF_TOKEN", "environment-login-only")

    result = upload_evaluation_repository(
        root,
        "OraRL/OraRL-Eval",
        revision="release",
        private=True,
        num_workers=7,
    )

    assert constructors == [({},)]
    assert calls == [
        {
            "repo_id": "OraRL/OraRL-Eval",
            "folder_path": str(root.resolve()),
            "repo_type": "dataset",
            "revision": "release",
            "private": True,
            "num_workers": 7,
        }
    ]
    assert "token" not in calls[0]
    assert result["rows"] == 1


def test_upload_failure_does_not_echo_authentication_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    build_evaluation_repository(_source_fixture(tmp_path), root)
    detail = "sensitive-auth-detail"
    module = types.ModuleType("huggingface_hub")

    class FakeApi:
        def upload_large_folder(self, **_kwargs: object) -> None:
            raise RuntimeError(f"authorization failed for {detail}")

    module.HfApi = FakeApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(UploadError) as captured:
        upload_evaluation_repository(root, "OraRL/OraRL-Eval")
    assert detail not in str(captured.value)


def test_upload_never_calls_hf_when_full_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    build_evaluation_repository(_source_fixture(tmp_path), root)
    video = next(
        path
        for path in (root / "media/video_qa/videomme/videos").rglob("*")
        if path.is_file()
    )
    video.write_bytes(b"tampered")
    module = types.ModuleType("huggingface_hub")

    class ForbiddenApi:
        def __init__(self) -> None:
            raise AssertionError("network client must not be constructed")

    module.HfApi = ForbiddenApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(StagingError, match="mismatch"):
        upload_evaluation_repository(root, "OraRL/OraRL-Eval")


def test_upload_cli_has_no_token_argument() -> None:
    parser = eval_data.create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "upload",
                "--root",
                "/tmp/repository",
                "--repo-id",
                "org/repo",
                "--token",
                "forbidden",
            ]
        )
