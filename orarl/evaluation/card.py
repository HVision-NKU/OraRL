"""Deterministic Hugging Face metadata for staged evaluation repositories."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .layout import ASSET_MANIFEST_FILENAME
from .manifest import load_dataset_manifest, validate_dataset_manifest
from .schema import (
    EVALUATION_FIELDS,
    EVALUATION_REQUIRED_FIELDS,
    EVALUATION_SCHEMA_VERSION,
)

DATASET_CARD_FILENAME = "README.md"
GIT_ATTRIBUTES_FILENAME = ".gitattributes"
DEFAULT_DATASET_REPO_ID = "OraRL/OraRL-Eval"
DEFAULT_INDEX_REPO_ID = "OraRL/OraRL-Data"
PUBLIC_EVAL_DATA_DIRECTORY = "OraRL-eval-data"
PAPER_URL = "https://arxiv.org/abs/2608.20492"
PROJECT_URL = "https://orarl.github.io/"
CODE_URL = "https://github.com/HVision-NKU/OraRL"

_PUBLIC_FAMILY_ORDER = (
    "temporal_grounding",
    "spatial_grounding",
    "segmentation",
    "tracking",
    "spatial_temporal_grounding",
    "video_qa",
    "spatial_intelligence",
)
_PUBLIC_FAMILY_LABELS = {
    "temporal_grounding": "Temporal grounding",
    "spatial_grounding": "Spatial grounding",
    "segmentation": "Segmentation",
    "tracking": "Visual tracking",
    "spatial_temporal_grounding": "Spatial-temporal grounding",
    "video_qa": "Video question answering",
    "spatial_intelligence": "Spatial intelligence",
}
_PUBLIC_BENCHMARK_LABELS = {
    "longvideobench": "LongVideoBench",
    "mindcube": "MindCube-Tiny",
    "mlvu": "MLVU",
    "mmsi": "MMSI-Bench",
    "mmvu": "MMVU",
    "mvbench": "MV-Bench",
    "revsi": "ReVSI",
    "segmentation": "RefCOCO / + / g, MeViS, ReasonVOS",
    "spatial_grounding": "RefCOCO / + / g",
    "stvg": "STVG",
    "temporal_grounding": "TimeLens: Charades-STA, ActivityNet, QVHighlights",
    "tracking": "GOT-10k",
    "videoholmes": "VideoHolmes",
    "videomme": "VideoMME",
    "videommev2": "VideoMME-v2",
    "vsi": "VSI-Bench",
}

_FIELD_TYPES = {
    "schema_version": "integer",
    "eval_task": "string",
    "sample_id": "string",
    "benchmark": "string",
    "split": "string",
    "problem": "string",
    "answer": "json",
    "images": "list[media]",
    "videos": "list[media]",
    "problem_type": "string",
    "source": "string",
    "family": "string",
    "choices": "list[string]",
    "subtitles": "list[media]",
    "preprocessed": "object",
    "task_payload": "object",
    "metadata": "object",
    "evaluation": "object",
}


class DatasetCardError(ValueError):
    """Raised when generated Hugging Face metadata is missing or stale."""


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise DatasetCardError(f"{path}:{line_number}: record must be an object")
                records.append(value)
    except json.JSONDecodeError as error:
        raise DatasetCardError(f"{path}: invalid JSON: {error}") from error
    except OSError as error:
        raise DatasetCardError(f"cannot read {path}: {error}") from error
    return records


def dataset_card_subsets(
    datasets: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one deterministic Hugging Face config per benchmark."""

    records = validate_dataset_manifest(datasets)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["benchmark"]), []).append(record)

    configs: list[dict[str, Any]] = []
    for benchmark in sorted(grouped):
        data_files = [
            {
                "split": str(record["split"]),
                "path": str(record["annotation_path"]),
            }
            for record in sorted(grouped[benchmark], key=lambda item: str(item["split"]))
        ]
        configs.append({"config_name": benchmark, "data_files": data_files})
    return configs


def _one_or_many(values: Iterable[str]) -> str | list[str]:
    ordered = sorted(set(values))
    return ordered[0] if len(ordered) == 1 else ordered


def _size_category(row_count: int) -> str:
    if row_count < 1_000:
        return "n<1K"
    if row_count < 10_000:
        return "1K<n<10K"
    if row_count < 100_000:
        return "10K<n<100K"
    if row_count < 1_000_000:
        return "100K<n<1M"
    if row_count < 10_000_000:
        return "1M<n<10M"
    if row_count < 100_000_000:
        return "10M<n<100M"
    if row_count < 1_000_000_000:
        return "100M<n<1B"
    return "n>1B"


def _human_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _display_value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _source_links(value: object) -> str:
    urls = value if isinstance(value, list) else [value]
    return "<br>".join(
        f"[source {index}]({url})" if len(urls) > 1 else f"[source]({url})"
        for index, url in enumerate(urls, start=1)
    )


def _asset_counts(
    assets: Sequence[Mapping[str, Any]],
    *,
    benchmark: str | None = None,
    kind: str | None = None,
) -> tuple[int, int]:
    selected = [
        record
        for record in assets
        if (benchmark is None or str(record.get("benchmark")) == benchmark)
        and (kind is None or str(record.get("kind")) == kind)
    ]
    return len(selected), sum(int(record.get("bytes", 0)) for record in selected)


def dataset_card_metadata(
    datasets: Iterable[Mapping[str, Any]],
    assets: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic dataset-card YAML metadata and release statistics."""

    records = validate_dataset_manifest(datasets)
    asset_records = sorted(
        (dict(record) for record in assets),
        key=lambda item: str(item.get("path", "")),
    )
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    annotation_bytes = {
        str(asset.get("path")): int(asset.get("bytes", 0))
        for asset in asset_records
        if asset.get("kind") == "annotations"
    }
    for record in records:
        grouped.setdefault(str(record["benchmark"]), []).append(record)

    benchmark_metadata: list[dict[str, Any]] = []
    dataset_info: list[dict[str, Any]] = []
    for benchmark in sorted(grouped):
        benchmark_records = sorted(
            grouped[benchmark],
            key=lambda item: str(item["split"]),
        )
        asset_count, byte_count = _asset_counts(asset_records, benchmark=benchmark)
        row_count = sum(int(record["expected_count"]) for record in benchmark_records)
        split_metadata = [
            {
                "name": str(record["split"]),
                "num_bytes": annotation_bytes.get(str(record["annotation_path"]), 0),
                "num_examples": int(record["expected_count"]),
            }
            for record in benchmark_records
        ]
        annotation_byte_count = sum(int(item["num_bytes"]) for item in split_metadata)
        dataset_info.append(
            {
                "config_name": benchmark,
                "splits": split_metadata,
                "download_size": byte_count,
                "dataset_size": annotation_byte_count,
            }
        )
        benchmark_metadata.append(
            {
                "config_name": benchmark,
                "task": _one_or_many(str(record["task"]) for record in benchmark_records),
                "family": _one_or_many(
                    str(record.get("family", record["task"])) for record in benchmark_records
                ),
                "license": _one_or_many(str(record["license"]) for record in benchmark_records),
                "source_url": _one_or_many(
                    str(record["source_url"]) for record in benchmark_records
                ),
                "splits": [str(record["split"]) for record in benchmark_records],
                "row_count": row_count,
                "asset_count": asset_count,
                "byte_count": byte_count,
            }
        )

    total_assets, total_bytes = _asset_counts(asset_records)
    total_rows = sum(int(record["expected_count"]) for record in records)
    return {
        "pretty_name": "OraRL Evaluation Data",
        "license": "other",
        "task_categories": ["video-text-to-text"],
        "tags": [
            "video",
            "multimodal",
            "evaluation",
            "benchmark",
            "temporal-grounding",
            "spatial-grounding",
            "object-tracking",
            "video-question-answering",
            "spatial-reasoning",
        ],
        "size_categories": [_size_category(total_rows)],
        "configs": dataset_card_subsets(records),
        "dataset_info": dataset_info,
        "orarl": {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "schema_fields": [
                {
                    "name": field,
                    "type": _FIELD_TYPES[field],
                    "required": field in EVALUATION_REQUIRED_FIELDS,
                }
                for field in EVALUATION_FIELDS
            ],
            "totals": {
                "benchmark_count": len(grouped),
                "split_count": len(records),
                "row_count": total_rows,
                "asset_count": total_assets,
                "byte_count": total_bytes,
            },
            "benchmarks": benchmark_metadata,
        },
    }


def _public_family_table(metadata: Mapping[str, Any]) -> str:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for benchmark in metadata["orarl"]["benchmarks"]:
        family = str(benchmark["family"])
        grouped.setdefault(family, []).append(benchmark)

    lines = [
        "| Task family | Benchmarks | Examples |",
        "| --- | --- | ---: |",
    ]
    ordered_families = [
        *[family for family in _PUBLIC_FAMILY_ORDER if family in grouped],
        *sorted(set(grouped) - set(_PUBLIC_FAMILY_ORDER)),
    ]
    for family in ordered_families:
        benchmarks = grouped[family]
        names = [
            _PUBLIC_BENCHMARK_LABELS.get(
                str(benchmark["config_name"]),
                str(benchmark["config_name"]),
            )
            for benchmark in benchmarks
        ]
        lines.append(
            f"| {_PUBLIC_FAMILY_LABELS.get(family, family.replace('_', ' ').title())} "
            f"| {', '.join(names)} "
            f"| {sum(int(benchmark['row_count']) for benchmark in benchmarks):,} |"
        )
    return "\n".join(lines)


def _public_provenance_table(metadata: Mapping[str, Any]) -> str:
    lines = [
        "| Config | Splits | Rows | Upstream |",
        "| --- | --- | ---: | --- |",
    ]
    for benchmark in metadata["orarl"]["benchmarks"]:
        splits = ", ".join(f"`{split}`" for split in benchmark["splits"])
        lines.append(
            f"| `{benchmark['config_name']}` "
            f"| {splits} "
            f"| {int(benchmark['row_count']):,} "
            f"| {_source_links(benchmark['source_url'])} |"
        )
    return "\n".join(lines)


def render_public_dataset_card(
    datasets: Iterable[Mapping[str, Any]],
    assets: Iterable[Mapping[str, Any]],
    *,
    repo_id: str = DEFAULT_INDEX_REPO_ID,
    data_directory: str = PUBLIC_EVAL_DATA_DIRECTORY,
) -> str:
    """Render the concise raw-media card published on Hugging Face."""

    data_directory = data_directory.strip("/")
    if not data_directory or data_directory in {".", ".."} or ".." in data_directory.split("/"):
        raise DatasetCardError("public data directory must be a safe relative path")
    metadata = dataset_card_metadata(datasets, assets)
    totals = metadata["orarl"]["totals"]
    metadata["pretty_name"] = "OraRL-Data"
    for config in metadata["configs"]:
        for data_file in config["data_files"]:
            data_file["path"] = f"{data_directory}/{data_file['path']}"
    public_metadata = {
        key: value
        for key, value in metadata.items()
        if key != "orarl"
    }
    yaml_header = yaml.safe_dump(
        public_metadata,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    ).rstrip()
    family_table = _public_family_table(metadata)
    provenance_table = _public_provenance_table(metadata)

    body = f"""# OraRL-Data

**The official evaluation data release for Video-ORA and OraRL.**

[Paper]({PAPER_URL}) · [Project page]({PROJECT_URL}) · [Code]({CODE_URL}) ·
[Video-ORA-9B](https://huggingface.co/OraRL/Video-ORA-9B)

OraRL-Data packages the canonical annotations and referenced raw media used by
the OraRL evaluation suite: **{totals["row_count"]:,} examples** across
**{totals["benchmark_count"]} benchmark configs** and
**{totals["split_count"]} splits**, with
**{_human_bytes(int(totals["byte_count"]))}** of manifested files.
The complete evaluation release lives under `{data_directory}/`, leaving room
for the separate OraRL training release in this repository.

> **Evaluation only.** This release is not training data. Preprocessed tensors
> and model-specific caches are intentionally excluded.

## Benchmarks

{family_table}

## Download

Download the complete release, including raw media:

```bash
pip install -U huggingface_hub

hf download {repo_id} \\
  --repo-type dataset \\
  --include "{data_directory}/**" \\
  --local-dir ./OraRL-Data
```

The Hub download is resumable. `{data_directory}/assets.jsonl` is the
authoritative file inventory; `{data_directory}/datasets.jsonl` records each
split's prompt, parser, metric, and preprocessing protocol.

## Load annotations

Each benchmark is exposed as a Hugging Face Datasets config:

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", "videomme", split="test")
sample = dataset[0]
print(sample["problem"])
print(sample["answer"])
print(sample["videos"])
```

`images`, `videos`, and `subtitles` contain paths relative to
`{data_directory}/` in the downloaded snapshot.

## Evaluate Video-ORA

```bash
orarl-eval \\
  --dataset ./OraRL-Data/{data_directory} \\
  --model OraRL/Video-ORA-9B \\
  --tasks paper \\
  --summary ./orarl-eval-summary.json \\
  --run
```

Exact frame sampling, resolution, prompts, parsers, and metric profiles are
declared in `{data_directory}/datasets.jsonl`; changing them defines a
different evaluation setting.

## Repository layout

```text
OraRL-Data/
├── README.md
└── {data_directory}/
    ├── annotations/    # canonical JSONL evaluation rows
    ├── media/          # referenced raw images, videos, and subtitles
    ├── assets.jsonl    # released-file inventory
    └── datasets.jsonl  # benchmark and protocol manifest
```

## License and provenance

OraRL-Data combines multiple upstream benchmarks and therefore uses
`license: other`. Every benchmark and media item remains subject to its
original terms; this repository does not replace or broaden those licenses.
Please cite the relevant upstream datasets in addition to OraRL.

<details>
<summary>Benchmark sources and splits</summary>

{provenance_table}

</details>

## Citation

```bibtex
@article{{li2026orarl,
  title   = {{Annotations as Rollouts: Efficient and Scalable
             Reinforcement Learning for Video MLLMs}},
  author  = {{Li, Yunheng and Mu, Guohong and Li, Hao and
             Qian, Shengsheng and Zhang, Dingwen and Hou, Qibin
             and Cheng, Ming-Ming}},
  journal = {{arXiv preprint arXiv:2608.20492}},
  year    = {{2026}},
  url     = {{https://arxiv.org/abs/2608.20492}}
}}
```
"""
    return f"---\n{yaml_header}\n---\n\n{body}"


def render_dataset_card(
    datasets: Iterable[Mapping[str, Any]],
    assets: Iterable[Mapping[str, Any]],
    *,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
) -> str:
    """Render the complete deterministic Hugging Face dataset card."""

    asset_records = [dict(record) for record in assets]
    if repo_id == DEFAULT_INDEX_REPO_ID:
        return render_public_dataset_card(
            datasets,
            asset_records,
            repo_id=repo_id,
        )
    metadata = dataset_card_metadata(datasets, asset_records)
    yaml_header = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    ).rstrip()
    totals = metadata["orarl"]["totals"]
    benchmark_lines = [
        "| Config | Family | Splits | Rows | Assets | Size | Terms | Upstream |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for benchmark in metadata["orarl"]["benchmarks"]:
        splits = ", ".join(f"`{split}`" for split in benchmark["splits"])
        benchmark_lines.append(
            f"| `{benchmark['config_name']}` "
            f"| {_display_value(benchmark['family'])} "
            f"| {splits} "
            f"| {int(benchmark['row_count']):,} "
            f"| {int(benchmark['asset_count']):,} "
            f"| {_human_bytes(int(benchmark['byte_count']))} "
            f"| {_display_value(benchmark['license'])} "
            f"| {_source_links(benchmark['source_url'])} |"
        )

    schema_lines = [
        "| Field | Type | Required |",
        "| --- | --- | :---: |",
    ]
    for field in metadata["orarl"]["schema_fields"]:
        schema_lines.append(
            f"| `{field['name']}` | `{field['type']}` "
            f"| {'yes' if field['required'] else 'no'} |"
        )
    artifact_description = (
        "- `artifacts/` contains deterministic preprocessed tensors required by selected\n"
        "  paper protocols."
        if any(record.get("kind") == "artifacts" for record in asset_records)
        else (
            "- `artifacts/` is intentionally omitted; evaluators decode the released raw\n"
            "  media using the declared preprocessing profiles."
        )
    )

    body = f"""# OraRL-Eval

[Paper]({PAPER_URL}) · [Project page]({PROJECT_URL}) · [Code]({CODE_URL})

This repository contains the canonical, content-addressed evaluation inputs for the
OraRL paper suite. It has {totals["benchmark_count"]} benchmark configs,
{totals["split_count"]} splits, {totals["row_count"]:,} rows,
{totals["asset_count"]:,} manifested assets, and
{_human_bytes(int(totals["byte_count"]))} of manifested files.

> **Evaluation only.** Do not use these benchmark splits for training,
> instruction tuning, model selection, or prompt optimization.

## Repository contents

- `datasets.jsonl` declares every benchmark split, preprocessing profile,
  parser, metric, source, and applicable terms.
- `annotations/` contains canonical JSONL rows with repository-relative media
  references.
- `media/` contains the authorized evaluation images, videos, and subtitles
  referenced by those rows.
{artifact_description}
- `assets.jsonl` records the size and SHA-256 digest of every released file.

## Benchmark configs and provenance

{chr(10).join(benchmark_lines)}

Each config follows its upstream benchmark terms. The repository-level
`license: other` reflects this mixed-license collection; it does not replace or
broaden any upstream license.

## Load annotations

Hugging Face Datasets can load one benchmark config at a time:

```python
from datasets import load_dataset

dataset = load_dataset(
    "{repo_id}",
    "temporal_grounding",
)
sample = dataset["charades_timelens"][0]
print(sample["problem"], sample["answer"], sample["videos"])
```

Media fields are repository-relative paths. For reproducible evaluation,
download the complete snapshot, including its LFS media and artifacts:

```bash
hf download {repo_id} \\
  --repo-type dataset \\
  --revision main \\
  --local-dir ./OraRL-Eval
```

## Canonical row schema

{chr(10).join(schema_lines)}

Task-specific targets remain structured in `answer`, `task_payload`, and
`metadata`. Evaluation behavior is fixed by the per-row `evaluation` profile
and the corresponding declaration in `datasets.jsonl`.

## Run the paper evaluation

```bash
orarl-eval \\
  --dataset {repo_id} \\
  --model /path/to/exported-model \\
  --tasks paper \\
  --summary ./orarl-eval-summary.json \\
  --run
```

Use the same command with `--dataset /path/to/OraRL-Eval` for a validated local
snapshot. The evaluator selects configs by their `task` metadata and resolves all
annotation, media, subtitle, and preprocessing paths relative to the snapshot.

## Integrity and reproducibility

`datasets.jsonl` declares each benchmark split and evaluation protocol.
`assets.jsonl` records every staged annotation and asset with its byte size and
SHA-256 digest. Validate a clean snapshot before evaluation:

```bash
orarl-eval-data validate --root /path/to/OraRL-Eval
```

The released preprocessing profiles are protocol-specific. Changing frame
sampling, resolution, prompt templates, answer parsing, or aggregation can
change benchmark scores and should be reported as a different setting.

## Intended use and limitations

OraRL-Eval is intended to reproduce and extend research evaluation across
temporal grounding, tracking, segmentation, spatial grounding,
spatial-temporal grounding, video question answering, and spatial intelligence.
It is not a training corpus.

The collection inherits coverage gaps, annotation errors, representational
biases, and potentially sensitive visual content from its upstream benchmarks.
Users are responsible for complying with upstream access conditions, privacy
requirements, and media licenses. Scores are not evidence of safety,
reliability, or suitability for deployment.

## Citation

```bibtex
@article{{li2026orarl,
  title   = {{Annotations as Rollouts: Efficient and Scalable
             Reinforcement Learning for Video MLLMs}},
  author  = {{Li, Yunheng and Mu, Guohong and Li, Hao and
             Qian, Shengsheng and Zhang, Dingwen and Hou, Qibin
             and Cheng, Ming-Ming}},
  journal = {{arXiv preprint arXiv:2608.20492}},
  year    = {{2026}},
  url     = {{https://arxiv.org/abs/2608.20492}}
}}
```

Please also cite the original benchmark sources listed above when using their
data or reporting results.
"""
    return f"---\n{yaml_header}\n---\n\n{body}"


def render_index_card(
    datasets: Iterable[Mapping[str, Any]],
    annotation_assets: Iterable[Mapping[str, Any]],
    *,
    repo_id: str = DEFAULT_INDEX_REPO_ID,
) -> str:
    """Render the annotation-only Hugging Face dataset card."""

    metadata = dataset_card_metadata(datasets, annotation_assets)
    metadata["pretty_name"] = "OraRL-Data"
    yaml_header = yaml.safe_dump(
        metadata,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=1000,
    ).rstrip()
    totals = metadata["orarl"]["totals"]
    benchmark_lines = [
        "| Config | Family | Splits | Rows | Terms | Upstream |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for benchmark in metadata["orarl"]["benchmarks"]:
        splits = ", ".join(f"`{split}`" for split in benchmark["splits"])
        benchmark_lines.append(
            f"| `{benchmark['config_name']}` "
            f"| {_display_value(benchmark['family'])} "
            f"| {splits} "
            f"| {int(benchmark['row_count']):,} "
            f"| {_display_value(benchmark['license'])} "
            f"| {_source_links(benchmark['source_url'])} |"
        )

    body = f"""# OraRL-Data

[Paper]({PAPER_URL}) · [Project page]({PROJECT_URL}) · [Code]({CODE_URL})

This is the portable, annotation-only evaluation index for OraRL. It contains
{totals["benchmark_count"]} benchmark configs, {totals["split_count"]} splits,
and {totals["row_count"]:,} canonical evaluation rows.

> **No media is redistributed.** Raw images, raw videos, subtitles, and derived
> preprocessing artifacts are intentionally excluded. Users must obtain them
> from the upstream benchmark sources under their respective terms.

> **Evaluation only.** Do not use these benchmark splits for training,
> instruction tuning, model selection, or prompt optimization.

## Repository contents

- `datasets.jsonl` declares each split, preprocessing profile, parser, metric,
  source, and applicable terms.
- `annotations/` contains canonical JSONL rows with portable repository-relative
  media and artifact references.
- `media/`, `artifacts/`, and `assets.jsonl` are intentionally absent.

## Benchmark configs and provenance

{chr(10).join(benchmark_lines)}

The repository-level `license: other` reflects the mixed upstream terms. It does
not replace or broaden any benchmark license.

## Load annotations

Hugging Face Datasets can load one benchmark config at a time without downloading
media:

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}", "segmentation")
sample = dataset["mevis"][0]
print(sample["problem"], sample["videos"])
```

## Prepare assets and evaluate

Download this index, obtain the corresponding benchmark media from the upstream
links above, and arrange it according to the repository-relative paths declared
in the annotations. The OraRL evaluator keeps metadata and licensed assets
separate:

```bash
hf download {repo_id} \\
  --repo-type dataset \\
  --revision main \\
  --local-dir ./OraRL-Data

orarl-eval \\
  --dataset ./OraRL-Data \\
  --asset-root /path/to/local/OraRL-Eval-assets \\
  --model /path/to/Video-ORA-9B \\
  --tasks paper \\
  --summary ./orarl-eval-summary.json \\
  --run
```

The released preprocessing profiles are protocol-specific. Changing frame
sampling, video reader, resolution, prompt templates, answer parsing, or
aggregation can change benchmark scores and should be reported as a different
setting. Video segmentation uses Decord, matching the archived paper evaluation.

## Canonical row schema

Rows use the fields documented by the OraRL evaluation schema, including
`problem`, `answer`, `images`, `videos`, `problem_type`, `source`, `family`, and
task-specific `task_payload`, `metadata`, and `evaluation` objects. Media values
are logical paths resolved against a separately prepared `--asset-root`.

## Intended use and limitations

OraRL-Data supports reproducible evaluation across temporal grounding, tracking,
segmentation, spatial grounding, spatial-temporal grounding, video question
answering, and spatial intelligence. It inherits annotation errors, coverage
gaps, biases, and potentially sensitive content from the upstream benchmarks.
Users are responsible for all access conditions, privacy requirements, and
licenses.

## Citation

```bibtex
@article{{li2026orarl,
  title   = {{Annotations as Rollouts: Efficient and Scalable
             Reinforcement Learning for Video MLLMs}},
  author  = {{Li, Yunheng and Mu, Guohong and Li, Hao and
             Qian, Shengsheng and Zhang, Dingwen and Hou, Qibin
             and Cheng, Ming-Ming}},
  journal = {{arXiv preprint arXiv:2608.20492}},
  year    = {{2026}},
  url     = {{https://arxiv.org/abs/2608.20492}}
}}
```

Please also cite the original benchmark sources when using their annotations or
reporting results.
"""
    return f"---\n{yaml_header}\n---\n\n{body}"


def render_gitattributes(
    datasets: Iterable[Mapping[str, Any]],
    *,
    path_prefix: str | None = None,
) -> str:
    """Render deterministic LFS rules for all declared data paths."""

    records = validate_dataset_manifest(datasets)
    prefix = path_prefix.strip("/") if path_prefix else ""
    patterns: set[str] = set()
    for record in records:
        for path in (*record["media_paths"], *record["artifact_paths"]):
            pattern = f"{path}/**"
            patterns.add(f"{prefix}/{pattern}" if prefix else pattern)
    lines = ["# Generated by orarl-eval-data; do not edit."]
    lines.extend(f"{pattern} filter=lfs diff=lfs merge=lfs -text" for pattern in sorted(patterns))
    return "\n".join(lines) + "\n"


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _repository_records(
    repository_root: str | os.PathLike[str],
) -> tuple[Path, list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    root = Path(repository_root).expanduser().resolve()
    datasets = load_dataset_manifest(root)
    assets = _read_jsonl(root / ASSET_MANIFEST_FILENAME)
    return root, datasets, assets


def write_huggingface_metadata(
    repository_root: str | os.PathLike[str],
    *,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
) -> dict[str, Any]:
    """Generate ``README.md`` and ``.gitattributes`` from staged manifests."""

    root, datasets, assets = _repository_records(repository_root)
    card = render_dataset_card(datasets, assets, repo_id=repo_id)
    attributes = render_gitattributes(
        datasets,
        path_prefix=(
            PUBLIC_EVAL_DATA_DIRECTORY
            if repo_id == DEFAULT_INDEX_REPO_ID
            else None
        ),
    )
    _atomic_write_text(root / DATASET_CARD_FILENAME, card)
    _atomic_write_text(root / GIT_ATTRIBUTES_FILENAME, attributes)
    return {
        "readme": DATASET_CARD_FILENAME,
        "gitattributes": GIT_ATTRIBUTES_FILENAME,
        "configs": len(dataset_card_subsets(datasets)),
    }


def validate_huggingface_metadata(
    repository_root: str | os.PathLike[str],
    *,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
) -> dict[str, Any]:
    """Require generated metadata to match the staged manifests byte-for-byte."""

    root, datasets, assets = _repository_records(repository_root)
    path_prefix = (
        PUBLIC_EVAL_DATA_DIRECTORY
        if repo_id == DEFAULT_INDEX_REPO_ID
        else None
    )
    expected = {
        DATASET_CARD_FILENAME: render_dataset_card(datasets, assets, repo_id=repo_id),
        GIT_ATTRIBUTES_FILENAME: render_gitattributes(
            datasets,
            path_prefix=path_prefix,
        ),
    }
    for name, content in expected.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise DatasetCardError(f"generated Hugging Face metadata is missing: {name}")
        if path.read_bytes() != content.encode("utf-8"):
            raise DatasetCardError(f"generated Hugging Face metadata is stale: {name}")
    return {
        "readme": DATASET_CARD_FILENAME,
        "gitattributes": GIT_ATTRIBUTES_FILENAME,
        "configs": len(dataset_card_subsets(datasets)),
    }


huggingface_dataset_configs = dataset_card_subsets
build_dataset_card_metadata = dataset_card_metadata
generate_dataset_card = render_dataset_card
generate_gitattributes = render_gitattributes


__all__ = [
    "DATASET_CARD_FILENAME",
    "DEFAULT_DATASET_REPO_ID",
    "DEFAULT_INDEX_REPO_ID",
    "DatasetCardError",
    "GIT_ATTRIBUTES_FILENAME",
    "PUBLIC_EVAL_DATA_DIRECTORY",
    "build_dataset_card_metadata",
    "dataset_card_metadata",
    "dataset_card_subsets",
    "generate_dataset_card",
    "generate_gitattributes",
    "huggingface_dataset_configs",
    "render_dataset_card",
    "render_index_card",
    "render_public_dataset_card",
    "render_gitattributes",
    "validate_huggingface_metadata",
    "write_huggingface_metadata",
]
