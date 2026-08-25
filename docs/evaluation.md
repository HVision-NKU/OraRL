<p align="right"><a href="evaluation_zh.md">简体中文</a></p>

# Evaluate Video-ORA

OraRL evaluates direct, task-native answers with canonical profiles for the
seven released task families. The corresponding runtime lives under
`eval/task/`; first
[validate the evaluation node](environment.md#validate-an-evaluation-only-node).

## 1. Materialize model and data

Install the Hugging Face CLI and download the released assets:

```bash
python -m pip install -U huggingface_hub

hf download OraRL/Video-ORA-9B \
  --local-dir "$PWD/models/Video-ORA-9B"

hf download OraRL/OraRL-Data \
  --repo-type dataset \
  --include "OraRL-eval-data/**" \
  --local-dir "$PWD/OraRL-Data"
```

The complete evaluation release is large. Downloads are resumable, and
`OraRL-eval-data/assets.jsonl` is the authoritative file inventory.

The local dataset layout is:

```text
OraRL-Data/OraRL-eval-data/
├── datasets.jsonl   # benchmark, prompt, parser, metric, and sampling profiles
├── assets.jsonl     # released-file inventory
├── annotations/     # canonical JSONL rows
└── media/           # raw images, videos, and subtitles
```

Derived preprocessing caches are intentionally excluded. Evaluators decode the
declared raw media when no compatible cache is present.

## 2. Run the canonical paper profile

Preview the resolved command first:

```bash
orarl-eval \
  --model "$PWD/models/Video-ORA-9B" \
  --tasks paper \
  --dataset "$PWD/OraRL-Data/OraRL-eval-data" \
  --summary "$PWD/outputs/Video-ORA-9B/evaluation.json"
```

`orarl-eval` is a dry run by default. After checking the model, dataset,
evaluator, task profiles, and output paths, add `--run`.

To validate the pipeline with a bounded smoke test:

```bash
orarl-eval \
  --model "$PWD/models/Video-ORA-9B" \
  --tasks videomme \
  --dataset "$PWD/OraRL-Data/OraRL-eval-data" \
  --max-samples 8 \
  --summary "$PWD/outputs/Video-ORA-9B/videomme-smoke.json" \
  --run
```

Smoke scores only validate execution and must not be reported as benchmark
results.

## 3. Compose a task suite

`--tasks paper` selects all released tasks. `--tasks video_qa` selects the
seven Video QA benchmarks. Individual task names may be comma-separated.

| Family | Task names |
| --- | --- |
| Video QA | `videomme`, `videommev2`, `mvbench`, `mmvu`, `videoholmes`, `longvideobench`, `mlvu` |
| Spatial intelligence | `vsi`, `mmsi`, `mindcube`, `revsi` |
| Temporal grounding | `temporal_grounding` |
| Spatial grounding | `spatial_grounding` |
| Tracking | `tracking` |
| Spatial-temporal grounding | `stvg` |
| Segmentation | `segmentation` |

The canonical profiles in `datasets.jsonl` pin frame sampling, resolution,
prompts, parsers, and metrics. Changing them defines a different evaluation
setting. ReVSI is reported separately from the three-benchmark
spatial-intelligence average.

## 4. Add segmentation post-processing

Segmentation inference runs without SAM2 post-processing by default. Enabling
`--segmentation-run-sam2` additionally requires:

- SAM2 weights (`SEGMENTATION_SAM2_CKPT`)
- the matching Hydra config (`SEGMENTATION_SAM2_CFG`)
- the official OneThinker `seg_post_sam2.py`
  (`SEGMENTATION_POSTPROCESSOR_PATH`)
- the `sam2` Python package

OraRL validates all three paths before launch.

## 5. Preserve reportable outputs

Each task evaluator writes its native summary, then `orarl-eval` creates the
requested aggregate JSON with requested, completed, and missing tasks, the
return code, and official metrics. Outputs are grouped by paper task family
under `outputs/<model>/`.

For a reportable run, retain:

- the OraRL source revision
- the model and data revisions
- the exact command and aggregate summary
- software versions and accelerator type/count
- every non-default profile or CLI override

See [`../eval/README.md`](../eval/README.md) for the evaluator map, video
decoding backend, and checkpoint-format notes.
