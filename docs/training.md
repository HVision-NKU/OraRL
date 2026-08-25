<p align="right"><a href="training_zh.md">简体中文</a></p>

# Train with OraRL

This workflow turns licensed source records into an auditable GRPO or OraRL
run. Complete the
[pinned environment installation](environment.md#install-the-pinned-stack)
first.

`orarl-train` is the public launch boundary: it validates paths and overrides,
resolves one of the released recipes, and then starts the trainer bundled with
this checkout. No second runtime repository is required.

## 1. Build the training manifest

The prepared training-data release will be uploaded separately and is not
included in this Git repository yet. Until it is available, obtain each
annotation and media source under its upstream license, then copy the example
manifest:

```bash
cp configs/data_sources.example.yaml ./data_sources.local.yaml
```

Replace every `../local_data` placeholder with a licensed local path. Relative
paths are resolved from the manifest location. Each source declares its input
annotations, task, family, quota, media root, and optional license/source-page
metadata.

The public example reproduces the 100,032-row paper mixture:

| Family | Rows |
| --- | ---: |
| Temporal grounding | 20,096 |
| Tracking | 13,952 |
| Segmentation | 12,032 |
| Spatial grounding | 7,040 |
| Spatial-temporal grounding | 9,536 |
| Video QA | 20,288 |
| Spatial intelligence | 17,088 |

Build the deterministic train/canary split:

```bash
orarl-prepare \
  --config ./data_sources.local.yaml \
  --output ./prepared/train.jsonl \
  --require-media
```

This writes:

```text
prepared/
├── train.jsonl
├── train.canary.jsonl
└── train.manifest.json
```

The builder validates local media, normalizes task records, enforces source
quotas and per-media caps, removes duplicate prompt identities, excludes
supplied benchmark identities, and keeps train/canary media disjoint. The audit
manifest records counts, shortfalls, source metadata, and SHA-256 checksums.

## 2. Choose GRPO or OraRL

| Recipe | Model scale | Method |
| --- | --- | --- |
| `grpo_4b.yaml` | 4B | GRPO baseline |
| `grpo_9b.yaml` | 9B | GRPO baseline |
| `orarl_4b.yaml` | 4B | OraRL |
| `orarl_9b.yaml` | 9B | OraRL |

The paper defaults use 64 prompts per rollout/update batch and eight policy
samples per prompt. The 100,032-row mixture therefore runs for 1,563 steps in
one epoch.

## 3. Preview, then launch

Set paths to a compatible local base model and the prepared data:

```bash
MODEL_DIR=/path/to/local/base-model
OUTPUT_DIR="$PWD/runs/orarl-4b"

orarl-train \
  --config orarl_4b.yaml \
  --model "$MODEL_DIR" \
  --train-data "$PWD/prepared/train.jsonl" \
  --val-data "$PWD/prepared/train.canary.jsonl" \
  --output "$OUTPUT_DIR" \
  --nodes 1 \
  --gpus-per-node 8
```

The command is a dry run by default. Inspect the resolved invocation, then add
`--run` to start training. Use `--set KEY=VALUE` for an explicit config
override; retain all overrides with the run artifacts.

For a one-update smoke test:

```bash
orarl-train \
  --config orarl_4b.yaml \
  --model "$MODEL_DIR" \
  --train-data "$PWD/prepared/train.jsonl" \
  --val-data "$PWD/prepared/train.canary.jsonl" \
  --output "$OUTPUT_DIR" \
  --nodes 1 \
  --gpus-per-node 8 \
  --set trainer.max_steps=1 \
  --run
```

Repeat with `grpo_4b.yaml` and a different output directory to validate the
baseline. Use the matching model and recipe for 9B runs.

## 4. Scale across nodes

All nodes must see the same source, model, data, and output paths:

```bash
HOSTS=node-a,node-b \
  bash scripts/launch_multinode.sh \
  --gpus-per-node 8 \
  -- \
  --config "$PWD/configs/orarl_4b.yaml" \
  --model "$MODEL_DIR" \
  --train-data "$PWD/prepared/train.jsonl" \
  --val-data "$PWD/prepared/train.canary.jsonl" \
  --output "$OUTPUT_DIR"
```

The launcher is also a dry run unless its own `--run` is supplied before the
`--` separator. SSH host-key checking is strict by default.

## 5. Accept a run

`scripts/smoke_training.sh` runs one GRPO update and one OraRL update with small
batches and saves a checkpoint for each:

```bash
bash scripts/smoke_training.sh \
  --model "$MODEL_DIR" \
  --train-data "$PWD/prepared/train.jsonl" \
  --val-data "$PWD/prepared/train.canary.jsonl" \
  --size 4b \
  --gpus-per-node 8
```

Add `--dry-run` to inspect the resolved commands without allocating GPUs.

Before a full experiment, verify that:

- GRPO and OraRL each complete one update with finite rewards, losses, gradient
  norms, and selection metrics.
- A checkpoint can be saved, reloaded, and used for another update.
- Multi-node runs form the expected Ray cluster and complete one update.
- The source revision, config, command, data-manifest checksum, environment
  versions, accelerator type, and all overrides are retained.

Use [Evaluation](evaluation.md) to evaluate an exported checkpoint.
