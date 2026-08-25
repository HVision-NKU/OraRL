# Evaluation profiles

This directory contains only the small, reviewable `datasets.jsonl` manifest
that pins the paper-suite splits, prompts, preprocessing settings, parsers, and
metrics.

Evaluation annotations and raw media are hosted separately at
[`OraRL/OraRL-Data`](https://huggingface.co/datasets/OraRL/OraRL-Data).
Model checkpoints are hosted under the
[`OraRL`](https://huggingface.co/OraRL) organization. They are intentionally not
duplicated in this Git repository.

Use the Hugging Face dataset ID or a complete local snapshot when evaluating:

```bash
orarl-eval \
  --model /path/to/Video-ORA-9B \
  --tasks paper \
  --dataset OraRL/OraRL-Data \
  --summary ./evaluation.json \
  --run
```

The manifest in this directory is release metadata, not a standalone dataset
root; it does not include `assets.jsonl` or media payloads.
