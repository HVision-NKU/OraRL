# Evaluation runtime

`eval/task/` holds the evaluators that produce every number in the paper. They
ship with this repository, so a single clone can reproduce the full suite.
`orarl-eval` is the supported entry point: it resolves the canonical dataset
layout into per-task environment variables and then runs `eval/task/eval.sh`.

```bash
orarl-eval \
  --model /path/to/exported-model \
  --tasks paper \
  --dataset OraRL/OraRL-Data \
  --summary ./evaluation.json \
  --run
```

Without `--run` the command prints the resolved plan and exits, which is the
fastest way to confirm data and model paths before allocating GPUs.
`eval/task/eval.sh` can also be called directly when you want to bypass the
canonical dataset resolution and set the per-task variables yourself.

## Task map

| Family | Benchmarks | Entry point |
| --- | --- | --- |
| Video QA | VideoMME, VideoMME-v2, MVBench, MMVU, Video-Holmes, LongVideoBench, MLVU | `eval_vllm.py` |
| Spatial intelligence | VSI-Bench, MindCube, MMSI-Bench, ReVSI | `eval_vllm.py`, `mmsi/eval_mmsi_transformers.py`, `revsi/eval_revsi_vllm.py` |
| Temporal grounding | TimeLens (Charades, ActivityNet, QVHighlights) | `temporal_grounding/eval_timelens_hf.py` |
| Spatial grounding | RefCOCO, RefCOCO+, RefCOCOg | `spatial_grounding/eval_refcoco_vllm.py` |
| Tracking | GOT-10k | `tracking/eval_tracking_vllm.py` |
| Spatio-temporal grounding | STVG | `spatial_temporal_grounding/eval_stvg_vllm.py` |
| Segmentation | RefCOCO series, MeViS, ReasonVOS | `segmentation/eval_seg_vllm.py` plus `segmentation/post_sam2.py` |

`eval_prompt.py` is the single source of truth for prompts, and
`canonical_data.py` adapts the canonical `ORARL_EVAL_*` layout for every
evaluator. MMSI-Bench runs through Transformers rather than vLLM; the other
families run through vLLM.

## Assets you must obtain separately

Annotations and media come from the `OraRL/OraRL-Data` dataset repository, and
model weights come from the released checkpoints. Neither is vendored here.

Segmentation additionally needs three inputs, all supplied by you under their
upstream licenses:

- SAM2 weights and the matching Hydra config, passed as
  `SEGMENTATION_SAM2_CKPT` and `SEGMENTATION_SAM2_CFG`.
- The official OneThinker `seg_post_sam2.py`, passed as
  `SEGMENTATION_POSTPROCESSOR_PATH`. `segmentation/post_sam2.py` is a thin
  wrapper that injects paths into it so mask metrics stay identical to the
  upstream implementation.
- The `sam2` Python package.

`orarl-eval` refuses to start segmentation with `--segmentation-run-sam2`
unless all three paths exist.

## Video decoding

Video evaluators pin the decord backend through `FORCE_QWENVL_VIDEO_READER`.
`eval/task/qwenvl_decord_patch.py` replaces `qwen_vl_utils.fetch_video` so the
backend is honoured across `qwen_vl_utils` releases that otherwise hard-code
torchvision. The backend is not cosmetic: switching MeViS and ReasonVOS from
torchcodec to decord moved MeViS J&F from 56.7 to 60.6. Set
`SEGMENTATION_VIDEO_READER` to `torchcodec` or `torchvision` only when
deliberately measuring that difference.

## Checkpoint format

Exported Hugging Face checkpoints run as-is. Pass `--force-merge` when the model
path is a sharded FSDP actor directory; `eval/task/eval.sh` then merges it with
`scripts/model_merger.py` before inference.
