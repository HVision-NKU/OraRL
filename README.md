<p align="right"><a href="README_zh.md">简体中文</a></p>

<div align="center">

# OraRL

### Annotations as Rollouts

**Efficient and scalable reinforcement learning for unified video MLLMs**

Yunheng Li · Guohong Mu · Hao Li · Shengsheng Qian · Dingwen Zhang ·
Qibin Hou · Ming-Ming Cheng

<p>
  <a href="https://arxiv.org/abs/2608.20492">📄 Paper</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://orarl.github.io/">🌐 Project Page</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/OraRL/video-ora-9b-demo">🎮 Live Demo</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="#models">🤗 Models (4B / 9B)</a>
</p>
<p>
  <a href="docs/environment.md">⚙️ Environment</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="docs/training.md">🚀 Training</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="docs/evaluation.md">📊 Evaluation</a>
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="LICENSE">⚖️ License</a>
</p>

<p align="center">
  <a href="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge-link?eval=28347"><img alt="Papers with Code: SOTA on ActivityNet-TimeLens" src="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge.svg?eval=28347&amp;live=1"></a>
  <a href="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge-link?eval=28346"><img alt="Papers with Code: SOTA on Charades-TimeLens" src="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge.svg?eval=28346&amp;live=1"></a>
  <a href="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge-link?eval=28350"><img alt="Papers with Code: SOTA on MeViS" src="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge.svg?eval=28350&amp;live=1"></a>
  <a href="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge-link?eval=28348"><img alt="Papers with Code: SOTA on QVHighlights-TimeLens Validation" src="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge.svg?eval=28348&amp;live=1"></a>
  <a href="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge-link?eval=28355"><img alt="Papers with Code: SOTA on VideoHolmes" src="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge.svg?eval=28355&amp;live=1"></a>
  <a href="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge-link?eval=28351"><img alt="Papers with Code: #2 on ReasonVOS" src="https://paperswithcode.co/api/v1/papers/2608.20492/leaderboard-badge.svg?eval=28351&amp;live=1"></a>
</p>

<p align="center">
  <a href="https://paperswithcode.co/paper/2608.20492#results">View all 11 verified evaluations on Papers with Code</a>
</p>

<a href="https://orarl.github.io/assets/orarl-teaser.mp4">
  <img src="assets/orarl-hero.gif"
       alt="Animated OraRL method preview" width="92%">
</a>

**▶ Click the image to watch the 1:38 project overview.**

</div>

## Why OraRL

- **Annotation-as-rollout:** annotations become reliable positive rollouts while
  policy samples retain an on-policy baseline.
- **Seven task families:** one update rule covers temporal and spatial grounding,
  segmentation, tracking, spatial-temporal grounding, video QA, and spatial
  intelligence.
- **Efficient training (4B):** sign-balanced pruning delivers **1.48× faster
  updates** (**92.5 → 62.4 s/step**) while reducing peak per-GPU memory from
  **62.4 to 50.9 GB**.
- **Efficient inference:** on one H20 with vLLM in BF16, weight loading occupies
  **8.6 GiB (4B)** and **17.6 GiB (9B)**. On ten-minute, 2-fps videos,
  answer-only decoding cuts median post-TTFT latency from **4.78 s to 130 ms**
  and total latency from **29.03 to 24.30 s**.
- **Multimodal veRL infrastructure:** a unified video contract carries cached
  artifacts, raw paths, or inline frame tensors through vLLM rollouts and FSDP
  updates, with decode-once frame reuse, temporal metadata, task-grouped
  batching, asynchronous Ray rewards, and safe hybrid-engine cache handling.

## OraRL in One Update

<p align="center">
  <img src="assets/orarl-method.gif"
       alt="Animated OraRL framework" width="96%">
</p>

An OraRL update separates reliable annotation guidance from on-policy
normalization:

1. **Build the group:** append one serialized annotation rollout to the policy
   samples generated for the same prompt.
2. **Keep the baseline on-policy:** estimate the group baseline from policy
   rewards only.
3. **Guide and select:** convert the annotation-policy reward gap into a
   correction, then retain a sign-balanced subset for the update.

This design uses task-native annotations directly and requires no
chain-of-thought supervision or decoding.

## Video-ORA Results

<p align="center">
  <img src="assets/paper-results.png"
       alt="Video-ORA-9B results across seven task families" width="100%">
</p>

### Dataset-Level Results

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="assets/video_ora_benchmark_matrix_dark.svg">
  <source media="(prefers-color-scheme: light)"
          srcset="assets/video_ora_benchmark_matrix_light.svg">
  <img src="assets/video_ora_benchmark_matrix_light.svg"
       alt="Dataset-level benchmark matrix comparing Video-ORA with multimodal baselines"
       width="100%">
</picture>

Video-ORA-9B leads the matched seven-family comparison without CoT decoding.
Best and second-best values are highlighted per row; `†` denotes an
original-report value whose frame, prompt, split, or decoding settings may
differ. Averages require complete family coverage.

<!-- <details>
<summary>Benchmark sources</summary>

Unmarked values come from Tables 1–8 and Appendix Table 20 of the latest
[OraRL paper](https://arxiv.org/abs/2608.20492). External entries follow the original
[LLaVA-OneVision-2](https://arxiv.org/abs/2605.25979),
[VideoChat3](https://github.com/MCG-NJU/VideoChat3), and
[OneThinker](https://arxiv.org/abs/2512.03043) reports. OneThinker is cited only
as the source of public Qwen3-VL scores. ReVSI uses each model's reported frame
setting; the paper's three-benchmark spatial-intelligence average excludes it.

</details> -->

### Model Scaling

<p align="center">
  <img src="assets/orarl-model-scaling.gif"
       alt="Animated Video-ORA model scaling from 0.8B to 9B" width="100%">
</p>

### Data Scaling

<p align="center">
  <img src="assets/orarl-data-scaling.gif"
       alt="Animated OraRL data scaling and reward dynamics" width="100%">
</p>

## Models

| Model | Backbone | Released recipe | Weights |
| --- | --- | --- | --- |
| **Video-ORA-9B** | Qwen3.5-9B | `orarl_9b.yaml` | [Hugging Face](https://huggingface.co/OraRL/Video-ORA-9B) |
| **Video-ORA-4B** | Qwen3.5-4B | `orarl_4b.yaml` | [Hugging Face](https://huggingface.co/OraRL/Video-ORA-4B) |

### vLLM Serving

Both Video-ORA checkpoints load directly with **vLLM 0.19.1** for
OpenAI-compatible serving:

```bash
MODEL=OraRL/Video-ORA-9B

vllm serve "$MODEL" \
  --served-model-name Video-ORA-9B \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --limit-mm-per-prompt '{"image": 1, "video": 1}'
```

Set `--tensor-parallel-size` to the GPU count for multi-GPU deployment and
lower `--max-model-len` on smaller-memory devices. Use
`enable_thinking=false` in the chat template for answer-only inference.

## Use OraRL

The release is organized around three user-facing workflows:

1. **[Environment](docs/environment.md):** install the pinned CUDA stack that
   covers both the bundled trainer and the evaluators.
2. **[Training](docs/training.md):** prepare licensed local training data and
   launch GRPO or OraRL on one or multiple nodes.
3. **[Evaluation](docs/evaluation.md):** download Video-ORA and OraRL-Data,
   then run a smoke test or the complete paper suite.

Training and evaluation are dry runs by default; inspect the resolved command
before adding `--run`. Checkpoints and evaluation media are hosted under the
[OraRL Hugging Face organization](https://huggingface.co/OraRL).

## Acknowledgements

OraRL is built on [veRL](https://github.com/volcengine/verl) — a
high-performance RL framework with HybridEngine. We thank its authors and
contributors for open-sourcing the training infrastructure.

## License

OraRL source is released under [Apache-2.0](LICENSE). Datasets, models,
benchmarks, and optional dependencies retain their original licenses; see
[NOTICE](NOTICE).

## Citation

If you find OraRL useful, please consider giving this repository a ⭐ and
citing our [paper](https://arxiv.org/abs/2608.20492).

```bibtex
@article{li2026orarl,
  title   = {Annotations as Rollouts: Efficient and Scalable
             Reinforcement Learning for Video MLLMs},
  author  = {Li, Yunheng and Mu, Guohong and Li, Hao and
             Qian, Shengsheng and Zhang, Dingwen and Hou, Qibin
             and Cheng, Ming-Ming},
  journal = {arXiv preprint arXiv:2608.20492},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.20492}
}
```
