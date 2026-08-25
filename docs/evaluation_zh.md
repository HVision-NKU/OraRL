<p align="right"><a href="evaluation.md">English</a></p>

# 评测 Video-ORA

OraRL 使用七个发布任务族的规范配置评测任务原生答案，对应运行时位于
`eval/task/`。运行前请先
[验证评测节点](environment_zh.md#验证仅用于评测的节点)。

## 1. 准备模型和数据

安装 Hugging Face CLI 并下载发布文件：

```bash
python -m pip install -U huggingface_hub

hf download OraRL/Video-ORA-9B \
  --local-dir "$PWD/models/Video-ORA-9B"

hf download OraRL/OraRL-Data \
  --repo-type dataset \
  --include "OraRL-eval-data/**" \
  --local-dir "$PWD/OraRL-Data"
```

完整评测数据体积较大，下载支持断点续传；
`OraRL-eval-data/assets.jsonl` 是权威文件清单。

本地数据结构为：

```text
OraRL-Data/OraRL-eval-data/
├── datasets.jsonl   # benchmark、prompt、parser、metric 与采样配置
├── assets.jsonl     # 发布文件清单
├── annotations/     # 规范化 JSONL 标注
└── media/           # 原始图像、视频和字幕
```

发布数据有意排除衍生预处理缓存；不存在兼容缓存时，评测器会直接解码清单中声明的
原始媒体。

## 2. 运行论文规范配置

先预览解析后的命令：

```bash
orarl-eval \
  --model "$PWD/models/Video-ORA-9B" \
  --tasks paper \
  --dataset "$PWD/OraRL-Data/OraRL-eval-data" \
  --summary "$PWD/outputs/Video-ORA-9B/evaluation.json"
```

`orarl-eval` 默认为 dry run。检查模型、数据、评测器、任务配置和输出路径后，
增加 `--run`。

可先运行限制样本数的 smoke test：

```bash
orarl-eval \
  --model "$PWD/models/Video-ORA-9B" \
  --tasks videomme \
  --dataset "$PWD/OraRL-Data/OraRL-eval-data" \
  --max-samples 8 \
  --summary "$PWD/outputs/Video-ORA-9B/videomme-smoke.json" \
  --run
```

Smoke test 分数仅用于验证流程，不应作为 benchmark 结果报告。

## 3. 组合任务集

`--tasks paper` 选择所有发布任务，`--tasks video_qa` 选择 7 个视频问答
benchmark；多个单项任务可用逗号分隔。

| 任务族 | 任务名 |
| --- | --- |
| 视频问答 | `videomme`, `videommev2`, `mvbench`, `mmvu`, `videoholmes`, `longvideobench`, `mlvu` |
| 空间智能 | `vsi`, `mmsi`, `mindcube`, `revsi` |
| 时序定位 | `temporal_grounding` |
| 空间定位 | `spatial_grounding` |
| 跟踪 | `tracking` |
| 时空定位 | `stvg` |
| 分割 | `segmentation` |

`datasets.jsonl` 中的规范配置固定了帧采样、分辨率、prompt、parser 和 metric；
修改这些值即代表不同的评测设置。ReVSI 单独报告，不计入三个 benchmark 的空间
智能平均分。

## 4. 添加分割后处理

分割推理默认不执行 SAM2 后处理。启用 `--segmentation-run-sam2` 时还需要：

- SAM2 权重（`SEGMENTATION_SAM2_CKPT`）
- 匹配的 Hydra 配置（`SEGMENTATION_SAM2_CFG`）
- OneThinker 官方 `seg_post_sam2.py`
  （`SEGMENTATION_POSTPROCESSOR_PATH`）
- `sam2` Python 包

OraRL 会在启动前检查三个路径。

## 5. 保存可报告的输出

每个任务评测器先写入原生 summary，随后 `orarl-eval` 生成指定的聚合 JSON，
其中包含请求、完成和缺失的任务、返回码以及官方 metric。输出按论文任务族组织在
`outputs/<model>/` 下。

正式报告结果时请保留：

- OraRL 源码版本
- 模型和数据版本
- 完整命令与聚合 summary
- 软件版本和加速卡类型/数量
- 所有非默认配置或命令行覆盖项

评测器映射、视频解码后端和 checkpoint 格式说明见
[`../eval/README.md`](../eval/README.md)。
