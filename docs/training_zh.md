<p align="right"><a href="training.md">English</a></p>

# 使用 OraRL 训练

本流程将已获许可的源数据整理为可审计的 GRPO 或 OraRL 训练任务。请先
[安装固定版本软件栈](environment_zh.md#安装固定版本软件栈)。

`orarl-train` 是公开的训练入口：它会检查路径与覆盖项、解析发布配置，再启动
本仓库自带的训练器，无需准备第二份运行时源码。

## 1. 构建训练清单

整理后的训练数据将稍后单独上传，目前尚未包含在本 Git 仓库中。在发布完成前，
请按上游许可证自行获取标注和媒体文件，然后复制示例清单：

```bash
cp configs/data_sources.example.yaml ./data_sources.local.yaml
```

将所有 `../local_data` 占位符替换为已获许可的本地路径。相对路径以清单文件所在
目录为基准。每个数据源需要声明标注文件、任务、任务族、配额和媒体根目录，并可
记录许可证与来源页面。

公开示例对应论文使用的 100,032 条训练混合：

| 任务族 | 条数 |
| --- | ---: |
| 时序定位 | 20,096 |
| 跟踪 | 13,952 |
| 分割 | 12,032 |
| 空间定位 | 7,040 |
| 时空定位 | 9,536 |
| 视频问答 | 20,288 |
| 空间智能 | 17,088 |

构建确定性的训练集和 canary 集：

```bash
orarl-prepare \
  --config ./data_sources.local.yaml \
  --output ./prepared/train.jsonl \
  --require-media
```

输出结构为：

```text
prepared/
├── train.jsonl
├── train.canary.jsonl
└── train.manifest.json
```

构建器会检查本地媒体、规范化任务记录、执行数据源配额和单媒体上限、去除重复
prompt、排除给定评测集中的样本，并保证训练集与 canary 集媒体互斥。审计清单会
记录计数、缺额、来源信息和 SHA-256 校验值。

## 2. 选择 GRPO 或 OraRL

| 配置 | 模型规模 | 方法 |
| --- | --- | --- |
| `grpo_4b.yaml` | 4B | GRPO baseline |
| `grpo_9b.yaml` | 9B | GRPO baseline |
| `orarl_4b.yaml` | 4B | OraRL |
| `orarl_9b.yaml` | 9B | OraRL |

论文默认每个 rollout/update batch 使用 64 个 prompt，每个 prompt 采样 8 个策略
回复，因此 100,032 条混合数据在一个 epoch 中对应 1,563 步。

## 3. 预览并启动

设置兼容的本地基础模型和已准备的数据路径：

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

命令默认只执行 dry run。检查解析后的调用后，增加 `--run` 开始训练。可通过
`--set KEY=VALUE` 显式覆盖配置，并应随运行产物保留全部覆盖项。

单步 smoke test：

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

再使用 `grpo_4b.yaml` 和独立输出目录验证 baseline。9B 任务需使用匹配的模型与
配置。

## 4. 扩展到多节点

所有节点必须看到相同的源码、模型、数据和输出路径：

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

启动器同样默认为 dry run；需要在 `--` 分隔符前加入其自身的 `--run`。SSH
主机密钥检查默认使用严格模式。

## 5. 验收一次运行

`scripts/smoke_training.sh` 会以较小的 batch 分别跑一步 GRPO 和一步 OraRL，
并各保存一个 checkpoint：

```bash
bash scripts/smoke_training.sh \
  --model "$MODEL_DIR" \
  --train-data "$PWD/prepared/train.jsonl" \
  --val-data "$PWD/prepared/train.canary.jsonl" \
  --size 4b \
  --gpus-per-node 8
```

加上 `--dry-run` 可在不占用 GPU 的情况下检查解析后的命令。

正式实验前确认：

- GRPO 和 OraRL 均可完成一步更新，reward、loss、梯度范数和选择指标均为有限值。
- checkpoint 可保存、重新加载并继续完成一步更新。
- 多节点任务能组成预期的 Ray 集群并完成一步更新。
- 保存源码版本、配置、命令、数据清单校验值、环境版本、加速卡类型和所有覆盖项。

使用[评测文档](evaluation_zh.md)评估导出的 checkpoint。
