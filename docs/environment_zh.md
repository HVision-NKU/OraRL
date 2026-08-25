<p align="right"><a href="environment.md">English</a></p>

# 复现 OraRL 环境

OraRL 使用同一套可复现软件栈完成策略优化与任务原生评测。发布配置在 NVIDIA
H20/Hopper 与 CUDA 12.9 上完成验证，本仓库同时包含两部分运行时。

## 环境要求

- `PATH` 中可用 Conda 的 Linux 系统
- 驱动版本较新的 NVIDIA GPU
- 复现论文环境时使用兼容 CUDA 12.9 的硬件

CUDA 12.9 GA 官方要求 NVIDIA Linux 驱动不低于 575.51.03。其他 NVIDIA GPU
也可能运行，但发布配置仅在 H20 上完成验证。

## 安装固定版本软件栈

```bash
bash scripts/create_conda_env.sh
conda activate orarl

python scripts/check_environment.py \
  --require-gpu \
  --model /path/to/local/model
```

脚本会固定 Python 3.11、PyTorch 2.10.0+cu129、Transformers 5.5.4、
vLLM 0.19.1、FlashAttention 2.8.3，以及 `requirements-cu129.txt` 中的其余依赖。
最后一步的可编辑安装会同时提供 `orarl-*` 命令和本仓库自带的 `verl` 训练包；
请勿再用 PyPI 上无关的 `verl` 版本覆盖它。

## 验证仅用于评测的节点

若某个节点只跑评测，可跳过训练侧检查：

```bash
bash scripts/create_conda_env.sh --evaluation-only
conda activate orarl

python scripts/check_environment.py --evaluation-only
```

## 验证发布源码

分配长任务前先执行：

```bash
python scripts/check_release.py
python -m pytest -q
ruff check .
```

在与论文一致的 H20 节点上，可为 `scripts/check_environment.py` 增加
`--require-gpu --require-h20`。多节点任务应在所有节点使用相同的源码版本和环境。

随后参阅[训练](training_zh.md)或[评测](evaluation_zh.md)。
