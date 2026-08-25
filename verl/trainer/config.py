# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.multimodal_contract import normalize_video_source_mode
from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    video_max_frames: int = 128
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    enable_thinking: bool = False
    """forwarded to the tokenizer/processor chat template; False preserves no-CoT runs"""
    response_prefix: str = ""
    """literal assistant prefix appended after the generation prompt, e.g. '<think>\\n'"""
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    image_min_pixels: Optional[int] = None
    image_max_pixels: Optional[int] = None
    video_min_pixels: Optional[int] = None
    video_max_pixels: Optional[int] = None
    video_total_pixels: Optional[int] = None
    val_video_fps: Optional[float] = None
    val_video_max_frames: Optional[int] = None
    val_video_min_pixels: Optional[int] = None
    val_video_max_pixels: Optional[int] = None
    val_video_total_pixels: Optional[int] = None
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16
    use_preprocessed_videos: bool = True
    """deprecated compatibility flag; prefer video_source_mode"""
    video_source_mode: Optional[str] = None
    """video source policy: prefer_preprocessed, preprocessed_only, realtime_only"""
    preprocessed_video_dir: Optional[str] = None
    """directory containing training preprocessed video files (.pt)"""
    val_preprocessed_video_dir: Optional[str] = None
    """directory containing validation preprocessed video files (.pt); defaults to preprocessed_video_dir"""
    val_video_source_mode: Optional[str] = None
    """validation video source policy; defaults to video_source_mode"""
    inline_video_tensors: bool = False
    """if True, decode the video once in the dataset and pass the decoded frames/metadata
    inline through multi_modal_data so vLLM rollout and FSDP worker forward passes do
    not re-decode the same mp4 (or reload the same .pt). Saves CPU at the cost of
    larger pickled batches; recommended for realtime-decode runs."""
    group_by_task: bool = False
    """if True, each batch contains samples from a single task type only
    (determined by the `group_by_task_key` field in the JSONL).  Avoids mixing
    modalities (image vs video) within a batch and concentrates gradients."""
    group_by_task_key: str = "problem_type"
    """JSONL field name used to identify the task type for task-grouped batching."""
    dataloader_num_workers: int = 8
    """number of subprocesses used by the training/validation StatefulDataLoader.
    Controls how many videos are decoded in parallel per training step."""

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")
        self.preprocessed_video_dir = get_abs_path(self.preprocessed_video_dir, prompt="Preprocessed video directory")
        self.val_preprocessed_video_dir = get_abs_path(self.val_preprocessed_video_dir, prompt="Validation preprocessed video directory")
        if self.image_min_pixels is None:
            self.image_min_pixels = self.min_pixels
        if self.image_max_pixels is None:
            self.image_max_pixels = self.max_pixels
        if self.video_min_pixels is None:
            self.video_min_pixels = self.min_pixels
        if self.video_max_pixels is None:
            self.video_max_pixels = self.max_pixels
        if self.val_video_fps is None:
            self.val_video_fps = self.video_fps
        if self.val_video_max_frames is None:
            self.val_video_max_frames = self.video_max_frames
        if self.val_video_min_pixels is None:
            self.val_video_min_pixels = self.video_min_pixels
        if self.val_video_max_pixels is None:
            self.val_video_max_pixels = self.video_max_pixels
        if self.val_video_total_pixels is None:
            self.val_video_total_pixels = self.video_total_pixels
        if self.val_preprocessed_video_dir is None:
            self.val_preprocessed_video_dir = self.preprocessed_video_dir
        self.video_source_mode = normalize_video_source_mode(
            self.video_source_mode,
            use_preprocessed_videos=self.use_preprocessed_videos,
        )
        if self.val_video_source_mode is None:
            self.val_video_source_mode = self.video_source_mode
        else:
            self.val_video_source_mode = normalize_video_source_mode(
                self.val_video_source_mode,
                use_preprocessed_videos=self.use_preprocessed_videos,
            )


@dataclass
class AlgorithmConfig:
    name: str = "grpo"
    """Algorithm selector; the released recipes are ``grpo`` and ``orarl``."""
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    scale_rewards: bool = True
    """Whether GRPO divides each group-centered outcome reward by the group's
    reward standard deviation. True preserves the historical normalized GRPO
    objective. False uses the TRL raw-centered objective
    `A_i = r_i - mean_group(r)`, avoiding amplification of tiny reward gaps."""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""

    # OraRL stages, in the order the paper applies them.
    oracle_injection: bool = False
    """Append one annotation-derived oracle response to each rollout group."""
    oracle_injection_mode: str = "append"
    """Oracle insertion mode. The released OraRL recipe requires ``append``."""
    oracle_builder: Optional[str] = None
    """Dotted ``module:function`` that constructs an oracle response."""
    oracle_replace_index: int = -1
    """Compatibility slot for non-released replacement mode."""
    oracle_log_exclude: bool = True
    """Exclude oracle rows from on-policy reward/accuracy logging."""
    directional_gain: bool = False
    """Enable the oracle-gap directional gain on on-policy utilities."""
    directional_gain_gamma: float = 0.25
    """Exponent for the clipped oracle-gap scale."""
    directional_gain_positive_only: bool = True
    """Amplify only utilities pointing toward the positive oracle."""
    directional_gain_recenter: bool = True
    """Re-center transformed on-policy utilities before selection."""
    detached_oracle_advantage: bool = False
    """Overwrite the oracle row with a detached positive anchor."""
    detached_oracle_advantage_scale: float = 2.0
    """Base magnitude of the detached oracle anchor."""
    detached_oracle_use_directional_gain: bool = False
    """Whether the policy directional gain also scales the oracle anchor."""
    detached_oracle_match_best_ratio: float = 1.2
    """Cap the oracle anchor relative to the strongest positive policy row."""
    detached_oracle_match_best_min: float = 0.05
    """Lower bound for the adaptive oracle cap."""
    detached_oracle_match_best_max: float = 1.0
    """Upper bound for the adaptive oracle cap."""
    oracle_reward_gate_beta: float = 2.0
    """Exponent for the normalized oracle reward-gap gate."""
    selection_prune_ratio: float = 0.0
    """Fraction of generated policy rows removed before actor backpropagation."""
    selection_keep_oracle: bool = True
    """Force-retain the detached oracle row."""
    selection_positive_quota: int = 0
    """Number of positive-advantage policy rows retained per group."""
    selection_negative_quota: int = 0
    """Number of negative-advantage policy rows retained per group."""
    selection_strict_sign_balance: bool = True
    """Select from true positive/negative buckets with deterministic fallback."""
    post_selection_recenter: bool = False
    """Restore zero mean over rows that actually receive gradients."""
    post_selection_rms_match: bool = False
    """Only downscale active RMS toward its pre-selection policy reference."""
    post_selection_rms_min_scale: float = 0.25
    """Lower bound for post-selection RMS downscaling."""


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "orarl"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    keep_optim_only_latest: bool = False
    """when True, only the latest checkpoint keeps optimizer / extra_state / dataloader;
    older checkpoints retained within `save_limit` are thinned down to model weights only.
    Mutually exclusive with `save_model_only` (when `save_model_only=True` optimizer is
    never saved at all, so this flag is a no-op). Useful for saving disk while still
    allowing resume from the latest step."""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""
    keep_best_train_ckpt: bool = False
    """When True, additionally save the model-only checkpoint at the step with the
    highest *smoothed* training reward into a separate `best_train/` subdirectory.
    This sidesteps the rolling `save_limit` window so a peak you discovered mid-run
    is not evicted by later (potentially overtrained) saves. No optimizer/dataloader
    state is kept, only model weights — purely for downstream evaluation."""
    best_train_metric_key: str = "reward/overall"
    """Which key in the per-step `metrics` dict to track. Common choices:
    `reward/overall` (default; combines IoU + format), `reward/iou`,
    `critic/rewards/mean`."""
    best_train_smooth_window: int = 5
    """Number of recent steps to average for the best-train signal. Reduces single-
    step noise so we lock onto a true plateau rather than a lucky spike. Set to 1
    to disable smoothing."""
    best_train_min_step: int = 10
    """Don't track best-train until this many steps have elapsed (skip warmup
    where reward jumps quickly and the highest single value is nearly meaningless
    for picking a good policy)."""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        # The actor sizes its per-rank mini-batch from the post-selection row
        # count k = floor(n*(1-P)) per group rather than the full n.
        self.worker.actor.selection_prune_ratio = (
            float(self.algorithm.selection_prune_ratio)
            if str(self.algorithm.name).strip().lower() == "orarl"
            else 0.0
        )

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        from .orarl_config import public_config_dict

        return public_config_dict(self)
