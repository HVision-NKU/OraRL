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

from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, TaskGroupedBatchSampler, collate_fn
from .config import DataConfig


def create_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    model_path: Optional[str] = None,
) -> None:
    # Auto-detect model_type from model config for proper position_ids routing
    model_type = None
    if model_path is not None:
        try:
            auto_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            model_type = getattr(auto_config, "model_type", None)
        except Exception:
            pass

    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        video_max_frames=config.video_max_frames,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        image_min_pixels=config.image_min_pixels,
        image_max_pixels=config.image_max_pixels,
        video_min_pixels=config.video_min_pixels,
        video_max_pixels=config.video_max_pixels,
        video_total_pixels=config.video_total_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
        use_preprocessed_videos=config.use_preprocessed_videos,
        video_source_mode=config.video_source_mode,
        preprocessed_video_dir=config.preprocessed_video_dir,
        inline_video_tensors=config.inline_video_tensors,
        enable_thinking=config.enable_thinking,
        response_prefix=config.response_prefix,
        model_type=model_type,
    )
    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    if config.group_by_task:
        batch_sampler = TaskGroupedBatchSampler(
            dataset=train_dataset,
            batch_size=train_batch_size,
            task_key=config.group_by_task_key,
            shuffle=config.shuffle,
            seed=config.seed,
            drop_last=True,
        )
        train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_sampler=batch_sampler,
            num_workers=config.dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
        )
    else:
        if config.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(config.seed)
            sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=train_dataset)

        train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=train_batch_size,
            sampler=sampler,
            num_workers=config.dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.val_video_fps,
        video_max_frames=config.val_video_max_frames,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        image_min_pixels=config.image_min_pixels,
        image_max_pixels=config.image_max_pixels,
        video_min_pixels=config.val_video_min_pixels,
        video_max_pixels=config.val_video_max_pixels,
        video_total_pixels=config.val_video_total_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        use_preprocessed_videos=config.use_preprocessed_videos,
        video_source_mode=config.val_video_source_mode,
        preprocessed_video_dir=config.val_preprocessed_video_dir,
        inline_video_tensors=config.inline_video_tensors,
        enable_thinking=config.enable_thinking,
        response_prefix=config.response_prefix,
        model_type=model_type,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
