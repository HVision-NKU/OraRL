# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Based on:
# https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
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

import itertools
from typing import Optional

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
)


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen3.5, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.

    Qwen3.5 uses mm_token_type_ids to distinguish modalities (0=text, 1=image, 2=video)
    and interleaved MRoPE with sections=[11,11,10].

    Returns position_ids of shape (3, seq_length) — [temporal, height, width].
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id

    # Since we use timestamps to separate videos,
    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>,
    # the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        attention_mask = attention_mask.to(input_ids.device)
        input_ids_masked = input_ids[attention_mask == 1]

        # Build mm_token_type_ids: 0=text, 1=image, 2=video
        mm_token_type_ids = torch.zeros_like(input_ids_masked)
        mm_token_type_ids[input_ids_masked == image_token_id] = 1
        mm_token_type_ids[input_ids_masked == video_token_id] = 2

        # Group tokens by modality type using itertools.groupby
        image_iter = iter(image_grid_thw) if image_grid_thw is not None else iter([])
        video_iter = iter(video_grid_thw) if video_grid_thw is not None else iter([])

        input_type_group = []
        for key, group in itertools.groupby(enumerate(mm_token_type_ids.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                # Text tokens: standard 1D position IDs expanded to 3D
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:
                # Vision tokens (image=1, video=2): compute 3D position IDs
                if modality_type == 1:
                    grid_thw = next(image_iter)
                else:
                    grid_thw = next(video_iter)

                # Compute vision position IDs (same logic as Qwen3_5Model.get_vision_position_ids)
                llm_grid_t = grid_thw[0].item() // 1  # temp_merge_size=1
                llm_grid_h = grid_thw[1].item() // spatial_merge_size
                llm_grid_w = grid_thw[2].item() // spatial_merge_size

                image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
                position_width = torch.arange(
                    current_pos, current_pos + llm_grid_w, device=input_ids.device
                ).repeat(llm_grid_h * llm_grid_t)
                position_height = torch.arange(
                    current_pos, current_pos + llm_grid_h, device=input_ids.device
                ).repeat_interleave(llm_grid_w * llm_grid_t)
                position_temporal = torch.full(
                    (image_seq_length,), current_pos, device=input_ids.device, dtype=torch.long
                )
                vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)
                llm_pos_ids_list.append(vision_position_ids)

                # Advance current_pos by max(h, w) // spatial_merge_size (Qwen3.5 convention)
                current_pos += max(grid_thw[1].item(), grid_thw[2].item()) // spatial_merge_size

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = torch.arange(input_ids.shape[0], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def _get_input_embeds(
    model: "Qwen3_5Model",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    """
    Compute input embeddings with visual features injected for Qwen3.5.

    Qwen3.5 uses model.visual (Qwen3_5VisionModel) which returns
    BaseModelOutputWithPooling with pooler_output containing split image embeds.
    No deepstack_visual_indexes in Qwen3.5-2B (empty list).
    """
    inputs_embeds = model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_outputs = model.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        image_embeds = image_outputs.pooler_output
        # pooler_output is already split by image; cat them back
        split_sizes = (image_grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        image_embeds = torch.cat(image_embeds, dim=0)

        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    else:
        image_mask = None

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_outputs = model.visual(pixel_values_videos, grid_thw=video_grid_thw, return_dict=True)
        video_embeds = video_outputs.pooler_output
        split_sizes = (video_grid_thw.prod(-1) // model.visual.spatial_merge_size**2).tolist()
        video_embeds = torch.split(video_embeds, split_sizes)
        video_embeds = torch.cat(video_embeds, dim=0)

        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    else:
        video_mask = None

    # Dummy gradient flow: when neither images nor videos are present,
    # run a dummy forward through the visual encoder to maintain gradient flow
    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        dummy_pixels = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        dummy_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        dummy_outputs = model.visual(dummy_pixels, grid_thw=dummy_grid_thw, return_dict=True)
        dummy_embeds = dummy_outputs.pooler_output
        if isinstance(dummy_embeds, (list, tuple)):
            for emb in dummy_embeds:
                inputs_embeds += 0.0 * emb.mean()
        else:
            inputs_embeds += 0.0 * dummy_embeds.mean()

    # Dummy gradient flow for image-video mixed training:
    # when only images present, create dummy video input
    if pixel_values is not None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        _video_grid_thw = video_grid_thw if (video_grid_thw is not None) \
            else torch.tensor([[2, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        _T, _H, _W = _video_grid_thw[0].tolist()
        _n_tokens = int(_T * _H * _W)
        _dummy_video_pixels = torch.zeros((_n_tokens, patch_dim),
                                          dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        _video_outputs = model.visual(_dummy_video_pixels, grid_thw=_video_grid_thw, return_dict=True)
        _video_embeds = _video_outputs.pooler_output
        if isinstance(_video_embeds, (list, tuple)):
            for emb in _video_embeds:
                inputs_embeds = inputs_embeds + 0.0 * emb.mean()
        else:
            inputs_embeds = inputs_embeds + 0.0 * _video_embeds.mean()

    # Dummy gradient flow: when only videos present, create dummy image input
    if pixel_values is None and pixel_values_videos is not None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        _image_grid_thw = image_grid_thw if (image_grid_thw is not None) \
            else torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        _T, _H, _W = _image_grid_thw[0].tolist()
        _n_tokens = int(_T * _H * _W)
        _dummy_image_pixels = torch.zeros((_n_tokens, patch_dim),
                                          dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        _image_outputs = model.visual(_dummy_image_pixels, grid_thw=_image_grid_thw, return_dict=True)
        _image_embeds = _image_outputs.pooler_output
        if isinstance(_image_embeds, (list, tuple)):
            for emb in _image_embeds:
                inputs_embeds = inputs_embeds + 0.0 * emb.mean()
        else:
            inputs_embeds = inputs_embeds + 0.0 * _image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
    }


def qwen3_5_base_forward(
    self: "Qwen3_5Model",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """
    Override Qwen3_5Model.forward for RL training with padding-free sequence packing.

    This replaces the original forward to:
    1. Handle visual embedding injection with dummy gradient flow
    2. Support the verl padding-free training pipeline
    """
    position_ids = kwargs.get("position_ids")
    if isinstance(position_ids, torch.Tensor) and position_ids.ndim not in (2, 3):
        raise ValueError("position_ids should have shape (batch_size, seq_length) or (3, batch_size, seq_length).")

    input_kwargs = _get_input_embeds(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_values_videos,
        image_grid_thw,
        video_grid_thw,
    )
    kwargs.update(input_kwargs)  # avoid lora module to have multiple keyword arguments
    outputs = self.language_model(input_ids=None, **kwargs)
    return Qwen3_5ModelOutputWithPast(last_hidden_state=outputs.last_hidden_state)


def qwen3_5_model_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor,
    labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputWithPast":
    """
    Override Qwen3_5ForConditionalGeneration.forward for RL training.

    Calls self.model (Qwen3_5Model) then self.lm_head to get logits.
    """
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3_5CausalLMOutputWithPast(logits=logits)
