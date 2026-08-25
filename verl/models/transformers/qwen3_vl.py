# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Based on:
# https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py
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

import os
import time
from typing import Optional

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

# Rate-limit state for the no-crash visual-alignment guard (see
# ``_reconcile_visual_features``). Keyed by modality ("image"/"video").
_VISUAL_GUARD_LAST_LOG: dict[str, float] = {}


def _reconcile_visual_features(
    embeds: torch.Tensor,
    n_tokens: int,
    kind: str,
) -> torch.Tensor:
    """Trim/pad visual ``embeds`` to exactly ``n_tokens`` rows so the FSDP
    forward does not crash on an aggregate token/feature mismatch.

    This is a DIAGNOSTIC guard for the no-cache RL pipeline: when enabled
    (``VERL_GUARD_VIDEO_MISMATCH=1``, the default), a mismatch is logged
    (rate-limited) and reconciled instead of raising, so a real run keeps
    collecting the per-sample culprit info emitted by the FSDP worker's
    ``_diagnose_video_alignment``. Reconciliation slightly corrupts the
    offending micro-batch's visual grounding, so it is meant for hunting the
    root cause, not for production training. Set ``VERL_GUARD_VIDEO_MISMATCH=0``
    to restore the hard ``ValueError``.
    """
    n_features = embeds.shape[0]
    if n_features == n_tokens:
        return embeds

    if os.environ.get("VERL_GUARD_VIDEO_MISMATCH", "1") != "1":
        raise ValueError(
            f"{kind.capitalize()} features and {kind} tokens do not match: "
            f"tokens: {n_tokens}, features {n_features}"
        )

    now = time.time()
    if now - _VISUAL_GUARD_LAST_LOG.get(kind, 0.0) >= 10.0:
        _VISUAL_GUARD_LAST_LOG[kind] = now
        print(
            f"[VIDEO-ALIGN][guard] {kind} tokens={n_tokens} features={n_features} "
            f"(delta={n_features - n_tokens}); reconciling to avoid crash "
            f"(set VERL_GUARD_VIDEO_MISMATCH=0 to hard-fail).",
            flush=True,
        )

    if n_features > n_tokens:
        return embeds[:n_tokens]
    # n_features < n_tokens: pad by repeating the last feature row.
    pad = embeds[-1:].expand(n_tokens - n_features, *embeds.shape[1:])
    return torch.cat([embeds, pad], dim=0)


def get_rope_index(
    processor: "Qwen3VLProcessor",
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen3-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L916
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id

    # Since we use timestamps to seperate videos,
    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>,
    # the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(input_ids.device)
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def _get_input_embeds(
    model: "Qwen3VLModel",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    def _visual_forward(pixel_tensor: torch.Tensor, grid_thw: torch.Tensor):
        """Normalize Qwen3-VL vision outputs across Transformers versions."""
        try:
            outputs = model.visual(pixel_tensor, grid_thw=grid_thw, return_dict=True)
        except TypeError:
            # Older Qwen3-VL implementations do not accept return_dict and
            # directly return (pooled_features, deepstack_features).
            outputs = model.visual(pixel_tensor, grid_thw=grid_thw)

        if hasattr(outputs, "pooler_output"):
            embeds = outputs.pooler_output
            deepstack = getattr(outputs, "deepstack_features", None)
        elif isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            embeds, deepstack = outputs[0], outputs[1]
        else:
            raise TypeError(
                "Unsupported Qwen3-VL vision output: expected a model output "
                "with pooler_output/deepstack_features or a 2-tuple."
            )

        # get_image_features/get_video_features may return one tensor per
        # image/video, while a direct visual forward returns one tensor.
        if isinstance(embeds, (tuple, list)):
            embeds = torch.cat(tuple(embeds), dim=0)
        if deepstack is None:
            deepstack = []
        return embeds, deepstack

    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask, video_mask = None, None
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds, deepstack_image_embeds = _visual_forward(pixel_values, image_grid_thw)
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

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds, deepstack_video_embeds = _visual_forward(pixel_values_videos, video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        # Guarded reconcile (no-cache RL diagnostic): trim/pad to the token count
        # instead of raising, so the run survives and the FSDP worker's
        # per-sample audit keeps naming the offending sample. The deepstack
        # tensors scatter into the same video positions, so reconcile them too.
        video_embeds = _reconcile_visual_features(video_embeds, n_video_tokens, "video")
        deepstack_video_embeds = [
            _reconcile_visual_features(d, n_video_tokens, "video") for d in deepstack_video_embeds
        ]

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds, dummy_deepstack_image_embeds = _visual_forward(pixel_values, image_grid_thw)
        inputs_embeds += 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds += 0.0 * emb.mean()

    # for image-video mixed training

    if pixel_values is not None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2

        _video_grid_thw = (
            video_grid_thw
            if (video_grid_thw is not None)
            else torch.tensor([[2, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        )
        _T, _H, _W = _video_grid_thw[0].tolist()
        _n_tokens = int(_T * _H * _W)

        _dummy_video_pixels = torch.zeros(
            (_n_tokens, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

        _video_embeds, _video_deepstack = _visual_forward(_dummy_video_pixels, _video_grid_thw)
        inputs_embeds = inputs_embeds + 0.0 * _video_embeds.mean()
        for _emb in _video_deepstack or []:
            inputs_embeds = inputs_embeds + 0.0 * _emb.mean()

    # for image-video mixed training: create dummy image when only video is present
    if pixel_values is None and pixel_values_videos is not None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2

        _image_grid_thw = (
            image_grid_thw
            if (image_grid_thw is not None)
            else torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        )
        _T, _H, _W = _image_grid_thw[0].tolist()
        _n_tokens = int(_T * _H * _W)

        _dummy_image_pixels = torch.zeros(
            (_n_tokens, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

        _image_embeds, _image_deepstack = _visual_forward(_dummy_image_pixels, _image_grid_thw)
        inputs_embeds = inputs_embeds + 0.0 * _image_embeds.mean()
        for _emb in _image_deepstack or []:
            inputs_embeds = inputs_embeds + 0.0 * _emb.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }


def qwen3_vl_base_forward(
    self: "Qwen3VLModel",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    position_ids = kwargs.get("position_ids")
    if isinstance(position_ids, torch.Tensor) and (position_ids.ndim != 3 or position_ids.size(0) != 4):
        # we concat the text position ids with the 3D vision position ids by default
        # see https://github.com/huggingface/transformers/pull/39447
        raise ValueError("position_ids should be a 3D tensor of shape (4, batch_size, seq_length).")

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
    return Qwen3VLModelOutputWithPast(last_hidden_state=outputs.last_hidden_state)


def qwen3_vl_model_forward(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    labels: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "Qwen3VLCausalLMOutputWithPast":
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3VLCausalLMOutputWithPast(logits=logits)
