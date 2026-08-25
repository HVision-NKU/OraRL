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

import inspect
import json
import math
import os
import time
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .multimodal_contract import (
    VIDEO_SOURCE_MODE_PREPROCESSED_ONLY,
    VIDEO_SOURCE_MODE_REALTIME_ONLY,
    build_video_multimodal_contract,
    normalize_video_source_mode,
    video_metadata_for_model,
)
from .prompt_template import build_prompt

try:
    _FETCH_VIDEO_PARAM_NAMES = frozenset(inspect.signature(fetch_video).parameters)
except (TypeError, ValueError):
    _FETCH_VIDEO_PARAM_NAMES = frozenset()


def _cap_int(sample_value: Any, config_value: int) -> int:
    try:
        return min(int(sample_value), int(config_value))
    except (TypeError, ValueError):
        return config_value


def _cap_optional_int(sample_value: Any, config_value: Optional[int]) -> Optional[int]:
    if sample_value is None:
        return config_value
    if config_value is None:
        try:
            return int(sample_value)
        except (TypeError, ValueError):
            return None
    return _cap_int(sample_value, config_value)


def _fetch_video_with_retry(vision_info: dict[str, Any], fetch_kwargs: dict[str, Any]) -> Any:
    """Retry transient FFmpeg/decord resource failures with path diagnostics."""
    try:
        attempts = max(1, int(os.getenv("MULTITASK_VIDEO_DECODE_RETRIES", "3")))
    except ValueError:
        attempts = 3
    try:
        retry_delay = max(0.0, float(os.getenv("MULTITASK_VIDEO_DECODE_RETRY_DELAY", "1.0")))
    except ValueError:
        retry_delay = 1.0

    video_path = vision_info.get("video")
    retryable_messages = (
        "eagain",
        "resource temporarily unavailable",
        "failed scaling graph",
        "failed initializing scaling graph",
        "cannot create buffer source",
        "errno 11",
    )
    for attempt in range(1, attempts + 1):
        try:
            return fetch_video(vision_info, **fetch_kwargs)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}".lower()
            retryable = any(message in error_text for message in retryable_messages)
            if not retryable:
                raise
            if attempt >= attempts:
                raise RuntimeError(
                    f"Video decode failed after {attempts} attempts for {video_path!r}: {error}"
                ) from error
            delay = retry_delay * attempt
            print(
                f"[video-decode] transient failure {attempt}/{attempts} for "
                f"{video_path!r}; retrying in {delay:.1f}s: {error}",
                flush=True,
            )
            if delay > 0:
                time.sleep(delay)

    raise AssertionError("unreachable")


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    all_keys: set[str] = set()
    for feature in features:
        all_keys.update(feature.keys())

    tensor_keys: set[str] = set()
    non_tensor_keys: set[str] = set()
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensor_keys.add(key)
            else:
                non_tensor_keys.add(key)

    result: dict[str, Any] = {}
    for key in tensor_keys:
        result[key] = torch.stack([f[key] for f in features], dim=0)

    for key in non_tensor_keys:
        result[key] = np.array([f.get(key, None) for f in features], dtype=object)

    return result


class LocalJsonlDataset(Dataset):
    """Random-access JSONL rows without imposing one Arrow schema.

    Multitask training files legitimately carry task-specific side channels:
    segmentation rows have mask metadata, temporal rows have spans, and
    spatial-intelligence rows have subtype fields. Hugging Face Datasets infers
    an Arrow schema from the first JSON chunk and rejects columns introduced by
    later chunks. This lightweight index keeps each row as its original Python
    mapping, while retaining random access for samplers and data-loader workers.
    """

    def __init__(
        self,
        path: str,
        *,
        offsets: Optional[list[int]] = None,
    ) -> None:
        self.path = os.path.abspath(path)
        self._offsets = offsets if offsets is not None else self._index_rows()
        self._handle = None
        self._handle_pid: Optional[int] = None

    def _index_rows(self) -> list[int]:
        offsets: list[int] = []
        with open(self.path, "rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        if not offsets:
            raise ValueError(f"JSONL dataset is empty: {self.path}")
        return offsets

    def _reader(self):
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid or self._handle.closed:
            if self._handle is not None and not self._handle.closed:
                self._handle.close()
            self._handle = open(self.path, "rb")
            self._handle_pid = pid
        return self._handle

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        handle = self._reader()
        handle.seek(self._offsets[index])
        raw = handle.readline()
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Invalid JSONL row {index + 1} in {self.path}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL row {index + 1} in {self.path} must be an object."
            )
        return row

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def filter(
        self,
        function,
        *,
        desc: Optional[str] = None,
        num_proc: Optional[int] = None,
    ) -> "LocalJsonlDataset":
        """Return a view containing rows accepted by ``function``.

        Filtering remains deterministic and sequential. Video prompt filtering
        is decode-heavy and should normally be completed during data
        preparation; released recipes disable it at training startup.
        """

        del num_proc
        if desc:
            print(f"{desc} (local JSONL)")
        selected = [
            offset
            for index, offset in enumerate(self._offsets)
            if function(self[index])
        ]
        return LocalJsonlDataset(self.path, offsets=selected)


class TaskGroupedBatchSampler(Sampler[list[int]]):
    """Yields batches where all samples belong to the same task type.

    Indices for each task group are shuffled independently each epoch, then
    chunked into batches of ``batch_size``. Groups are interleaved in
    round-robin order so that every task is visited roughly equally often.
    Incomplete tail batches are dropped.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        task_key: str = "problem_type",
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        group_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            row = dataset.dataset[idx]
            if task_key not in row:
                invalid_reason = "missing"
                task = None
            else:
                task = row[task_key]
                if task is None:
                    invalid_reason = "null"
                elif not isinstance(task, str):
                    invalid_reason = f"non-string ({type(task).__name__})"
                elif not task.strip():
                    invalid_reason = "empty"
                else:
                    invalid_reason = None

            if invalid_reason is not None:
                raise ValueError(
                    f"TaskGroupedBatchSampler: row index {idx} has a {invalid_reason} "
                    f"task key {task_key!r} (value={task!r})."
                )
            group_indices[task].append(idx)

        self.group_names = sorted(group_indices.keys())
        self.group_indices = {k: group_indices[k] for k in self.group_names}

        self._total_batches = 0
        for indices in self.group_indices.values():
            n = len(indices) // batch_size if drop_last else math.ceil(len(indices) / batch_size)
            self._total_batches += n

        groups_str = ", ".join(f"{k}({len(v)})" for k, v in self.group_indices.items())
        print(
            f"[TaskGroupedBatchSampler] {len(self.group_names)} groups: {groups_str} | "
            f"batch_size={batch_size} | total_batches={self._total_batches}"
        )

    def __iter__(self):
        group_batches: dict[str, list[list[int]]] = {}
        for name, indices in self.group_indices.items():
            idx = list(indices)
            if self.shuffle:
                perm = torch.randperm(len(idx), generator=self.generator).tolist()
                idx = [idx[i] for i in perm]
            batches = []
            for i in range(0, len(idx), self.batch_size):
                batch = idx[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
            group_batches[name] = batches

        queues = {name: iter(batches) for name, batches in group_batches.items()}
        active_names = list(self.group_names)

        while active_names:
            next_round = []
            for name in active_names:
                batch = next(queues[name], None)
                if batch is not None:
                    yield batch
                    next_round.append(name)
            active_names = next_round

    def __len__(self):
        return self._total_batches


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def _build_fallback_video_metadata(video_data: Union[torch.Tensor, list[ImageObject]], sample_fps: float) -> dict[str, Any]:
    if isinstance(video_data, torch.Tensor):
        total_num_frames = int(video_data.shape[0])
        height = int(video_data.shape[-2]) if video_data.ndim >= 3 else None
        width = int(video_data.shape[-1]) if video_data.ndim >= 3 else None
    else:
        total_num_frames = len(video_data)
        height = width = None
        if total_num_frames > 0:
            first_frame = video_data[0]
            if isinstance(first_frame, ImageObject):
                width, height = first_frame.size
            elif isinstance(first_frame, torch.Tensor) and first_frame.ndim >= 2:
                height = int(first_frame.shape[-2])
                width = int(first_frame.shape[-1])

    metadata = {
        "fps": float(sample_fps),
        "frames_indices": list(range(total_num_frames)),
        "total_num_frames": total_num_frames,
    }
    if sample_fps > 0:
        metadata["duration"] = total_num_frames / sample_fps
    if width is not None and height is not None:
        metadata["width"] = width
        metadata["height"] = height
    return metadata


def process_video(
    video: Any,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 64 * 32 * 32,
    max_frames: int = 128,
    video_fps: float = 2.0,
    total_pixels: Optional[int] = None,
    return_fps: bool = False,
) -> Any:
    if isinstance(video, dict):
        min_pixels = _cap_int(video.get("min_pixels"), min_pixels)
        max_pixels = _cap_int(video.get("max_pixels"), max_pixels)
        max_frames = _cap_int(video.get("max_frames"), max_frames)
        video_fps = video.get("fps", video.get("video_fps", video_fps))
        total_pixels = _cap_optional_int(video.get("total_pixels"), total_pixels)
        video = video.get("video") or video.get("path")

    if not isinstance(video, str) or not video.strip():
        raise ValueError("Video entry must be a non-empty string or a dict with a non-empty 'video' or 'path' string.")

    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }
    if total_pixels is not None:
        vision_info["total_pixels"] = total_pixels

    fetch_kwargs = {}
    # Qwen3-VL video processor reshapes frames with 16x16 patches.
    # Ensure offline and online preprocessing resize to a 16-aligned grid.
    if "image_patch_size" in _FETCH_VIDEO_PARAM_NAMES:
        fetch_kwargs["image_patch_size"] = 16
    elif "image_factor" in _FETCH_VIDEO_PARAM_NAMES:
        fetch_kwargs["image_factor"] = 16
    if return_fps and "return_video_sample_fps" in _FETCH_VIDEO_PARAM_NAMES:
        fetch_kwargs["return_video_sample_fps"] = True
    if return_fps and "return_video_metadata" in _FETCH_VIDEO_PARAM_NAMES:
        fetch_kwargs["return_video_metadata"] = True

    result = _fetch_video_with_retry(vision_info, fetch_kwargs)
    if not return_fps:
        return result

    if isinstance(result, tuple) and len(result) == 2:
        video_data, sample_fps = result
    else:
        video_data, sample_fps = result, video_fps

    sample_fps = float(sample_fps)
    if isinstance(video_data, tuple) and len(video_data) == 2:
        return video_data, sample_fps

    return (video_data, _build_fallback_video_metadata(video_data, sample_fps)), sample_fps


def _video_path_from_entry(video: Any) -> str:
    if isinstance(video, dict):
        video = video.get("video") or video.get("path")
    if not isinstance(video, str) or not video.strip():
        raise ValueError("Video entry must contain a non-empty string path.")
    return video


def _resolve_video_entries(
    videos: list[Any], image_dir: Optional[str] = None
) -> tuple[list[Any], list[str]]:
    resolved_entries: list[Any] = []
    video_paths: list[str] = []
    for video in videos:
        video_path = _video_path_from_entry(video)
        if image_dir is not None and not os.path.isabs(video_path):
            video_path = os.path.join(image_dir, video_path)
        video_paths.append(video_path)

        if isinstance(video, dict):
            resolved_video = dict(video)
            resolved_video["video"] = video_path
            resolved_entries.append(resolved_video)
        else:
            resolved_entries.append(video_path)

    return resolved_entries, video_paths


def _align_media_placeholders(
    prompt: str,
    marker: str,
    media_count: int,
) -> str:
    """Match prompt placeholders to the media entries actually supplied.

    Some legacy image-sequence rows retain placeholders for every source frame
    even though their media list was capped during data preparation. Extra
    placeholders have no corresponding pixels and must be removed before the
    processor expands multimodal tokens. Missing placeholders are ambiguous and
    remain a hard error instead of inventing an ordering.
    """

    placeholder_count = prompt.count(marker)
    if placeholder_count < media_count:
        raise ValueError(
            f"Prompt has {placeholder_count} {marker} placeholder(s) for "
            f"{media_count} media item(s)."
        )
    if placeholder_count == media_count:
        return prompt

    parts = prompt.split(marker)
    return marker.join(parts[: media_count + 1]) + "".join(
        parts[media_count + 1 :]
    )


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        video_max_frames: int = 128,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        video_total_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        use_preprocessed_videos: bool = True,
        video_source_mode: Optional[str] = None,
        preprocessed_video_dir: Optional[str] = None,
        inline_video_tensors: bool = False,
        enable_thinking: bool = False,
        response_prefix: str = "",
        model_type: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_type = model_type
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.video_max_frames = video_max_frames
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.video_total_pixels = video_total_pixels
        self.use_preprocessed_videos = use_preprocessed_videos
        self.video_source_mode = normalize_video_source_mode(
            video_source_mode,
            use_preprocessed_videos=use_preprocessed_videos,
        )
        self.preprocessed_video_dir = preprocessed_video_dir
        self.inline_video_tensors = bool(inline_video_tensors)
        self.enable_thinking = bool(enable_thinking)
        self.response_prefix = str(response_prefix or "")

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            if data_path.casefold().endswith(".jsonl"):
                self.dataset = LocalJsonlDataset(data_path)
            else:
                file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
                self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _ensure_single_vision_modality(self, has_images: bool, has_videos: bool) -> None:
        if has_images and has_videos:
            raise NotImplementedError(
                "A single sample containing both images and videos is not supported in this training contract yet."
            )

    def _append_response_prefix(self, prompt: str) -> str:
        """Start generation from an explicit assistant prefix when configured."""
        return prompt + self.response_prefix

    def _resolve_preprocessed_video_path(self, example: dict[str, Any], *, pop_value: bool) -> Optional[str]:
        if pop_value:
            preprocessed_video_file = example.pop("preprocessed_video", None)
        else:
            preprocessed_video_file = example.get("preprocessed_video")

        if not preprocessed_video_file:
            if self.video_source_mode == VIDEO_SOURCE_MODE_PREPROCESSED_ONLY:
                problem_id = example.get("problem_id", "unknown")
                raise FileNotFoundError(
                    f"video_source_mode=preprocessed_only but sample {problem_id!r} has no preprocessed_video field."
                )
            return None

        if self.preprocessed_video_dir is not None:
            preprocessed_video_path = os.path.join(self.preprocessed_video_dir, preprocessed_video_file)
        else:
            preprocessed_video_path = preprocessed_video_file

        if os.path.exists(preprocessed_video_path):
            return preprocessed_video_path

        if self.video_source_mode == VIDEO_SOURCE_MODE_PREPROCESSED_ONLY:
            problem_id = example.get("problem_id", "unknown")
            raise FileNotFoundError(
                f"video_source_mode=preprocessed_only but artifact is missing for sample {problem_id!r}: "
                f"{preprocessed_video_path}"
            )

        return None

    def _should_use_preprocessed_video(self, preprocessed_video_path: Optional[str]) -> bool:
        return (
            preprocessed_video_path is not None
            and self.video_source_mode != VIDEO_SOURCE_MODE_REALTIME_ONLY
        )

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            # Pass every example field to the Jinja template so it can route on
            # problem_type, data_type, options, data_source, and so on.
            prompt_str = format_prompt.render(
                content=prompt_str,  # older templates expect `content`
                problem=prompt_str,
                **{k: v for k, v in example.items() if k != self.prompt_key},
            )
        else:
            # Only add task-specific instructions when no template was supplied;
            # records that already carry full instructions would get them twice.
            prompt_str = build_prompt(prompt_str, example)

        # Accept both list and numpy.ndarray media columns.
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, "__len__")
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, "__len__")
            and len(videos_data) > 0
        )
        self._ensure_single_vision_modality(has_images, has_videos)

        if has_images:
            prompt_str = _align_media_placeholders(
                prompt_str,
                "<image>",
                len(images_data),
            )
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif has_videos:
            prompt_str = _align_media_placeholders(
                prompt_str,
                "<video>",
                len(videos_data),
            )
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        # Accept both list and numpy.ndarray media columns.
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, "__len__")
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, "__len__")
            and len(videos_data) > 0
        )

        if has_images:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
            )
            prompt = self._append_response_prefix(prompt)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif has_videos:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
            )
            prompt = self._append_response_prefix(prompt)

            preprocessed_video_path = self._resolve_preprocessed_video_path(example, pop_value=False)
            if self._should_use_preprocessed_video(preprocessed_video_path):
                preprocessed_data = torch.load(preprocessed_video_path, map_location="cpu", weights_only=False)
                processed_videos = [preprocessed_data["frames"]]
                video_metadatas = [
                    video_metadata_for_model(
                        preprocessed_data["metadata"],
                        preprocessed_data["frames"],
                    )
                ]
                model_inputs = self.processor(
                    videos=processed_videos,
                    text=[prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                    video_metadata=video_metadatas,
                    do_resize=False,
                    do_sample_frames=False,
                )
                return model_inputs["input_ids"].size(-1) <= self.max_prompt_length

            # Fall back to decoding the source video at request time.
            videos, _ = _resolve_video_entries(example[self.video_key], self.image_dir)

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_metadatas = []
            for video in videos:
                result = process_video(
                    video,
                    min_pixels=self.video_min_pixels if self.video_min_pixels else 4 * 32 * 32,
                    max_pixels=self.video_max_pixels if self.video_max_pixels else 64 * 32 * 32,
                    max_frames=self.video_max_frames,
                    video_fps=self.video_fps,
                    total_pixels=self.video_total_pixels,
                    return_fps=True,
                )
                if isinstance(result, tuple) and len(result) == 2:
                    video_data, _ = result  # Unpack (video_data, sample_fps)
                    if isinstance(video_data, tuple) and len(video_data) == 2:
                        frames, metadata = video_data
                        processed_videos.append(frames)
                        video_metadatas.append(video_metadata_for_model(metadata, frames))
                    else:
                        processed_videos.append(video_data)
                        video_metadatas = None
                        break
                else:
                    processed_videos.append(result)
                    video_metadatas = None
                    break

            if video_metadatas is not None and len(video_metadatas) > 0:
                model_inputs = self.processor(
                    videos=processed_videos,
                    text=[prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                    video_metadata=video_metadatas,
                    do_resize=False,
                    do_sample_frames=False,
                )
            else:
                model_inputs = self.processor(
                    videos=processed_videos,
                    text=[prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                    do_sample_frames=False,
                )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
            )
            prompt = self._append_response_prefix(prompt)
            input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)

        # Accept both list and numpy.ndarray media columns.
        images_data = example.get(self.image_key)
        has_images = (
            self.image_key in example
            and images_data is not None
            and hasattr(images_data, "__len__")
            and len(images_data) > 0
        )
        # Check if videos exist and is a non-empty list/array
        videos_data = example.get(self.video_key)
        has_videos = (
            self.video_key in example
            and videos_data is not None
            and hasattr(videos_data, "__len__")
            and len(videos_data) > 0
        )

        if has_images:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
            )
            prompt = self._append_response_prefix(prompt)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif has_videos:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
            )
            prompt = self._append_response_prefix(prompt)
            videos = example.pop(self.video_key)
            videos, video_paths = _resolve_video_entries(videos, self.image_dir)

            preprocessed_video_path = self._resolve_preprocessed_video_path(example, pop_value=True)
            if self._should_use_preprocessed_video(preprocessed_video_path):
                preprocessed_data = torch.load(preprocessed_video_path, map_location="cpu", weights_only=False)
                processed_videos = [preprocessed_data["frames"]]
                video_metadatas = [
                    video_metadata_for_model(
                        preprocessed_data["metadata"],
                        preprocessed_data["frames"],
                    )
                ]
                video_kwargs = {"do_sample_frames": False}
            else:
                processed_videos = [] if len(videos) != 0 else None  # text-only data
                video_kwargs = {"do_sample_frames": False}  # For Qwen3-VL
                for video in videos:
                    processed_video, _ = process_video(
                        video,
                        min_pixels=self.video_min_pixels if self.video_min_pixels else 4 * 32 * 32,
                        max_pixels=self.video_max_pixels if self.video_max_pixels else 64 * 32 * 32,
                        max_frames=self.video_max_frames,
                        video_fps=self.video_fps,
                        total_pixels=self.video_total_pixels,
                        return_fps=True,
                    )
                    processed_videos.append(processed_video)

            # Handle video_metadata for Qwen3-VL
            if processed_videos is not None and len(processed_videos) > 0:
                # video_metadatas is already set when a preprocessed tensor was loaded.
                if "video_metadatas" in locals() and video_metadatas is not None and len(video_metadatas) > 0:
                    # Preprocessed video: processed_videos already holds the frames.
                    processed_video_frames = processed_videos
                else:
                    # Realtime path: process_video returns (frames, metadata) when return_fps=True.
                    # Historical runs normalize metadata against decoded frames.
                    # Upstream parity preserves original fps/frame indices so Qwen3-VL
                    # renders the exact upstream timestamp tokens. Apply the selected
                    # contract at the source so input ids, rollout and FSDP forward agree.
                    processed_video_frames = []
                    video_metadatas = []
                    for pv in processed_videos:
                        if isinstance(pv, tuple) and len(pv) == 2:
                            frames, metadata = pv
                            processed_video_frames.append(frames)
                            video_metadatas.append(video_metadata_for_model(metadata, frames))
                        else:
                            processed_video_frames.append(pv)
                            video_metadatas = None
                            break

                if video_metadatas is not None and len(video_metadatas) > 0:
                    model_inputs = self.processor(
                        text=[prompt],
                        videos=processed_video_frames,
                        add_special_tokens=False,
                        video_metadata=video_metadatas,
                        return_tensors="pt",
                        do_resize=False,
                        **video_kwargs,
                    )
                else:
                    model_inputs = self.processor(
                        videos=processed_video_frames,
                        text=[prompt],
                        add_special_tokens=False,
                        return_tensors="pt",
                        **video_kwargs,
                    )
            else:
                model_inputs = self.processor(
                    videos=processed_videos,
                    text=[prompt],
                    add_special_tokens=False,
                    return_tensors="pt",
                )

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

            inline_frames = None
            inline_metadatas = None
            if (
                self.inline_video_tensors
                and "processed_video_frames" in locals()
                and processed_video_frames is not None
                and len(processed_video_frames) > 0
                and "video_metadatas" in locals()
                and video_metadatas is not None
                and len(video_metadatas) == len(processed_video_frames)
            ):
                inline_frames = processed_video_frames
                inline_metadatas = video_metadatas

            example["multi_modal_data"] = build_video_multimodal_contract(
                video_paths=video_paths,
                preprocessed_video_path=preprocessed_video_path,
                video_source_mode=self.video_source_mode,
                inline_frames=inline_frames,
                inline_metadatas=inline_metadatas,
            )
        else:
            # Text-only sample, with neither images nor videos.
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.enable_thinking
            )
            prompt = self._append_response_prefix(prompt)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            # Text-only rows still need an empty dict so the batch keys stay
            # uniform; None would break vllm_rollout.
            example["multi_modal_data"] = {}

        # Clean up images/videos keys if they still exist
        if self.image_key in example:
            example.pop(self.image_key, None)
        if self.video_key in example:
            example.pop(self.video_key, None)

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            # Qwen3.5 and Qwen3-VL share the same Qwen3VLProcessor,
            # so we distinguish by model_type when available
            if self.model_type == "qwen3_5":
                from ..models.transformers.qwen3_5 import get_rope_index
            elif "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)

        return example
