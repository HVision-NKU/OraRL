"""Monkey-patch qwen_vl_utils to force the decord video backend.

Why this is needed
==================

* Evaluators call ``process_vision_info(..., return_video_metadata=True)``,
  which is the **new-API** form of ``qwen_vl_utils``. New-API
  ``fetch_video`` expects the underlying backend reader to return a
  3-tuple ``(video_tensor, video_metadata, sample_fps)``.

* Newer ``qwen_vl_utils`` (>= 0.0.10) hard-codes
  ``VIDEO_READER_BACKENDS["torchvision"](ele)`` inside ``fetch_video``, so
  ``FORCE_QWENVL_VIDEO_READER=decord`` no longer takes effect. The
  torchvision reader on some machines fails with
  ``KeyError: 'video_fps'`` because the mp4 metadata is incomplete.

* Older ``_read_video_decord`` (<= 0.0.8) returns a single tensor; if we
  just alias ``"torchvision" -> _read_video_decord``, the 3-value unpack
  in ``fetch_video`` blows up with ``RuntimeError: Error reading <path>``.

Strategy
========

Replace ``vp.fetch_video`` with our own version that uses ``decord``
directly and returns the **new-API 3-tuple** the rest of qwen_vl_utils
expects. This way both old- and new-API call sites work, and we never
have to ship a patched copy of qwen_vl_utils itself.

Disable the patch with ``DISABLE_QWENVL_DECORD_PATCH=1`` (e.g. when
debugging on a machine where torchvision works fine).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any


def _smart_nframes(ele: dict, *, total_frames: int, video_fps: float) -> int:
    """Mirror qwen_vl_utils.vision_process.smart_nframes: pick the number
    of frames to sample given fps / max_frames hints in the message dict.

    We re-implement instead of importing because qwen_vl_utils versions
    differ on the public name (``smart_nframes`` vs internal helper).
    """
    FPS = float(ele.get("fps", 2.0))
    MIN_FRAMES = int(ele.get("min_frames", 4))
    MAX_FRAMES = int(ele.get("max_frames", 768))
    nframes = ele.get("nframes")
    if nframes is not None:
        return max(1, min(int(nframes), total_frames))
    duration = total_frames / max(1e-6, video_fps)
    if FPS <= 0:
        FPS = 2.0
    n = int(round(duration * FPS))
    n = max(MIN_FRAMES, min(MAX_FRAMES, n))
    n = max(1, min(n, total_frames))
    return n


def _resolve_video_path(video_path: str) -> str:
    """If the input path doesn't exist, try a few fallbacks before giving up.

    Benchmark annotations join a base prefix with a relative media path, so a
    layout that nests media one level differently than the annotation expects
    (for example ``./Evaluation/got10k/<file>.mp4`` against an on-disk
    ``<base>/got10k/<file>.mp4``) would otherwise fail to decode.

    Resolution order:
      0. exact path (default)
      1. drop ``/Evaluation/`` -> retry
      2. drop leading ``Evaluation/`` from the raw component if EVAL_BASE_PREFIX_OVERRIDE
      3. take basename + walk under override prefix (slow last resort)
    """
    if os.path.isfile(video_path):
        return video_path

    candidates = []

    # (1) Drop the spurious "/Evaluation/" component anywhere in the path.
    if "/Evaluation/" in video_path:
        candidates.append(video_path.replace("/Evaluation/", "/", 1))

    # (2) Override-prefix-based rewrites.
    override = os.getenv("EVAL_BASE_PREFIX_OVERRIDE", "").rstrip("/")
    base_name = os.path.basename(video_path)
    parent = os.path.dirname(video_path)
    if override:
        rel = video_path
        for prefix in ("Evaluation/", "evaluation/", "./Evaluation/", "./"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
        candidates.append(os.path.join(override, rel))
        candidates.append(os.path.join(override, base_name))
        pdir = os.path.basename(parent)
        if pdir:
            candidates.append(os.path.join(override, pdir, base_name))

    for c in candidates:
        if os.path.isfile(c):
            print(f"[qwenvl_decord_patch] path-resolved {video_path!r} → {c!r}",
                  file=sys.stderr)
            return c

    # (3) Last resort: walk under override (or the path's grandparent).
    walk_root = override or os.path.dirname(parent) or parent
    if walk_root and os.path.isdir(walk_root):
        for root, dirs, files in os.walk(walk_root):
            if base_name in files:
                hit = os.path.join(root, base_name)
                print(f"[qwenvl_decord_patch] walked-resolved {video_path!r} → {hit!r}",
                      file=sys.stderr)
                return hit
            depth = root[len(walk_root):].count(os.sep)
            if depth >= 4:
                dirs.clear()

    return video_path


def _decord_fetch_video_new_api(ele: dict, *args, **kwargs):
    """Drop-in replacement for ``qwen_vl_utils.vision_process.fetch_video``.

    The exact signature of fetch_video varies across qwen_vl_utils
    versions, but we observe the following calling conventions in the
    wild:

      v0.0.8     : fetch_video(ele, image_factor)             → tensor
      v0.0.10+   : fetch_video(ele, image_factor=…,
                                  return_video_sample_fps=True,
                                  return_video_metadata=False) → (video, sample_fps)
      v0.0.10+   : fetch_video(ele, …, return_video_metadata=True) → (video, metadata, sample_fps)

    We therefore inspect ``return_video_sample_fps`` /
    ``return_video_metadata`` kwargs and return whichever shape the
    caller asked for. The actual decode is always decord-based.
    """
    import torch
    import decord

    video_path = ele["video"]
    if isinstance(video_path, str) and video_path.startswith("file://"):
        video_path = video_path[len("file://"):]

    # Multi-tier fallback path resolution (see _resolve_video_path docstring)
    video_path = _resolve_video_path(video_path)

    if "video_start" in ele or "video_end" in ele:
        # Trim window. decord supports ranges via numeric indices, but no
        # evaluated benchmark sets these, so we keep it simple.
        raise NotImplementedError(
            "video_start/video_end not implemented in decord patch; "
            "the evaluated benchmarks do not need this."
        )

    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    video_fps = float(vr.get_avg_fps())
    if total_frames <= 0 or not math.isfinite(video_fps) or video_fps <= 0:
        raise RuntimeError(
            f"decord-patch: invalid video {video_path!r} "
            f"(total_frames={total_frames}, video_fps={video_fps})"
        )

    nframes = _smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()
    arr = vr.get_batch(idx).asnumpy()  # (T, H, W, C)
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)

    sample_fps = nframes / (total_frames / video_fps) if total_frames > 0 else float(video_fps)

    # metadata fields are chosen to be compatible with transformers'
    # VideoMetadata dataclass (qwen3-vl image processor instantiates this
    # from the dict we return). Field set across qwen2/qwen3-vl versions:
    #   fps, total_num_frames, duration, frames_indices, video_backend
    # We deliberately DO NOT include "video_fps" — that name causes
    # ``VideoMetadata.__init__() got an unexpected keyword argument 'video_fps'``
    # on newer transformers.
    metadata = {
        "fps": float(video_fps),
        "total_num_frames": int(total_frames),
        "duration": float(total_frames / video_fps),
        "frames_indices": [int(i) for i in idx],
        "video_backend": "decord",
    }

    # Decide return shape based on what the caller asked for.
    # NOTE on shapes: qwen_vl_utils versions disagree.
    #   * Some versions:   fetch_video(...) → (video, metadata, sample_fps)
    #   * Other versions:  fetch_video(...) → ((video, metadata), sample_fps)
    #                        (i.e. the wrapper bundles the metadata INSIDE the
    #                         video return so that process_vision_info can keep
    #                         a 2-tuple unpack.)
    # The pinned qwen_vl_utils release uses the 2-tuple-with-bundled-metadata
    # variant, so we default to that when both flags are set. Override with
    # QWENVL_RETURN_SHAPE=3tuple if your stack expects the 3-element form.
    want_metadata = bool(kwargs.get("return_video_metadata", False))
    want_sample_fps = bool(kwargs.get("return_video_sample_fps", False))
    return_shape = os.getenv("QWENVL_RETURN_SHAPE", "auto").strip().lower()

    if want_metadata and want_sample_fps:
        if return_shape == "3tuple":
            return video, metadata, float(sample_fps)
        # default "auto" / "bundled" → 2-tuple form
        return (video, metadata), float(sample_fps)
    if want_metadata:
        return video, metadata
    if want_sample_fps:
        return video, float(sample_fps)
    return video


def _apply_patch() -> None:
    if os.getenv("DISABLE_QWENVL_DECORD_PATCH", "0") == "1":
        print("[qwenvl_decord_patch] disabled via env", file=sys.stderr)
        return

    try:
        from qwen_vl_utils import vision_process as vp
    except Exception as exc:  # pragma: no cover
        print(f"[qwenvl_decord_patch] qwen_vl_utils import failed: {exc}",
              file=sys.stderr)
        return

    # Make sure decord is importable up front so we fail loudly here, not
    # mid-eval, if it's not installed.
    try:
        import decord  # noqa: F401
    except Exception as exc:
        print(f"[qwenvl_decord_patch] decord not installed: {exc}",
              file=sys.stderr)
        print("                       Install with: pip install decord",
              file=sys.stderr)
        return

    # Belt: try the env-based switch (no-op on new versions).
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

    # Suspenders: replace fetch_video itself.
    if not hasattr(vp, "fetch_video"):
        print("[qwenvl_decord_patch] vp.fetch_video missing, abort",
              file=sys.stderr)
        return

    # Also alias the torchvision backend to our decord function in case any
    # downstream code re-reads VIDEO_READER_BACKENDS directly.
    backends = getattr(vp, "VIDEO_READER_BACKENDS", None)
    if isinstance(backends, dict):
        # Wrap to satisfy the SINGLE-tensor return convention used by some
        # internal callers (the older ``_read_video_torchvision`` returns
        # just a tensor in 0.0.8 path).
        def _single_tensor_decord(ele):
            v, _meta, _fps = _decord_fetch_video_new_api(ele)
            return v
        backends["torchvision"] = _single_tensor_decord

    vp.fetch_video = _decord_fetch_video_new_api

    print(
        f"[qwenvl_decord_patch] applied. "
        f"vp.fetch_video → decord (return shape auto-adapts to "
        f"return_video_metadata / return_video_sample_fps kwargs; "
        f"override with QWENVL_RETURN_SHAPE=3tuple). "
        f"file={vp.__file__}",
        file=sys.stderr,
    )


_apply_patch()
