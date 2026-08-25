#!/usr/bin/env python3
"""Single source of truth for evaluation prompts.

Exactly ONE prompt per task — the one used by ``eval/task/eval.sh`` — kept
byte-for-byte aligned with the SFT training data in
``data/joint/sft_joint_all.jsonl``.

Task -> prompt mapping (verbatim from sft_joint_all.jsonl):

  video_qa_mc (videomme, videommmu, mmvu, mvbench, videoholmes,
               longvideobench, lvbench, mlvu)
      "{question}\\nOptions:\\n{opts}\\n" + VIDEO_QA_MC_TAIL

  spatial_intelligence (vsi)
      MC      -> "{q}\\nOptions:\\n{opts}\\n" + VSI_MC_TAIL
      integer -> "{q} " + VSI_INTEGER_TAIL
      meters  -> "{q} " + VSI_METERS_TAIL
      cm      -> "{q} " + VSI_CENTIMETERS_TAIL
      room m2 -> "{q} " + VSI_SQUARE_METERS_TAIL
      (each prefixed with "These are frames of a video.\\n")

  sensenova_si / image_sequence_mc (mmsi, mindcube)
      "{q}\\n" + SENSENOVA_MC_TAIL   (alias PROMPT_TAIL)

  spatial grounding (spatial_grounding)
      build_spatial_grounding_prompt(expr)  == QWEN_NATIVE_PROMPT_SG

  tracking
      GROUNDING_QUESTION_TEMPLATE_NO_THINK.format(Question=q) + TRACKING_TAIL

  stvg
      GROUNDING_QUESTION_TEMPLATE_NO_THINK.format(
          Question=TRAIN_STVG_QUESTION_PREFIX.format(query=q)) + STVG_TAIL

  segmentation
      "{question}\\n" + TRAIN_SEG_IMAGE_TAIL / TRAIN_SEG_VIDEO_TAIL

  temporal grounding (temporal_grounding)
      TEMPORAL_GROUNDING_PROMPT.format(event)

Keep this file dependency-free (stdlib only) so it can be imported from every
worker regardless of the runtime environment.
"""

# ===========================================================================
# Multiple-choice / numeric answer tails
# ===========================================================================

# video_qa_mc — videomme, videommmu, mmvu, mvbench, videoholmes,
# longvideobench, lvbench, mlvu.
VIDEO_QA_MC_TAIL = (
    "Answer with the option letter only within <answer>...</answer> tags. "
    "Example: <answer>A</answer>"
)

# spatial_intelligence multiple choice — vsi (object_rel_distance,
# object_rel_direction, obj_appearance_order, route_plan, ...).
VSI_MC_TAIL = (
    "Answer with the option letter within <answer>...</answer> tags. "
    "Example: <answer>A</answer>"
)

# spatial_intelligence numeric — vsi.
VSI_INTEGER_TAIL = (
    "Answer with an integer within <answer>...</answer> tags. Example: <answer>3</answer>"
)
VSI_METERS_TAIL = (
    "Answer with a number in meters within <answer>...</answer> tags. Example: <answer>2.3</answer>"
)
VSI_CENTIMETERS_TAIL = (
    "Answer with a number in centimeters within <answer>...</answer> tags. Example: <answer>120.5</answer>"
)
VSI_SQUARE_METERS_TAIL = (
    "Answer with a number in square meters within <answer>...</answer> tags. Example: <answer>25.5</answer>"
)

# sensenova_si / image_sequence_mc_answer_only — mmsi, mindcube.
SENSENOVA_MC_TAIL = (
    "Choose the best answer from the options. "
    "Put exactly one uppercase option letter inside <answer>...</answer> "
    "Do not explain. Example: <answer>A</answer>"
)
# Backwards-compatible alias used throughout eval_vllm.py.
PROMPT_TAIL = SENSENOVA_MC_TAIL


# ===========================================================================
# Tracking
# ===========================================================================

# Question scaffold shared by tracking + stvg (the tail carries the full
# answer-format spec).
GROUNDING_QUESTION_TEMPLATE_NO_THINK = (
    "{Question}\n"
    "Please answer this question based on the visual content. "
)

TRACKING_TAIL = (
    "Please track the target object throughout the video and provide one bounding box per second, "
    "ONLY up to 32 seconds, within the <answer>...</answer> tags.\n"
    "Example:\n"
    "<answer>{\"boxes\": {\"1\": [405, 230, 654, 463], \"2\": [435, 223, 678, 446], "
    "\"32\": [415, 203, 691, 487]}}</answer>\n"
    "Note: Each key in 'boxes' must correspond to a second (1, 2, 3, ..., 32) "
    "and contain a 4-number bounding box [x1, y1, x2, y2]."
)


# ===========================================================================
# Spatial-temporal grounding (STVG)
# ===========================================================================

STVG_TAIL = (
    "Please provide only the time span in seconds and bounding boxes as JSON "
    "within the <answer>...</answer> tags.\n"
    "You MUST output one bounding box for every integer second within the "
    "given time span (inclusive).\n"
    "Example:\n"
    "<answer>{\"time\": [8.1, 13.5], \"boxes\": {\"9\": [317, 422, 582, 997], "
    "\"10\": [332, 175, 442, 369], \"11\": [340, 180, 450, 370]}}</answer>\n"
    "Note: Each key in 'boxes' must be an integer second within the span, "
    "and its value must be a 4-number bounding box [x1, y1, x2, y2]."
)

# Rewrites an eval question into the exact wording used during STVG training.
TRAIN_STVG_QUESTION_PREFIX = (
    'Given the query "{query}", when and where does the described content '
    'occur in the video? please firstly give the start and end time, spatial '
    'bounding box corresponding to each integer second.'
)


# ===========================================================================
# Segmentation
# ===========================================================================

# Matches sft_joint_all.jsonl seg_image / seg_video samples.
TRAIN_SEG_IMAGE_TAIL = (
    "Please answer this question based on the visual content. "
    "This task prepares inputs for image object segmentation with a specialized model (e.g., SAM2).\n"
    "Please provide ONE bounding box, 3 positive points (clearly INSIDE the object), "
    "and 3 negative points (clearly OUTSIDE the object) within the <answer>...</answer> tags.\n"
    "Choose informative points that help distinguish object vs. background. Prefer negatives on clear non-object "
    "pixels INSIDE the box when safe; otherwise place them just outside on obvious background. "
    "Negatives must NEVER be on the object or on its boundary.\n"
    "Example: <answer>{\"boxes\": [x1, y1, x2, y2], \"positive_points\": [[x,y],[x,y],[x,y]], "
    "\"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
)

TRAIN_SEG_VIDEO_TAIL = (
    "Please answer this question based on the visual content. "
    "This task prepares inputs for video object segmentation with a specialized model (e.g., SAM2).\n"
    "Please select ONE representative time (in seconds), and provide ONE bounding box, "
    "3 positive points (clearly INSIDE the object), and 3 negative points (clearly OUTSIDE the object) "
    "within the <answer>...</answer> tags.\n"
    "Choose informative points that help distinguish object vs. background. Prefer negatives on clear non-object "
    "pixels INSIDE the box when safe; otherwise place them just outside on obvious background. "
    "Negatives must NEVER be on the object or on its boundary.\n"
    "Example: <answer>{\"time\": <time_in_seconds>, \"boxes\": [x1, y1, x2, y2], "
    "\"positive_points\": [[x,y],[x,y],[x,y]], \"negative_points\": [[x,y],[x,y],[x,y]]}</answer>"
)


# ===========================================================================
# Spatial grounding (RefCOCO / spatial_grounding)
# ===========================================================================

# Matches sft_joint_all.jsonl spatial grounding samples after the <image> token.
QWEN_NATIVE_PROMPT_SG = (
    'Locate "{}" in the image. Output its bounding box in JSON format '
    'within <answer>...</answer> tags. '
    'Example: <answer>[{{"bbox_2d": [123, 30, 404, 846]}}]</answer>'
)


# ===========================================================================
# Temporal grounding (TimeLens / temporal_grounding)
# ===========================================================================

# Matches sft_joint_all.jsonl temporal grounding samples (after <video>).
TEMPORAL_GROUNDING_PROMPT = (
    'To accurately pinpoint the event "{}" in the video, '
    "determine the precise time period of the event. "
    "Provide the start and end times (in seconds) "
    'in the format "start time to end time" within <answer> </answer> tags. '
    "Example: <answer> 12 to 18 </answer>"
)
# Exact prompt used by the official TimeLens evaluation.
TIMELENS_OFFICIAL_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)
# Alias used by eval_timelens_hf.py.
PROMPT_WO_THINK = TEMPORAL_GROUNDING_PROMPT
TEMPORAL_GROUNDING_PROMPT_MODES = {
    "same": TEMPORAL_GROUNDING_PROMPT,
    "timelens_official": TIMELENS_OFFICIAL_PROMPT,
}


# ===========================================================================
# Builder helpers
# ===========================================================================

def build_video_qa_mc_prompt(question, options_block):
    """`{question}\\nOptions:\\n{options_block}\\n{VIDEO_QA_MC_TAIL}`."""
    return f"{question}\nOptions:\n{options_block}\n{VIDEO_QA_MC_TAIL}"


def build_sensenova_mc_prompt(question):
    """`{question}\\n{SENSENOVA_MC_TAIL}`."""
    return f"{question}\n{SENSENOVA_MC_TAIL}"


def build_spatial_grounding_prompt(expression):
    """Training-aligned spatial grounding prompt."""
    expression = (expression or "").strip()
    if expression and expression[-1] not in ".?!":
        expression += "."
    return QWEN_NATIVE_PROMPT_SG.format(expression)


def build_tracking_prompt(question):
    return GROUNDING_QUESTION_TEMPLATE_NO_THINK.format(Question=question) + TRACKING_TAIL


def build_stvg_prompt(raw_query):
    q = TRAIN_STVG_QUESTION_PREFIX.format(query=raw_query)
    return GROUNDING_QUESTION_TEMPLATE_NO_THINK.format(Question=q) + STVG_TAIL


def build_seg_prompt(question, data_type):
    tail = TRAIN_SEG_VIDEO_TAIL if str(data_type).strip().lower() == "video" else TRAIN_SEG_IMAGE_TAIL
    return f"{question}\n{tail}"


def build_temporal_grounding_prompt(event, prompt_mode="same"):
    try:
        template = TEMPORAL_GROUNDING_PROMPT_MODES[prompt_mode]
    except KeyError as exc:
        choices = ", ".join(sorted(TEMPORAL_GROUNDING_PROMPT_MODES))
        raise ValueError(
            f"Unknown temporal grounding prompt mode {prompt_mode!r}; "
            f"choose one of: {choices}"
        ) from exc
    return template.format(event)
