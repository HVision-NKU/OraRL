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

from __future__ import annotations

from typing import Any


QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer this question based on the visual content."
    "Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    "At the end, you must output the final answer in the format:\n"
    "<answer><your_answer_here></answer>\n"
)

# NOTE: <think> tag is NOT required. Only <answer> tag is mandatory.
# This allows both Instruct and Thinking models to work without format penalty.
# QUESTION_TEMPLATE = (
#     "{Question}\n"
#     "Please answer this question based on the visual content. "
#     "Provide your final answer within the <answer>...</answer> tags.\n"
# )

TYPE_TEMPLATE = {
    "multiple choice": (
        "Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A</answer>"
    ),
    "numerical": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>3.14</answer>"
    ),
    "OCR": (
        "Please provide only the transcribed text within the <answer>...</answer> tags.\n"
        "Example:\n<answer>Hello World</answer>"
    ),
    "ocr": (
        "Please provide only the transcribed text within the <answer>...</answer> tags.\n"
        "Example:\n<answer>Hello World</answer>"
    ),
    "open-ended": (
        "Please provide only your text answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The capital of France is Paris.</answer>"
    ),
    "regression": (
        "Please provide only the numerical value within the <answer>...</answer> tags.\n"
        "Example:\n<answer>42.7</answer>"
    ),
    "math": (
        "Please provide only the final answer within the <answer>...</answer> tags.\n"
        "For multiple choice questions, provide the option letter (e.g., A, B, C, D).\n"
        "For calculation problems, provide the numerical result or LaTeX formula.\n"
        "Examples:\n"
        "<answer>B</answer>\n"
        "<answer>42</answer>\n"
        "<answer>$$-\\dfrac{3}{2}$$</answer>"
    ),
    "temporal grounding": (
        "Please provide only the time span in seconds as JSON within the <answer>...</answer> tags.\n"
        'Example:\n<answer>{"time": [12.3, 25.7]}</answer>'
    ),
    "spatial grounding": (
        "Please provide only the bounding box as JSON with key 'boxes' within the <answer>...</answer> tags.\n"
        'Example:\n<answer>{"boxes": [35, 227, 437, 932]}</answer>'
    ),
    "spatial-temporal grounding": (
        "Please provide only the time span in seconds and bounding boxes as JSON within the <answer>...</answer> tags.\n"
        "You MUST output one bounding box for every integer second within the given time span (inclusive).\n"
        "Example:\n"
        '<answer>{"time": [8.125, 13.483], "boxes": {"9": [317, 422, 582, 997], '
        '"10": [332, 175, 442, 369], "11": [340, 180, 450, 370]}}</answer>\n'
        "Note: Each key in 'boxes' must be an integer second within the span, and its value must be a 4-number bounding box [x1, y1, x2, y2]."
    ),
    "tracking": (
        "Please track the target object throughout the video and provide one bounding box per second, "
        "ONLY up to 32 seconds, within the <answer>...</answer> tags.\n"
        "Example:\n"
        '<answer>{"boxes": {"1": [405, 230, 654, 463], "2": [435, 223, 678, 446], ..., '
        '"32": [415, 203, 691, 487]}}</answer>\n'
        "Note: Each key in 'boxes' must correspond to a second (1, 2, 3, ..., 32) and contain a 4-number bounding box [x1, y1, x2, y2]."
    ),
    "segmentation_image": (
        "This task prepares inputs for image object segmentation with a specialized model (e.g., SAM2).\n"
        "Please provide ONE bounding box, 3 positive points (clearly INSIDE the object), and 3 negative points "
        "(clearly OUTSIDE the object) within the <answer>...</answer> tags.\n"
        "Choose informative points that help distinguish object vs. background. Prefer negatives on clear non-object "
        "pixels INSIDE the box when safe; otherwise place them just outside on obvious background. "
        "Negatives must NEVER be on the object or on its boundary.\n"
        "Example:\n"
        '<answer>{"boxes": [x1, y1, x2, y2], "positive_points": [[x,y],[x,y],[x,y]], '
        '"negative_points": [[x,y],[x,y],[x,y]]}</answer>'
    ),
    "segmentation_video": (
        "This task prepares inputs for video object segmentation with a specialized model (e.g., SAM2).\n"
        "Please select ONE representative time (in seconds), and provide ONE bounding box, "
        "3 positive points (clearly INSIDE the object), and 3 negative points (clearly OUTSIDE the object) "
        "within the <answer>...</answer> tags.\n"
        "Choose informative points that help distinguish object vs. background. Prefer negatives on clear non-object "
        "pixels INSIDE the box when safe; otherwise place them just outside on obvious background. "
        "Negatives must NEVER be on the object or on its boundary.\n"
        "Example:\n"
        '<answer>{"time": <time_in_seconds>, "boxes": [x1, y1, x2, y2], '
        '"positive_points": [[x,y],[x,y],[x,y]], "negative_points": [[x,y],[x,y],[x,y]]}</answer>'
    ),
    # ===== Additional task types =====
    "code": (
        "Please provide only the complete Python code within the <answer>...</answer> tags.\n"
        "Make sure your code is properly formatted and includes all necessary imports.\n"
        "Example:\n"
        "<answer>\n"
        "def solve(lst):\n"
        "    # Your implementation\n"
        "    return result\n"
        "</answer>"
    ),
    "svg-code": (
        "Please provide only the complete SVG code within the <answer>...</answer> tags.\n"
        "Example:\n"
        "<answer>\n"
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">\n'
        '  <circle cx="50" cy="50" r="40" fill="blue"/>\n'
        "</svg>\n"
        "</answer>"
    ),
    "html-code": (
        "Please provide only the complete HTML code within the <answer>...</answer> tags.\n"
        "Example:\n"
        "<answer>\n"
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>Page</title></head>\n"
        "<body><h1>Hello</h1></body>\n"
        "</html>\n"
        "</answer>"
    ),
    "boolean": (
        "Please provide only 'Yes' or 'No' within the <answer>...</answer> tags.\nExample:\n<answer>Yes</answer>"
    ),
    "binary classification": (
        "Please provide only 'Yes' or 'No' within the <answer>...</answer> tags.\nExample:\n<answer>Yes</answer>"
    ),
    "llava": (
        "Please compare the two responses and determine which one is better.\n"
        "Provide your answer as one of: 'Response 1', 'Response 2', or 'Tie' "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>Response 1</answer>"
    ),
    "video qa": (
        "Please provide only your text answer based on the video content "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>The person is playing basketball.</answer>"
    ),
    "video description": (
        "Please provide a detailed description of what you see in the video "
        "within the <answer>...</answer> tags.\n"
        "Example:\n<answer>A man is walking down the street carrying a bag.</answer>"
    ),
    "free-form": (
        "Please provide your answer within the <answer>...</answer> tags.\n"
        "Example:\n<answer>Your answer here.</answer>"
    ),
}


def build_prompt(prompt_str: str, example: dict[str, Any]) -> str:
    data_type = (example.get("data_type") or "").strip().lower()
    problem_type = example.get("problem_type") or ""
    question = prompt_str

    if problem_type == "multiple choice" and isinstance(example.get("options"), list) and example["options"]:
        opts = "\n".join(example["options"])
        question = f"{question}\nOptions:\n{opts}"

    if problem_type == "segmentation":
        type_key = "segmentation_video" if data_type == "video" else "segmentation_image"
    else:
        type_key = problem_type

    tail = TYPE_TEMPLATE.get(type_key, "")
    return QUESTION_TEMPLATE.format(Question=question) + tail
