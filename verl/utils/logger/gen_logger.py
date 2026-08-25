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


import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from ..py_functional import is_package_available


if is_package_available("wandb"):
    import wandb  # type: ignore


if is_package_available("swanlab"):
    import swanlab  # type: ignore


GenerationSample = Union[Tuple[str, str, str, float], Tuple[str, str, str, float, Any]]


def _unpack_generation_sample(sample: GenerationSample) -> Tuple[str, str, str, float, Any]:
    """Unify generation sample shape with backward compatibility."""
    if len(sample) == 4:
        inp, out, lab, score = sample
        return inp, out, lab, score, None
    if len(sample) == 5:
        inp, out, lab, score, problem_id = sample
        return inp, out, lab, score, problem_id
    raise ValueError(f"Invalid generation sample format with length={len(sample)}.")


@dataclass
class GenerationLogger(ABC):
    config: dict[str, Any]

    @abstractmethod
    def log(self, samples: List[GenerationSample], step: int) -> None: ...


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    def log(self, samples: List[GenerationSample], step: int) -> None:
        for sample in samples:
            inp, out, lab, score, problem_id = _unpack_generation_sample(sample)
            print(
                f"[problem_id] {problem_id}\n[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n"
            )


@dataclass
class FileGenerationLogger(GenerationLogger):
    def log(self, samples: List[GenerationSample], step: int) -> None:
        with open(os.path.join(self.config["trainer"]["save_checkpoint_path"], "generations.log"), "a") as f:
            for sample in samples:
                inp, out, lab, score, problem_id = _unpack_generation_sample(sample)
                f.write(
                    f"[problem_id] {problem_id}\n[prompt] {inp}\n[output] {out}\n[ground_truth] {lab}\n[score] {score}\n\n"
                )


@dataclass
class WandbGenerationLogger(GenerationLogger):
    def log(self, samples: List[GenerationSample], step: int) -> None:
        # Create column names for all samples
        columns = ["step"] + sum(
            [
                [f"problem_id_{i + 1}", f"input_{i + 1}", f"output_{i + 1}", f"label_{i + 1}", f"score_{i + 1}"]
                for i in range(len(samples))
            ],
            [],
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = [step]
        for sample in samples:
            inp, out, lab, score, problem_id = _unpack_generation_sample(sample)
            row_data.extend([problem_id, inp, out, lab, score])

        new_table.add_data(*row_data)
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table


@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: List[GenerationSample], step: int) -> None:
        swanlab_text_list = []
        for i, sample in enumerate(samples):
            inp, out, lab, score, problem_id = _unpack_generation_sample(sample)
            row_text = "\n\n---\n\n".join(
                (
                    f"problem_id: {problem_id}",
                    f"input: {inp}",
                    f"output: {out}",
                    f"label: {lab}",
                    f"score: {score}",
                )
            )
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        swanlab.log({"val/generations": swanlab_text_list}, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "file": FileGenerationLogger,
    "wandb": WandbGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


class AggregateGenerationsLogger:
    def __init__(self, loggers: List[str], config: Optional[dict[str, Any]] = None):
        self.loggers: List[GenerationLogger] = []

        for logger in loggers:
            if logger in GEN_LOGGERS:
                self.loggers.append(GEN_LOGGERS[logger](config))

    def log(self, samples: List[GenerationSample], step: int) -> None:
        for logger in self.loggers:
            logger.log(samples, step)
