# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GRPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    max_completion_length_gen: Optional[int] = field(
        default=1600,
        metadata={"help": "Maximum length of the generated completion."},
    )
    min_completion_length: Optional[int] = field(
        default=1100,
        metadata={"help": "Minimum length of the generated completion."},
    )
    KLreward_threshold: Optional[float] = field(
        default=1.0,
        metadata={"help": "Minimum length of the generated completion."},
    )
    semantic_thres: Optional[float] = field(
        default=0.85,
        metadata={"help": "Minimum length of the generated completion."},
    )
    klreward_weight: Optional[float] = field(
        default=0.5,
        metadata={"help": "klreward_weight."},
    )
    seman_weight: Optional[float] = field(
        default=25,
        metadata={"help": "klreward_weight."},
    )
    ppl_weight: Optional[float] = field(
        default=0.1,
        metadata={"help": "ppl_weight."},
    )
    generater_path: Optional[str] = field(
        default=None,
        metadata={"help": "generator path."},
    )
    train_data_path: Optional[str] = field(
        default='path/to/RLCracker/ghostbuster_wp_rewritten/GRPO_shortData/500pieces/llama3.1-500toks-train-500.json',
        metadata={"help": "train data path."},
    )
    test_data_path: Optional[str] = field(
        default='path/to/RLCracker/ghostbuster_wp_rewritten/GRPO_shortData/llama3.1-500toks-test-20.json',
        metadata={"help": "test data path."},
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
