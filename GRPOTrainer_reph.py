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

import os
import re
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch
from sentence_transformers import SentenceTransformer, util
import torch
from contextlib import nullcontext
import math
import torch.nn.functional as F
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
import deepspeed
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax
from torch.nn.utils.rnn import pad_sequence
from metrics.p_sp_utils.models import load_model
from metrics.p_sp import GRPO_P_SP
from vllm.config import CompilationConfig, CompilationLevel


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the dataset.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility (only affects this sampler).

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(
    ...     ["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4
    ... )
    >>> list(sampler)
    [4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     4, 4, 3, 3, 0, 0,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6,
     1, 1, 2, 2, 6, 6]
    ```

    ```txt
    mini_repeat_count = 3
          -   -   -
         [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                repeat_count = 2
          0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
          4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
          8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
          ---------   ---------   ---------   ---------
           ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                         batch_size = 12
    ```
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.shuffle = shuffle
        self.seed = seed

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            if seed is not None:
                self.generator.manual_seed(seed)

    def __iter__(self):
        if self.shuffle:
            # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
            indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indexes = list(range(self.num_samples))

        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count



class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT is required to use `peft_config`. Run `pip install peft`.")
            model = get_peft_model(model, peft_config)

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token


        # Reward functions
        # if not isinstance(reward_funcs, list):
        #     reward_funcs = [reward_funcs]
        # for i, reward_func in enumerate(reward_funcs):
        #     if isinstance(reward_func, str):
        #         reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
        #             reward_func, num_labels=1, **model_init_kwargs
        #         )
        # self.reward_funcs = reward_funcs
        self.reward_funcs = [self._get_semantic_similarity_reward()]
        self.klreward_threshold = args.KLreward_threshold
        self.semantic_thres = args.semantic_thres
        self.klreward_weight = args.klreward_weight
        self.ppl_weight = args.ppl_weight
        self.seman_weight = args.seman_weight
        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs)+1:
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.min_completion_length = args.min_completion_length
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.use_liger_loss = args.use_liger_loss
        # self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.mask_truncated_completions = args.mask_truncated_completions

        self.beta = args.beta

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset
        self.num_iterations = args.num_iterations


        # Reference model
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=self.accelerator.device)
        self.psp_args = {
                'gpu': 0,#1 if torch.cuda.is_available() else 0,
                'load_file': 'metrics/p_sp_utils/psp/model.para.lc.100.pt',
                'sp_model': 'metrics/p_sp_utils/psp/paranmt.model',
                'gpu_id': self.accelerator.device.index,
            }
        self.similarity_model, _ = load_model(None, self.psp_args)
        

    
        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                if args.vllm_server_base_url is not None:
                    base_url = args.vllm_server_base_url
                else:
                    base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                self.vllm_client.init_communicator()

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                self.llm = LLM(
                    model=model.name_or_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.gradient_accumulation_steps,
                    dtype=torch.bfloat16, #self.args.vllm_dtype,
                    max_model_len=self.max_prompt_length + self.max_completion_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    compilation_config=CompilationConfig(level=CompilationLevel.PIECEWISE,# By default, it goes up to max_num_seqscudagraph
                                                      cudagraph_capture_sizes=[1,2,4])
                )

            # vLLM specific sampling arguments
            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            # self.sampling_params = SamplingParams(
            #     temperature=args.temperature,
            #     top_p=0.95,
            #     top_k=20,
            #     max_tokens=self.max_completion_length,
            #     min_tokens=self.min_completion_length,
            #     # presence_penalty=0.1,
            #     # frequency_penalty=0.1,
            #     stop_token_ids=[self.processing_class.eos_token_id]
            # )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            
            self.accelerator.wait_for_everyone()
            
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": processing_class.pad_token_id,
                "bos_token_id": processing_class.bos_token_id,
                "eos_token_id": processing_class.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        # print('111111111111111111')
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)
        # print('222222222222222222')
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                # print('33333333333333333333')
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                # print('44444444444444444444')
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            # print('5555555555555555')
            print("Syncing ref model") 
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)


    def _get_semantic_similarity_reward(
        self,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
        ):
        
        def semantic_reward_p_sp(wm_completions, watermarked, **kwargs):
            # generated_texts = [completion[0]["content"] for completion in completions]

            
            y = (0.95 + 1) / 2
            k = math.log(y / (1 - y)) / (1 - self.semantic_thres)

            generated_texts = [completion for completion in wm_completions]

            scores = GRPO_P_SP(
                self.similarity_model,
                self.psp_args,
                generated_texts,
                watermarked,
            )
            rewards = []
            # Map PSP scores to a smooth reward:
            # - score = 0 maps close to -1
            # - score = midpoint maps close to 0
            # - score = 1 maps close to +1
            for score in scores:
                x = k * (score - self.semantic_thres)
                rewards.append(2 * (1 / (1 + math.exp(-x))) - 1)
            # rewards = scores
            return rewards

        return semantic_reward_p_sp
    

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    
    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded

        # logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]

        return selective_log_softmax(logits, input_ids)  #  compute logprobs for the input tokens
    

    def _get_per_token_logits(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded

        # logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]

        return logits

    def _get_per_token_entropy(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        # try:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        # except:
        #     logits = model(input_ids=input_ids, attention_mask=attention_mask, num_logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, -logits_to_keep:]

        logits = logits[:, -logits_to_keep:]
        return -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1)
    
    def _get_loss_perplexity(self, model, input_ids, attention_mask):
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
        # print(loss)

        # with torch.no_grad():
        #     logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).logits

        #     # Shift logits and labels to align predictions and targets.
        #     shift_logits = logits[:, :-1, :].contiguous()
        #     shift_labels = input_ids[:, 1:].contiguous()
        #     shift_attention_mask = attention_mask[:, 1:].contiguous()

        #     # Compute each token's loss.
        #     loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        #     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        #     loss = loss.view(shift_labels.size())

        #     # Compute average sentence loss and exponentiate it into perplexity.
        #     sum_loss = torch.sum(loss * shift_attention_mask, dim=1)
        #     lengths = torch.sum(shift_attention_mask, dim=1)
        #     avg_loss = sum_loss / lengths
        #     perplexities = torch.exp(avg_loss)

        return loss


    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                self._sync_fsdp_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
            else:
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()


    def _soft_clip(self, x, threshold=10.0, softness=2.0):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return threshold * torch.tanh(x / softness)
        # return scale * (1 - torch.exp(-x / threshold))
    
    def _standardized_soft_clip(self, x, threshold=25, eps=1e-8):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        mean = x.mean()
        std = x.std()
        x_std = (x - mean) / (std + eps)
        return threshold * torch.tanh(x_std)
    

    def _KL_smoother(self, delta, beta=2.0, threshold=1, eps=1e-8):
        """
        Map input deltas to the (-1, 1) interval while preserving rank order.
        
        Args:
            delta: List[float] or a 1D Tensor.
            beta: Controls the compression slope; larger values are smoother.

        Returns:
            Normalized rewards in (-1, 1).
        """
        if not torch.is_tensor(delta):
            reward_tensor = torch.tensor(delta, dtype=torch.float32)
        else:
            reward_tensor = delta.float()
        
        mean = reward_tensor.mean()
        std  = reward_tensor.std(unbiased=False) + eps
        normalized = (reward_tensor - mean) / std
        # normalized = torch.tanh(z / beta)
        return normalized
        # min_v = reward_tensor.amin(dim=1, keepdim=True)
        # max_v = reward_tensor.amax(dim=1, keepdim=True)
        # kl_norm = (reward_tensor - min_v) / (max_v - min_v + 1e-8)  # [0, 1]
        # kl_norm = (kl_norm - 0.5) * 2    # in [-1,1]
        # return kl_norm  # [-1, 1]

    def _get_zero_count_reward(self, kl_count, thres = 0.15):
            y = (0.95 + 1) / 2
            k = math.log(y / (1 - y)) / (1 - thres)
            x = k * (kl_count - thres)
            return -(2 * torch.sigmoid(x) - 1)


    def _compute_rewards(self, inputs, completion_ids, completion_mask, logits_to_keep):
        # prompts for similating human distribution 
        
        # question_text = [maybe_apply_chat_template(example, self.processing_class)["question"] for example in inputs]
        # original_ans = [f"{maybe_apply_chat_template(example, self.processing_class)['question']} "+f"{maybe_apply_chat_template(example, self.processing_class)['watermarked']}" for example in inputs]
        # raw_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] + '######Instruction: Rewrite the target text above using different words but keeping the same meaning and similar length.\n\n######Your Response:' for example in inputs]
        # raw_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] + '######Instruction: Generate a version of the target text above that means the same and is about the same length.\n\n######Your Response:' for example in inputs]

        # question_text = [f"######Instruction: {inputs[i]['question']}\n\n######Requirement: Your response should starts with: {self.processing_class.decode(completion_ids[i][:20], skip_special_tokens=True)}\n\n######Your Response: "  for i in range(len(inputs))]
        # question_text = [f"{inputs[i]['question']} "  for i in range(len(inputs))]
        question_text = [self.processing_class.apply_chat_template([{"role": "user", "content": inputs[i]['question']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)  for i in range(len(inputs))]
        # question_text = [f"{inputs[i]['question']}\n\n{self.processing_class.decode(completion_ids[i][:20], skip_special_tokens=True)}"  for i in range(len(inputs))]

        # raw_text = [f'######Target Text: {inputs[i]["watermarked"]}\n\n######Instruction: Generate a version of the target text above that means the same and is about the same length.\n\n######Your Response: ' for i in range(len(inputs))]

        raw_text = [self.processing_class.apply_chat_template([{"role": "user", "content": f"######Target Text: {inputs[i]['watermarked']}\n\n######Instruction: Generate a version of the target text above that means the same and is about the same length.\n\n######Your Response: "}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        for i in range(len(inputs))]

        # original_ans = [f"{example['question']} "+f"{example['watermarked']}" for example in inputs]

        question_inputs = self.processing_class(
            question_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        raw_inputs = self.processing_class(
            raw_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        # original_ans_inputs = self.processing_class(
        #     original_ans, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        # )

        question_inputs = super()._prepare_inputs(question_inputs)
        raw_inputs = super()._prepare_inputs(raw_inputs)
        # original_ans_inputs = super()._prepare_inputs(original_ans_inputs)

        question_ids, question_mask = question_inputs["input_ids"], question_inputs["attention_mask"]
        raw_ids, raw_mask = raw_inputs["input_ids"], raw_inputs["attention_mask"]

        ph_ids, ph_mask = torch.cat([question_ids, completion_ids], dim=1), torch.cat([question_mask, completion_mask], dim=1)
        pwm_ids, pwm_mask = torch.cat([raw_ids, completion_ids], dim=1), torch.cat([raw_mask, completion_mask], dim=1)

        with torch.inference_mode():
            if self.ref_model is not None:
                ph_per_token_logps = self._get_per_token_logps(
                            self.ref_model, ph_ids, ph_mask, logits_to_keep
                    )
                pwm_per_token_logps = self._get_per_token_logps(
                            self.ref_model, pwm_ids, pwm_mask, logits_to_keep
                    )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ph_per_token_logps = self._get_per_token_logps(
                                self.model, ph_ids, ph_mask, logits_to_keep
                        )
                    pwm_per_token_logps = self._get_per_token_logps(
                                self.model, pwm_ids, pwm_mask, logits_to_keep
                        )
            pa_per_token_logps = self._get_per_token_logps(
                self.model, pwm_ids, pwm_mask, logits_to_keep
            )

        # original_ans_perplex_loss = self._get_loss_perplexity(self.model, original_ans_inputs["input_ids"], original_ans_inputs["attention_mask"])
        perplex_loss = self._get_loss_perplexity(self.model, ph_ids, ph_mask)
            # original_ans_perplex_loss = self._get_loss_perplexity(self.ref_model, original_ans_inputs["input_ids"], original_ans_inputs["attention_mask"])
        # print(f"original_ans_perplex_loss: {original_ans_perplex_loss}")
        # print(f"perplex_loss: {perplex_loss}")

        # perplex_loss = torch.abs(perplex_loss - original_ans_perplex_loss)  # shape: [batch, logits_to_keep]

        KL_h = torch.exp(ph_per_token_logps - pa_per_token_logps) - (ph_per_token_logps - pa_per_token_logps) - 1
        KL_wm = torch.exp(pwm_per_token_logps - pa_per_token_logps) - (pwm_per_token_logps - pa_per_token_logps) - 1

        kl_reward = KL_wm - KL_h

        

        kl_reward_std = 0# torch.std(kl_reward, dim=-1, unbiased=False)

        # Reducing the distance between ph and pw also works reasonably well.
        # KL_wmh = torch.exp(ph_per_token_logps - pwm_per_token_logps) - (ph_per_token_logps - pwm_per_token_logps) - 1
        # ph || pwm
        # KL_wmh = torch.exp(pwm_per_token_logps - ph_per_token_logps) - (pwm_per_token_logps - ph_per_token_logps) - 1
        soft_zero_mask = torch.sigmoid((1e-6 - kl_reward) * 200)
        kl_zero_bonus = soft_zero_mask.sum(dim=-1) / logits_to_keep  # shape: [batch]
        # Increase the distance between them.

        # return combined_kl.tolist()
        return kl_reward, kl_zero_bonus, perplex_loss, kl_reward_std#original_ans_perplex_loss KL_h, KL_wm, 
        
   
    def _generating_contexts(self, prompt_ids, prompt_mask, prompts_text, prompts, device, is_conversation):
        
        
        local_processing_class = self.processing_class
            

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation, this can generate the context a
        if self.use_vllm:
            # First, update the vLLM weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
                    with profiling_context(self, "vLLM.generate"):
                        completion_ids = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            guided_decoding_regex=self.guided_decoding_regex,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                if self.guided_decoding_regex:
                    guided_decoding = GuidedDecodingParams(backend="outlines", regex=self.guided_decoding_regex)
                else:
                    guided_decoding = None

                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "min_tokens": self.min_completion_length,
                    "guided_decoding": guided_decoding,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text = prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(all_prompts_text, sampling_params=sampling_params, use_tqdm=False)

                completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = completion_ids[tp_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                with (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == local_processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Decode the generated completions
        completions_text = local_processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversation:
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        return completion_ids, completion_mask, completions, completions_text, prompt_completion_ids, attention_mask, logits_to_keep
    

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts = [f"{x['prompt']}######Instruction: Generate a version of the target text above that means the same and is about the same length.\n\n######Your Response:" for x in inputs]

        # prompts = [f"{x['prompt']}######Instruction: Generate a version of the target text above that means the same and is about the same length, without any modifiers. Output only the rephrased text.\n\n######Your Response:" for x in inputs]

        # prompts_text = [f"{maybe_apply_chat_template(example, self.processing_class)['prompt']}######Instruction: Generate a version of the target text above that means the same and is about the same length.\n\n######Your Response:" for example in inputs]
        prompts_text = [self.processing_class.apply_chat_template([{"role": "user", "content": f"{example['prompt']}######Instruction: Generate a version of the target text above that means the same and is about the same length.\n\n######Your Response: "}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        for example in inputs]

        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )


        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        completion_ids, completion_mask, completions, completions_text, prompt_completion_ids, attention_mask, logits_to_keep = self._generating_contexts(prompt_ids, prompt_mask, prompts_text, prompts, device, is_conversational(inputs[0]))

        # print(completions)

        ####################################
        # prepare for generating the \hat{W}
        ####################################
        
        # print("attention_mask:", torch.all(attention_mask == 1))
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        rewards_per_func = torch.zeros(len(prompts), int(len(self.reward_funcs)+1), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, wm_completions=completions, wm_completion_ids=completion_ids, **reward_kwargs) # the similarity
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        
        
        kl_reward, kl_zero_bonus, perplex_loss, kl_reward_std = self._compute_rewards(inputs, completion_ids, completion_mask, logits_to_keep)
        
        # zero_mask = torch.isclose(KL_wmh, torch.tensor(0.0, dtype=KL_wmh.dtype, device=KL_wmh.device), atol=1e-6)

        # Count near-zero tokens in each sample as a bonus.
        # kl_zero_bonus = zero_mask.float().sum(dim=-1)/logits_to_keep  # shape: [batch]
        # kl_zero_bonus = self._get_zero_count_reward(kl_zero_bonus)

        # softmask
        # soft_zero_mask = torch.exp(-KL_wmh*10)
        
        rewards_per_func[:,-1] = 0 #self._get_zero_count_reward(kl_zero_bonus) 

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # # Compute grouped-wise rewards ORIGINAL GRPO
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        

        # GRPO Original
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # advantages = rewards

        # AGPO
        # advantages = rewards - mean_grouped_rewards

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        sim_rewards = rewards[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        # self._metrics[f"rewards/KlZeroNum"].append(reward_per_func[-1].item())
        self._metrics[f"rewards/perplexity"].append(perplex_loss.item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        # question_text = [f"{inputs[i]['question']}"  for i in range(len(inputs))]
        # question_inputs = self.processing_class(
        #             question_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        #         )
        # question_inputs = super()._prepare_inputs(question_inputs)
        # question_ids, question_mask = question_inputs["input_ids"], question_inputs["attention_mask"]

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "sim_rewards": sim_rewards,
            "kl_reward": kl_reward, 
            "kl_zero_bonus": kl_zero_bonus,
            "kl_reward_std": kl_reward_std,
            "perplex_loss": perplex_loss,
            # "KL_wmh": KL_wmh,
            # "original_ans_perplex_loss": original_ans_perplex_loss
        }

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        kl_reward = inputs["kl_reward"]
        perplex_loss = inputs['perplex_loss']
        sim_rewards = inputs['sim_rewards']

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]

        # original
        # total_reward = advantages.unsqueeze(1) + self.klreward_weight*kl_reward # [B, L]0.5

        #dynamic weights
        # kl_mean = torch.abs(kl_reward.median()) + 1e-6       # shape: []
        # flat = torch.abs(kl_reward.flatten().to(torch.float32))
        # flat = kl_reward.to(torch.float32)
        # q05, q95 = torch.quantile(flat, 0.1), torch.quantile(flat, 0.9)
        # kl_mean = self.klreward_weight*torch.abs(flat[(flat >= q05) & (flat <= q95)].mean())    # shape: []
        # # kl_mean = flat.mean()
        # adv_mean = torch.abs(advantages).mean()    # shape: []

        # flat = kl_reward.to(torch.float32)
        # kl_means = []
        # for i in range(kl_reward.shape[0]):
        #     row = flat[i]  # shape: [L]
        #     q10, q90 = torch.quantile(row, 0.05), torch.quantile(row, 0.95)
        #     filtered = row[(row >= q10) & (row <= q90)]
        #     kl_means.append(torch.abs(filtered.mean()))
        # kl_mean = torch.stack(kl_means)
        # adv_mean = torch.abs(advantages)     # shape: []
        # alpha =  kl_mean*self.klreward_weight / (adv_mean + 1e-8)
        # alpha = alpha.unsqueeze(1)

        # for Unigram new
        # alpha = self.seman_weight*(1-sim_rewards.mean())*kl_mean if self.seman_weight*(1-sim_rewards.mean())*(self.klreward_weight*kl_mean)>1 else 1

        # for PF,EWD Unigram old
        alpha = self.seman_weight*(1-sim_rewards.mean()) if self.seman_weight*(1-sim_rewards.mean())>1 else 1


        # for PF,EWD Unigram New
        # alpha = self.seman_weight*(1-sim_rewards.mean())

        # print(alpha)

        total_reward = alpha * advantages.unsqueeze(1) + self.klreward_weight*kl_reward # [B, L]0.5

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * total_reward


        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl + advantages.unsqueeze(1))
        per_token_loss = -(per_token_loss - self.beta * per_token_kl) # - KL_wmh
        
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean() + perplex_loss * self.ppl_weight #+ 0.5*kl_zero_bonus.mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        mean_kl_rewards = ((kl_reward * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl_reward"].append(self.accelerator.gather_for_metrics(mean_kl_rewards).mean().item())

        # self._metrics["kl_mean"].append(self.accelerator.gather_for_metrics(kl_mean).mean().item())
        # self._metrics["kl_reward-std"].append(self.accelerator.gather_for_metrics(kl_reward_std.mean()).mean().item())
        # mean_kl_zero_bonus = kl_zero_bonus.mean()
        # self._metrics["kl_zero_bonus"].append(self.accelerator.gather_for_metrics(mean_kl_zero_bonus).mean().item())

        return loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()


    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
