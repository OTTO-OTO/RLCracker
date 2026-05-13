# RLCracker: Evaluating the Worst-Case Vulnerability of LLM Watermarks with Adaptive RL Attacks

Official code repository for the ICML 2026 accepted paper:

**RLCracker: Evaluating the Worst-Case Vulnerability of LLM Watermarks with Adaptive RL Attacks**

**Paper link:** https://arxiv.org/abs/2509.20924

RLCracker is an RL algorithm for evaluating the worst-case robustness of LLM text watermarks under adaptive paraphrasing attacks. It trains a detector-free rephrasing policy from a small number of prompt--watermarked response pairs and uses distributional rewards to move generated text away from watermark-induced behavior while preserving semantic fidelity.

This repository contains the code for reproducing the RLCracker training and evaluation pipeline.

## Overview

Large language model watermarking is commonly evaluated against fixed prompts, standard paraphrasers, or non-adaptive rewriting attacks. RLCracker is designed as a stronger robustness stress test: it adaptively trains a rephraser to remove watermark signals without querying the watermark detector.

At a high level, RLCracker:

- trains from prompt--watermarked response pairs;
- does not require access to the watermark detector during training or evaluation;
- optimizes semantic preservation together with token-level distributional rewards;
- evaluates watermark robustness using evasion success rate, semantic similarity, and quality metrics;
- supports experiments across multiple watermarking schemes, model sizes, and text lengths.

The training code is built from [Hugging Face Open-R1](https://github.com/huggingface/open-r1), and the watermarked data is generated with [THU-BPM/MarkLLM](https://github.com/THU-BPM/MarkLLM). The `MarkLLM/` directory in this repository stores the MarkLLM configs and datasets used by the experiments.

## Repository Layout

```text
.
├── config_grpo_demo_reph.yaml      # Example GRPO config
├── grpo_reph.py                    # Training entry point
├── GRPOTrainer_reph.py             # Modified GRPO trainer and custom reward logic
├── evaluation.py                   # Rephrasing and watermark-detection evaluation
├── data_utils.py                   # Data preparation and result aggregation helpers
├── multi_thread.py                 # Multi-GPU batch launcher for training/evaluation
├── open_r1/                        # Code adapted from Hugging Face Open-R1
├── watermarks/                     # Watermark detection utilities
├── metrics/                        # PSP, semantic similarity, and retrieval metrics
└── MarkLLM/                        # MarkLLM configs and generated/used datasets
```

## Setup

Create a Python 3.10+ environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Some runs require multi-GPU execution with `accelerate`, `vllm`, and `trl`. For gated Hugging Face models such as Llama, accept the model license and log in first:

```bash
huggingface-cli login
```

## Models and Paths

The scripts use Hugging Face model IDs where possible. Common model IDs are:

- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen3-8B`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `sentence-transformers/all-MiniLM-L6-v2`

Machine-specific absolute paths have been replaced by placeholders such as:

```text
path/to/RLCracker/datasets/...
path/to/RLCracker/TRAINED_MODELS/...
```

Replace these placeholders with your local data, checkpoint, and result directories before running the scripts.

## Data Flow

1. Generate watermarked text with MarkLLM using configs under `MarkLLM/config/` and source datasets under `MarkLLM/dataset/`.

2. Convert or filter the MarkLLM outputs into train/eval JSON files under a structure such as:

```text
path/to/RLCracker/datasets/TRAIN_DATA/WM_GEN_Results_Filtered_short/<generator_model>/<watermark>/train_100.json
path/to/RLCracker/datasets/TRAIN_DATA/WM_GEN_Results_Filtered_short/<generator_model>/<watermark>/validation_20.json
path/to/RLCracker/datasets/TEST_DATA/WM_GEN_Results_Filtered_short/<generator_model>/Watermark_Test_<watermark>.json
```

Each training/evaluation record is expected to contain at least:

```json
{
  "question": "...",
  "watermarked": "..."
}
```

`data_utils.py` contains helper functions for preparing train/test splits, computing success rates, exporting TensorBoard scalars, and merging comparative Pass@K results.

## Run Instructions

This section describes how to run RLCracker training and evaluation, including single-run commands and the multi-GPU batch launcher in `multi_thread.py`.

### 1. Single Training Run

`grpo_reph.py` loads the train/validation JSON files, wraps each `watermarked` field as a rephrasing prompt, and trains with the custom `GRPOTrainer` from `GRPOTrainer_reph.py`.

Example training command:

```bash
accelerate launch \
  --config_file path/to/RLCracker/open_r1/accelerate_configs/zero2.yaml \
  --num_processes 4 \
  path/to/RLCracker/grpo_reph.py \
  --config path/to/RLCracker/config_grpo_demo_reph.yaml \
  --model_name_or_path Qwen/Qwen3-8B \
  --max_completion_length 600 \
  --min_completion_length 10 \
  --per_device_eval_batch_size 6 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2 \
  --vllm_tensor_parallel_size 4 \
  --vllm_gpu_memory_utilization 0.08 \
  --output_dir path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/Qwen3_8B-EWD-Sem09KLRew01PPL5e7_dynaWeiSimAdv15 \
  --temperature 0.7 \
  --learning_rate 5e-7 \
  --klreward_weight 0.9 \
  --seman_weight 15 \
  --ppl_weight 0.1 \
  --train_data_path path/to/RLCracker/datasets/TRAIN_DATA/WM_GEN_Results_Filtered_short/Meta-Llama-3.1-8B-Instruct/EWD/train_100.json \
  --test_data_path path/to/RLCracker/datasets/TRAIN_DATA/WM_GEN_Results_Filtered_short/Meta-Llama-3.1-8B-Instruct/EWD/validation_20.json
```

In this command:

- `--model_name_or_path` specifies the rephraser model to train.
- `--train_data_path` points to the 100-example training set.
- `--test_data_path` points to the 20-example validation set.
- `--klreward_weight`, `--seman_weight`, and `--ppl_weight` control the distributional reward, semantic reward, and perplexity penalty.
- `--vllm_tensor_parallel_size` should usually match `--num_processes`.

### 2. Single Evaluation Run

Use `evaluation.py` to run a trained checkpoint as the rephraser and evaluate whether the rewritten outputs still trigger watermark detection.

Example evaluation command for short-form evaluation:

```bash
python path/to/RLCracker/evaluation.py \
  --ckpt_path path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/Qwen3_4B-KGW-Sem01KLRew01PPL1e6_dynaWeiSimAdv6/checkpoint-250 \
  --data_path WM_GEN_Results_Filtered_short/Meta-Llama-3.1-8B-Instruct/KGW \
  --run_num 1 \
  --use_cuda True \
  --is_short True \
  --max_new_tokens 600 \
  --min_new_tokens 100 \
  --withSysPro True \
  --WhetherThinking False
```

Example evaluation command for long-form evaluation:

```bash
python path/to/RLCracker/evaluation.py \
  --ckpt_path path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/Qwen3_4B-KGW-Sem01KLRew01PPL1e6_dynaWeiSimAdv6/checkpoint-250 \
  --data_path WM_GEN_Results_Filtered/Meta-Llama-3.1-8B-Instruct/KGW \
  --run_num 1 \
  --use_cuda True \
  --is_short True \
  --max_new_tokens 1600 \
  --min_new_tokens 100 \
  --withSysPro True \
  --WhetherThinking False
```

The evaluation script writes JSONL files containing the source question, watermarked text, rephrased output, PSP score, sentence similarity score, and watermark detection results when detection is enabled for the selected watermark setting.

### 3. Multi-GPU Batch Launcher

The repository also provides `multi_thread.py`, which launches multiple training or evaluation jobs across GPU groups.

Run:

```bash
python multi_thread.py
```

By default, the script calls:

```python
RunGRPO(node_num=4)
RunEval(node_num=1)
```

The `node_num` argument controls how many GPUs are assigned to each job.

For example:

- `RunGRPO(node_num=4)` launches each training job with 4 GPUs.
- `RunEval(node_num=1)` launches each evaluation job with 1 GPU.
- If the machine has 8 GPUs and `node_num=4`, the script creates two GPU groups: `0,1,2,3` and `4,5,6,7`.
- If the machine has 8 GPUs and `node_num=1`, the script creates eight GPU groups: `0`, `1`, `2`, ..., `7`.

Each GPU group runs one queued command at a time. The script sets:

```bash
CUDA_VISIBLE_DEVICES=<gpu_group>
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_DISABLE_COMPILE_CACHE=1
```

for each job.

### 4. Multi-GPU Training with `RunGRPO`

`RunGRPO(node_num)` creates a queue of training commands and assigns them to GPU groups.

The default training loop in `multi_thread.py` is:

```python
for algorithm in ['EWD']: # 'SWEET','PF'
    for model_name in ['Meta-Llama-3.1-8B-Instruct']:
        ...
```

The corresponding training command template is:

```bash
accelerate launch \
  --main_process_port <port> \
  --config_file path/to/RLCracker/open_r1/accelerate_configs/zero2.yaml \
  --num_processes <node_num> \
  path/to/RLCracker/grpo_reph.py \
  --config path/to/RLCracker/config_grpo_demo_reph.yaml \
  --model_name_or_path Qwen/Qwen3-8B \
  --max_completion_length 600 \
  --min_completion_length 10 \
  --per_device_eval_batch_size <24/node_num> \
  --per_device_train_batch_size <24/node_num> \
  --gradient_accumulation_steps 2 \
  --vllm_tensor_parallel_size <node_num> \
  --vllm_gpu_memory_utilization 0.08 \
  --output_dir path/to/RLCracker/TRAINED_MODELS/Short/<generator_model>/Qwen3_8B-<watermark>-Sem09KLRew01PPL5e7_dynaWeiSimAdv15 \
  --temperature 0.7 \
  --learning_rate 5e-7 \
  --klreward_weight 0.9 \
  --seman_weight 15 \
  --ppl_weight 0.1 \
  --train_data_path path/to/RLCracker/datasets/TRAIN_DATA/WM_GEN_Results_Filtered_short/<generator_model>/<watermark>/train_100.json \
  --test_data_path path/to/RLCracker/datasets/TRAIN_DATA/WM_GEN_Results_Filtered_short/<generator_model>/<watermark>/validation_20.json
```

To train on more watermarking schemes, edit:

```python
for algorithm in ['EWD']:
```

to, for example:

```python
for algorithm in ['EWD', 'SWEET', 'PF']:
```

To train on another generator model, edit:

```python
for model_name in ['Meta-Llama-3.1-8B-Instruct']:
```

To change the attacker model, edit:

```bash
--model_name_or_path Qwen/Qwen3-8B
```

and update the `output_dir` name accordingly.

Each generated command is saved to:

```text
<output_dir>/command.txt
```

which makes it easier to reproduce or debug a run.

### 5. Multi-GPU Evaluation with `RunEval`

`RunEval(node_num)` creates a queue of evaluation commands and assigns them to GPU groups.

The default evaluation loop in `multi_thread.py` evaluates checkpoints:

```python
for ckpt_step in [25,50,75,100,125,150,175,200,225,250]:
```

for the setting:

```python
mod = 'Qwen3_4B'
algo = 'KGW'
model_name = 'Meta-Llama-3.1-8B-Instruct'
ckpt_name = 'Sem01KLRew01PPL1e6_dynaWeiSimAdv6'
```

For each checkpoint, the script runs three evaluation commands:

#### Short-form evaluation with system prompt

```bash
python path/to/RLCracker/evaluation.py \
  --ckpt_path path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/Qwen3_4B-KGW-Sem01KLRew01PPL1e6_dynaWeiSimAdv6/checkpoint-<step> \
  --data_path WM_GEN_Results_Filtered_short/Meta-Llama-3.1-8B-Instruct/KGW \
  --run_num 1 \
  --use_cuda True \
  --is_short True \
  --max_new_tokens 600 \
  --min_new_tokens 100 \
  --withSysPro True \
  --WhetherThinking False
```

#### Long-form evaluation without system prompt

```bash
python path/to/RLCracker/evaluation.py \
  --ckpt_path path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/Qwen3_4B-KGW-Sem01KLRew01PPL1e6_dynaWeiSimAdv6/checkpoint-<step> \
  --data_path WM_GEN_Results_Filtered/Meta-Llama-3.1-8B-Instruct/KGW \
  --run_num 1 \
  --use_cuda True \
  --is_short True \
  --max_new_tokens 1600 \
  --min_new_tokens 100 \
  --withSysPro False \
  --WhetherThinking False
```

#### Long-form evaluation with system prompt

```bash
python path/to/RLCracker/evaluation.py \
  --ckpt_path path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/Qwen3_4B-KGW-Sem01KLRew01PPL1e6_dynaWeiSimAdv6/checkpoint-<step> \
  --data_path WM_GEN_Results_Filtered/Meta-Llama-3.1-8B-Instruct/KGW \
  --run_num 1 \
  --use_cuda True \
  --is_short True \
  --max_new_tokens 1600 \
  --min_new_tokens 100 \
  --withSysPro True \
  --WhetherThinking False
```

To evaluate a different watermark, change:

```python
for algo in ['KGW']:
```

To evaluate a different model or checkpoint naming pattern, change:

```python
for mod in ['Qwen3_4B']:
for ckpt_name in ['Sem01KLRew01PPL1e6_dynaWeiSimAdv6']:
```

To evaluate only selected checkpoints, edit:

```python
for ckpt_step in [25,50,75,100,125,150,175,200,225,250]:
```

### 6. Recommended Workflow

A typical workflow is:

1. Generate watermarked data with MarkLLM.
2. Prepare `train_100.json` and `validation_20.json`.
3. Run single training for debugging.
4. Run `RunGRPO(node_num=4)` for batch training.
5. Run one evaluation command manually to verify paths.
6. Run `RunEval(node_num=1)` for checkpoint evaluation.
7. Aggregate results with `data_utils.py`.

## Metrics

The main evaluation metric is evasion success rate, which measures the fraction of rewritten outputs that both evade watermark detection and preserve semantic similarity.

The repository also supports additional metrics used in the paper, including:

- PSP semantic similarity;
- sentence-level similarity;
- watermark detection result before and after rewriting;
- removal rate without semantic filtering;
- quality-related statistics used for result aggregation.

Metric utilities are implemented under `metrics/`, and aggregation helpers are provided in `data_utils.py`.

## End-to-End Pipeline

1. Generate watermarked data with MarkLLM.
2. Prepare `TRAIN_DATA` and `TEST_DATA` JSON files with `question` and `watermarked` fields.
3. Train the rephraser with `grpo_reph.py`.
4. Evaluate checkpoints with `evaluation.py`.
5. Aggregate success rates or Pass@K results with `data_utils.py`.

## Responsible Use

RLCracker is released for research on watermark robustness, security evaluation, and pre-deployment stress testing. The code is intended to help watermark designers identify vulnerabilities in existing schemes and develop stronger defenses.

Because adaptive watermark-removal methods may be misused to evade provenance mechanisms, users should apply this repository only in authorized research or evaluation settings and follow the policies of the models, datasets, and watermarking systems they use.

## Notes

- `open_r1/` is adapted from Hugging Face Open-R1.
- `MarkLLM/` stores the MarkLLM configs and data used by this project.
- PSP metric files are expected under `metrics/p_sp_utils/psp/`; place `model.para.lc.100.pt` there if it is not included.
- The batch launcher in `multi_thread.py` is an example template. Adjust model names, watermark names, checkpoint steps, and GPU counts for your run.
- When using `multi_thread.py`, verify that `node_num`, available GPU count, `vllm_tensor_parallel_size`, and batch sizes are compatible.

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{huang2025rlcrackerexposingvulnerabilityllm,
      title={RLCracker: Exposing the Vulnerability of LLM Watermarks with Adaptive RL Attacks}, 
      author={Hanbo Huang and Yiran Zhang and Hao Zheng and Xuan Gong and Yihan Li and Lin Liu and Shiyu Liang},
      year={2025},
      eprint={2509.20924},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2509.20924}, 
}
```