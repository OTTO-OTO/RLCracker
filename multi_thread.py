import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
import torch
import os
from threading import Semaphore
import threading
import queue


def RunGRPO(node_num):

    pool_size = int(torch.cuda.device_count()/node_num)
    print(pool_size)
    commands = queue.Queue()

    ports = [29507,25841,29536,24856]
    port_cycle = cycle(range(len(ports))) 

    for algorithm in ['EWD']: # 'SWEET','PF'
        for model_name in ['Meta-Llama-3.1-8B-Instruct']:
            port_num = next(port_cycle)
            output_dir = f'path/to/RLCracker/TRAINED_MODELS/Short/{model_name}/Qwen3_8B-{algorithm}-Sem09KLRew01PPL5e7_dynaWeiSimAdv15'
            command = f'accelerate launch --main_process_port {ports[port_num]} \
                        --config_file path/to/RLCracker/open_r1/accelerate_configs/zero2.yaml\
                        --num_processes {node_num} \
                        path/to/RLCracker/grpo_reph.py \
                        --config path/to/RLCracker/config_grpo_demo_reph.yaml \
                        --model_name_or_path Qwen/Qwen3-8B \
                        --max_completion_length 600 \
                        --min_completion_length 10 \
                        --per_device_eval_batch_size {int(24/node_num)} \
                        --per_device_train_batch_size {int(24/node_num)} \
                        --gradient_accumulation_steps 2 \
                        --vllm_tensor_parallel_size {node_num} \
                        --vllm_gpu_memory_utilization 0.08\
                        --output_dir {output_dir} \
                        --temperature 0.7\
                        --learning_rate 5e-7 \
                        --klreward_weight 0.9 \
                        --seman_weight 15 \
                        --ppl_weight 0.1\
                        --train_data_path path/to/RLCracker/datasets/TRAIN_DATA/WM_GEN_Results_Filtered_short/{model_name}/{algorithm}/train_100.json \
                        --test_data_path path/to/RLCracker/datasets/TRAIN_DATA/WM_GEN_Results_Filtered_short/{model_name}/{algorithm}/validation_20.json'
            commands.put(command)
            os.makedirs(output_dir, exist_ok=True)
            with open(f'{output_dir}/command.txt', 'a') as f:
                f.write(command)


    print("Current queue items:", list(commands.queue))
    gpu_pairs=[]
    for i in range(0, torch.cuda.device_count(), node_num):
        temp = []
        for j in range(node_num):
            if i+j >= torch.cuda.device_count():
                break
            temp.append(str(i+j))
        gpu_pairs.append(','.join(temp))
    print(gpu_pairs)

    # Create one semaphore per GPU group.
    gpu_semaphores = [Semaphore(1) for _ in range(len(gpu_pairs))]
    
    threads = []

    for i, gpu in enumerate(gpu_pairs):
        t = threading.Thread(target=gpu_worker, args=(gpu, gpu_semaphores[i], commands))
        t.start()
        threads.append(t)

    # Wait until all queued jobs finish.
    commands.join()
    for t in threads:
        t.join()
    print("All commands have been executed.")


def RunEval(node_num):

    pool_size = int(torch.cuda.device_count()/node_num)
    print(pool_size)

    commands = queue.Queue()

    for i in [1]:

        for mod in ['Qwen3_4B']:
            for algo in ['KGW']:
                for model_name in ['Meta-Llama-3.1-8B-Instruct']:
                    for ckpt_name in ['Sem01KLRew01PPL1e6_dynaWeiSimAdv6']:
                        for ckpt_step in [25,50,75,100,125,150,175,200,225,250]:
                            command = f'python path/to/RLCracker/evaluation.py --ckpt_path path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/{mod}-{algo}-{ckpt_name}/checkpoint-{ckpt_step} --data_path WM_GEN_Results_Filtered_short/{model_name}/{algo} --run_num {i} --use_cuda True --is_short True --max_new_tokens 600 --min_new_tokens 100 --withSysPro True --WhetherThinking False'
                            commands.put(command)
                            command = f'python path/to/RLCracker/evaluation.py --ckpt_path path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/{mod}-{algo}-{ckpt_name}/checkpoint-{ckpt_step} --data_path WM_GEN_Results_Filtered/{model_name}/{algo} --run_num {i} --use_cuda True --is_short True --max_new_tokens 1600 --min_new_tokens 100 --withSysPro False --WhetherThinking False'
                            commands.put(command)
                            command = f'python path/to/RLCracker/evaluation.py --ckpt_path path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/{mod}-{algo}-{ckpt_name}/checkpoint-{ckpt_step} --data_path WM_GEN_Results_Filtered/{model_name}/{algo} --run_num {i} --use_cuda True --is_short True --max_new_tokens 1600 --min_new_tokens 100 --withSysPro True --WhetherThinking False'
                            commands.put(command)

    gpu_pairs=[]
    for i in range(0, torch.cuda.device_count(), node_num):
        temp = []
        for j in range(node_num):
            if i+j >= torch.cuda.device_count():
                break
            temp.append(str(i+j))
        gpu_pairs.append(','.join(temp))
    print(gpu_pairs)

    # Create one semaphore per GPU group.
    gpu_semaphores = [Semaphore(1) for _ in range(len(gpu_pairs))]
    threads = []

    for i, gpu in enumerate(gpu_pairs):
        t = threading.Thread(target=gpu_worker, args=(gpu, gpu_semaphores[i], commands))
        t.start()
        threads.append(t)

    # Wait until all queued jobs finish.
    commands.join()
    for t in threads:
        t.join()



def gpu_worker(gpu, semaphore, task_queue):
    while True:
        try:
            command = task_queue.get(timeout=3)  # wait for task; exit if idle for 3 sec
        except queue.Empty:
            break
        run_command(command, gpu, semaphore)
        task_queue.task_done()

def run_command(command, gpu, semaphore):

    with semaphore:
        # Execute the command
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        env['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
        env['VLLM_DISABLE_COMPILE_CACHE']='1'
        try:
            subprocess.run(command, shell=True, executable="/bin/bash", check=True, env=env)
            print(f"[{gpu}] ✅ success: {command}")
        except subprocess.CalledProcessError as e:
            print(f"[{gpu}] ❌ failed: {command}\n  Exit Code: {e.returncode}")
        except Exception as e:
            print(f"[{gpu}] ❌ unexpected error: {e}")
        print(command)


if __name__ == '__main__':
    RunGRPO(node_num=4)  # Adjust node_num based on your GPU count and desired parallelism
    RunEval(node_num=1)
