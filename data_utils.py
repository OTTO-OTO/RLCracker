import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
import threading
import numpy as np
import concurrent.futures
from filelock import FileLock
from metrics.p_sp import cal_p_sp, GRPO_P_SP, GRPO_Sim
from metrics.p_sp_utils.models import load_model

import numpy # for gradio hot reload
import gradio as gr
import random
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import torch
import gzip
import os
import pandas as pd
from glob import glob
import json
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)
from metrics.p_sp import cal_p_sp
try:
    from wm_eval_utils.eval_utils import get_deepseek_retrieve, get_kimi_retrieve
except ImportError:
    def get_deepseek_retrieve(*args, **kwargs):
        raise ImportError("wm_eval_utils is not available. Provide your retrieval client before using retrieval helpers.")

    def get_kimi_retrieve(*args, **kwargs):
        raise ImportError("wm_eval_utils is not available. Provide your retrieval client before using retrieval helpers.")
from watermarks.extended_watermark_processor_vllm import WatermarkDetector
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config, GRPOTrainer

# Strictly ensure that all key details, meaning and core message remain the same and the sentence length is unchanged. Respond only with the rewritten text, without any additional details or modifiers.
end_prompt = "Make sure all key details and the original meaning and core messages remain identical, and that the sentence length does not change. Respond only with the rewritten text, without any extra details or modifications."

def data_length_count(model_path, file_path='path/to/RLCracker/datasets/Qwen2.5-3B-rewrite-expanded-vllm_detectorFilted.jsonl'):
    wp_data = []
    with open(file_path, 'r') as f:
        wp_data = [json.loads(line) for line in f]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    para_length = []
    for data in wp_data:
        outputs = data['outputs']
        for key in outputs.keys():
            para_length.append(len(tokenizer.encode(outputs[key])))
    print(len(para_length))
    print(max(para_length))
    print(min(para_length))
    print(sum(para_length)/len(para_length))


def retrieve_data_deepseek(prompt,is_expand):
    # input = (f'Please retrieve a human-written text from the internet that has very high semantic similarity and a similar length to the text below. Only return the retrieved text and do not include any additional information or modifiers.\n\n'
    #          f'Original Text:\n{prompt}') 
    input = (
        "As an expert copy-editor, please rewrite the target text above in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Do not add any new facts, modifiers, stylistic flourishes, or interpretations. Return only the expanded version of the text—no explanations or extra output.\n\n"
        f"Target Text:\n{prompt}\n\n"
        "Your Response: "
    )
    # if is_expand:
        # input = (
        #     "Expand the following text so that the total output is approximately 1200 tokens.\n"
        #     "For each sentence, rewrite it in a way that makes it at least 2–3 times longer in word count, while keeping the original meaning exactly the same.\n"
        #     "Expand each sentence individually and maintain the original structure, sequence, and paragraph breaks. Do not add any new facts, modifiers, stylistic flourishes, or interpretations. Return only the expanded version of the text—no explanations or extra output.\n\n"
        #     f"Target Text:\n{prompt}\n\n"
        #     "Your Response: "
        # )
        # input = (
        #     "Paraphrase the following text while preserving the original structure, meaning, and length. Do not add any new facts, modifiers, stylistic flourishes, or interpretations. Return only the expanded version of the text—no explanations or extra output.\n\n"
        #     f"Target Text:\n{prompt}\n\n"
        #     "Your Response: "
        # )
    # else:
        # input = (
        #     "Summarize the following text to approximately 500 tokens.\n"
        #     "Please preserve the original structure and sequence of content.\n"
        #     "Do not add commentary, introductions, or stylistic changes—return only the concise version of the original text.\n\n"
        #     "Target Text:\n"
        #     f"{prompt}\n\n"
        #     "Your Response: "
        # )
        # input = (
        #     "Rewrite the following text in a condensed form, make it a bit concise, aiming for around 600 tokens.\n"
        #     "Retain the original structure and sequence of ideas as much as possible.\n"
        #     "Avoid adding commentary, introductions, or stylistic embellishments—focus only on producing a shortened version of the original.\n"
        #     "Ensure the result is complete, informative, and logically ordered.\n\n"
        #     "Target Text:\n"
        #     f"{prompt}\n\n"
        #     "Your Response: "
        # )
    try:
        cur_try = 0
        while cur_try < 6:
            # text = get_chat_kimi_reflect(prompt=prompt, temperature=0.3, max_tokens = 1800)
            text = get_deepseek_retrieve(prompt=input, temperature=0.3, max_tokens = 1800)
            # text = get_kimi_retrieve(prompt=input, temperature=0.3, max_tokens = 1800)
            # dumb way to do this
            if len(text.strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        # print(prompt)
        print(e)
        import sys
        sys.exit(1)


def retrieve_thread(data, save_path, semaphore, psp_model, psp_args, is_expand=True):
    semaphore.acquire()
    try:
        if is_expand:
            prompt= data['watermarked']
            for i in range(1):
                expanded = retrieve_data_deepseek(prompt, is_expand)
                psp_score = GRPO_P_SP(psp_model, psp_args, [expanded], [prompt])
                data['expanded'] = {'expanded': expanded, 'p_sp_score': f"{psp_score[0]}"}
                
                lock = FileLock(save_path + ".lock")
                with lock:
                    with open(save_path, 'a') as wf:
                        wf.write(json.dumps(data) + '\n')
                        wf.flush()
        else:
            prompt = data['expanded']['expanded']
            summarized = retrieve_data_deepseek(prompt, is_expand)
            psp_score = GRPO_P_SP(psp_model, psp_args, [summarized], [prompt])
            data['summarized'] = {'summarized': summarized, 'p_sp_score': f"{psp_score[0]}"}
            
            lock = FileLock(save_path + ".lock")
            with lock:
                with open(save_path, 'a') as wf:
                    wf.write(json.dumps(data) + '\n')
                    wf.flush()
    except Exception as e:
        print(f'raising: {e}')
    finally:
        semaphore.release()


def retrieve_expand_watermarkText(save_path = 'path/to/RLCracker/datasets/BT_dataset/deepseekExpanded.jsonl', max_threads=10, is_expand=True):
    if is_expand:
        with open('path/to/RLCracker/datasets/TEST_DATA/WM_GEN_Results_Filtered_short/Meta-Llama-3.1-8B-Instruct/Watermark_Test_PF.json', 'r') as f:
            # test_set = [json.loads(line) for line in f]
            # test_set = json.load(f)
            wp_data = json.load(f)
        # wp_data = wp_data[:50]

    if not is_expand:
        with open('path/to/RLCracker/datasets/BT_dataset/deepseekExpand.jsonl', 'r') as f:
            wp_data = [json.loads(line) for line in f if line.strip()]

    # Limit concurrent retrieval requests.
    semaphore = threading.Semaphore(max_threads)
    # for sample in tqdm(samples):

    psp_args = {
                    'gpu': 1, #1 if torch.cuda.is_available() else 0,
                    'load_file': 'metrics/p_sp_utils/psp/model.para.lc.100.pt',
                    'sp_model': 'metrics/p_sp_utils/psp/paranmt.model',
                    'gpu_id':0
                }
    psp_model, _ = load_model(None, psp_args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        list(tqdm(executor.map(lambda s: retrieve_thread(s, save_path, semaphore, psp_model, psp_args, is_expand), wp_data), total=len(wp_data)))

    print('DONE')


def retrieve_deepseek_watermarkText_pass1(save_path, watermark_path, max_threads=10, is_expand=True):

    
    with open(f"path/to/RLCracker/datasets/TEST_DATA/{watermark_path.split('/')[0]}/{watermark_path.split('/')[1]}/Watermark_Test_{watermark_path.split('/')[-1]}.json", 'r') as f:
        wp_data = json.load(f)

    # Limit concurrent retrieval requests.
    semaphore = threading.Semaphore(max_threads)
    # for sample in tqdm(samples):

    psp_args = {
                    'gpu': 1, #1 if torch.cuda.is_available() else 0,
                    'load_file': 'metrics/p_sp_utils/psp/model.para.lc.100.pt',
                    'sp_model': 'metrics/p_sp_utils/psp/paranmt.model',
                    'gpu_id':0
                }
    psp_model, _ = load_model(None, psp_args)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        list(tqdm(executor.map(lambda s: retrieve_thread(s, save_path, semaphore, psp_model, psp_args, is_expand), wp_data), total=len(wp_data)))

    print('DONE')



def re_unify_train_data():
    import shutil
    for algo in ['KGW_self']: # ,'Unigram','UPV','SWEET','PF','XSIR','SIR','SynthID','EWD'
        for model_name in ['Meta-Llama-3.1-8B-Instruct','Qwen2.5-1.5B-Instruct']:
            for folder in ['WM_GEN_Results_Filtered_short','WM_GEN_Results_Filtered']:

                tail ='Short' if 'short' in folder else 'Long'
                root_path = f"path/to/RLCracker/evaluation_results/DetectedResults/{folder}/{model_name}/Qwen2.5-3B-Instruct-{tail}/{algo}/Qwen2.5-3B-Instruct"
                target_path = f"path/to/RLCracker/datasets/TRAIN_DATA/{folder}/{model_name}/{algo}"
                
                save_data = []
                rest_data = []
                for dirpath, folder, file_names in os.walk(root_path):
                    for file_name in file_names:
                        if file_name.endswith('.jsonl') and ('train' in file_name or 'test' in file_name or 'wp-prompts' in file_name):
                            detected_file_path = os.path.join(dirpath, file_name)
                            wp_data = []
                            with open(detected_file_path, 'r') as f:
                                wp_data = [json.loads(line) for line in f if line.strip()]
                            for data in wp_data:
                                if 'p_sp_score' not in data:
                                    continue
                                psp_score = float(data['p_sp_score'])
                                is_watermarked = bool(data['is_watermarked'])
                                if psp_score > 0.7:
                                    if is_watermarked:
                                        save_data.append({"question": data['question'], "watermarked": data['watermarked_text']})
                                    else:
                                        rest_data.append({"question": data['question'], "watermarked": data['watermarked_text']})

                os.makedirs(target_path, exist_ok=True)
                print(len(save_data))
                if len(save_data) < 300:
                    print(target_path)
                    save_data += rest_data[:300-len(save_data)]
                with open(f"{target_path}/train_100.json", 'a') as f:
                    json.dump(save_data[:100], f, indent=4)
                with open(f"{target_path}/validation_20.json", 'a') as f:
                    json.dump(save_data[100:120], f, indent=4)


def construct_test_data():
    import shutil
    for algo in ['KGW_self']:#,'Unigram','UPV','SWEET','PF','XSIR','SIR','SynthID','EWD'
        for model_name in ['Meta-Llama-3.1-8B-Instruct','Qwen2.5-1.5B-Instruct']:
            for folder in ['WM_GEN_Results_Filtered_short','WM_GEN_Results_Filtered']:

                tail ='Short' if 'short' in folder else 'Long'
                root_path = f"path/to/RLCracker/evaluation_results/DetectedResults/{folder}/{model_name}/Qwen2.5-3B-Instruct-{tail}/{algo}/Qwen2.5-3B-Instruct"
                target_path = f"path/to/RLCracker/datasets/TEST_DATA/{folder}/{model_name}"
                
                save_data = []
                true_data = []

                with open(f"{target_path.replace('TEST_DATA','TRAIN_DATA')}/{algo}/train_100.json", 'r') as f:
                    train_data = json.load(f)
                questions_train = set([data['question'] for data in train_data])

                for dirpath, folder, file_names in os.walk(root_path):
                    for file_name in file_names:
                        detected_file_path = os.path.join(dirpath, file_name)
                        wp_data = []
                        with open(detected_file_path, 'r') as f:
                            wp_data = [json.loads(line) for line in f if line.strip()]
                        
                        if file_name.endswith('.jsonl') and ('test' in file_name or 'train' in file_name or 'wp-prompts' in file_name):
                            for data in wp_data:
                                if 'p_sp_score' not in data:
                                    continue
                                
                                save_data.append({"question": data['question'], "watermarked": data['watermarked_text']}) 

                        if file_name.endswith('.jsonl') and ('MMW_' in file_name):
                            for data in wp_data:
                                if 'p_sp_score' not in data:
                                    continue
                                true_data.append({"question": data['question'], "watermarked": data['watermarked_text']})

                        if file_name.endswith('.jsonl') and ('lfqa' in file_name):
                            random.seed(42)
                            random.shuffle(wp_data)
                            for data in wp_data[:100]:
                                true_data.append({"question": data['question'], "watermarked": data['watermarked_text']})
                                
                random.seed(42)
                random.shuffle(save_data)
                if len(true_data) < 400:
                    for data in save_data:
                        if data['question'] not in questions_train:
                            true_data.append(data)
                        if len(true_data) >= 400:
                            break

                os.makedirs(target_path, exist_ok=True)
                # print(len(save_data))
                # if len(save_data) < 500:
                #     # print(target_path)
                #     save_data += rest_data[:500-len(save_data)]
                with open(f"{target_path}/Watermark_Test_{algo}.json", 'a') as f:
                    json.dump(true_data, f, indent=4)
                # with open(f"{target_path}/Watermark_Test_{algo}.json", 'a') as f:
                #     json.dump(save_data[100:], f, indent=4)
                

def delete_global_step_dirs(root_dir):
    import shutil
    count = 0
    for dirpath, dirnames, _ in os.walk(root_dir):
        for dirname in dirnames:
            if dirname.startswith("global_step"):
                full_path = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(full_path)
                    print(f"Deleted folder: {full_path}")
                    count += 1
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")
    print(f"\n✅ Done. {count} global_step folders deleted.")


def cal_success_rate():

    for algo in ['KGW_self','KGW','Unigram','UPV','SWEET','PF','XSIR','SIR','SynthID','EWD']: #  
        for model_name in ['Meta-Llama-3.1-8B-Instruct', 'Qwen2.5-1.5B-Instruct']:
            for folder_name in ['WM_GEN_Results_Filtered_short','WM_GEN_Results_Filtered']:
                success_rate = []
                removed_count = []
                data_len = []
                psp_mean = []
                psp_off_watermarked = []
                tail = 'Short' if 'short' in folder_name else 'Long'
                for ckpt_step in [25,50,75,100,125,150,175,200,225,250]:
                    file_path = f'path/to/RLCracker/evaluation_results/DetectedResults/{folder_name}/{model_name}/Qwen3B-{algo}-{tail}/checkpoint-{ckpt_step}/Watermark_Test_{algo}-7.jsonl'
                    with open(file_path, 'r') as f:
                        data = [json.loads(line) for line in f if line.strip()]
                    succes = 0
                    removed = 0
                    psp = 0
                    total_psp = 0
                    for da in data:
                        total_psp += float(da['p_sp_score'])
                        if not bool(da['is_watermarked']):
                            removed += 1
                            psp += float(da['p_sp_score'])
                            if float(da['p_sp_score']) > 0.7:
                                succes += 1
                    
                    success_rate.append(succes/len(data))
                    removed_count.append(removed)
                    data_len.append(len(data))
                    psp_mean.append(total_psp/len(data))
                    psp_off_watermarked.append(psp/removed)
                with open(f'path/to/RLCracker/evaluation_results/DetectedResults/{folder_name}/{model_name}/Qwen3B-{algo}-{tail}/success_rate.txt', 'w') as f:
                    f.write(algo+'\n')
                    for i in range(len(success_rate)):
                        f.write(f"{success_rate[i]}, {removed_count[i]}, {data_len[i]}, {psp_off_watermarked[i]}, {psp_mean[i]}\n")

    
    # for model_name in ['Meta-Llama-3.1-8B-Instruct', 'Qwen2.5-1.5B-Instruct']:
    #     for folder_name in ['WM_GEN_Results_Filtered_short','WM_GEN_Results_Filtered']:
    #         tail = 'Short' if 'short' in folder_name else 'Long'
    #         for algo in ['KGW_self','KGW','Unigram','UPV','SWEET','PF','XSIR','SIR','SynthID','EWD']: 
    #             success_rate = []
    #             removed_count = []
    #             data_len = []
    #             psp_mean = []
    #             psp_off_watermarked = []
    #             file_path = f'path/to/RLCracker/evaluation_results/DetectedResults/{folder_name}/{model_name}/model_base-{tail}/Qwen2.5-3B-Instruct/Watermark_Test_{algo}-7.jsonl'
    #             with open(file_path, 'r') as f:
    #                 data = [json.loads(line) for line in f if line.strip()]
    #             succes = 0
    #             removed = 0
    #             psp = 0
    #             total_psp = 0
    #             for da in data:
    #                 total_psp += float(da['p_sp_score'])
    #                 if not bool(da['is_watermarked']):
    #                     removed += 1
    #                     psp += float(da['p_sp_score'])
    #                     if float(da['p_sp_score']) > 0.7:
    #                         succes += 1
    #             success_rate.append(succes/len(data))
    #             removed_count.append(removed)
    #             data_len.append(len(data))
    #             psp_mean.append(total_psp/len(data))
    #             psp_off_watermarked.append(psp/removed)
    #             with open(f'path/to/RLCracker/evaluation_results/DetectedResults/{folder_name}/{model_name}/model_base-{tail}/success_rate.txt', 'a') as f:
    #                 f.write(algo+'\n')
    #                 for i in range(len(success_rate)):
    #                     f.write(f"{success_rate[i]}, {removed_count[i]}, {data_len[i]}, {psp_off_watermarked[i]}, {psp_mean[i]}\n")


def cal_success_rate_path(root_path):
    algo = None
    for al in ['KGW_bias8','KGW_gamma075','KSEMSTAMP','SEMSTAMP','KGW_self','KGW','Unigram','UPV','SWEET','PF','XSIR','SIR','SynthID','EWD']: #  
        if f'-{al}' in root_path:
            algo = al
            break
    if algo is None:
        return
    success_rate = []
    removed_count = []
    data_len = []
    psp_mean = []
    psp_off_watermarked = []
    success_psp_mean=[]
    # for ckpt_step in [72,144,216,288,360,432,504,576,648,720]:
    # for ckpt_step in [38,76,114,152,190,228,266,304,342,380]:
    for ckpt_step in [25,50,75,100,125,150,175,200,225,250]:
    # for ckpt_step in [50,100,150,200,250,300,350,400,450,500]:
        file_path = f'{root_path}/checkpoint-{ckpt_step}/Watermark_Test_{algo}-7.jsonl'
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            data = [json.loads(line) for line in f if line.strip()]
        data = data[:400]
        succes = 0
        removed = 0
        psp = 0
        total_psp = 0
        suceese_psp = 0
        for da in data:
            total_psp += float(da['p_sp_score'])
            if not bool(da['is_watermarked']):
                removed += 1
                psp += float(da['p_sp_score'])
                if float(da['p_sp_score']) > 0.7:
                    succes += 1
                    suceese_psp += float(da['p_sp_score'])
        
        success_rate.append(succes/len(data))
        removed_count.append(removed)
        data_len.append(len(data))
        psp_mean.append(total_psp/len(data))
        if succes == 0:
            success_psp_mean.append(0)
        else:
            success_psp_mean.append(suceese_psp/succes)
        if removed == 0:
            psp_off_watermarked.append(0)
        else:
            psp_off_watermarked.append(psp/removed)
    with open(f'{root_path}/success_rate.txt', 'w') as f:
        f.write(algo+'\n')
        f.write('success_rate\t removed_count\t removed_rate\t data_len\t psp_success\t psp_off_watermarked\t Total psp_mean\n')
        for i in range(len(success_rate)):
            f.write(f"{success_rate[i]},\t{removed_count[i]},\t{removed_count[i]/(data_len[i])},\t{data_len[i]},\t{success_psp_mean[i]},\t{psp_off_watermarked[i]},\t{psp_mean[i]}\n")


def cal_success_pass1(root_path):

    
    success_rate = []
    removed_count = []
    data_len = []
    psp_mean = []
    psp_off_watermarked = []
    success_psp_mean=[]
    algo_list = []
    for file_name in os.listdir(root_path):
        if '.jsonl' not in file_name:
            continue
        for al in ['KGW_bias8','KGW_gamma075','KSEMSTAMP','SEMSTAMP','KGW_self','KGW','Unigram','UPV','SWEET','PF','XSIR','SIR','SynthID','EWD']: #  
            if f'{al}' in file_name:
                algo = al
                if algo=='SIR' and 'XSIR' in file_name:
                    algo = 'XSIR'
                break
        file_path = os.path.join(root_path, file_name)
        if '.jsonl' not in file_path:
            continue
        try:
            with open(file_path, 'r') as f:
                data = [json.loads(line) for line in f if line.strip()]
                if len(data)==0:
                    continue
        except :
            print(f"Error reading file: {file_path}")
        
        algo_list.append(algo)
        succes = 0
        removed = 0
        psp = 0
        total_psp = 0
        suceese_psp = 0
        for da in data:
            total_psp += float(da['p_sp_score'])
            if not bool(da['is_watermarked']):
                removed += 1
                psp += float(da['p_sp_score'])
                if float(da['p_sp_score']) > 0.7:
                    succes += 1
                    suceese_psp += float(da['p_sp_score'])
        
        success_rate.append(succes/len(data))
        removed_count.append(removed)
        data_len.append(len(data))
        psp_mean.append(total_psp/len(data))
        if succes == 0:
            success_psp_mean.append(0)
        else:
            success_psp_mean.append(suceese_psp/succes)
        if removed == 0:
            psp_off_watermarked.append(0)
        else:
            psp_off_watermarked.append(psp/removed)

    with open(f'{root_path}/success_rate.txt', 'a') as f:
        f.write('Pass@1\n')
        f.write('Watermark\t\t success_rate\t removed_count\t removed_rate\t data_len\t psp_success\t psp_off_watermarked\t Total psp_mean\n')
        for i in range(len(success_rate)):
            f.write(f"{algo_list[i]},\t{success_rate[i]},\t{removed_count[i]},\t{removed_count[i]/(data_len[i])},\t{data_len[i]},\t{success_psp_mean[i]},\t{psp_off_watermarked[i]},\t{psp_mean[i]}\n")


def tensor_to_json_by_step(event_file):
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    step_data = defaultdict(dict)

    for tag in ea.Tags().get('scalars', []):
        for e in ea.Scalars(tag):
            step_data[e.step]["step"] = e.step
            step_data[e.step][tag] = round(e.value, 6)

    # Sort by training step.
    output = [step_data[k] for k in sorted(step_data)]

    with open(f'path/to/RLCracker/run_logs_modelTrain/{event_file.split("/")[7]}.jsonl', "w") as f:
        for record in output:
            f.write(json.dumps(record) + "\n")

    print(f"Saved all scalars")

    
def merge_comparative_results_by_prompt() -> None:
    
    def collect_watermark_data(model_dir: str, tag: str, withSys=False) -> dict:
        """
        Collect all watermark result tables under a model directory.

        Returns a mapping from prompt ID to a DataFrame containing the
        corresponding watermark result columns.
        """
        prompt_map = {}
        for watermark in os.listdir(model_dir):
            path_to_csvs = os.path.join(model_dir, watermark, "effectiveResults_csv")
            if (withSys and '_sys' not in path_to_csvs) or (not withSys and '_sys' in path_to_csvs):
                continue
            if not os.path.exists(path_to_csvs):
                path_to_csvs = os.path.join(model_dir, watermark, "Qwen2.5-3B-Instruct","effectiveResults_csv")

            watermark = watermark.replace('_sys', '')
            for csv_path in glob(os.path.join(path_to_csvs, "*.csv")):
                filename = os.path.basename(csv_path)
                if not filename.endswith(".csv"):
                    continue
                if "promp" not in filename:
                    continue
                prompt = filename.replace(".csv", "").split("-")[-1]  # promp1 ~ promp8
                df = pd.read_csv(csv_path)
                if df.columns[0] != 'Pass@K':
                    df.columns = ['Pass@K', 'SuccessNum', 'TotalNum', 'TotalRemoved', 'PSPMean', 'SuccessRate']
                sub_df = df[['Pass@K', 'SuccessRate', 'PSPMean']].copy()
                sub_df = sub_df.rename(columns={
                    'SuccessRate': f'{watermark}_{tag}_SuccessRate',
                    'PSPMean': f'{watermark}_{tag}_PSPMean'
                })

                if prompt not in prompt_map:
                    prompt_map[prompt] = sub_df
                else:
                    prompt_map[prompt] = pd.merge(prompt_map[prompt], sub_df, on='Pass@K', how='outer')
        return prompt_map
    
    for result_path in ['Short', 'Long']:
        base_dir=f'path/to/RLCracker/Pass@K/DetectedResults/{result_path}/Meta-Llama-3.1-8B-Instruct'
        rl_dir=f'path/to/RLCracker/Pass@K/DetectedResults/{result_path}/Meta-Llama-3.1-8B-Instruct_RLModel'
        for withSys in [False,True]:
            os.makedirs(f'path/to/RLCracker/Pass@K/MergedResults/{result_path}/',exist_ok=True)
            output_path = os.path.join(f'path/to/RLCracker/Pass@K/MergedResults/{result_path}/', f"merged_comparative_results_{withSys}.xlsx")
            # Collect base model and RL model results.
            base_results = collect_watermark_data(base_dir, tag="Base", withSys=withSys)
            your_results = collect_watermark_data(rl_dir, tag="RLModel", withSys=withSys)

            # Merge results from both models.
            all_prompts = sorted(set(base_results) | set(your_results))

            with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
                for prompt in all_prompts:
                    df_base = base_results.get(prompt, pd.DataFrame())
                    df_your = your_results.get(prompt, pd.DataFrame())

                    if df_base.empty and df_your.empty:
                        continue
                    elif df_base.empty:
                        merged = df_your
                    elif df_your.empty:
                        merged = df_base
                    else:
                        merged = pd.merge(df_base, df_your, on="Pass@K", how="outer")

                    merged = merged.sort_values("Pass@K")

                    # Collect all watermark names.
                    watermarks = set(col.split('_')[0] for col in merged.columns if '_' in col)
                    watermarks.add('KGW_self')
                    watermarks = sorted(watermarks)
                    # watermarks = sorted(set(
                    #     col for col in merged.columns #if '_sys' in col
                    # ))

                    # Define the output column order.
                    new_columns = ['Pass@K']
                    metrics = ['SuccessRate', 'PSPMean']

                    for watermark in watermarks:
                        for metric in metrics:
                            base_col = f"{watermark}_Base_{metric}"
                            rl_col = f"{watermark}_RLModel_{metric}"
                            if base_col in merged.columns:
                                new_columns.append(base_col)
                            if rl_col in merged.columns:
                                new_columns.append(rl_col)

                    merged = merged[new_columns]
                    merged.set_index('Pass@K', inplace=True)
                    merged = merged.transpose().reset_index()
                    merged[['Watermark', 'Model', 'Metric']] = merged['index'].str.extract(r'(.+?)_(Base|RLModel)_(.+)')
                    merged.drop(columns=['index'], inplace=True)
                    cols = ['Watermark', 'Model', 'Metric'] + [col for col in merged.columns if col not in ['Watermark', 'Model', 'Metric']]
                    merged = merged[cols]
                    merged.sort_values(by=['Watermark', 'Model'], inplace=True)
                    merged.to_excel(writer, sheet_name=prompt, index=False)

            print(f"All prompt-level comparative results were saved to: {output_path}")

def discover_experiments_under_base(base_dir: str):
    """
    Discover experiment root paths ending at result-mode directories such as
    `.../Norm` without depending on version, model directory, or experiment
    naming conventions.

    This only depends on the stable result structure:
      .../Norm/checkpoint-25/Watermark_Test_*-7.jsonl
    """
    pattern = os.path.join(
        base_dir,
        "WM_GEN_Results_Filtered_short",
        "*",    # model_dir Meta-Llama-3.1-8B-Instruct
        "*",    # exp_name
        "*",
        # "*",
        "checkpoint-*",
        "Watermark_Test_*-7.jsonl"
    )

    anchors = glob(pattern)

    experiments = []
    for jsonl_path in anchors:
        ckpt_dir = os.path.dirname(jsonl_path)
        norm_dir = os.path.dirname(ckpt_dir)

        experiments.append({"root_path": norm_dir, 'ckpt_dir': ckpt_dir})

    # Deduplicate so each result-mode directory is processed once.
    uniq = {}
    for e in experiments:
        uniq[e["root_path"]] = e
        # uniq[e["ckpt_dir"]] = e
    return list(uniq.values())

if __name__ == "__main__":
    # cal_success_rate() 

    BASE_DIR = "path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test"
    # BASE_DIR = "path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_pass1"
    
    exps = discover_experiments_under_base(BASE_DIR)
    print(f"[Found {len(exps)} experiments]")

    for exp in exps:
        root_path = exp["root_path"]
        # root_path = exp["ckpt_dir"]
        print(f"[Run] {root_path}")
        cal_success_rate_path(root_path)
        # cal_success_pass1(root_path)

    
    # delete_global_step_dirs('path/to/RLCracker/TRAINED_MODELS_rebuttal')



    # delete_global_step_dirs('path/to/RLCracker/TRAINED_MODELS_rebuttal')
    # tensor_to_json_by_step('path/to/RLCracker/TRAINED_MODELS/Short/Meta-Llama-3.1-8B-Instruct/Qwen3_4B-Unigram-Sem03KLRew01PPL5e7_dynaWeiSimAdv10/runs/Aug04_16-34-41_7638e02f0a6124c7624c322a8ef1134e-taskrole1-0/events.out.tfevents.1754296530.7638e02f0a6124c7624c322a8ef1134e-taskrole1-0.1930546.0')

    # model_name 'Sem05KlReward02Ppl1e6'.  Sem09KLRew01PPL2e7_dynaWeiSimAdv12
    # EWD','PF','UPV','SWEET','SIR','XSIR
    # for algo in['EWD','PF','SWEET']:
    #     for model_name in ['Meta-Llama-3.1-8B-Instruct']: # 'Meta-Llama-3.1-8B-Instruct','Qwen2.5-1.5B-Instruct' 'Qwen3_4B','Qwen3_1_7B','Qwen3_06B'
    #         for ckpt in ['Sem09KLRew01PPL3e7_dynaWeiSimAdv12_Qwen8B']:
    #             for mod in['Qwen3_06B']:
    #                 cal_success_rate_path(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test/WM_GEN_Results_Filtered_short/{model_name}/{mod}-{algo}-{ckpt}-Short/Norm_chat")
    #                 cal_success_rate_path(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test/WM_GEN_Results_Filtered/{model_name}/{mod}-{algo}-{ckpt}-Short/Norm_chat")


    # for model in ['Sem20KLRew01PPL1e6_dynaWeiSimAdv0_99MatrixTest']:
    #     cal_success_rate_path(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test/WM_GEN_Results_Filtered_short/Qwen3-1.7B/Qwen3_1_7B-EWD-{model}-Short/Norm_chat")
    #     cal_success_rate_path(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test/WM_GEN_Results_Filtered_short/Qwen3-0.6B/Qwen3_06B-EWD-{model}-Short/Norm_chat")
    #     cal_success_rate_path(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test/WM_GEN_Results_Filtered_short/Qwen3-4B/Qwen3_4B-EWD-{model}-Short/Norm_chat")

    # cal_success_rate_path(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test/WM_GEN_Results_Filtered_short/Qwen3-1.7B/Qwen3_1_7B-EWD-Sem2KLRew01PPL2e7_dynaWeiSimAdv0_99Matrix-Short/Norm_chat")
    # cal_success_rate_path(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test/WM_GEN_Results_Filtered_short/Qwen3-0.6B/Qwen3_06B-EWD-Sem2KLRew01PPL2e7_dynaWeiSimAdv0_99Matrix-Short/Norm_chat")
    # # cal_success_rate_path(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_Test/WM_GEN_Results_Filtered_short/Qwen3-4B/Qwen3_4B-EWD-Sem09KLRew01PPL2e7_dynaWeiSimAdv0_99Matrix-Short/Norm_chat")



    # for algo in['EWD']:
    #     for model_name in ['Qwen3-1.7B','Qwen3-0.6B','Qwen3-4B']: # 'Meta-Llama-3.1-8B-Instruct','Qwen2.5-1.5B-Instruct' 'Qwen3_4B','Qwen3_1_7B','Qwen3_06B'
    #         for mod in['Qwen3-1.7B','Qwen3-0.6B','Qwen3-4B']:
                
    #             cal_success_pass1(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_pass1/WM_GEN_Results_Filtered_short/{model_name}/{mod}-Short/Norm/Ins7")
    #             cal_success_pass1(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_pass1/WM_GEN_Results_Filtered_short/{model_name}/{mod}-Short/Norm_chat/Ins7")
    #             cal_success_pass1(f"path/to/RLCracker/evaluation_results_rebuttal/DetectedResults_pass1/WM_GEN_Results_Filtered_short/{model_name}/{mod}-Short/Think/Ins7")
    

    
