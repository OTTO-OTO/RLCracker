import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import argparse
import re
from functools import partial
from vllm.config import CompilationConfig, CompilationLevel

import random
from tqdm import tqdm
import torch
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from metrics.p_sp import cal_p_sp, GRPO_P_SP, GRPO_Sim
from metrics.p_sp_utils.models import load_model
from sentence_transformers import SentenceTransformer
from trl.data_utils import maybe_apply_chat_template
from watermarks.extended_watermark_processor import WatermarkDetector


PROJECT_ROOT = "path/to/RLCracker"
PSP_MODEL_PATH = f"{PROJECT_ROOT}/metrics/p_sp_utils/psp/model.para.lc.100.pt"
PSP_SENTENCEPIECE_PATH = f"{PROJECT_ROOT}/metrics/p_sp_utils/psp/paranmt.model"
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_ROOT = f"{PROJECT_ROOT}/datasets"
RESULT_ROOT = f"{PROJECT_ROOT}/evaluation_results_rebuttal"

MODEL_REPOS = {
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Meta-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
    "Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "Qwen3-4B": "Qwen/Qwen3-4B",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"yes", "true", "t", "1", "y"}:
        return True
    if normalized in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def resolve_model_id(model_name):
    return MODEL_REPOS.get(model_name, model_name)


rephrase_Ins = {0:"\nRewrite the paragraph above using different wording, but keep the original meaning and length unchanged.",
                    1:"\n\n######Instruction: Paraphrase the above paragraphs.\n\nYour Response:",
                    2:"\n\n######Instruction: You are an expert copy-editor. Please rewrite the target text above in your own voice and paraphrase all sentences. \n Ensure that the final output contains the same information as the original text and has roughly the same length. \n Do not leave out any important details when rewriting in your own voice.\n\n######Your Response:",
                    3:"\n\n######Instruction: As an expert copy-editor, please rewrite the target text above in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases.\n\n######Your Response:",
                    4:"\n\n######Instruction: Rewrite the target text above using different words but keeping the same meaning and similar length.\n\n######Your Response:",
                    5:"\n\n######Instruction: Paraphrase the target text above without changing its meaning or length.\n\n######Your Response:",
                    6:"\n\n######Instruction: Restate the target text above using different wording. Keep the meaning and text length nearly the same.\n\n######Your Response:",
                    7:"\n\n######Instruction: Generate a version of the target text above that means the same and is about the same length.\n\n######Your Response:",
                    8:"\n\n######Instruction: Write a similar target text to the one above, keeping both the meaning and the number of words roughly the same.\n\n######Your Response:",}


def load_model_vllm(model_name_or_path):
    """Load and return the model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # return model, tokenizer, device   
    device_memo = torch.cuda.get_device_properties(0).total_memory/(1024 ** 2)//1000
    gpu_memory_utilization = 0.8 if device_memo <50 else 0.9
    if device_memo < 30:
        gpu_memory_utilization = 0.7
    model = LLM(model=model_name_or_path, tensor_parallel_size=torch.cuda.device_count(), dtype=torch.bfloat16, trust_remote_code=True, gpu_memory_utilization=gpu_memory_utilization,
                compilation_config=CompilationConfig(level=CompilationLevel.PIECEWISE,# By default, it goes up to max_num_seqscudagraph
                                                      cudagraph_capture_sizes=[1,2,4])
                ) #  , quantization="bitsandbytes", load_format="bitsandbytes"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, tokenizer, device


def cal_z_score(input_texts, args, device=None, tokenizer=None):
    eval_results={}
    # for ignore_repeated_ngrams in [True, False]:
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=args.gamma,
                                        seeding_scheme=args.seeding_scheme,
                                        device=device,
                                        tokenizer=tokenizer,
                                        z_threshold=args.detection_z_threshold,
                                        normalizers=args.normalizers,
                                        ignore_repeated_ngrams=args.ignore_repeated_bigrams,
                                        select_green_tokens=args.select_green_tokens)
    # eval_results[f'z_score_ignore_{ignore_repeated_ngrams}'] = {}
    # eval_results[f'p_value_ignore_{ignore_repeated_ngrams}'] = {}
    eval_results[f'z_score'] = {}
    eval_results[f'p_value'] = {}

    for i in range(len(input_texts)):
        if len(input_texts[i])>4:
            score_dict = watermark_detector.detect(input_texts[i],return_z_at_T=False)
            eval_results[f'z_score'][i] = score_dict['z_score']
            eval_results[f'p_value'][i] = score_dict['p_value']
        else:

            eval_results[f'z_score'][i] = 5
            eval_results[f'p_value'][i] = 0
                
    return eval_results


def paraphrase_actor_local(inputs, paraphraser, tokenizer, args):
    generate_rephrased = partial(
        paraphraser.generate,
        sampling_params = SamplingParams(temperature=0.7,#args.sampling_temp,
                                         presence_penalty=0.1,
                                         frequency_penalty=0.1,
                                        top_k=-1,
                                        max_tokens=args.max_new_tokens,
                                        min_tokens=args.min_new_tokens,
                                        stop_token_ids=[tokenizer.eos_token_id,
                                        # tokenizer.convert_tokens_to_ids("<|eot_id|>")
                                        ]),
        use_tqdm=False)
    torch.manual_seed(args.generation_seed)
    rephrased_output = generate_rephrased(inputs)
    decoded_output = []
    for out in rephrased_output:
        for output in out.outputs:
            decoded_output.append(output.text.strip())
    return decoded_output


def paraphrase_actor_local_thinking(inputs, paraphraser, tokenizer, args):
    generate_rephrased = partial(
        paraphraser.generate,
        sampling_params = SamplingParams(
                                        temperature=0.6,#args.sampling_temp,
                                        top_p=0.95,
                                        top_k=20,
                                        max_tokens=int(2048+int(args.max_new_tokens)),
                                        stop_token_ids=[tokenizer.eos_token_id,
                                        # tokenizer.convert_tokens_to_ids("<|eot_id|>")
                                        ]),
        use_tqdm=False)
    torch.manual_seed(args.generation_seed)
    rephrased_output = generate_rephrased(inputs)
    decoded_output = []
    for out in rephrased_output:
        for output in out.outputs:
            match = re.search(r"<think>(.*?)</think>\s*(.*)", output.text, re.S)
            if match:
                reasoning = match.group(1).strip()
                answer = match.group(2).strip()
            else:
                reasoning, answer = "", output.text.strip()
            decoded_output.append(answer.strip())
    return decoded_output


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1600,
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--max_new_tokens_prompt",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate for the prompt.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--is_decoder_only_model",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=True,
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7,
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="selfhash",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.25,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.0,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=True,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )

    parser.add_argument(
        "--run_num",
        type=str,
        default='0',
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--rephraser_path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=1100,
        help="Maximmum number of new tokens to generate.",
    )

    parser.add_argument(
        "--lora_path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--use_cuda",
        type=str2bool,
        default=True
    )
    parser.add_argument(
        "--is_short",
        type=str2bool,
        default=True
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default='Meta-Llama-3.1-8B-Instruct/KGW'
    )
    parser.add_argument(
        "--withSysPro",
        type=str2bool,
        default=False
    )
    parser.add_argument(
        "--WhetherThinking",
        type=str2bool,
        default=False
    )
    # args = parser.parse_args()
    args, _ = parser.parse_known_args()
    return args


def pure_rewrite_rephraseTrained_selfhash(args, rephraser_path="meta-llama/Meta-Llama-3-8B-Instruct", text_nums=[1,2,3,4,5,6,7,8], is_short=False):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print('Use_cuda', args.use_cuda)
    print('max_new_tokens:', args.max_new_tokens)
    print('is_short:', is_short)
    print('WhetherThinking', args.WhetherThinking)

    psp_args = {
                'gpu': 1 if args.use_cuda else 0,
                'load_file': PSP_MODEL_PATH,
                'sp_model': PSP_SENTENCEPIECE_PATH,
                'gpu_id':0
            }

    psp_model, _ = load_model(None, psp_args)
    sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME, device='cuda' if args.use_cuda else 'cpu')

    rephraser, rephrase_tokenizer, device = load_model_vllm(rephraser_path)
    
    # folder_name = 'WM_GEN_Results_Filtered_short' if 'Short' in rephraser_path or is_short else 'WM_GEN_Results_Filtered'
    folder_name = args.data_path.split("/")[0]

    data_pathes = [f'{DATA_ROOT}/TEST_DATA/{folder_name}/{args.data_path.split("/")[-2]}/Watermark_Test_{args.data_path.split("/")[-1]}.json']

    for data_path in data_pathes:
        wp_data = []
        with open(data_path, 'r') as f:
            wp_data = json.load(f)

        random.seed(42)
        # random.shuffle(wp_data)
        # wp_data = wp_data[:200]
        print(len(wp_data))

        if args.WhetherThinking:
            ends_str = 'Think_chat' if args.withSysPro else 'Think'
        else:
            ends_str = 'Norm_chat' if args.withSysPro else 'Norm'

        if 'Short' in rephraser_path or is_short:
            # ends_str = '_chat-Short' if args.withSysPro else '-Short'
            folder = args.data_path.split("/")[-2]+"/"+rephraser_path.split("/")[-2] + '-Short/' + ends_str
        else:
            # ends_str = '_chat-Long' if args.withSysPro else '-Long'
            folder = args.data_path.split("/")[-2]+"/"+rephraser_path.split("/")[-2] + '-Long/' + ends_str


        file_name = f'Watermark_Test_{args.data_path.split("/")[-1]}'

        result_save_path = f'{RESULT_ROOT}/DetectedResults_Test/{folder_name}/{folder}/{rephraser_path.split("/")[-1]}'
        os.makedirs(result_save_path, exist_ok=True)
        for text_num in text_nums:
            count = 0
            if os.path.exists(f'{result_save_path}/{file_name}-{text_num}.jsonl'):
                with open(f'{result_save_path}/{file_name}-{text_num}.jsonl', 'r') as f:
                    temp_data = [json.loads(line) for line in f]
                wp_data = wp_data[len(temp_data):]
                print(f'Continue from {len(temp_data)}')

            with open(f'{result_save_path}/{file_name}-{text_num}.jsonl', 'a') as f:
                rephrase_batch =[]
                watermarked_batch = []
                prefix_batch = []
                for data in tqdm(wp_data):
                    original_text = '######Target Text: '+ data['watermarked'] + rephrase_Ins[text_num]
                    prefix_batch.append(data['question'])
                    watermarked_batch.append(data['watermarked'])

                    if args.withSysPro:
                        rephrase_batch.append(rephrase_tokenizer.apply_chat_template(
                                    [{"role": "system", "content": "You are an AI assistant skilled in rewriting prompts in diverse and effective ways. Your can provide well-structured and detailed rewordings that maintain the original meaning while improving clarity and variety."},
                                    {"role": "user", "content": original_text}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=args.WhetherThinking
                                ))
                        
                    else:
                        # rephrase_batch.append(original_text)
                        rephrase_batch.append(rephrase_tokenizer.apply_chat_template(
                                    [{"role": "user", "content": original_text}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=args.WhetherThinking
                                ))

                    # rephrase_batch.append('######Watermarked Text:\n' + data['watermarked_text'] + "\n\n######Instruction:" + data['prompt_gen'] + "\n\n######Rephrased Text:")

                    if len(prefix_batch) == 100:
                        count += 100
                        if args.WhetherThinking:
                            rephrase_gen = paraphrase_actor_local_thinking(rephrase_batch, rephraser, rephrase_tokenizer, args)
                        else:
                            rephrase_gen = paraphrase_actor_local(rephrase_batch, rephraser, rephrase_tokenizer, args)

                        p_sp_scores = GRPO_P_SP(psp_model, psp_args, watermarked_batch, rephrase_gen)
                        sentence_similarity_scores = GRPO_Sim(sentence_model, watermarked_batch, rephrase_gen)
                        eval_results = cal_z_score(rephrase_gen, args, device=device, tokenizer=AutoTokenizer.from_pretrained(resolve_model_id(args.data_path.split("/")[-2])))

                        for i in range(len(prefix_batch)):
                            f.write(json.dumps({
                                'question': prefix_batch[i],
                                'watermarked_text': watermarked_batch[i],
                                'rephrase_gen': rephrase_gen[i],
                                'p_sp_score': f'{p_sp_scores[i]}',
                                'sentence_similarity': f'{sentence_similarity_scores[i]}',
                                'z_score': f'{eval_results["z_score"][i]}',
                                'p_value': f"{eval_results['p_value'][i]}",
                                'is_watermarked': bool(eval_results["z_score"][i] > args.detection_z_threshold),
                            })+'\n')
                        f.flush()
                        rephrase_batch =[]
                        watermarked_batch = []
                        prefix_batch = []

                if len(rephrase_batch) > 0:
                    rephrase_gen = paraphrase_actor_local(rephrase_batch, rephraser, rephrase_tokenizer, args)

                    p_sp_scores = GRPO_P_SP(psp_model, psp_args, watermarked_batch, rephrase_gen)
                    sentence_similarity_scores = GRPO_Sim(sentence_model, watermarked_batch, rephrase_gen)
                    eval_results = cal_z_score(rephrase_gen, args, device=device, tokenizer=AutoTokenizer.from_pretrained(resolve_model_id(args.data_path.split("/")[-2])))

                    for i in range(len(prefix_batch)):
                        f.write(json.dumps({
                            'question': prefix_batch[i],
                            'watermarked_text': watermarked_batch[i],
                            'rephrase_gen': rephrase_gen[i],
                            'p_sp_score': f'{p_sp_scores[i]}',
                            'sentence_similarity': f'{sentence_similarity_scores[i]}',
                            'z_score': f'{eval_results["z_score"][i]}',
                            'p_value': f"{eval_results['p_value'][i]}",
                            'is_watermarked': bool(eval_results["z_score"][i] > args.detection_z_threshold),
                        })+'\n')
                    f.flush()
                    rephrase_batch =[]
                    watermarked_batch = []
                    prefix_batch = []



def pure_rewrite_rephraseTrained(args, rephraser_path="meta-llama/Meta-Llama-3-8B-Instruct", text_nums=[1,2,3,4,5,6,7,8], is_short=False):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    if 'KGW_self' not in args.data_path:
        print('RUNNING NORMAL', args.data_path)
        RunNormal = True
    elif 'KGW_self' in args.data_path:
        print('RUNNING SELF HASH', args.data_path)
        RunNormal = False
    else:
        raise ValueError("Invalid data path, should contain 'KGW_self' or not.")


    print('Use_cuda', args.use_cuda)
    print('max_new_tokens:', args.max_new_tokens)
    print('is_short:', is_short)
    print('WhetherThinking', args.WhetherThinking)


    psp_args = {
                'gpu': 1 if args.use_cuda else 0,
                'load_file': PSP_MODEL_PATH,
                'sp_model': PSP_SENTENCEPIECE_PATH,
                'gpu_id':0
            }
    psp_model, _ = load_model(None, psp_args)
    sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME, device='cuda' if args.use_cuda else 'cpu')

    rephraser, rephrase_tokenizer, device = load_model_vllm(rephraser_path)
    
    # folder_name = 'WM_GEN_Results_Filtered_short' if 'Short' in rephraser_path or is_short else 'WM_GEN_Results_Filtered'
    folder_name = args.data_path.split("/")[0]
    
    if 'semmark' in rephraser_path:
        data_pathes = [f'{DATA_ROOT}/TRAIN_DATA/WM_GEN_Results_semamark_c4/semamark_test_400.jsonl']
    else:
        data_pathes = [f'{DATA_ROOT}/TEST_DATA/{folder_name}/{args.data_path.split("/")[-2]}/Watermark_Test_{args.data_path.split("/")[-1]}.json']

    for data_path in data_pathes:
        wp_data = []
        if '.jsonl' in data_path:
            with open(data_path, 'r') as f:
                wp_data = [json.loads(line) for line in f]
        else:
            with open(data_path, 'r') as f:
                wp_data = json.load(f)

        random.seed(42)
        # random.shuffle(wp_data)
        # wp_data = wp_data[:200]
        print(len(wp_data))


        if args.WhetherThinking:
            ends_str = 'Think_chat' if args.withSysPro else 'Think'
        else:
            ends_str = 'Norm_chat' if args.withSysPro else 'Norm'
        
        if 'ShortShort' in rephraser_path:
            # ends_str = '_chat-Short' if args.withSysPro else '-Short'
            folder = args.data_path.split("/")[-2]+"/"+rephraser_path.split("/")[-2] + '-ShortShort/' + ends_str
        elif 'Short' in rephraser_path or is_short:
            # ends_str = '_chat-Short' if args.withSysPro else '-Short'
            folder = args.data_path.split("/")[-2]+"/"+rephraser_path.split("/")[-2] + '-Short/' + ends_str
        else:
            # ends_str = '_chat-Long' if args.withSysPro else '-Long'
            folder = args.data_path.split("/")[-2]+"/"+rephraser_path.split("/")[-2] + '-Long/' + ends_str


        file_name = f'Watermark_Test_{args.data_path.split("/")[-1]}'

        if RunNormal:
            result_save_path = f'{RESULT_ROOT}/Temp_Test/{folder_name}/{folder}/{rephraser_path.split("/")[-1]}'
        else:
            result_save_path = f'{RESULT_ROOT}/DetectedResults_Test/{folder_name}/{folder}/{rephraser_path.split("/")[-1]}'


        os.makedirs(result_save_path, exist_ok=True)
        for text_num in text_nums:
            count = 0
            if os.path.exists(f'{result_save_path}/{file_name}-{text_num}.jsonl'):
                with open(f'{result_save_path}/{file_name}-{text_num}.jsonl', 'r') as f:
                    temp_data = [json.loads(line) for line in f]
                wp_data = wp_data[len(temp_data):]
                print(f'Continue from {len(temp_data)}')

            with open(f'{result_save_path}/{file_name}-{text_num}.jsonl', 'a') as f:
                rephrase_batch =[]
                watermarked_batch = []
                prefix_batch = []
                for data in tqdm(wp_data):
                    original_text = '######Target Text: '+ data['watermarked'] + rephrase_Ins[text_num]
                    prefix_batch.append(data['question'])
                    watermarked_batch.append(data['watermarked'])

                    if args.withSysPro:
                        rephrase_batch.append(rephrase_tokenizer.apply_chat_template(
                                    [{"role": "system", "content": "You are an AI assistant skilled in rewriting prompts in diverse and effective ways. Your can provide well-structured and detailed rewordings that maintain the original meaning while improving clarity and variety."},
                                    {"role": "user", "content": original_text}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=args.WhetherThinking
                                ))
                        
                    else:
                        # rephrase_batch.append(original_text)
                        rephrase_batch.append(rephrase_tokenizer.apply_chat_template(
                                    [{"role": "user", "content": original_text}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=args.WhetherThinking
                                ))

                    # rephrase_batch.append('######Watermarked Text:\n' + data['watermarked_text'] + "\n\n######Instruction:" + data['prompt_gen'] + "\n\n######Rephrased Text:")

                    if len(prefix_batch) == 100:
                        count += 100
                        if args.WhetherThinking:
                            rephrase_gen = paraphrase_actor_local_thinking(rephrase_batch, rephraser, rephrase_tokenizer, args)
                        else:
                            rephrase_gen = paraphrase_actor_local(rephrase_batch, rephraser, rephrase_tokenizer, args)

                        p_sp_scores = GRPO_P_SP(psp_model, psp_args, watermarked_batch, rephrase_gen)
                        sentence_similarity_scores = GRPO_Sim(sentence_model, watermarked_batch, rephrase_gen)
                        if RunNormal:
                            for i in range(len(prefix_batch)):
                                f.write(json.dumps({
                                    'question': prefix_batch[i],
                                    'watermarked_text': watermarked_batch[i],
                                    'rephrase_gen': rephrase_gen[i],
                                    'p_sp_score': f'{p_sp_scores[i]}',
                                    'sentence_similarity': f'{sentence_similarity_scores[i]}'
                                })+'\n')
                            f.flush()
                        
                        else:
                            eval_results = cal_z_score(rephrase_gen, args, device=device, tokenizer=AutoTokenizer.from_pretrained(resolve_model_id(args.data_path.split("/")[-2])))

                            for i in range(len(prefix_batch)):
                                f.write(json.dumps({
                                    'question': prefix_batch[i],
                                    'watermarked_text': watermarked_batch[i],
                                    'rephrase_gen': rephrase_gen[i],
                                    'p_sp_score': f'{p_sp_scores[i]}',
                                    'sentence_similarity': f'{sentence_similarity_scores[i]}',
                                    'z_score': f'{eval_results["z_score"][i]}',
                                    'p_value': f"{eval_results['p_value'][i]}",
                                    'is_watermarked': bool(eval_results["z_score"][i] > args.detection_z_threshold),
                                })+'\n')
                            f.flush()
                        
                        rephrase_batch =[]
                        watermarked_batch = []
                        prefix_batch = []

                
def pure_rewrite_rephraseTrainedOOD(args, rephraser_path="meta-llama/Meta-Llama-3-8B-Instruct", text_nums=[1,2,3,4,5,6,7,8], is_short=False):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    if 'KGW_self' not in args.data_path:
        print('RUNNING NORMAL', args.data_path)
        RunNormal = True
    elif 'KGW_self' in args.data_path:
        print('RUNNING SELF HASH', args.data_path)
        RunNormal = False
    else:
        raise ValueError("Invalid data path, should contain 'KGW_self' or not.")


    print('Use_cuda', args.use_cuda)
    print('max_new_tokens:', args.max_new_tokens)
    print('is_short:', is_short)
    print('WhetherThinking', args.WhetherThinking)


    psp_args = {
                'gpu': 1 if args.use_cuda else 0,
                'load_file': PSP_MODEL_PATH,
                'sp_model': PSP_SENTENCEPIECE_PATH,
                'gpu_id':0
            }
    psp_model, _ = load_model(None, psp_args)
    sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME, device='cuda' if args.use_cuda else 'cpu')

    rephraser, rephrase_tokenizer, device = load_model_vllm(rephraser_path)
    
    # folder_name = 'WM_GEN_Results_Filtered_short' if 'Short' in rephraser_path or is_short else 'WM_GEN_Results_Filtered'
    folder_name = args.data_path.split("/")[0]
    
    data_pathes = [f'{DATA_ROOT}/TEST_DATA/{folder_name}/{args.data_path.split("/")[-2]}/Watermark_Test_{args.data_path.split("/")[-1]}.json']

    for data_path in data_pathes:
        wp_data = []
        with open(data_path, 'r') as f:
            wp_data = json.load(f)

        random.seed(42)
        # random.shuffle(wp_data)
        # wp_data = wp_data[:200]
        print(len(wp_data))


        if args.WhetherThinking:
            ends_str = 'Think_chat' if args.withSysPro else 'Think'
        else:
            ends_str = 'Norm_chat' if args.withSysPro else 'Norm'
        
        if 'ShortShort' in rephraser_path:
            # ends_str = '_chat-Short' if args.withSysPro else '-Short'
            folder = args.data_path.split("/")[-2]+"/"+rephraser_path.split("/")[-2] + '-ShortShort/' + ends_str
        elif 'Short' in rephraser_path or is_short:
            # ends_str = '_chat-Short' if args.withSysPro else '-Short'
            folder = args.data_path.split("/")[-2]+"/"+rephraser_path.split("/")[-2] + '-Short/' + ends_str
        else:
            # ends_str = '_chat-Long' if args.withSysPro else '-Long'
            folder = args.data_path.split("/")[-2]+"/"+rephraser_path.split("/")[-2] + '-Long/' + ends_str


        file_name = f'Watermark_Test_{args.data_path.split("/")[-1]}'

        if RunNormal:
            result_save_path = f'{RESULT_ROOT}/Temp_TestOOD/{folder_name}/{folder}/{rephraser_path.split("/")[-1]}'
        else:
            result_save_path = f'{RESULT_ROOT}/DetectedResults_TestOOD/{folder_name}/{folder}/{rephraser_path.split("/")[-1]}'


        os.makedirs(result_save_path, exist_ok=True)
        for text_num in text_nums:
            count = 0
            if os.path.exists(f'{result_save_path}/{file_name}-{text_num}.jsonl'):
                with open(f'{result_save_path}/{file_name}-{text_num}.jsonl', 'r') as f:
                    temp_data = [json.loads(line) for line in f]
                wp_data = wp_data[len(temp_data):]
                print(f'Continue from {len(temp_data)}')

            with open(f'{result_save_path}/{file_name}-{text_num}.jsonl', 'a') as f:
                rephrase_batch =[]
                watermarked_batch = []
                prefix_batch = []
                for data in tqdm(wp_data):
                    original_text = '######Target Text: '+ data['watermarked'] + rephrase_Ins[text_num]
                    prefix_batch.append(data['question'])
                    watermarked_batch.append(data['watermarked'])

                    if args.withSysPro:
                        if 'Qwen2.5-3B' in rephraser_path:
                            rephrase_batch.append(maybe_apply_chat_template({'prompt': [{"role": "system", "content": "You are an helpful AI rewriting assistant skilled in rewriting sentences in diverse and effective ways. You can rewrite input sentences into diverse, well-structured forms that preserve meaning and approximate length."},
                                                                                {"role": "user", "content": original_text}]}, rephrase_tokenizer)['prompt'])
                        else:

                            rephrase_batch.append(rephrase_tokenizer.apply_chat_template(
                                        [{"role": "system", "content": "You are an AI assistant skilled in rewriting prompts in diverse and effective ways. Your can provide well-structured and detailed rewordings that maintain the original meaning while improving clarity and variety."},
                                        {"role": "user", "content": original_text}],
                                        tokenize=False,
                                        add_generation_prompt=True,
                                        enable_thinking=args.WhetherThinking
                                    ))
                        
                    else:
                        if 'Qwen2.5-3B' in rephraser_path:
                            rephrase_batch.append(original_text)
                        else:
                            rephrase_batch.append(rephrase_tokenizer.apply_chat_template(
                                    [{"role": "user", "content": original_text}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=args.WhetherThinking
                                ))

                    # rephrase_batch.append('######Watermarked Text:\n' + data['watermarked_text'] + "\n\n######Instruction:" + data['prompt_gen'] + "\n\n######Rephrased Text:")

                    if len(prefix_batch) == 100:
                        count += 100
                        if args.WhetherThinking:
                            rephrase_gen = paraphrase_actor_local_thinking(rephrase_batch, rephraser, rephrase_tokenizer, args)
                        else:
                            rephrase_gen = paraphrase_actor_local(rephrase_batch, rephraser, rephrase_tokenizer, args)

                        p_sp_scores = GRPO_P_SP(psp_model, psp_args, watermarked_batch, rephrase_gen)
                        sentence_similarity_scores = GRPO_Sim(sentence_model, watermarked_batch, rephrase_gen)
                        if RunNormal:
                            for i in range(len(prefix_batch)):
                                f.write(json.dumps({
                                    'question': prefix_batch[i],
                                    'watermarked_text': watermarked_batch[i],
                                    'rephrase_gen': rephrase_gen[i],
                                    'p_sp_score': f'{p_sp_scores[i]}',
                                    'sentence_similarity': f'{sentence_similarity_scores[i]}'
                                })+'\n')
                            f.flush()
                        
                        else:
                            eval_results = cal_z_score(rephrase_gen, args, device=device, tokenizer=AutoTokenizer.from_pretrained(resolve_model_id(args.data_path.split("/")[-2])))

                            for i in range(len(prefix_batch)):
                                f.write(json.dumps({
                                    'question': prefix_batch[i],
                                    'watermarked_text': watermarked_batch[i],
                                    'rephrase_gen': rephrase_gen[i],
                                    'p_sp_score': f'{p_sp_scores[i]}',
                                    'sentence_similarity': f'{sentence_similarity_scores[i]}',
                                    'z_score': f'{eval_results["z_score"][i]}',
                                    'p_value': f"{eval_results['p_value'][i]}",
                                    'is_watermarked': bool(eval_results["z_score"][i] > args.detection_z_threshold),
                                })+'\n')
                            f.flush()
                        
                        rephrase_batch =[]
                        watermarked_batch = []
                        prefix_batch = []


if __name__ == "__main__":
    args = parse_args()
    # if args.is_short:
    #     args.max_new_tokens = 500
    # args.run_num = '1'
    pure_rewrite_rephraseTrained(args, rephraser_path=args.ckpt_path, text_nums=[7], is_short=args.is_short)

    # pure_rewrite_rephraseTrainedOOD(args, rephraser_path=args.ckpt_path, text_nums=[7], is_short=args.is_short)
