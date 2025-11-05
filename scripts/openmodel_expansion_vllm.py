import os
import re
import json
import pandas as pd
import numpy as np
import torch
import random
import argparse
from tqdm import tqdm
from typing import Union, Optional
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils import load_hf_dataset
from datasets import Dataset
import time 

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated-data-directory', type=str, default='./datasets/')
    parser.add_argument('--source-dataset', type=str)
    parser.add_argument('--source-dataset-lang', type=str)
    parser.add_argument('--source-dataset-split', type=str, default='train')
    parser.add_argument('--source-dataset-subset-size', type=int, default=None)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--n-generated', type=int, default=20)
    parser.add_argument('--n-repeats', type=int, default=250)
    parser.add_argument('--random-seed', type=int, default=3535)
    parser.add_argument('--prompt', type=str, default="Help me expand a {0} Named Entity Recognition dataset. Below are a set of examples of datapoints. On the basis of these examples, please give me {1} new datapoints, formatted as a JSON object. Make sure the examples are unique and diverse!")
    parser.add_argument('--system-prompt', type=str, default="You are a helpful model that helps build text-based datasets, but does not produce any conversation besides the text it is asked to produce. You only output JSON strings.")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--quantization', type=str, default=None)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--results-tag', type=str)
    args = parser.parse_args()
    print(f"Arguments: {args}")
    generate(args)
    end_time = time.time()
    print(f"Time taken:")
    print(f"{(end_time - start_time)/60} minutes")


# Clear cache on all available GPUs
def clear_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(i)
            print(f"Cleared memory on GPU {i}")


def extract_json_objects(input_string):
    # Regular expression to capture a valid JSON object
    json_pattern = r'\{(?:[^{}])*?\}'
    # Search for the first occurrence of a JSON object
    matches = re.findall(json_pattern, input_string)
    if len(matches)>0:
        return matches
    else:
        return None

def get_lang_name(lang):
    lang_dict = {"ta": "Tamil", "kn": "Kannada", "te": "Telugu", "ml": "Malayalam", "kin": "Kinyarwanda", "ibo": "Igbo", "swa": "Swahili", "yor": "Yoruba", "da": "Danish", "sk": "Slovak", "sv": "Swedish"}
    return lang_dict[lang]

def get_model_shorthand(model_name):
    # Removes e.g. "meta-llama/" from "meta-llama/Llama-3.1-8B-Instruct"
    if "meta-llama" in model_name:  
        return model_name[11:]
    elif "CohereLabs" in model_name:
        return model_name[11:]
    elif "fsaudm" in model_name:
        return model_name[7:]
    elif "unsloth" in model_name:
        return model_name[8:]
    elif "hugging-quants" in model_name:
        return model_name[15:]


def load_data(dataset_path: str, language: str, split='train'):
    data = load_hf_dataset(dataset_path, language, split=split, trust_remote_code=True)
    return pd.DataFrame(data)

def vllm_setup(model_name: str, quantization: str, tensor_parallel_size: int, max_model_len: int):
    if quantization==False:
        vllm_llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)
    else:
        vllm_llm = LLM(model=model_name, quantization=quantization, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len, trust_remote_code=True)
    #
    return vllm_llm

def vllm_batch_call(vllm_llm, pipeline_input_list, sampling_params):
    vllm_outputs = vllm_llm.generate(pipeline_input_list, sampling_params)
    raw_outputs = [output.outputs[0].text for output in vllm_outputs]
    return raw_outputs


def check_and_add_data(raw_output, data_list, verbose=True):
    json_object_candidates = extract_json_objects(raw_output)
    if json_object_candidates:
        for json_object_candidate in json_object_candidates:
            try:
                data = json.loads(json_object_candidate)
                data_list.append(data)
            except:
                print(f"Error with datapoint: {json_object_candidate}")
    else:
        print(f"Error with data batch: {raw_output}")


def prepare_pipeline_inputs(tokenizer, base_df: pd.DataFrame, lang: str, n_calls: int, sample_size: int, n_generated_per_call: int, prompt: str, system_prompt: str, random_seed: int):
    random.seed(random_seed)
    lang_full = get_lang_name(lang)
    prompt_formatted = prompt.format(lang_full, n_generated_per_call)
    message_list = []
    for i in range(n_calls):
        random_state = random.randint(0,1000)
        sample = base_df.sample(n=sample_size, random_state=random_state)
        examples = sample.to_json(orient='records',lines=True)
        total_prompt = prompt_formatted+"\n\n"+examples+"\n"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": total_prompt}]
        message_list.append(messages)
    #
    pipeline_input_list = [tokenizer.apply_chat_template(x, tokenize=False) for x in message_list]
    return pipeline_input_list


def generate_data_from_df(vllm_llm, tokenizer, sampling_params, base_df: pd.DataFrame, lang: str, n_calls: int, sample_size: int, n_generated_per_call: int, prompt: str, system_prompt: str, random_seed=None, verbose=True):
    random.seed(random_seed)
    data_list = []
    # raw_data = []
    pipeline_input_list = prepare_pipeline_inputs(tokenizer, base_df, lang, n_calls, sample_size, n_generated_per_call, prompt, system_prompt, random_seed)
    print("Starting on Calls to Model...")
    raw_data = vllm_batch_call(vllm_llm, pipeline_input_list, sampling_params)
    #
    print("Processing Data...")
    for raw_datapoint in raw_data:
        check_and_add_data(raw_datapoint, data_list)
    #
    print("Done!")
    return {'data_list': data_list, 'raw_data': raw_data}  


def write_data_to_json(data_list: list, output_directory: str, filename: str):
    with open(os.path.join(output_directory, filename), 'w') as file:
        for dictionary in data_list:
            json_line = json.dumps(dictionary)
            file.write(json_line + '\n')

def write_raw_outputs_to_json(raw_data: list, output_directory: str, filename: str):
    with open(os.path.join(output_directory, filename), 'w') as file:
        for i, raw_output in enumerate(raw_data):
            dictionary = {'call_idx': i, 'raw_output': raw_output}
            json_line = json.dumps(dictionary)
            file.write(json_line + '\n')

def generate(args):
    model_name = args.model
    model_shorthand = get_model_shorthand(model_name)
    generated_data_directory = args.generated_data_directory
    full_generated_data_directory = os.path.join(generated_data_directory, f"{model_shorthand}_generations")
    if not os.path.exists(full_generated_data_directory):
        os.makedirs(full_generated_data_directory)
    #
    source_dataset_path = args.source_dataset
    source_dataset_lang = args.source_dataset_lang
    source_dataset_split = args.source_dataset_split
    source_dataset_subset_size = args.source_dataset_subset_size
    sample_size = args.sample_size
    n_generated_per_call = args.n_generated
    n_calls = args.n_repeats
    random_seed = args.random_seed
    prompt = args.prompt
    system_prompt = args.system_prompt
    quantization = args.quantization
    top_p = args.top_p
    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    max_model_len = args.max_model_len
    tensor_parallel_size = args.n_gpus
    verbose = args.verbose
    results_tag = args.results_tag
    #
    print("Loading vllm model and HF tokenizer...")
    # vllm_llm = vllm_setup(model_name, quantization, tensor_parallel_size, max_model_len)
    vllm_llm = LLM(model_name, quantization=quantization, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    print("Done!")
    print("Loading dataset...")
    base_df = load_data(source_dataset_path, source_dataset_lang, source_dataset_split)
    if source_dataset_subset_size:
        base_df = base_df.sample(n=source_dataset_subset_size, random_state=random_seed)
    print("Done!")
    print("Generating data...")
    data = generate_data_from_df(vllm_llm=vllm_llm, tokenizer=tokenizer, sampling_params=sampling_params, base_df=base_df, lang=source_dataset_lang, n_calls=n_calls, sample_size=sample_size, n_generated_per_call=n_generated_per_call, prompt=prompt, system_prompt=system_prompt, random_seed=random_seed, verbose=verbose)
    print("Done!")
    data_list = data['data_list']
    raw_data = data['raw_data']
    filename = f"{results_tag}_{model_shorthand}.jsonl"
    raw_data_filename = f"{results_tag}_{model_shorthand}_raw_outputs.jsonl"
    print(f"Writing data to file at {filename}...")
    write_data_to_json(data_list=data_list, output_directory=full_generated_data_directory, filename=filename)
    write_raw_outputs_to_json(raw_data=raw_data, output_directory=full_generated_data_directory, filename=raw_data_filename)
    print("Done!")
    print("Clearing GPU memory...")
    del(vllm_llm)
    clear_gpu_memory()
    print("All done!")

if __name__ == "__main__":
    main()




