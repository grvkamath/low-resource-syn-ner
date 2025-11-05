import os
import argparse
import pandas as pd
import numpy as np 
from openai import OpenAI
from pydantic import BaseModel
from datasets import load_dataset 
import random 
from utils import load_hf_dataset
import json

client = OpenAI(max_retries=5)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-file-directory', type=str, default="./api_files/expansion")
    parser.add_argument('--source-dataset', type=str)
    parser.add_argument('--source-dataset-lang', type=str)
    parser.add_argument('--source-dataset-split', type=str, default='train')
    parser.add_argument('--source-dataset-subset-size', type=int, default=None)
    parser.add_argument('--sample-size', type=int, default=10)
    parser.add_argument('--n-generated', type=int, default=20)
    parser.add_argument('--n-repeats', type=int, default=50)
    parser.add_argument('--random-seed', type=int, default=3535)
    parser.add_argument('--json-format', type=bool, default=True)
    parser.add_argument('--prompt', type=str, default="Help me make a {0} Named Entity Recognition dataset. Please give me {1} new datapoints, formatted as a single JSON object. Make sure the examples are unique and diverse. Here are some examples to get you started:")
    parser.add_argument('--system-prompt', type=str, default="You are a helpful model that helps build text-based datasets, but does not produce any conversation besides the text it is asked to produce.")
    parser.add_argument('--model', type=str, default='gpt-4.1-2025-04-14')
    parser.add_argument('--results-tag', type=str)
    args = parser.parse_args()
    print(f"Arguments: {args}")
    make_batch_API_request(args)


def get_language_full_name(language: str):
    lang_dict = {'da':'Danish', 'sv': 'Swedish', 'sk': 'Slovak', 'ibo': 'Igbo', 'kin': 'Kinyarwanda', 'swa': 'Swahili', 'yor': 'Yoruba', 'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'kn': 'Kannada', 'en': 'English'}
    return lang_dict[language]

def load_data(dataset_path: str, language: str, split='train'):
    data = load_hf_dataset(dataset_path, language, split=split, trust_remote_code=True)
    return pd.DataFrame(data)


def API_dict_from_sample(sample: pd.DataFrame, language_full: str, n_generated: int, prompt: str, system_prompt:str, json_format=True, model='gpt-4.1-2025-04-14', **kwargs):
    if json_format==True:
        if ('gpt-4o' in model) or ('gpt-4-' in model): # Older models don't support full json schemas
            response_format={ "type": "json_object" }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ner_batch_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "ner_datapoints": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "tokens": {
                                            "type": "array",
                                            "items": {"type": "string"}
                                        },
                                        "ner_tags": {
                                            "type": "array",
                                            "items": {"type": "integer"}
                                        }
                                    },
                                    "required": ["tokens", "ner_tags"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["ner_datapoints"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
    else:
        response_format=None
    #
    prompt_formatted = prompt.format(language_full, n_generated)
    examples = sample.to_json(orient='records',lines=True)
    total_prompt = prompt_formatted+"\n\n"+examples+"\n"
    messages_for_api = [{"role": "system", "content": system_prompt}, {"role": "user", "content": total_prompt}]    
    output_dict = {"method": "POST", "url": "/v1/chat/completions", "body": {"model": model, "messages": messages_for_api, "response_format": response_format} }
    return output_dict

def generate_API_dictlist(base_df: pd.DataFrame, language_full: str, n_generated: int, n_repeats: int, sample_size: int, prompt: str, system_prompt: str, json_format=True, model='gpt-4.1-2025-04-14', random_seed=3535, **kwargs):
    random.seed(random_seed)
    API_dicts = []
    for i in range(n_repeats):
        random_state = random.randint(0,1000)
        sample = base_df.sample(n=sample_size, random_state=random_state)
        API_dict = API_dict_from_sample(sample=sample, language_full=language_full, n_generated=n_generated, prompt=prompt, system_prompt=system_prompt, json_format=json_format, model=model)
        API_dict['custom_id'] = f"request-{i}"
        API_dicts.append(API_dict)
    return API_dicts


def write_jsonl_file(dictlist, path):
    with open(path, 'w') as jsonl_file:
        for d in dictlist:
            jsonl_file.write(json.dumps(d) + '\n')


def make_batch_API_request(args):
    api_file_directory = args.api_file_directory
    if not os.path.exists(api_file_directory):
        os.makedirs(api_file_directory)
    source_dataset_path = args.source_dataset 
    source_dataset_lang = args.source_dataset_lang
    source_dataset_lang_full = get_language_full_name(source_dataset_lang)
    source_dataset_split = args.source_dataset_split
    source_dataset_subset_size = args.source_dataset_subset_size
    sample_size = args.sample_size
    n_generated_per_call = args.n_generated
    n_calls = args.n_repeats
    random_seed = args.random_seed
    json_format = args.json_format
    prompt = args.prompt
    system_prompt = args.system_prompt
    model = args.model
    results_tag = args.results_tag
    #
    API_filepath = os.path.join(api_file_directory, f"{results_tag}.jsonl")
    print("Loading dataset...")
    base_df = load_data(source_dataset_path, source_dataset_lang, source_dataset_split)
    if source_dataset_subset_size:
        base_df = base_df.sample(n=source_dataset_subset_size, random_state=random_seed)
    print("Done!")
    print("Processing data and writing .jsonl file for API batching...")
    dictlist = generate_API_dictlist(base_df=base_df, language_full=source_dataset_lang_full, n_generated=n_generated_per_call, n_repeats=n_calls, sample_size=sample_size, prompt=prompt, system_prompt=system_prompt, json_format=json_format, model=model, random_seed=random_seed)
    write_jsonl_file(dictlist, API_filepath)
    print("Done!")
    print("Uploading .jsonl file...")
    batch_input_file = client.files.create(
        file=open(API_filepath, "rb"),
        purpose="batch"
        )
    print("Done!")
    print("Creating batch...")
    batch_input_file_id = batch_input_file.id
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": results_tag
        })
    print("Done!")
    batch_info_dict = batch_info.to_dict()
    batch_info_filepath = os.path.join(api_file_directory, f"{results_tag}_batch_info.json")
    with open(batch_info_filepath, 'w') as json_file:
        json.dump(batch_info_dict, json_file, indent=4)
    #
    print(f"Batch info saved at {batch_info_filepath}!")

if __name__=="__main__":
    main()


    
