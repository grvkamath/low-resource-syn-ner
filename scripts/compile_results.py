import pandas as pd
import numpy as np
import json
import os

langs = ['kn', 'ta', 'te', 'ml', 'kin', 'swa', 'yor', 'ibo', 'da', 'sv', 'sk']

fromscratch_synth_eval_results_path = "./results/fromscratch_synth_eval"
finetuning_synth_eval_results_path = "./results/finetuning_synth_eval"

def extract_finetuning_synth_data(lang):
    filepath = f"{lang}.json"
    full_filepath = os.path.join(finetuning_synth_eval_results_path, filepath)
    # Load the JSON data
    with open(full_filepath, 'r') as file:
        data = json.load(file)
    # Prepare a list to collect rows for the DataFrame
    rows = []
    
    # Get the zero-shot F1 score
    zero_shot_f1 = data["zero-shot"]["eval_f1"]
    
    # Iterate through the keys and extract data
    for key, value in data.items():
        if key == "args":
            continue  # Skip the args key
        if key == "pre_training":
            rows.append({"lang": lang, "raw_data_source": "pre_training", "size": 0, "f1": value["eval_f1"]})
        elif key == "zero-shot":
            # Skip this for now; we'll handle it through synthetic sources
            continue
        else:
            # Parse the key for source and size
            parts = key.split('_')
            size = int(parts[-1])
            source = '_'.join(parts[:-1])
            
            # Add the actual data point
            rows.append({"lang": lang, "raw_data_source": source, "size": size, "f1": value["eval_f1"]})
    
    # Add zero-shot entries for all unique sources (apart from pre_training)
    unique_sources = {key.split('_')[0] for key in data if key not in ["args", "pre_training", "zero-shot"]}
    for source in unique_sources:
        rows.append({"lang": lang, "raw_data_source": source, "size": 0, "f1": zero_shot_f1})
    
    # Create the DataFrame
    df = pd.DataFrame(rows)
    
    return df


def extract_fromscratch_synth_data(lang):
    filepath = f"{lang}.json"
    full_filepath = os.path.join(fromscratch_synth_eval_results_path, filepath)
    # Load the JSON data
    with open(full_filepath, 'r') as file:
        data = json.load(file)
    # Prepare a list to collect rows for the DataFrame
    rows = []
    
    # Iterate through the keys and extract data
    for key, value in data.items():
        if key == "args":
            continue  # Skip the args key
        if key == "pre_training":
            rows.append({"lang": lang, "raw_data_source": "pre_training", "size": 0, "f1": value["eval_f1"]})
        elif key == "zero-shot":
            # Skip this for now; we'll handle it through synthetic sources
            continue
        else:
            # Parse the key for source and size
            parts = key.split('_')
            size = int(parts[-1])
            source = '_'.join(parts[:-1])
            
            # Add the actual data point
            rows.append({"lang": lang, "raw_data_source": source, "size": size, "f1": value["eval_f1"]})
    
    # Add zero-shot entries for all unique sources (apart from pre_training)
    unique_sources = {key.split('_')[0] for key in data if key not in ["args", "pre_training", "zero-shot"]}
    for source in unique_sources:
        rows.append({"lang": lang, "raw_data_source": source, "size": 0, "f1": 0})
    # Create the DataFrame
    df = pd.DataFrame(rows)
    
    return df

def get_proper_source_name(data_source):
    if 'ai4bharat' in data_source:
        return 'Naamapadam'
    elif 'universal' in data_source:
        return 'Universal NER'
    elif 'masakhane' in data_source:
        return 'MasakhaNER 2'
    elif 'gpt-4' in data_source:
        return 'GPT-4.1'
    elif 'aya-expanse-32b' in data_source:
        return 'Aya Expanse 32B'
    elif 'Llama-3.1-8B' in data_source:
        return 'Llama-3.1-8B-Instruct'

def check_organic_or_synthetic(raw_data_source):
    if ('ai4bharat' in raw_data_source) or ('universal_ner' in raw_data_source) or ('masakhane' in raw_data_source):
        return 'Organic'
    else:
        return 'Synthetic'

finetuned_synth_results = pd.DataFrame()
for lang in langs:
    lang_df = extract_finetuning_synth_data(lang)
    finetuned_synth_results = pd.concat([finetuned_synth_results, lang_df])

finetuned_synth_results['data_source'] = finetuned_synth_results['raw_data_source'].apply(lambda x: get_proper_source_name(x))
finetuned_synth_results['data_type'] = finetuned_synth_results['raw_data_source'].apply(lambda x: check_organic_or_synthetic(x))

finetuned_synth_results.to_csv("./results/finetuning_synth_eval.csv", index=False)

fromscratch_synth_results = pd.DataFrame()
for lang in langs:
    lang_df = extract_fromscratch_synth_data(lang)
    fromscratch_synth_results = pd.concat([fromscratch_synth_results, lang_df])

fromscratch_synth_results['data_source'] = fromscratch_synth_results['raw_data_source'].apply(lambda x: get_proper_source_name(x))
fromscratch_synth_results['data_type'] = fromscratch_synth_results['raw_data_source'].apply(lambda x: check_organic_or_synthetic(x))

fromscratch_synth_results.to_csv("./results/fromscratch_synth_eval.csv", index=False)


