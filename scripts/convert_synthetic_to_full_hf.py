from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, ClassLabel, Sequence, Value, Features
from utils import convert_json_to_hf_dataset, detect_languages, load_hf_dataset
import os
from tqdm import tqdm

langs = ['kn', 'ta', 'te', 'ml', 'kin', 'swa', 'ibo', 'yor', 'da', 'sv', 'sk']

def lang_to_dataset_tag(lang: str):
    if lang in ['kn', 'ta', 'te', 'ml']:
        return 'ai4b'
    elif lang in ['kin', 'swa', 'ibo', 'yor']:
        return 'masakhaner2'
    elif lang in ['da', 'sv', 'sk']:
        return 'uniner'

def lang_to_dataset_path(lang: str):
    if lang in ['kn', 'ta', 'te', 'ml']:
        return 'ai4bharat/naamapadam'
    elif lang in ['kin', 'swa', 'ibo', 'yor']:
        return 'masakhane/masakhaner2'
    elif lang in ['da', 'sv', 'sk']:
        if lang == 'da':
            return 'datasets/universal_ner/da_ddt'
        elif lang == 'sv':
            return 'datasets/universal_ner/sv_talbanken'
        elif lang == 'sk':
            return 'datasets/universal_ner/sk_snk'


def synthetic_data_jsons_to_full_hf_dataset(lang: str, model="gpt-4.1-2025-04-14", synthetic_data_dir="./datasets/gpt-4_generations/"):
    train_tag = f"{lang}_train"
    validation_tag = f"{lang}_validation"
    test_tag = f"{lang}_test"
    #
    train_path = os.path.join(synthetic_data_dir, f"{train_tag}_{model}.jsonl")
    validation_path = os.path.join(synthetic_data_dir, f"{validation_tag}_{model}.jsonl")
    test_path = os.path.join(synthetic_data_dir, f"{test_tag}_{model}.jsonl")
    #
    organic_hf_dataset_path = lang_to_dataset_path(lang)
    organic_hf_dataset = load_hf_dataset(organic_hf_dataset_path, lang)
    #
    train_split = convert_json_to_hf_dataset(train_path, organic_hf_dataset)
    validation_split = convert_json_to_hf_dataset(validation_path, organic_hf_dataset)
    test_split = convert_json_to_hf_dataset(test_path, organic_hf_dataset)
    #
    hf_dataset = DatasetDict({'train': train_split, 'validation': validation_split, 'test': test_split})
    hf_dataset_local_path = os.path.join(synthetic_data_dir, f"{lang}-{model}")
    hf_dataset.save_to_disk(hf_dataset_local_path)

for model_shorthand in ['aya-expanse-32b', 'Llama-3.1-8B-Instruct']:
    for lang in tqdm(langs):
        print(model_shorthand, lang)
        synthetic_data_jsons_to_full_hf_dataset(lang, model=model_shorthand, synthetic_data_dir=f"./datasets/{model_shorthand}_generations/")

for lang in tqdm(langs):
    synthetic_data_jsons_to_full_hf_dataset(lang, model="gpt-4.1-2025-04-14", synthetic_data_dir=f"./datasets/gpt-4_generations/")



