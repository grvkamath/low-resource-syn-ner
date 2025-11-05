import os 
import json
import pandas as pd 
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, ClassLabel, Sequence, Value, Features
import evaluate
import uuid
import numpy as np
import unicodedata
# from torch.utils.data import Dataset as torch_dataset

def load_seqeval_metric():
    # Try to make a stable-but-unique id per process
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    exp_id = f"seqeval-{rank}-{uuid.uuid4()}-{slurm_job_id}"
    return evaluate.load("seqeval", experiment_id=exp_id)


auth_token=os.getenv('AUTH_TOKEN')

def load_hf_dataset(path, lang, **kwargs):
    if 'datasets/' in path: # This should catch cases where the dataset is local; pretty sure we won't get false positives
        data = load_from_disk(path)
        if 'split' in kwargs:
            data = data[kwargs['split']]
        return data
    else:
        return load_dataset(path, lang, **kwargs) 

def get_sample_from_hf_dataset(hf_dataset_split, sample_size, random_seed=3535):
    pandas_version = pd.DataFrame(hf_dataset_split)
    if sample_size < len(pandas_version):
        pandas_version_sample = pandas_version.sample(n=sample_size, random_state=random_seed)
    else:
        pandas_version_sample = pandas_version
        print(f"Sample size greater than dataset size, so returning original dataset of length {len(pandas_version)}")
    return Dataset.from_pandas(pandas_version_sample, preserve_index=False)

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def get_tokenized_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(tokenizer, x), batched=True)
    return tokenized_dataset


def is_valid_utf8(text):
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False

def convert_json_to_hf_dataset(jsonl_filepath, hf_dataset):
    # Load the .jsonl file into a pandas DataFrame
    jsonl_data = []
    with open(jsonl_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                assert isinstance(data, dict), f"Datapoint may be faulty: {line}"
                jsonl_data.append(json.loads(line))
            except:
                continue
    #
    jsonl_df = pd.DataFrame(jsonl_data)
    
    # Get the column names of the train split of the Huggingface dataset
    train_columns = hf_dataset['train'].column_names
    train_columns = [col for col in train_columns if col != 'id']
    
    # Get the possible labels from the train split of the Huggingface dataset
    labels = [tag for taglist in hf_dataset['train']['ner_tags'] for tag in taglist]
    unique_labels = set(labels)
    
    # Filter the DataFrame to retain only true (non-hallucinated) columns, and remove any NA entries
    filtered_df = jsonl_df[train_columns].dropna()
    
    # Also remove any faulty rows where len(tokens) and len(ner_tags) don't match
    bool_filter = filtered_df['ner_tags'].apply(lambda x: len(x)) == filtered_df['tokens'].apply(lambda x: len(x))
    filtered_df = filtered_df[bool_filter]
    
    # Also remove any faulty rows that include ner tags not present in the training set of the HF dataset
    bool_filter = filtered_df['ner_tags'].apply(lambda x: set(x).issubset(unique_labels))
    filtered_df = filtered_df[bool_filter]
    
    # Get features from HF dataset:
    features = hf_dataset['train'].features

    # Standardize data types:
    filtered_df['tokens'] = filtered_df['tokens'].apply(lambda x: [str(y) for y in x])
    filtered_df['ner_tags'] = filtered_df['ner_tags'].apply(lambda x: [int(y) for y in x])
    if 'id' in filtered_df.columns:
        filtered_df['id'] = filtered_df['id'].apply(lambda x: str(x))
    
    # Remove any cases of bad utf-8 encoding:
    bool_filter = filtered_df['tokens'].apply(lambda lst: all(is_valid_utf8(text) for text in lst))
    filtered_df = filtered_df[bool_filter]
    
    # Convert the filtered DataFrame into a Huggingface Dataset
    filtered_dataset = Dataset.from_pandas(filtered_df, features=features, preserve_index=False)
    
    return filtered_dataset

def append_jsonl_to_hf_dataset(jsonl_filepath, hf_dataset):
    train_columns = hf_dataset['train'].column_names
    jsonl_converted = convert_json_to_hf_dataset(jsonl_filepath, hf_dataset)
    
    # Append the filtered data to the train split of the Huggingface dataset
    hf_dataset['train'] = Dataset.from_dict({
        key: hf_dataset['train'][key] + jsonl_converted[key]
        for key in train_columns
    })
    
    return hf_dataset

def get_label2id(hf_dataset):
    label2id_dict = {}
    label_list = hf_dataset["train"].features[f"ner_tags"].feature.names
    id2label = {k: v for k, v in enumerate(label_list)}
    label2id = {v: k for k, v in enumerate(label_list)}
    label2id_dict['label_list'] = label_list
    label2id_dict['id2label'] = id2label
    label2id_dict['label2id'] = label2id
    return label2id_dict

def create_compute_metrics(label2id_dict):
    def compute_metrics(p):
        seqeval = load_seqeval_metric()
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        label_list  = label2id_dict['label_list']
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return compute_metrics


def convert_uniner_iob2_to_hf(filepath, sep="\t"):
    # Using IOB2 Schema:
    label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    # Extremely hacky, but I'm not sure what else to do: the sr_set and hr_set datasets specifically have got a 'MISC' too, labelled 'OTH'
    if 'sr_set' in filepath or 'hr_set' in filepath:
        label_names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-OTH', 'I-OTH']
    
    class_label = ClassLabel(names=label_names)
    
    master_tokens = []
    master_ner_tags = []
    
    with open(filepath, 'r') as file:
        datapoint_tokens = []
        datapoint_ner_tags = []
        for line in file:
            if len(line.strip()) < 1:
                if len(datapoint_tokens) > 0:
                    master_tokens.append(datapoint_tokens)
                if len(datapoint_ner_tags) > 0:
                    master_ner_tags.append([class_label.str2int(tag) for tag in datapoint_ner_tags])
                datapoint_tokens = []
                datapoint_ner_tags = []
            else:
                if line.strip()[0] != '#':
                    line_data = line.strip().split(sep)
                    token = line_data[1]
                    ner_tag = line_data[2]
                    datapoint_tokens.append(token)
                    datapoint_ner_tags.append(ner_tag)
    
    output_df = pd.DataFrame()
    output_df['tokens'] = master_tokens
    output_df['ner_tags'] = master_ner_tags
    
    features = Features({
        'tokens': Sequence(feature=Value(dtype='string')),
        'ner_tags': Sequence(feature=class_label)
    })
    
    output_df_hf = Dataset.from_pandas(output_df, features=features, preserve_index=False)
    
    return output_df_hf

