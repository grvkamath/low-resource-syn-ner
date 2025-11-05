from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
import argparse
from utils import get_tokenized_dataset, convert_json_to_hf_dataset, get_label2id, create_compute_metrics, load_hf_dataset, get_sample_from_hf_dataset
import torch
import os
import json
from evaluate_synthetic_data_finetuning import pretrain_base_ner_model
scratch_dir = os.path.join(os.getenv('SCRATCH'), 'tmp')
job = os.getenv("SLURM_JOB_ID")
scratch_dir = os.path.join(os.getenv('SCRATCH'), 'tmp')

model_output_dir = os.path.join(scratch_dir, f"job_{job}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--pre-base-hf-dataset', type=str)
    parser.add_argument('--pre-base-hf-dataset-sample-size', type=int, default=None)
    parser.add_argument('--pre-base-lang', type=str)
    parser.add_argument('--organic-hf-dataset', type=str)
    parser.add_argument('--organic-hf-dataset-lang', type=str)
    parser.add_argument('--synthetic-data-paths', nargs='+', type=str)
    parser.add_argument('--synthetic-data-langs', nargs='+', type=str)
    parser.add_argument('--synthetic-data-sample-sizes', nargs='+', type=int)
    parser.add_argument('--random-seed', type=int, default=3535)
    parser.add_argument('--results-directory', type=str, default='./results/')
    parser.add_argument('--results-tag', type=str, default='')
    args = parser.parse_args()
    train_and_evaluate(args)


def finetune_on_synthetic_data(synthetic_data, name, organic_tokenized_dataset, label2id_dict, trained_base_model_path, tokenizer, results_dict):
    trained_model = AutoModelForTokenClassification.from_pretrained(trained_base_model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=scratch_dir,
        report_to="none",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    
    compute_metrics_fn = create_compute_metrics(label2id_dict)
    
    trainer = Trainer(
            model=trained_model,
            args=training_args,
            train_dataset=synthetic_data,
            eval_dataset=organic_tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn,
        )
    
    trainer.train()
    print("Base dataset test set performance:")
    results = trainer.evaluate(organic_tokenized_dataset["test"])
    print(results)
    results_dict[f'FT_{name}'] = results


def train_synthetic_only_model(synthetic_tokenized_dataset, name, organic_tokenized_dataset, label2id_dict, model, tokenizer, results_dict, save_path=None):
    # save_path = myconfig["save_path"]
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=scratch_dir,
        report_to="none",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    
    compute_metrics_fn = create_compute_metrics(label2id_dict)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=synthetic_tokenized_dataset,
        eval_dataset=organic_tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    print("Model being trained only on synthetic data...")
    
    trainer.train()
    
    print("Test set performance:")
    results = trainer.evaluate(organic_tokenized_dataset["test"])
    print(results)
    results_dict[f'BASE_{name}'] = results



def train_and_evaluate(args):
    print(args)
    model_name = args.model
    pre_base_hf_dataset_path = args.pre_base_hf_dataset
    pre_base_hf_dataset_sample_size = args.pre_base_hf_dataset_sample_size
    pre_base_lang = args.pre_base_lang
    organic_hf_dataset_path = args.organic_hf_dataset
    organic_hf_dataset_lang = args.organic_hf_dataset_lang
    synthetic_data_paths = args.synthetic_data_paths
    synthetic_data_langs = args.synthetic_data_langs
    synthetic_data_sample_sizes = args.synthetic_data_sample_sizes
    random_seed = args.random_seed
    results_directory = args.results_directory
    results_tag = args.results_tag
    results_filename = os.path.join(results_directory, f"eval_{results_tag}.json")
    master_results = {}
    master_results['args'] = str(args)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pre_base_hf_dataset = load_hf_dataset(pre_base_hf_dataset_path, pre_base_lang, trust_remote_code=True)
    organic_hf_dataset = load_hf_dataset(organic_hf_dataset_path, organic_hf_dataset_lang, trust_remote_code=True)
    label2id_dict = get_label2id(pre_base_hf_dataset)
    synthetic_dataset_dictionary = {}
    for path, lang, sample_size in zip(synthetic_data_paths, synthetic_data_langs, synthetic_data_sample_sizes):
        if sample_size==0:
            sample_size=None
        if 'datasets/' in path: # If it's local json files
            synthetic_data = convert_json_to_hf_dataset(path, organic_hf_dataset)
        else: # If it's e.g. Wikiann
            synthetic_data = load_hf_dataset(path, lang)['train']
        if sample_size:
            synthetic_data = get_sample_from_hf_dataset(synthetic_data, sample_size, random_seed)
        synthetic_tokenized_dataset = get_tokenized_dataset(synthetic_data, tokenizer)
        synthetic_dataset_dictionary[f"{path}_{sample_size}"] = synthetic_tokenized_dataset
    #
    pre_base_tokenized_dataset = get_tokenized_dataset(pre_base_hf_dataset, tokenizer)
    organic_tokenized_dataset = get_tokenized_dataset(organic_hf_dataset, tokenizer)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id_dict['label_list']),
        id2label=label2id_dict['id2label'],
        label2id=label2id_dict['label2id']
        )
    #
    print(f"-------------------\nPre-training on {pre_base_lang} dataset...\n------------------------")
    if pre_base_hf_dataset_sample_size:
        print(f"NOTE: Using only a sample of {pre_base_hf_dataset_sample_size} datapoints from NER pre-training dataset, using random seed {random_seed}")
    trained_base_model_path = pretrain_base_ner_model(pre_base_tokenized_dataset, organic_tokenized_dataset, label2id_dict, model, tokenizer, pre_base_hf_dataset_sample_size, random_seed, master_results)
    print(f"-------------------\nStarting on fine-tuning on synthetic datasets...\n------------------------")
    for name, synthetic_tokenized_dataset in synthetic_dataset_dictionary.items():
        print(f"-------------------\nFine-tuning on synthetic data titled: {name}:\n------------------------")
        finetune_on_synthetic_data(synthetic_tokenized_dataset, name, organic_tokenized_dataset, label2id_dict, trained_base_model_path, tokenizer, master_results)
    #
    print(f"-------------------\nStarting on training from scratch using synthetic datasets...\n------------------------")
    for name, synthetic_tokenized_dataset in synthetic_dataset_dictionary.items():
        print(f"-------------------\nTraining from scratch on synthetic data titled: {name}:\n------------------------")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id_dict['label_list']),
            id2label=label2id_dict['id2label'],
            label2id=label2id_dict['label2id']
            )
        train_synthetic_only_model(synthetic_tokenized_dataset, name, organic_tokenized_dataset, label2id_dict, model, tokenizer, master_results)
    #
    print(f"Writing all results to {results_filename}...")
    with open(results_filename, 'w') as json_file:
        json.dump(master_results, json_file, indent=4)
    print("All done!")

if __name__=='__main__':
    main()

