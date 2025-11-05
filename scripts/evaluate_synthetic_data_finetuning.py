from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
import argparse
from utils import get_tokenized_dataset, convert_json_to_hf_dataset, get_label2id, create_compute_metrics, load_hf_dataset, get_sample_from_hf_dataset
import torch
import os
import json
job = os.getenv("SLURM_JOB_ID")
scratch_dir = os.path.join(os.getenv('SCRATCH'), 'tmp')

model_output_dir = os.path.join(scratch_dir, f"job_{job}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='FacebookAI/xlm-roberta-large')
    parser.add_argument('--gold-standard-dataset', type=str)
    parser.add_argument('--target-lang', type=str)
    parser.add_argument('--pretraining-dataset', type=str)
    parser.add_argument('--pretraining-lang', type=str)
    parser.add_argument('--pretraining-dataset-subset-size', type=int, default=8000)
    parser.add_argument('--synthetic-sources', nargs='+', type=str)
    parser.add_argument('--train-sizes', nargs='+', default=[100,250,500,1000,2500,5000])
    parser.add_argument('--results-directory', type=str, default='./results/finetuning_synth_eval')
    parser.add_argument('--random-seed', type=int, default=3535)
    parser.add_argument('--results-tag', type=str)
    args = parser.parse_args()
    print(args)
    run_experiment(args)



def pretrain_base_ner_model(tokenized_pretraining_dataset, tokenized_gold_standard_dataset, label2id_dict, model, tokenizer, pretraining_dataset_subset_size, random_seed, results_dict, save_path=os.path.join(model_output_dir, 'pretrained_model.model'), num_train_epochs=5):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        report_to="none",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )
    
    compute_metrics_fn = create_compute_metrics(label2id_dict)
    if pretraining_dataset_subset_size:
        train_dataset = get_sample_from_hf_dataset(tokenized_pretraining_dataset["train"], pretraining_dataset_subset_size, random_seed)
    else:
        train_dataset = tokenized_pretraining_dataset["train"]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=tokenized_pretraining_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    
    trainer.train()
    
    print("Same-language test set performance:")
    results = trainer.evaluate(tokenized_pretraining_dataset["test"])
    print(results)
    results_dict['pre_training'] = results
    #
    print("Target language test set performance (zero-shot):")
    results = trainer.evaluate(tokenized_gold_standard_dataset["test"])
    print(results)
    results_dict['zero-shot'] = results
        
    if not save_path == 'None':
        print("Saving the trained model at: ", save_path)
        trainer.save_model(save_path)
    else:
        print("Model not saved.")
    
    return save_path



def finetune_and_evaluate_model(tokenized_train_set, tokenized_validation_set, tokenized_gold_standard_dataset, label2id_dict, model, tokenizer, results_dict, key_name, save_path=None):
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        report_to="none",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True
    )
    
    compute_metrics_fn = create_compute_metrics(label2id_dict)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_validation_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
        
    trainer.train()
    
    print("Gold standard test set performance:")
    results = trainer.evaluate(tokenized_gold_standard_dataset["test"])
    print(results)
    results_dict[key_name] = results

def run_experiment(args):
    model_name = args.model
    gold_standard_dataset_path = args.gold_standard_dataset
    target_lang = args.target_lang
    pretraining_dataset_path = args.pretraining_dataset
    pretraining_lang = args.pretraining_lang
    pretraining_dataset_subset_size = args.pretraining_dataset_subset_size
    synthetic_sources = args.synthetic_sources
    train_sizes = args.train_sizes
    results_directory = args.results_directory
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    #
    random_seed = args.random_seed
    results_tag = args.results_tag
    results_filename = os.path.join(results_directory, f"{results_tag}.json")
    master_results = {}
    master_results['args'] = str(args)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #
    print("Loading and processing datasets...")
    gold_standard_dataset = load_hf_dataset(gold_standard_dataset_path, target_lang)
    label2id_dict = get_label2id(gold_standard_dataset)
    tokenized_gold_standard_dataset = get_tokenized_dataset(gold_standard_dataset, tokenizer)
    pretraining_dataset = load_hf_dataset(pretraining_dataset_path, pretraining_lang)
    tokenized_pretraining_dataset = get_tokenized_dataset(pretraining_dataset, tokenizer)
    synthetic_datasets = [load_hf_dataset(path, target_lang) for path in synthetic_sources]
    tokenized_synthetic_datasets = [get_tokenized_dataset(x, tokenizer) for x in synthetic_datasets]
    print("Done!")
    print(f"Starting pre-training on {pretraining_lang} NER Data:")
    model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels = len(label2id_dict['label_list']),
                id2label = label2id_dict['id2label'],
                label2id = label2id_dict['label2id']
            )
    pretrained_model_path = pretrain_base_ner_model(
        tokenized_pretraining_dataset=tokenized_pretraining_dataset,
        tokenized_gold_standard_dataset=tokenized_gold_standard_dataset,
        label2id_dict=label2id_dict,
        model=model,
        tokenizer=tokenizer,
        pretraining_dataset_subset_size=pretraining_dataset_subset_size,
        random_seed=random_seed,
        results_dict=master_results
    )
    print("Finished pretraining, now finetuning on target language...") 
    for train_size in train_sizes:
        # train_size_results = {}
        if len(gold_standard_dataset["train"]) < train_size:
            key_name = f'{gold_standard_dataset_path}_{len(gold_standard_dataset["train"])}'
        else:
            key_name = f'{gold_standard_dataset_path}_{train_size}'
        #
        if key_name not in master_results.keys():
            torch.cuda.empty_cache()
            model = AutoModelForTokenClassification.from_pretrained(pretrained_model_path)
            tokenized_gold_standard_train_set_sample = get_sample_from_hf_dataset(tokenized_gold_standard_dataset["train"], train_size, random_seed)
            tokenized_gold_standard_validation_set_sample = get_sample_from_hf_dataset(tokenized_gold_standard_dataset["validation"], train_size, random_seed)
            print(f"Training on {gold_standard_dataset_path}, size: {train_size}")
            finetune_and_evaluate_model(
                tokenized_train_set=tokenized_gold_standard_train_set_sample,
                tokenized_validation_set=tokenized_gold_standard_validation_set_sample,
                tokenized_gold_standard_dataset=tokenized_gold_standard_dataset,
                label2id_dict=label2id_dict,
                model=model,
                tokenizer=tokenizer,
                results_dict=master_results,
                key_name=key_name)
        #
        for tokenized_synthetic_dataset, synthetic_data_path in zip(tokenized_synthetic_datasets, synthetic_sources):
            if len(tokenized_synthetic_dataset["train"]) < train_size:
                key_name = f'{synthetic_data_path}_{len(tokenized_synthetic_dataset["train"])}'
            else:
                key_name = f'{synthetic_data_path}_{train_size}'
            #
            if key_name not in master_results.keys():
                torch.cuda.empty_cache()
                model = AutoModelForTokenClassification.from_pretrained(pretrained_model_path)
                tokenized_synthetic_train_set_sample = get_sample_from_hf_dataset(tokenized_synthetic_dataset["train"], train_size, random_seed)
                tokenized_synthetic_validation_set_sample = get_sample_from_hf_dataset(tokenized_synthetic_dataset["validation"], train_size, random_seed)
                print(f"Training on {synthetic_data_path}, size: {train_size}")
                finetune_and_evaluate_model(
                    tokenized_train_set=tokenized_synthetic_train_set_sample,
                    tokenized_validation_set=tokenized_synthetic_validation_set_sample,
                    tokenized_gold_standard_dataset=tokenized_gold_standard_dataset,
                    label2id_dict=label2id_dict,
                    model=model,
                    tokenizer=tokenizer,
                    results_dict=master_results,
                    key_name=key_name)
            #
        #
    #
    print(f"Writing all results to {results_filename}...")
    with open(results_filename, 'w') as json_file:
        json.dump(master_results, json_file, indent=4)
    print("All done!")
    

if __name__=="__main__":
    main()


