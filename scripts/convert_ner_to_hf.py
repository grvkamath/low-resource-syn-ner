'''
Script to convert Universal-NER .iob2 files into Huggingface-style datasets.
Searches the raw files, and where we have train, validation and test splits, formats them into one single HF-style dataset.
'''

import os
import pandas as pd
from datasets import Dataset, DatasetDict
from utils import convert_uniner_iob2_to_hf
from tqdm import tqdm 

raw_file_directory = os.getenv("RAW_UNER_DIRECTORY") # Directory containing the raw UNER .iob2 files
processed_file_directory = './datasets/universal_ner'
subfolders = [x for x in os.listdir(raw_file_directory) if os.path.isdir(os.path.join(raw_file_directory, x))]

for subfolder in tqdm(subfolders):
    full_subfolder_filepath = os.path.join(raw_file_directory, subfolder)
    dataset_files = os.listdir(full_subfolder_filepath)
    train_file = next((filename for filename in dataset_files if filename.endswith('-train.iob2')), None)
    validation_file = next((filename for filename in dataset_files if filename.endswith('-dev.iob2')), None)
    test_file = next((filename for filename in dataset_files if filename.endswith('-test.iob2')), None)
    if train_file and validation_file and test_file:
        train_hf = convert_uniner_iob2_to_hf(os.path.join(full_subfolder_filepath, train_file))
        validation_hf = convert_uniner_iob2_to_hf(os.path.join(full_subfolder_filepath, validation_file))
        test_hf = convert_uniner_iob2_to_hf(os.path.join(full_subfolder_filepath, test_file))
        hf_final = DatasetDict({'train': train_hf, 'test': test_hf, 'validation': validation_hf})
        hf_final.save_to_disk(os.path.join(processed_file_directory, f"{subfolder}"))

