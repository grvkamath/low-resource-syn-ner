This folder contains the synthetically generated datasets we use in the paper, the raw model responses generated when trying to build them, and Universal NER datasets, for Danish, Swedish, Slovak and English, in a HuggingFace Datasets format.
Note that during generation, we actually generated synthetic training, validation and test splits, though in the paper we only ever use non-synthetic data as test sets.

* `universal_ner`: Each subfolder within this folder is a language-specific dataset from Universal NER, formatted as a HuggingFace Dataset, which can be loaded with `datasets.load_from_disk(path_to_subfolder)`. 

* `{model}_generations`: These contain both the raw responses from models, as well as cleaned datasets (in the HuggingFace Datasets format) containing whatever generations were usable. 
    * `{language}-{model}`: These subfolders are HuggingFace Datasets, containing the final, clean synthetically-generated data. These can be loaded with `datasets.load_from_disk(path_to_subfolder)`.
    *  `{language}-{split}-{model}_raw.jsonl`: These files contain the raw responses from models. Note that in the case of GPT-4.1, in this repository we omit details like batch request IDs and other API request metadata. 
    * `{language}-{split}-{model}.jsonl`: These files contain whatever readable JSON objects could be recovered from the raw model outputs. 