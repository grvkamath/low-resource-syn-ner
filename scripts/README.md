* `compare_synthetic_data.py`: Script used to compare different synthetic datasets in terms of model performance.

* `compile_results.py`: Takes language-specific `.json` results files and compiles them into `results/finetuning_synth_eval.csv` and `results/finetuning_synth_eval.csv`.

* `convert_ner_to_hf.py`: Script used to convert Universal NER .iob2 data files to the HuggingFace Datasets format.

* `convert_synthetic_to_full_hf.py`: Script used to convert synthetically generated datapoints to the HuggingFace Datasets format.

* `evaluate_synthetic_data_finetuning.py`: Script used to evaluate synthetic data in the NER Fine-tuning setting (see Section 3.2 of the paper).

* `evaluate_synthetic_data_fromscratch.py`: Script used to evaluate synthetic data in the NER From Scratch setting (see Section 3.2 of the paper).

* `generate_plots.R`: R script used to generate results plots.

* `gpt-4_expansion_api_batching.py`: Script used to submit OpenAI batch API requests for data generation. 

* `gpt-4_expansion_api_retrieval.py`: Script used to retrieve processed OpenAI batch API requests (for data generation), and write them to raw and cleaned `.jsonl` files. 

* `openmodel_expansion_vllm.py`: Script used to generate synthetic data from open-source models using the vLLM library.
