This folder contains all results from our study.

*   `finetuning_synth_eval`: Contains language-wise results for the NER Fine-tuning setting (see Section 3.2 of the paper), in JSON format.

*   `fromscratch_synth_eval`: Contains language-wise results for the NER From Scratch setting (see Section 3.2 of the paper), in JSON format.

*   `wikiann_evaluations`: Contains language-wise results for training on WikiANN, in JSON format. Note that the WikiANN performance we report in the paper is based on these results, but the organic dataset performance we report is instead based on organic dataset performance from `finetuning_synth_eval` and `fromscratch_synth_eval` respectively.

*   `finetuning_synth_eval.csv`: Compiled results for the NER Fine-tuning setting.

*   `fromscratch_synth_eval.csv`: Compiled results for the NER From Scratch setting.

* `wikiann.csv`: Compiled results on WikiANN. Note that the organic dataset results we report in the paper is based on these results, but the organic dataset performance we report is instead based on organic dataset performance from `finetuning_synth_eval` and `fromscratch_synth_eval` respectively.