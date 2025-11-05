# Does Synthetic Data Help Named Entity Recognition for Low-Resource Languages?

*Gaurav Kamath \& Sowmya Vajjala*

This repository contains code and data for the paper "Does Synthetic Data Help Named Entity Recognition for Low-Resource Languages?", to be presented at [AACL 2025 ](https://2025.aaclnet.org) (Main Conference). This work was conducted as part of an internship Gaurav Kamath did at the National Research Council, Canada.

## Contents

Below is the structure of this repository:

* `scripts`: Scripts used for all steps of this study, including both data generation and model training.

* `datasets`: Datasets used for this study. This folder includes synthetic, LLM-generated datasets, as well as the raw outputs from models when generating them. This folder also contains portions of the [Universal NER](https://www.universalner.org) dataset (Mayhew et al. 2023), which was not available on HuggingFace Datasets at the time this study began. The [MasakhaNER 2](https://huggingface.co/datasets/masakhane/masakhaner2) (Adelani et al. 2022) and [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) (Mhaske et al. 2023) datasets can be found on HuggingFace Datasets.

* `results`: All results files from our experiments.

### References
Adelani, D. I., Neubig, G., Ruder, S., Rijhwani, S., Beukman, M., Palen-Michel, C., ... & Klakow, D. (2022, December). [MasakhaNER 2.0: Africa-centric Transfer Learning for Named Entity Recognition.](https://aclanthology.org/2022.emnlp-main.298/) In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 4488-4508).

Mayhew, S., Blevins, T., Liu, S., Šuppa, M., Gonen, H., Imperial, J. M., ... & Pinter, Y. (2024, June). [Universal NER: A Gold-Standard Multilingual Named Entity Recognition Benchmark.](https://aclanthology.org/2024.naacl-long.243/) In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)* (pp. 4322-4337).

Mhaske, A., Kedia, H., Doddapaneni, S., Khapra, M. M., Kumar, P., Murthy, R., & Kunchukuttan, A. (2023, July). [Naamapadam: A Large-Scale Named Entity Annotated Data for Indic Languages.](https://aclanthology.org/2023.acl-long.582/) In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 10441-10456).
