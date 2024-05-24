# Large language models streamline automated machine learning for clinical studies


Overview
------

* This is the official repository of the paper [**Large language models streamline automated machine learning for clinical studies**](https://doi.org/10.1038/s41467-024-45879-8).
* Pre-print version: [https://arxiv.org/abs/2308.14120](https://arxiv.org/abs/2308.14120)

Abstract
------
A knowledge gap persists between machine learning (ML) developers (e.g., data scientists) and practitioners (e.g., clinicians), hampering the full utilization of ML for clinical data analysis. We investigated the potential of the ChatGPT Advanced Data Analysis (ADA), an extension of GPT-4, to bridge this gap and perform ML analyses efficiently. Real-world clinical datasets and study details from large trials across various medical specialties were presented to ChatGPT ADA without specific guidance. ChatGPT ADA autonomously developed state-of-the-art ML models based on the original study’s training data to predict clinical outcomes such as cancer development, cancer progression, disease complications, or biomarkers such as pathogenic gene sequences. Following the re-implementation and optimization of the published models, the head-to-head comparison of the ChatGPT ADA-crafted ML models and their respective manually crafted counterparts revealed no significant differences in traditional performance metrics. Strikingly, the ChatGPT ADA-crafted ML models often outperformed their counterparts. In conclusion, ChatGPT ADA offers a promising avenue to democratize ML in medicine by simplifying complex data analyses, yet should enhance, not replace, specialized training and resources, to promote broader applications in medical research and practice.


### Prerequisites

The software is developed in **Python 3.9**.



Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate llmmed
```

**Note:** This might take a few minutes.


Code structure
---

Our source code for training and evaluation of the deep neural networks, image analysis and preprocessing, and data augmentation are available here.

1. *./main_LLMmed.py*: includes all the training codes.
2. *./statistics_LLMmed.py*: includes all the statistical analysis and evaluation metrics.
3. *./utils.py* and *./shapp_LLMmed*: include all the illustrations.


------
### In case you use this repository, please cite the original paper:

S. Tayebi Arasteh, T. Han, M. Lotfinia, et al. *Large language models streamline automated machine learning for clinical studies*. Nature Communications, 15(1), 1603 (2024), https://doi.org/10.1038/s41467-024-45879-8.

### BibTex

    @article {gptada_arasteh,
      author = {Tayebi Arasteh, Soroosh and Han, Tianyu and Lotfinia, Mahshad and Kuhl, Christiane and Kather, Jakob Nikolas and Truhn, Daniel and Nebelung, Sven},
      title = {Large language models streamline automated machine learning for clinical studies},
      year = {2024},
      volume = {15},
      number = {1603},
      doi = {10.1038/s41467-024-45879-8},
      publisher = {Nature Portfolio},
      journal = {Nature Communications}
    }
