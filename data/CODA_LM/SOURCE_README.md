---
license: mit
configs:
- config_name: Chinese
  data_files:
  - split: Mini
    path: Chinese/Mini-*
  - split: Test
    path: Chinese/Test-*
  - split: Val
    path: Chinese/Val-*
  - split: Train
    path: Chinese/Train-*
- config_name: English
  data_files:
  - split: Mini
    path: English/Mini-*
  - split: Test
    path: English/Test-*
  - split: Val
    path: English/Val-*
  - split: Train
    path: English/Train-*
dataset_info:
- config_name: Chinese
  features:
  - name: id
    dtype: string
  - name: image
    dtype: image
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  splits:
  - name: Mini
    num_bytes: 150295510
    num_examples: 193
  - name: Test
    num_bytes: 1686467830.01
    num_examples: 2123
  - name: Val
    num_bytes: 14194833543.85
    num_examples: 18455
  - name: Train
    num_bytes: 16251864387.65
    num_examples: 20495
  download_size: 32254046770
  dataset_size: 32283461271.510002
- config_name: English
  features:
  - name: id
    dtype: string
  - name: image
    dtype: image
  - name: conversations
    list:
    - name: from
      dtype: string
    - name: value
      dtype: string
  splits:
  - name: Mini
    num_bytes: 150333933
    num_examples: 193
  - name: Test
    num_bytes: 1686807094.01
    num_examples: 2123
  - name: Val
    num_bytes: 14198216149.85
    num_examples: 18455
  - name: Train
    num_bytes: 16255087877.65
    num_examples: 20495
  download_size: 32255463905
  dataset_size: 32290445054.510002
task_categories:
- image-to-text
language:
- en
- zh
tags:
- autonomous_driving
- corner_cases
---

# CODA-LM Dataset Card

[CODA-LM](https://coda-dataset.github.io/coda-lm/) is the multi-modal version of the [CODA](https://coda-dataset.github.io/) dataset, used in the [CODA-LM](https://arxiv.org/abs/2404.10595) paper. Both English and Chinese annotations are available. Check detailed usage in our [Github repo](https://github.com/DLUT-LYZ/CODA-LM).

This repo contains the CODA-LM dataset, which has been reorganized in the LLaVA data format.
You are also welcome to check the [original CODA-LM](https://huggingface.co/datasets/KaiChen1998/coda-lm) data which contains more metadata vanilla annotations.

## Usage

```python
from datasets import load_dataset

# name can be selected from ['English', 'Chinese']
# split can be selected from ['Mini', 'Train', 'Val', 'Test']
dataset = load_dataset("KaiChen1998/coda-lm-llava-format", name="English", split='Train')

# should be a dictionary containing
# {"id": sample identification, 'image': PIL Image, 'conversations': with <image> token}
for data in dataset:
    print(data)
```

## Citation

```bibtex
@article{li2024automated,
  title={Automated Evaluation of Large Vision-Language Models on Self-driving Corner Cases},
  author={Li, Yanze and Zhang, Wenhua and Chen, Kai and Liu, Yanxin and Li, Pengxiang and Gao, Ruiyuan and Hong, Lanqing and Tian, Meng and Zhao, Xinhai and Li, Zhenguo and others},
  journal={arXiv preprint arXiv:2404.10595},
  year={2024}
}
```
