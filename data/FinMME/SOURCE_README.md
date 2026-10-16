---
license: mit
dataset_info:
  features:
  - name: id
    dtype: int32
  - name: image
    dtype: image
  - name: question_text
    dtype: string
  - name: question_type
    dtype: string
  - name: options
    dtype: string
  - name: answer
    dtype: string
  - name: unit
    dtype: string
  - name: tolerance
    dtype: float32
  - name: verified_caption
    dtype: string
  - name: related_sentences
    dtype: string
  splits:
  - name: train
    num_bytes: 419829046.637
    num_examples: 11099
  download_size: 398554212
  dataset_size: 419829046.637
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
task_categories:
- multiple-choice
- question-answering
language:
- en
tags:
- finance
- multimodal
- reasoning
pretty_name: FinMME
size_categories:
- 10K<n<100K
---



Multimodal Large Language Models (MLLMs) have experienced rapid development in recent years. However, there is a notable lack of effective and specialized multimodal evaluation datasets in the financial domain. To advance the development of MLLMs in the finance domain, we introduce FinMME, encompassing more than 11,000 high-quality financial research samples across 18 financial domains and 6 asset classes, featuring 10 major chart types and 21 subtypes. We ensure data quality through 20 annotators and carefully designed validation mechanisms. Additionally, we develop FinScore, an evaluation system incorporating hallucination penalties and multi-dimensional capability assessment to provide an unbiased evaluation.


## Usage

Please refer to https://github.com/luo-junyu/FinMME for the evaluation protocol.

## Citation

Paper Link: https://arxiv.org/abs/2505.24714

If you find our work helpful, please consider citing our work:

```BibTex
@inproceedings{finmme,
  title={FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation},
  author={Junyu Luo and Zhizhuo Kou and Liming Yang and Xiao Luo and Jinsheng Huang and Zhiping Xiao and Jingshu Peng and Chengzhong Liu and Jiaming Ji and Xuanzhe Liu and Sirui Han and Ming Zhang and Yike Guo},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year={2025}
}
```
