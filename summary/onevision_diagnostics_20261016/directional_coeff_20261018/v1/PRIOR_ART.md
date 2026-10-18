# 借鉴来源与本次改造

- [PiSSA，NeurIPS2024](https://papers.nips.cc/paper_files/paper/2024/hash/db36f4d603cc9e3a2a5e10b93e6428f2-Abstract-Conference.html)：利用预训练权重的主奇异方向适配。借鉴“已有方向可直接利用”，本试验不是PiSSA复现。
- [LoRA-XS作者v3全文](https://arxiv.org/html/2405.17604)：作者声明ECAI2025接收，冻结SVD方向，只训练r×r矩阵。[作者代码](https://github.com/MohammadrezaBanaei/LoRA-XS)，已通过SSH取得，commit e50b1a82b5d7b7e87a0c46bc62e07e54f6648d31；utils/latent_utils.py及initialization_utils.py等五文件SHA见code/LoRA_XS/files_sha256.json。仅阅读，不运行／安装作者入口，不把整个source/入主Git。

我们独立实现图文持续版本：rank16/qv、R=0、scale1、dropout.1和统一LR1e-4；原作者R为std1e-5高斯，含其任务超参。本试验保留一个R跨19任务，不合并／重置，初始严格同native基座。随机方向以同一奇异缩放匹配参数预算，用于对照，非测试驱动选择。

公式及思想属于既有PEFT，不声称原创。当前目标是快速检验该已发表思路能否在小模型／1%多领域连续QA中提高F1并降低实际训练成本，而非先构造复杂模块或继续细分所有原因。
