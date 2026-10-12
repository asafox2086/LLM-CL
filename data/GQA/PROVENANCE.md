# GQA 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Drew A. Hudson、Christopher D. Manning。原始论文：[Drew A. Hudson 等：GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering](https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html)（CVPR 2019）。

官方资源：https://cs.stanford.edu/people/dorarad/gqa/

划分：优先公开官方划分；hidden test 不造标签；开放式任务处理需记录与论文的差异。

限制与版本：目前本地 JSONL 为开放式派生视图，不等同于论文全部任务或原始完整计数；下载镜像不是提出者。

后续实际使用证据：

- [Ziqi Wang 等：SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)（ICCV 2025）。用途：continual instruction tuning and evaluation。定位：Main paper: upstream continual learning section; results table 1。证据概述：TextVQA, GQA and VQAv2 are sequential upstream tasks with per-dataset results; includes additional classification tasks that we do not adopt. [依据](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html)
- [Ziqin Wang 等：CoPE: Continual Probe-guided Expansion for Large Vision-Language Models](https://proceedings.mlr.press/v306/wang26le.html)（ICML 2026）。用途：continual instruction tuning and evaluation。定位：Main paper: table 1 and task-order table 3。证据概述：CoIN results explicitly report TextVQA, GQA and VQAv2; not just references in the bibliography. [依据](https://proceedings.mlr.press/v306/wang26le.html)
