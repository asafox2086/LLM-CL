# GeoChat_Instruct 来源和实际使用证据

完整总表：[DATASET_PROVENANCE.md](../DATASET_PROVENANCE.md)。核验日期：2026-10-12。

提出者：Kartik Kuckreja、Muhammad Sohail Danish、Muzammal Naseer、Abhijit Das、Salman Khan、Fahad Shahbaz Khan。原始论文：[Kartik Kuckreja 等：GeoChat: Grounded Large Vision-Language Model for Remote Sensing](https://github.com/mbzuai-oryx/GeoChat)（CVPR 2024）。

官方资源：https://github.com/mbzuai-oryx/GeoChat

划分：当前不构建第五域任务；如以后使用须采用论文明确的任务协议。

限制与版本：原始混合集含分类和定位，不能整包算开放 QA；本次 pinned JSON 实际 308,861 条与 318k brochure 不同。

后续实际使用证据：

- [Jeremy Irvin 等：TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html)（ICLR 2025）。用途：multimodal instruction training。定位：Section 3.1.1 and training-data appendix。证据概述：Explicitly incorporates GeoChat instructions in training. [依据](https://proceedings.iclr.cc/paper_files/paper/2025/file/ac3af725ae398b6184faae0828bdbd6c-Paper-Conference.pdf)
