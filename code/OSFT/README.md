2026-10-18现状：OneVision正式V2运行已完整核验19阶段/74格/21454QA，AP46.357；旧获取阶段状态以下保留为历史。实际使用固定25%剩余奇异秩作者核心、七类解码投影、113766144可训练参数、初LR1e-5；没有执行论文按层输入输出重要性自适应分配。较SeqFT AP+6.071伴随习得+9.283/BWT更低3.391，不能单独归因保护，详见[六栏及完整证据](../../summary/onevision_diagnostics_20261016/osft_acquisition_20261018/RESEARCH_FINDINGS.md)。原冻结方法代码/源码快照/manifest不变，尚未实现或准入OSFT轻量研究版本。

# OSFT

论文：[ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html)；分类：regularization。

作者仓库：https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer

核验依据：paper appendix A.13 author implementation URL

状态：作者源码已下载；待适配。固定版本：`fd5b552177bde9a551d30e73660d6a96205c97c0`。

备注：原作者模型和任务需适配；未运行新实验。

`source/` 保存作者源码，未安装或执行上游代码；源文件和许可证哈希见 manifest。 尚未完成 Qwen2-VL/十任务适配和 smoke test。
