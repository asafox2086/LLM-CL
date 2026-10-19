# FOREVER独立v2保存修复

训练与优化器保持v1，只在保存前移除HF AMP封装，并实测参数及共同TRAIN NLL精确不变。源/协议已冻结，实际门槛在旧GPU0；尚未准入正式训练，不能填主表分数。详见[修复记录](../../summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v2/README.md)。不复用v1预检权重。
