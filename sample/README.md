# 报告展示图片

这里只保存报告实际展示的四张原始图片和八张科学图，供 GitHub 直接渲染。完整图像数据集留在 `CV_data/`。

## 真实输入示例

### 自然图像问答（VQAv2）

![自然图像问答（VQAv2）](natural_vqav2_COCO_val2014_000000388829.jpg)

### 自然图像问答（VQAv2） / number

![自然图像问答（VQAv2） / number](natural_vqav2_COCO_val2014_000000343606.jpg)

### 自然图像问答（VQAv2） / other

![自然图像问答（VQAv2） / other](natural_vqav2_COCO_val2014_000000356949.jpg)

### 医学图像问答（VQA-RAD）

![医学图像问答（VQA-RAD）](medical_vqa_rad_synpic29265.jpg)

四张图片复用于九任务示例及五个图像答案类别，未按模型分数挑选；输入、参考及回答见 [完整实例](../summary_cv/examples.md)。

## 过程图能回答什么

[学习收益](../summary_cv/learning_gain.md) 量化学前→学后收益和保留；[任务全过程](../summary_cv/process.md) 定位各任务与领域的变化；[外部知识](../summary_cv/knowledge_probe.md) 观察固定知识题的准确率变化；[T5 过程](../summary/learning_process.md) 展示语言主线。

图中的变化是固定测试集表现，不能直接换算为内部知识总量。

## 文件来源与复现

[图片清单与 SHA256](image_manifest.json) 保留原路径及实例；展示图片按字节复制，没有压缩、替换或修改医学图像。

[图表清单与数据哈希](figure_manifest.json) 标出 300 DPI PNG 与 [矢量 PDF](../summary_cv/figures/)；原始精确矩阵和重算脚本随报告保存。

绘图依赖与独立环境：

```bash
python -m venv --system-site-packages .plot-env
.plot-env/bin/python -m pip install -r summary_cv/plot_requirements.txt
.plot-env/bin/python summary_cv/plot_process.py
```
