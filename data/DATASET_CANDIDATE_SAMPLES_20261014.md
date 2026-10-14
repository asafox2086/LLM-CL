# 新任务候选数据真实样例

这些是已下载的原始候选数据，尚未替换正式十任务配置。以下图像为原始文件复制，问答原样展示，不生成答案。

## GeoChat 遥感图像

```text
<image>
 How many airports are present in the image?
```

原始答案：
```text
There is only one airport present in the image.
```

![GeoChat 原始图像](../../sample/dataset_candidates_20261014/geochat.png)

## FinChart 金融图表

```text
What is the percentage increase in Net Bookings from Q3 '14 to Q4 '14?
```

原始答案：
```text
145
```

![FinChart 原始图像](../../sample/dataset_candidates_20261014/finchart.jpg)

原始 reasoning 字段只保存在标注中，不作为模型输入。

## GeoQuestions 地理文本

官方问题：
```text
What is the population of Central Greece?
```

官方答案文件对应记录（CSV 原样）：
```text
population
570922

```

原始答案文件还有几何和实体集合格式，文本任务子集待确定。
