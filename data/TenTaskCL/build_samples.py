"""Publish the first frozen training example per current task; never modify source QA."""
import hashlib,html,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    policy=json.loads((ROOT/'exp/ten_task_cl_20261012/execution_policy.json').read_text())
    manifest=json.loads((ROOT/policy['manifest']).read_text())
    output=ROOT/'sample/ten_task_cl_20261012';output.mkdir(parents=True,exist_ok=True)
    examples=[]
    lines=['# 本轮十任务真实数据示例','',
      '每个任务展示固定训练清单中的第一条样本（index 0），按同一规则选取，不按答案内容或模型分数挑选。问题、上下文、答案和图像来自当前实际训练输入；示例不是模型预测。完整原始字段另存 [samples.json](samples.json)。原始数据文件保持不变。','']
    domains={'natural':'自然／通用','medical':'医学','medicine':'医学','law':'法学','legal':'法学','finance':'金融','art':'艺术文化','culture':'艺术文化','art_culture':'艺术文化'}
    for task in manifest['tasks']:
        info=task['files']['train'];path=ROOT/info['path']
        before=hashlib.sha256(path.read_bytes()).hexdigest();assert before==info['sha256']
        row=json.loads(path.read_text().splitlines()[0]);images=[]
        for index,name in enumerate(row.get('images',[])):
            source=ROOT/name
            assert source.is_file(),source
            target=output/f'T{task["task"]:02d}_image_{index+1}{source.suffix.lower()}'
            shutil.copy2(source,target)
            assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(target.read_bytes()).hexdigest()
            images.append(str(target.relative_to(ROOT)))
        example={'task':task['task'],'domain':task['domain'],'modality':task['modality'],'dataset':row.get('dataset'),
          'source_view':info['path'],'source_view_sha256':before,'selection_index':0,'sample_images':images,'raw_example':row}
        examples.append(example)
        lines+=['## T%02d'%task['task'],'',f'领域：{domains.get(task["domain"],task["domain"])}；模态：{task["modality"]}；数据集：{row.get("dataset","—")}；样本 ID：`{row["id"]}`。','',
          '**问题（原文）**','',html.escape(row['question']),'','**参考答案（原文）**','']
        answers=row.get('answers',[row['answer']])
        for answer in answers:lines+=['- '+html.escape(str(answer))]
        if row.get('context'):
            lines+=['','<details>','<summary>原始上下文（完整）</summary>','',html.escape(row['context']),'','</details>']
        for image in images:lines+=['',f'![T{task["task"]:02d} 原始输入图像](../../{image})']
        lines+=['']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    (ROOT/'data/TenTaskCL/samples.json').write_text(json.dumps({'selection_rule':'first example in each frozen training view, index 0','question_answer_annotations_unchanged':True,'examples':examples},ensure_ascii=False,indent=2)+'\n')
    (ROOT/'data/TenTaskCL/samples.md').write_text('\n'.join(lines))
    return examples
if __name__=='__main__':main()
