"""Validate actual fixed QA inputs with the real local Qwen2-VL processor, CPU."""
import json
from transformers import AutoProcessor
from methods import ROOT
from run import input_batch


def main():
    manifest=json.loads((ROOT/'data/TenTaskCL/final_task_manifest.json').read_text())
    assert manifest['formal_ready']
    processor=AutoProcessor.from_pretrained(ROOT/'model/Qwen2-VL-2B-Instruct',local_files_only=True)
    results=[]
    for task in manifest['tasks']:
        rows=[json.loads(line) for line in (ROOT/task['files']['dev']['path']).open()]
        largest=max(rows,key=lambda row:len(row.get('context','')[:6000])+len(row['question'])+len(row['images'])*1000)
        for row in [rows[0],largest]:
            batch,details=input_batch(row,task,processor,'cpu')
            if row['images']:
                grid=batch['image_grid_thw']
                image_tokens=int(grid.prod(1).sum())//4
                assert int((batch['input_ids']==151655).sum())==image_tokens
            else:
                assert 'pixel_values' not in batch
                image_tokens=0
            results.append({'task':task['task_id'],'qa_id':row['id'],'image_count':len(row['images']),
                            'image_tokens':image_tokens,'prompt_tokens':details['prompt_tokens'],'passed':True})
    (ROOT/'exp/ten_task_cl_20261012/input_verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    print('PASS: real Qwen2-VL processor on 20 actual QA inputs from all ten tasks; image placeholders match grids')


if __name__=='__main__':
    main()
