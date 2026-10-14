"""Nested equal-budget subsets; raw examples/splits stay unchanged."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from methods import ROOT

COUNTS={'train':100,'dev':50,'test':50}
OUT=ROOT/'data/TenTaskCL/equal_budget_100_50_50_seed42'


def main():
    source=ROOT/'data/TenTaskCL/final_task_manifest.json'
    original=json.loads(source.read_text())
    assert original['formal_ready']
    OUT.mkdir(exist_ok=True)
    tasks=[]
    initial=OUT/'frozen_initial_evaluations'
    initial.mkdir(exist_ok=True)
    previous=ROOT/'exp/ten_task_cl_20261012/runs/SeqLoRA/seed_42/20261012_200725'
    if json.loads((previous/'config.json').read_text())['manifest_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError('Old frozen baseline used a different source manifest')
    proof=[]
    all_ids={split:set() for split in COUNTS}
    for task in original['tasks']:
        updated=dict(task,files={})
        for split,count in COUNTS.items():
            info=task['files'][split]
            path=ROOT/info['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
                raise ValueError('Source view changed')
            rows=[json.loads(line) for line in path.open()]
            source_counts=Counter(row['dataset'] for row in rows)
            quotas={name:math.floor(n*count/len(rows)) for name,n in source_counts.items()}
            remainder=count-sum(quotas.values())
            ordering=sorted(source_counts,key=lambda name:(-(source_counts[name]*count/len(rows)-quotas[name]),name))
            for name in ordering[:remainder]:
                quotas[name]+=1
            selected=[]
            for dataset,quota in sorted(quotas.items()):
                candidates=[row for row in rows if row['dataset']==dataset]
                candidates.sort(key=lambda row:hashlib.sha256(('42:'+row['id']).encode()).hexdigest())
                selected+=candidates[:quota]
            selected.sort(key=lambda row:hashlib.sha256(('42:'+row['id']).encode()).hexdigest())
            assert len(selected)==count
            keys={row['id'] for row in selected}
            if len(keys)!=count or all_ids[split]&keys:
                raise ValueError('QA IDs overlap between tasks')
            all_ids[split].update(keys)
            target=OUT/f'T{task["task"]:02d}_{split}.jsonl'
            target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
            updated['files'][split]={'path':str(target.relative_to(ROOT)),'count':count,
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'dataset_counts':quotas,
                'selected_qa_ids':[row['id'] for row in selected]}
            if split=='test':
                cached_path=previous/'evaluations/stage_00'/f'T{task["task"]:02d}_test_predictions.jsonl'
                cache={row['id']:row for row in map(json.loads,cached_path.read_text().splitlines())}
                records=[]
                for row in selected:
                    record=cache[row['id']]
                    assert record['references']==row['answers']
                    records.append(dict(record,dataset=row['dataset'],initial_evaluation_reused=True))
                cached_target=initial/f'T{task["task"]:02d}_test_predictions.jsonl'
                cached_target.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records))
                proof.append({'task':task['task'],'count':count,'source':str(cached_path.relative_to(ROOT)),
                    'source_sha256':hashlib.sha256(cached_path.read_bytes()).hexdigest(),
                    'cache_sha256':hashlib.sha256(cached_target.read_bytes()).hexdigest()})
        tasks.append(updated)
    assert not any(all_ids[a]&all_ids[b] for a,b in [('train','dev'),('train','test'),('dev','test')])
    manifest=dict(original,tasks=tasks,counts_per_task=COUNTS,seed=42,
        protocol='Nested deterministic source-stratified subset; official/source splits and original QA/images unchanged',
        parent_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        formal_ready=True,initial_evaluation_cache=str(initial.relative_to(ROOT)))
    target=OUT/'manifest.json'
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'initial_cache_proof.json').write_text(json.dumps({'source':'Untrained zero-adapter Qwen2-VL base only',
        'model_asset_manifest_sha256':hashlib.sha256((ROOT/'model/Qwen2-VL-2B-Instruct/asset_manifest.json').read_bytes()).hexdigest(),
        'trained_stage_results_are_not_reused':True,'verified_nested_source_rows_and_answer_references':True,'tasks':proof},indent=2)+'\n')
    policy={'manifest':str(target.relative_to(ROOT)),'counts_per_task':COUNTS,'seed':42,
        'save_checkpoints':False,'experiment_budget_id':'100_50_50_seed42','reuse_initial_base_predictions_only':True,
        'research_context':'Small models, small datasets, lightweight continual learning'}
    (ROOT/'exp/ten_task_cl_20261012/execution_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    print('READY: all 10 tasks 100/50/50; 500 verified untrained-base test predictions reusable; original QA unchanged')


if __name__=='__main__':
    main()
