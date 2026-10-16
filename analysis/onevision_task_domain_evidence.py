"""Read-only CPU snapshot of completed task/domain intervention cases.

Report all finished cases, not partial stage scores. Default numerical backend
is retained as a limitation; do not infer method superiority from tiny gaps.
"""
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from onevision_knowledge_intervention import case_complete
from onevision_research_worker import digest
from onevision_task_domain_intervention import METHODS,SEQUENCES
from run import freeze


def main():
    manifest,data,domains,tests,generation,probes,protocol=freeze()
    report=ROOT/'summary/onevision_diagnostics_20261016'
    info=json.loads((ROOT/'summary/onevision_mechanisms_20261016/task_domain_attempt.json').read_text())
    run=ROOT/info['run'];anchor=[r for r in tests['medical'] if r['_task']==4]
    assert len(anchor)==23
    rows=[];sources={};weights={};pending=[]
    for sequence in SEQUENCES:
        for method in METHODS:
            case=sequence+'__'+method;proof=run/'cases'/(case+'_complete.json')
            if not proof.exists():pending.append(case);continue
            marker=json.loads(proof.read_text());assert marker['complete'] and marker['source_sha256']==info['spec']['source_sha256']
            stages={}
            for stage in [1,3,5]:
                label=f'{case}_after{stage}'
                assert case_complete(run,label,{'anchor':anchor})
                path=run/'evaluations'/label/'anchor_image.jsonl';sources[str(path.relative_to(ROOT))]=digest(path)
                qa=[json.loads(line) for line in path.read_text().splitlines()]
                assert all('scores' in r for r in qa)
                stages[stage]={'count':len(qa),'mean_nll':sum(r['nll'] for r in qa)/len(qa),
                    'correct':sum(r['scores']['exact_match']==100 for r in qa),
                    'token_f1':sum(r['scores']['token_f1'] for r in qa)/len(qa)}
            path=run/'adapters'/case/'stage_01.pt'
            meta=json.loads(path.with_suffix('.json').read_text());assert digest(path)==meta['sha256']
            weights[case]=torch.load(path,map_location='cpu',weights_only=True)['adapter']
            sources[str(path.relative_to(ROOT))]=meta['sha256']
            rows.append({'sequence':sequence,'method':method,'stages':stages})
    reference='repeat_anchor__SeqFT_LoRA';deltas={}
    if reference in weights:
        base=weights[reference]
        for name,a in weights.items():
            deltas[name]=max(float((a[k][b]-base[k][b]).abs().max()) for k in base for b in ['A','B'])
    result={'run':info['run'],'code_sha256':digest(Path(__file__)),
            'complete_cases_at_snapshot':len(rows),'planned_cases':len(SEQUENCES)*len(METHODS),
            'pending_or_live_cases':pending,'rows':rows,'source_sha256':sources,
            'same_anchor_stage1_max_parameter_difference_vs_reference':deltas,
            'limitations':['Single23-question medical Organ anchor; diagnostic, not full test.',
                           'Correct counts are normalized exact-reference matches, not expert semantic/clinical judgments.',
                           'Default numerical backend; same seed did not give identical stage1 weights across cases.',
                           'Same first stage has no replay/Fisher/history/projection; small case gaps are not algorithmic proof.',
                           'Repeat sequence reuses same8TRAIN QA for5 stages;40 target exposures, not40 distinct QA.',
                           'ER/Fisher/gradient-reference use extra historical data/compute, recorded separately.',
                           'No cross-domain attribution until corresponding cases complete; no ranking by interim results.']}
    (report/'task_domain_interim.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 任务数量与领域数量：已完成对照快照','',f'已完成{len(rows)}/{result["planned_cases"]}个完整序列／方法组合。仅纳入已完成标记及校验通过的阶段1/3/5，未完成组合不列成绩。', '',
           '| 序列 | 机制 | S1 NLL | S3 NLL | S5 NLL | S1正确/23 | S3正确/23 | S5正确/23 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        s=row['stages'];lines.append('| '+row['sequence']+' | '+row['method']+' | '+
             ' | '.join(f"{s[i]['mean_nll']:.3f}" for i in [1,3,5])+' | '+
             ' | '.join(f"{s[i]['correct']}/23" for i in [1,3,5])+' |')
    lines+=['','S1对所有序列／方法均是同一个Organ锚任务、相同8条QA和4次更新。其目标函数／初始状态没有历史回放或Fisher／投影项；默认计算下已观察到阶段1权重差异。细小方法差距不能直接归因于机制。原生顺序／独立任务的固定后端配对重跑等待现有队列完成；本4×6机制对照自身的确定性复核仍待补齐。','','重复锚序列增加优化次数但不增加独立训练QA；若同域新任务也改善锚性能，说明“任务一多必然遗忘”不成立于该前缀，但仍需跨域控制、更多锚／顺序和稳定数值设置。', '', '限制：','']
    lines+=['- '+item for item in result['limitations']]
    (report/'TASK_DOMAIN_INTERIM.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete_cases':len(rows),'planned_cases':result['planned_cases'],
                      'max_same_stage1_parameter_gap':max(deltas.values()) if deltas else None}))


if __name__=='__main__':main()
