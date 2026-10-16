"""Read-only CPU evidence from fully completed knowledge intervention cases."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_knowledge_intervention import case_complete
from onevision_research_worker import digest
from run import freeze


def main():
    manifest,data,domains,tests,generation,probes,protocol=freeze()
    report=ROOT/'summary/onevision_diagnostics_20261016';mp=ROOT/'summary/onevision_mechanisms_20261016'
    attempt=json.loads((mp/'knowledge_attempt.json').read_text());run=ROOT/attempt['run']
    selected={'medical':tests['medical'],'medical_KG':data[11,'test']}
    names=[f'context_{base}_{kind}' for base in ['base','shared'] for kind in ['none','correct','unrelated','permuted']]
    names += [f'residual_rank{rank}_{kind}' for rank in [1,2] for kind in ['correct','unrelated','permuted']]
    names += [f'continue_rank8_{kind}' for kind in ['correct','unrelated','permuted']]
    rows=[];sources={};completed=[]
    for case in names:
        if not (run/'evaluations'/case/'durable_complete.json').exists():continue
        assert case_complete(run,case,selected)
        completed.append(case)
        for group,expected in selected.items():
            path=run/'evaluations'/case/(group+'_image.jsonl');sources[str(path.relative_to(ROOT))]=digest(path)
            records=[json.loads(line) for line in path.read_text().splitlines()]
            assert [r['id'] for r in records]==[r['id'] for r in expected] and all('scores' in r for r in records)
            rows.append({'case':case,'group':group,'count':len(records),
                         'mean_nll':sum(r['nll'] for r in records)/len(records),
                         'EM_correct':sum(r['scores']['exact_match']==100 for r in records),
                         **{m:sum(r['scores'][m] for r in records)/len(records) for m in ['exact_match','token_f1','rougeL']}})
    overlap=len({r['id'] for r in selected['medical']}&{r['id'] for r in selected['medical_KG']})
    result={'run':attempt['run'],'code_sha256':digest(Path(__file__)),'completed_cases':completed,'planned_cases':17,
            'prompt':'predeclared both, shared across all knowledge cases; not original-prompt visual ablation',
            'rows':rows,'source_sha256':sources,'group_overlap_qa':overlap,
            'limitations':['Medical100 diagnostic and nativeKG267 overlap; never pool as independent367 QA.',
                           'Unrelated is low-relevance author anatomy knowledge, not nonmedical unrelated text.',
                           'Contexts differ in token lengths; no claim of strictly length-matched controls.',
                           'Lower teacher-forced NLL need not improve generated medical answers.',
                           'Only completed cases shown; no test-based prompt/capacity/hyperparameter selection.',
                           'Knowledge encoding sanity and target-test generalization are separate; tiny-LoRA efficacy not assumed.']}
    (report/'knowledge_interim.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 16条知识干预：已完成结果', '',f'已完成{len(completed)}/17个预定干预。医学100题与KG267题重叠{overlap}题，分开报告，不合并样本量。', '',
           '本组统一使用预先固定的both提示，未根据测试挑提示。无知识NLL11.169不能与原提示视觉消融的9.595混算；上下文比较只在本组同提示内进行。', '',
           '| 干预 | 组 | 题数 | NLL↓ | EM匹配数 | Token F1↑ | ROUGE-L↑ |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['group']} | {r['count']} | {r['mean_nll']:.3f} | {r['EM_correct']}/{r['count']} | {r['token_f1']:.3f} | {r['rougeL']:.3f} |")
    lines+=['','注意：相关与低相关解剖知识都可能降低NLL；只有正确知识相对负对照的增量，才有助于定位知识内容作用。'
            '低相关组仍属医学，不能据其同样有效就断言所有专业知识均不需要。KG生成分数与NLL可能反向变化，不能以loss下降代替生成性能证据。', '',
            '另有只读事实编码核验排在容量试验前：检查辅助16事实训练目标是否被记住及固定改写是否能回答。它不是额外医学测试，也不是独立事实泛化。', '', '限制：','']
    lines+=['- '+x for x in result['limitations']]
    (report/'KNOWLEDGE_INTERIM.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':len(completed),'planned':17,'overlap':overlap}))


if __name__=='__main__':main()
