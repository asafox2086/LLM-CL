"""Original Answer F1 from sealed formal-cache artifacts; CPU/stdlib only."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from answer_f1 import answer_f1

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016/formal_v2_state_cache_20261018/v1'
DOMAINS=['natural','medical','geography','finance','traffic']
VIEWS=['seq_geometry','seq_quality','cache_geometry','cache_quality']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    spec=read(REPORT/'CPU_INPUT_CHECK.json')['spec']
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path)==spec['manifest_sha256'];manifest=read(manifest_path)
    found={p.name:sha(p) for p in REPORT.glob('*_complete.json')}
    destination=REPORT/'VERIFIED_EVIDENCE.json'
    if destination.exists() and read(destination)['proof_sha256']==found:return
    rows=[]
    for name,value in sorted(found.items()):
        proof=read(REPORT/name);assert proof['complete'] and proof['spec']==spec
        run=ROOT/proof['run'];assert run.resolve().is_relative_to((ROOT/'exp/onevision_formal_state_cache_20261018/v1').resolve())
        for rel,expected in proof['artifact_sha256'].items():
            p=run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==expected
        if proof['phase']=='build':
            assert read(run/'original_exact_gate.json')['original_formal_SeqFT_exact']
            mapping=read(run/'view_labels.json')
        else:
            assert proof['full_QA']==21454 and read(run/'candidate_reload_exact.json')['passed']
            mapping={proof['view']:201}
        for view,label in mapping.items():
            task_means=[];domain_values=defaultdict(list);count=0
            for task in manifest['tasks']:
                t=task['task'];p=run/f'evaluations/stage_{label:02d}/T{t:02d}_test_predictions.jsonl'
                rr=[json.loads(s) for s in p.read_text().splitlines()]
                goldpath=ROOT/task['files']['test']['path'];assert sha(goldpath)==task['files']['test']['sha256']
                gold=[json.loads(s) for s in goldpath.read_text().splitlines()]
                if proof['phase']=='build':gold=[r for r in gold if r['id'] in set(spec['generation_ids'][task['domain']])]
                assert [r['id'] for r in rr]==[r['id'] for r in gold]
                scores=[]
                for r,g in zip(rr,gold):
                    assert r['references']==g['answers'];f=answer_f1(r['prediction'],r['references'])
                    assert f==r['scores']['token_f1'];scores.append(f)
                task_means.append(sum(scores)/len(scores));domain_values[task['domain']]+=scores;count+=len(scores)
            assert count==(155 if proof['phase']=='build' else 21454)
            domains={d:sum(domain_values[d])/len(domain_values[d]) for d in DOMAINS}
            rows.append(dict(scope='diagnostic155' if proof['phase']=='build' else 'full21454',view=view,
                domains=domains,AP=sum(task_means)/19,domain_macro=sum(domains.values())/5,FWT=None,BWT=None))
    result=dict(CPU_only=True,formal_common_input_FP16=True,proof_sha256=found,rows=rows,
        no_TEST_selection=True,report_source_sha256=sha(Path(__file__)),
        nativeAnyRes_results_not_mixed=True,full_FWT_BWT_require_separate_sparse_cells=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    text='# 公共V2输入/FP16：连续习得与功能缓存\n\n'
    for scope in ['diagnostic155','full21454']:
        text+='## '+scope+'\n\n| 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP | FWT | BWT |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
        for v in VIEWS if scope=='diagnostic155' else VIEWS[2:]:
            r=next((r for r in rows if r['scope']==scope and r['view']==v),None)
            values=[f"{r['domains'][d]:.3f}" for d in DOMAINS]+[f"{r['AP']:.3f}"] if r else ['—']*6
            text+='| '+v+' | '+' | '.join(values+['—','—'])+' |\n'
        text+='\n'
    text+='主训练精确复用正式SeqFT已封存的19阶段，原93次更新成本仍计入；不重新训练或复用不兼容预测。全部视图/两个完整候选保留。最终单点评测不能补FWT/BWT；CPU测试不是模型收益，内存/额外计算须按原训练与本阶段构建/推理分别计费。\n'
    (REPORT/'VERIFIED_RESULTS.md').write_text(text)


if __name__=='__main__':main()
