"""Read sealed initialization-source controls; report all five, not a winner."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from answer_f1 import answer_f1

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    spec=read(REPORT/'CPU_PLAN_CHECK.json')['spec']
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    proofs={p.name:sha(p) for p in REPORT.glob('*_complete.json')}
    dest=REPORT/'VERIFIED_EVIDENCE.json'
    if dest.exists() and read(dest)['proof_sha256']==proofs:return
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path)==spec['manifest_sha256'];task=read(manifest_path)['tasks'][3]
    p=ROOT/task['files']['test']['path'];assert sha(p)==task['files']['test']['sha256']
    gold=[json.loads(s) for s in p.read_text().splitlines()];byid={r['id']:r for r in gold}
    results=[];scores={}
    for name in spec['cases']:
        pointer=REPORT/(name+'_complete.json')
        if not pointer.exists():continue
        proof=read(pointer);assert proof['complete'] and proof['spec']==spec and proof['target_full_TEST_QA']==507
        run=ROOT/proof['run'];assert run.resolve().is_relative_to((ROOT/'exp/onevision_initialization_source_20261018/v1').resolve())
        for rel,s in proof['artifact_sha256'].items():
            p=run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==s
        assert read(run/'initial_function_gate.json')['passed']
        events=[json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
        updates=[e for e in events if e.get('event')=='train' and e.get('label')=='target_T04']
        assert len(updates)==2 and all(e['learning_rate_used']>0 and math.isfinite(e['loss']) and e['gradient_l2_preclip']>0 for e in updates)
        scopes={}
        for label,ids in [('initial_probe',spec['initial_probe_ids']),('target_full507',[r['id'] for r in gold])]:
            rr=[json.loads(s) for s in (run/f'evaluations/{label}/medical_image.jsonl').read_text().splitlines()]
            assert [r['id'] for r in rr]==ids
            for r in rr:
                assert r['references']==byid[r['id']]['answers'] and math.isfinite(r['nll'])
                assert answer_f1(r['prediction'],r['references'])==r['scores']['token_f1']
            scopes[label]=dict(QA=len(rr),F1=sum(r['scores']['token_f1'] for r in rr)/len(rr),NLL=sum(r['nll'] for r in rr)/len(rr))
            if label=='target_full507':scores[name]={r['id']:r['scores']['token_f1'] for r in rr}
        results.append(dict(condition=name,**scopes,target_updates=2,run=proof['run']))
    contrasts=[]
    if 'cold' in scores:
        for name in spec['cases'][1:]:
            if name not in scores:continue
            blocks=defaultdict(list)
            for r in gold:blocks[tuple(r['images'])].append(scores[name][r['id']]-scores['cold'][r['id']])
            groups=list(blocks.values());rng=random.Random(42);draws=[]
            for _ in range(10000):
                chosen=[groups[rng.randrange(len(groups))] for _ in groups]
                draws.append(sum(map(sum,chosen))/sum(map(len,chosen)))
            draws.sort();contrasts.append(dict(condition=name,reference='cold',
                delta_F1=sum(sum(x) for x in groups)/507,image_95CI=[draws[249],draws[9749]],uncorrected=True))
    out=dict(CPU_only=True,proof_sha256=proofs,rows=results,contrasts=contrasts,
        all5_required=True,TEST_selection=False,report_source_sha256=sha(Path(__file__)),
        limits=['One target/two fixed source tasks/one seed; content and task type still differ.',
                'Initial32 and final507 are different cohorts: never subtract their means as acquisition.',
                'Full inheritance changes both initial function and first-step A access; not pure knowledge cause.',
                'A-only drops old output but tests learned basis; geometric familiarity is not mastery.',
                'Extra source12QA/2updates charged; independent controls are not formal19-stage CL metrics.'])
    dest.write_text(json.dumps(out,indent=2)+'\n')
    text='# 从零、完整继承、只继承A：同26TRAIN／507最终测试\n\n| 条件 | 初始32 F1 | 最终507 F1 | 最终507 NLL | 相对冷启动F1 |\n|---|---:|---:|---:|---:|\n'
    for name in spec['cases']:
        row=next((r for r in results if r['condition']==name),None);c=next((c for c in contrasts if c['condition']==name),None)
        values=[f"{row['initial_probe']['F1']:.3f}",f"{row['target_full507']['F1']:.3f}",f"{row['target_full507']['NLL']:.4f}"] if row else ['—']*3
        effect=f"{c['delta_F1']:+.3f}" if c else '0（参考）' if name=='cold' and row else '—'
        text+='| '+name+' | '+' | '.join(values+[effect])+' |\n'
    text+='\n所有五格保留；源历史每组6条TRAIN、重复两轮/2更新，目标26条/2更新相同。不强迫跨域复用，不预设复用有益。初始32不作全507初始分或验证调参；A-only真实基座函数门槛、原始梯度、实际A/B更新、CPU/GPU纯TRAIN窗口及checkpoint见各run。当前不是新方法的大幅提升或正式CL主表。\n'
    (REPORT/'VERIFIED_RESULTS.md').write_text(text)


if __name__=='__main__':main()
