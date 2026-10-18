"""Report all fixed feedback conditions from complete SHA-bound predictions."""
import hashlib,json,math
from pathlib import Path
from answer_f1 import answer_f1

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1'
SOURCE=ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1'
VIEWS=['cache_AnswerF1','cache_AnswerF1_metric']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(proof,family):
    assert proof['complete'];run=ROOT/proof['run'];assert run.resolve().is_relative_to(family.resolve())
    for rel,s in proof['artifact_sha256'].items():
        p=run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p)==s
    return run


def complete(phase,view,spec):
    p=REPORT/(('build' if phase=='build' else 'full_'+view)+'_complete.json')
    if not p.exists():return False
    proof=read(p);assert proof['spec']==spec
    run=verify(proof,ROOT/'exp/onevision_answer_compatibility_20261018/v1')
    assert read(run/'original_cache_geometry_exact.json')['passed']
    if phase=='full':assert proof['full_QA']==21454 and read(run/'feedback_reload_exact.json')['passed']
    return True


def main():
    spec=read(REPORT/'CPU_PLAN_CHECK.json')['spec']
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    proofs={p.name:sha(p) for p in REPORT.glob('*_complete.json')};dest=REPORT/'VERIFIED_EVIDENCE.json'
    if dest.exists() and read(dest)['proof_sha256']==proofs:return
    sourceproof=read(SOURCE/'build_complete.json');assert sha(SOURCE/'build_complete.json')==spec['parent_complete_sha256']
    source=verify(sourceproof,ROOT/'exp/onevision_state_cache_20261018/v1')
    build=None
    if complete('build',None,spec):
        build=ROOT/read(REPORT/'build_complete.json')['run']
        event=read(build/'router_TRAIN.json')
        assert len(event['events'])==40 and len(event['TRAIN_ids'])==20 and event['teacher_generation_QA']==60
        assert all(math.isfinite(e['loss']) and math.isfinite(e['raw_gradient_l2']) and e['LR']==.03 for e in event['events'])
        assert event['no_TEST_input_to_fit'] and event['source93updates_charged']
    manifest=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest)==spec['parent_spec']['manifest_sha256'];tasks=read(manifest)['tasks'];gold={};train={};full_ids={};gold_tasks={}
    for task in tasks:
        for split,destination in [('test',gold),('train',train)]:
            f=task['files'][split];p=ROOT/f['path'];assert sha(p)==f['sha256']
            destination.update({r['id']:r for r in [json.loads(s) for s in p.read_text().splitlines()]})
            if split=='test':
                ids=[r['id'] for r in [json.loads(s) for s in p.read_text().splitlines()]]
                full_ids.setdefault(task['domain'],[]).extend(ids)
                gold_tasks.update({i:task['task'] for i in ids})
    def score(run,label,full=False):
        per_task={};per_domain={};count=0
        for domain,ids in spec['parent_spec']['generation_ids'].items():
            p=run/f'evaluations/{label}/{domain}_image.jsonl'
            rows=[json.loads(s) for s in p.read_text().splitlines()]
            expected=full_ids[domain] if full else [json.loads(s)['id'] for s in
                (source/f'evaluations/seq_geometry/{domain}_image.jsonl').read_text().splitlines()]
            assert [r['id'] for r in rows]==expected and all(math.isfinite(r['nll']) for r in rows)
            generated=[r for r in rows if r.get('prediction') is not None]
            if not full:assert {r['id'] for r in generated}==set(ids)
            vals=[]
            for r in generated:
                g=gold[r['id']];assert r['references']==g['answers']
                f1=answer_f1(r['prediction'],r['references']);assert f1==r['scores']['token_f1']
                assert r['native_task']==gold_tasks[r['id']]
                task=str(r['native_task']);per_task.setdefault(task,[]).append(f1);vals.append(f1);count+=1
            per_domain[domain]=sum(vals)/len(vals)
        assert count==(21454 if full else 155) and len(per_task)==19
        return dict(QA=count,domains=per_domain,AP19=sum(sum(v)/len(v) for v in per_task.values())/19,
            domain_macro=sum(per_domain.values())/5,full_FWT=None,full_BWT=None)
    conditions={}
    for view,label in [('seq','seq_geometry'),('cache_geometry','cache_geometry'),('cache_NLL','cache_quality')]:
        conditions[view]=dict(source='original sealed reference, not a new rerun',scores=score(source,label))
    if build:
        for view in VIEWS:conditions[view]=dict(source='new complete build',scores=score(build,view))
    teacher=None
    if build:
        teacher=[]
        for state in [12,18,19]:
            p=build/f'evaluations/TRAIN_teacher_state{state}/TRAIN_memory_image.jsonl'
            rows=[json.loads(s) for s in p.read_text().splitlines()]
            assert [r['id'] for r in rows]==event['TRAIN_ids']
            for r in rows:
                assert r['references']==train[r['id']]['answers']
                assert r['scores']['token_f1']==answer_f1(r['prediction'],r['references'])
            teacher.append(dict(state=state,QA=len(rows),mean_TRAIN_F1=sum(r['scores']['token_f1'] for r in rows)/20))
    full={}
    for view in VIEWS:
        if complete('full',view,spec):
            run=ROOT/read(REPORT/('full_'+view+'_complete.json'))['run']
            full[view]=score(run,'final_full_input_router',True)
    result=dict(CPU_only=True,proof_sha256=proofs,reporter_sha256=sha(Path(__file__)),
        diagnostic_conditions=conditions,teacher_TRAIN=teacher,full_conditions=full,
        limits=['155 diagnostic is not full21454 or formalV2.',
                'TRAIN head fit/image exclusion is supervised training, not independent validation.',
                'Source93updates/pool3 and all feedback cost charged; no blanket memory superiority.',
                'All conditions retained, no TEST choosing winner or adjusting settings.'])
    dest.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# TRAIN输出质量反馈：所有预定条件','',
        '| 条件 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP19（155诊断） |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for view in ['seq','cache_geometry','cache_NLL']+VIEWS:
        c=conditions.get(view)
        if c:
            s=c['scores'];values=[s['domains'][d] for d in ['natural','medical','geography','finance','traffic']]+[s['AP19']]
            lines.append('| '+view+' | '+' | '.join(f'{x:.3f}' for x in values)+' |')
        else:lines.append('| '+view+' | — | — | — | — | — | — |')
    lines+=['','前三行明确引用同协议原封存结果；新两行仅接收完整SHA证明。缺失为待测，不是零收益。TRAIN合成检查／拟合分数不代表模型提升；完整21454和资源核验仍必需。',
        '','[协议](PROTOCOL.md) · [完整数据与来源](VERIFIED_EVIDENCE.json)']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
