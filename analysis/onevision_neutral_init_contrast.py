"""Audit same source factors/budget, retained vs compensated source function.

Only new complete conditions trigger this independent audit. No model, GPU,
optimizer, new predictions, bootstrap tuning or selection of a TEST winner.
"""
import hashlib,json
from pathlib import Path
import torch
import onevision_function_neutral_init_report_v2 as verified
from onevision_function_neutral_gradients import gradient_values

ROOT=verified.ROOT
SOURCE=ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1'
REPORT=verified.REPORT/'function_vs_factor'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return torch.load(p,map_location='cpu',weights_only=True)
def same(a,b):return a.keys()==b.keys() and all(torch.equal(a[k][v],b[k][v]) for k in a for v in ['A','B'])


def main():
    torch.set_num_threads(1)
    verified.main();spec=read(verified.REPORT/'CPU_PLAN_CHECK.json')['spec']
    score=read(verified.REPORT/'VERIFIED_EVIDENCE.json')
    proof_set={p.name:sha(p) for p in verified.REPORT.glob('*_complete.json')}
    REPORT.mkdir(parents=True,exist_ok=True);destination=REPORT/'EVIDENCE.json'
    if destination.exists():
        previous=read(destination)
        if previous['proof_sha256']==proof_set and previous['analyzer_sha256']==sha(Path(__file__)):return
    results={}
    for index,name in [(1,'within'),(2,'cross')]:
        if not verified.complete(index,spec):continue
        original=read(SOURCE/(name+'_full_complete.json'))
        assert original['complete'] and original['spec']==spec['parent_spec']
        full=ROOT/original['run']
        assert full.resolve().is_relative_to((ROOT/'exp/onevision_initialization_source_20261018/v1').resolve())
        for rel,s in original['artifact_sha256'].items():
            p=full/rel;assert p.resolve().is_relative_to(full.resolve()) and sha(p)==s
        compensated_proof=read(verified.REPORT/(name+'_compensated_complete.json'))
        compensated=ROOT/compensated_proof['run']
        start_full=load(full/'initial_adapter.pt')['adapter']
        start_comp=load(compensated/'initial_adapter.pt')['adapter']
        reference=load(compensated/'reference_adapter.pt')['adapter']
        assert len(start_full)==48 and same(start_full,start_comp) and same(start_comp,reference)
        assert sum(p.numel() for pair in reference.values() for p in pair.values())==540672
        assert all(torch.count_nonzero(pair['B'])>0 for pair in reference.values())
        initial={};events={};gradients={}
        for condition,run in [('full',full),('compensated',compensated)]:
            rows=[json.loads(s) for s in (run/'evaluations/initial_probe/medical_image.jsonl').read_text().splitlines()]
            assert [r['id'] for r in rows]==spec['parent_spec']['initial_probe_ids']
            initial[condition]=dict(QA=32,mean_NLL=sum(r['nll'] for r in rows)/32,
                raw_file_sha256=sha(run/'evaluations/initial_probe/medical_image.jsonl'))
            ev=[r for r in [json.loads(s) for s in (run/'events.jsonl').read_text().splitlines()]
                if r.get('event')=='train' and r.get('label')=='target_T04']
            assert len(ev)==2
            events[condition]=[{k:r[k] for k in ['qa_seen','total_qa','learning_rate_used','warmup_updates']} for r in ev]
            gradients[condition]=[gradient_values(load(run/f'raw_update_gradients/target_T04/step{i}.pt')) for i in [1,2]]
            assert all(g['preclip_A_gradient_l2']>0 and g['preclip_B_gradient_l2']>0 for g in gradients[condition])
        assert events['full']==events['compensated']
        assert [g['TRAIN_ids'] for g in gradients['full']]==[g['TRAIN_ids'] for g in gradients['compensated']]
        contrast=next(c for c in score['all_available_pairwise_contrasts']
            if c['target']==name+'_compensated' and c['reference']==name+'_full')
        results[name]=dict(source_factor_values_exact=True,trainable_parameters=540672,
            frozen_reference_parameters_compensated_only=540672,
            target_TRAIN_QA=26,target_updates=2,source_history_QA_exposures=12,source_history_updates=2,
            actual_target_budgets=events['full'],initial32=initial,raw_preclip_gradients=gradients,
            final507={k:score['rows'][name+'_'+k] for k in ['full','compensated']},contrast=contrast,
            full_complete_sha256=sha(SOURCE/(name+'_full_complete.json')),
            compensated_complete_sha256=sha(verified.REPORT/(name+'_compensated_complete.json')))
    result=dict(CPU_only=True,analyzer_sha256=sha(Path(__file__)),proof_sha256=proof_set,
        scoring_audit_sha256=sha(verified.REPORT/'VERIFIED_EVIDENCE.json'),sources=results,
        pending=[n for n in ['within','cross'] if n not in results],new_model_updates=0,new_GPU_or_predictions=False,
        limits=['This exact compensation changes source function, activations, gradients and translated function family.',
                'Not a pure knowledge intervention or universal proof nonzero initialization is ineffective.',
                'Initial32 is not full507 acquisition reference; never subtract their means.',
                'Same source history charged to both; one target/two fixed sources/single seed.',
                'All reported paired image intervals are uncorrected, not multi-seed uncertainty.',
                'Larger raw A-gradient norm does not identify better Adam updates or causal semantic capacity.'])
    destination.write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 非零首步A梯度是否足以恢复完整继承收益？','',
        '**探究的问题：** 先前温启动收益是否仅因首步A变得可学习？','',
        '**做的实验：** 同一已封存源A/B，目标26TRAIN／2更新、源12曝光／2更新均相同；完整保留源增量，对比减去固定源增量。两边首步A/B都实际非零。补偿组初始32逐项精确等于cold，冷补偿全部507精确等价门槛已通过。','',
        '| 来源 | 完整继承507 F1 | 补偿507 F1 | 补偿−完整 | 图像95%区间 | 首步A梯度：完整／补偿 |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for name in ['within','cross']:
        if name not in results:lines.append('| '+name+' | — | — | — | — | — |');continue
        x=results[name];c=x['contrast'];lo,hi=c['image_95CI'];f=x['final507'];g=x['raw_preclip_gradients']
        lines.append(f"| {name} | {f['full']['F1']:.3f} | {f['compensated']['F1']:.3f} | {c['delta_F1']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {g['full'][0]['preclip_A_gradient_l2']:.3f}／{g['compensated'][0]['preclip_A_gradient_l2']:.3f} |")
    lines+=['','**结果／证明了什么：** 首步A确实可学习、源因子值相同，也不能直接推出完整继承的收益会保留。各固定源的差异及区间以表为准；不能把首步A为0当作唯一原因，或把较大梯度范数当作更好学习。此干预包含源功能／激活／优化路径变化，不等于已经分离“纯知识”。',
        '','**改进的方法：** 优先检验完整已学功能的连续复用及正确调用；不只复制微小变化的A或追求非零／更大梯度。新TRAIN-F1匹配原型保留原轨迹与功能状态。',
        '','**改进的效果：** 这只是原因解释的约束，尚无新五域方法收益；已完成来源与待测列表见表和JSON，不取消预定条件。新匹配试验已排队，两个新条件均保留完整确认。',
        '','[参数／实际预算／原始梯度／SHA](EVIDENCE.json) · [全部配对评分](../VERIFIED_EVIDENCE.json)']
    (REPORT/'RESEARCH_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(accepted=list(results),pending=result['pending'],no_new_GPU=True)))


if __name__=='__main__':main()
