"""CPU observer: accept only complete, SHA-verified matched-control cases.

Never changes training, evaluation, queue policy, or original experiment files.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import json
import math
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
import torch
from answer_f1 import answer_f1,VERSION

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/matched_task_domain'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    tmp=path.with_suffix(path.suffix+'.observer.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def paired(values,images):
    groups=defaultdict(list)
    for v,image in zip(values,images):groups[tuple(image)].append(v)
    assert len(values)==len(images) and values and all(math.isfinite(v) for v in values)
    clusters=list(groups.values());rng=random.Random(42);boot=[]
    for _ in range(2000):
        chosen=[clusters[rng.randrange(len(clusters))] for _ in clusters]
        boot.append(sum(map(sum,chosen))/sum(map(len,chosen)))
    boot.sort()
    return {'mean':sum(values)/len(values),'image_cluster_95CI':[boot[49],boot[1949]],
            'QA':len(values),'image_clusters':len(clusters),'scope':'conditional on fitted models; uncorrected QA cluster interval, not seed/order uncertainty'}


def checked_rows(path,expected,original):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert [r['id'] for r in rows]==expected, 'Incomplete, duplicate, or reordered fixed cohort'
    assert len(set(expected))==len(expected)
    for row in rows:
        assert row['references']==original[row['id']]['answers'], 'Changed original gold'
        assert math.isfinite(row['nll']) and 'prediction' in row and 'scores' in row
        assert answer_f1(row['prediction'],row['references'])==row['scores']['token_f1'], 'Raw F1 mismatch'
    return rows


def same_initial(a,b):
    assert [(r['id'],r['prediction'],r['nll'],r['scores']['token_f1']) for r in a]==[
        (r['id'],r['prediction'],r['nll'],r['scores']['token_f1']) for r in b], 'Same stage1 parameters but unequal stage1 outputs; no comparisons accepted'


def transition(a,b,original):
    assert [r['id'] for r in a]==[r['id'] for r in b]
    values=[y['scores']['token_f1']-x['scores']['token_f1'] for x,y in zip(a,b)]
    counts={'increased':sum(v>0 for v in values),'decreased':sum(v<0 for v in values),'unchanged':sum(v==0 for v in values)}
    return {'QA':len(a),'Answer_F1_before':sum(r['scores']['token_f1'] for r in a)/len(a),
            'Answer_F1_after':sum(r['scores']['token_f1'] for r in b)/len(b),'quality_changes':counts,
            'delta_Answer_F1':paired(values,[original[x['id']]['images'] for x in a]),
            'delta_NLL':paired([y['nll']-x['nll'] for x,y in zip(a,b)],[original[x['id']]['images'] for x in a])}


def compare(a0,a1,b0,b1,original):
    same_initial(a0,b0)
    assert [r['id'] for r in a1]==[r['id'] for r in b1]==[r['id'] for r in a0]
    images=[original[r['id']]['images'] for r in a0]
    out={'Answer_F1':paired([t1['scores']['token_f1']-t0['scores']['token_f1']-c1['scores']['token_f1']+c0['scores']['token_f1']
                        for t0,t1,c0,c1 in zip(a0,a1,b0,b1)],images)
         }
    out['NLL']=paired([t1['nll']-t0['nll']-c1['nll']+c0['nll'] for t0,t1,c0,c1 in zip(a0,a1,b0,b1)],images)
    return out


def checkpoint(path):
    side=json.loads(path.with_suffix('.json').read_text())
    assert sha(path)==side['sha256']
    return torch.load(path,map_location='cpu',weights_only=True)


def verify_adapter(a,b):
    assert a.keys()==b.keys()
    for name in a:
        for f in ['A','B']:
            assert torch.isfinite(a[name][f]).all() and torch.equal(a[name][f],b[name][f]), 'Stage1 canonical equality failed'


def build():
    assert not torch.cuda.is_initialized()
    metric_lock=json.loads((ROOT/'summary/primary_metric_20261017/metric_lock.json').read_text())
    assert metric_lock['version']==VERSION and metric_lock['metric_sha256']==sha(ROOT/'analysis/answer_f1.py')
    assert metric_lock['option_mapping'] is False
    plan=json.loads((REPORT/'CPU_PLAN_CHECK.json').read_text())
    assert plan['passed'] and plan['GPU_used'] is False
    spec=plan['spec'];sources={str((REPORT/'CPU_PLAN_CHECK.json').relative_to(ROOT)):sha(REPORT/'CPU_PLAN_CHECK.json')}
    for name,digest in spec['source_sha256'].items():assert sha(ROOT/name)==digest
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path)==spec['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());original={}
    for task in manifest['tasks']:
        f=task['files']['test'];path=ROOT/f['path'];assert sha(path)==f['sha256'];sources[f['path']]=f['sha256']
        original.update({r['id']:r for r in map(json.loads,path.read_text().splitlines())})
    accepted={};tables=[];canonical={};all_unchanged={}
    for anchor,sequence,method in spec['cases']:
        name=f'T{anchor:02d}_{sequence}__{method}';proof_path=REPORT/(name+'_complete.json')
        if not proof_path.exists():continue
        proof=json.loads(proof_path.read_text());assert proof['complete'] and proof['spec']==spec and proof['stage1_exact'] and not proof['full21454']
        run=ROOT/proof['run'];assert run.resolve().is_relative_to((ROOT/'exp/onevision_matched_task_domain_20261017').resolve())
        for artifact,digest in proof['artifact_sha256'].items():
            path=run/artifact;assert path.resolve().is_relative_to(run.resolve()) and sha(path)==digest, artifact
            all_unchanged[str(path.relative_to(ROOT))]=digest
        sources[str(proof_path.relative_to(ROOT))]=sha(proof_path)
        gate=json.loads((run/'stage1_exact_gate.json').read_text());assert gate['passed'] and gate['no_tolerance']
        canon=REPORT/f'canonical_anchor{anchor}.pt';assert sha(canon)==gate['canonical_sha256']
        reference=checkpoint(canon);assert reference['source_sha256']==spec['source_sha256']
        verify_adapter(checkpoint(run/'adapters/stage_01.pt')['adapter'],reference['adapter'])
        ids=spec['anchor_probe_ids'][str(anchor)]
        endpoints={stage:checked_rows(run/f'evaluations/anchor_after{stage}/anchor_image.jsonl',ids,original) for stage in [1,3,5]}
        if anchor in canonical:same_initial(endpoints[1],canonical[anchor])
        else:canonical[anchor]=endpoints[1]
        plasticity={};train=[];states=[];events_resources=[json.loads(line) for line in (run/'resources.jsonl').read_text().splitlines()]
        fields=['VmRSS_bytes','Pss_bytes','cpu_uss_bytes','gpu_allocated_bytes','gpu_reserved_bytes','gpu_process_nvml_bytes']
        memory=[]
        for stage,task in enumerate(spec['sequences'][str(anchor)][sequence],1):
            p=run/f'training/{name}/stage_{stage:02d}.json';t=json.loads(p.read_text());events=t['events']
            assert len(events)==4 and all(e['lr_used']>0 and math.isfinite(e['loss']) and math.isfinite(e['gradient_l2_preclip']) for e in events)
            assert [ident for e in events for ident in e['train_ids']]==spec['TRAIN_ids'][str(task)]
            state=checkpoint(run/f'adapters/stage_{stage:02d}.pt')['adapter']
            changed=None if not states else any(not torch.equal(state[k][f],states[-1][k][f]) for k in state for f in ['A','B'])
            assert changed is not False, 'No parameter update at later trained stage'
            states.append(state);train.append({'stage':stage,'task':task,'events':events,'new_plus_replay_exposures':t['new_plus_replay_exposures'],
                                              'fisher_extra_QA':t['fisher_extra_qa_next'],'parameters_changed_vs_previous_stage':changed})
            before=[v for v in events_resources if v.get('phase')==f'{name}_stage{stage}_before']
            after=[v for v in events_resources if v.get('phase')==f'{name}_stage{stage}_after']
            assert len(before)==len(after)==1
            a,b=before[0],after[0];assert a['unix']<=b['unix']
            samples=[v for v in events_resources if a['unix']<=v['unix']<=b['unix']]
            memory.append({'stage':stage,'before':{k:a.get(k) for k in fields},'after':{k:b.get(k) for k in fields},
                           'growth':{k:b[k]-a[k] if a.get(k) is not None and b.get(k) is not None else None for k in fields},
                           'sampled_CPU_peak':{k:max((v[k] for v in samples if v.get(k) is not None),default=None) for k in fields[:3]},
                           'CUDA_allocator_peak_training_window':{k:b.get(k) for k in ['gpu_peak_allocated_bytes','gpu_peak_reserved_bytes']},
                           'note':'CPU/NVML sampled; CUDA allocator peak reset before stage. VmHWM is not a stage peak. Fisher and whole-case windows remain separately in original artifacts.'})
            if stage>1:
                plasticity[stage]=checked_rows(run/f'evaluations/plasticity_after{stage}/current_image.jsonl',spec['plasticity_ids'][str(task)],original)
        current={'Answer_F1':sum(r['scores']['token_f1'] for rows in plasticity.values() for r in rows)/sum(map(len,plasticity.values()))}
        entry={'case':name,'anchor':anchor,'sequence':sequence,'method':method,'run':proof['run'],
               'anchor_stage1_to3':transition(endpoints[1],endpoints[3],original),
               'anchor_stage1_to5':transition(endpoints[1],endpoints[5],original),
               'current_task_scores_after_stages2_to5':current,
               'current_scope':'32 stage-observations, possibly repeated IDs; not32 independent QA or old-task retention.',
               'training':train,'TRAIN_memory_bytes':memory,
               'whole_case_memory':json.loads((run/'memory_summary.json').read_text())}
        full=None
        if method=='SeqFT_LoRA':
            full=checked_rows(run/'evaluations/final_native_anchor/anchor_image.jsonl',spec['final_SeqFT_full_anchor_ids'][str(anchor)],original)
            entry['full_native_anchor_final']={'Answer_F1':sum(r['scores']['token_f1'] for r in full)/len(full)}
            entry['full_native_anchor_final']['QA']=len(full)
        accepted[anchor,sequence,method]=(endpoints,plasticity,full);tables.append(entry)
    comparisons=[]
    sequence_pairs=[('within_medical','repeat_anchor'),('cross_short','within_medical'),('cross_five_domains','within_medical'),('cross_five_domains','cross_short')]
    for anchor in spec['anchors']:
        for method in spec['methods']:
            for treatment,control in sequence_pairs:
                ka,kb=(anchor,treatment,method),(anchor,control,method)
                if ka not in accepted or kb not in accepted:continue
                a,b=accepted[ka][0],accepted[kb][0]
                item={'anchor':anchor,'method':method,'treatment':treatment,'control':control,
                      'anchor_difference_in_changes':compare(a[1],a[5],b[1],b[5],original)}
                if method=='SeqFT_LoRA':
                    af,bf=accepted[ka][2],accepted[kb][2]
                    item['full_native_anchor_final_only_difference']=transition(bf,af,original)
                    item['full_scope']='Full anchor final-only paired scores; no measured full stage1/FWT/BWT.'
                comparisons.append(item)
    method_pairs=[]
    for anchor in spec['anchors']:
        for sequence in spec['sequences'][str(anchor)]:
            for method in spec['methods']:
                if method=='SeqFT_LoRA':continue
                ka,kb=(anchor,sequence,method),(anchor,sequence,'SeqFT_LoRA')
                if ka not in accepted or kb not in accepted:continue
                a,b=accepted[ka],accepted[kb]
                method_pairs.append({'anchor':anchor,'sequence':sequence,'method':method,'control':'SeqFT_LoRA',
                                     'anchor_difference_in_changes':compare(a[0][1],a[0][5],b[0][1],b[0][5],original),
                                     'current_task_difference':{'Answer_F1':sum(x['scores']['token_f1']-y['scores']['token_f1']
                                        for stage in [2,3,4,5] for x,y in zip(a[1][stage],b[1][stage]))/32
                                        },
                                     'current_scope':'Same sequence and current IDs, stage-observation mean; extra ER/Fisher/reference cost recorded, not equal total compute.'})
    for name,digest in all_unchanged.items():assert sha(ROOT/name)==digest
    result={'complete_cases':len(tables),'planned_cases':48,'GPU_study_complete':len(tables)==48,
            'case_results':tables,'sequence_contrasts':comparisons,'method_vs_SeqFT':method_pairs,
            'primary_metric':'Answer F1','metric_version':VERSION,'metric_sha256':sha(ROOT/'analysis/answer_f1.py'),
            'sources':sources,'verified_artifact_count':len(all_unchanged),'reporter_sha256':sha(Path(__file__)),
            'limits':['Zero accepted cases means pending, not zero effect. No incomplete-run inference.',
                      'Two medical anchors, one fixed order/seed and32QA probes; no universal pure domain-count causal claim.',
                      'ER, Fisher and anchor reference use extra history/compute; interpret old and new scores together.',
                      'QA bootstrap intervals conditional on trained weights, uncorrected; cannot replace training-seed/order replication.',
                      'Full native-anchor evaluation only final SeqFT507/228, not all methods/full21454/FWT/BWT.',
                      'Original scores only; previous auxiliary format parsers do not replace primary values.']}
    write(REPORT/'VERIFIED_RESULTS.json',result)
    lines=['# 双锚确定性机制复核：自动核验结果','',f'已接受完整轨迹 **{len(tables)}/48**。仅接受完整SHA、共同stage1参数和输出精确一致、原ID/金标/评分一致的结果。未完成不填0。','',
           '| 锚 | 序列 | 机制 | 旧锚Answer F1 S1→S5 | 质量下降/上升题数 | 当前任务Answer F1 | 旧锚NLL变化 |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for row in tables:
        t=row['anchor_stage1_to5'];c=t['quality_changes']
        lines.append(f"| T{row['anchor']:02d} | {row['sequence']} | {row['method']} | {t['Answer_F1_before']:.3f}→{t['Answer_F1_after']:.3f} | {c['decreased']}/{c['increased']} | {row['current_task_scores_after_stages2_to5']['Answer_F1']:.3f} | {t['delta_NLL']['mean']:+.4f} |")
    lines+=['','同域/跨域差异、方法对SeqFT的配对差异、当前任务学习、实际训练loss/LR/梯度、额外信息预算及每阶段训练前后/峰值内存见[JSON](VERIFIED_RESULTS.json)。没有GPU结果时不得称该报告已证明领域原因。','']+['- '+v for v in result['limits']]
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    assert not torch.cuda.is_initialized()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2);previous=None
    if args.watch:
        import fcntl
        observer_lock=(REPORT/'observer.lock').open('a')
        fcntl.flock(observer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        proofs=sorted((p.name,p.stat().st_mtime_ns) for p in REPORT.glob('T*_complete.json'))
        signature=hashlib.sha256(json.dumps(proofs).encode()).hexdigest()
        if signature!=previous:
            result=build();previous=signature
            write(REPORT/'observer_status.json',{'phase':'complete' if result['GPU_study_complete'] else 'waiting_completed_GPU_cases',
                  'pid':os.getpid(),'accepted_cases':result['complete_cases'],'no_GPU_allocation':True})
            print(json.dumps({'verified_cases':result['complete_cases'],'planned':48,'CPU_only':True}),flush=True)
            if result['GPU_study_complete']:return
        if not args.watch:return
        time.sleep(45)


if __name__=='__main__':
    try:main()
    except Exception:
        write(REPORT/'observer_needs_attention.json',{'traceback':traceback.format_exc(),'pid':os.getpid(),'CPU_only':True,
                                                   'original_GPU_worker_not_stopped':True});raise
