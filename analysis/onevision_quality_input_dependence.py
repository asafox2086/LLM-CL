"""TRAIN-only frozen-head input controls; no fit, generation or model load."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import onevision_functional_router_core as core

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'summary/onevision_diagnostics_20261016/functional_router'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    tick=time.monotonic()
    p=BASE/'pilot_complete.json'
    proof=json.loads(p.read_text())
    assert proof['complete'] and proof['scope']=='pilot' and proof['LoRA_optimizer_updates']==0
    run=ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_functional_router_20261018/pilot').resolve())
    cache,summary=run/'calibration.pt',run/'calibration_summary.json'
    assert sha(cache)==proof['artifact_sha256']['calibration.pt']
    assert sha(summary)==proof['artifact_sha256']['calibration_summary.json']
    saved=torch.load(cache,map_location='cpu',weights_only=True)
    assert saved['spec']==proof['spec'] and not saved['TEST_used']
    for name,h in saved['spec']['source_sha256'].items():
        assert sha(ROOT/name)==h,name
    original=BASE/'calibration_audit/EVIDENCE.json'
    old=json.loads(original.read_text())
    assert old['CPU_only'] and not old['TEST_opened'] and old['exact_reproduced_head_and_gradient_and_loss']
    assert old['calibration_sha256']==sha(cache) and old['summary_sha256']==sha(summary)
    cos,nll,head=saved['cosines'],saved['TRAIN_NLL'],saved['weight']
    rows=old['paired_TRAIN_rows']
    assert [r['id'] for r in rows]==saved['TRAIN_ids']
    assert cos.shape==(43,3) and nll.shape==(43,4) and head.shape==(4,4)
    assert torch.equal(nll,torch.tensor([r['four_actual_NLL'] for r in rows]))
    reference=[r['fit_class'] for r in rows]
    calls=0
    def choose(inputs):
        nonlocal calls
        labels,probs=[],[]
        for value in inputs:
            index,_,prob=core.choose('quality_hard',value,head,0.)
            labels.append(0 if index is None else index+1)
            probs.append(prob)
            calls+=1
        return labels,torch.tensor(probs)
    aligned,prob=choose(cos)
    assert aligned==reference
    def stats(labels,probabilities):
        selected=nll[torch.arange(43),torch.tensor(labels)]
        return {'QA':43,'choices_base_e0_e1_e2':[labels.count(j) for j in range(4)],
                'changed_hard_calls_vs_aligned':sum(a!=b for a,b in zip(labels,aligned)),
                'mean_actual_TRAIN_NLL':float(selected.mean()),
                'mean_NLL_delta_vs_aligned':float((selected-nll[torch.arange(43),torch.tensor(aligned)]).mean()),
                'max_probability_delta_vs_aligned':float((probabilities-prob).abs().max()),
                'single_hard_choice_for_all':len(set(labels))==1}
    controls={'aligned':stats(aligned,prob)}
    for name,inputs in [('bias_only_zero_cosines',torch.zeros_like(cos)),
                        ('TRAIN_mean_cosines',cos.mean(0).expand_as(cos)),
                        ('reverse_TRAIN_input_rows',cos.flip(0))]:
        labels,probs=choose(inputs)
        controls[name]=stats(labels,probs)
    cyclic=[]
    for offset in range(43):
        labels,probs=choose(torch.roll(cos,shifts=offset,dims=0))
        row=stats(labels,probs)
        cyclic.append({'offset':offset,**row})
        assert row['choices_base_e0_e1_e2']==controls['aligned']['choices_base_e0_e1_e2']
    assert cyclic[0]=={'offset':0,**controls['aligned']}
    means=[r['mean_actual_TRAIN_NLL'] for r in cyclic]
    original_classes=[r['original_class'] for r in rows]
    changes={}
    for a,b in zip(original_classes,aligned):
        key=f'{a}->{b}';changes[key]=changes.get(key,0)+1
    target=BASE/'input_dependence_20261018/EVIDENCE.json'
    assert not target.exists(), 'Preserve previous analyses'
    result={'CPU_only':True,'new_GPU_operations':0,'TEST_opened':False,'model_loaded':False,
      'new_generation':0,'new_optimizer_updates':0,'new_LM_forwards':0,'new_head_fits':0,
      'frozen_TRAIN_QA':43,'head_choices_evaluated':calls,'controls':controls,
      'all_cyclic_controls':cyclic,'mean_over_all_cyclic_TRAIN_NLL':sum(means)/len(means),
      'fixed_candidate_mean_TRAIN_NLL_base_e0_e1_e2':nll.mean(0).tolist(),
      'original_geometry_to_quality_transitions':changes,'analysis_seconds':time.monotonic()-tick,
      'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                       'analysis/onevision_functional_router_core.py':sha(ROOT/'analysis/onevision_functional_router_core.py')},
      'parent_sha256':{str(p.relative_to(ROOT)):sha(p),str(cache.relative_to(ROOT)):sha(cache),
                       str(summary.relative_to(ROOT)):sha(summary),str(original.relative_to(ROOT)):sha(original)},
      'limits':['All controls are frozen-head TRAIN routing counterfactuals, not new generated F1.',
                'Every cyclic offset is reported; no TEST or best shift selects a deployed router.',
                'Cosine-row permutations alter input correspondence; teacher-forced candidate NLL remains fixed.',
                'A quality gain can overlap cold-start repair; two mechanisms need the queued four-cell ablation.',
                'No expert deletion, head replacement, GPU source edit or queue modification follows this audit.']}
    assert not torch.cuda.is_initialized()
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ['controls','mean_over_all_cyclic_TRAIN_NLL','original_geometry_to_quality_transitions','head_choices_evaluated','analysis_seconds']}))


if __name__=='__main__':
    main()
