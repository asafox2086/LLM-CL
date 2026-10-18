"""Compare sealed cold/warm hard-routing decisions; never change a router.

TEST diagnostic files are read for route metadata only. No answer scoring,
head/threshold selection, model forward, or full-test reuse is performed.
"""
import hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'summary/onevision_diagnostics_20261016'
OUT=BASE/'warm_quality_20261018/calibration_audit/ROUTE_STATE_COMPARISON.json'
DOMAINS=['natural','medical','geography','finance','traffic']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'Do not repeat completed route-state comparison'
    cold_proof_path=BASE/'functional_router/pilot_complete.json'
    cold=json.loads(cold_proof_path.read_text());assert cold['complete'] and cold['LoRA_optimizer_updates']==0
    case_path=BASE/'warm_quality_20261018/predeclared_hard_case/VERIFIED_EVIDENCE.json'
    case=json.loads(case_path.read_text());assert case['condition_complete']
    attempt=BASE/'warm_quality_20261018/pilot_attempt.json'
    warm=json.loads(attempt.read_text());wrun=ROOT/warm['run'];crun=ROOT/cold['run']
    durable_path=wrun/'evaluations/quality_hard/durable_complete.json'
    durable=json.loads(durable_path.read_text());assert durable['complete']
    source={str(p.relative_to(ROOT)):sha(p) for p in [cold_proof_path,case_path,attempt,durable_path]}
    domains=[];all_changed=[];prob_changed=0;generated=0
    for domain in DOMAINS:
        rel=f'evaluations/quality_hard/{domain}_image.jsonl'
        a,b=crun/rel,wrun/rel
        assert sha(a)==cold['artifact_sha256'][rel]
        assert sha(b)==durable['files'][b.name]==case['input_sha256'][str(b.relative_to(ROOT))]
        source[str(a.relative_to(ROOT))]=sha(a);source[str(b.relative_to(ROOT))]=sha(b)
        left=[json.loads(s) for s in a.read_text().splitlines()];right=[json.loads(s) for s in b.read_text().splitlines()]
        assert len(left)==len(right)==100 and [r['id'] for r in left]==[r['id'] for r in right]
        changed=[];counts=[]
        for records in [left,right]:counts.append(dict(Counter(str(r['route_expert']) for r in records)))
        for x,y in zip(left,right):
            assert x['route_policy']==y['route_policy']=='quality_hard'
            assert x['route_residual_gain']==y['route_residual_gain']==1
            if x['route_expert']!=y['route_expert']:changed.append(x['id'])
            prob_changed+=x['route_quality_probabilities']!=y['route_quality_probabilities']
            generated+=int('prediction' in x)
        all_changed+=changed
        domains.append({'domain':domain,'QA':100,'changed_expert_count':len(changed),
                        'changed_ids':changed,'cold_route_counts':counts[0],'warm_route_counts':counts[1]})
    assert generated==155
    result={'CPU_only':True,'new_GPU_or_optimizer_operations':0,'scope':'Same500 diagnostic route decisions; not full21454',
            'TEST_prediction_files_read':True,'TEST_answers_or_scores_used_for_selection':False,
            'new_answer_F1_computation':False,'domains':domains,'changed_hard_routes':len(all_changed),
            'changed_quality_probabilities_rows':prob_changed,'hard_gain_always_one':True,
            'input_sha256':source,'source_sha256':sha(Path(__file__)),
            'limits':['Observed equal hard decisions do not establish equivalence for arbitrary/full21454 inputs.',
                      'No router is replaced, frozen or pruned based on this TEST comparison.',
                      'Changing expert functions and changing routing decisions are separate explanations.',
                      'All original auxiliary views, TRAIN controls and full candidates remain required.']}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['changed_hard_routes','changed_quality_probabilities_rows','domains']}))
if __name__=='__main__':main()
