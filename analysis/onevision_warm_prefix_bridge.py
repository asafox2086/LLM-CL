"""Check whether inherited experts reproduce actual sequential prefixes.

Only sealed small LoRA snapshots and existing predictions/NLL are read.
No model forward, optimization, routing choice, or score transplantation.
"""
import hashlib,json,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'summary/onevision_diagnostics_20261016'
REPORT=BASE/'warm_start_20261018/prefix_bridge'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def same(a,b):return a.keys()==b.keys() and all(torch.equal(a[k][l],b[k][l]) for k in a for l in ['A','B'])
def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    torch.set_num_threads(1)
    assert not (REPORT/'EVIDENCE.json').exists(),'Do not repeat a completed bridge'
    paths=[BASE/'warm_start_20261018/warm_inherit_K3_v1_train_complete.json',BASE/'deterministic_native/complete.json']
    proofs=[json.loads(p.read_text()) for p in paths]
    assert all(p['complete'] for p in proofs)
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    def artifact(which,rel):
        proof=proofs[which];p=ROOT/proof['run']/rel
        assert sha(p)==proof['artifact_sha256'][rel],rel
        sources[str(p.relative_to(ROOT))]=sha(p);return p
    comparisons=[]
    for stage in range(2,12):
        a=torch.load(artifact(0,f'candidates/reuse_T{stage:02d}.pt'),map_location='cpu',weights_only=True)
        b=torch.load(artifact(1,f'adapters/sequential_T{stage:02d}.pt'),map_location='cpu',weights_only=True)
        comparisons.append({'stage':stage,'expert':a['expert'],'all_A_B_bitwise_equal':same(a['adapter'],b['adapter'])})
    final=torch.load(artifact(0,'latest.pt'),map_location='cpu',weights_only=True)['pool']
    prefix2=torch.load(artifact(1,'adapters/sequential_T02.pt'),map_location='cpu',weights_only=True)['adapter']
    prefix11=torch.load(artifact(1,'adapters/sequential_T11.pt'),map_location='cpu',weights_only=True)['adapter']
    final_native=torch.load(artifact(1,'adapters/sequential_T19.pt'),map_location='cpu',weights_only=True)['adapter']
    bridges={'final_e1_exact_native_prefix2':same(final['e1'],prefix2),
             'final_e2_exact_native_prefix11':same(final['e2'],prefix11),
             'final_e0_exact_native_final19':same(final['e0'],final_native)}
    case=BASE/'warm_quality_20261018/predeclared_hard_case/VERIFIED_EVIDENCE.json'
    metadata=json.loads(case.read_text());sources[str(case.relative_to(ROOT))]=sha(case)
    attempt=json.loads((BASE/'warm_quality_20261018/pilot_attempt.json').read_text())
    p=ROOT/attempt['run']/'evaluations/quality_hard/medical_image.jsonl'
    assert sha(p)==metadata['input_sha256'][str(p.relative_to(ROOT))]
    sources[str(p.relative_to(ROOT))]=sha(p)
    later=[json.loads(s) for s in p.read_text().splitlines()]
    earlier=[json.loads(s) for s in artifact(1,'evaluations/sequential_T11/medical_image.jsonl').read_text().splitlines()]
    assert [r['id'] for r in earlier]==[r['id'] for r in later] and len(later)==100
    assert all(r['route_expert']=='e2' for r in later)
    overlap=[(a,b) for a,b in zip(earlier,later) if 'prediction' in a]
    comparison={'same_medical_diagnostic_QA':100,
                'all100_NLL_bitwise_equal':all(a['nll']==b['nll'] for a,b in zip(earlier,later)),
                'native_prefix_generated_overlap_QA':len(overlap),
                'overlap_predictions_and_original_scores_exact':all(a['prediction']==b['prediction'] and a['scores']==b['scores'] for a,b in overlap),
                'all100_references_exact':all(a['references']==b['references'] for a,b in zip(earlier,later))}
    result={'CPU_only':True,'GPU_operations':0,'new_optimizer_or_model_forward':0,
            'prefixes':comparisons,'final_expert_bridges':bridges,'existing_function_comparison':comparison,
            'input_sha256':sources,'source_sha256':sha(Path(__file__)),
            'limits':['Checkpoint equality does not prove all-domain professional knowledge sufficiency.',
                      'Actual prefix has only its native-task10 generation overlap, not all73 medical generations.',
                      'No old-prefix prediction is transplanted into a CL main table or full21454 confirmation.',
                      'The learned router, actual pool residency and full goal remain separately evaluated.']}
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['prefixes','final_expert_bridges','existing_function_comparison']}))
if __name__=='__main__':main()
