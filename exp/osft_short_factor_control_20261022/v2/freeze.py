"""One necessary CPU admission; freeze this independent single-condition study."""
import json
import time
from pathlib import Path
from core import ROOT,REPORT,study,sha,write,cpu_gate
from common import data

def main():
    assert not (REPORT/'READY.json').exists()
    review=json.loads((REPORT/'REVIEW.json').read_text());assert review['decision']=='accept'
    for n,h in review['files_sha256'].items():assert sha(ROOT/n)==h,n
    old=json.loads((study.REPORT/'READY.json').read_text())
    for n,h in old['sources'].items():assert sha(ROOT/n)==h,n
    path,m,_,split,_,_,_,plan=data()
    config=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json'
    t=json.loads(config.read_text())['text_config']
    assert (t['hidden_size'],t['intermediate_size'],t['num_hidden_layers'],t['num_attention_heads'],t['num_key_value_heads'])==(896,4864,24,14,2)
    shape=ROOT/'summary/edge_deployment_20261019/update_geometry_literature_20261022/PROTECTED_BLOCK_COUNTS.json'
    rows=[]
    for name,x in json.loads(shape.read_text())['by_position'].items():
        m,n,h,r,L=x['m'],x['n'],x['preserved_rank'],x['trainable_rank'],x['layers']
        assert h+r==min(m,n)
        rows.append(dict(position=name,layers=L,frozen='V_low' if m>=n else 'U_low',
            original=L*r*(m+n+1),remaining=L*r*(max(m,n)+1),frozen_short=L*r*min(m,n),
            short_constraint_normals=L*h*r,short_basis_gauge=L*r*r,S_low=L*r))
    totals={k:sum(r[k] for r in rows) for k in ('original','remaining','frozen_short','short_constraint_normals','short_basis_gauge','S_low')}
    assert totals==dict(original=113766144,remaining=89485056,frozen_short=24281088,
        short_constraint_normals=18210816,short_basis_gauge=6070272,S_low=28416)
    write(REPORT/'COUNTS.json',dict(by_position=rows,totals=totals,shape_source_sha256=sha(shape),
        original_USV_permanent_storage_unchanged=True,static_coordinates_not_inference_or_runtime_peak=True))
    write(REPORT/'INTERVENTION.json',dict(short_side_frozen=True,policy='m>=n freeze V_low; m<n freeze U_low; reapply after each original reSVD; S_low remains trainable',
        parameter_projection='wide skip_u=True; tall/square original V projection on detached temporary clone',
        trainable_coordinates=89485056,target_modules=168,unfreeze_rank_ratio=.25,learning_rate=1e-5,
        unchanged='Original seven-target construction/forward/author kernels/HostAdamW/offload/stagewise re-SVD',
        scope='Same single-stage ideal allowed set, not matched optimizer trajectory/step magnitude or fresh controlled time comparison'))
    begin=time.monotonic();gate=cpu_gate();gate['seconds']=time.monotonic()-begin
    write(REPORT/'CPU_GATE.json',gate)
    sources=dict(old['sources']);sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')})
    for n in ['analysis/osft_increment_paired_analysis_20261022.py','analysis/onevision_research_device_entry.py']:
        sources[n]=sha(ROOT/n)
    evidence=[REPORT/n for n in ['PROOF.md','PROTOCOL.md','REPAIR_SCOPE.md','V1_FAILURE_SEAL.json','REVIEW.json','COUNTS.json','INTERVENTION.json','CPU_GATE.json']]
    evidence+=[study.REPORT/n for n in ['READY.json','SPLIT.json','SCREEN_COMPLETE.json','OSFT_all_COMPLETE.json','LoRA_qv_COMPLETE.json']]
    evidence.extend([shape,config])
    for name in ['OSFT_all','LoRA_qv']:
        v=json.loads((study.REPORT/(name+'_COMPLETE.json')).read_text())
        evidence.extend(ROOT/n for n in v['prediction_sha256'])
    write(REPORT/'READY.json',dict(sources=sources,evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
        base_files=old['base_files'],manifest_sha256=sha(path),split_sha256=sha(study.REPORT/'SPLIT.json'),
        training_ast_sha256=study.training_ast()[1],training_plan=plan,short_side_frozen=True,
        trainable_coordinates=89485056,FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,
        one_new_condition=True,automatic_full_confirmation=False,scientific_sources_FROZEN=True))
    print(json.dumps(dict(ready_sha256=sha(REPORT/'READY.json'),CPU_gate_passed=True,actual_model_not_started=True)))
if __name__=='__main__':main()
