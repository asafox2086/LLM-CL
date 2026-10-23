"""One necessary tiny-HF CPU admission; CUDA hidden before all Torch imports."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import json,time,traceback
from core import ROOT,PORT,REPORT,study,sha,write,cpu_gate,RANK,EXTRA,FINAL,TRAINABLE,CASES
from common import data

def main():
    assert not (REPORT/'READY.json').exists() and not (REPORT/'CPU_GATE.json').exists()
    assert not (REPORT/'CPU_GATE_FAILURE.json').exists(),'Retain a failed admission; no blind repeated gate'
    review=json.loads((REPORT/'SOURCE_REVIEW.json').read_text());assert review['decision']=='accept'
    for n,h in review['files_sha256'].items():assert sha(ROOT/n)==h,n
    old=json.loads((study.REPORT/'READY.json').read_text())
    for n,h in old['sources'].items():assert sha(ROOT/n)==h,n
    path,m,split,fit,dev,mapped,plan,groups=data()
    cfg=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json';text=json.loads(cfg.read_text())['text_config']
    assert (text['hidden_size'],text['intermediate_size'],text['num_hidden_layers'])==(896,4864,24)
    best_path=ROOT/'summary/edge_deployment_20261019/position_coordinate_control_20261022/v1/VERIFIED.json'
    best=json.loads(best_path.read_text());assert best['complete'] and best['quality']['domain_macro']==43.79198740853657
    import torch
    assert not torch.cuda.is_initialized()
    start=time.monotonic();gate=cpu_gate();gate['seconds']=time.monotonic()-start
    assert gate['passed'] and not torch.cuda.is_initialized()
    write(REPORT/'CPU_GATE.json',gate)
    write(REPORT/'INTERVENTION.json',dict(rank=RANK,temporary_rank=EXTRA,FFN_matrices=72,
        final_coefficients=FINAL,temporary_training_coordinates=TRAINABLE,temporary_total_coefficients=FINAL+TRAINABLE,
        compression='data-free SVD, no recovery; shared untrained init',teacher=False,TEST_QA=0,
        cases=list(CASES),scale=4,dropout=.1,learning_rate=1e-4,
        current_QA_limit=2,current_payload_limit=16*1024**2,bank_group_limit=8,bank_payload_limit=32*1024**2,
        auxiliary_answer_QA_upper_bound=570,auxiliary_backward_QA_upper_bound=190,
        worker_budget_seconds=7200,aggregate_model_budget_seconds=14400,automatic_full_confirmation=False))
    sources=dict(old['sources'])
    sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')})
    for n in ['analysis/osft_increment_paired_analysis_20261022.py','analysis/onevision_research_device_entry.py']:
        sources[n]=sha(ROOT/n)
    base_files=dict(old['base_files'])
    for p in cfg.parent.iterdir():
        if p.is_file() and p.suffix in ('.json','.txt','.model'):base_files[p.relative_to(ROOT).as_posix()]=sha(p)
    evidence=[REPORT/n for n in ['PROTOCOL.md','NECESSITY_REVIEW.json','SOURCE_REVIEW.json','INTERVENTION.json','CPU_GATE.json']]
    evidence.extend([study.REPORT/'READY.json',study.REPORT/'SPLIT.json',cfg,best_path])
    write(REPORT/'READY.json',dict(sources=sources,evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
        base_files=base_files,manifest_sha256=sha(path),split_sha256=sha(study.REPORT/'SPLIT.json'),
        training_ast_sha256=study.training_ast()[1],training_plan=plan,rank=RANK,temporary_rank=EXTRA,
        worker_budget_seconds=7200,aggregate_model_budget_seconds=14400,cases=list(CASES),
        reference_best_macro=best['quality']['domain_macro'],reference_proof_sha256=sha(best_path),
        FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,scientific_sources_FROZEN=True,
        automatic_full_confirmation=False))
    print(json.dumps(dict(ready_sha256=sha(REPORT/'READY.json'),CPU_gate_seconds=gate['seconds'],actual_model_not_started=True)))
if __name__=='__main__':
    try:main()
    except Exception:write(REPORT/'CPU_GATE_FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
