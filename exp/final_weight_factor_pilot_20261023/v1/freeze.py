"""Necessary one-time tiny CPU admission before READY; no actual base loaded."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,time
from pathlib import Path
from core import ROOT,REPORT,PORT,study,sha,write,cpu_gate,FINAL,TRAINABLE
from common import data

def main():
    assert not (REPORT/'READY.json').exists()
    assert not (REPORT/'CPU_GATE.json').exists(),'CPU admission is unique; do not repeat after partial freeze'
    review=json.loads((REPORT/'SOURCE_REVIEW.json').read_text());assert review['decision']=='accept'
    for n,h in review['files_sha256'].items():assert sha(ROOT/n)==h,n
    old=json.loads((study.REPORT/'READY.json').read_text())
    for n,h in old['sources'].items():assert sha(ROOT/n)==h,n
    path,m,_,split,_,_,_,plan=data()
    cfg=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json'
    text=json.loads(cfg.read_text())['text_config']
    assert (text['hidden_size'],text['intermediate_size'],text['num_hidden_layers'])==(896,4864,24)
    import torch
    assert not torch.cuda.is_initialized()
    begin=time.monotonic();gate=cpu_gate();gate['seconds']=time.monotonic()-begin
    assert not torch.cuda.is_initialized()
    write(REPORT/'CPU_GATE.json',gate)
    write(REPORT/'INTERVENTION.json',dict(rank=540,FFN_matrices=72,factors_frozen=True,
        original_FFN_coefficients=313786368,factor_coefficients=223948800,
        final_folded_coefficients=FINAL,unfolded_coefficients=FINAL+TRAINABLE,
        trainable_coordinates=TRAINABLE,compression_data_free=True,teacher=False,
        adaptation='original qv-r8/alpha32/dropout.1/LR1e-4,19 FIT stages,87 updates',
        TEST_QA=0,worker_budget_seconds=7200,no_search_or_automatic_full_confirmation=True))
    sources=dict(old['sources'])
    sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')})
    for n in ['analysis/osft_increment_paired_analysis_20261022.py','analysis/onevision_research_device_entry.py']:
        sources[n]=sha(ROOT/n)
    evidence=[REPORT/n for n in ['PROTOCOL.md','QUESTION.md','NECESSITY_REVIEW.json','SOURCE_REVIEW.json','INTERVENTION.json','CPU_GATE.json']]
    evidence.extend(study.REPORT/n for n in ['READY.json','SPLIT.json','SCREEN_COMPLETE.json','LoRA_qv_COMPLETE.json','OSFT_all_COMPLETE.json'])
    evidence.append(cfg)
    refs=json.loads((study.REPORT/'SCREEN_COMPLETE.json').read_text())['all_four_results']
    for case in ['LoRA_qv','OSFT_all']:evidence.extend(ROOT/n for n in refs[case]['prediction_sha256'])
    write(REPORT/'READY.json',dict(sources=sources,evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
        base_files=old['base_files'],manifest_sha256=sha(path),split_sha256=sha(study.REPORT/'SPLIT.json'),
        training_ast_sha256=study.training_ast()[1],training_plan=plan,rank=540,worker_budget_seconds=7200,
        FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,final_folded_coefficients=FINAL,
        scientific_sources_FROZEN=True,automatic_full_confirmation=False))
    print(json.dumps(dict(ready_sha256=sha(REPORT/'READY.json'),CPU_gate_passed=True,actual_model_not_started=True)))
if __name__=='__main__':main()
