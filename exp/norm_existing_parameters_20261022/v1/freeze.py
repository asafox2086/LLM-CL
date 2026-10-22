"""One necessary tiny-HF CPU admission; freeze independent scientific version."""
import json
import time
from pathlib import Path
from core import ROOT,REPORT,study,sha,write,cpu_gate,original_norm_source
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
    assert (t['hidden_size'],t['num_hidden_layers'])==(896,24)
    source=original_norm_source();assert source['transformers']=='4.46.3'
    write(REPORT/'INTERVENTION.json',dict(method='RMSNorm_only',target_modules=48,trainable_coordinates=43008,
        targets=['input_layernorm','post_attention_layernorm'],final_norm_frozen=True,learning_rate=1e-4,
        master_dtype='float32',forward_weight_dtype='float16',optimizer_state_dtype='float32',
        reference_names=['LoRA_qv','OSFT_all'],single_recipe_no_LR_search=True,
        strongest_PAS_substitution_not_tested=True,DEV_already_explored=True,automatic_full_confirmation=False))
    begin=time.monotonic();gate=cpu_gate();gate['seconds']=time.monotonic()-begin
    write(REPORT/'CPU_GATE.json',gate)
    sources=dict(old['sources'])
    sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')})
    for n in ['analysis/osft_increment_paired_analysis_20261022.py','analysis/onevision_research_device_entry.py']:
        sources[n]=sha(ROOT/n)
    evidence=[REPORT/n for n in ['PROTOCOL.md','REVIEW.json','INTERVENTION.json','CPU_GATE.json']]
    evidence+=[study.REPORT/n for n in ['READY.json','SPLIT.json','SCREEN_COMPLETE.json','LoRA_qv_COMPLETE.json','OSFT_all_COMPLETE.json']]
    evidence.append(config)
    for name in ['LoRA_qv','OSFT_all']:
        v=json.loads((study.REPORT/(name+'_COMPLETE.json')).read_text())
        evidence.extend(ROOT/n for n in v['prediction_sha256'])
    write(REPORT/'READY.json',dict(sources=sources,evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
        base_files=old['base_files'],manifest_sha256=sha(path),split_sha256=sha(study.REPORT/'SPLIT.json'),
        training_ast_sha256=study.training_ast()[1],training_plan=plan,method='RMSNorm_only',
        original_norm_source=source,trainable_coordinates=43008,FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,
        one_new_condition=True,automatic_full_confirmation=False,scientific_sources_FROZEN=True))
    print(json.dumps(dict(ready_sha256=sha(REPORT/'READY.json'),CPU_gate_passed=True,actual_model_not_started=True)))
if __name__=='__main__':main()
