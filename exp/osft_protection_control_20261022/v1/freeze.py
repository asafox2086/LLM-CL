"""One CPU admission then freeze; never starts the actual model or GPU worker."""
import json
import subprocess
import time
from pathlib import Path
from core import ROOT,PORT,PILOT,REPORT,study,sha,write,cpu_gate
from common import data

def main():
    assert not (REPORT/'READY.json').exists()
    review=json.loads((REPORT/'REVIEW.json').read_text());assert review['decision']=='accept'
    for n,h in review['files_sha256'].items():assert sha(ROOT/n)==h,n
    old=json.loads((study.REPORT/'READY.json').read_text())
    for n,h in old['sources'].items():assert sha(ROOT/n)==h,n
    path,m,_,split,_,_,_,plan=data()
    config=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json'
    t=json.loads(config.read_text())['text_config'];assert (t['hidden_size'],t['intermediate_size'],t['num_hidden_layers'],t['num_attention_heads'],t['num_key_value_heads'])==(896,4864,24,14,2)
    count=113766144
    write(REPORT/'INTERVENTION.json',dict(per_step_projection_package='disabled',
        callbacks=['post_backward','after_update'],expected_skipped_requests=dict(gradient=1213,parameter=87),
        trainable_coordinates=count,target_modules=168,unfreeze_rank_ratio=.25,learning_rate=1e-5,
        unchanged='Original seven-target construction/forward/author kernels/HostAdamW/offload/stagewise re-SVD',
        scope='Package net effect; feasible set expands; not separate hook effects or pure optimizer geometry'))
    prior_path='summary/edge_deployment_20261019/update_geometry_literature_20261022/FINDINGS.md'
    prior=subprocess.check_output(['git','show','cc445788e7a43f6f817abc21cfe8b833adce4651:'+prior_path],cwd=ROOT).decode()
    lines=[line for line in prior.splitlines() if '逐步投影' in line];assert lines
    write(REPORT/'QUESTION_PROVENANCE.json',dict(prior_commit='cc445788e7a43f6f817abc21cfe8b833adce4651',
        prior_file=prior_path,matching_lines=lines,prior_blob_sha256=__import__('hashlib').sha256(prior.encode()).hexdigest(),
        current_protocol_written_after_confirmation=True,current_TEST_not_configuration_selection=True))
    begin=time.monotonic();gate=cpu_gate();gate['seconds']=time.monotonic()-begin
    write(REPORT/'CPU_GATE.json',gate)
    sources=dict(old['sources']);sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')})
    sources['analysis/osft_increment_paired_analysis_20261022.py']=sha(ROOT/'analysis/osft_increment_paired_analysis_20261022.py')
    sources['analysis/onevision_research_device_entry.py']=sha(ROOT/'analysis/onevision_research_device_entry.py')
    evidence=[REPORT/n for n in ['PROTOCOL.md','REVIEW.json','INTERVENTION.json','QUESTION_PROVENANCE.json','CPU_GATE.json']]
    evidence+=[study.REPORT/n for n in ['READY.json','SPLIT.json','SCREEN_COMPLETE.json','OSFT_all_COMPLETE.json']]
    evidence.append(config)
    for name in ['OSFT_all']:
        v=json.loads((study.REPORT/(name+'_COMPLETE.json')).read_text())
        evidence.extend(ROOT/n for n in v['prediction_sha256'])
    write(REPORT/'READY.json',dict(sources=sources,evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
        base_files=old['base_files'],manifest_sha256=sha(path),split_sha256=sha(study.REPORT/'SPLIT.json'),
        training_ast_sha256=study.training_ast()[1],training_plan=plan,per_step_projection_package='disabled',
        trainable_coordinates=count,FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,
        one_new_condition=True,automatic_full_confirmation=False,scientific_sources_FROZEN=True))
    print(json.dumps(dict(ready_sha256=sha(REPORT/'READY.json'),CPU_gate_passed=True,actual_model_not_started=True)))
if __name__=='__main__':main()
