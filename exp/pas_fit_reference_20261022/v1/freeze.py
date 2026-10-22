"""One necessary CPU admission and freeze of the fresh PASs reference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import json
import time
from pathlib import Path
from core import ROOT,REPORT,PORT,study,sha,write,cpu_gate
from common import data,CANDIDATE

def main():
    assert not (REPORT/'READY.json').exists() and not (REPORT/'CPU_GATE.json').exists()
    review=json.loads((REPORT/'REVIEW.json').read_text());assert review['decision']=='accept'
    for n,h in review['files_sha256'].items():assert sha(ROOT/n)==h,n
    inherited=json.loads((study.REPORT/'READY.json').read_text())
    for n,h in inherited['sources'].items():assert sha(ROOT/n)==h,n
    path,m,_,split,_,_,_,plan=data()
    assert len(plan)==19 and sum(p['QA'] for p in plan)==1213
    candidate=json.loads((CANDIDATE/'VERIFIED.json').read_text())
    assert sha(CANDIDATE/'VERIFIED.json')=='6289cfecb65f989e75ab8e031e9ee85700a4b18b5005d5ef0203b0789ea9d1fc'
    assert candidate['complete'] and candidate['quality']['domain_macro']==42.30132909166463
    intervention=dict(method='original_PASs_MoE_fresh_FIT_only_reference',method_name_kept='PASs_MoE',learning_rate=1e-4,
        original_optimizer='HostAdamW',rank=128,experts=2,alpha=256,dropout=.05,gate_rank=16,
        router='a',gate_tie='A_T',original_learner_and_callbacks_unchanged=True,
        author_save_checkpoints=False,full_graph_checkpoints=19,
        numerical_exception='FP16 base/outer autocast, FP32 adapters internally autocast disabled and CPU saved tensors',
        only_new_pair='ShortFixed_minus_fresh_PAS',no_old_pair_or_bootstrap_repeat=True,
        fixed_candidate_macro=42.30132909166463,strict_point_rule_not_changed_by_cross_zero_interval=True)
    write(REPORT/'INTERVENTION.json',intervention)
    begin=time.monotonic();gate=cpu_gate();gate['seconds']=time.monotonic()-begin
    write(REPORT/'CPU_GATE.json',gate)
    sources=dict(inherited['sources']);sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in Path(__file__).parent.glob('*.py')})
    upstream=ROOT/'code/PASs_MoE'
    for p in [upstream/'files_sha256.json',upstream/'source_manifest.json',upstream/'source/CoIN/peft/tuners/coinmoelora.py',upstream/'source/cl_learner/moelora.py',
              PORT/'pass_adapter.py',ROOT/'analysis/osft_increment_paired_analysis_20261022.py',ROOT/'analysis/onevision_research_device_entry.py']:
        sources[p.relative_to(ROOT).as_posix()]=sha(p)
    evidence=[REPORT/n for n in ['PROTOCOL.md','REVIEW.json','INTERVENTION.json','CPU_GATE.json']]
    evidence += [study.REPORT/n for n in ['READY.json','SPLIT.json','SCREEN_COMPLETE.json']]
    evidence += [CANDIDATE/n for n in ['READY.json','COMPLETE.json','VERIFIED.json','OUTCOME.json','FINDINGS.md','RESULT_REVIEW.json']]
    evidence += [ROOT/n for n in candidate['quality']['prediction_sha256']]
    write(REPORT/'READY.json',dict(sources=sources,evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
        base_files=inherited['base_files'],manifest_sha256=sha(path),split_sha256=sha(study.REPORT/'SPLIT.json'),
        training_ast_sha256=study.training_ast()[1],training_plan=plan,FIT_QA=1213,DEV_QA=100,updates=87,TEST_QA=0,
        original_PAS_configuration=True,one_new_reference=True,automatic_full_confirmation=False,scientific_sources_FROZEN=True))
    print(json.dumps(dict(ready_sha256=sha(REPORT/'READY.json'),CPU_gate_passed=True,CPU_seconds=gate['seconds'],actual_model_not_started=True)))

if __name__=='__main__':main()
