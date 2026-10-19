"""Read-only CPU checkpoint-lineage audit for interrupted Qwen SeqFT.

No model forward, optimizer, CUDA, TEST scoring or artifact normalization.
Checkpoint pickle loading is limited to our own SHA-verified saved graphs.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib
import json
import pickle
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
FRAME=ROOT/'exp/two_backbone_native_cl_20261017_v2'
RUN=FRAME/'runs/qwen/pct01/SeqFT_LoRA/training_20261017_161209'
REPORT=ROOT/'summary/two_backbone_pct01_20261017/qwen_seqft_io_recovery_20261019'
sys.path.insert(0,str(FRAME))
import torch
from training_validity import parameter_digest


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(2)
    assert not (REPORT/'LINEAGE_VERIFIED.json').exists()
    config=read(RUN/'config.json')
    for name,h in config['execution_source_sha256'].items():
        assert sha(FRAME/name)==sha(RUN/'runner_source'/name)==h
    checkpoints={}
    for stage in [17,18,19]:
        p=RUN/f'checkpoints/stage_{stage:02d}.pt';meta=read(p.with_suffix('.json'))
        assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']
        obj=torch.load(p,map_location='cpu',pickle_module=pickle,weights_only=False,mmap=True)
        assert obj['stage']==stage and obj['checkpoint_phase']=='after_stage_evaluation'
        for k in ['method','condition','manifest_sha256','execution_source_sha256']:
            assert obj['config'][k]==config[k]
        model=obj['method'].model
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        digest=parameter_digest([p for _,p in named])
        validity=read(RUN/f'validity/stage_{stage:02d}.json')
        assert digest==validity['parameters_after_sha256']
        checkpoints[stage]=dict(checkpoint_sha256=meta['sha256'],trainable_sha256=digest,
            trainable_parameters=sum(p.numel() for _,p in named),stage=stage)
        del model,obj,named
    for stage in range(2,20):
        assert read(RUN/f'validity/stage_{stage:02d}.json')['parameters_before_sha256']==read(RUN/f'validity/stage_{stage-1:02d}.json')['parameters_after_sha256']
    assert checkpoints[17]['trainable_sha256']==read(RUN/'validity/stage_18.json')['parameters_before_sha256']
    assert checkpoints[18]['trainable_sha256']==read(RUN/'validity/stage_19.json')['parameters_before_sha256']
    rows=[];bad=[]
    for i,line in enumerate((RUN/'resources.jsonl').read_bytes().splitlines(),1):
        try:rows.append(json.loads(line))
        except (json.JSONDecodeError,UnicodeDecodeError):bad.append(i)
    boundaries=[{k:x.get(k) for k in ['phase','stage','unix','pid']} for x in rows
                if x.get('stage') in [17,18,19] and x.get('phase')!='periodic']
    phases=[x for x in boundaries if x['phase']=='stage_training_end_persistent' and x['stage']==18]
    assert len(phases)==2 and phases[0]['pid']!=phases[1]['pid']
    archived=list((RUN/'interrupted_evaluations').glob('*/stage_18'))
    assert len(archived)==1
    archive_time=int(archived[0].parent.name)/1e9
    assert phases[0]['unix']<archive_time<phases[1]['unix']
    current=[]
    for tid in [18,19]:
        p=RUN/f'evaluations/stage_18/T{tid:02d}_test_predictions.jsonl'
        assert p.stat().st_mtime>phases[1]['unix']
        current.append(dict(path=str(p.relative_to(RUN)),sha256=sha(p),mtime=p.stat().st_mtime))
    assert (RUN/'checkpoints/stage_18.pt').stat().st_mtime>max(x['mtime'] for x in current)
    assert (RUN/'checkpoints/stage_19.pt').stat().st_mtime>(RUN/'checkpoints/stage_18.pt').stat().st_mtime
    # The frozen resume code quarantines every evaluation above its durable
    # checkpoint before re-fitting; it never resumes the old stage18 files.
    code=(FRAME/'run.py').read_text()
    assert "if int(p.name.split('_')[-1])>resume_obj['stage']:" in code
    assert "shutil.move(str(p),str(target))" in code
    events=[json.loads(s) for s in (RUN/'logs/events.jsonl').read_text().splitlines()]
    e18=[(i,e) for i,e in enumerate(events) if e.get('event')=='train_update' and e['stage']==18]
    assert [e['step'] for _,e in e18]==list(range(1,8))*2
    assert max(i for i,e in enumerate(events) if e.get('event')=='train_update' and e['stage']==17)<e18[0][0]
    assert e18[-1][0]<min(i for i,e in enumerate(events) if e.get('event')=='train_update' and e['stage']==19)
    drop=[i for i,_ in e18[:7]]
    proof=dict(verified=True,CPU_only=True,no_CUDA_initialized=not torch.cuda.is_initialized(),
        no_forward_or_optimizer_or_TEST_scoring=True,source=str(RUN.relative_to(ROOT)),
        audit_source_sha256=sha(Path(__file__)),checkpoints=checkpoints,
        validity_chain_continuous_all19=True,stage18_fresh_evaluations_after_restart=True,
        original_interrupted_evaluations_quarantined_by_frozen_runner=True,
        archived_folder=str(archived[0].relative_to(RUN)),archive_time=archive_time,
        current_stage18_files=current,boundaries=boundaries,resource_invalid_line_numbers=bad,
        original_events_sha256=sha(RUN/'logs/events.jsonl'),interrupted_event_indices=drop,
        original_resources_sha256=sha(RUN/'resources.jsonl'),discarded_updates=7,
        all_original_files_unchanged=True,
        limitation='Does not publish completion or rescore predictions; an explicit derivative branch must retain discarded compute and use the unchanged full verifier.')
    (REPORT/'LINEAGE_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({k:proof[k] for k in ['verified','CPU_only','stage18_fresh_evaluations_after_restart','discarded_updates','no_CUDA_initialized']}))


if __name__=='__main__':main()
