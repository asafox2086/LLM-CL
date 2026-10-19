"""Admission: frozen TRAIN/source IDs, real optimizer grouping, safe imports."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
import onevision_learned_reuse_trial as native
import onevision_train_only_update_pilot as worker
from onevision_research_worker import digest
from run import hashed, write


def main():
    assert not torch.cuda.is_initialized()
    _,values=native.plan();_,data,_,_,_,original,proof=values
    anchor=original/'adapters/sequential_T01.pt'
    assert digest(anchor)==proof['artifact_sha256']['adapters/sequential_T01.pt']
    weights=torch.load(anchor,map_location='cpu',weights_only=True)['adapter']
    assert sum(w['A'].numel()+w['B'].numel() for w in weights.values())==540672
    assert sum(w['B'].numel() for w in weights.values())==196608
    assert all(w['A'].shape[0]==w['B'].shape[1]==8 for w in weights.values())
    controls=[]
    for frozen in [False,True]:
        for ratio in [1,16]:
            torch.manual_seed(42)
            a=nn.Parameter(torch.randn(2,5),requires_grad=not frozen)
            b=nn.Parameter(torch.randn(3,2));old_a=a.detach().clone();old_b=b.detach().clone()
            x=torch.randn(4,5);target=torch.randn(4,3)
            loss=F.mse_loss(4*F.linear(F.linear(x,a),b),target);loss.backward()
            assert b.grad is not None and torch.isfinite(b.grad).all()
            assert (a.grad is None)==frozen
            optimizer=torch.optim.AdamW([dict(params=[] if frozen else [a],lr=1e-4),dict(params=[b],lr=1e-4*ratio)],weight_decay=0)
            scheduler=get_cosine_schedule_with_warmup(optimizer,0,1)
            assert [g['lr'] for g in optimizer.param_groups]==[1e-4,1e-4*ratio]
            optimizer.step();scheduler.step()
            assert not torch.equal(b,old_b) and (torch.equal(a,old_a)==frozen)
            controls.append(dict(frozen_A=frozen,ratio=ratio,positive_LR_and_nonzero_updates=True))
    target={str(t):[r['id'] for r in sorted(data[t,'train'],key=lambda r:hashed(r,'training'))] for t in worker.TARGETS}
    old=[r['id'] for r in sorted(data[1,'train'],key=lambda r:hashed(r,'fixed-old-TRAIN-update-pilot'))[:4]]
    assert list(map(len,target.values()))==[11,26,12]
    assert not set(old)&{i for rr in target.values() for i in rr}
    for gpu in ['0','1']:
        script="import os,sys;os.environ['CUDA_VISIBLE_DEVICES']="+repr(gpu)+";sys.path.insert(0,"+repr(str(worker.ROOT/'analysis'))+");import onevision_train_only_update_pilot;import torch;assert os.environ['CUDA_VISIBLE_DEVICES']=="+repr(gpu)+";assert not torch.cuda.is_initialized()"
        subprocess.run([sys.executable,'-c',script],check=True)
    assert not torch.cuda.is_initialized()
    worker.REPORT.mkdir(parents=True,exist_ok=True)
    plan=dict(worker_sha256=digest(Path(worker.__file__)),source_manifest_sha256=native.plan()[0]['manifest_sha256'],
        anchor_sha256=digest(anchor),anchor=str(anchor.relative_to(worker.ROOT)),
        target_TRAIN_ids=target,old_TRAIN_ids=old,views=worker.VIEWS,
        source_history_QA=len(data[1,'train']),source_history_updates=(len(data[1,'train'])+15)//16,
        actual_model_updates=16,TRAIN_exposures=196,target_NLL_forwards=392,old_NLL_forwards=96,
        no_TEST_based_selection=True,full_CL_result=False)
    write(worker.REPORT/'PLAN.json',plan)
    write(worker.REPORT/'CPU_CHECK.json',dict(passed=True,checks=7,no_CUDA_initialized=True,
        actual_source_weights_checked=True,synthetic_optimizer_controls=controls,
        expected_trainable_AB=540672,expected_trainable_B=196608,
        worker_sha256=digest(Path(worker.__file__)),cpu_source_sha256=digest(Path(__file__)),GPU_started=False))
    print(json.dumps(dict(passed=True,CPU_only=True,checks=7,conditions=12,actual_model_updates=16)))


if __name__=='__main__':main()
