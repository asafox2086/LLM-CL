"""TRAIN-only 2x2 pilot: retain current adaptation, freeze A, faster B updates.

Borrow LoRA-FA's frozen-A topology and LoRA+'s asymmetric learning rates.
No TEST metric selects these fixed conditions. This is not a full CL result,
nor a reproduction of LoRA-FA's gradient-corrected optimizer.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import fcntl
import gc
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/train_update_pilot_20261019/v1'
TARGETS = [2, 4, 11]
VIEWS = ['continue_AB', 'continue_AB_plus16', 'fixed_A', 'fixed_A_plus16']


def main(gpu):
    import onevision_research_device_entry as device
    cards = device.inventory(); assert device.select(cards, os.getuid()) == gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    from transformers import get_cosine_schedule_with_warmup
    import onevision_learned_reuse_trial as native
    from onevision_knowledge_intervention import reset_seed, valid_checkpoint
    from onevision_research_worker import atomic_torch, digest
    from run import hashed, write
    torch.set_num_threads(6); torch.use_deterministic_algorithms(True); reset_seed()
    REPORT.mkdir(parents=True, exist_ok=True)
    assert not (REPORT/'attempt.json').exists(), 'Preserve failed evidence; no blind retry'
    lock=(REPORT/'worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((REPORT/'PLAN.json').read_text())
    assert digest(Path(__file__)) == plan['worker_sha256']
    _, values = native.plan(); manifest, data, domains, _, _, original, proof = values
    anchor_path=original/'adapters/sequential_T01.pt'
    assert valid_checkpoint(anchor_path)
    assert digest(anchor_path) == proof['artifact_sha256']['adapters/sequential_T01.pt'] == plan['anchor_sha256']
    anchor=torch.load(anchor_path,map_location='cpu',weights_only=True)['adapter']
    rows={t:sorted(data[t,'train'],key=lambda r:hashed(r,'training')) for t in TARGETS}
    old_rows=sorted(data[1,'train'],key=lambda r:hashed(r,'fixed-old-TRAIN-update-pilot'))[:4]
    assert {str(t):[r['id'] for r in rr] for t,rr in rows.items()} == plan['target_TRAIN_ids']
    assert [r['id'] for r in old_rows] == plan['old_TRAIN_ids']
    run=ROOT/'exp/onevision_train_update_pilot_20261019/v1'/time.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    write(REPORT/'attempt.json',dict(run=str(run.relative_to(ROOT)),pid=os.getpid(),plan_sha256=digest(REPORT/'PLAN.json')))
    write(run/'physical_device_execution.json',dict(gpu=gpu,inventory=cards))
    x=native.MetricExperiment(manifest,run)
    results=[]

    def evaluate(selected):
        x.model.eval(); x.cache.clear()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]),torch.no_grad():
            return [float(x.loss(r)[0]) for r in selected]

    initial_nll={}
    for task in TARGETS:
        for view in VIEWS:
            prefix=f'T{task:02d}_{view}'; x.configure(anchor,finetune_shared=True)
            frozen=view.startswith('fixed_A'); ratio=16 if view.endswith('plus16') else 1
            for layer in x.layers.values():layer.A.weight.requires_grad_(not frozen)
            start=x.snapshot()
            assert all(torch.equal(start[n][k],anchor[n][k]) for n in anchor for k in ['A','B'])
            before_target=evaluate(rows[task]); before_old=evaluate(old_rows)
            if task not in initial_nll:initial_nll[task]=(before_target,before_old)
            assert (before_target,before_old)==initial_nll[task]
            reset_seed(); x.model.train(); x.cache.clear(); gc.collect(); torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            before=x.monitor.snapshot(prefix+'_TRAIN_before',model=x.model)
            group_a=[l.A.weight for l in x.layers.values() if l.A.weight.requires_grad]
            group_b=[l.B.weight for l in x.layers.values()]
            parameters=group_a+group_b
            expected=196608 if frozen else 540672
            assert sum(p.numel() for p in parameters)==expected
            optimizer=torch.optim.AdamW([dict(params=group_a,lr=1e-4),dict(params=group_b,lr=1e-4*ratio)],weight_decay=0)
            updates=math.ceil(len(rows[task])/16)
            scheduler=get_cosine_schedule_with_warmup(optimizer,min(int(.03*updates),max(0,updates-1)),updates)
            events=[]
            named=[(n,p) for n,p in x.model.named_parameters() if p.requires_grad]
            for step,offset in enumerate(range(0,len(rows[task]),16),1):
                optimizer.zero_grad(set_to_none=True); batch=rows[task][offset:offset+16]; losses=[]
                for row in batch:
                    loss,_=x.loss(row); assert torch.isfinite(loss)
                    (loss/len(batch)).backward(); losses.append(float(loss.detach()))
                raw=torch.cat([p.grad.detach().float().cpu().flatten() for _,p in named])
                assert torch.isfinite(raw).all() and float(raw.norm())>0
                atomic_torch(run/f'raw_gradients/{prefix}/step{step}.pt',dict(raw_gradient=raw,
                    layout=[dict(name=n,shape=list(p.shape)) for n,p in named],TRAIN_ids=[r['id'] for r in batch]))
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm)
                rates=[g['lr'] for g in optimizer.param_groups];assert min(rates)>0
                optimizer.step();scheduler.step()
                events.append(dict(step=step,TRAIN_ids=[r['id'] for r in batch],loss=sum(losses)/len(losses),LR=rates,
                                   raw_gradient_l2=float(raw.norm()),gradient_l2_preclip=float(norm)))
            after=x.monitor.snapshot(prefix+'_TRAIN_after',model=x.model)
            end=x.snapshot(); changes={}
            for k in ['A','B']:
                changes[k]=math.sqrt(sum(float((end[n][k].double()-anchor[n][k].double()).square().sum()) for n in anchor))
            assert changes['B']>0 and (changes['A']==0 if frozen else changes['A']>0)
            atomic_torch(run/f'states/{prefix}.pt',dict(adapter=end,source_anchor_sha256=digest(anchor_path),view=view,task=task))
            write(run/f'events/{prefix}.json',events)
            optimizer.zero_grad(set_to_none=True);del optimizer,scheduler,loss;gc.collect();torch.cuda.empty_cache()
            target_after=evaluate(rows[task]);old_after=evaluate(old_rows)
            item=dict(task=task,view=view,trainable_parameters=expected,target_ids=[r['id'] for r in rows[task]],
                target_before=before_target,target_after=target_after,old_before=before_old,old_after=old_after,
                A_B_displacement_l2=changes,updates=updates,TRAIN_memory=dict(before=before,after=after,
                peak_allocated_bytes=after['gpu_peak_allocated_bytes'],peak_reserved_bytes=after['gpu_peak_reserved_bytes']))
            write(run/f'results/{prefix}.json',item);results.append(item)
    write(run/'RESULTS.json',dict(conditions=results,full_CL_result=False,TEST_predictions=0,
        paired_comparison_TRAIN_only=True,renderer_gate_includes_existing_TEST_sentinel='FinMME:5082',
        source_history_QA=len(data[1,'train']),source_history_updates=math.ceil(len(data[1,'train'])/16),
        source_refit_updates=0,source_history_charged=True,no_F1_claim=True))
    files={str(p.relative_to(run)):digest(p) for p in run.rglob('*') if p.is_file() and p.suffix in ['.pt','.json','.jsonl']}
    write(REPORT/'complete.json',dict(complete=True,run=str(run.relative_to(ROOT)),plan_sha256=digest(REPORT/'PLAN.json'),
        artifact_sha256=files,conditions=12,TEST_used_for_selection=False,no_F1_claim=True))
    print(json.dumps(dict(complete=True,run=str(run.relative_to(ROOT)),conditions=12)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    try:main(p.parse_args().gpu)
    except Exception:
        import traceback
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True))+'\n')
        raise
