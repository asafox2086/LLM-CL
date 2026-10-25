"""Three fixed-r8 replay/EMA controls; blind terminal readouts only.

The inherited training AST retains current-QA accumulation, optimizer, scheduler,
clipping and scaler order. Its existing method hooks implement replay and EMA.
No model work occurs on import; all numerical model work runs on CUDA.
"""
from pathlib import Path
import ast
import gc
import hashlib
import json
import random
import sys
import traceback
from types import SimpleNamespace
from common import ROOT,RUN,ASSET,ARMS,Context,Batcher,read,sha,write,module

COORDINATES=540672
PARAMETERS=96
GRADIENT_BYTES=2162688
OPTIMIZER_BYTES=4325760


def same_receipt(actual,expected):
    for key,value in expected.items():
        if key!='device':assert actual[key]==value,key


def row_key(task,row):
    return dict(task=task['task'],id=row['id'],domain=task['domain'])


def order_rows(task,rows):
    return sorted(rows,key=lambda r:hashlib.sha256(f"42:{task['task']}:{r['id']}".encode()).digest())


class ReplaySchedule:
    """Past-task-only reservoir; independent metadata RNGs never touch dropout."""
    def __init__(self,ctx,arm):
        self.ctx,self.arm=ctx,arm
        self.pool=[];self.seen=0;self.insert_rng=random.Random(42)
        self.draw_rng=random.Random(42);self.stage=0;self.step=0;self.global_update=0
        self.current=[];self.replays=[];self.masks=[];self.index=0;self.stats=[]
        self.events=[];self.memory_events=[];self.stage_starts=[];self.stage_ends=[]
        self.gate_complete=read(RUN/'train_gate_COMPLETE.json') if arm=='shuffle' else None

    def begin(self,stage):
        assert not self.stats and self.stage==stage-1
        self.stage=stage;self.step=0
        task=self.ctx.manifest['tasks'][stage-1]
        self.order=order_rows(task,self.ctx.fit[task['task']])
        self.stage_starts.append(dict(stage=stage,cached_QA=len(self.pool),seen_past_QA=self.seen,
            cached_IDs=[row_key(t,r) for t,r in self.pool]))

    def prepare(self,processor=None):
        assert processor is None or processor is self.ctx.processor
        assert not self.stats and self.index==len(self.current)
        self.step+=1;self.global_update+=1
        task=self.ctx.manifest['tasks'][self.stage-1]
        self.current=self.order[(self.step-1)*16:self.step*16]
        assert self.current
        self.replays=[self.pool[self.draw_rng.randrange(len(self.pool))] for _ in self.current] if self.pool else []
        assert bool(self.replays)==(self.stage>1)
        self.index=0;self.masks=[];self.reference=None
        if self.arm=='shuffle':
            filename=f'gate_MASK_S{self.stage:02}_{self.step:03}.json'
            assert sha(RUN/filename)==self.gate_complete['files'][filename]
            self.reference=read(RUN/filename)
            assert self.reference['stage']==self.stage and self.reference['step']==self.step
            assert self.reference['current_QA']==[row_key(task,r) for r in self.current]
            assert self.reference['replay_QA']==[row_key(t,r) for t,r in self.replays]
            self.masks=list(self.reference['natural_gate'])
            seed=int(hashlib.sha256(f'42:reference-mask:{self.stage}:{self.step}'.encode()).hexdigest(),16)
            random.Random(seed).shuffle(self.masks)
            assert sum(self.masks)==sum(self.reference['natural_gate'])

    def take(self):
        assert self.index<len(self.current)
        row=self.current[self.index]
        replay=self.replays[self.index] if self.replays else None
        applied=self.masks[self.index] if self.arm=='shuffle' and replay is not None else None
        self.index+=1
        return row,replay,applied

    def finish_update(self,step,updates,method):
        assert step==self.step and self.index==len(self.current)
        assert len(self.stats)==len(self.replays)
        assert method.ema_update_count==self.global_update
        task=self.ctx.manifest['tasks'][self.stage-1]
        flags=[int(s['natural_gate']) for s in self.stats]
        applied=[int(s['applied_mask']) for s in self.stats]
        assert all(v in (0,1) for v in flags+applied)
        if self.arm=='original':assert applied==[1]*len(self.replays)
        if self.arm=='gate':assert applied==flags
        if self.arm=='shuffle':assert applied==self.masks and sum(applied)==sum(self.reference['natural_gate'])
        filename=f'{self.arm}_MASK_S{self.stage:02}_{self.step:03}.json'
        proof=dict(stage=self.stage,step=self.step,global_update=self.global_update,stage_updates=updates,
            current_QA=[row_key(task,r) for r in self.current],replay_QA=[row_key(t,r) for t,r in self.replays],
            natural_gate=flags,applied_mask=applied,stats=self.stats,
            EMA_updates=method.ema_update_count,accepted=sum(applied),replay_count=len(self.replays),
            denominator='all_replay_QA; unchanged current minibatch accumulation',
            reference_mask_sha256=sha(RUN/f'gate_MASK_S{self.stage:02}_{self.step:03}.json') if self.arm=='shuffle' else None)
        write(filename,proof);self.events.append(dict(file=filename,sha256=sha(RUN/filename),
            stage=self.stage,step=self.step,current_QA=len(self.current),replay_QA=len(self.replays),accepted=sum(applied)))
        self.stats=[]

    def end(self,stage):
        assert stage==self.stage and not self.stats and self.index==len(self.current)
        task=self.ctx.manifest['tasks'][stage-1]
        for row in self.order:
            seen_before=self.seen;self.seen+=1
            if len(self.pool)<500:
                slot=len(self.pool);self.pool.append((task,row));retained=True
            else:
                slot=self.insert_rng.randrange(self.seen);retained=slot<500
                if retained:self.pool[slot]=(task,row)
            self.memory_events.append(dict(stage=stage,source=row_key(task,row),seen_before=seen_before,
                draw=slot,retained=retained,cached_QA=len(self.pool)))
        # These are encoded immutable QA-reference metadata bytes, never RAM/RSS.
        encoded=json.dumps([row_key(t,r) for t,r in self.pool],sort_keys=True,separators=(',',':')).encode()
        self.stage_ends.append(dict(stage=stage,cached_QA=len(self.pool),seen_QA=self.seen,
            reference_ledger_bytes=len(encoded),reference_ledger_sha256=hashlib.sha256(encoded).hexdigest()))


def create_initial(ctx,arm):
    import deployment
    model,base=deployment.load_base(ctx.base,ctx.device,ctx.storage)
    calibration=read(RUN/'calibration_COMPLETE.json')
    assert base['digest']==calibration['initial_state_digest']
    assert base['all_buffers_digest']==calibration['all_buffers_digest']
    assert base['layout']==calibration['layout']
    same_receipt(base,read(RUN/'prior_naive_train_COMPLETE.json')['original_base'])
    return deployment.export(model,RUN/(arm+'_initial.pt'),ctx.storage),base


def train(ctx,arm):
    import torch
    import adaptation,deployment,loss_core
    adaptation.configure_gpu_math()
    initial_cp,base=create_initial(ctx,arm);gc.collect();torch.cuda.empty_cache()
    model,loaded=deployment.load_complete(RUN/(arm+'_initial.pt'),ctx.device,ctx.storage);ctx.bind(model)
    same_receipt(loaded,base);initial_invariant=deployment.finish(model,loaded,ctx.storage)
    ctx.reset();method=adaptation.make_plain(model,ASSET,ctx.storage)
    prior_initial=read(RUN/'prior_naive_ADAPTER_INITIAL.json')
    assert method.initialization==prior_initial['initialization']
    assert method.rank_plan==prior_initial['rank_plan']
    slow=adaptation.ema_state(method)
    assert len(slow)==PARAMETERS and sum(t.numel() for t in slow.values())==COORDINATES
    assert all(t.is_cuda and t.dtype==torch.float32 and not t.requires_grad for t in slow.values())
    slow_ids={id(t) for t in slow.values()}
    canonical_slow={n:b for n,b in model.core.named_buffers() if id(b) in slow_ids}
    assert len(canonical_slow)==PARAMETERS and {id(t) for t in canonical_slow.values()}==slow_ids
    assert all(n.endswith(('._ilora_slow_A','._ilora_slow_B')) for n in canonical_slow)
    ordinary=lambda:{n:b for n,b in model.core.named_buffers() if n not in canonical_slow}
    assert sum(t.numel()*t.element_size() for t in ordinary().values())==9032
    write(arm+'_ADAPTER_INITIAL.json',dict(coordinates=COORDINATES,parameters=PARAMETERS,
        initialization=method.initialization,rank_plan=method.rank_plan,initialization_exact_to_naive=True,
        slow_coordinates=COORDINATES,slow_bytes=GRADIENT_BYTES,slow_buffers=list(canonical_slow),
        slow_initial_digest=ctx.storage.digest(slow),learned_gate_parameters=0))
    batcher=Batcher(ctx,model,'TRAIN');replay_batcher=Batcher(ctx,model,'REPLAY')
    frozen=lambda:ctx.storage.digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
    frozen_before=frozen();buffers_before=ctx.storage.digest(ordinary())
    adapter_before=ctx.storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    formal=module('arc_original_train',ASSET/'exp/two_backbone_native_cl_20261017_v2/run.py')
    directory=RUN/(arm+'_training');directory.mkdir();(directory/'logs').mkdir()
    monitor=formal.ResourceMonitor(directory);monitor.snapshot('training_before',model=model)
    mapped={};plan=[]
    for task in ctx.manifest['tasks']:
        stage=task['task'];mapped[stage,'train']=ctx.fit[stage];mapped[stage,'dev']=ctx.dev[stage]
        order=order_rows(task,ctx.fit[stage])
        task['training_order_sha256']=hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        plan.append(dict(stage=stage,QA=len(order),updates=(len(order)+15)//16,order_sha256=task['training_order_sha256']))
    assert plan==ctx.ready['training_plan']
    schedule=ReplaySchedule(ctx,arm);counts={};holder=[];raw_clip=torch.nn.utils.clip_grad_norm_
    def begin_stage(stage):
        adaptation.sync_slow(method);assert method.ema_stage_sync_count==stage
        schedule.begin(stage)
    def after_update(step,updates):
        adaptation.update_slow(method);schedule.finish_update(step,updates,method)
    method.begin_stage=begin_stage;method.prepare_update=schedule.prepare
    method.after_update=after_update;method.end_stage=schedule.end
    def answer_loss(m,batch,answer,processor):
        assert m is model
        row,past,mask=schedule.take()
        task=ctx.manifest['tasks'][schedule.stage-1]
        assert answer==row['answer']
        current,n=ctx.rt.answer_loss(model,batch,answer,processor)
        if past is None:return current,n
        past_task,past_row=past
        assert past_task['task']<task['task']
        replay,_=replay_batcher(past_row,past_task,processor,model.device)
        replay_ce,_,stats=loss_core.answer_loss(model,replay,past_row['answer'],processor,
            method=method,replay=True,arm=arm,applied_mask=mask)
        consistency=stats.pop('consistency_loss_tensor')
        assert current.ndim==replay_ce.ndim==consistency.ndim==0
        assert all(torch.isfinite(value).all() for value in (current,replay_ce,consistency))
        schedule.stats.append(stats)
        return (current+replay_ce)/2+consistency,n
    def clip(parameters,*args,**kw):
        parameters=list(parameters);stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        grads=[p.grad for p in parameters if p.grad is not None]
        assert len(grads)==PARAMETERS
        assert all(g.is_cuda and g.dtype==torch.float32 and torch.isfinite(g).all() for g in grads)
        assert sum(g.numel()*g.element_size() for g in grads)==GRADIENT_BYTES
        write(f'{arm}_grad_S{stage:02}_{counts[stage]:03}.json',dict(parameters=PARAMETERS,finite=True,gradient_bytes=GRADIENT_BYTES))
        return raw_clip(parameters,*args,**kw)
    def optimizer(m):
        assert not holder
        opt=torch.optim.AdamW([p for p in m.model.parameters() if p.requires_grad],lr=1e-4,
            betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False,capturable=True)
        holder.append(opt);return opt
    def save_stage(stage):
        assert frozen()==frozen_before and ctx.storage.digest(ordinary())==buffers_before
        opt=holder.pop();states=[t for state in opt.state.values() for t in state.values() if isinstance(t,torch.Tensor)]
        assert states and all(t.is_cuda and torch.isfinite(t).all() for t in states)
        assert sum(t.numel()*t.element_size() for t in states)==OPTIMIZER_BYTES
        adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        assert len(adapter)==PARAMETERS
        torch.save(adapter,RUN/f'{arm}_adapter_S{stage:02}.pt')
        ema=adaptation.ema_state(method)
        write(f'{arm}_STAGE_{stage:02}_COMPLETE.json',dict(stage=stage,updates=counts[stage],
            adapter_digest=ctx.storage.digest(adapter),bytes=GRADIENT_BYTES,optimizer_state_bytes=OPTIMIZER_BYTES,
            all_optimizer_state_GPU=True,frozen_unchanged=True,ordinary_buffers_unchanged=True,
            EMA_digest=ctx.storage.digest(ema),EMA_coordinates=COORDINATES,EMA_bytes=GRADIENT_BYTES,
            EMA_updates=method.ema_update_count,EMA_stage_syncs=method.ema_stage_sync_count,
            memory=schedule.stage_ends[-1]))
    namespace=dict(vars(formal),model=model,method=method,processor=ctx.processor,manifest=ctx.manifest,
        data=mapped,run=directory,monitor=monitor,args=SimpleNamespace(method='SeqFT_LoRA'),resume_obj=None,
        input_batch=batcher,answer_loss=answer_loss,prepare=lambda *a:None,visual_cache_stats=ctx.fv.stats,
        make_optimizer=optimizer,save_stage=save_stage)
    prefix,training_ast=ctx.study.training_ast();assert training_ast==ctx.ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),
        iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),
        body=prefix+ast.parse('save_stage(stage)').body,orelse=[])
    ctx.reset();torch.nn.utils.clip_grad_norm_=clip
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<unchanged87_answer_reliability>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert sum(counts.values())==87 and not holder and len(batcher.records)==1213
    assert len(replay_batcher.records)==886 and len(schedule.events)==87
    assert schedule.seen==1213 and len(schedule.pool)==500
    assert method.ema_update_count==87 and method.ema_stage_sync_count==19
    assert frozen()==frozen_before and ctx.storage.digest(ordinary())==buffers_before
    assert batcher.records==read(RUN/'prior_naive_FIT_INPUTS.json')
    monitor.snapshot('training_after',model=model);memory=monitor.finish(model);assert not monitor.thread.is_alive()
    adapter_after=ctx.storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})
    assert adapter_after!=adapter_before
    model.core.language_model.disable_input_require_grads();model.core.language_model.gradient_checkpointing_disable()
    final_ema_digest=ctx.storage.digest(adaptation.ema_state(method))
    namespace.clear();fold=adaptation.fold(model,method)
    assert not any(n.endswith(('._ilora_slow_A','._ilora_slow_B')) for n,_ in model.core.named_buffers())
    cp=deployment.export(model,RUN/(arm+'_complete.pt'),ctx.storage)
    ledger=dict(events=schedule.events,reservoir_events=schedule.memory_events,stage_starts=schedule.stage_starts,
        stage_ends=schedule.stage_ends,past_only=True,capacity=500,with_replacement=True,
        task_end_insert=True,current_QA=1213,replay_QA=886,updates=87,
        extra_full_decoder_for_gate=0,working_replay_forwards=886,slow_replay_forwards=886,
        max_target_head_calls=886,reference_mask_counterfactual=arm=='shuffle')
    write(arm+'_REPLAY_LEDGER.json',ledger);write(arm+'_FIT_INPUTS.json',batcher.records)
    write(arm+'_REPLAY_INPUTS.json',replay_batcher.records);write(arm+'_TRAIN_MEMORY.json',memory)
    files=[p for p in RUN.iterdir() if p.is_file() and p.name.startswith(arm+'_') and p.suffix=='.json']
    files.extend(p for p in directory.rglob('*') if p.is_file())
    files.extend(RUN/f'{arm}_adapter_S{s:02}.pt' for s in range(1,20))
    files.extend([RUN/(arm+'_initial.pt'),RUN/(arm+'_complete.pt')])
    ctx.end(dict(arm=arm,new_generations=0,FIT_backwards=1213,current_QA=1213,replay_QA=886,
        working_replay_forwards=886,slow_replay_forwards=886,max_target_head_calls=886,
        optimizer_updates=87,coordinates=COORDINATES,frozen_EMA_coordinates=COORDINATES,
        training_plan=plan,training_ast_sha256=training_ast,original_base=base,
        initial_checkpoint=initial_cp,initial_reload=loaded,initial_invariant=initial_invariant,
        checkpoint=cp,fold=fold,naive_initialization_exact=True,naive_current_inputs_exact=True,
        learned_gate_parameters=0,adapter_before=adapter_before,adapter_after=adapter_after,
        EMA_updates=87,EMA_stage_syncs=19,final_EMA_digest=final_ema_digest,
        final_default_only=True,final_EMA_removed=True,reference_masks=arm=='shuffle',
        frozen_unchanged=True,ordinary_buffers_unchanged=True),files)


def evaluate(ctx,arm):
    import adaptation,deployment,readouts
    adaptation.configure_gpu_math()
    model,before=deployment.load_complete(RUN/(arm+'_complete.pt'),ctx.device,ctx.storage);ctx.bind(model)
    assert before['digest']==read(RUN/('train_'+arm+'_COMPLETE.json'))['checkpoint']['digest']
    batcher=Batcher(ctx,model,'RESOURCE')
    memory,_=readouts.resource_windows(model,ctx.processor,ctx.rows,ctx.storage,
        batch_provider=batcher.readout,generation_policy=ctx.stop)
    write(arm+'_memory.json',memory);batcher.phase='DEV_FINAL'
    values,_=readouts.evaluate_dev(model,ctx.processor,ctx.rows,ctx.storage,
        batch_provider=batcher.readout,generation_policy=ctx.stop)
    write(arm+'_final_predictions.json',values);invariant=deployment.finish(model,before,ctx.storage)
    ctx.end(dict(arm=arm,new_generations=105,FIT_backwards=0,optimizer_updates=0,
        reloaded=before,invariant=invariant,final_default_only=True),
        [RUN/(arm+'_memory.json'),RUN/(arm+'_final_predictions.json')])


if __name__=='__main__':
    phase=sys.argv[1]
    try:
        mode,arm=phase.split('_',1);assert arm in ARMS
        ctx=Context(phase)
        assert ctx.gpu==(2 if mode=='train' else 1)
        if mode=='train':train(ctx,arm)
        else:assert mode=='eval';evaluate(ctx,arm)
    except BaseException:
        write(phase+'_FAILURE.json',dict(unix=__import__('time').time(),
            traceback=traceback.format_exc(),no_quality_claim=True))
        raise
