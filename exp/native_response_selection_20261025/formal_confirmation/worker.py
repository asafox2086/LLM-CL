"""One fresh full-TRAIN wide-QV learning and final-only blind TEST confirmation."""
from pathlib import Path
import ast,gc,json,sys,traceback
from types import SimpleNamespace
from common import ROOT,RUN,ASSET,Context,Batcher,read,sha,write,module

def same_receipt(actual,expected):
    keys=('layout','digest','all_buffers_digest','buffer_manifest','parameter_alias_groups','buffer_alias_groups',
        'functional_config_digest','generation_config_digest','language_generation_config_digest',
        'language_attention_classes','vision_attention_classes','response')
    for key in keys:assert actual[key]==expected[key],key

def create_initial(ctx):
    import control_deployment as deployment
    model,base=deployment.load_base(ctx.base,ctx.device,ctx.storage)
    calibration=read(RUN/'calibration_COMPLETE.json')
    assert base['digest']==calibration['initial_state_digest']
    assert base['all_buffers_digest']==calibration['all_buffers_digest']
    assert base['layout']==calibration['layout']
    prior=read(RUN/'prior_qv_train_COMPLETE.json')
    same_receipt(base,prior['original_base'])
    return deployment.export(model,RUN/'qv_initial.pt',ctx.storage),base

def train(ctx):
    arm="qv"
    import torch
    import control_deployment as deployment,control_adaptation as adaptation
    adaptation.configure_gpu_math();initial_cp,base=create_initial(ctx);gc.collect();torch.cuda.empty_cache()
    model,loaded=deployment.load_complete(RUN/(arm+'_initial.pt'),ctx.device,ctx.storage);ctx.bind(model)
    same_receipt(loaded,base)
    initial_invariant=deployment.finish(model,loaded,ctx.storage)
    # Both paired arms reset42 immediately before the same BudgetQV constructor.
    ctx.reset();method=adaptation.make_wide(model,ASSET,ctx.storage)
    prior_initial=read(RUN/'prior_qv_ADAPTER_INITIAL.json')
    assert method.initialization==prior_initial['initialization'] and method.rank_plan==prior_initial['rank_plan']
    write('qv_ADAPTER_INITIAL.json',dict(coordinates=4399104,parameters=96,initialization=method.initialization,rank_plan=method.rank_plan,
        paired_initialization_exact=True,no_response_module=True))
    batcher=Batcher(ctx,model,'TRAIN')
    frozen=lambda:ctx.storage.digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
    frozen_before=frozen();buffers_before=ctx.storage.digest(dict(model.core.named_buffers()));adapter_before=ctx.storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    formal=module('response_original_train',ASSET/'exp/two_backbone_native_cl_20261017_v2/run.py')
    directory=RUN/(arm+'_training');directory.mkdir();(directory/'logs').mkdir();monitor=formal.ResourceMonitor(directory);monitor.snapshot('training_before',model=model)
    mapped={};plan=[]
    for task in ctx.manifest['tasks']:
        stage=task['task'];mapped[stage,'train']=ctx.fit[stage];mapped[stage,'dev']=ctx.dev[stage]
        order=sorted(ctx.fit[stage],key=lambda r:__import__('hashlib').sha256(f"42:{stage}:{r['id']}".encode()).digest())
        task['training_order_sha256']=__import__('hashlib').sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        plan.append(dict(stage=stage,QA=len(order),updates=(len(order)+15)//16,order_sha256=task['training_order_sha256']))
    assert plan==ctx.ready['training_plan'];counts={};holder=[];raw_clip=torch.nn.utils.clip_grad_norm_
    def clip(parameters,*args,**kw):
        parameters=list(parameters);stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        grads=[p.grad for p in parameters if p.grad is not None];assert len(grads)==96
        assert all(g.is_cuda and g.dtype==torch.float32 and torch.isfinite(g).all() for g in grads)
        assert sum(g.numel()*g.element_size() for g in grads)==17596416
        write(f'{arm}_grad_S{stage:02}_{counts[stage]:03}.json',dict(parameters=96,finite=True,gradient_bytes=17596416))
        return raw_clip(parameters,*args,**kw)
    def optimizer(m):
        assert not holder
        opt=torch.optim.AdamW([p for p in m.model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False,capturable=True)
        holder.append(opt);return opt
    def save_stage(stage):
        assert frozen()==frozen_before and ctx.storage.digest(dict(model.core.named_buffers()))==buffers_before
        opt=holder.pop();states=[t for state in opt.state.values() for t in state.values() if isinstance(t,torch.Tensor)]
        assert states and all(t.is_cuda and torch.isfinite(t).all() for t in states)
        assert sum(t.numel()*t.element_size() for t in states)==35193216
        adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad};p=RUN/f'{arm}_adapter_S{stage:02}.pt';torch.save(adapter,p)
        write(f'{arm}_STAGE_{stage:02}_COMPLETE.json',dict(stage=stage,updates=counts[stage],adapter_digest=ctx.storage.digest(adapter),bytes=17596416,
            optimizer_state_bytes=sum(t.numel()*t.element_size() for t in states),all_optimizer_state_GPU=True,frozen_unchanged=True))
    batcher.phase='TRAIN';offset=len(batcher.records)
    namespace=dict(vars(formal),model=model,method=method,processor=ctx.processor,manifest=ctx.manifest,data=mapped,run=directory,monitor=monitor,
        args=SimpleNamespace(method='SeqFT_LoRA'),resume_obj=None,input_batch=batcher,answer_loss=ctx.rt.answer_loss,
        prepare=lambda *a:None,visual_cache_stats=ctx.fv.stats,make_optimizer=optimizer,save_stage=save_stage)
    prefix,training_ast=ctx.study.training_ast();assert training_ast==ctx.ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_stage(stage)').body,orelse=[])
    ctx.reset();torch.nn.utils.clip_grad_norm_=clip
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<unchanged93_formal_GPUAdamW>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert sum(counts.values())==93 and not holder and len(batcher.records)-offset==1313
    assert frozen()==frozen_before and ctx.storage.digest(dict(model.core.named_buffers()))==buffers_before
    monitor.snapshot('training_after',model=model);memory=monitor.finish(model);assert not monitor.thread.is_alive()
    adapter_after=ctx.storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad});assert adapter_after!=adapter_before
    model.core.language_model.disable_input_require_grads();model.core.language_model.gradient_checkpointing_disable()
    namespace.clear();fold=adaptation.fold(model,method)
    cp=deployment.export(model,RUN/(arm+'_complete.pt'),ctx.storage)
    fit_inputs=batcher.records[offset:]
    assert [r['id'] for r in fit_inputs]==[r['id'] for t in ctx.manifest['tasks'] for r in sorted(ctx.fit[t['task']],key=lambda r:__import__('hashlib').sha256(f"42:{t['task']}:{r['id']}".encode()).digest())]
    write(arm+'_TRAIN_INPUTS.json',fit_inputs);write(arm+'_TRAIN_MEMORY.json',memory)
    files=[p for p in RUN.iterdir() if p.is_file() and p.name.startswith(arm+'_') and p.suffix=='.json']+[p for p in directory.rglob('*') if p.is_file()]
    files.extend(RUN/f'{arm}_adapter_S{s:02}.pt' for s in range(1,20));files.extend([RUN/(arm+'_initial.pt'),RUN/(arm+'_complete.pt')])
    ctx.end(dict(arm=arm,new_generations=0,TRAIN_backwards=1313,optimizer_updates=93,coordinates=4399104,training_plan=plan,training_ast_sha256=training_ast,
        original_base=base,initial_checkpoint=initial_cp,initial_reload=loaded,initial_invariant=initial_invariant,
        checkpoint=cp,fold=fold,paired_initialization_exact=True,original_full_TRAIN_order_exact=True,no_response_module=True,adapter_before=adapter_before,adapter_after=adapter_after,frozen_unchanged=True,buffers_unchanged=True),files)

def resource(ctx):
    import control_deployment as deployment,readouts,control_adaptation as adaptation
    adaptation.configure_gpu_math();model,before=deployment.load_complete(RUN/'qv_complete.pt',ctx.device,ctx.storage);ctx.bind(model)
    train=read(RUN/'train_COMPLETE.json');same_receipt(before,train['checkpoint'])
    batcher=Batcher(ctx,model,'RESOURCE')
    memory,_=readouts.resource_windows(model,ctx.processor,ctx.rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop)
    prior=read(RUN/'prior_PAS_memory.json');assert len(prior)==len(memory)==5
    for old,new in zip(prior,memory):
        for key in ('id','domain','raw_input_sha256','transport_sha256','prepared_sha256','input_shape','input_tensor_shapes','new_tokens'):
            assert old[key]==new[key],key
    write('resource_memory.json',memory);invariant=deployment.finish(model,before,ctx.storage)
    ctx.end(dict(new_generations=5,forced_new_tokens=160,TRAIN_backwards=0,optimizer_updates=0,TEST_QA=0,reloaded=before,invariant=invariant,
        same_PAS_resource_inputs=True,checkpoint_source_sha256=train['checkpoint']['sha256']),[RUN/'resource_memory.json'])

def test(ctx,shard):
    import control_deployment as deployment,readouts,control_adaptation as adaptation
    from common import shard_for
    adaptation.configure_gpu_math();model,before=deployment.load_complete(RUN/'qv_complete.pt',ctx.device,ctx.storage);ctx.bind(model)
    train=read(RUN/'train_COMPLETE.json');same_receipt(before,train['checkpoint'])
    ctx.reset();batcher=Batcher(ctx,model,'TEST_FINAL')
    rows=[(t,r) for t in ctx.manifest['tasks'] for r in ctx.data[t['task'],'test'] if shard_for(t['task'],r['id'])==shard]
    assert rows and len({r['id'] for t,r in rows})==len(rows)
    path,proof=readouts.evaluate_test_shard(model,ctx.processor,rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop,
        output=RUN/f'test_{shard}_predictions.jsonl')
    invariant=deployment.finish(model,before,ctx.storage)
    import hashlib
    ctx.end(dict(shard=shard,new_generations=len(rows),TEST_QA=len(rows),TRAIN_backwards=0,optimizer_updates=0,new_scores=0,
        reloaded=before,invariant=invariant,checkpoint_source_sha256=train['checkpoint']['sha256'],test_proof=proof,
        assigned_ID_sha256=hashlib.sha256(''.join(f"{t['task']}:{r['id']}\n" for t,r in rows).encode()).hexdigest(),
        task_counts={str(t['task']):sum(tt['task']==t['task'] for tt,r in rows) for t in ctx.manifest['tasks']}),[path])

if __name__=='__main__':
    phase=sys.argv[1]
    try:
        ctx=Context(phase)
        if phase=='train':train(ctx)
        elif phase=='resource':resource(ctx)
        else:assert phase in {f'test_{i}' for i in range(4)};test(ctx,int(phase[-1]))
    except BaseException:write(phase+'_FAILURE.json',dict(unix=__import__('time').time(),traceback=traceback.format_exc(),no_quality_claim=True));raise
