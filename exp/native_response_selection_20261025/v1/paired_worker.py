"""Fixed paired learning under the original87-update AST; blind DEV outputs."""
from pathlib import Path
import ast,gc,json,sys,traceback
from types import SimpleNamespace
from paired_common import ROOT,RUN,ASSET,Context,Batcher,read,sha,write,module

def beta_values(model):return [float(layer.mlp.beta.detach()) for layer in model.model.layers]

def create_initial(ctx,arm):
    import paired_deployment as deployment,response
    model,base=deployment.load_base(ctx.base,ctx.device,ctx.storage)
    calibration=read(RUN/'calibration_COMPLETE.json')
    response.install(model,arm,calibration,ctx.device);model.requires_grad_(False).eval()
    assert all(v==0 for v in beta_values(model))
    return deployment.export(model,RUN/(arm+'_initial.pt'),ctx.storage),base

def train(ctx,arm):
    import torch
    import paired_deployment as deployment,paired_adaptation as adaptation,paired_readouts as readouts,response
    adaptation.configure_gpu_math();initial_cp,base=create_initial(ctx,arm);gc.collect();torch.cuda.empty_cache()
    model,loaded=deployment.load_complete(RUN/(arm+'_initial.pt'),ctx.device,ctx.storage);ctx.bind(model)
    batcher=Batcher(ctx,model,'DEV_INITIAL')
    predictions,_=readouts.evaluate_dev(model,ctx.processor,ctx.rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop)
    write(arm+'_initial_predictions.json',predictions);initial_invariant=deployment.finish(model,loaded,ctx.storage)
    ctx.reset();method=adaptation.make_wide(model,ASSET,ctx.storage)
    write(arm+'_ADAPTER_INITIAL.json',dict(coordinates=4399128,parameters=120,initialization=method.initialization,rank_plan=method.rank_plan,beta=beta_values(model),response=response.metadata(model)))
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
        grads=[p.grad for p in parameters if p.grad is not None];assert len(grads)==120
        assert all(g.is_cuda and g.dtype==torch.float32 and torch.isfinite(g).all() for g in grads)
        assert sum(g.numel()*g.element_size() for g in grads)==17596512
        write(f'{arm}_grad_S{stage:02}_{counts[stage]:03}.json',dict(parameters=120,finite=True,gradient_bytes=17596512))
        return raw_clip(parameters,*args,**kw)
    def optimizer(m):
        assert not holder
        opt=torch.optim.AdamW([p for p in m.model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False,capturable=True)
        holder.append(opt);return opt
    def save_stage(stage):
        assert frozen()==frozen_before and ctx.storage.digest(dict(model.core.named_buffers()))==buffers_before
        opt=holder.pop();states=[t for state in opt.state.values() for t in state.values() if isinstance(t,torch.Tensor)]
        assert states and all(t.is_cuda and torch.isfinite(t).all() for t in states)
        adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad};p=RUN/f'{arm}_adapter_S{stage:02}.pt';torch.save(adapter,p)
        write(f'{arm}_STAGE_{stage:02}_COMPLETE.json',dict(stage=stage,updates=counts[stage],adapter_digest=ctx.storage.digest(adapter),bytes=17596512,
            optimizer_state_bytes=sum(t.numel()*t.element_size() for t in states),all_optimizer_state_GPU=True,frozen_unchanged=True,beta=beta_values(model)))
    batcher.phase='TRAIN';offset=len(batcher.records)
    namespace=dict(vars(formal),model=model,method=method,processor=ctx.processor,manifest=ctx.manifest,data=mapped,run=directory,monitor=monitor,
        args=SimpleNamespace(method='SeqFT_LoRA'),resume_obj=None,input_batch=batcher,answer_loss=ctx.rt.answer_loss,
        prepare=lambda *a:None,visual_cache_stats=ctx.fv.stats,make_optimizer=optimizer,save_stage=save_stage)
    prefix,training_ast=ctx.study.training_ast();assert training_ast==ctx.ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_stage(stage)').body,orelse=[])
    ctx.reset();torch.nn.utils.clip_grad_norm_=clip
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<unchanged87_response_GPUAdamW>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert sum(counts.values())==87 and not holder and len(batcher.records)-offset==1213
    assert frozen()==frozen_before and ctx.storage.digest(dict(model.core.named_buffers()))==buffers_before
    monitor.snapshot('training_after',model=model);memory=monitor.finish(model);assert not monitor.thread.is_alive()
    adapter_after=ctx.storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad});assert adapter_after!=adapter_before
    learned_beta=beta_values(model);expected=[]
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        for layer in model.model.layers:
            mlp=layer.mlp;value=mlp.beta.detach().clone()
            if arm=='static':value=value*mlp.calibration_scale
            expected.append(value)
        expected_digest=ctx.storage.digest({'effective_beta':torch.stack(expected)})
    model.core.language_model.disable_input_require_grads();model.core.language_model.gradient_checkpointing_disable()
    namespace.clear();fold=adaptation.fold(model,method);compiled=response.compile_final(model)
    actual_digest=ctx.storage.digest({'effective_beta':torch.stack([l.mlp.beta for l in model.model.layers])});assert actual_digest==expected_digest
    write(arm+'_COMPILE.json',dict(learned_beta=learned_beta,final_beta=beta_values(model),effective_beta_digest=actual_digest,expected_effective_beta_digest=expected_digest,metadata=compiled))
    cp=deployment.export(model,RUN/(arm+'_complete.pt'),ctx.storage)
    write(arm+'_FIT_INPUTS.json',batcher.records[offset:]);write(arm+'_TRAIN_MEMORY.json',memory)
    files=[p for p in RUN.iterdir() if p.is_file() and p.name.startswith(arm+'_') and p.suffix=='.json']+[p for p in directory.rglob('*') if p.is_file()]
    files.extend(RUN/f'{arm}_adapter_S{s:02}.pt' for s in range(1,20));files.extend([RUN/(arm+'_initial.pt'),RUN/(arm+'_complete.pt')])
    ctx.end(dict(arm=arm,new_generations=100,FIT_backwards=1213,optimizer_updates=87,coordinates=4399128,training_plan=plan,training_ast_sha256=training_ast,
        original_base=base,initial_checkpoint=initial_cp,initial_reload=loaded,initial_invariant=initial_invariant,
        checkpoint=cp,fold=fold,compile=compiled,adapter_before=adapter_before,adapter_after=adapter_after,frozen_unchanged=True,buffers_unchanged=True),files)

def evaluate(ctx,arm):
    import paired_deployment as deployment,paired_readouts as readouts,paired_adaptation as adaptation
    adaptation.configure_gpu_math();model,before=deployment.load_complete(RUN/(arm+'_complete.pt'),ctx.device,ctx.storage);ctx.bind(model)
    assert before['digest']==read(RUN/('train_'+arm+'_COMPLETE.json'))['checkpoint']['digest']
    batcher=Batcher(ctx,model,'RESOURCE');memory,_=readouts.resource_windows(model,ctx.processor,ctx.rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop)
    write(arm+'_memory.json',memory);batcher.phase='DEV_FINAL'
    values,_=readouts.evaluate_dev(model,ctx.processor,ctx.rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop)
    write(arm+'_final_predictions.json',values);invariant=deployment.finish(model,before,ctx.storage)
    ctx.end(dict(arm=arm,new_generations=105,FIT_backwards=0,optimizer_updates=0,reloaded=before,invariant=invariant),[RUN/(arm+'_memory.json'),RUN/(arm+'_final_predictions.json')])

if __name__=='__main__':
    phase=sys.argv[1]
    try:
        ctx=Context(phase)
        if phase.startswith('train_'):train(ctx,phase.removeprefix('train_'))
        else:assert phase.startswith('eval_');evaluate(ctx,phase.removeprefix('eval_'))
    except BaseException:write(phase+'_FAILURE.json',dict(unix=__import__('time').time(),traceback=traceback.format_exc(),no_quality_claim=True));raise
