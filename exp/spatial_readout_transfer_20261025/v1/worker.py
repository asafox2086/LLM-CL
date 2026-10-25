"""Fixed image-only fit, paired fresh learning, and blind complete evaluations."""
from pathlib import Path
import ast,gc,hashlib,json,os,sys,time,traceback,weakref
from types import SimpleNamespace
from common import ROOT,RUN,ASSET,Context,Batcher,read,sha,write,module

def fit(ctx):
    import torch
    from safetensors import safe_open
    import deployment,interface,alignment
    alignment.configure_gpu_math();model,before=deployment.load_base(ctx.base,ctx.device,ctx.storage);ctx.bind(model)
    tower,student_cfg=deployment.load_student(RUN/'student_vision_fp16.pt',ctx.device)
    with safe_open(str(ASSET/'model/SmolVLM-500M-Instruct_a7da5b9/model.safetensors'),framework='pt',device=ctx.device) as f:
        native_weight=f.get_tensor('model.connector.modality_projection.proj.weight').to(torch.float16)
    assert tuple(native_weight.shape)==(960,12288)
    s1={r['id']:r for r in ctx.fit[1]};groups=[]
    for gid,g in ctx.split['groups'].items():
        selected=sorted(set(g['ids']).intersection(s1))
        if selected:assert not g['dev'];groups.append([gid,selected[0]])
    groups.sort(key=lambda g:hashlib.sha256(('vision-interface-20261025:'+g[0]).encode()).digest());assert groups==ctx.ready['groups'] and len(groups)==325
    batcher=Batcher(ctx,model,'FIT_ALIGN');pending={}
    def pre(layer,args):
        assert not pending
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):pending['raw']=interface.patch_features(tower,args[0])
        pending.update(raw_pixels_SHA256=ctx.storage.digest({'pixels':args[0]}),tiles=args[0].shape[0])
    def post(layer,args,value):pending['teacher']=value.detach()
    hooks=[model.core.vision_tower.register_forward_pre_hook(pre),model.core.multi_modal_projector.register_forward_hook(post)]
    def stream(selected):
        for gid,rid in selected:
            assert not pending
            with torch.inference_mode():batch,_=batcher(s1[rid],ctx.manifest['tasks'][0])
            raw=pending.pop('raw');target=pending.pop('teacher');crop_count=pending.pop('tiles');assert crop_count==raw.shape[0]
            meta=dict(pending,id=rid,group_id=gid,task=1,domain='natural');pending.clear();del batch
            yield raw,target,meta
    try:compiled,proof=alignment.fit_streams(native_weight,stream(groups[:293]),stream(groups[293:]),ctx.storage)
    finally:
        for hook in hooks:hook.remove()
    invariant=deployment.finish(model,before,ctx.storage,False)
    paths=[];payloads={}
    for arm,values in compiled.items():
        p=RUN/(arm+'_compiled.pt');torch.save({n:t.detach().cpu() for n,t in values.items()},p);paths.append(p)
        payloads[arm]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),digest=ctx.storage.digest(values))
    write('ALIGNMENT.json',proof)
    ctx.end(dict(new_generations=0,new_scores=0,FIT_backwards=0,optimizer_updates=0,teacher_decoder_requests=0,
        teacher_vision_requests=325,student_vision_requests=325,groups=groups,compiled=payloads,invariant=invariant),paths+[RUN/'ALIGNMENT.json'])

def create_initial(ctx,arm):
    import torch
    import deployment,interface
    model,base=deployment.load_base(ctx.base,ctx.device,ctx.storage)
    tower,cfg=deployment.load_student(RUN/'student_vision_fp16.pt',ctx.device)
    values=torch.load(RUN/(arm+'_compiled.pt'),map_location='cpu',weights_only=True)
    assert ctx.storage.digest(values)==read(RUN/'fit_COMPLETE.json')['compiled'][arm]['digest']
    part=interface.compile_interface(tower,values['weight'].to(ctx.device),values['bias'].to(ctx.device));interface.install(model,part)
    cp=deployment.export(model,RUN/(arm+'_initial.pt'),cfg,ctx.storage)
    return cp,cfg,base

def train(ctx,arm):
    import torch
    import deployment,adaptation,readouts
    adaptation.configure_gpu_math();initial_cp,student_cfg,original_base=create_initial(ctx,arm)
    gc.collect();torch.cuda.empty_cache()
    model,loaded=deployment.load_complete(RUN/(arm+'_initial.pt'),ctx.device,ctx.storage);ctx.bind(model)
    batcher=Batcher(ctx,model,'DEV_INITIAL')
    predictions,_=readouts.evaluate_dev(model,ctx.processor,ctx.rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop)
    write(arm+'_initial_predictions.json',predictions);initial_invariant=deployment.finish(model,loaded,ctx.storage)
    ctx.reset();method=adaptation.make_all(model,ASSET,ctx.storage)
    write(arm+'_ADAPTER_INITIAL.json',dict(coordinates=4399104,parameters=336,initialization=method.initialization,rank_plan=method.rank_plan))
    frozen=lambda:ctx.storage.digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
    frozen_before=frozen();buffers_before=ctx.storage.digest(dict(model.core.named_buffers()));adapter_before=ctx.storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    formal=module('spatial_original_train',ASSET/'exp/two_backbone_native_cl_20261017_v2/run.py')
    directory=RUN/(arm+'_training');directory.mkdir();(directory/'logs').mkdir();monitor=formal.ResourceMonitor(directory);monitor.snapshot('training_before',model=model)
    mapped={};plan=[]
    for task in ctx.manifest['tasks']:
        stage=task['task'];mapped[stage,'train']=ctx.fit[stage];mapped[stage,'dev']=ctx.dev[stage]
        order=sorted(ctx.fit[stage],key=lambda r:hashlib.sha256(f"42:{stage}:{r['id']}".encode()).digest())
        task['training_order_sha256']=hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        plan.append(dict(stage=stage,QA=len(order),updates=(len(order)+15)//16,order_sha256=task['training_order_sha256']))
    assert plan==ctx.ready['training_plan'];counts={};holder=[];raw_clip=torch.nn.utils.clip_grad_norm_
    def clip(parameters,*args,**kw):
        parameters=list(parameters);stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        grads=[p.grad for p in parameters if p.grad is not None];assert len(grads)==336
        assert all(g.is_cuda and g.dtype==torch.float32 and torch.isfinite(g).all() for g in grads)
        write(f'{arm}_grad_S{stage:02}_{counts[stage]:03}.json',dict(parameters=336,finite=True,gradient_bytes=17596416))
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
        write(f'{arm}_STAGE_{stage:02}_COMPLETE.json',dict(stage=stage,updates=counts[stage],adapter_digest=ctx.storage.digest(adapter),bytes=17596416,optimizer_state_bytes=sum(t.numel()*t.element_size() for t in states),all_optimizer_state_GPU=True,frozen_unchanged=True))
    batcher.phase='TRAIN';offset=len(batcher.records)
    namespace=dict(vars(formal),model=model,method=method,processor=ctx.processor,manifest=ctx.manifest,data=mapped,run=directory,monitor=monitor,
        args=SimpleNamespace(method='SeqFT_LoRA'),resume_obj=None,input_batch=batcher,answer_loss=ctx.rt.answer_loss,
        prepare=lambda *a:None,visual_cache_stats=ctx.fv.stats,make_optimizer=optimizer,save_stage=save_stage)
    prefix,training_ast=ctx.study.training_ast();assert training_ast==ctx.ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_stage(stage)').body,orelse=[])
    ctx.reset();torch.nn.utils.clip_grad_norm_=clip
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<unchanged87_GPUAdamW>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert sum(counts.values())==87 and not holder and len(batcher.records)-offset==1213
    assert frozen()==frozen_before and ctx.storage.digest(dict(model.core.named_buffers()))==buffers_before
    monitor.snapshot('training_after',model=model);memory=monitor.finish(model);assert not monitor.thread.is_alive()
    adapter_after=ctx.storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad});assert adapter_after!=adapter_before
    model.core.language_model.disable_input_require_grads();model.core.language_model.gradient_checkpointing_disable()
    namespace.clear();fold=adaptation.fold(model,method);cp=deployment.export(model,RUN/(arm+'_complete.pt'),student_cfg,ctx.storage)
    write(arm+'_FIT_INPUTS.json',batcher.records[offset:]);write(arm+'_TRAIN_MEMORY.json',memory)
    files=[p for p in RUN.iterdir() if p.is_file() and p.name.startswith(arm+'_') and p.suffix=='.json']+[p for p in directory.rglob('*') if p.is_file()]
    # Stage PT artifacts are separately bound to their immutable phase proof.
    files.extend(RUN/f'{arm}_adapter_S{s:02}.pt' for s in range(1,20))
    ctx.end(dict(arm=arm,new_generations=100,FIT_backwards=1213,optimizer_updates=87,coordinates=4399104,training_plan=plan,training_ast_sha256=training_ast,
        original_base=original_base,initial_checkpoint=initial_cp,initial_reload=loaded,initial_invariant=initial_invariant,
        checkpoint=cp,fold=fold,adapter_before=adapter_before,adapter_after=adapter_after,frozen_unchanged=True,buffers_unchanged=True),files)

def evaluate(ctx,arm):
    import deployment,readouts,adaptation
    adaptation.configure_gpu_math();model,before=deployment.load_complete(RUN/(arm+'_complete.pt'),ctx.device,ctx.storage);ctx.bind(model)
    assert before['digest']==read(RUN/('train_'+arm+'_COMPLETE.json'))['checkpoint']['digest']
    batcher=Batcher(ctx,model,'RESOURCE');memory,_=readouts.resource_windows(model,ctx.processor,ctx.rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop,kind='parent')
    write(arm+'_memory.json',memory);batcher.phase='DEV_FINAL'
    values,_=readouts.evaluate_dev(model,ctx.processor,ctx.rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop)
    write(arm+'_final_predictions.json',values);invariant=deployment.finish(model,before,ctx.storage)
    ctx.end(dict(arm=arm,new_generations=105,FIT_backwards=0,optimizer_updates=0,reloaded=before,invariant=invariant),[RUN/(arm+'_memory.json'),RUN/(arm+'_final_predictions.json')])

def parent_resource(ctx):
    import adaptation,readouts
    adaptation.configure_gpu_math();reference=module('spatial_parent_reference',ASSET/'exp/native_prior_learning_20261025/compatibility/deployment.py')
    model,before=reference.load_parent_static(RUN/'wide_parent_static.pt',ctx.base,ctx.storage,ctx.device);ctx.bind(model)
    batcher=Batcher(ctx,model,'RESOURCE');memory,_=readouts.resource_windows(model,ctx.processor,ctx.rows,ctx.storage,batch_provider=batcher.readout,generation_policy=ctx.stop,kind='parent',prior_windows=read(RUN/'native_memory.json'))
    assert all(a['raw_input_sha256']==b['raw_input_sha256'] for a,b in zip(memory,read(RUN/'permuted_memory.json')))
    write('parent_memory.json',memory);invariant=reference.finish_proof(model,before,ctx.storage,'parent')
    ctx.end(dict(new_generations=5,FIT_backwards=0,optimizer_updates=0,reference=before,invariant=invariant),[RUN/'parent_memory.json'])

if __name__=='__main__':
    phase=sys.argv[1]
    try:
        ctx=Context(phase)
        if phase=='fit':fit(ctx)
        elif phase.startswith('train_'):train(ctx,phase.removeprefix('train_'))
        elif phase.startswith('eval_'):evaluate(ctx,phase.removeprefix('eval_'))
        else:assert phase=='parent_resource';parent_resource(ctx)
    except BaseException:write(phase+'_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),no_quality_claim=True));raise
