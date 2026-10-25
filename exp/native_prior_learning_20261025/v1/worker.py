"""One GPU worker, four fixed phases, no scoring or outcome-dependent branch."""
from pathlib import Path
import ast,gc,hashlib,importlib.util,json,math,os,signal,sys,time,traceback,weakref
from types import SimpleNamespace
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=Path(os.environ['LLMCL_ASSET'])
LIMITS={'initial':600,'train':3600,'native_eval':600,'parent_resource':240}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    with (RUN/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def main():
    ready=read(RUN/'READY.json');rh=sha(RUN/'READY.json')
    for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
    for n,h in ready['runtime_files'].items():assert sha(ASSET/n)==h,n
    for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
    for n,h in ready['environment'].items():assert sha(n)==h,n
    assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
    spec=importlib.util.spec_from_file_location('ids',ASSET/'exp/paired_image_support_20261024/v1/common.py');ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
    gpu=int(os.environ['LLMCL_GPU']);card=ids.inventory()[gpu];assert not card['owners'] and card['memory_MiB']<500 and card['utilization']<10
    who=ids.identity(os.getpid());assert who['pid']==who['pgid'];global_start=time.monotonic()
    write('STARTED.json',dict(unix=time.time(),identity=who,GPU=gpu,ready_sha256=rh,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    phase_tick=None;phase=None
    def expired(*args):raise TimeoutError('Fixed native qualification budget expired: '+str(phase))
    signal.signal(signal.SIGALRM,expired)
    def begin(name):
        nonlocal phase_tick,phase
        phase=name;phase_tick=time.monotonic();signal.alarm(LIMITS[name])
        write(name+'_STARTED.json',dict(unix=time.time(),identity=who,ready_sha256=rh,limit=LIMITS[name]))
    def end(proof,files):
        value=dict(identity=who,ready_sha256=rh,files={p.relative_to(RUN).as_posix():sha(p) for p in files},**proof)
        value['seconds']=time.monotonic()-phase_tick
        write(phase+'_COMPLETE.json',value);signal.alarm(0)
    begin('initial')
    os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import random,numpy as np,torch
    from transformers import AutoProcessor,GenerationConfig
    torch.set_num_threads(8);torch.cuda.set_device(gpu)
    sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'))
    sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
    import study,runtime_utils as rt,onevision_bridge as bridge
    import runtime,adaptation,deployment
    payload_refs=[];optimizer_refs=[]
    def track_payload(m):
        payload_refs.extend(weakref.ref(x) for x in [m,*m.parameters(),*m.buffers()])
    def released_boundary(name):
        gc.collect();torch.cuda.empty_cache()
        assert all(ref() is None for ref in payload_refs+optimizer_refs),'Prior model or optimizer payload retained'
        value=dict(prior_payload_released=True,tracked_references=len(payload_refs)+len(optimizer_refs),
            remaining_allocated_bytes=torch.cuda.memory_allocated(gpu),same_process_CUDA_context=True,
            note='CUDA library workspaces may persist; no arbitrary zero-allocation threshold.')
        write(name,value);payload_refs.clear();optimizer_refs.clear()
        return value
    spec=importlib.util.spec_from_file_location('storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
    math_receipt=adaptation.configure_gpu_math()
    def reset():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    reset();native_base=ASSET/'model/SmolVLM-500M-Instruct_a7da5b9';parent_base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(native_base,local_files_only=True)
    model,initial=deployment.load_native(native_base,storage,f'cuda:{gpu}')
    compat=runtime.configure_processor(processor,model.config,read(native_base/'processor_config.json'))
    stop=runtime.policy(processor,GenerationConfig.from_pretrained(native_base,local_files_only=True));stop['context_limit']=8192
    path,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
    assert sha(path)==ready['manifest_sha256'] and sha(study.REPORT/'SPLIT.json')==ready['split_sha256']
    assert split==read(study.REPORT/'SPLIT.json')
    rows=[(t,r) for t in manifest['tasks'] for r in dev[t['task']]];assert len(rows)==100
    nr=runtime.NativeRuntime(processor,model,RUN,stop,image_SHA256=split['image_sha256'],asset_root=ASSET)
    def native_batch(m,t,r):
        assert nr.model is m
        return nr.input_batch(r,t,processor,m.device)
    nr.phase='DEV'
    values,ep=deployment.evaluate_dev(model,processor,rows,storage,batch_provider=native_batch,generation_policy=stop)
    write('initial_predictions.json',values);write('INITIAL_NATIVE.json',dict(model=initial,processor=compat,policy=stop,math=math_receipt))
    invariant=deployment.finish_proof(model,initial,storage)
    end(dict(new_generations=100,FIT_model_requests=0,backwards=0,optimizer_updates=0,invariants=invariant),[RUN/'initial_predictions.json',RUN/'INITIAL_NATIVE.json'])

    begin('train');reset();method=adaptation.make_qv(model,storage)
    track_payload(model);optimizer_holder=[]
    frozen=lambda:storage.digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
    frozen_before=frozen();buffer_before=storage.digest(dict(model.named_buffers()))
    adapter_before=storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})
    write('INITIAL_ADAPTER.json',dict(initialization=method.initialization,rank_plan=method.rank_plan,
        digest=adapter_before,coordinates=819200,parameters=128,frozen_digest=frozen_before,buffers_digest=buffer_before))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.model.text_model.enable_input_require_grads()
    spec=importlib.util.spec_from_file_location('formal',ASSET/'exp/two_backbone_native_cl_20261017_v2/run.py');formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)
    tr=RUN/'training';tr.mkdir();(tr/'logs').mkdir();monitor=formal.ResourceMonitor(tr)
    monitor.snapshot('training_before',model=model)
    mapped={};plan=[]
    for t in manifest['tasks']:
        stage=t['task'];mapped[stage,'train']=fit[stage];mapped[stage,'dev']=dev[stage]
        order=sorted(fit[stage],key=lambda r:hashlib.sha256(f"42:{stage}:{r['id']}".encode()).digest())
        t['training_order_sha256']=hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        plan.append(dict(stage=stage,QA=len(order),updates=(len(order)+15)//16,order_sha256=t['training_order_sha256']))
    assert plan==ready['training_plan'] and sum(p['QA'] for p in plan)==1213 and sum(p['updates'] for p in plan)==87
    counts={};raw_clip=torch.nn.utils.clip_grad_norm_
    def clip(parameters,*args,**kwargs):
        parameters=list(parameters);stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        grads={n:p.grad for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert len(grads)==128 and all(g.is_cuda and g.dtype==torch.float32 and torch.isfinite(g).all() for g in grads.values())
        write(f'grad_S{stage:02}_{counts[stage]:03}.json',dict(parameters=128,gradient_bytes=3276800,finite=True))
        return raw_clip(parameters,*args,**kwargs)
    def training_input(*args,**kwargs):nr.phase='TRAIN';return nr.input_batch(*args,**kwargs)
    def optimizer(m):
        assert not optimizer_holder
        opt=torch.optim.AdamW([p for p in m.model.parameters() if p.requires_grad],lr=1e-4,
            betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False,capturable=True)
        optimizer_holder.append(opt);optimizer_refs.append(weakref.ref(opt));return opt
    def save_stage(stage):
        assert frozen()==frozen_before and storage.digest(dict(model.named_buffers()))==buffer_before
        assert len(optimizer_holder)==1
        opt=optimizer_holder.pop();states=[x for state in opt.state.values() for x in state.values() if isinstance(x,torch.Tensor)]
        assert states and all(x.is_cuda and torch.isfinite(x).all() for x in states)
        optimizer_refs.extend(weakref.ref(x) for x in states)
        adapter={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
        cp=RUN/f'adapter_S{stage:02}.pt';torch.save(adapter,cp)
        write(f'STAGE_{stage:02}_COMPLETE.json',dict(stage=stage,updates=counts[stage],adapter_digest=storage.digest(adapter),
            bytes=3276800,optimizer_state_bytes=sum(x.numel()*x.element_size() for x in states),all_optimizer_state_GPU=True,frozen_unchanged=True))
    namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=manifest,data=mapped,run=tr,monitor=monitor,
        args=SimpleNamespace(method='SeqFT_LoRA'),resume_obj=None,input_batch=training_input,answer_loss=nr.answer_loss,
        prepare=lambda *a:None,visual_cache_stats=lambda *a:{},make_optimizer=optimizer,save_stage=save_stage)
    prefix,training_ast=study.training_ast();assert training_ast==ready['training_ast_sha256']
    node=ast.For(target=ast.Name(id='stage',ctx=ast.Store()),iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),body=prefix+ast.parse('save_stage(stage)').body,orelse=[])
    reset();torch.nn.utils.clip_grad_norm_=clip
    try:exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<original87_GPUAdamW>', 'exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=raw_clip
    assert sum(counts.values())==87 and len(nr.requests)==1213 and all(v['kind']=='answer_NLL' for v in nr.requests)
    assert frozen()==frozen_before and storage.digest(dict(model.named_buffers()))==buffer_before
    monitor.snapshot('training_after',model=model)
    training_memory=monitor.finish(model)
    assert not monitor.thread.is_alive()
    training_memory['note']='Whole training process, including temporary stage optimizer retention for receipts. Not matched inference or a CPU memory advantage.'
    (tr/'memory_summary.json').write_text(json.dumps(training_memory,indent=2)+'\n')
    adapter_after=storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad});assert adapter_after!=adapter_before
    model.model.text_model.disable_input_require_grads();model.gradient_checkpointing_disable()
    fold=adaptation.fold(model,method);cp=deployment.export_native(model,RUN/'native_complete.pt',storage)
    track_payload(model)
    write('FIT_INPUTS.json',nr.input_records[100:]);assert len(nr.input_records[100:])==1213
    files=[RUN/'INITIAL_ADAPTER.json',RUN/'FIT_INPUTS.json']+[p for p in RUN.iterdir() if p.name.startswith(('grad_','adapter_S','STAGE_'))]+[p for p in tr.rglob('*') if p.is_file()]
    train_proof=dict(new_generations=0,FIT_model_requests=1213,backwards=1213,optimizer_updates=87,training_plan=plan,training_ast_sha256=training_ast,
        adapter_before=adapter_before,adapter_after=adapter_after,frozen_unchanged=True,buffers_unchanged=True,fold=fold,checkpoint=cp,coordinates=819200,training_memory=training_memory)
    assert not optimizer_holder
    namespace.clear();del namespace,nr,method,model;bridge.ACTIVE=None;bridge.LAST_NATIVE=None;gc.collect();torch.cuda.empty_cache()
    train_proof['fresh_boundary']=released_boundary('FRESH_TRAIN_BOUNDARY.json')
    end(train_proof,files+[RUN/'FRESH_TRAIN_BOUNDARY.json'])

    begin('native_eval')
    model,loaded=deployment.load_native(native_base,storage,f'cuda:{gpu}',package_path=RUN/'native_complete.pt',expected_digest=cp['digest'])
    track_payload(model)
    nr=runtime.NativeRuntime(processor,model,RUN,stop,image_SHA256=split['image_sha256'],asset_root=ASSET)
    nr.phase='RESOURCE'
    mem,mp=deployment.resource_windows(model,processor,rows,storage,batch_provider=native_batch,generation_policy=stop)
    write('native_memory.json',mem);nr.phase='DEV'
    values,ep=deployment.evaluate_dev(model,processor,rows,storage,batch_provider=native_batch,generation_policy=stop)
    write('final_predictions.json',values)
    invariant=deployment.finish_proof(model,loaded,storage)
    del model,nr;bridge.ACTIVE=None;bridge.LAST_NATIVE=None;gc.collect();torch.cuda.empty_cache()
    boundary=released_boundary('FRESH_EVAL_BOUNDARY.json')
    end(dict(new_generations=105,FIT_model_requests=0,backwards=0,optimizer_updates=0,loaded=loaded,invariants=invariant,fresh_boundary=boundary),[RUN/'native_memory.json',RUN/'final_predictions.json',RUN/'FRESH_EVAL_BOUNDARY.json'])

    begin('parent_resource')
    model,loaded=deployment.load_parent_static(RUN/'wide_parent_static.pt',parent_base,storage,f'cuda:{gpu}')
    parent_processor=AutoProcessor.from_pretrained(parent_base,local_files_only=True)
    from retrieval import retrieve
    def parent_batch(m,t,r):
        for name in r['images']:assert sha(ASSET/name)==split['image_sha256'][name]
        bridge.bind(m);batch,details=rt.input_batch(r,t,parent_processor,m.device)
        context=r.get('context','');indexes=[]
        if len(context)>6000:
            chunks=[context[i:i+1000] for i in range(0,len(context),800)];indexes=sorted(retrieve(r['question'],chunks,5));context='\n'.join(chunks[i] for i in indexes)
        assert details['retrieved_chunk_indexes']==indexes
        contract=dict(images=[dict(path=n,sha256=split['image_sha256'][n]) for n in r['images']],instruction=t['instruction'],question=r['question'],original_context=r.get('context',''),resolved_context=context,retrieved_chunk_indexes=indexes,image_shapes=details['image_shapes'],common_image_area_cap=401408)
        details.update(raw_input_sha256=hashlib.sha256(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),image_SHA256=contract['images'])
        return batch,details
    pp=dict(generation_stop_ids=parent_processor.tokenizer.eos_token_id,pad_id=parent_processor.tokenizer.pad_token_id,context_limit=None)
    parent_mem,mp=deployment.resource_windows(model,parent_processor,rows,storage,batch_provider=parent_batch,generation_policy=pp,kind='parent',prior_windows=mem)
    write('parent_memory.json',parent_mem);invariant=deployment.finish_proof(model,loaded,storage,'parent')
    end(dict(new_generations=5,FIT_model_requests=0,backwards=0,optimizer_updates=0,loaded=loaded,invariants=invariant),[RUN/'parent_memory.json'])
    import resource
    write('COMPLETE.json',dict(complete=True,identity=who,seconds=time.monotonic()-global_start,ready_sha256=rh,
        new_generations=210,FIT_model_requests=1213,backwards=1213,optimizer_updates=87,TEST_QA=0,new_scores=0,
        phase_complete_sha256={n:sha(RUN/(n+'_COMPLETE.json')) for n in LIMITS},checkpoint=cp,process_RSS_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    assert time.monotonic()-global_start<=5040

if __name__=='__main__':
    try:main()
    except BaseException:write('FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),no_quality_claim=True));raise
