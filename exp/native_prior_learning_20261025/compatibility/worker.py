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
    for n,h in ready['original_files'].items():assert sha(RUN/n)==h,n
    for n,h in ready['environment'].items():assert sha(n)==h,n
    assert sha(RUN/'ORIGINAL_READY.json')==ready['original_ready_sha256']
    assert read(RUN/'ORIGINAL_EXIT.json')['returncode']==1
    assert "GenerationConfig' object has no attribute '_original_object_hash" in read(RUN/'ORIGINAL_FAILURE.json')['traceback']
    spec=importlib.util.spec_from_file_location('ids',ASSET/'exp/paired_image_support_20261024/v1/common.py');ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
    gpu=int(os.environ['LLMCL_GPU']);assert gpu==ready['GPU'];card=ids.inventory()[gpu];assert not card['owners'] and card['memory_MiB']<500
    who=ids.identity(os.getpid());assert who['pid']==who['pgid'];original_start=read(RUN/'ORIGINAL_STARTED.json')['unix']
    global_start=time.monotonic()-(time.time()-original_start)
    write('STARTED.json',dict(unix=time.time(),identity=who,GPU=gpu,ready_sha256=rh,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),original_first_STARTED=original_start))
    phase=None;phase_tick=None
    def expired(*args):raise TimeoutError('Original native qualification deadline expired: '+str(phase))
    signal.signal(signal.SIGALRM,expired)
    def begin(name):
        nonlocal phase,phase_tick
        phase=name;phase_tick=time.monotonic();remaining=ready['native_eval_absolute_deadline']-time.time() if name=='native_eval' else LIMITS[name]
        assert remaining>0;signal.alarm(max(1,math.ceil(remaining)))
        write(name+'_STARTED.json',dict(unix=time.time(),identity=who,ready_sha256=rh,limit=remaining,original_deadline=ready['native_eval_absolute_deadline'] if name=='native_eval' else None))
    def end(proof,files):
        value=dict(identity=who,ready_sha256=rh,files={p.relative_to(RUN).as_posix():sha(p) for p in files},**proof)
        value['seconds']=time.monotonic()-phase_tick
        if phase=='native_eval':assert time.time()<=ready['native_eval_absolute_deadline']
        assert time.time()-original_start<=5040
        write(phase+'_COMPLETE.json',value);signal.alarm(0)
    os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import random,numpy as np,torch
    from transformers import AutoProcessor,GenerationConfig
    torch.set_num_threads(8);torch.cuda.set_device(gpu)
    sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
    import study,runtime_utils as rt,onevision_bridge as bridge
    import runtime,adaptation,deployment
    spec=importlib.util.spec_from_file_location('storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
    adaptation.configure_gpu_math();random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    native_base=ASSET/'model/SmolVLM-500M-Instruct_a7da5b9';parent_base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(native_base,local_files_only=True)
    # The same processor policy is restored below immediately after load.
    original_load=deployment.load_native
    def load(*a,**kw):
        result=original_load(*a,**kw);runtime.configure_processor(processor,result[0].config,read(native_base/'processor_config.json'));return result
    deployment.load_native=load
    stop=read(RUN/'INITIAL_NATIVE.json')['policy'];cp=read(RUN/'train_COMPLETE.json')['checkpoint']
    path,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
    assert sha(path)==ready['manifest_sha256'] and sha(study.REPORT/'SPLIT.json')==ready['split_sha256']
    rows=[(t,v) for t in manifest['tasks'] for v in dev[t['task']]];assert len(rows)==100
    def native_batch(m,t,v):assert nr.model is m;return nr.input_batch(v,t,processor,m.device)
    payload_refs=[]
    def track_payload(m):payload_refs.extend(weakref.ref(x) for x in [m,*m.parameters(),*m.buffers()])
    def released_boundary(name):
        gc.collect();torch.cuda.empty_cache();assert all(ref() is None for ref in payload_refs)
        value=dict(prior_payload_released=True,tracked_references=len(payload_refs),remaining_allocated_bytes=torch.cuda.memory_allocated(gpu),same_process_CUDA_context=True,note='CUDA workspaces may persist')
        write(name,value);payload_refs.clear();return value
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
        phase_complete_sha256={n:sha(RUN/(n+'_COMPLETE.json')) for n in LIMITS},original_ready_sha256=ready['original_ready_sha256'],original_failure_preserved=True,no_retraining=True,new_generations_this_continuation=110,checkpoint=cp,process_RSS_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    assert time.monotonic()-global_start<=5040

if __name__=='__main__':
    try:main()
    except BaseException:write('FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),no_quality_claim=True));raise
