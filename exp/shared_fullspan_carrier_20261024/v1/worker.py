"""Single fixed carrier construction; no continual-learning or model search."""
import ast, gc, hashlib, json, os, random, signal, sys, time, traceback
from collections import OrderedDict
from pathlib import Path
from common import ROOT,HERE,PORT,REPORT,WORK,BASE,BUDGET,CACHE_LIMIT,ready,data,loss_node,identity,sha,write

def run(gpu):
    r=ready();assert gpu in r['allowed_GPUs']
    assert not (REPORT/'STARTED.json').exists() and not WORK.exists()
    from dispatch import available
    if gpu not in available():return 75
    tick=time.monotonic();who=identity(os.getpid());assert who['uid']==os.getuid() and who['pid']==who['pgid']
    write(REPORT/'STARTED.json',dict(**who,ready_sha256=sha(REPORT/'READY.json'),started_monotonic=tick,
        started_unix=time.time(),GPU=gpu,budget_seconds=BUDGET,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    def expired(*unused):raise TimeoutError('Fixed carrier model budget expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,BUDGET-(time.monotonic()-tick)))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import torch,numpy as np
    from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
    from core import Factor,PackedFactor,full_initialize,recover_group,pack_group
    sys.path.insert(0,str(PORT))
    import runtime_utils,onevision_bridge
    from resource_monitor import ResourceMonitor
    assert runtime_utils.ROOT.resolve()==ROOT and __import__('frozen_vision').ROOT.resolve()==ROOT
    WORK.mkdir();monitor=ResourceMonitor(WORK)
    task,rows,plan=data();assert plan==r['data']
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    node,loss_sha=loss_node();assert loss_sha==r['full_loss_AST_sha256']
    namespace=dict(vars(runtime_utils));exec(compile(ast.Module(body=[node],type_ignores=[]),'<full-gold-answer>','exec'),namespace)
    full_loss=namespace['full_answer_loss']

    def tsha(t):
        v=t.detach().cpu().contiguous();h=hashlib.sha256(str((tuple(v.shape),str(v.dtype))).encode())
        raw=v.reshape(-1).view(torch.uint8).numpy()
        for i in range(0,raw.size,8*1024**2):h.update(raw[i:i+8*1024**2].tobytes())
        return h.hexdigest()

    def digest(model,non_ffn=False):
        h=hashlib.sha256()
        for kind,seq in (('p',model.named_parameters()),('b',model.named_buffers())):
            for name,value in seq:
                if non_ffn and name.startswith('model.layers.') and '.mlp.' in name:continue
                h.update(json.dumps([kind,name,list(value.shape),str(value.dtype),tsha(value)]).encode())
        return h.hexdigest()

    def layout(model):
        params=list(model.parameters());buffers=list(model.buffers())
        factors=[m for m in model.modules() if isinstance(m,PackedFactor)]
        basis={id(m.basis):m.basis for m in factors}
        packed_values=sum(m.values.numel() for m in factors)
        unique={}
        # Include tensors reachable through both the public bridge and hidden native core.
        for obj in (model,model.core):
            for value in list(obj.parameters())+list(obj.buffers()):
                st=value.untyped_storage();key=(str(value.device),st.data_ptr(),st.nbytes())
                unique[key]=st.nbytes()
        return dict(weight_values=sum(p.numel() for p in params)+packed_values,
            trainable=sum(p.numel() for p in params if p.requires_grad),all_registered_storage_bytes=sum(unique.values()),
            factor_layers=len(factors),dense_factors=sum(isinstance(m,Factor) for m in model.modules()),
            shared_bases=len(basis),shared_basis_values=sum(b.weight.numel() for b in basis.values()),
            bitmap_bytes=sum(m.bits.numel() for m in factors),packed_values=packed_values,
            other_buffer_bytes=sum(b.numel()*b.element_size() for b in buffers)-sum(m.values.numel()*m.values.element_size()+m.bits.numel() for m in factors))

    def clear_visual(model):
        model._llmcl_visual_cache=OrderedDict();model._llmcl_visual_cache_bytes=0

    def save_model(model,name):
        import cloudpickle
        clear_visual(model);model.eval()
        assert not any(p.requires_grad for p in model.parameters())
        p=WORK/(name+'.pt');assert not p.exists();begin=time.monotonic()
        with p.open('xb') as f:
            torch.save(model,f,pickle_module=cloudpickle);f.flush();os.fsync(f.fileno())
        return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),seconds=time.monotonic()-begin)

    def inputs(model,row):
        batch,details=runtime_utils.input_batch(row,task,processor,model.device)
        target=processor.tokenizer(row['answer'],add_special_tokens=False,truncation=False)['input_ids']
        assert target and processor.tokenizer.eos_token_id is not None
        target=target+[processor.tokenizer.eos_token_id]
        assert batch['input_ids'].shape[1]+len(target)<=model.config.max_position_embeddings
        assert batch['input_ids'].shape[1]+128<=model.config.max_position_embeddings
        binding={k:tsha(v) for k,v in batch.items()}
        binding['target_sha256']=hashlib.sha256(json.dumps(target).encode()).hexdigest()
        return batch,target,binding

    def diagnose(model,state):
        path=WORK/(state+'_diagnostic.jsonl');assert not path.exists();model.eval()
        with path.open('x',encoding='utf8') as f:
            for record in [p for p in plan['rows'] if p['role']=='diagnostic']:
                row=rows[record['id']]
                with torch.inference_mode():
                    batch,target,binding=inputs(model,row)
                    with torch.autocast('cuda',dtype=torch.float16):
                        loss,n=full_loss(model,batch,row['answer'],processor)
                        output=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=128,use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
                    assert bool(torch.isfinite(loss)) and n==len(target)
                    length=batch['input_ids'].shape[1];assert torch.equal(output[0,:length],batch['input_ids'][0])
                    response=output[0,length:];assert 0<len(response)<=128
                    text=processor.tokenizer.decode(response,skip_special_tokens=True)
                    result=dict(id=row['id'],NLL=float(loss),target_tokens=n,inputs=binding,prediction=text,
                        output_token_ids=response.tolist(),output_tokens=len(response),generation_cap=128,
                        last_token=int(response[-1]),stopped_eos=int(response[-1])==processor.tokenizer.eos_token_id)
                f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                del batch,output,response,loss
        return path

    try:
        for n,h in r['base_weights'].items():assert sha(ROOT/n)==h
        processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
        assert isinstance(processor.tokenizer.eos_token_id,int)
        native,loading=LlavaOnevisionForConditionalGeneration.from_pretrained(BASE,local_files_only=True,
            torch_dtype=torch.float16,attn_implementation='sdpa',output_loading_info=True)
        assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),loading
        model=onevision_bridge.OneVisionBridge(native).cuda();onevision_bridge.bind(model);del native
        for p in model.parameters():p.requires_grad_(False)
        model.eval();model._llmcl_visual_cache_hits=0;model._llmcl_visual_cache_misses=0;clear_visual(model)
        # PackedVision already accepts packed features. Keep its class method picklable.
        model._llmcl_visual_original=model.visual.forward
        assert len(model.model.layers)==24
        original_layout=layout(model);assert original_layout['weight_values']==893675552
        parent_before=digest(model);stable_before=digest(model,True)
        monitor.snapshot('parent_before',model=model)
        parent_package=save_model(model,'parent')
        parent_predictions=diagnose(model,'parent');parent_after=digest(model);assert parent_before==parent_after
        clear_visual(model)
        selected=[p for p in plan['rows'] if p['role']=='recovery'];cache=[];cache_bytes=0;active={};handles=[]
        def make_hook(index):
            def hook(module,args,output):
                nonlocal cache_bytes
                assert index not in active
                x,y=args[0],output
                assert x.shape==y.shape and x.ndim==3 and x.shape[0]==1 and x.shape[-1]==896
                assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
                x=x.detach().to('cpu',dtype=torch.float16).clone();y=y.detach().to('cpu',dtype=torch.float16).clone()
                cache_bytes+=x.untyped_storage().nbytes()+y.untyped_storage().nbytes()
                assert cache_bytes<=CACHE_LIMIT,'Fixed teacher tensor cache cap exceeded'
                active[index]=(x,y)
            return hook
        for i,layer in enumerate(model.model.layers):handles.append(layer.mlp.register_forward_hook(make_hook(i)))
        cache_index=[]
        try:
            with torch.no_grad():
                for record in selected:
                    row=rows[record['id']];active={};batch,target,binding=inputs(model,row)
                    with torch.autocast('cuda',dtype=torch.float16):loss,n=full_loss(model,batch,row['answer'],processor)
                    assert bool(torch.isfinite(loss)) and n==len(target) and set(active)==set(range(24))
                    cache.append([active[i] for i in range(24)])
                    cache_index.append(dict(id=row['id'],inputs=binding,layers=[dict(shape=list(x.shape),input_sha256=tsha(x),output_sha256=tsha(y)) for x,y in cache[-1]]))
                    del batch,loss
        finally:
            for h in handles:h.remove()
        layer=None
        assert len(cache)==32 and digest(model,True)==stable_before
        write(WORK/'cache_index.json',dict(QA=32,layer_pairs=768,tensor_payload_bytes=cache_bytes,rows=cache_index))
        clear_visual(model);active.clear();handles.clear();gc.collect();monitor.snapshot('cache_complete',model=model)
        updates=WORK/'updates.jsonl';group_results=[]
        with updates.open('x',encoding='utf8') as journal:
            for group in range(6):
                mlps=[model.model.layers[i].mlp for i in range(group*4,group*4+4)]
                assert all(m is model.core.language_model.model.layers[i].mlp for m,i in zip(mlps,range(group*4,group*4+4)))
                basis,factors=full_initialize(mlps,1152)
                group_cache=[[sample[i] for i in range(group*4,group*4+4)] for sample in cache]
                def emit(event):
                    event['group']=group;journal.write(json.dumps(event,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                result=recover_group(mlps,group_cache,torch.device('cuda'),emit)
                pack_group(mlps)
                group_results.append(dict(group=group,**result))
                del basis,factors,group_cache,mlps;gc.collect();torch.cuda.empty_cache();monitor.snapshot('group_packed',stage=group+1,model=model)
        del cache,cache_index;gc.collect();clear_visual(model);torch.cuda.empty_cache()
        assert all(not p.requires_grad for p in model.parameters())
        assert stable_before==digest(model,True)
        compact_layout=layout(model);packed_before=digest(model)
        compact_package=save_model(model,'compact')
        # Drop every live model reference before actual physical package reload.
        onevision_bridge.ACTIVE=None;namespace.pop('model',None);del model;gc.collect();torch.cuda.empty_cache()
        model=torch.load(ROOT/compact_package['path'],map_location='cuda',weights_only=False)
        onevision_bridge.bind(model);model.eval();reload_layout=layout(model);reloaded=digest(model)
        assert reload_layout==compact_layout and reloaded==packed_before
        compact_predictions=diagnose(model,'compact');packed_after=digest(model);assert packed_after==packed_before
        stable_after=digest(model,True);assert stable_after==stable_before
        monitor.done.set();monitor.thread.join(timeout=6);monitor.snapshot('complete',model=model)
        files=[parent_predictions,compact_predictions,updates,WORK/'cache_index.json',monitor.path]
        elapsed=time.monotonic()-tick;assert elapsed<=BUDGET
        assert identity(os.getpid())==who
        write(REPORT/'COMPLETE.json',dict(ready_sha256=sha(REPORT/'READY.json'),model_RUN_seconds=elapsed,end_identity=who,
            parent_digest_before=parent_before,parent_digest_after=parent_after,non_FFN_before=stable_before,non_FFN_after=stable_after,
            packed_digest_before=packed_before,packed_digest_after=packed_after,reload_digest=reloaded,
            parent_layout=original_layout,packed_layout=compact_layout,reload_layout=reload_layout,
            parent_package=parent_package,compact_package=compact_package,group_results=group_results,
            cache_QA=32,cache_layer_pairs=768,cache_payload_bytes=cache_bytes,TEST_QA=0,
            files={str(p.relative_to(ROOT)):sha(p) for p in files}))
        signal.setitimer(signal.ITIMER_REAL,0);return 0
    finally:
        monitor.done.set();monitor.thread.join(timeout=6)

if __name__=='__main__':
    try:sys.exit(run(int(sys.argv[1])))
    except Exception:
        write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),fixed_point_closed=True,no_retry=True,not_quality_negative=True));raise
