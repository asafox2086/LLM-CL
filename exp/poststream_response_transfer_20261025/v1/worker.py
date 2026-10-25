"""One fixed GPU-only model phase; no QA scoring or adaptive retries."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,signal,sys,time,traceback
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=Path(os.environ['LLMCL_ASSET'])
LIMITS={'CE_train':1200,'KD_train':1800,'CE_eval':450,'KD_eval':450}
DOMAINS=('natural','medical','geography','finance','traffic')

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    with (RUN/n).open('x',encoding='utf8') as f:
        json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def main(case):
    ready=read(RUN/'READY.json');rh=sha(RUN/'READY.json')
    for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
    for n,h in ready['runtime_files'].items():assert sha(ASSET/n)==h,n
    for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
    assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
    spec=importlib.util.spec_from_file_location('ids_only',ASSET/'exp/paired_image_support_20261024/v1/common.py')
    ids=importlib.util.module_from_spec(spec);spec.loader.exec_module(ids)
    student_gpu=int(os.environ['LLMCL_STUDENT_GPU']);teacher_gpu=int(os.environ['LLMCL_TEACHER_GPU'])
    cards=ids.inventory();required=[student_gpu]+([teacher_gpu] if case=='KD_train' else [])
    assert student_gpu!=teacher_gpu
    for gpu in required:assert not cards[gpu]['owners'] and cards[gpu]['memory_MiB']<500 and cards[gpu]['utilization']<10
    who=ids.identity(os.getpid());assert who['pid']==who['pgid'];tick=time.monotonic()
    write(case+'_STARTED.json',dict(unix=time.time(),identity=who,student_GPU=student_gpu,
        teacher_GPU=teacher_gpu if case=='KD_train' else None,ready_sha256=rh,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    def expired(*args):raise TimeoutError('Fixed GPU response-transfer phase budget expired')
    signal.signal(signal.SIGALRM,expired);signal.alarm(LIMITS[case])
    os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import random
    import numpy as np
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(8);torch.cuda.set_device(student_gpu)
    sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'))
    sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
    import onevision_bridge as bridge,frozen_vision as fv,runtime_utils as rt,study
    import training,deployment
    spec=importlib.util.spec_from_file_location('storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py')
    storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
    math=training.configure_gpu_math();fv.LIMIT=0
    for gpu in required:
        with torch.cuda.device(gpu):torch.cuda.reset_peak_memory_stats(gpu)
    base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    mp,manifest,original=study.original_train();split,fit,dev=study.split_train(manifest,original)
    assert sha(mp)==ready['manifest_sha256'] and sha(study.REPORT/'SPLIT.json')==ready['split_sha256']
    assert split==read(study.REPORT/'SPLIT.json')
    tasks={t['task']:t for t in manifest['tasks']}
    fitrows={r['id']:(t,r) for k,t in tasks.items() for r in fit[k]}
    devrows=[(t,r) for k,t in tasks.items() for r in dev[k]]
    assert len(fitrows)==1213 and len(devrows)==100
    memory_meta=read(RUN/'MEMORY.json');memory=[fitrows[x['id']] for x in memory_meta['order']]
    assert all((t['task'],t['domain'])==(x['task'],x['domain']) for (t,_),x in zip(memory,memory_meta['order']))
    for d in DOMAINS:
        ordered=sorted((tr for tr in fitrows.values() if tr[0]['domain']==d),
            key=lambda tr:(hashlib.sha256(('response-memory-20261025:'+tr[1]['id']).encode()).digest(),tr[1]['id']))[:32]
        assert [r['id'] for t,r in memory if t['domain']==d]==[r['id'] for _,r in ordered]
    prior=read(RUN/'prior_COMPLETE.json');prior_preds=read(RUN/'prior_parent_predictions.json')
    assert read(RUN/'prior_VERIFIED.json')['source_COMPLETE_sha256']==sha(RUN/'prior_COMPLETE.json')
    checked=set()
    for t,row in memory+devrows:
        for image in row['images']:
            if image not in checked:assert sha(ASSET/image)==split['image_sha256'][image];checked.add(image)
    def batch_provider(model,task,row):
        bridge.bind(model);batch,_=rt.input_batch(row,task,processor,model.device)
        details=dict(id=row['id'],task=task['task'],domain=task['domain'],
            prepared_sha256=storage.digest(batch),input_sha256=storage.digest(batch),
            transport_sha256=storage.digest({k:v for k,v in batch.items() if k!='pixel_values'}),
            packed_tokens=len(batch['pixel_values']),prompt_tokens=batch['input_ids'].shape[1])
        assert model._llmcl_visual_cache_bytes==0
        return batch,details
    def reset():
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    def event(name,value):
        with (RUN/name).open('a',encoding='utf8') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
    arm=case[:2];files=[]
    if case.endswith('_train'):
        reset()
        model,initial=deployment.load_static(RUN/'pruned_complete.pt',base,storage,
            device=f'cuda:{student_gpu}',expected_digest=prior['metadata']['pruned']['digest'],
            expected_layout=prior['metadata']['pruned']['layout'])
        method,init=training.install_remaining_lora(model,study.TargetLoRA)
        frozen=lambda:storage.digest({n:p for n,p in model.named_parameters() if not p.requires_grad})
        frozen_before=frozen();buffers_before=storage.digest(dict(model.core.named_buffers()))
        adapter_initial=storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})
        write(arm+'_INITIALIZATION.json',dict(initial_static=initial,adapter=init,
            adapter_initial_digest=adapter_initial,frozen_digest=frozen_before,buffer_digest=buffers_before,math=math))
        cache=None;head=None;teacher_receipt=None;cache_receipt=None
        if arm=='KD':
            teacher,teacher_receipt=deployment.load_static(RUN/'wide_parent_static.pt',base,storage,
                device=f'cuda:{teacher_gpu}',expected_digest=prior['metadata']['parent']['digest'],
                expected_layout=prior['metadata']['parent']['layout'])
            assert storage.digest({'head':teacher.lm_head.weight})==storage.digest({'head':model.lm_head.weight})
            cache,cache_receipt=training.cache_teacher(teacher,memory,processor,batch_provider,
                on_item=lambda record,entry:event('KD_cache.jsonl',record))
            assert cache_receipt['hidden_bytes']<=147087360
            head=teacher.lm_head
            head.requires_grad_(False)
            del teacher
            bridge.ACTIVE=None;bridge.LAST_NATIVE=None;gc.collect();torch.cuda.empty_cache()
            write('KD_TEACHER.json',dict(parent=teacher_receipt,cache=cache_receipt,
                head_identical_to_student=True,head_device=str(head.weight.device),head_frozen=True))
        torch.cuda.set_device(student_gpu);bridge.bind(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        # Cache construction and model loading must not alter paired training RNG.
        reset()
        grads_seen=[]
        def updated(record,grads):
            assert len(grads)==288 and sum(g.numel() for g in grads.values())==3293184
            assert all(g.is_cuda and torch.isfinite(g).all() for g in grads.values())
            grads_seen.append(record)
            event(arm+'_updates.jsonl',record)
        receipt=training.train_arm(model,method,memory,processor,batch_provider,arm,
            teacher_cache=cache,teacher_head=head,
            on_qa=lambda record:event(arm+'_QA.jsonl',record),on_update=updated)
        assert len(grads_seen)==20 and frozen()==frozen_before
        assert storage.digest(dict(model.core.named_buffers()))==buffers_before
        adapter_final=storage.digest({n:p for n,p in model.named_parameters() if p.requires_grad})
        assert adapter_final!=adapter_initial
        write(arm+'_TRAINING.json',receipt)
        model.core.language_model.disable_input_require_grads();model.core.gradient_checkpointing_disable()
        del cache,head
        gc.collect();torch.cuda.empty_cache();torch.cuda.set_device(student_gpu)
        fold=training.fold_gpu(model,method);assert len(fold)==144
        checkpoint=deployment.export_static(model,RUN/(arm+'_complete.pt'),storage)
        proof=dict(initial_adapter_digest=adapter_initial,final_adapter_digest=adapter_final,initial_static_digest=prior['metadata']['pruned']['digest'],
            frozen_unchanged=True,buffers_unchanged=True,training=receipt,checkpoint=checkpoint,fold=fold,
            teacher_decoder_requests=160 if arm=='KD' else 0,teacher_head_requests=320 if arm=='KD' else 0,
            FIT_model_requests=320,backwards=320,optimizer_updates=20,new_generations=0,
            DEV_used_for_optimization=0,TEST_QA=0)
        files=[RUN/(arm+n) for n in ['_INITIALIZATION.json','_TRAINING.json','_QA.jsonl','_updates.jsonl']]
        if arm=='KD':files.extend([RUN/'KD_TEACHER.json',RUN/'KD_cache.jsonl'])
    else:
        train=read(RUN/(arm+'_train_COMPLETE.json'));cp=RUN/(arm+'_complete.pt')
        expected=train['checkpoint'];assert cp.stat().st_size==expected['bytes']
        model,initial=deployment.load_static(cp,base,storage,device=f'cuda:{student_gpu}',
            expected_digest=expected['digest'],expected_layout=expected['layout'])
        values,eval_proof=deployment.evaluate_dev(model,processor,devrows,prior_preds,storage,batch_provider=batch_provider)
        write(arm+'_predictions.json',values)
        measurements,memory_proof=deployment.resource_windows(model,processor,devrows,prior_preds,
            read(RUN/'prior_parent_memory.json'),storage,batch_provider=batch_provider)
        write(arm+'_memory.json',measurements)
        invariant=deployment.finish_proof(model,initial,storage)
        proof=dict(initial=initial,evaluation=eval_proof,memory=memory_proof,invariants=invariant,
            checkpoint=expected,new_generations=105,FIT_model_requests=0,backwards=0,optimizer_updates=0,
            teacher_decoder_requests=0,teacher_head_requests=0,DEV_used_for_optimization=0,TEST_QA=0)
        files=[RUN/(arm+'_predictions.json'),RUN/(arm+'_memory.json')]
    import resource
    torch.cuda.synchronize(student_gpu)
    phase_gpu={str(gpu):dict(peak_allocated=torch.cuda.max_memory_allocated(gpu),
        peak_reserved=torch.cuda.max_memory_reserved(gpu),scope='whole training phase including fresh construction; not matched inference') for gpu in required} if case.endswith('_train') else {}
    write(case+'_COMPLETE.json',dict(complete=True,identity=who,seconds=time.monotonic()-tick,
        ready_sha256=rh,math=math,process_RSS_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        phase_GPU=phase_gpu,files={p.name:sha(p) for p in files},**proof))
    signal.alarm(0)

if __name__=='__main__':
    case=sys.argv[1]
    try:main(case)
    except BaseException:
        write(case+'_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),no_quality_claim=True));raise
