"""Read-only likelihood measurements; no generation, fitting or checkpoint write."""
import argparse
import ast
import hashlib
import inspect
import json
import os
import pickle
import random
import signal
import sys
import time
import traceback
from common import ROOT,HERE,PORT,SCREEN,REPORT,WORK,STATES,BASE,BUDGET,PHYSICAL_GPU,check_ready,data,identity,available,loss_contract,sha,write

def tensor_sha(value):
    import torch
    v=value.detach().cpu().contiguous();h=hashlib.sha256(str((tuple(v.shape),str(v.dtype))).encode())
    raw=v.reshape(-1).view(torch.uint8).numpy()
    for start in range(0,raw.size,8*1024**2):h.update(raw[start:start+8*1024**2].tobytes())
    return h.hexdigest()

def weights_digest(model):
    h=hashlib.sha256();layout=[]
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
        for name,value in items:
            item=dict(kind=kind,name=name,shape=list(value.shape),dtype=str(value.dtype),sha256=tensor_sha(value))
            layout.append(item);h.update(json.dumps(item,sort_keys=True).encode())
    return h.hexdigest(),layout

def cache_receipt(model,row,batch,details):
    images=[]
    for rel in row['images']:
        p=ROOT/rel;st=p.stat();images.append((rel,st.st_size,st.st_mtime_ns))
    key=hashlib.sha256(json.dumps(dict(images=images,native_layout=batch['image_grid_thw'].tolist(),dtype=str(model.visual.get_dtype()))).encode()).hexdigest()
    feature_sha=tensor_sha(batch['pixel_values'])
    if key in model._llmcl_visual_cache:assert tensor_sha(model._llmcl_visual_cache[key])==feature_sha
    elif details['visual_cache_hit']:raise AssertionError('Claimed cache hit not bound to actual donor image key')
    return dict(image_paths=row['images'],cache_key=key,cache_hit=details['visual_cache_hit'],feature_sha256=feature_sha,
        feature_shape=list(batch['pixel_values'].shape),cached_key_present=key in model._llmcl_visual_cache)

def run(state):
    ready=check_ready();marker=REPORT/(state+'_STARTED.json');run=WORK/state
    assert state in STATES and not marker.exists() and not run.exists()
    ok,cards=available()
    if not ok:return 75
    tick=time.monotonic();who=identity(os.getpid());assert who['uid']==os.getuid() and who['pgid']==who['pid']
    write(marker,dict(**who,boot_id=__import__('pathlib').Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        started_monotonic=tick,started_unix=time.time(),budget_seconds=BUDGET,ready_sha256=sha(REPORT/'MODEL_READY.json'),inventory=cards,state=state))
    def expired(*unused):raise TimeoutError('Fixed paired-image diagnostic state budget expired')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,BUDGET-(time.monotonic()-tick)))
    os.environ.update(CUDA_VISIBLE_DEVICES=str(PHYSICAL_GPU),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import torch,numpy as np
    from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
    sys.path.insert(0,str(ROOT/'exp/osft_increment_screen_20261019/v1'));sys.path.insert(0,str(PORT))
    import study,onevision_bridge,runtime_utils,frozen_vision
    from resource_monitor import ResourceMonitor
    assert study.ROOT.resolve()==ROOT and runtime_utils.ROOT.resolve()==ROOT and frozen_vision.ROOT.resolve()==ROOT
    assert __import__('pathlib').Path(inspect.getfile(runtime_utils.answer_loss)).resolve()==(PORT/'runtime_utils.py').resolve()
    node,h=loss_contract();assert h==ready['full_loss_AST_sha256']
    namespace=dict(vars(runtime_utils));exec(compile(ast.Module(body=[node],type_ignores=[]),'<paired-full-answer-loss>','exec'),namespace)
    full_loss=namespace['full_answer_loss']
    manifest,p,pairs,rows=data();tasks={t['task']:t for t in manifest['tasks']}
    run.mkdir(parents=True);monitor=ResourceMonitor(run);monitor.snapshot('process_before_model')
    try:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
        processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
        case='LoRA_qv' if state.startswith('qv_') else 'OSFT_all'
        if state.endswith('_before'):
            for n,digest in ready['base_weights'].items():assert sha(ROOT/n)==digest
            core,loading=LlavaOnevisionForConditionalGeneration.from_pretrained(BASE,torch_dtype=torch.float16,
                attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
            assert not any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),loading
            model=onevision_bridge.OneVisionBridge(core).cuda();onevision_bridge.bind(model)
            torch.manual_seed(42);torch.cuda.manual_seed_all(42)
            method=study.make_method(model,case,[t['instruction'] for t in manifest['tasks']])
            provenance=dict(kind='current_reconstructed_unadapted',not_historical_S0_bitwise_replica=True,seed=42)
        else:
            bound=ready['checkpoints'][case];path=ROOT/bound['path']
            assert path.stat().st_size==bound['metadata']['bytes'] and sha(path)==bound['metadata']['sha256']
            obj=torch.load(path,pickle_module=pickle,weights_only=False)
            assert obj['format']=='llmcl-method-graph-v1' and obj['stage']==19 and obj['config']['method']==case
            assert obj['config']['manifest_sha256']==p['manifest_sha256'] and obj['config']['split_sha256']==p['split_sha256']
            assert obj['config']['FIT_QA']==1213 and obj['config']['DEV_QA']==100 and obj['config']['TEST_QA']==0
            method=obj['method'];model=method.model
            provenance=dict(kind='existing_legal_FIT_final_checkpoint',checkpoint=bound,current_common_processor_used=True)
            del obj
        assert model.llmcl_backbone=='onevision' and model.dtype==torch.float16
        for parameter in model.parameters():parameter.requires_grad_(False)
        model.eval();onevision_bridge.bind(model);onevision_bridge.NATIVE_GATE_MODE=False
        # Fresh cache; loaded training/evaluation cache is not an image intervention.
        model._llmcl_visual_cache=__import__('collections').OrderedDict()
        model._llmcl_visual_cache_bytes=0;model._llmcl_visual_cache_hits=0;model._llmcl_visual_cache_misses=0
        if not hasattr(model,'_llmcl_visual_original'):method.vision_original=frozen_vision.install(model)
        assert all(not parameter.requires_grad for parameter in model.parameters())
        initial,layout=weights_digest(model);write(run/'INITIAL_BINDING.json',dict(provenance=provenance,digest=initial,layout=layout))
        monitor.snapshot('inference_before',model=model);out=run/'measurements.jsonl';assert not out.exists();requests=0
        eos=processor.tokenizer.eos_token_id;assert isinstance(eos,int) and 0<=eos<model.config.vocab_size
        with out.open('x',encoding='utf8') as stream:
            for record in pairs:
                row=rows[record['id']];donor=rows[record['donor_id']];wrong=dict(row,images=list(donor['images']))
                assert [sha(ROOT/n) for n in wrong['images']]==next(r['image_sha256'] for r in pairs if r['id']==record['donor_id'])
                method.set_instruction(record['task'])
                with torch.no_grad():
                    original,original_details=runtime_utils.input_batch(row,tasks[record['task']],processor,model.device)
                    replaced,wrong_details=runtime_utils.input_batch(wrong,tasks[record['task']],processor,model.device)
                    assert set(original)==set(replaced)
                    for key in original:
                        if key=='pixel_values':assert original[key].shape==replaced[key].shape
                        else:assert torch.equal(original[key],replaced[key]),(row['id'],key)
                    target=processor.tokenizer(row['answer'],add_special_tokens=False,truncation=False)['input_ids']
                    assert target and isinstance(row['answer'],str);target=target+[eos]
                    assert max(target)<model.config.vocab_size and min(target)>=0
                    assert original['input_ids'].shape[1]+len(target)<=model.config.max_position_embeddings
                    ids=torch.cat([original['input_ids'],torch.tensor([target],device=model.device)],dim=1)
                    positions,_=model.get_rope_index(ids,original['image_grid_thw'],attention_mask=torch.ones_like(ids))
                    wrong_ids=torch.cat([replaced['input_ids'],torch.tensor([target],device=model.device)],dim=1)
                    wrong_positions,_=model.get_rope_index(wrong_ids,replaced['image_grid_thw'],attention_mask=torch.ones_like(wrong_ids))
                    assert torch.equal(positions,wrong_positions)
                    features=[cache_receipt(model,r,b,d) for r,b,d in ((row,original,original_details),(wrong,replaced,wrong_details))]
                    inputs={k:tensor_sha(v) for k,v in original.items() if k!='pixel_values'}
                    inputs['positions_sha256']=tensor_sha(positions)
                    target_sha=hashlib.sha256(json.dumps(target).encode()).hexdigest()
                    binding=hashlib.sha256(json.dumps({k:row.get(k) for k in ('id','question','context','answer','answers')},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                    values=[]
                    for batch in (original,replaced):
                        with torch.autocast('cuda',dtype=model.dtype):value,n=full_loss(model,batch,row['answer'],processor)
                        requests+=1;assert n==len(target) and torch.isfinite(value)
                        values.append(float(value));del value
                result=dict(id=record['id'],task=record['task'],domain=record['domain'],group=record['group'],donor_id=record['donor_id'],
                    original_NLL=values[0],wrong_NLL=values[1],C=values[1]-values[0],target_tokens=len(target),
                    target_sha256=target_sha,QA_binding_sha256=binding,nonvisual_input_sha256=inputs,features=features,
                    original_image_sha256=record['image_sha256'],wrong_image_sha256=next(r['image_sha256'] for r in pairs if r['id']==record['donor_id']),
                    image_shapes=original_details['image_shapes'],wrong_image_shapes=wrong_details['image_shapes'])
                stream.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                del original,replaced,ids,wrong_ids,positions,wrong_positions
        assert requests==122
        final,final_layout=weights_digest(model);assert final==initial and final_layout==layout
        assert not any(parameter.grad is not None for parameter in model.parameters())
        monitor.snapshot('inference_after',model=model)
        monitor.done.set();monitor.thread.join(timeout=6);monitor.snapshot('run_finished',model=model)
        resource_rows=[json.loads(x) for x in monitor.path.read_text().splitlines()]
        fields=('VmRSS_bytes','cpu_uss_bytes','gpu_allocated_bytes','gpu_reserved_bytes','gpu_process_nvml_bytes')
        peaks={k:max(x[k] for x in resource_rows if x.get(k) is not None) if any(x.get(k) is not None for x in resource_rows) else None for k in fields}
        elapsed=time.monotonic()-tick;assert elapsed<BUDGET
        write(REPORT/(state+'_COMPLETE.json'),dict(state=state,ready_sha256=sha(REPORT/'MODEL_READY.json'),
            measurement_path=str(out.relative_to(ROOT)),measurement_sha256=sha(out),NLL_requests=requests,QA=61,
            weight_digest_before=initial,weight_digest_after=final,initial_binding_path=str((run/'INITIAL_BINDING.json').relative_to(ROOT)),
            initial_binding_sha256=sha(run/'INITIAL_BINDING.json'),resources_path=str(monitor.path.relative_to(ROOT)),resources_sha256=sha(monitor.path),
            observed_sample_peaks_bytes=peaks,model_RUN_seconds=elapsed,visual_cache=frozen_vision.stats(model),training_updates=0,generation_requests=0,TEST_QA=0))
        signal.setitimer(signal.ITIMER_REAL,0);return 0
    finally:
        monitor.done.set();monitor.thread.join(timeout=6)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('state',choices=STATES);args=p.parse_args()
    try:sys.exit(run(args.state))
    except Exception:
        write(REPORT/(args.state+'_FAILURE.json'),dict(traceback=traceback.format_exc(),not_quality_negative=True,no_retry=True));raise
