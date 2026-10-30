"""DRAFT interfaces only. No asset loader, CLI, admission, GPU choice or controller.
External worker must full-load original method-graph CP and bind all original sources.
Not admitted: do not call before independent source/asset/runtime admission.
"""
import copy,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import StoppingCriteria,StoppingCriteriaList

def durable(path,record):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(record,f,ensure_ascii=False,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())

def tensor_signature(batch):
    out={}
    for k,v in sorted(batch.items()):
        if not isinstance(v,torch.Tensor):raise TypeError('unreviewed non-tensor generation input')
        raw=v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        out[k]=dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(raw).hexdigest())
    return out

class ObserveFirst32(StoppingCriteria):
    # No eos_token_id property; no max_length override; no RNG or output edits.
    def __init__(self,initial_input_ids_length):self.initial_length=initial_input_ids_length
    def __call__(self,input_ids,scores,**kwargs):
        stop=input_ids.shape[-1]-self.initial_length>=32
        return torch.full((input_ids.shape[0],),stop,dtype=torch.bool,device=input_ids.device)

def generation_contract(model,processor,task):
    original=model.core.language_model.generation_config.to_dict()
    kwargs=dict(do_sample=False,num_beams=1,max_new_tokens=task['max_new_tokens'],use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
    effective=copy.deepcopy(model.core.language_model.generation_config)
    effective.update(**kwargs)
    mode=effective.get_generation_mode()
    if getattr(mode,'value',mode)!='greedy_search':raise RuntimeError('original effective generation is not GREEDY_SEARCH')
    if effective.max_time is not None:raise RuntimeError('time-based stop prevents prefix equivalence')
    if kwargs['max_new_tokens']<1:raise ValueError('original generation cap')
    return kwargs,dict(original_config=original,effective_config=effective.to_dict(),
        generation_mode=getattr(mode,'value',str(mode)),native_kwargs=kwargs,
        added_stop='Only ObserveFirst32; original task cap, EOS/pad/cache and logits processors unchanged')

def observe_qa(*,method,processor,task,row,rt,fv,hooks_module,statistics,old_tf_c,qa_dir):
    """One question-only native generate; call only in reviewed externally claimed run.
    row outputs never include question/gold/reference text. Prefix token IDs are recorded.
    Per-step moments permit later phase/QA-equal aggregation; no fit or readout here.
    """
    model=method.model
    if model.training or torch.is_grad_enabled():raise RuntimeError('caller must use eval + inference_mode')
    safe={k:row[k] for k in ('id','question','images','context') if k in row}
    assert not any(k in safe for k in ('answer','answers','references'))
    qa_dir=Path(qa_dir);qa_dir.mkdir(parents=True,exist_ok=False)
    durable(qa_dir/'STARTED.json',dict(qa_id_sha256=hashlib.sha256(safe['id'].encode()).hexdigest(),
        task=task['task'],domain=task['domain'],new_QA=0,TRAIN_only=True,no_retry=True))
    method.set_instruction(task['task'])
    decoder=model.core.language_model.model;counter={'all':0}
    def count(module,args):counter['all']+=1
    count_handle=decoder.register_forward_pre_hook(count)
    handles=[];hooks=None;stream=None;rows=0
    try:
        batch,details=rt.input_batch(safe,task,processor,model.device)
        feature_before=counter['all'];cache_hit=None
        # Exact original runtime_utils.evaluate preparation ordering.
        if hasattr(method,'vision_original'):
            fv.prepare(batch,safe,model,method.vision_original)
            if hasattr(method,'prepare_row'):
                cache_hit=safe['question'] in method.feature_cache
                method.prepare_row(safe,batch)
        if method.name=='CL_MoE' and not hasattr(method,'vision_original'):
            raise RuntimeError('original CL question feature preparation path missing')
        feature_forwards=counter['all']-feature_before
        signatures=tensor_signature(batch);initial_length=int(batch['input_ids'].shape[1])
        kwargs,contract=generation_contract(model,processor,task)
        stream=(qa_dir/'PREDICTION_RECORDS.jsonl').open('x',encoding='utf8')
        def emit(record):
            nonlocal rows
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+chr(10));rows+=1
        hooks=hooks_module.NativeMomentHooks(model=model,method_name=method.name,statistics=statistics,
            emit=emit,closure_atol=0.,closure_rtol=0.)
        hooks.begin_native_qa(qa_id=safe['id'],domain=task['domain'],prompt_length=initial_length,old_tf_c=old_tf_c)
        handles.append(decoder.register_forward_pre_hook(hooks.decoder_pre,with_kwargs=True))
        handles.append(decoder.register_forward_hook(hooks.decoder_post,with_kwargs=True))
        started=time.monotonic();before_generation=counter['all']
        with torch.autocast('cuda',dtype=model.dtype),hooks.dispatch():
            generated=model.generate(**batch,**kwargs,
                stopping_criteria=StoppingCriteriaList([ObserveFirst32(initial_length)]))
        seconds=time.monotonic()-started
        if generated.ndim!=2 or generated.shape[0]!=1:raise RuntimeError('native batch1 token output required')
        if not torch.equal(generated[:,:initial_length],batch['input_ids']):raise RuntimeError('generate must retain original prompt prefix')
        prefix=generated[0,initial_length:].detach().cpu().tolist();n=len(prefix)
        summary=hooks.end_native_qa(n);steps=summary['steps']
        if counter['all']-before_generation!=n or rows!=168*n:raise RuntimeError('native forward/token/target-layer counts')
        if model.core.language_model.generation_config.to_dict()!=contract['original_config']:
            raise RuntimeError('generation config mutated')
        stream.flush();os.fsync(stream.fileno());stream.close();stream=None
        phase_path=qa_dir/'QA_PHASE_MOMENTS.jsonl'
        with phase_path.open('x',encoding='utf8') as f:
            for record in summary['phase_rows']:f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+chr(10))
            f.flush();os.fsync(f.fileno())
        if len(summary['phase_rows'])!=168*(1+int(n>1)):raise RuntimeError('QA-phase row count')
        assert steps[0]['phase']=='prefill' and sum(x['phase']=='prefill' for x in steps)==1
        metadata=dict(qa_id_sha256=hashlib.sha256(safe['id'].encode()).hexdigest(),
            task=task['task'],domain=task['domain'],instruction_sha256=hashlib.sha256(task['instruction'].encode()).hexdigest(),
            prepared_input_signatures=signatures,input_details=details,generation_contract=contract,
            initial_input_ids_length=initial_length,prefix_token_ids=prefix,actual_prediction_positions=n,
            prefill_positions=1,decode_positions=n-1,decode_eligible=n>1,
            original_cap=task['max_new_tokens'],added_prefix_stop_limit=32,steps=steps,
            question_feature_decoder_forwards=feature_forwards,question_feature_cache_hit=cache_hit,
            generation_decoder_forwards=counter['all']-before_generation,
            generation_and_observation_seconds=seconds,new_scores=0,new_CE=0,new_F1=0,new_checkpoints=0,
            moments_sha256=hashlib.sha256(phase_path.read_bytes()).hexdigest(),
            moments_rows=len(summary['phase_rows']),position_record_rows=rows,
            position_records_sha256=hashlib.sha256((qa_dir/'PREDICTION_RECORDS.jsonl').read_bytes()).hexdigest(),
            no_gold_tokens_passed=True,own_model_native_trajectory=True)
        durable(qa_dir/'COMPLETE.json',metadata)
        return metadata
    finally:
        if stream is not None:stream.flush();os.fsync(stream.fileno());stream.close()
        for h in handles:h.remove()
        if hooks is not None:hooks.close()
        count_handle.remove()
