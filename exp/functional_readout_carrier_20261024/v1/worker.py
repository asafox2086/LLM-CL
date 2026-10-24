"""One bounded phase: stream parent output moments, or reload and evaluate a small readout."""
import gc,json,os,random,signal,sys,time,traceback
from common import ROOT,HERE,PORT,BASE,REPORT,WORK,CASES,BUDGETS,COUNTS,MOMENT_BYTES,sha,write,identity,h,selection,dev_data,parent_reference,ready


def main(case,gpu):
    r=ready();admission=json.loads((REPORT/'ADMISSION_REVIEW.json').read_bytes())
    assert admission['model_admission'] is True and admission['ready_sha256']==sha(REPORT/'READY.json')
    from dispatch import available
    if gpu not in available():return 75
    assert not (REPORT/(case+'_STARTED.json')).exists()
    tick=time.monotonic();who=identity(os.getpid())
    write(REPORT/(case+'_STARTED.json'),dict(**who,started_monotonic=tick,unix=time.time(),GPU=gpu,
          boot_id=r['boot_id'],ready_sha256=sha(REPORT/'READY.json')))
    def expired(*unused):raise TimeoutError('Fixed functional-readout phase budget')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,BUDGETS[case])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import AutoProcessor
    from core import initialize,install,digest,layout,cpu_state,add_qa_moment
    sys.path.insert(0,str(PORT))
    import runtime_utils as rt,onevision_bridge as bridge
    from frozen_vision import install as install_visual
    from resource_monitor import ResourceMonitor
    torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    run=WORK/case;run.mkdir(parents=True,exist_ok=False);monitor=ResourceMonitor(run);monitor.snapshot('before_model')
    processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
    assert isinstance(processor.tokenizer.eos_token_id,int)
    checked=set();known_hashes={}
    def clear(model):
        model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0;gc.collect();torch.cuda.empty_cache()
    def base():
        model=bridge.Factory.from_pretrained(BASE,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
        model.requires_grad_(False);model.eval();bridge.bind(model);install_visual(model);return model
    def state_digest(model):return digest(list(model.core.named_parameters())+list(model.core.named_buffers()))
    def unchanged_digest(model):
        items=[(n,p) for n,p in model.core.named_parameters() if not any(n.endswith('.mlp.'+v) for v in ('left','right','down_proj.weight'))]
        return digest(items+list(model.core.named_buffers()))
    def inputs(model,row,task,s):
        for image in row['images']:
            if image not in checked:assert sha(ROOT/image)==s['image_sha256'][image];checked.add(image)
        return rt.input_batch(row,task,processor,model.device)
    def save(path,obj):
        with path.open('xb') as f:torch.save(obj,f);f.flush();os.fsync(f.fileno())
    def finish(extra):
        monitor.done.set();monitor.thread.join(timeout=6);assert not monitor.thread.is_alive();monitor.snapshot('complete')
        files={str(p.relative_to(WORK)):known_hashes.get(str(p.relative_to(WORK))) or sha(p) for p in run.rglob('*') if p.is_file()}
        write(REPORT/(case+'_COMPLETE.json'),dict(complete=True,worker=who,ready_sha256=sha(REPORT/'READY.json'),
              seconds=time.monotonic()-tick,files=files,TEST_QA=0,optimizer_updates=0,backward_calls=0,**extra))
        signal.setitimer(signal.ITIMER_REAL,0)
    try:
        if case=='init':
            assert sha(BASE/'model.safetensors')=='07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e'
            model=base();parent=state_digest(model);stable=unchanged_digest(model)
            _,_,ref=parent_reference();assert parent==ref['parent_digest']
            assert layout(model)['weights']['coefficients']==893675552
            task,rows,selected=selection();from common import metadata
            _,s=metadata();assert selected==r['selected']
            moments=[torch.zeros(896,896,dtype=torch.float32) for _ in range(24)]
            assert sum(t.untyped_storage().nbytes() for t in moments)==MOMENT_BYTES
            parts={arm:[] for arm in CASES[1:]};index=[];active={};current={};handles=[]
            diagnostic={arm:[dict(error=0.,energy=0.) for _ in range(24)] for arm in CASES[1:]}
            candidate_calls=0
            class CaptureDone(Exception):pass
            def hook(layer_index):
                def capture(module,args,output):
                    nonlocal candidate_calls
                    assert layer_index not in active
                    z=args[0];out=output
                    assert z.ndim==out.ndim==3 and z.shape[:2]==out.shape[:2] and z.shape[0]==1
                    assert z.shape[-1]==4864 and out.shape[-1]==896
                    if layer_index==0:
                        length=out.shape[1]
                        positions=sorted(sorted(range(length),key=lambda p:h('42:gate-response-position:'+current['id']+':'+str(p)))[:64])
                        assert positions;current.update(positions=positions,sequence_length=length)
                    assert out.shape[1]==current['sequence_length']
                    pos=torch.tensor(current['positions'],device=out.device)
                    yy=out[0].index_select(0,pos).detach().to('cpu',dtype=torch.float32)
                    assert torch.isfinite(yy).all()
                    row=dict(layer=layer_index,output_sha256=digest([('h',yy)]),positions=len(pos))
                    if current['role']=='fit':add_qa_moment(moments[layer_index],yy)
                    else:
                        zz=z[0].index_select(0,pos).detach()
                        assert torch.isfinite(zz).all();row['input_sha256']=digest([('z',zz)])
                        for arm in CASES[1:]:
                            left,right=parts[arm][layer_index]
                            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                                pred=F.linear(F.linear(zz,right.to(device=z.device,dtype=torch.float16)),left.to(device=z.device,dtype=torch.float16))
                            assert torch.isfinite(pred).all();candidate_calls+=1
                            pp=pred.detach().cpu().float();acc=diagnostic[arm][layer_index]
                            acc['error']+=float((pp-yy).square().mean())/16;acc['energy']+=float(yy.square().mean())/16
                    active[layer_index]=row
                    if layer_index==23:raise CaptureDone()
                return capture
            for i,layer in enumerate(model.model.layers):handles.append(layer.mlp.down_proj.register_forward_hook(hook(i)))
            def capture_rows(items):
                nonlocal active,current
                for item in items:
                    active={};current=dict(id=item['id'],role=item['role']);row=rows[item['id']]
                    batch,details=inputs(model,row,task,s);input_sha=digest(list(batch.items()))
                    try:
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):rt.answer_loss(model,batch,row['answer'],processor)
                    except CaptureDone:pass
                    else:raise AssertionError('Teacher pass reached output head instead of capture boundary')
                    assert set(active)==set(range(24))
                    index.append(dict(selection=item,input_sha256=input_sha,positions=current['positions'],sequence_length=current['sequence_length'],
                                      layers=[active[i] for i in range(24)],**details))
                    write(run/'capture_index.json',dict(rows=index));del batch
            try:
                capture_rows(selected[:32]);monitor.snapshot('moments_complete',model=model)
                save(run/'moments.pt',moments);decomposition=[]
                for i,layer in enumerate(model.model.layers):
                    values,evidence=initialize(layer.mlp.down_proj.weight,moments[i])
                    evidence.update(layer=i,factors={})
                    for arm,pair in values.items():
                        assert all(torch.isfinite(t.half()).all() for t in pair)
                        parts[arm].append(pair)
                        evidence['factors'][arm]=dict(left_shape=list(pair[0].shape),right_shape=list(pair[1].shape),
                            finite_after_FP16_cast=True,digest=digest([('left',pair[0]),('right',pair[1])]))
                    decomposition.append(evidence)
                write(run/'decomposition.json',dict(layers=decomposition));save(run/'initial_factors.pt',parts)
                capture_rows(selected[32:])
            finally:
                for handle in handles:handle.remove()
            assert len(index)==48 and candidate_calls==16*24*2
            after=state_digest(model);assert after==parent and unchanged_digest(model)==stable
            scores={arm:[dict(layer=i,mean_squared_error=v['error'],mean_target_energy=v['energy'],error_ratio=v['error']/max(v['energy'],1e-12)) for i,v in enumerate(values)] for arm,values in diagnostic.items()}
            write(run/'local_diagnostic.json',dict(QA=16,actual_local_candidate_readouts=candidate_calls,arms=scores,
                teacher_inputs_only_not_whole_decoder_error=True))
            write(run/'parent_binding.json',dict(parent_digest=parent,parent_VERIFIED_sha256=sha(ROOT/'parent_reference/VERIFIED.json'),parent_layout=layout(model)))
            clear(model);finish(dict(parent_digest=parent,parent_digest_after=after,teacher_QA=48,moment_QA=32,diagnostic_QA=16,moment_tensor_bytes=MOMENT_BYTES));return 0
        init=json.loads((REPORT/'init_COMPLETE.json').read_bytes())
        for name,expected in init['files'].items():assert sha(WORK/name)==expected
        parts=torch.load(WORK/'init/initial_factors.pt',map_location='cpu',weights_only=False)[case]
        model=base();_,_,ref=parent_reference();assert state_digest(model)==ref['parent_digest']
        stable=unchanged_digest(model);install(model,parts);del parts
        model.requires_grad_(False);clear(model)
        assert unchanged_digest(model)==stable
        final_layout=layout(model);final_digest=state_digest(model)
        assert final_layout['weights']['coefficients']==COUNTS[case]
        package=run/'final_complete.pt';state=cpu_state(model)
        save(package,dict(case=case,state=state,layout=final_layout,state_digest=digest(state.items())))
        package_meta=dict(bytes=package.stat().st_size,sha256=sha(package));known_hashes[str(package.relative_to(WORK))]=package_meta['sha256']
        bridge.ACTIVE=None;del state,model;gc.collect();torch.cuda.empty_cache()
        loaded=torch.load(package,map_location='cpu',weights_only=False)
        model=base();zeros=[(torch.zeros(896,200),torch.zeros(200,4864)) for _ in range(24)]
        install(model,zeros);del zeros;model.core.load_state_dict(loaded['state'],strict=True)
        assert digest(model.core.state_dict().items())==loaded['state_digest'];del loaded
        reload_layout=layout(model);reload_digest=state_digest(model)
        assert reload_layout==final_layout and reload_digest==final_digest and unchanged_digest(model)==stable
        m,s,dev=dev_data();_,old_pred,_=parent_reference();clear(model);before=state_digest(model)
        with (run/'predictions.jsonl').open('x',encoding='utf8') as f:
            for task in m['tasks']:
                for row in dev[task['task']]:
                    batch,details=inputs(model,row,task,s);input_sha=digest(list(batch.items()))
                    assert input_sha==old_pred[row['id']]['input_sha256']
                    class Finite:
                        calls=0
                        def __call__(self,input_ids,scores):
                            assert not torch.isnan(scores).any() and not torch.isposinf(scores).any() and torch.isfinite(scores).any(-1).all()
                            self.calls+=1;return scores
                    guard=Finite()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                        result=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=task['max_new_tokens'],use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[guard])
                    ids=result[0,batch['input_ids'].shape[1]:].tolist()
                    record=dict(id=row['id'],task=task['task'],references=row['answers'],input_sha256=input_sha,
                        images={n:s['image_sha256'][n] for n in row['images']},generation_cap=task['max_new_tokens'],
                        eos_token_id=processor.tokenizer.eos_token_id,output_token_ids=ids,finite_score_checks=guard.calls,
                        prediction=processor.tokenizer.decode(ids,skip_special_tokens=True),**details)
                    f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        after=state_digest(model);assert before==after==final_digest
        clear(model);finish(dict(layout=final_layout,reload_layout=reload_layout,final_digest=final_digest,reload_digest=reload_digest,
            evaluation_digest_before=before,evaluation_digest_after=after,unchanged_weights_before=stable,
            unchanged_weights_after=unchanged_digest(model),checkpoint=package_meta,
            physical_original_D_absent=not any(n.endswith('down_proj.weight') for n in model.core.state_dict())));return 0
    finally:
        monitor.done.set();monitor.thread.join(timeout=6)


if __name__=='__main__':
    assert sys.argv[1] in CASES
    try:sys.exit(main(sys.argv[1],int(sys.argv[2])))
    except Exception:
        write(REPORT/(sys.argv[1]+'_FAILURE.json'),dict(traceback=traceback.format_exc(),no_retry=True));raise
