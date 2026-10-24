"""One bounded phase: shared teacher capture, then two independent carriers."""
import gc,importlib.util,json,math,os,random,signal,sys,time,traceback
from common import ROOT,HERE,PORT,BASE,REPORT,WORK,CASES,BUDGETS,COUNTS,CAP,sha,write,identity,h,selection,dev_data,parent_reference,ready


def main(case,gpu):
    r=ready();a=json.loads((REPORT/'ADMISSION_REVIEW.json').read_bytes())
    assert a['model_admission'] is True and a['ready_sha256']==sha(REPORT/'READY.json')
    from dispatch import available
    if gpu not in available():return 75
    assert not (REPORT/(case+'_STARTED.json')).exists()
    tick=time.monotonic();who=identity(os.getpid())
    write(REPORT/(case+'_STARTED.json'),dict(**who,started_monotonic=tick,unix=time.time(),GPU=gpu,
        boot_id=r['boot_id'],ready_sha256=sha(REPORT/'READY.json')))
    def expired(*_):raise TimeoutError('Fixed gate-response phase budget')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,BUDGETS[case])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from core import Carrier,initialize,install,digest,layout,cpu_state
    sys.path.insert(0,str(PORT))
    import runtime_utils as rt,onevision_bridge as bridge
    from frozen_vision import install as install_visual
    from resource_monitor import ResourceMonitor
    from host_adam import HostAdamW
    torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    run=WORK/case;run.mkdir(parents=True,exist_ok=False);monitor=ResourceMonitor(run)
    monitor.snapshot('before_model')
    processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
    assert isinstance(processor.tokenizer.eos_token_id,int)
    checked=set()
    def clear(model):
        model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0
        gc.collect();torch.cuda.empty_cache()
    def base():
        model=bridge.Factory.from_pretrained(BASE,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
        model.requires_grad_(False);model.eval();bridge.bind(model);install_visual(model);return model
    def state_digest(model):return digest(list(model.core.named_parameters())+list(model.core.named_buffers()))
    def frozen_digest(model):
        return digest([(n,p) for n,p in model.core.named_parameters() if not any(n.endswith('.mlp.'+v) for v in ('left','right','row_scale'))]+list(model.core.named_buffers()))
    def inputs(model,row,task,s):
        for image in row['images']:
            if image not in checked:assert sha(ROOT/image)==s['image_sha256'][image];checked.add(image)
        return rt.input_batch(row,task,processor,model.device)
    known_hashes={}
    def finish(extra):
        monitor.done.set();monitor.thread.join(timeout=6);assert not monitor.thread.is_alive()
        monitor.snapshot('complete')
        files={str(p.relative_to(WORK)):(known_hashes[str(p.relative_to(WORK))] if str(p.relative_to(WORK)) in known_hashes else sha(p)) for p in run.rglob('*') if p.is_file()}
        write(REPORT/(case+'_COMPLETE.json'),dict(complete=True,worker=who,ready_sha256=sha(REPORT/'READY.json'),
            seconds=time.monotonic()-tick,files=files,TEST_QA=0,**extra))
        signal.setitimer(signal.ITIMER_REAL,0)
    try:
        if case=='init':
            assert sha(BASE/'model.safetensors')=='07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e'
            model=base();parent=state_digest(model)
            _,_,ref=parent_reference();assert parent==ref['parent_digest']
            assert layout(model)['weights']['coefficients']==893675552
            task,rows,selected=selection();from common import metadata
            _,s=metadata();assert selected==r['selected']
            cache=[];cache_index=[];payload=0;active={};current={};handles=[]
            class CaptureDone(Exception):pass
            def hook(index):
                def capture(module,args,output):
                    nonlocal payload
                    x=args[0];y=output
                    assert x.shape==y.shape and x.ndim==3 and x.shape[0]==1 and x.shape[-1]==896
                    assert index not in active
                    if index==0:
                        length=x.shape[1]
                        positions=sorted(sorted(range(length),key=lambda p:h('42:gate-response-position:'+current['id']+':'+str(p)))[:64])
                        assert positions
                        current.update(positions=positions,sequence_length=length)
                    assert x.shape[1]==current['sequence_length']
                    positions=torch.tensor(current['positions'],device=x.device)
                    need=len(positions)*896*2*2
                    assert payload+need<=CAP,'Sampled teacher feature payload cap exceeded'
                    xx=x[0].index_select(0,positions).detach().to('cpu',dtype=torch.float16).clone()
                    yy=y[0].index_select(0,positions).detach().to('cpu',dtype=torch.float16).clone()
                    assert torch.isfinite(xx).all() and torch.isfinite(yy).all()
                    payload+=xx.untyped_storage().nbytes()+yy.untyped_storage().nbytes();assert payload<=CAP
                    active[index]=(xx,yy)
                    if index==23:raise CaptureDone()
                return capture
            for i,layer in enumerate(model.model.layers):handles.append(layer.mlp.register_forward_hook(hook(i)))
            try:
                for item in selected:
                    active={};current=dict(id=item['id']);row=rows[item['id']]
                    batch,details=inputs(model,row,task,s);input_sha=digest(list(batch.items()))
                    try:
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):rt.answer_loss(model,batch,row['answer'],processor)
                    except CaptureDone:pass
                    else:raise AssertionError('Teacher pass reached output head instead of capture boundary')
                    assert set(active)==set(range(24))
                    pairs=[active[i] for i in range(24)];cache.append(pairs)
                    entry=dict(selection=item,input_sha256=input_sha,positions=current['positions'],sequence_length=current['sequence_length'],
                        layers=[dict(shape=list(x.shape),input_sha256=digest([('x',x)]),output_sha256=digest([('h',y)])) for x,y in pairs],
                        payload_bytes=sum(x.untyped_storage().nbytes()+y.untyped_storage().nbytes() for x,y in pairs),**details)
                    cache_index.append(entry);write(run/'capture_index.json',dict(rows=cache_index,payload_bytes=payload))
                    del batch
            finally:
                for handle in handles:handle.remove()
            assert len(cache)==48 and state_digest(model)==parent
            with (run/'features.pt').open('xb') as f:torch.save(cache,f);f.flush();os.fsync(f.fileno())
            values={}
            for arm in CASES[1:]:
                values[arm]=[initialize(layer.mlp,arm) for layer in model.model.layers]
                assert state_digest(model)==parent
            with (run/'initial_factors.pt').open('xb') as f:torch.save(values,f);f.flush();os.fsync(f.fileno())
            write(run/'parent_binding.json',dict(parent_digest=parent,parent_VERIFIED_sha256=sha(ROOT/'parent_reference/VERIFIED.json'),parent_layout=layout(model)))
            clear(model);finish(dict(parent_digest=parent,capture_QA=48,payload_bytes=payload));return 0
        init=json.loads((REPORT/'init_COMPLETE.json').read_bytes())
        for name,expected in init['files'].items():assert sha(WORK/name)==expected
        cache=torch.load(WORK/'init/features.pt',map_location='cpu',weights_only=False)
        parts=torch.load(WORK/'init/initial_factors.pt',map_location='cpu',weights_only=False)[case]
        assert len(cache)==48 and len(parts)==24
        model=base();_,_,ref=parent_reference();assert state_digest(model)==ref['parent_digest']
        install(model,case,parts,master=True);model.requires_grad_(False);del parts
        assert layout(model)['weights']['coefficients']==COUNTS[case]
        stable=frozen_digest(model);clear(model);updates=0;fit_states=None;layer_results=[]
        with (run/'updates.jsonl').open('x',encoding='utf8') as journal:
            for index,layer in enumerate(model.model.layers):
                block=layer.mlp;assert isinstance(block,Carrier)
                train=[]
                for name in ('left','right','row_scale'):
                    p=getattr(block,name)
                    if p is not None:p.data=p.data.float();p.requires_grad_(True);train.append((name,p))
                x=torch.cat([sample[index][0] for sample in cache[:32]])
                y=torch.cat([sample[index][1] for sample in cache[:32]])
                hx=torch.cat([sample[index][0] for sample in cache[32:]])
                hy=torch.cat([sample[index][1] for sample in cache[32:]])
                fit_states=len(x) if fit_states is None else fit_states;assert len(x)==fit_states
                denominator=float(y.float().square().mean().clamp_min(1e-12))
                def check():
                    error=energy=0.;n=0
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                        for start in range(0,len(hx),128):
                            yy=hy[start:start+128].cuda();pred=block(hx[start:start+128].cuda())
                            assert torch.isfinite(pred).all()
                            error+=float((pred.float()-yy.float()).square().sum());energy+=float(yy.float().square().sum());n+=yy.numel()
                    return error/max(energy,1e-12*n)
                before=check();initial_params=digest(train)
                optimizer=HostAdamW([p for _,p in train],lr=1e-4,eps=1e-8,betas=(.9,.999),weight_decay=0.)
                scaler=torch.amp.GradScaler('cuda',init_scale=1.)
                for epoch in range(4):
                    for start in range(0,len(x),128):
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast('cuda',dtype=torch.float16):
                            predicted=block(x[start:start+128].cuda());target=y[start:start+128].cuda()
                            loss=(predicted.float()-target.float()).square().mean()/denominator
                        assert torch.isfinite(loss)
                        scaler.scale(loss).backward();scaler.unscale_(optimizer)
                        assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in train)
                        grad=torch.nn.utils.clip_grad_norm_([p for _,p in train],1.);assert torch.isfinite(grad)
                        count_before=sum(float(st.get('step',0)) for st in optimizer.state.values())
                        scaler.step(optimizer);scaler.update()
                        count_after=sum(float(st.get('step',0)) for st in optimizer.state.values())
                        assert count_after>count_before and all(torch.isfinite(p).all() for _,p in train)
                        moments=[v for state in optimizer.state.values() for k,v in state.items() if k in ('exp_avg','exp_avg_sq')]
                        assert all(t.device.type=='cpu' for t in moments)
                        updates+=1
                        record=dict(layer=index,epoch=epoch+1,offset=start,states=min(128,len(x)-start),step=updates,loss=float(loss),
                            finite=True,positive_lr=optimizer.param_groups[0]['lr']>0,actual_step=True,
                            trainable=sum(p.numel() for _,p in train),parameter_ids={name:id(p) for name,p in train},
                            gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for _,p in train),
                            cpu_moments_bytes=sum(t.numel()*t.element_size() for t in moments))
                        journal.write(json.dumps(record,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                after=check();changed=initial_params!=digest(train)
                layer_results.append(dict(layer=index,held_feature_before=before,held_feature_after=after,parameters_changed=changed))
                assert changed
                for _,p in train:p.requires_grad_(False);p.grad=None;p.data=p.data.half()
                del optimizer,scaler,train,x,y,hx,hy,predicted,target,loss,moments
                gc.collect();torch.cuda.empty_cache();monitor.snapshot('layer_fitted',stage=index+1,model=model)
        assert updates==24*4*math.ceil(fit_states/128) and frozen_digest(model)==stable
        del cache;clear(model);model.eval();model.requires_grad_(False)
        final_layout=layout(model);final_digest=state_digest(model);assert final_layout['weights']['coefficients']==COUNTS[case]
        package=run/'final_complete.pt';state=cpu_state(model)
        with package.open('xb') as f:torch.save(dict(case=case,state=state,layout=final_layout,state_digest=digest(state.items())),f);f.flush();os.fsync(f.fileno())
        package_meta=dict(bytes=package.stat().st_size,sha256=sha(package))
        known_hashes[str(package.relative_to(WORK))]=package_meta['sha256']
        bridge.ACTIVE=None;del state,block,layer,model;gc.collect();torch.cuda.empty_cache()
        loaded=torch.load(package,map_location='cpu',weights_only=False)
        model=base();rank=200 if case=='Static' else 199
        zeros=[(torch.zeros(4864,rank),torch.zeros(rank,896),None if case=='Static' else torch.zeros(4864)) for _ in range(24)]
        install(model,case,zeros);del zeros
        model.core.load_state_dict(loaded['state'],strict=True)
        assert digest(model.core.state_dict().items())==loaded['state_digest'];del loaded
        reload_layout=layout(model);reload_digest=state_digest(model)
        assert reload_layout==final_layout and reload_digest==final_digest
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
        write(run/'fit_summary.json',dict(layers=layer_results,fit_states=fit_states,updates=updates,checkpoint=package_meta))
        clear(model);finish(dict(layout=final_layout,reload_layout=reload_layout,final_digest=final_digest,reload_digest=reload_digest,
            evaluation_digest_before=before,evaluation_digest_after=after,frozen_before=stable,frozen_after=frozen_digest(model),
            updates=updates,fit_states=fit_states,checkpoint=package_meta));return 0
    finally:
        monitor.done.set();monitor.thread.join(timeout=6)


if __name__=='__main__':
    assert sys.argv[1] in CASES
    try:sys.exit(main(sys.argv[1],int(sys.argv[2])))
    except Exception:
        write(REPORT/(sys.argv[1]+'_FAILURE.json'),dict(traceback=traceback.format_exc(),no_retry=True));raise
