"""Shared trained S1 prefix, followed by two independent boundary-intervention branches."""
import ast,gc,hashlib,importlib.util,json,math,os,random,signal,sys,time,traceback
from pathlib import Path
from types import MethodType,SimpleNamespace
from common import ROOT,HERE,PORT,BASE,REPORT,WORK,CASES,BUDGETS,TRAINABLE,sha,write,identity,data,ready,training_prefix


def main(case,gpu):
    r=ready();admission=json.loads((REPORT/'ADMISSION_REVIEW.json').read_bytes())
    assert admission['model_admission'] and admission['ready_sha256']==sha(REPORT/'READY.json')
    from dispatch import available
    if gpu not in available():return 75
    import shutil
    assert shutil.disk_usage(ROOT).free>24*1024**3
    assert not (REPORT/(case+'_STARTED.json')).exists()
    tick=time.monotonic();who=identity(os.getpid())
    write(REPORT/(case+'_STARTED.json'),dict(**who,started_monotonic=tick,unix=time.time(),GPU=gpu,
        boot_id=r['boot_id'],ready_sha256=sha(REPORT/'READY.json')))
    def expired(*unused):raise TimeoutError('Fixed OSFT stage-refresh phase budget')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,BUDGETS[case])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from core import StageMethod,digest,layout,cpu_state,rng_digest
    sys.path.insert(0,str(PORT))
    import runtime_utils as rt,onevision_bridge as bridge
    from frozen_vision import install as install_visual,prepare
    from resource_monitor import ResourceMonitor
    spec=importlib.util.spec_from_file_location('stage_refresh_original_training_run',PORT/'run.py')
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)
    torch.set_num_threads(8);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    run=WORK/case;run.mkdir(parents=True,exist_ok=False)
    for name in ('logs','validity','witness','stages','boundaries'):(run/name).mkdir()
    monitor=ResourceMonitor(run);monitor.snapshot('before_model')
    processor=AutoProcessor.from_pretrained(BASE,local_files_only=True);assert isinstance(processor.tokenizer.eos_token_id,int)
    m,s,fit,dev,plan=data();checked=set();known_hashes={};updates=0;exposures=0
    def seed():random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    def state_digest(model):return digest(list(model.core.named_parameters())+list(model.core.named_buffers()))
    def stable_digest(model):
        ignored={'U_high','S_high','V_high','U_low','S_low','V_low'}
        return digest([(n,t) for n,t in list(model.core.named_parameters())+list(model.core.named_buffers()) if n.rsplit('.',1)[-1] not in ignored])
    def clear(model):model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0;gc.collect();torch.cuda.empty_cache()
    def base_and_method():
        model=bridge.Factory.from_pretrained(BASE,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
        bridge.bind(model);method=StageMethod(model,case)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        method.vision_original=install_visual(model)
        assert len(method.layers)==168 and sum(p.numel() for p in model.parameters() if p.requires_grad)==TRAINABLE
        return model,method
    def input_batch(row,task,processor,device):
        for image in row['images']:
            if image not in checked:assert sha(ROOT/image)==s['image_sha256'][image];checked.add(image)
        return rt.input_batch(row,task,processor,device)
    def save(path,value):
        with path.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
        meta=dict(bytes=path.stat().st_size,sha256=sha(path));known_hashes[str(path.relative_to(WORK))]=meta['sha256'];return meta
    def finish(extra):
        monitor.done.set();monitor.thread.join(timeout=6);assert not monitor.thread.is_alive();monitor.snapshot('complete')
        files={str(p.relative_to(WORK)):known_hashes.get(str(p.relative_to(WORK))) or sha(p) for p in run.rglob('*') if p.is_file()}
        write(REPORT/(case+'_COMPLETE.json'),dict(complete=True,worker=who,seconds=time.monotonic()-tick,
            ready_sha256=sha(REPORT/'READY.json'),files=files,actual_updates=updates,actual_FIT=exposures,TEST_QA=0,**extra))
        signal.setitimer(signal.ITIMER_REAL,0)
    def generate(model,method,tasks,path,phase):
        before=state_digest(model);model.eval()
        with path.open('a',encoding='utf8') as stream:
            for task in tasks:
                method.set_instruction(task['task'])
                for row in dev[task['task']]:
                    batch,details=input_batch(row,task,processor,model.device);prepare(batch,row,model,method.vision_original)
                    # Bind the prepared tensors actually supplied to the model in every state.
                    input_sha=digest(list(batch.items()))
                    class Finite:
                        calls=0
                        def __call__(self,ids,scores):
                            assert not torch.isnan(scores).any() and not torch.isposinf(scores).any() and torch.isfinite(scores).any(-1).all()
                            self.calls+=1;return scores
                    guard=Finite()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                        result=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=task['max_new_tokens'],use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[guard])
                    ids=result[0,batch['input_ids'].shape[1]:].tolist()
                    record=dict(id=row['id'],task=task['task'],state=phase,references=row['answers'],input_sha256=input_sha,
                        images={n:s['image_sha256'][n] for n in row['images']},generation_cap=task['max_new_tokens'],
                        eos_token_id=processor.tokenizer.eos_token_id,output_token_ids=ids,finite_score_checks=guard.calls,
                        prediction=processor.tokenizer.decode(ids,skip_special_tokens=True),**details)
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        after=state_digest(model);assert before==after;return after
    try:
        assert sha(BASE/'model.safetensors')=='07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e'
        model,method=base_and_method();stable=stable_digest(model);loaded_shared=None
        if case=='prefix':
            initial=state_digest(model);generate(model,method,m['tasks'],run/'initial.jsonl','initial')
            seed();stages=[1]
        else:
            prefix=json.loads((REPORT/'prefix_COMPLETE.json').read_bytes())
            shared=WORK/'prefix/shared_S1.pt';assert sha(shared)==prefix['checkpoint']['sha256']
            package=torch.load(shared,map_location='cpu',weights_only=False)
            assert package['method_stage']==1 and package['train_updates']==21 and package['train_FIT']==327
            model.core.load_state_dict(package['state'],strict=True)
            method.stage=package['method_stage'];assert method.stage==1
            assert state_digest(model)==package['state_digest']==prefix['shared_S1_digest']
            assert stable_digest(model)==stable
            rt.restore_rng(package['rng'])
            assert rng_digest(rt.rng())==package['rng_digest']==prefix['shared_rng_digest']
            loaded_shared=dict(checkpoint=prefix['checkpoint'],digest=prefix['shared_S1_digest'],rng_sha256=prefix['shared_rng_sha256'],rng_digest=prefix['shared_rng_digest'])
            restored_method_stage=method.stage;del package;stages=list(range(2,20))
        method.write_boundary=lambda stage,record:write(run/'boundaries'/f'S{stage:02d}.json',record)
        rows_nodes,ast_sha=training_prefix();assert ast_sha==r['training_ast_sha256']
        train_module=compile(ast.fix_missing_locations(ast.Module(body=rows_nodes,type_ignores=[])),'original-TRAIN-prefix','exec')
        mapped={(k,'train'):v for k,v in fit.items()}
        namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=m,data=mapped,
            run=run,monitor=monitor,args=SimpleNamespace(method='OSFT'),resume_obj=None)
        witness_steps={}
        def measured_optimizer(current):
            opt=rt.make_optimizer(current);original=type(opt).step
            expected={id(p) for layer in current.layers for p in (layer.U_low,layer.S_low,layer.V_low)}
            assert expected=={id(p) for group in opt.param_groups for p in group['params']}
            def step(self,*args,**kwargs):
                nonlocal updates
                params=[p for group in self.param_groups for p in group['params']]
                assert {id(p) for p in params}==expected and sum(p.numel() for p in params)==TRAINABLE
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in params)
                old_steps=sum(float(v.get('step',0)) for v in self.state.values())
                value=original(self,*args,**kwargs)
                assert sum(float(v.get('step',0)) for v in self.state.values())>old_steps
                assert all(torch.isfinite(p).all() for p in params)
                moments=[v[k] for v in self.state.values() for k in ('exp_avg','exp_avg_sq')]
                assert all(x.device.type=='cpu' for x in moments)
                updates+=1;stage=current.stage;witness_steps[stage]=witness_steps.get(stage,0)+1
                write(run/'witness'/f'S{stage:02d}_{witness_steps[stage]:03d}.json',dict(stage=stage,step=witness_steps[stage],
                    finite=True,actual_step=True,positive_lr=all(g['lr']>0 for g in self.param_groups),
                    coordinates=TRAINABLE,parameter_ids={n:id(p) for n,p in model.named_parameters() if p.requires_grad},
                    gradient_bytes=gradient_bytes,cpu_moments_bytes=sum(x.numel()*x.element_size() for x in moments)))
                return value
            opt.step=MethodType(step,opt);return opt
        journal=(run/'train_inputs.jsonl').open('x',encoding='utf8')
        def training_input(row,task,processor,device):
            nonlocal exposures
            batch,details=input_batch(row,task,processor,device);exposures+=1
            journal.write(json.dumps(dict(id=row['id'],stage=task['task'],input_sha256=digest(list(batch.items()))))+'\n')
            journal.flush();os.fsync(journal.fileno());return batch,details
        namespace.update(make_optimizer=measured_optimizer,input_batch=training_input)
        clear(model);monitor.snapshot('training_before',model=model)
        try:
            for stage in stages:
                namespace['stage']=stage;exec(train_module,namespace)
                after_train=state_digest(model)
                after_eval=generate(model,method,[m['tasks'][stage-1]],run/'acquired.jsonl',f'acquired_S{stage:02d}')
                assert after_train==after_eval and stable_digest(model)==stable
                write(run/'stages'/f'S{stage:02d}.json',dict(stage=stage,method_stage=method.stage,
                    digest_after_train=after_train,digest_after_acquired=after_eval,layout=layout(model),
                    updates=witness_steps[stage],FIT=len(fit[stage])))
        finally:journal.close()
        assert exposures==sum(len(fit[k]) for k in stages) and updates==sum(plan[k-1]['updates'] for k in stages)
        if case=='prefix':
            clear(model);current=state_digest(model);rng=rt.rng()
            rng_path=run/'rng.pt';rng_meta=save(rng_path,rng)
            checkpoint=save(run/'shared_S1.pt',dict(state=cpu_state(model),state_digest=current,rng=rng,rng_digest=rng_digest(rng),
                method_stage=method.stage,train_updates=updates,train_FIT=exposures))
            finish(dict(layout=layout(model),checkpoint=checkpoint,shared_S1_digest=current,shared_rng_sha256=rng_meta['sha256'],shared_rng_digest=rng_digest(rng),
                unchanged_before=stable,unchanged_after=stable_digest(model),initial_digest=initial));return 0
        assert len(method.boundary_rows)==18
        clear(model);model.requires_grad_(False);model.eval();final_layout=layout(model);final_digest=state_digest(model)
        state=cpu_state(model);checkpoint=save(run/'final_complete.pt',dict(state=state,state_digest=final_digest,method_stage=19,layout=final_layout))
        # Namespace also owns the old model/parameters; drop it before the actual independent reload.
        bridge.ACTIVE=None;del namespace,state,method,model;gc.collect();torch.cuda.empty_cache()
        model,method=base_and_method();package=torch.load(run/'final_complete.pt',map_location='cpu',weights_only=False)
        assert package['method_stage']==19;method.stage=19
        model.core.load_state_dict(package['state'],strict=True);model.requires_grad_(False);del package
        reload_layout=layout(model);reload_digest=state_digest(model)
        assert reload_layout==final_layout and reload_digest==final_digest and stable_digest(model)==stable
        before=state_digest(model);after=generate(model,method,m['tasks'],run/'final.jsonl','final_S19')
        clear(model);finish(dict(layout=final_layout,reload_layout=reload_layout,checkpoint=checkpoint,
            final_digest=final_digest,reload_digest=reload_digest,evaluation_digest_before=before,evaluation_digest_after=after,
            unchanged_before=stable,unchanged_after=stable_digest(model),loaded_shared_S1=loaded_shared,restored_method_stage=restored_method_stage));return 0
    finally:
        monitor.done.set();monitor.thread.join(timeout=6)


if __name__=='__main__':
    assert sys.argv[1] in CASES
    try:sys.exit(main(sys.argv[1],int(sys.argv[2])))
    except Exception:
        write(REPORT/(sys.argv[1]+'_FAILURE.json'),dict(traceback=traceback.format_exc(),no_retry=True));raise
