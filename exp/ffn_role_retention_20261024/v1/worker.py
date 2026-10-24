"""One fresh serial phase of the two fixed role allocations; no score in worker."""
import ast,gc,hashlib,importlib.util,json,os,random,signal,sys,time,traceback,types
from pathlib import Path
from types import SimpleNamespace
from common import ROOT,HERE,PORT,REPORT,WORK,BASE,CASES,BUDGETS,sha,write,identity,data,ready,train_ast


def main(case,gpu):
    from dispatch import available
    r=ready()
    admission=json.loads((REPORT/'ADMISSION_REVIEW.json').read_bytes())
    assert admission['model_admission'] is True and admission['ready_sha256']==sha(REPORT/'READY.json')
    if gpu not in available():return 75
    assert not (REPORT/(case+'_STARTED.json')).exists()
    who=identity(os.getpid());started=time.monotonic()
    write(REPORT/(case+'_STARTED.json'),dict(**who,started_monotonic=started,unix=time.time(),GPU=gpu,boot_id=r['boot_id'],ready_sha256=sha(REPORT/'READY.json')))
    def expired(*_):raise TimeoutError('Fixed phase model budget exceeded')
    signal.signal(signal.SIGALRM,expired);signal.alarm(BUDGETS[case])
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu);os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    import numpy as np
    import torch
    from transformers import AutoProcessor
    from core import FactorLinear,replace,restore,digest,layout,state_cpu,fold
    sys.path.insert(0,str(PORT))
    spec=importlib.util.spec_from_file_location('role_original_run',PORT/'run.py');formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)
    from methods import Method
    from frozen_vision import install
    m,s,fit,dev,plan=data();run=WORK/case;run.mkdir(parents=True,exist_ok=False)
    for folder in ('logs','validity','gradients','stage_adapters'):(run/folder).mkdir()
    monitor=formal.ResourceMonitor(run)
    monitor.snapshot('process_before_model')
    torch.set_num_threads(8)
    def seed():
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    seed();processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
    def base():
        model=formal.onevision_bridge.Factory.from_pretrained(BASE,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
        model.requires_grad_(False);model.eval();formal.onevision_bridge.bind(model);install(model);return model
    def clear(model):
        model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0;gc.collect();torch.cuda.empty_cache()
    def base_digest(model):
        return digest(list(model.core.named_parameters())+list(model.core.named_buffers()))
    def frozen_digest(model):
        return digest([(n,p) for n,p in model.core.named_parameters() if not p.requires_grad]+list(model.core.named_buffers()))
    checked_images=set()
    def inputs(row,task,processor,device):
        for name in row['images']:
            if name not in checked_images:assert sha(ROOT/name)==s['image_sha256'][name];checked_images.add(name)
        batch,details=formal.input_batch(row,task,processor,device)
        return batch,details
    def evaluate(model,phase):
        model.eval();clear(model);before=base_digest(model);p=run/(phase+'.jsonl');assert not p.exists()
        with p.open('w',encoding='utf8') as f:
            for task in m['tasks']:
                for row in dev[task['task']]:
                    batch,details=inputs(row,task,processor,model.device)
                    input_sha=digest(list(batch.items()))
                    class FiniteScores:
                        def __init__(self):self.calls=0
                        def __call__(self,input_ids,scores):
                            assert not torch.isnan(scores).any() and not torch.isposinf(scores).any()
                            assert torch.isfinite(scores).any(dim=-1).all()
                            self.calls+=1;return scores
                    guard=FiniteScores()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
                        generated=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=task['max_new_tokens'],use_cache=True,
                              pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id,logits_processor=[guard])
                    ids=generated[0,batch['input_ids'].shape[1]:].tolist()
                    record=dict(id=row['id'],task=task['task'],worker=who,input_sha256=input_sha,
                        images={n:s['image_sha256'][n] for n in row['images']},references=row['answers'],
                        output_token_ids=ids,prediction=processor.tokenizer.decode(ids,skip_special_tokens=True),
                        generation_cap=task['max_new_tokens'],eos_token_id=processor.tokenizer.eos_token_id,finite_score_checks=guard.calls,**details)
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            os.fsync(f.fileno())
        assert before==base_digest(model),'Evaluation changed weights or buffers'
        clear(model);return dict(before=before,after=before,prediction_sha256=sha(p))
    def save(model,path,tag):
        clear(model);tick=time.monotonic();state=state_cpu(model.core)
        payload=dict(case=tag,rank=200,state=state,config=model.core.config.to_dict(),generation_config=model.core.generation_config.to_dict(),
                     layout=layout(model),state_digest=digest(state.items()))
        tmp=path.with_suffix('.pt.tmp');torch.save(payload,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        tmp.replace(path);del state,payload
        return dict(path=str(path.relative_to(WORK)),sha256=sha(path),bytes=path.stat().st_size,seconds=time.monotonic()-tick)
    def load(path,tag):
        package=torch.load(path,map_location='cpu',weights_only=False)
        assert package['case']==tag and package['rank']==200
        model=base();branch={'KeepGate':'up_proj','KeepUp':'gate_proj'}[tag]
        for layer in model.model.layers:
            original=getattr(layer.mlp,branch)
            left=torch.zeros((original.out_features,200),device=model.device,dtype=torch.float16)
            right=torch.zeros((200,original.in_features),device=model.device,dtype=torch.float16)
            setattr(layer.mlp,branch,FactorLinear(left,right))
        model.core.load_state_dict(package['state'],strict=True)
        assert digest(model.core.state_dict().items())==package['state_digest']
        assert layout(model)==package['layout']
        del package;gc.collect();return model
    checkpoints=[]
    if case=='init':
        assert sha(BASE/'model.safetensors')=='07b3362c3412de79baf2379e44e5b0b2a8f4b965ebebd11d7b5b3eb4450fe96e'
        model=base();assert layout(model)['weights']['coefficients']==893675552
        parent=base_digest(model);parent_evaluation=evaluate(model,'parent');allocations={}
        for tag in CASES[1:]:
            stash,details=replace(model,tag)
            assert layout(model)['weights']['coefficients']==816728096
            allocations[tag]=details;checkpoints.append(save(model,WORK/(tag+'_CP0.pt'),tag))
            restore(stash);del stash;assert base_digest(model)==parent
            gc.collect();torch.cuda.empty_cache()
        write(run/'initializations.json',dict(parent_digest=parent,parent_layout=layout(model),allocations=allocations,parent_evaluation=parent_evaluation,checkpoints=checkpoints))
        monitor.done.set();monitor.thread.join(timeout=6)
        assert not monitor.thread.is_alive();monitor.snapshot('complete',model=model)
        files={str(p.relative_to(WORK)):sha(p) for p in (run/'parent.jsonl',run/'initializations.json',run/'resources.jsonl')}
        files.update({x['path']:x['sha256'] for x in checkpoints})
        write(REPORT/(case+'_COMPLETE.json'),dict(complete=True,worker=who,ready_sha256=sha(REPORT/'READY.json'),files=files,
                  seconds=time.monotonic()-started,TEST_QA=0));signal.alarm(0);return 0
    model=load(WORK/(case+'_CP0.pt'),case)
    seed();method=Method(model,'SeqLoRA',[t['instruction'] for t in m['tasks']]);method.host_optimizer=True;method.offload_activations=True
    method.vision_original=model._llmcl_visual_original
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==540672
    qv_initial=digest([(n,p) for n,p in model.core.named_parameters() if p.requires_grad])
    initial_evaluation=evaluate(model,'initial')
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    seed();clear(model);frozen_before=frozen_digest(model);monitor.snapshot('training_before',model=model)
    seen=[];counts={};global_step=0
    def training_inputs(row,task,processor,device):
        seen.append(row['id']);return inputs(row,task,processor,device)
    original_clip=torch.nn.utils.clip_grad_norm_
    def record_clip(parameters,*args,**kwargs):
        nonlocal global_step
        parameters=list(parameters);global_step+=1;stage=monitor.stage;counts[stage]=counts.get(stage,0)+1
        gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert len(gradients)==96 and all(torch.isfinite(t).all() for t in gradients.values())
        torch.save(gradients,run/'gradients'/f'{global_step:03d}.pt')
        return original_clip(parameters,*args,**kwargs)
    def optimizer_factory(method):
        optimizer=formal.make_optimizer(method);original_step=optimizer.step
        def recorded(this,*args,**kwargs):
            names={id(p):n for n,p in model.named_parameters() if p.requires_grad}
            actual=[p for group in this.param_groups for p in group['params']]
            assert {id(p) for p in actual}==set(names)
            ids=seen[:];seen.clear();assert ids
            result=original_step(*args,**kwargs)
            moments=[v for state in this.state.values() for k,v in state.items() if k in ('exp_avg','exp_avg_sq')]
            assert all(v.device.type=='cpu' for v in moments)
            record=dict(stage=monitor.stage,step=global_step,ids=ids,parameter_count=sum(p.numel() for p in actual),
                 parameter_ids={names[id(p)]:id(p) for p in actual},positive_lr=any(g['lr']>0 for g in this.param_groups),
                 finite=all(bool(torch.isfinite(p).all()) for p in actual),
                 gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in actual if p.grad is not None),
                 cpu_moments_bytes=sum(v.numel()*v.element_size() for v in moments))
            with (run/'optimizer.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
            return result
        optimizer.step=types.MethodType(recorded,optimizer);return optimizer
    def save_stage(stage):
        state={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        torch.save(state,run/'stage_adapters'/f'{stage:02d}.pt')
    training_data={(t['task'],'train'):fit[t['task']] for t in m['tasks']}
    namespace=dict(vars(formal),model=model,method=method,processor=processor,manifest=m,data=training_data,run=run,monitor=monitor,
                  args=SimpleNamespace(method='SeqFT_LoRA'),resume_obj=None,input_batch=training_inputs,make_optimizer=optimizer_factory,save_stage=save_stage)
    node,train_sha=train_ast();assert train_sha==r['train_AST_sha256']
    torch.nn.utils.clip_grad_norm_=record_clip
    try:exec(compile(node,'<original-TRAIN-prefix>','exec'),namespace)
    finally:torch.nn.utils.clip_grad_norm_=original_clip
    assert global_step==87 and len(counts)==19 and not seen
    frozen_after=frozen_digest(model);assert frozen_before==frozen_after
    model.eval();model.gradient_checkpointing_disable() if hasattr(model,'gradient_checkpointing_disable') else model.core.language_model.gradient_checkpointing_disable()
    folded=fold(model);assert len(folded)==48
    final_layout=layout(model);assert final_layout['weights']['coefficients']==816728096
    final_digest=base_digest(model);final_meta=save(model,run/'final_complete.pt',case)
    # Release every training reference before the actual complete-package reload.
    namespace.clear();method.model=None;method.layers.clear();del method,model
    formal.onevision_bridge.ACTIVE=None;gc.collect();torch.cuda.empty_cache()
    model=load(run/'final_complete.pt',case);reload_layout=layout(model);reload_digest=base_digest(model)
    assert reload_layout==final_layout and reload_digest==final_digest
    final_evaluation=evaluate(model,'final')
    write(run/'result_proof.json',dict(initial_evaluation=initial_evaluation,final_evaluation=final_evaluation,folded_layers=folded,
          final_checkpoint=final_meta,counts=counts,initial_qv_sha256=qv_initial))
    monitor.done.set();monitor.thread.join(timeout=6);assert not monitor.thread.is_alive();monitor.snapshot('complete',model=model)
    files={str(p.relative_to(WORK)):sha(p) for p in run.rglob('*') if p.is_file()}
    write(REPORT/(case+'_COMPLETE.json'),dict(complete=True,worker=who,ready_sha256=sha(REPORT/'READY.json'),files=files,updates=87,stages=19,FIT_QA=1213,
           frozen_before=frozen_before,frozen_after=frozen_after,final_layout=final_layout,reload_layout=reload_layout,final_digest=final_digest,reload_digest=reload_digest,
           qv_initial_sha256=qv_initial,seconds=time.monotonic()-started,TEST_QA=0))
    signal.alarm(0);return 0


if __name__=='__main__':
    case=sys.argv[1];assert case in CASES
    try:sys.exit(main(case,int(sys.argv[2])))
    except Exception:
        write(REPORT/(case+'_FAILURE.json'),dict(traceback=traceback.format_exc(),no_retry=True));raise
