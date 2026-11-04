"""Exploratory paired generation on the existing 40 TRAIN diagnostic questions."""
from pathlib import Path
import json, hashlib, importlib.util, sys, os, subprocess, traceback
R=Path('/data2/liyapeng_grp/program/LLMCL')
O=R/'exp/routing_logit_heldout_20261104/runs/confirmation_v1'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (O/n).open('x') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def main():
    b=read(R/'summary/routing_value_probe_20261104/BINDING.json')
    assert sha(b['runner'])==b['sources']['runner']
    runner=load('subset_original_scope',b['runner'])
    old,port,run,cfg,rows,candidates,_=runner.scope(b)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
    assert not subprocess.check_output(['nvidia-smi','--id=0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    O.mkdir(parents=True,exist_ok=False)
    write('STARTED.json',dict(pid=os.getpid(),source_sha256=sha(__file__),TRAIN_only=False))
    controller=None
    try:
        import torch
        torch.set_num_threads(8);sys.path.insert(0,str(port))
        import checkpoints,runtime_utils as rt,onevision_bridge as bridge,frozen_vision as fv
        for m in (checkpoints,rt,bridge,fv):assert sha(m.__file__)==cfg['execution_source_sha256'][Path(m.__file__).name]
        assert sha(run/'checkpoints/stage_19.pt')==runner.CP_SHA
        payload=checkpoints.load(run,expected=cfg);assert payload['stage']==19
        method=payload['method'];processor=payload['processor'];del payload
        model=method.model.eval();model.requires_grad_(False);bridge.bind(model);bridge.NATIVE_GATE_MODE=False
        storage=load('subset_storage',old['storage_helper']);gate=load('subset_gate',b['gate'])
        probe=load('subset_nll_controls',b['probe'])
        before=storage.digest(model.core);buffers=storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
        config=model.core.language_model.generation_config.to_dict()
        full_binding=read(R/'summary/cl_fixed_unlimited_20261103/BINDING.json')
        ref=Path(full_binding['old_TF_c_path'])
        fullscope=load('heldout_full_scope',R/'exp/cl_fixed_unlimited_20261103/run.py')
        assert sha(ref)==fullscope.REF_SHA
        c=read(ref)['methods']['CL_MoE']
        manifest=read(Path(old['manifest']))
        pools={}
        for task in manifest['tasks']:
            path=R/task['files']['test']['path']
            assert sha(path)==task['files']['test']['sha256']
            examples=[json.loads(line) for line in path.read_text().splitlines()]
            pools.setdefault(task['domain'],[]).extend((task,row) for row in examples)
        selected=[]
        for domain,pool in sorted(pools.items()):
            pool.sort(key=lambda x:hashlib.sha256(('routing-logit-confirm-v1:'+x[1]['id']).encode()).digest())
            selected.extend(pool[:20])
        assert len(selected)==100 and len({row['id'] for _,row in selected})==100
        write('SELECTION.json',dict(rule='SHA256 routing-logit-confirm-v1 plus id; first20 per domain; no score-based selection',ids=[row['id'] for _,row in selected],fixed_source=str(ref),fixed_source_sha256=sha(ref)))
        write('MODEL_LOADED.json',dict(checkpoint_sha256=runner.CP_SHA,QA=100,arms=['off','fixed','dynamic'],fixed_reference='same as complete TEST',teacher_forced=True))
        # Existing bitwise generation passthrough controls; no quality scoring.
        control=load('subset_controls',R/'exp/cl_fixed_effective_gate_qwen_donor_20261031/controls.py')
        controller=gate.GateIntervention(method,c,port/'clmoe_adapter.py',cfg['execution_source_sha256']['clmoe_adapter.py'])
        with torch.inference_mode():checks=control.run_controls(controller=controller,method=method,processor=processor,rows=rows,rt=rt,fv=fv)
        controller.close();controller=None;write('CONTROLS_COMPLETE.json',checks)
        records=[]
        for i,(task,row) in enumerate(selected):
            target=processor.tokenizer(row['answer'],add_special_tokens=False,truncation=True,max_length=512)['input_ids']+[processor.tokenizer.eos_token_id]
            indices=sorted(set(round(j*(len(target)-1)/max(1,min(32,len(target))-1)) for j in range(min(32,len(target)))))
            values={};state=rt.rng()
            for arm in ['off','fixed','dynamic']:
                rt.restore_rng(state)
                if arm!='dynamic':
                    coeff={n:[0.]*8 for n in c} if arm=='off' else c
                    controller=gate.GateIntervention(method,coeff,port/'clmoe_adapter.py',cfg['execution_source_sha256']['clmoe_adapter.py'])
                    controller.install();controller.begin('fixed')
                safe={k:row[k] for k in ['id','question','images','context'] if k in row}
                method.set_instruction(task['task'])
                batch,_=rt.input_batch(safe,task,processor,model.device)
                fv.prepare(batch,safe,model,method.vision_original);method.prepare_row(safe,batch)
                captured=[]
                def capture(module,args,out):
                    assert out.ndim==3 and out.shape[1]==len(target)
                    captured.append(out[0,indices].detach().float().cpu())
                hook=model.lm_head.register_forward_hook(capture)
                try:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
                        loss,n=rt.answer_loss(model,batch,row['answer'],processor)
                    assert n==len(target) and len(captured)==1
                    values[arm]=captured[0]
                finally:
                    hook.remove()
                    if controller:controller.close();controller=None
            labels=torch.tensor([target[j] for j in indices])
            def centered(x):return x-x.mean(-1,keepdim=True)
            base_change=centered(values['fixed']-values['off'])
            route_change=centered(values['dynamic']-values['fixed'])
            def margin(x):
                gold=x.gather(1,labels[:,None]).squeeze(1)
                other=x.clone();other.scatter_(1,labels[:,None],float('-inf'))
                return gold-other.max(-1).values
            m={k:margin(v) for k,v in values.items()}
            pred={k:v.argmax(-1) for k,v in values.items()}
            record=dict(id=row['id'],domain=task['domain'],target_count=n,selected_target_positions=indices,
                fixed_minus_off_centered_logit_squared=float(base_change.double().square().sum()),
                dynamic_minus_fixed_centered_logit_squared=float(route_change.double().square().sum()),
                fixed_off_top1_changes=int((pred['fixed']!=pred['off']).sum()),
                dynamic_fixed_top1_changes=int((pred['dynamic']!=pred['fixed']).sum()),
                gold_top1_counts={k:int((v==labels).sum()) for k,v in pred.items()},
                gold_margin={k:v.tolist() for k,v in m.items()},
                fixed_off_gold_margin_abs_mean=float((m['fixed']-m['off']).abs().mean()),
                dynamic_fixed_gold_margin_abs_mean=float((m['dynamic']-m['fixed']).abs().mean()))
            records.append(record);write('QA_%02d.json'%i,record)
            print('QA',i+1,flush=True)
        assert storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers
        assert model.core.language_model.generation_config.to_dict()==config
        denominator=sum(x['fixed_minus_off_centered_logit_squared'] for x in records)
        numerator=sum(x['dynamic_minus_fixed_centered_logit_squared'] for x in records)
        write('RESULT.json',dict(records=records,centered_logit_energy_ratio=numerator/denominator,
            relative_centered_logit_L2=(numerator/denominator)**.5,
            sampled_answer_positions=sum(len(x['selected_target_positions']) for x in records),
            fixed_off_top1_changes=sum(x['fixed_off_top1_changes'] for x in records),
            dynamic_fixed_top1_changes=sum(x['dynamic_fixed_top1_changes'] for x in records),
            TRAIN_only=False,teacher_forced=True,no_generation_F1_claim=True,
            sampling='at most 32 evenly spaced answer positions per QA; same gold prefix across arms',
            fixed_reference='identical coefficients to verified full TEST comparison'))
        write('COMPLETE.json',dict(result_sha256=sha(O/'RESULT.json')))
    except BaseException:
        write('FAILURE.json',dict(traceback=traceback.format_exc()));raise
    finally:
        if controller:controller.close()
if __name__=='__main__':main()
