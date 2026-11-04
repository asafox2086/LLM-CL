"""Exploratory paired generation on the existing 40 TRAIN diagnostic questions."""
from pathlib import Path
import json, hashlib, importlib.util, sys, os, subprocess, traceback
R=Path('/data2/liyapeng_grp/program/LLMCL')
O=R/'exp/routing_strength_f1_20261104/runs/strength2_v1'
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
        write('MODEL_LOADED.json',dict(checkpoint_sha256=runner.CP_SHA,QA=100,arms=['dynamic_x2','fixed_x2'],fixed_reference='same as complete TEST',coefficient_multiplier=2,teacher_forced=False))
        # Existing bitwise generation passthrough controls; no quality scoring.
        control=load('subset_controls',R/'exp/cl_fixed_effective_gate_qwen_donor_20261031/controls.py')
        controller=gate.GateIntervention(method,c,port/'clmoe_adapter.py',cfg['execution_source_sha256']['clmoe_adapter.py'])
        with torch.inference_mode():checks=control.run_controls(controller=controller,method=method,processor=processor,rows=rows,rt=rt,fv=fv)
        controller.close();controller=None;write('CONTROLS_COMPLETE.json',checks)
        class ScaledGate(gate.GateIntervention):
            def __init__(self,*args,dynamic=False,factor=2.,**kw):
                super().__init__(*args,**kw);self.dynamic=dynamic;self.factor=factor
            def apply(self,m,w):
                fixed=super().apply(m,w)
                if self.mode=='passthrough':return fixed*self.factor
                assert self.mode=='fixed'
                return (w if self.dynamic else fixed)*self.factor
        # Factor-one control exercises new multiplication against original
        # dynamic forward, including byte-exact full projection outputs.
        controller=ScaledGate(method,c,port/'clmoe_adapter.py',cfg['execution_source_sha256']['clmoe_adapter.py'],dynamic=True,factor=1.)
        with torch.inference_mode():scaled_checks=control.run_controls(controller=controller,method=method,processor=processor,rows=rows,rt=rt,fv=fv)
        controller.close();controller=None;write('SCALE_ONE_CONTROLS_COMPLETE.json',scaled_checks)
        arms=['dynamic_x2','fixed_x2'];records=[];paired_rng={}
        for arm in arms:
            controller=ScaledGate(method,c,port/'clmoe_adapter.py',cfg['execution_source_sha256']['clmoe_adapter.py'],dynamic=(arm=='dynamic_x2'),factor=2.)
            controller.install();controller.begin('fixed')
            for i,(task,row) in enumerate(selected):
                if arm==arms[0]:paired_rng[row['id']]=rt.rng()
                else:rt.restore_rng(paired_rng[row['id']])
                dest=O/arm/('q%03d'%i)
                with torch.inference_mode():rt.evaluate(method,processor,[task],{(task['task'],'test'):[row]},dest,19,split='test')
                p=dest/'evaluations/stage_19'/('T%02d_test_predictions.jsonl'%task['task'])
                v=read(p);assert v['id']==row['id']
                records.append(dict(arm=arm,id=row['id'],domain=task['domain'],task=task['task'],F1=v['scores']['token_f1'],prediction=v['prediction']))
                print(json.dumps(dict(arm=arm,done=i+1,total=100)),flush=True)
            write(arm+'_GATE_COVERAGE.json',dict(report=controller.report(),effective_mode=arm,coefficient_multiplier=2,original_dynamic_G_scaled=(arm=='dynamic_x2')))
            controller.close();controller=None
            assert storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers
            assert model.core.language_model.generation_config.to_dict()==config
        domains=sorted(pools)
        scores={a:{d:sum(v['F1'] for v in records if v['arm']==a and v['domain']==d)/20 for d in domains} for a in arms}
        write('RESULT.json',dict(records=records,domain_F1=scores,macro_F1={a:sum(v.values())/5 for a,v in scores.items()},TRAIN_only=False,weights_unchanged=True,coefficient_multiplier=2,selection='same 100 prespecified TEST IDs as logit confirmation',comparison='dynamic_x2 vs fixed_x2; x1 predictions reused from verified complete TEST',limits=['changes total strength, not preferential weighting of newest expert','does not establish data scarcity causality']))
        write('COMPLETE.json',dict(result_sha256=sha(O/'RESULT.json')))
    except BaseException:
        write('FAILURE.json',dict(traceback=traceback.format_exc()));raise
    finally:
        if controller:controller.close()
if __name__=='__main__':main()
