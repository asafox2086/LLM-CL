"""Exploratory paired generation on the existing 40 TRAIN diagnostic questions."""
from pathlib import Path
import json, hashlib, importlib.util, sys, os, subprocess, traceback
R=Path('/data2/liyapeng_grp/program/LLMCL')
O=R/'exp/expert_mass_control_20261104/runs/diagnostic_v1'
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
    write('STARTED.json',dict(pid=os.getpid(),source_sha256=sha(__file__),TRAIN_only=True))
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
        ids={q for d in candidates['split'].values() for q in d['diagnose']}
        selected=[(t,r) for t,r in rows if r['id'] in ids];assert len(selected)==40
        c={n:v['C64'] for n,v in candidates['candidates']['global']['layers'].items()}
        # Complementary halves chosen before looking at generation outcomes.
        arms={'all_scaled_eighth':{n:[x/8 for x in v] for n,v in c.items()}}
        for k in range(8):
            arms['only_mass_matched_'+str(k)]={n:[sum(v) if j==k else 0. for j in range(8)] for n,v in c.items()}
        write('MODEL_LOADED.json',dict(checkpoint_sha256=runner.CP_SHA,QA=40,arms=list(arms),subset_mass_preserved='only_mass_matched yes; all_scaled_eighth no',expert_slots_not_semantically_aligned=True))
        # Existing bitwise generation passthrough controls; no quality scoring.
        control=load('subset_controls',R/'exp/cl_fixed_effective_gate_qwen_donor_20261031/controls.py')
        controller=gate.GateIntervention(method,c,port/'clmoe_adapter.py',cfg['execution_source_sha256']['clmoe_adapter.py'])
        with torch.inference_mode():checks=control.run_controls(controller=controller,method=method,processor=processor,rows=rows,rt=rt,fv=fv)
        controller.close();controller=None;write('CONTROLS_COMPLETE.json',checks)
        paired_rng={};records=[]
        for arm in list(arms):
            if arm!='dynamic':
                controller=gate.GateIntervention(method,arms[arm],port/'clmoe_adapter.py',cfg['execution_source_sha256']['clmoe_adapter.py']);controller.install();controller.begin('fixed')
            for i,(task,row) in enumerate(selected):
                if arm==next(iter(arms)):paired_rng[row['id']]=rt.rng()
                else:rt.restore_rng(paired_rng[row['id']])
                dest=O/arm/f'q{i:02d}'
                # Native generate, decoding and scoring; split name explicitly TRAIN.
                with torch.inference_mode():rt.evaluate(method,processor,[task],{(task['task'],'train'):[row]},dest,19,split='train')
                p=dest/'evaluations/stage_19'/f"T{task['task']:02d}_train_predictions.jsonl"
                v=json.loads(p.read_text());assert v['id']==row['id']
                records.append(dict(arm=arm,id=row['id'],domain=task['domain'],F1=v['scores']['token_f1'],prediction=v['prediction']))
                print(json.dumps(dict(arm=arm,done=i+1,total=40)),flush=True)
            if controller:controller.close();controller=None
            assert storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers
            assert model.core.language_model.generation_config.to_dict()==config
        domains=sorted({t['domain'] for t,_ in selected})
        summary={a:{d:sum(v['F1'] for v in records if v['arm']==a and v['domain']==d)/8 for d in domains} for a in list(arms)}
        write('RESULT.json',dict(records=records,domain_F1=summary,macro_F1={a:sum(v.values())/5 for a,v in summary.items()},TRAIN_exploratory_only=True,weights_unchanged=True,subset_rule='8 single slots matched to full coefficient sum, and all experts scaled by 1/8; coefficient sum is not output norm; no repeated prior arms'))
        write('COMPLETE.json',dict(result_sha256=sha(O/'RESULT.json')))
    except BaseException:
        write('FAILURE.json',dict(traceback=traceback.format_exc()));raise
    finally:
        if controller:controller.close()
if __name__=='__main__':main()
