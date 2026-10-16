"""Fixed-anchor task-count/domain-count interventions and CL mechanism controls.

Eight original frozen TRAIN QA/task, microbatch2, four positive-lr updates/task.
Exploratory anchor diagnostic subset only; no claim of full benchmark results.
"""
import hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'exp/onevision_diagnostics_20261016'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'analysis'))
from run import freeze,write
from onevision_research_worker import atomic_torch,digest
from onevision_knowledge_intervention import Intervention,case_complete,valid_checkpoint,reset_seed

SEQUENCES={'repeat_anchor':[4,4,4,4,4],'within_medical':[4,2,3,6,11],
           'cross_short':[4,1,13,15,16],'cross_five_domains':[4,1,13,16,18]}
METHODS=['SeqFT_LoRA','ER_LoRA','EWC_lambda1','EWC_lambda100','anchor_gradient_noop','anchor_gradient_projection']

def flat_gradient(parameters):
    return torch.cat([p.grad.detach().float().flatten() if p.grad is not None else torch.zeros(p.numel(),device=p.device) for p in parameters])

def project(current,reference):
    dot=torch.dot(current.double(),reference.double());norm=reference.double().square().sum()
    if float(dot)<0 and float(norm)>0:current=current-(dot/norm).to(current.dtype)*reference
    return current,float(dot)

class Mechanism(Intervention):
    def fisher(self,rows):
        self.model.eval();named=[(k,p) for k,p in self.model.named_parameters() if p.requires_grad]
        diagonal={k:torch.zeros_like(p,device='cpu',dtype=torch.float32) for k,p in named}
        for row in rows[:4]:
            self.model.zero_grad(set_to_none=True);loss,_=self.loss(row);loss.backward()
            for k,p in named:
                if p.grad is not None:diagonal[k].add_(p.grad.detach().float().cpu().square()/min(4,len(rows)))
        self.model.zero_grad(set_to_none=True)
        return {'diagonal':diagonal,'mean':{k:p.detach().float().cpu().clone() for k,p in named},'ids':[r['id'] for r in rows[:4]]}

    def step_task(self,case,stage,rows,method,history,anchor,fishers):
        reset_seed();self.model.train();named=[(k,p) for k,p in self.model.named_parameters() if p.requires_grad];parameters=[p for _,p in named]
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
        memory=sorted({r['id']:r for r in history}.values(),key=lambda r:hashlib.sha256(('42:memory:'+r['id']).encode()).digest())[:8]
        self.monitor.snapshot(f'{case}_stage{stage}_before',model=self.model)
        exposures=0;records=[]
        for offset in range(0,8,2):
            reference=None
            if method.startswith('anchor_gradient_') and stage>1:
                self.model.eval();optimizer.zero_grad(set_to_none=True)
                for row in anchor:
                    loss,_=self.loss(row);(loss/len(anchor)).backward()
                reference=flat_gradient(parameters).detach();self.model.train()
            optimizer.zero_grad(set_to_none=True);batch=rows[offset:offset+2];replay=[]
            if method=='ER_LoRA' and memory:
                for row in batch:
                    index=int.from_bytes(hashlib.sha256(f"42:{stage}:{row['id']}".encode()).digest()[:8],'big')%len(memory);replay.append(memory[index])
            all_rows=batch+replay;losses=[];target_tokens=0
            for row in all_rows:
                loss,details=self.loss(row);assert torch.isfinite(loss);(loss/len(all_rows)).backward();losses.append(float(loss.detach()));target_tokens+=details['target_tokens']
            penalty_value=0.
            if method.startswith('EWC_') and fishers:
                coefficient=1. if method=='EWC_lambda1' else 100.
                penalty=sum((f['diagonal'][k].to(p.device)*(p.float()-f['mean'][k].to(p.device)).square()).sum() for f in fishers for k,p in named)*(.5*coefficient)
                assert torch.isfinite(penalty);penalty.backward();penalty_value=float(penalty.detach())
            current=flat_gradient(parameters);raw=current.detach().cpu().clone();dot=None
            if reference is not None:
                projected,dot=project(current,reference)
                if method=='anchor_gradient_projection':current=projected
                position=0
                for p in parameters:
                    if p.grad is None:p.grad=torch.zeros_like(p)
                    p.grad.copy_(current[position:position+p.numel()].reshape_as(p));position+=p.numel()
            grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad);optimizer.step();exposures+=len(all_rows)
            records.append({'stage':stage,'update':offset//2+1,'train_ids':[r['id'] for r in batch],'replay_ids':[r['id'] for r in replay],
                'loss':sum(losses)/len(losses),'penalty':penalty_value,'gradient_l2_preclip':float(grad),'lr_used':1e-4,
                'anchor_gradient_dot':dot,'target_tokens':target_tokens,'memory_size':len(memory) if method=='ER_LoRA' else 0,
                'projection_applied':method=='anchor_gradient_projection' and dot is not None and dot<0,
                'reference_extra_QA':len(anchor) if reference is not None else 0})
            if offset==0:
                atomic_torch(self.run/'raw_step_gradients'/case/f'stage_{stage:02d}.pt',{'pre_intervention_gradient':raw,
                    'reference_gradient':reference.detach().cpu() if reference is not None else None,'parameter_layout':[(k,list(p.shape)) for k,p in named],
                    'train_ids':[r['id'] for r in batch],'reference_train_ids':[r['id'] for r in anchor] if reference is not None else []})
            print(json.dumps({'event':'mechanism_update','case':case,**records[-1]}),flush=True)
        optimizer.zero_grad(set_to_none=True);del optimizer
        self.monitor.snapshot(f'{case}_stage{stage}_after',model=self.model)
        write(self.run/'training'/case/f'stage_{stage:02d}.json',{'original_train_qa':8,'updates':4,'new_plus_replay_exposures':exposures,
            'fisher_extra_qa_next':min(4,len(rows)) if method.startswith('EWC_') else 0,'events':records})

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(6);reset_seed()
    manifest,data,domains,tests,generation,probes,original_protocol=freeze()
    selected={t:sorted(data[t,'train'],key=lambda r:hashlib.sha256(('42:task-domain:'+r['id']).encode()).digest())[:8]
              for t in {t for seq in SEQUENCES.values() for t in seq}}
    assert all(len(rows)==8 for rows in selected.values())
    anchor=[r for r in tests['medical'] if r['_task']==4];assert len(anchor)==23
    report=ROOT/'summary/onevision_mechanisms_20261016';report.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'analysis/onevision_knowledge_intervention.py',HERE/'run.py']}
    spec={'train_ids':{str(t):[r['id'] for r in rows] for t,rows in selected.items()},'anchor_test_ids':[r['id'] for r in anchor],
          'sequences':SEQUENCES,'methods':METHODS,'source_sha256':hashes,'seed':42,'qa_per_task':8,'updates_per_task':4,
          'test_status':'23 frozen anchor probes; exploratory, not full507 anchor test or full21454 five-domain test',
          'method_scope':'standard replay and empirical-diagonal EWC controls; anchor-gradient projection is a causal intervention, not a newly published algorithm',
          'confounds':['Task contents and target lengths differ across domains. cross_short removes traffic long answers but is four domains, not five.',
                       'ER and projection/Fisher access extra historical TRAIN data/compute; record exposures, not falsely equal total compute.',
                       'Single medical anchor; robust attribution needs more anchors/orders, not only this pilot.']}
    pointer=report/'task_domain_attempt.json'
    if pointer.exists():
        info=json.loads(pointer.read_text());assert info['spec']==spec;run=ROOT/info['run']
    else:
        run=ROOT/'exp/onevision_mechanisms_20261016/task_domain'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
        write(pointer,{'run':str(run.relative_to(ROOT)),'spec':spec});write(run/'protocol.json',spec)
    x=Mechanism(manifest,run);initial=x.original_initial
    gate=run/'mechanism_gpu_gate.json'
    if not gate.exists():
        checked=[]
        for method in METHODS:
            x.configure(initial,finetune_shared=True);fishers=[];name='gate_'+method
            x.step_task(name,1,selected[4],method,[],selected[4],fishers)
            if method.startswith('EWC_'):fishers.append(x.fisher(selected[4]))
            x.step_task(name,2,selected[2],method,selected[4],selected[4],fishers)
            assert all(torch.isfinite(p).all() for p in x.model.parameters() if p.requires_grad)
            checked.append(method)
        write(gate,{'passed':True,'methods':checked,'native_task_ids':[4,2],'finite_forward_backward_optimizer':True,
                    'diagnostic_updates_discarded':True,'source_sha256':hashes})
    # Native base reference, same exact anchor questions as every condition.
    x.configure(initial,finetune_shared=True)
    if not case_complete(run,'anchor_base',{'anchor':anchor}):x.model.eval();x.evaluate('anchor_base',{'anchor':anchor},{'anchor':anchor})
    for sequence,tasks in SEQUENCES.items():
        for method in METHODS:
            case=sequence+'__'+method
            if (run/'cases'/f'{case}_complete.json').exists():continue
            x.configure(initial,finetune_shared=True);fishers=[];start=1
            latest=run/'case_states'/case/'latest.pt'
            if valid_checkpoint(latest):
                saved=torch.load(latest,map_location='cpu',weights_only=True);assert saved['source_sha256']==hashes
                x.restore(saved['adapter']);fishers=saved['fishers'];start=saved['trained_stage']
            else:saved={'trained_stage':0}
            for stage in range(start,6):
                write(report/'status.json',{'phase':'task_domain_intervention','case':case,'stage':stage,'run':str(run.relative_to(ROOT)),'pid':os.getpid()})
                task=tasks[stage-1];rows=selected[task]
                if stage>saved['trained_stage']:
                    history=[row for previous in tasks[:stage-1] for row in selected[previous]]
                    x.step_task(case,stage,rows,method,history,selected[4],fishers)
                    if method.startswith('EWC_'):fishers.append(x.fisher(rows))
                    saved={'trained_stage':stage,'adapter':x.snapshot(),'fishers':fishers,'source_sha256':hashes}
                    atomic_torch(latest,saved)
                    atomic_torch(run/'adapters'/case/f'stage_{stage:02d}.pt',{'adapter':saved['adapter'],'rank':8,'alpha':32,'native_task':task})
                if stage in [1,3,5]:
                    label=f'{case}_after{stage}'
                    if not case_complete(run,label,{'anchor':anchor}):x.model.eval();x.evaluate(label,{'anchor':anchor},{'anchor':anchor})
                # Plasticity on current native-task probes, same IDs for the same task across all methods.
                current=[r for ds in tests.values() for r in ds if r['_task']==task][:20]
                label=f'{case}_plasticity{stage}'
                if not case_complete(run,label,{'current_task':current}):x.model.eval();x.evaluate(label,{'current_task':current},{'current_task':current})
            write(run/'cases'/f'{case}_complete.json',{'complete':True,'source_sha256':hashes})
    x.monitor.snapshot('training_after',model=x.model);x.monitor.finish(x.model)
    write(report/'task_domain_complete.json',{'complete':True,'run':str(run.relative_to(ROOT)),'formal_full_test':False,'source_sha256':hashes})

if __name__=='__main__':main()
