"""FormalV2-input version of the fixed continuous-state-cache candidate.

This driver reuses the exact formal SeqFT training trajectory. It is deliberately
outside the frozen baseline tree. No baseline sources or hyperparameters change.
GPU admission is separate; do not run beside an existing research stream.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
FRAME=ROOT/'exp/two_backbone_native_cl_20261017_v2'
sys.path.insert(0,str(FRAME))
import torch
import torch.nn.functional as F
from transformers import AutoProcessor
import onevision_bridge as bridge
from methods import Method
from frozen_vision import install
from runtime_utils import input_batch, answer_loss, evaluate, write
from resource_monitor import ResourceMonitor
from run import load_protocol
import onevision_state_cache_core as core

REPORT=ROOT/'summary/onevision_diagnostics_20261016/formal_v2_state_cache_20261018/v1'
FAMILY=ROOT/'exp/onevision_formal_state_cache_20261018/v1'
STATES=FAMILY/'source_states'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def plan():
    policy,manifest,digest,data=load_protocol('pct01')
    export=read(STATES/'COMPLETE.json')
    assert export['complete'] and export['stage_count']==19
    conf=export['original_training_config']
    assert digest==conf['manifest_sha256'] and conf['model_dtype']=='torch.float16'
    assert all(sha(FRAME/n)==v for n,v in conf['execution_source_sha256'].items())
    parent=read(ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/CPU_PLAN_CHECK.json')['spec']
    train={r['id']:(t,r) for (t,split),rs in data.items() if split=='train' for r in rs}
    assert len(train)==1313 and all(train[i][0]==t for i,t in parent['TRAIN_memory_arrivals'].items())
    assert all(set(parent['prototype_TRAIN_ids'][str(t)])<=set(r['id'] for r in data[t,'train']) for t in range(1,20))
    spec=dict(formal_V2=True, condition='pct01',manifest_sha256=digest,
        seed=42,main_epochs=1,main_TRAIN_QA=1313,main_updates_charged=93,
        main_training='exact completed formal SeqFT19; module choices do not change main updates',
        model_dtype='torch.float16',image_area_cap=401408,visual_cache_limit_bytes=134217728,
        generation_settings='unchanged formal task instruction and max_new_tokens; greedy',
        shared_source_sha256=conf['execution_source_sha256'],
        formal_reference155_sha256=sha(FAMILY/'SOURCE_REFERENCE155.json'),
        source_export_sha256=sha(STATES/'COMPLETE.json'),export_stage_sha256={str(r['stage']):r['export_sha256'] for r in export['stages']},
        TRAIN_memory_arrivals=parent['TRAIN_memory_arrivals'],prototype_TRAIN_ids=parent['prototype_TRAIN_ids'],
        generation_ids=parent['generation_ids'],Kmax=core.CAP,penalty_nats=core.PENALTY,
        kernel_temperature=core.TEMPERATURE,views=list(core.VIEWS),
        full_candidates=['cache_geometry','cache_quality'],TEST_selection=False,
        stage_source_keys='only already arrived states loaded; discarded states never reconsidered',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'analysis/onevision_state_cache_core.py']})
    return spec,manifest,data,train,export


def load_state(t,spec):
    p=STATES/f'stage_{t:02}.pt'
    assert sha(p)==spec['export_stage_sha256'][str(t)]
    obj=torch.load(p,map_location='cpu',weights_only=True)
    assert obj['stage']==t and obj['manifest_sha256']==spec['manifest_sha256']
    return obj['state']


class CacheMethod(Method):
    def __init__(self,model,states,view):
        super().__init__(model,'SeqLoRA',[])
        self.states,self.view=states,view
        self.active=None;self.last_route={};self.feature_calls=0
    def restore_tensors(self,state):
        own=self.model.state_dict()
        assert len(state)==96 and sum(v.numel() for v in state.values())==540672
        assert all(k in own and v.shape==own[k].shape for k,v in state.items())
        with torch.no_grad():
            for k,v in state.items():own[k].copy_(v)
        assert all(torch.equal(own[k].cpu(),v) for k,v in state.items())
        self.active=None
    def set_instruction(self,task):
        # Formal task instructions are handled by the unchanged common formatter.
        # Task labels never enter our router.
        pass
    def feature(self,row,batch):
        self.feature_calls+=1
        ids=self.processor.tokenizer(row['question'].replace('<image>',''),add_special_tokens=False)['input_ids']
        assert ids and batch['pixel_values'].ndim==2
        with torch.no_grad():
            q=self.model.get_input_embeddings()(torch.tensor([ids],device=self.model.device))[0].float().mean(0).cpu()
            v=batch['pixel_values'].float().mean(0).cpu()
        out=F.normalize(torch.cat([F.normalize(q,dim=0),F.normalize(v,dim=0)]),dim=0)
        assert out.shape==(1792,) and torch.isfinite(out).all()
        return out
    def prepare_row(self,row,batch):
        state=self.states;keys=[state['stage']] if self.view.startswith('seq_') else sorted(state['pool'])
        if len(keys)==1:chosen,details=keys[0],dict(single_state_identity=True)
        else:
            chosen,details=core.route(self.feature(row,batch),keys,torch.stack([state['prototypes'][k] for k in keys]),
                state['memory_features'],torch.tensor([[state['losses'][k][i] for i in state['memory_ids']] for k in keys]),
                self.view.endswith('_quality'))
        if self.active!=chosen:
            self.restore_tensors(state['pool'][chosen]);self.active=chosen
        self.last_route=dict(state=chosen,view=self.view,**details)
        # Every query route is saved separately without modifying common predictions.
        if hasattr(self,'route_stream'):
            self.route_stream.write(json.dumps(dict(id=row['id'],**self.last_route))+'\n');self.route_stream.flush()


def build(run,spec,manifest,data,train,method,monitor):
    states={};prototypes={};losses={};features={};events=[];calls=[]
    arrivals=spec['TRAIN_memory_arrivals']
    method.model.eval()
    for t in range(1,20):
        states[t]=load_state(t,spec)
        method.restore_tensors(states[t])
        monitor.snapshot('cache_before_auxiliary',t,method.model)
        ids=[i for i,a in arrivals.items() if a<=t]
        proto=[]
        for i in spec['prototype_TRAIN_ids'][str(t)]:
            row=train[i][1];batch,_=input_batch(row,manifest['tasks'][t-1],method.processor,method.model.device)
            proto.append(method.feature(row,batch))
        prototypes[t]=F.normalize(torch.stack(proto).mean(0),dim=0)
        for i in ids:
            if i not in features:
                task,row=train[i];batch,_=input_batch(row,manifest['tasks'][task-1],method.processor,method.model.device)
                features[i]=method.feature(row,batch)
        for s in sorted(states):
            method.restore_tensors(states[s]);losses.setdefault(s,{})
            for i in ids:
                if i in losses[s]:continue
                task,row=train[i];assert task<=t and s<=t
                batch,_=input_batch(row,manifest['tasks'][task-1],method.processor,method.model.device)
                with torch.no_grad(),torch.autocast('cuda',dtype=method.model.dtype):
                    loss,_=answer_loss(method.model,batch,row['answer'],method.processor)
                value=float(loss);assert math.isfinite(value)
                losses[s][i]=value;calls.append(dict(stage=t,state=s,TRAIN_id=i,NLL=value))
        selected,objective=core.retain(t,{s:[losses[s][i] for i in ids] for s in states})
        events.append(dict(stage=t,selected_states=selected,candidates=sorted(states),**objective))
        states={s:states[s] for s in selected};prototypes={s:prototypes[s] for s in selected};losses={s:losses[s] for s in selected}
        state=dict(stage=t,pool=states,prototypes=prototypes,losses=losses,memory_ids=ids,
                   memory_features=torch.stack([features[i] for i in ids]))
        p=run/f'pool/stage_{t:02}.pt';p.parent.mkdir(exist_ok=True)
        tmp=p.with_suffix('.pt.partial');torch.save(state,tmp);tmp.replace(p)
        monitor.snapshot('cache_after_auxiliary',t,method.model)
        write(run/'decisions.json',dict(events=events,TRAIN_NLL_calls=calls,feature_calls=method.feature_calls,
            pool_tensor_bytes=sum(v.numel()*v.element_size() for a in states.values() for v in a.values()),
            router_trainable_parameters=0,original_main_training_updates_charged=93))
        print(json.dumps(dict(event='formal_cache_stage',**events[-1])),flush=True)
    return state


def main(check,phase,view):
    spec,manifest,data,train,export=plan()
    REPORT.mkdir(parents=True,exist_ok=True)
    if check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        for t in range(1,20):
            a=load_state(t,spec);assert len(a)==96 and sum(v.numel() for v in a.values())==540672
        write(REPORT/'CPU_INPUT_CHECK.json',dict(passed=True,CPU_only=True,spec=spec,
            original19_states_finite_shape_checked=True,GPU_started=False,performance_evidence=False))
        return
    # No manual overlapping launch: an explicit admission from the sole research
    # dispatcher is required. The active native worker remains untouched.
    admission=read(REPORT/'ADMISSION.json')
    assert admission['spec']==spec and admission['single_authoritative_research_queue']
    assert admission['worker_source_sha256']==sha(Path(__file__))
    assert read(REPORT/'CPU_INPUT_CHECK.json')['spec']==spec
    physical=int(os.environ['CUDA_VISIBLE_DEVICES']);assert physical in [0,1]
    import onevision_research_device_entry as device
    cards=device.inventory();assert device.select(cards,os.getuid())==physical
    lock=(REPORT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert phase=='build' or view in spec['full_candidates']
    name='build' if phase=='build' else 'full_'+view
    assert not (REPORT/(name+'_attempt.json')).exists(), 'No blind retry or partial splice'
    run=FAMILY/name/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
    write(REPORT/(name+'_attempt.json'),dict(spec=spec,run=str(run.relative_to(ROOT)),pid=os.getpid()))
    write(run/'spec.json',spec)
    write(run/'physical_device_execution.json',dict(physical_GPU=physical,admission_inventory=cards,
        allowed_GPUs=[0,1],max_our_GPU_cards=3,actual_CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES']))
    import random
    import numpy as np
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    monitor=ResourceMonitor(run);monitor.snapshot('process_before_model')
    model_path=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=bridge.Factory.from_pretrained(model_path,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda()
    bridge.bind(model)
    method=CacheMethod(model,None,None);method.processor=processor;method.vision_original=install(model)
    model.eval()
    monitor.snapshot('model_loaded',model=model)
    bridge.NATIVE_GATE_MODE=True
    gates=[bridge.forward_gate(data[t,'train'][0],manifest['tasks'][t-1],processor) for t in [1,2,12,16,17]]
    # Positive geometry-failure control is retained rather than filtering it out.
    known=next(r for r in data[16,'test'] if r['id']=='FinMME:5082')
    gates.append(bridge.forward_gate(known,manifest['tasks'][15],processor))
    bridge.NATIVE_GATE_MODE=False
    write(run/'official_forward_gate.json',dict(passed=True,cases=gates,optimizer_updates=0))
    if phase=='build':
        monitor.snapshot('functional_selection_before',model=model)
        state=build(run,spec,manifest,data,train,method,monitor)
        monitor.snapshot('functional_selection_after',model=model)
    else:
        proof=read(REPORT/'build_complete.json');assert proof['spec']==spec and proof['complete']
        parent=ROOT/proof['run']
        p=parent/'pool/stage_19.pt';assert sha(p)==proof['artifact_sha256']['pool/stage_19.pt']
        state=torch.load(p,map_location='cpu',weights_only=True)
    method.states=state
    diag={(t,'test'):[r for r in data[t,'test'] if r['id'] in set(spec['generation_ids'][manifest['tasks'][t-1]['domain']])] for t in range(1,20)}
    assert sum(map(len,diag.values()))==155

    def eval_view(which,label,selected):
        method.view=which;method.active=None
        with (run/(str(label)+'_routes.jsonl')).open('w') as stream:
            method.route_stream=stream
            evaluate(method,processor,manifest['tasks'],selected,run,label)
        del method.route_stream
        result=[]
        from answer_f1 import answer_f1
        for t in range(1,20):
            p=run/f'evaluations/stage_{label:02d}/T{t:02}_test_predictions.jsonl'
            rr=[json.loads(s) for s in p.read_text().splitlines()]
            assert [r['id'] for r in rr]==[r['id'] for r in selected[t,'test']]
            for row,gold in zip(rr,selected[t,'test']):
                assert row['references']==gold['answers']
                assert answer_f1(row['prediction'],row['references'])==row['scores']['token_f1']
                result.append(dict(id=row['id'],prediction=row['prediction'],references=row['references'],scores=row['scores']))
        return result

    # Formal runtime formats stage with :02d. Numbered labels are explicit and
    # bound in the completed proof; they are not CL training-stage measurements.
    if phase=='build':
        outputs={}
        for index,which in enumerate(core.VIEWS,100):outputs[which]=eval_view(which,index,diag)
        reference=read(FAMILY/'SOURCE_REFERENCE155.json')
        assert reference['source_export_proof_sha256']==spec['source_export_sha256']
        expected={r['id']:r for r in reference['rows']}
        assert len(expected)==155
        for row in outputs['seq_geometry']:
            assert row==expected[row['id']], 'Formal SeqFT exact prediction reload differs; retain evidence, no score claim'
        assert outputs['seq_geometry']==outputs['seq_quality']
        write(run/'original_exact_gate.json',dict(passed=True,generated=155,original_formal_SeqFT_exact=True,no_tolerance=True))
        write(run/'view_labels.json',dict(zip(core.VIEWS,range(100,104))))
    else:
        current=eval_view(view,200,diag)
        mapping=read(parent/'view_labels.json');label=mapping[view]
        expected=[]
        for t in range(1,20):
            for row in [json.loads(s) for s in (parent/f'evaluations/stage_{label:02d}/T{t:02}_test_predictions.jsonl').read_text().splitlines()]:
                expected.append({k:row[k] for k in ['id','prediction','references','scores']})
        assert current==expected, 'Fresh-process exact reload failed'
        write(run/'candidate_reload_exact.json',dict(passed=True,generated=155,no_tolerance=True))
        full={(t,'test'):data[t,'test'] for t in range(1,20)}
        final=eval_view(view,201,full);assert len(final)==21454
    monitor.snapshot('method_after',model=model);monitor.finish(model)
    files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.pt']}
    write(REPORT/(name+'_complete.json'),dict(complete=True,spec=spec,run=str(run.relative_to(ROOT)),
        phase=phase,view=view,artifact_sha256=files,main_training_updates_charged=93,new_optimizer_updates=0,
        full_QA=21454 if phase=='full' else None,FWT=None,BWT=None,
        sparse_transition_cells_still_required_for_complete_FWT_BWT=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true')
    p.add_argument('--phase',choices=['build','full']);p.add_argument('--view',choices=core.VIEWS)
    a=p.parse_args();main(a.check,a.phase,a.view)
