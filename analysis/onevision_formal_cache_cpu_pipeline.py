"""Isolated synthetic CPU integration of the new formal cache driver.

Exercises the real build/retention/routing/common evaluator/proof paths. Model,
images, weights, loss, and generation are synthetic. Never performance evidence.
"""
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
import torch
torch.set_num_threads(1)
import onevision_formal_state_cache as target
import runtime_utils
import onevision_research_device_entry as device


class CPU:
    cuda=SimpleNamespace(manual_seed_all=lambda _:None)
    def __getattr__(self,name):return getattr(torch,name)
    def autocast(self,*a,**k):return contextlib.nullcontext()
    def set_num_threads(self,*a):torch.set_num_threads(1)


class Model:
    device=torch.device('cpu');dtype=torch.float16
    active_stage=0
    def cuda(self):return self
    def eval(self):return self
    def generate(self,**batch):return torch.tensor([[0,0,7]])


class Monitor:
    def __init__(self,run):self.run=run;self.boundaries=[]
    def snapshot(self,label,*a,**k):self.boundaries.append(label)
    def finish(self,*a):target.write(self.run/'toy_resource_calls.json',dict(boundaries=self.boundaries,synthetic=True))


def main():
    real_report=target.REPORT;source=target.sha(Path(target.__file__))
    # Exercise the real feature function before replacing model components.
    # Changing answers/domain labels must not change its vector; changing the
    # question must change it (positive control against a constant stub).
    probe=object.__new__(target.CacheMethod);probe.feature_calls=0
    embedding=torch.nn.Embedding(128,896)
    probe.model=SimpleNamespace(device=torch.device('cpu'),get_input_embeddings=lambda:embedding)
    probe.processor=SimpleNamespace(tokenizer=lambda text,**k:dict(input_ids=[ord(c)%128 for c in text]))
    row=dict(question='ask',images=['toy'],answer='gold',domain='medical')
    batch=dict(pixel_values=torch.ones((3,896)))
    a=probe.feature(row,batch)
    b=probe.feature(dict(row,answer='different',answers=['different'],domain='finance',native_task=99),batch)
    c=probe.feature(dict(row,question='why'),batch)
    assert torch.equal(a,b) and not torch.equal(a,c) and a.shape==(1792,)
    with tempfile.TemporaryDirectory(prefix='llmcl_formal_cpu_') as folder:
        root=Path(folder);assert root.is_relative_to(Path('/tmp'))
        target.ROOT=root;target.REPORT=root/'report';target.FAMILY=root/'family'
        target.FAMILY.mkdir();target.REPORT.mkdir()
        target.torch=CPU();runtime_utils.torch=CPU()
        device.inventory=lambda:{i:{'owners':[],'memory_MiB':0,'utilization':0} for i in range(4)}
        device.select=lambda *a:1
        target.ResourceMonitor=Monitor
        target.bridge.Factory=SimpleNamespace(from_pretrained=lambda *a,**k:Model())
        target.bridge.bind=lambda m:None
        target.bridge.forward_gate=lambda *a,**k:dict(CPU_synthetic_only=True)
        target.install=lambda m:None
        tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=1,decode=lambda *a,**k:'synthetic')
        processor=SimpleNamespace(tokenizer=tokenizer)
        target.AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**k:processor)

        def input_batch(row,task,processor,device):
            return dict(input_ids=torch.zeros((1,2),dtype=torch.long),_toy_task=row['_toy_task']),{}
        target.input_batch=input_batch;runtime_utils.input_batch=input_batch
        target.answer_loss=lambda model,batch,*a:(torch.tensor((model.active_stage-batch['_toy_task'])**2/100+.1),1)
        target.load_state=lambda t,spec:dict(_stage=torch.tensor([t]))
        class Method(target.CacheMethod):
            def __init__(self,model,state,view):
                self.model,self.states,self.view=model,state,view
                self.active=None;self.feature_calls=0
            def restore_tensors(self,state):self.model.active_stage=int(state['_stage'][0]);self.active=None
            def feature(self,row,batch):
                self.feature_calls+=1
                v=torch.tensor([1.,float(row['_toy_task'])/19]);return torch.nn.functional.normalize(v,dim=0)
        target.CacheMethod=Method
        domains=['natural']+['medical']*10+['geography']*4+['finance']+['traffic']*3
        fullcounts=[5000]+[209]*9+[213]+[2501]*4+[2233,707,708,708]
        gen=[10]+[7]*9+[10]+[8]*4+[10,10,10,10]
        assert sum(fullcounts)==21454 and sum(gen)==155
        data={};train={};tasks=[];generation={d:[] for d in set(domains)};protos={};arrivals={}
        for t in range(1,20):
            def row(index,split):
                return dict(id=f'toy:{split}:{t}:{index}',question='synthetic',images=[],answer='synthetic',
                            answers=['synthetic'],_toy_task=t)
            rs=[row(i,'test') for i in range(fullcounts[t-1])]
            if t==16:rs[-1]['id']='FinMME:5082'
            data[t,'test']=rs
            n=1295 if t==1 else 1
            data[t,'train']=[row(i,'train') for i in range(n)]
            train.update({r['id']:(t,r) for r in data[t,'train']})
            protos[str(t)]=[data[t,'train'][0]['id']]
            arrivals[data[t,'train'][0]['id']]=t
            generation[domains[t-1]] += [r['id'] for r in rs[:gen[t-1]]]
            tasks.append(dict(task=t,domain=domains[t-1],instruction='synthetic',max_new_tokens=4))
        arrivals[data[1,'train'][1]['id']]=1
        spec=dict(TRAIN_memory_arrivals=arrivals,prototype_TRAIN_ids=protos,generation_ids=generation,
                  full_candidates=['cache_geometry','cache_quality'],source_export_sha256='toy')
        manifest=dict(tasks=tasks)
        target.plan=lambda:(copy.deepcopy(spec),manifest,data,train,{})
        target.write(target.REPORT/'ADMISSION.json',dict(spec=spec,single_authoritative_research_queue=True,worker_source_sha256=source))
        target.write(target.REPORT/'CPU_INPUT_CHECK.json',dict(spec=spec))
        reference=[]
        for t in range(1,20):
            for r in data[t,'test'][:gen[t-1]]:
                reference.append(dict(id=r['id'],prediction='synthetic',references=r['answers'],scores=runtime_utils.score('synthetic',r['answers'])))
        target.write(target.FAMILY/'SOURCE_REFERENCE155.json',dict(source_export_proof_sha256='toy',rows=reference))
        os.environ['CUDA_VISIBLE_DEVICES']='1' # fake namespace only; no CUDA call permitted
        assert not torch.cuda.is_initialized()
        target.main(False,'build',None)
        proof=json.loads((target.REPORT/'build_complete.json').read_text());run=root/proof['run']
        events=json.loads((run/'decisions.json').read_text())['events']
        assert len(events)==19 and all(e['stage'] in e['selected_states'] and len(e['selected_states'])<=3 for e in events)
        for view in spec['full_candidates']:
            target.main(False,'full',view)
            full=json.loads((target.REPORT/('full_'+view+'_complete.json')).read_text())
            assert full['complete'] and full['full_QA']==21454
        assert not torch.cuda.is_initialized()
        evidence=dict(CPU_only=True,real_model_or_QA_used=False,GPU_operations=0,
            actual_feature_function_synthetic_answer_domain_invariance_and_question_positive_control=True,
            actual_formal_build19_retention_and_all4_common_evaluator_views_exercised=True,
            both_full21454_branches_and_reload_gates_exercised=True,
            no_toy_predictions_or_scores_published=True,
            worker_source_sha256=source,harness_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            not_model_equivalence_or_performance_or_resource_evidence=True)
    real_report.mkdir(parents=True,exist_ok=True);target.write(real_report/'CPU_PIPELINE_CHECK.json',evidence)
    print(json.dumps(evidence))


if __name__=='__main__':main()
