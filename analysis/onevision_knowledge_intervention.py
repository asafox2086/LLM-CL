"""Controlled prompt / author-KG / tiny residual-LoRA intervention on GPU1.

Exploratory fixed diagnostic subsets plus the complete native KG test267;
never advertised as the full five-domain21454 test or a published CL method.
"""
import argparse,copy,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'exp/onevision_diagnostics_20261016'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'analysis'))
from run import Experiment,freeze,write
from onevision_research_worker import atomic_torch,digest

PROMPTS={'original':'','answer_only':'\nGive only the answer, without an introductory phrase or explanation.',
         'question_language':'\nAnswer in the same language as the question.',
         'both':'\nGive only the answer, without an introductory phrase or explanation. Answer in the same language as the question.'}

def render_facts(facts):
    return '\n'.join(f"{f['subject']} | {f['relation']} | {f['value']}" for f in facts)

def case_complete(run,case,tests):
    folder=run/'evaluations'/case;proof=folder/'durable_complete.json'
    if not proof.exists():return False
    check=json.loads(proof.read_text())
    for domain,rows in tests.items():
        p=folder/f'{domain}_image.jsonl'
        if not p.exists() or digest(p)!=check['files'].get(p.name):return False
        actual=[json.loads(line) for line in p.read_text().splitlines()]
        if [x['id'] for x in actual]!=[x['id'] for x in rows]:return False
    return True

def valid_checkpoint(path):
    if not path.exists():return False
    meta=path.with_suffix('.json')
    if not meta.exists():
        destination=path.parent/'interrupted';destination.mkdir(exist_ok=True);path.rename(destination/(str(time.time_ns())+'_'+path.name));return False
    assert digest(path)==json.loads(meta.read_text())['sha256'],'Saved adapter checksum mismatch'
    return True

class ResidualLoRA(nn.Module):
    def __init__(self,frozen,rank):
        super().__init__();self.frozen=frozen;self.weight,self.bias=frozen.weight,frozen.bias
        self.A=nn.Linear(frozen.A.in_features,rank,bias=False,device=self.weight.device,dtype=torch.float32)
        self.B=nn.Linear(rank,frozen.B.out_features,bias=False,device=self.weight.device,dtype=torch.float32)
        nn.init.kaiming_uniform_(self.A.weight,a=5**.5);nn.init.zeros_(self.B.weight);self.drop=nn.Dropout(.1)
    def forward(self,x):
        # The reused adapter is fixed in both weights and dropout behavior.
        self.frozen.eval();y=self.frozen(x)
        with torch.autocast('cuda',enabled=False):delta=4*self.B(self.A(self.drop(x.float())))
        return y+delta.to(y.dtype)

class Intervention(Experiment):
    def __init__(self,manifest,run):
        super().__init__(manifest,run);self.pristine=dict(self.layers);self.original_initial=self.initial
        self.original_manifest=copy.deepcopy(manifest);self.knowledge=None

    def prompt(self,variant):
        self.manifest=copy.deepcopy(self.original_manifest)
        for task in self.manifest['tasks']:task['instruction']+=PROMPTS[variant]

    def inputs(self,row,view='image',wrong=None,retain_pixels=False):
        if self.knowledge:
            row=dict(row);row['context']=row.get('context','')+'\nReference knowledge (use only if relevant):\n'+self.knowledge
        return super().inputs(row,view,wrong,retain_pixels)

    def evaluate(self,case,tests,generation,views=('image',)):
        folder=self.run/'evaluations'/case
        for domain in tests:
            path=folder/f'{domain}_image.jsonl'
            if not path.exists():continue
            lines=path.read_text().splitlines();valid=[]
            for index,line in enumerate(lines):
                try:json.loads(line);valid.append(line)
                except json.JSONDecodeError:
                    if index!=len(lines)-1:raise
                    archive=folder/'interrupted';archive.mkdir(exist_ok=True)
                    (archive/(str(time.time_ns())+'_'+path.name)).write_bytes(path.read_bytes())
                    path.write_text('\n'.join(valid)+('\n' if valid else ''))
        super().evaluate(case,tests,generation,views)
        files={}
        for domain in tests:
            path=folder/f'{domain}_image.jsonl'
            with path.open('rb') as stream:os.fsync(stream.fileno())
            files[path.name]=digest(path)
        write(folder/'durable_complete.json',{'complete':True,'files':files})

    def configure(self,shared,rank=None,finetune_shared=False):
        for p in self.model.parameters():p.requires_grad=False
        self.layers={}
        for key,old in self.pristine.items():
            with torch.no_grad():old.A.weight.copy_(shared[key]['A']);old.B.weight.copy_(shared[key]['B'])
            if rank is None:
                module=old
                if finetune_shared:
                    for p in (old.A.weight,old.B.weight):p.requires_grad=True
            else:module=ResidualLoRA(old,rank)
            parent,name=key.rsplit('.',1);setattr(self.model.get_submodule(parent),name,module);self.layers[key]=module
        self.initial=self.snapshot()

    def train_facts(self,case,facts,epochs=4):
        assert len(facts)==16
        rows=[]
        for f in facts:
            question=(f"What is the {f['relation']} of {f['subject']}?" if f['language']=='en' else f"{f['subject']}的{f['relation']}是什么？")
            rows.append({'id':'external-KG:'+f['id'],'dataset':'SLAKE-author-KG-auxiliary','images':[],
                'question':question,'answer':f['value'],'answers':[f['value']],'_task':11})
        parameters=[p for p in self.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
        self.model.train();self.monitor.snapshot(case+'_training_before',model=self.model)
        records=[]
        for epoch in range(epochs):
            order=sorted(rows,key=lambda x:hashlib.sha256(f"42:{epoch}:{x['id']}".encode()).digest())
            for offset in [0,8]:
                optimizer.zero_grad(set_to_none=True);losses=[];tokens=0
                for row in order[offset:offset+8]:
                    loss,details=self.loss(row);assert torch.isfinite(loss);(loss/8).backward();losses.append(float(loss.detach()));tokens+=details['target_tokens']
                grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad);optimizer.step()
                record={'case':case,'epoch':epoch,'qa_seen':epoch*16+offset+8,'external_fact_pool':16,'loss':sum(losses)/8,
                        'gradient_l2_preclip':float(grad),'lr_used':1e-4,'target_tokens':tokens,'trainable_parameters':sum(p.numel() for p in parameters)}
                records.append(record);print(json.dumps({'event':'knowledge_aux_train',**record}),flush=True)
        optimizer.zero_grad(set_to_none=True);del optimizer
        self.monitor.snapshot(case+'_training_after',model=self.model)
        write(self.run/'auxiliary_training'/f'{case}.json',{'synthetic_QA_rendering_of_author_graph':True,'original_target_train_qa_added':0,
            'test_gold_used':False,'fixed_epochs':epochs,'optimizer_updates':len(records),'additional_external_QA_exposures':epochs*16,'events':records})

def reset_seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['format','knowledge'],required=True);args=ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    torch.set_num_threads(6);reset_seed()
    manifest,data,domains,tests,generation,probes,protocol=freeze()
    report=ROOT/'summary/onevision_mechanisms_20261016';report.mkdir(parents=True,exist_ok=True)
    source={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),HERE/'run.py',ROOT/'analysis/onevision_research_worker.py']}
    pool_path=ROOT/'data/OneVisionMechanisms/20261016/knowledge16.json';pool=json.loads(pool_path.read_text())
    assert pool['control_version']==2 and all(a['value']!=b['value'] for a,b in zip(pool['correct'],pool['permuted']))
    source['knowledge16.json']=digest(pool_path)
    pointer=report/(args.mode+'_attempt.json')
    if pointer.exists():
        info=json.loads(pointer.read_text());assert info['source_sha256']==source
        run=ROOT/info['run']
    else:
        run=ROOT/'exp/onevision_mechanisms_20261016'/args.mode/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
        info={'run':str(run.relative_to(ROOT)),'source_sha256':source,'mode':args.mode,'seed':42,'target_train_ids_unchanged':True,
            'test_ids_source':'original frozen diagnostic500; knowledge also complete native KG267; groups overlap, never summed',
            'shared_adapter_selection':'first domain by original task order, not best test adapter',
            'source_knowledge_status':'author-provided benchmark knowledge graph, not independent clinical corpus',
            'knowledge_fact_count':16,'rank_grid':[1,2],'scale_alpha_over_rank':4,'auxiliary_epochs':4,
            'full_five_domain_test_complete':False,'adaptive_capacity_method':False}
        write(pointer,info);write(run/'protocol.json',info)
    x=Intervention(manifest,run)
    x.configure(x.original_initial)
    if args.mode=='format':
        # Medical generation now covers all100 fixed probes; other domains retain the pre-fixed10 each.
        generation=dict(generation);generation['medical']=tests['medical']
        for variant in PROMPTS:
            case='format_'+variant
            if case_complete(run,case,tests):continue
            x.prompt(variant);x.knowledge=None;x.model.eval()
            write(report/'status.json',{'phase':'format_control','case':case,'run':str(run.relative_to(ROOT)),'pid':os.getpid()})
            x.evaluate(case,tests,generation)
    else:
        # Test gold enters evaluation only. Fact selection was frozen beforehand from TRAIN questions.
        tests={'medical':tests['medical'],'medical_KG':data[11,'test']};generation=tests
        shared_path=HERE/'runs/20261016_164638/adapters/isolated_natural.pt'
        shared_obj=torch.load(shared_path,map_location='cpu',weights_only=True);shared=shared_obj['adapter']
        write(run/'shared_adapter_reference.json',{'path':str(shared_path.relative_to(ROOT)),'sha256':digest(shared_path),
            'selection':'first original domain, predeclared; not test-score selection'})
        x.prompt('both')
        gate=run/'tiny_adapter_gpu_gate.json'
        if not gate.exists():
            probe=probes['medical'][0];cases=[]
            x.configure(shared);x.model.eval()
            with torch.no_grad():reference,_=x.loss(probe)
            for rank in [1,2]:
                reset_seed();x.configure(shared,rank);x.model.eval()
                parameters=[p for p in x.model.parameters() if p.requires_grad]
                with torch.no_grad():zero_loss,_=x.loss(probe)
                assert abs(float(zero_loss)-float(reference))<.005
                optimizer=torch.optim.AdamW(parameters,lr=1e-4);optimizer.zero_grad(set_to_none=True)
                loss,_=x.loss(probe);loss.backward();grad=torch.nn.utils.clip_grad_norm_(parameters,1.)
                assert torch.isfinite(loss) and torch.isfinite(grad);optimizer.step()
                assert sum(p.numel() for p in parameters)==67584*rank
                for key,old in x.pristine.items():
                    assert torch.equal(old.A.weight.detach().cpu(),shared[key]['A']) and torch.equal(old.B.weight.detach().cpu(),shared[key]['B'])
                cases.append({'rank':rank,'zero_residual_loss_error':abs(float(zero_loss)-float(reference)),
                    'finite_backward':True,'finite_optimizer_update':True,'frozen_shared_unchanged':True,'trainable_parameters':67584*rank})
                optimizer.zero_grad(set_to_none=True);del optimizer,loss
            write(gate,{'passed':True,'train_probe_id':probe['id'],'diagnostic_updates_discarded':True,'cases':cases})
        # Context-only comparison on both raw base and the fixed first-domain shared adapter.
        for name,base in [('base',x.original_initial),('shared',shared)]:
            for kind in ['none','correct','unrelated','permuted']:
                case=f'context_{name}_{kind}'
                if case_complete(run,case,tests):continue
                x.configure(base);x.knowledge=None if kind=='none' else render_facts(pool[kind]);x.model.eval()
                write(report/'status.json',{'phase':'knowledge_context','case':case,'run':str(run.relative_to(ROOT)),'pid':os.getpid()})
                write(run/'contexts'/f'{case}.json',{'knowledge':x.knowledge,'utf8_bytes':len((x.knowledge or '').encode()),
                    'token_count':len(x.processor.tokenizer(x.knowledge or '',add_special_tokens=False)['input_ids']),
                    'fact_ids':[] if kind=='none' else [f['id'] for f in pool[kind]],'synthetic_mismatch':kind=='permuted'})
                x.evaluate(case,tests,generation)
        # Reuse fixed shared LoRA, add a tiny residual; compare correct vs unrelated vs mismatched facts.
        for rank in [1,2]:
            for kind in ['correct','unrelated','permuted']:
                case=f'residual_rank{rank}_{kind}'
                if case_complete(run,case,tests):continue
                reset_seed();x.configure(shared,rank);x.knowledge=None;x.prompt('both')
                path=run/'adapters'/f'{case}.pt'
                write(report/'status.json',{'phase':'knowledge_residual','case':case,'run':str(run.relative_to(ROOT)),'pid':os.getpid()})
                if valid_checkpoint(path):
                    meta=json.loads(path.with_suffix('.json').read_text());assert digest(path)==meta['sha256'];obj=torch.load(path,map_location='cpu',weights_only=True);x.restore(obj['adapter'])
                else:
                    x.train_facts(case,pool[kind])
                    atomic_torch(path,{'adapter':x.snapshot(),'rank':rank,'alpha':4*rank,'shared_adapter_sha256':digest(shared_path),
                        'knowledge_sha256':source['knowledge16.json'],'source_sha256':source,'training_kind':kind})
                x.evaluate(case,tests,generation)
        # Same external knowledge budget, no module growth: continue the original rank8 shared adapter.
        for kind in ['correct','unrelated','permuted']:
            case=f'continue_rank8_{kind}'
            if case_complete(run,case,tests):continue
            reset_seed();x.configure(shared,finetune_shared=True);x.knowledge=None
            path=run/'adapters'/f'{case}.pt'
            if valid_checkpoint(path):
                meta=json.loads(path.with_suffix('.json').read_text());assert digest(path)==meta['sha256'];x.restore(torch.load(path,map_location='cpu',weights_only=True)['adapter'])
            else:
                x.train_facts(case,pool[kind]);atomic_torch(path,{'adapter':x.snapshot(),'rank':8,'alpha':32,'training_kind':kind,'source_sha256':source})
            x.evaluate(case,tests,generation)
    x.monitor.snapshot('training_after',model=x.model);x.monitor.finish(x.model)
    write(report/(args.mode+'_complete.json'),{'complete':True,'run':str(run.relative_to(ROOT)),'source_sha256':source,'formal_full_test':False})

if __name__=='__main__':main()
