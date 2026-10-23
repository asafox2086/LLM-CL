"""Physically compact FFNs with bounded temporary directions; no dense residual."""
from pathlib import Path
import hashlib, importlib.util, json, math, os
ROOT=Path(__file__).resolve().parents[3]
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/final_weight_reduction_20261023/answer_guided_pilot_v1'
WORK=Path(__file__).parent/'private'
RANK=200; EXTRA=16; BASE=893675552; FINAL=662833184; TRAINABLE=6635520
CURRENT_LIMIT=16*1024**2; BANK_LIMIT=32*1024**2; BANK_GROUPS=8
CASES=('Spectral','Answer')
spec=importlib.util.spec_from_file_location('answer_renewal_frozen_study',ROOT/'exp/osft_increment_screen_20261019/v1/study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sha=study.sha

def write(path,value):
    study.write(path,value)
    fd=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def digest(params):
    from training_validity import parameter_digest
    return parameter_digest(list(params))

def factor_class():
    import torch
    from torch import nn
    class Factors(nn.Module):
        def __init__(self,left,right,bias):
            super().__init__()
            self.left=nn.Parameter(left,requires_grad=False)
            self.right=nn.Parameter(right,requires_grad=False)
            self.bias=None if bias is None else nn.Parameter(bias.detach().clone(),requires_grad=False)
            self.register_parameter('A',None);self.register_parameter('B',None);self.register_parameter('gate',None)
            self.dropout=nn.Dropout(.1)
            self.in_features=right.shape[1];self.out_features=left.shape[0]
        def forward(self,x):
            h=torch.nn.functional.linear(x,self.right)
            if self.gate is not None:h=(h.float()*self.gate).to(h.dtype)
            y=torch.nn.functional.linear(h,self.left,self.bias)
            if self.A is not None:
                z=torch.nn.functional.linear(torch.nn.functional.linear(self.dropout(x.float()),self.B),self.A)
                y=y+(4*z).to(y.dtype)
            return y
    return Factors

def compress(model,rank=RANK):
    import time,torch
    from torch import nn
    cls=factor_class();records=[]
    for p in model.parameters():p.requires_grad=False
    rng=torch.get_rng_state().clone()
    for i,block in enumerate(model.model.layers):
        for name in ('gate_proj','up_proj','down_proj'):
            old=getattr(block.mlp,name);assert type(old) is nn.Linear
            m,n=old.weight.shape;assert rank<min(m,n) and rank*(m+n)<m*n
            tick=time.monotonic()
            with torch.no_grad():
                w=old.weight.detach().float()
                u,s,vh=torch.linalg.svd(w,full_matrices=False,**({'driver':'gesvd'} if w.is_cuda else {}))
                root=s[:rank].sqrt()
                layer=cls((u[:,:rank]*root).to(old.weight.dtype).contiguous(),
                          (root[:,None]*vh[:rank]).to(old.weight.dtype).contiguous(),old.bias)
            setattr(block.mlp,name,layer)
            records.append(dict(name=f'model.layers.{i}.mlp.{name}',shape=[m,n],rank=rank,
                removed_coefficients=m*n,retained_coefficients=rank*(m+n),seconds=time.monotonic()-tick))
            del old,w,u,s,vh,root,layer
    assert torch.equal(rng,torch.get_rng_state())
    return records

def layout(model):
    ps=dict(model.named_parameters());bs=dict(model.named_buffers())
    aliases=[(str(p.device),p.untyped_storage().data_ptr()) for p in ps.values()]
    assert len(aliases)==len(set(aliases)),'Distinct parameter views need explicit storage review'
    return dict(coefficients=sum(p.numel() for p in ps.values()),trainable=sum(p.numel() for p in ps.values() if p.requires_grad),
        parameter_bytes=sum(p.numel()*p.element_size() for p in ps.values()),
        parameters={n:dict(shape=list(p.shape),dtype=str(p.dtype),trainable=p.requires_grad) for n,p in ps.items()},
        buffers={n:dict(shape=list(p.shape),dtype=str(p.dtype),bytes=p.numel()*p.element_size()) for n,p in bs.items()},
        tied_parameters_deduplicated=True,distinct_parameter_storage_aliases=0)

def priority(prefix,value):return hashlib.sha256((prefix+str(value)).encode()).hexdigest()

def entry_bytes(entry):
    tensor=sum(x.numel()*x.element_size() for x in entry['batch'].values())
    info={k:v for k,v in entry.items() if k!='batch'}
    info['batch']={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in entry['batch'].items()}
    metadata=len(json.dumps(info,ensure_ascii=False,sort_keys=True).encode())
    return tensor+metadata

def memory_ledger(method):
    tensors=[v for e in list(method.bank.values())+list(method.current.values()) for v in e['batch'].values()]
    tensors+=list(getattr(method.model,'_llmcl_visual_cache',{}).values())
    storages={}
    for t in tensors:
        assert str(t.device)=='cpu'
        storages[t.untyped_storage().data_ptr()]=t.untyped_storage().nbytes()
    union=sum(storages.values())
    assert union<=128*1024**2+CURRENT_LIMIT+BANK_LIMIT
    return dict(actual_CPU_tensor_storage_union_bytes=union,CPU_tensor_storage_count=len(storages),
        logical_current_payload_bytes=sum(entry_bytes(e) for e in method.current.values()),
        logical_bank_payload_bytes=sum(entry_bytes(e) for e in method.bank.values()),
        excludes_Python_allocator_overhead_and_other_process_memory=True)

class Renewal:
    name='CompactRenewal';learning_rate=1e-4;host_optimizer=True;offload_activations=True
    def __init__(self,model,rank=RANK,extra=EXTRA):
        self.model=model;self.rank=rank;self.extra=extra;self.current={};self.bank={};self.selected={}
        self.layers=[(n,l) for n,l in model.named_modules() if n.startswith('model.layers.') and n.endswith(('.gate_proj','.up_proj','.down_proj'))]
        assert self.layers and all(hasattr(l,'left') for _,l in self.layers)
    def set_instruction(self,stage):pass
    def begin_stage(self,stage):
        import torch
        self.stage=stage
        for name,l in self.layers:
            m,n=l.out_features,l.in_features;assert self.rank+self.extra<=min(m,n)
            gen=torch.Generator(device='cpu');gen.manual_seed(int(priority('42:answer-renewal:',str(stage)+':'+name)[:16],16)%(2**63))
            b=torch.empty(self.extra,n,dtype=torch.float32).uniform_(-1/math.sqrt(n),1/math.sqrt(n),generator=gen)
            l.A=torch.nn.Parameter(torch.zeros(m,self.extra,device=l.left.device,dtype=torch.float32))
            l.B=torch.nn.Parameter(b.to(l.left.device));l.gate=None
        self.stage_frozen_digest=digest([p for _,l in self.layers for p in (l.left,l.right)]+[l.bias for _,l in self.layers if l.bias is not None])
    def choose_current(self,rows,groups,task):
        self.current={};self.selected={};self.current_task=dict(task);self.current_rejected=[]
        used=set()
        for row in sorted(rows,key=lambda r:priority('20261023:calibration:',r['id'])):
            group=groups[row['id']]
            if group in used:continue
            self.selected[row['id']]=group;used.add(group)
            if len(self.selected)==2:break
    def prepare_row(self,row,batch):
        if row['id'] not in self.selected or row['id'] in self.current:return
        entry=dict(id=row['id'],group=self.selected[row['id']],stage=self.stage,domain=self.current_task['domain'],
                   answer=row['answer'],batch=batch)
        if sum(entry_bytes(e) for e in self.current.values())+entry_bytes(entry)>CURRENT_LIMIT:
            self.current_rejected.append(priority('',row['id']));del self.selected[row['id']];return
        entry['batch']={k:v.detach().cpu().clone() for k,v in batch.items()}
        self.current[row['id']]=entry
    def merge_bank(self):
        for e in self.current.values():
            gid=e['group'];old=self.bank.get(gid)
            if old is not None and priority('20261023:anchor:',old['id'])<=priority('20261023:anchor:',e['id']):continue
            self.bank[gid]=e
            while len(self.bank)>BANK_GROUPS or sum(entry_bytes(v) for v in self.bank.values())>BANK_LIMIT:
                worst=max(self.bank,key=lambda g:(priority('20261023:anchor:',g),g))
                del self.bank[worst]
        self.current={};self.selected={};self.current_task={}
    def remove_training_memory(self):
        self.bank={};self.current={};self.selected={};self.current_task={}

def parent_factors(l):
    """Thin QR/core SVD of rank <=r+b; never materialize a dense original W."""
    import torch
    with torch.no_grad():
        left=torch.cat([l.left.float(),4*l.A.detach()],dim=1)
        right=torch.cat([l.right.float(),l.B.detach()],dim=0)
        ql,rl=torch.linalg.qr(left,mode='reduced');qr,rr=torch.linalg.qr(right.T,mode='reduced')
        core=rl@rr.T
        u,s,vh=torch.linalg.svd(core,full_matrices=False,**({'driver':'gesvd'} if core.is_cuda else {}))
        root=s.clamp_min(0).sqrt()
        lf=((ql@u)*root).to(l.left.dtype).contiguous()
        rf=(root[:,None]*(vh@qr.T)).to(l.right.dtype).contiguous()
    return lf,rf,s.detach().cpu().tolist()

def install_pool(method,pool,masks=None,gated=False):
    import torch
    for name,l in method.layers:
        left,right,_=pool[name]
        if masks is not None:
            ix=torch.tensor(masks[name],device=left.device,dtype=torch.long)
            left=left.index_select(1,ix).contiguous();right=right.index_select(0,ix).contiguous()
        l.A=None;l.B=None
        l.left=torch.nn.Parameter(left,requires_grad=False);l.right=torch.nn.Parameter(right,requires_grad=False)
        l.gate=torch.nn.Parameter(torch.ones(right.shape[0],device=left.device,dtype=torch.float32)) if gated else None

def weighted_entries(method):
    current=list(method.current.values());old=list(method.bank.values());result=[]
    if current:result.extend((e,(.5 if old else 1.)/len(current)) for e in current)
    if old:result.extend((e,(.5 if current else 1.)/len(old)) for e in old)
    return result

def scalar_objective(method,entries,loss_fn,backward):
    import torch
    from contextlib import nullcontext
    value=0.;seen=0
    for entry,weight in entries:
        with (torch.enable_grad() if backward else torch.no_grad()):
            loss=loss_fn(entry)
            assert loss.ndim==0 and torch.isfinite(loss)
            value+=float(loss.detach())*weight;seen+=1
            if backward:(loss*weight).backward()
    assert math.isfinite(value)
    return value,seen

def consolidate(method,case,loss_fn):
    import torch,time
    assert case in CASES
    tick=time.monotonic();method.model.eval()
    record=dict(stage=method.stage,case=case,current_QA=len(method.current),old_QA=len(method.bank),
        current_rejected_id_sha256=getattr(method,'current_rejected',[]),memory_before=memory_ledger(method),
        current_payload_bytes=sum(entry_bytes(e) for e in method.current.values()),
        old_payload_bytes=sum(entry_bytes(e) for e in method.bank.values()),
        current_coverage=[dict(id_sha256=priority('',e['id']),group=e['group'],domain=e['domain']) for e in method.current.values()],
        old_coverage=[dict(id_sha256=priority('',e['id']),group=e['group'],domain=e['domain']) for e in method.bank.values()],
        gradient_QA=0,candidate_forward_QA=0,mask_different=False,selected='spectral',fallback=None,
        count_before=layout(method.model)['coefficients'])
    pool={name:parent_factors(l) for name,l in method.layers}
    spectral={name:list(range(method.rank)) for name,_ in method.layers}
    entries=weighted_entries(method)
    answer=None
    if case=='Answer' and entries:
        install_pool(method,pool,gated=True)
        grad_value,seen=scalar_objective(method,entries,loss_fn,True)
        answer={};gate_stats={}
        for name,l in method.layers:
            assert l.gate.grad is not None and torch.isfinite(l.gate.grad).all()
            deletion=(-l.gate.grad.detach().cpu()).tolist()
            answer[name]=sorted(sorted(range(len(deletion)),key=lambda i:(-deletion[i],i))[:method.rank])
            gate_stats[name]=dict(norm=float(l.gate.grad.detach().float().norm()),finite=True,
                deletion_costs=deletion,signed_score_sha256=priority('',json.dumps(deletion)))
        record.update(gradient_QA=seen,parent_gate_NLL=grad_value,gate_stats=gate_stats,
                      mask_different=any(answer[n]!=spectral[n] for n in spectral))
        if record['mask_different']:
            install_pool(method,pool,spectral)
            spec_value,n1=scalar_objective(method,entries,loss_fn,False)
            install_pool(method,pool,answer)
            ans_value,n2=scalar_objective(method,entries,loss_fn,False)
            record.update(candidate_forward_QA=n1+n2,spectral_NLL=spec_value,answer_NLL=ans_value)
            if ans_value<spec_value:record['selected']='answer'
        else:record['fallback']='identical_masks'
    elif case=='Answer':record['fallback']='empty_calibration_and_history'
    chosen=answer if record['selected']=='answer' else spectral
    install_pool(method,pool,chosen)
    record.update(spectral_masks=spectral,answer_masks=answer,
        count_after=layout(method.model)['coefficients'],final_trainable=layout(method.model)['trainable'])
    assert record['final_trainable']==0
    assert record['gradient_QA']<=10 and record['candidate_forward_QA']<=20
    method.merge_bank()
    record.update(bank_after_QA=len(method.bank),bank_after_bytes=sum(entry_bytes(e) for e in method.bank.values()),
                  memory_after=memory_ledger(method),seconds=time.monotonic()-tick)
    return record

def save_gradient(model,run,stage):
    import torch
    p=run/'gradients'/f'S{stage:02d}_first_raw.pt';p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},p)
    return p

def measured_optimizer(method,factory,record):
    from types import MethodType
    opt=factory(method);expected={id(p) for p in method.model.parameters() if p.requires_grad};original=type(opt).step
    def step(self,*args,**kwargs):
        params=[p for group in self.param_groups for p in group['params']]
        assert {id(p) for p in params}==expected and all(p.grad is not None for p in params)
        grad=sum(p.grad.numel()*p.grad.element_size() for p in params)
        value=original(self,*args,**kwargs)
        moments=[self.state[p][k] for p in params for k in ('exp_avg','exp_avg_sq')]
        record(dict(parameter_IDs_match=True,coordinates=sum(p.numel() for p in params),gradient_bytes=grad,
            two_moment_bytes=sum(t.numel()*t.element_size() for t in moments),moment_devices=sorted({str(t.device) for t in moments})))
        return value
    opt.step=MethodType(step,opt)
    return opt

def cpu_gate():
    import copy,io,pickle,tempfile,torch,cloudpickle
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from types import SimpleNamespace
    import sys
    sys.path.insert(0,str(PORT));from runtime_utils import make_optimizer
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    base=Qwen2ForCausalLM(cfg).half().eval();before=sum(p.numel() for p in base.parameters())
    records=compress(base,8);initial=copy.deepcopy(base)
    assert len(records)==6 and sum(p.numel() for p in base.parameters())==before-6*(32*64-8*96)
    ids=torch.tensor([[2,4,7,9]]);target=torch.tensor([5,6]);reference=initial(ids).logits
    results={}
    for case in CASES:
        m=copy.deepcopy(initial);method=Renewal(m,8,2)
        assert torch.equal(m(ids).logits,reference)
        opt_records=[];consolidations=[]
        for stage in (1,2):
            rng=torch.get_rng_state().clone();method.begin_stage(stage);assert torch.equal(rng,torch.get_rng_state())
            assert all(torch.count_nonzero(l.A)==0 for _,l in method.layers)
            with tempfile.TemporaryDirectory() as td:
                def commit(v):
                    p=Path(td)/'new_optimizer'/f'{len(opt_records)}.json';write(p,v)
                    assert json.loads(p.read_text())==v;opt_records.append(v)
                opt=measured_optimizer(method,make_optimizer,commit)
                frozen=[p for p in m.parameters() if not p.requires_grad];frozen_before=digest(frozen)
                for step in (1,2):
                    m.train();opt.zero_grad(set_to_none=True)
                    with torch.autocast('cpu',dtype=torch.float16):
                        loss=torch.nn.functional.cross_entropy(m(ids).logits[0,-2:].float(),target)
                    loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
                    if step==1:
                        assert all(torch.count_nonzero(l.B.grad)==0 for _,l in method.layers)
                        p=save_gradient(m,Path(td)/'missing_run',stage);saved=torch.load(p,weights_only=True)
                        actual={n:p.grad for n,p in m.named_parameters() if p.requires_grad}
                        assert saved.keys()==actual.keys() and all(torch.equal(saved[n],actual[n]) for n in saved)
                    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.)
                    opt.step()
                assert digest(frozen)==frozen_before
            m.eval()
            method.current={'x':dict(id=f'id{stage}',group=f'g{stage}',stage=stage,domain='tiny',answer='unused',batch={'input_ids':ids})}
            def tiny_loss(entry):
                with torch.autocast('cpu',dtype=torch.float16):return torch.nn.functional.cross_entropy(m(entry['batch']['input_ids']).logits[0,-2:].float(),target)
            consolidations.append(consolidate(method,case,tiny_loss))
            assert all(l.left.shape[1]==8 and l.A is None and l.B is None and l.gate is None for _,l in method.layers)
            assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in initial.parameters())
            assert not any(n.endswith(('.gate_proj.weight','.up_proj.weight','.down_proj.weight')) for n in m.state_dict())
            del opt,frozen
        m.eval();method.remove_training_memory()
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):expected=m(ids).logits
        stream=io.BytesIO();torch.save(method,stream,pickle_module=cloudpickle);stream.seek(0)
        restored=torch.load(stream,pickle_module=pickle,weights_only=False)
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):assert torch.equal(expected,restored.model(ids).logits)
        assert layout(restored.model)==layout(m)
        results[case]=dict(actual_optimizer_updates=len(opt_records),consolidations=consolidations,
            final_coefficients=layout(m)['coefficients'],full_graph_roundtrip_exact=True)
    # Signed deletion ordering, physical masks, and index-free bounded replacement.
    costs=[-2.,4.,1.,-5.];assert sorted(sorted(range(4),key=lambda i:(-costs[i],i))[:2])==[1,2]
    assert not torch.cuda.is_initialized()
    return dict(passed=True,tiny_HF_only=True,not_actual_model_quality=True,cases=results,
        missing_directory_gradient_roundtrip=True,persistent_optimizer_roundtrip=True,initial_shared_forward_exact=True,
        stage_local_initialization_preserves_RNG=True,first_B_gradient_zero=True,final_physical_shapes_checked=True,
        entire_graph_reload_exact=True,CUDA_initialized=False)
