"""Physical FFN factor replacement; original dense weights are not retained."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
PILOT=ROOT/'exp/osft_increment_screen_20261019/v1'
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/final_weight_reduction_20261023/factor_pilot_v1'
WORK=Path(__file__).parent/'private'
spec=importlib.util.spec_from_file_location('factor_pilot_original_study',PILOT/'study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sha=study.sha
def write(path,value):
    import os
    study.write(path,value)
    descriptor=os.open(str(Path(path).parent),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)
RANK=540
BASE=893675552
FINAL=803837984
TRAINABLE=540672

def digest(params):
    from training_validity import parameter_digest
    return parameter_digest(list(params))

def layout(model):
    """Parameters de-duplicate tied objects; also disclose all registered buffers."""
    ps=dict(model.named_parameters());bs=dict(model.named_buffers())
    storages=[(str(p.device),p.untyped_storage().data_ptr()) for p in ps.values()]
    # named_parameters already removes tied Parameter aliases. Detect additional
    # distinct Parameter views before using its element sum as a weight count.
    assert len(set(storages))==len(storages),'Unexpected distinct Parameters sharing storage; count must be reviewed'
    return dict(weight_coefficients=sum(p.numel() for p in ps.values()),
        distinct_Parameter_storage_aliases=0,tied_Parameter_objects_deduplicated=True,
        trainable_coordinates=sum(p.numel() for p in ps.values() if p.requires_grad),
        parameter_bytes=sum(p.numel()*p.element_size() for p in ps.values()),
        buffers={n:dict(shape=list(b.shape),dtype=str(b.dtype),bytes=b.numel()*b.element_size()) for n,b in bs.items()},
        parameters={n:dict(shape=list(p.shape),dtype=str(p.dtype),trainable=p.requires_grad) for n,p in ps.items()})

def compress(model,rank=RANK):
    import time,torch
    from torch import nn
    class Factors(nn.Module):
        def __init__(self,left,right,bias):
            super().__init__()
            self.left=nn.Parameter(left,requires_grad=False)
            self.right=nn.Parameter(right,requires_grad=False)
            self.bias=None if bias is None else nn.Parameter(bias.detach().clone(),requires_grad=False)
            self.in_features=right.shape[1];self.out_features=left.shape[0]
        def forward(self,x):
            return torch.nn.functional.linear(torch.nn.functional.linear(x,self.right),self.left,self.bias)
    records=[];cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None
    for i,block in enumerate(model.model.layers):
        for name in ['gate_proj','up_proj','down_proj']:
            old=getattr(block.mlp,name);assert type(old) is nn.Linear
            m,n=old.weight.shape;assert 0<rank<min(m,n) and rank*(m+n)<m*n
            tick=time.monotonic()
            with torch.no_grad():
                w=old.weight.detach().float()
                kwargs={'driver':'gesvd'} if w.is_cuda else {}
                u,s,vh=torch.linalg.svd(w,full_matrices=False,**kwargs)
                root=s[:rank].sqrt()
                left=(u[:,:rank]*root).to(old.weight.dtype).contiguous()
                right=(root[:,None]*vh[:rank]).to(old.weight.dtype).contiguous()
                replacement=Factors(left,right,old.bias)
                assert all(torch.isfinite(p).all() for p in replacement.parameters())
            records.append(dict(name=f'model.layers.{i}.mlp.{name}',original_shape=[m,n],rank=rank,
                original_coefficients=m*n,factor_coefficients=rank*(m+n),seconds=time.monotonic()-tick))
            setattr(block.mlp,name,replacement)
            del old,w,u,s,vh,left,right,root,replacement
    assert torch.equal(cpu_rng,torch.get_rng_state())
    if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
    assert len(records)==3*len(model.model.layers)
    return records

def save_gradient(model,run,stage):
    import torch
    folder=run/'gradients';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'S{stage:02d}_first_raw.pt'
    torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},path)
    return path

def measured_optimizer(method,factory,record):
    from types import MethodType
    opt=factory(method);expected={id(p) for p in method.model.parameters() if p.requires_grad}
    original=type(opt).step
    def step(self,*args,**kwargs):
        params=[p for g in self.param_groups for p in g['params']]
        assert {id(p) for p in params}==expected and all(p.grad is not None for p in params)
        grad=sum(p.grad.numel()*p.grad.element_size() for p in params)
        value=original(self,*args,**kwargs)
        moments=[self.state[p][k] for p in params for k in ['exp_avg','exp_avg_sq']]
        record(dict(parameter_IDs_match=True,coordinates=sum(p.numel() for p in params),
            gradient_bytes=grad,two_moment_bytes=sum(t.numel()*t.element_size() for t in moments),
            moment_devices=sorted({str(t.device) for t in moments})))
        return value
    opt.step=MethodType(step,opt)
    return opt

def persist_optimizer(run,value,stage,step):
    value=dict(value,stage=stage,step=step)
    write(run/'optimizer'/f'S{stage:02d}_{step:03d}.json',value)
    return value

def cpu_gate():
    import copy,io,pickle,tempfile,torch,cloudpickle
    from transformers import Qwen2Config,Qwen2ForCausalLM
    sys.path.insert(0,str(PORT))
    from runtime_utils import make_optimizer
    torch.set_num_threads(2);torch.manual_seed(41)
    cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    reference=Qwen2ForCausalLM(cfg).half().eval();model=copy.deepcopy(reference)
    before=sum(p.numel() for p in model.parameters())
    records=compress(model,rank=16)
    assert sum(p.numel() for p in model.parameters())==before-6*(32*64-16*96)
    torch.manual_seed(42);refmethod=study.make_method(reference,'LoRA_qv',['x'])
    torch.manual_seed(42);method=study.make_method(model,'LoRA_qv',['x'])
    for n,p in model.named_parameters():
        if '.A.' in n or '.B.' in n:assert torch.equal(p,dict(reference.named_parameters())[n])
    assert not any(n.endswith(('.gate_proj.weight','.up_proj.weight','.down_proj.weight')) for n in model.state_dict())
    frozen=[p for p in model.parameters() if not p.requires_grad];before_frozen=digest(frozen)
    ids=torch.tensor([[2,4,7,9]]);model.train()
    with torch.autocast('cpu',dtype=torch.float16):loss=model(ids).logits.float().square().mean()
    loss.backward();assert all(p.grad is None for p in frozen)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    with tempfile.TemporaryDirectory() as td:
        records_opt=[]
        def commit(v):
            entry=persist_optimizer(Path(td)/'new_run',v,1,1);records_opt.append(entry)
        opt=measured_optimizer(method,make_optimizer,commit)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        opt.step();assert digest(frozen)==before_frozen and len(records_opt)==1
        import json
        assert json.loads((Path(td)/'new_run/optimizer/S01_001.json').read_text())==records_opt[0]
        path=save_gradient(model,Path(td)/'new_run',1)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        actual={n:p.grad for n,p in model.named_parameters() if p.requires_grad}
        assert saved.keys()==actual.keys() and all(torch.equal(saved[n],actual[n]) for n in saved)
    model.eval()
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):y=model(ids).logits
    stream=io.BytesIO();torch.save(method,stream,pickle_module=cloudpickle);stream.seek(0)
    recovered=torch.load(stream,pickle_module=pickle,weights_only=False)
    assert layout(recovered.model)==layout(model)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):assert torch.equal(y,recovered.model(ids).logits)
    # Full rank is intentionally not used: the tiny gate verifies actual physical reduction.
    assert not torch.cuda.is_initialized()
    return dict(passed=True,physical_factor_modules=len(records),original_dense_keys_absent=True,
        qv_initialization_exact=True,compression_RNG_unchanged=True,finite_backward=True,
        frozen_factors_and_other_base_unchanged=True,actual_optimizer_step=True,
        optimizer_witness_persisted_and_reloaded=True,
        missing_gradient_directory_roundtrip_exact=True,full_method_graph_roundtrip_exact=True,
        CPU_only=True,not_actual_model_quality=True)
