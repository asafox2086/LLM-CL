"""One budget-derived q/v allocation, retaining the frozen r8 initialization."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
PILOT=ROOT/'exp/osft_increment_screen_20261019/v1'
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/edge_deployment_20261019/position_coordinate_control_20261022/v1'
WORK=Path(__file__).parent/'private'
spec=importlib.util.spec_from_file_location('position_control_inherited_study',PILOT/'study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sha,write=study.sha,study.write

def rank_plan(layers=24):
    v=[f'model.layers.{i}.self_attn.v_proj' for i in range(layers)]
    chosen=set(sorted(v,key=lambda n:hashlib.sha256(('42:coordinate-remainder:'+n).encode()).hexdigest())[:6])
    return {f'model.layers.{i}.self_attn.{t}_proj':65+int(f'model.layers.{i}.self_attn.{t}_proj' in chosen)
            for i in range(layers) for t in ('q','v')}

class BudgetQV:
    def __init__(self,model,plan):
        import torch
        from torch import nn
        sys.path.insert(0,str(PORT));from methods import SeqLinear
        bases={n:l for n,l in model.named_modules() if n in plan and isinstance(l,nn.Linear)}
        assert set(bases)==set(plan)
        old=study.make_method(model,'LoRA_qv',['unused'])
        self.model=model;self.name=old.name;self.embedding=None;self.layers=[]
        self.learning_rate=1e-4;self.original_revision='one-exact-coordinate-budget-qv-allocation-scale4'
        self.rank_plan=dict(plan);self.initialization=[]
        for name,base in bases.items():
            original=model.get_submodule(name);rank=plan[name]
            devices=[base.weight.device.index] if base.weight.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                seed=int(hashlib.sha256(('42:coordinate-extra-init:'+name).encode()).hexdigest()[:8],16)
                torch.manual_seed(seed)
                layer=SeqLinear(base,rank=rank,alpha=4*rank)
                with torch.no_grad():
                    layer.A.weight[:8].copy_(original.A.weight)
                    layer.B.weight[:,:8].copy_(original.B.weight)
            assert layer.scale==4 and torch.equal(layer.A.weight[:8],original.A.weight)
            assert torch.equal(layer.B.weight[:,:8],original.B.weight) and torch.count_nonzero(layer.B.weight)==0
            assert layer.weight is base.weight and layer.bias is base.bias
            parent,child=name.rsplit('.',1);setattr(model.get_submodule(parent),child,layer)
            self.layers.append(layer)
            self.initialization.append(dict(module=name,rank=rank,alpha=4*rank,scale=4,prefix_r8_exact=True,
                                            extra_B_zero=True,extra_A_seed=seed))
        assert len(self.layers)==len(plan)
    def set_instruction(self,task):pass
    def begin_stage(self,stage):pass

def cpu_gate():
    import copy,torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.set_num_threads(2)
    torch.manual_seed(41)
    # Actual 24-layer q/v names, on a tiny HF model; ranks remain within in/out widths here.
    cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=24,
                   num_attention_heads=4,num_key_value_heads=2)
    base=Qwen2ForCausalLM(cfg).eval();ref=copy.deepcopy(base);wide=copy.deepcopy(base)
    torch.manual_seed(42);study.make_method(ref,'LoRA_qv',['x']);reference_rng=torch.get_rng_state().clone()
    tiny={n:12+int(r==66) for n,r in rank_plan().items()}
    torch.manual_seed(42);method=BudgetQV(wide,tiny)
    assert torch.equal(reference_rng,torch.get_rng_state())
    assert len(method.layers)==48 and all(l.scale==4 for l in method.layers)
    for name,parameter in base.named_parameters():
        assert torch.equal(parameter,dict(wide.named_parameters())[name]),name
        assert not dict(wide.named_parameters())[name].requires_grad
    for name,rank in tiny.items():
        a,b=ref.get_submodule(name),wide.get_submodule(name)
        assert torch.equal(a.A.weight,b.A.weight[:8]) and torch.equal(a.B.weight,b.B.weight[:,:8])
        assert b.A.weight.shape[0]==rank and torch.count_nonzero(b.B.weight)==0
    ref.eval();wide.eval()
    ids=torch.tensor([[2,4,7,9]])
    with torch.no_grad():assert torch.equal(base(ids).logits,ref(ids).logits) and torch.equal(ref(ids).logits,wide(ids).logits)
    # Nonzero extra coordinates must act through the inherited forward, including FP32 adapters.
    layer=method.layers[0];x=torch.randn(2,32)
    with torch.no_grad():
        layer.B.weight[:,8:].fill_(.01)
        actual=layer(x);expected=torch.nn.functional.linear(x,layer.weight,layer.bias)+4*layer.B(layer.A(x))
        assert torch.equal(actual,expected) and not torch.equal(actual,torch.nn.functional.linear(x,layer.weight,layer.bias))
    out=wide(ids).logits;out.float().square().mean().backward()
    grads=[p.grad for p in wide.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert not torch.cuda.is_initialized()
    return dict(passed=True,tiny_HF_layers=24,target_modules=48,prefix_and_frozen_base_and_RNG_exact=True,
        zero_initialized_function_exact=True,nonzero_extra_forward_exact=True,finite_backward=True,
        tiny_ranks='12/13 for structural admission; not the actual 65/66 experiment',new_actual_model_requests=0)
