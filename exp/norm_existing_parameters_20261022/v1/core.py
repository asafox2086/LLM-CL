"""Existing block RMSNorm gains only; preserve original FP16 forward arithmetic."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
PILOT=ROOT/'exp/osft_increment_screen_20261019/v1'
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/edge_deployment_20261019/norm_existing_parameters_20261022/v1'
WORK=Path(__file__).parent/'private'
spec=importlib.util.spec_from_file_location('norm_inherited_study',PILOT/'study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sha,write=study.sha,study.write

def original_norm_source():
    import hashlib,inspect,transformers
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    source=inspect.getsource(Qwen2RMSNorm)
    return dict(transformers=transformers.__version__,class_source=source,
                class_sha256=hashlib.sha256(source.encode()).hexdigest())

class NormOnly:
    def __init__(self,model):
        import torch
        from torch import nn
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
        class MasterRMSNorm(nn.Module):
            def __init__(self,original):
                super().__init__()
                self.forward_weight_dtype=original.weight.dtype
                self.weight=nn.Parameter(original.weight.detach().float().clone())
                self.variance_epsilon=original.variance_epsilon
            def forward(self,hidden_states):
                input_dtype=hidden_states.dtype
                hidden_states=hidden_states.to(torch.float32)
                variance=hidden_states.pow(2).mean(-1,keepdim=True)
                hidden_states=hidden_states*torch.rsqrt(variance+self.variance_epsilon)
                # FP32 master for optimizer; original FP16 weight is used by forward.
                return self.weight.to(self.forward_weight_dtype)*hidden_states.to(input_dtype)
        for p in model.parameters():p.requires_grad_(False)
        self.model=model;self.name='RMSNorm_only';self.learning_rate=1e-4
        self.layers=[];self.names=[];self.update_evidence=[]
        self.host_optimizer=False;self.offload_activations=False
        for i,block in enumerate(model.model.layers):
            for name in ('input_layernorm','post_attention_layernorm'):
                original=getattr(block,name)
                assert type(original) is Qwen2RMSNorm and original.weight.dtype==torch.float16
                assert set(dict(original.named_parameters()))=={'weight'}
                replacement=MasterRMSNorm(original);setattr(block,name,replacement)
                self.layers.append(replacement);self.names.append(f'model.layers.{i}.{name}.weight')
        assert len(self.layers)==2*len(model.model.layers)
        assert {id(p) for p in model.parameters() if p.requires_grad}=={id(l.weight) for l in self.layers}
    def set_instruction(self,task):pass
    def snapshot(self):
        return {n:dict(master=l.weight.detach().cpu().clone(),forward=l.weight.detach().to(l.forward_weight_dtype).cpu().clone())
                for n,l in zip(self.names,self.layers)}
    def begin_stage(self,stage):
        self.stage=stage;self._stage_start=self.snapshot();self._previous=self._stage_start
    def after_update(self,step,updates):
        import torch
        now=self.snapshot()
        def differences(before):
            result={}
            for kind in ('master','forward'):
                a=torch.cat([before[n][kind].reshape(-1) for n in self.names])
                b=torch.cat([now[n][kind].reshape(-1) for n in self.names])
                delta=b.float()-a.float()
                assert torch.isfinite(delta).all()
                result[kind]=dict(changed=int((a!=b).sum()),l2=float(delta.norm()),total=a.numel())
            return result
        entry=dict(stage=self.stage,step=step,updates=updates,from_previous=differences(self._previous))
        if step==updates:entry['from_stage_start']=differences(self._stage_start)
        self.update_evidence.append(entry);self._previous=now
    def method_state(self,stage):
        return dict(stage=stage,trainable_names=self.names,coordinates=sum(l.weight.numel() for l in self.layers),
                    master_dtype='float32',forward_weight_dtype='float16',existing_gains_only=True)

def cpu_gate():
    import copy,io,pickle,torch,cloudpickle
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.set_num_threads(2);torch.manual_seed(41)
    cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    ref=Qwen2ForCausalLM(cfg).half().eval();model=copy.deepcopy(ref)
    rng=torch.get_rng_state().clone();method=NormOnly(model)
    assert torch.equal(rng,torch.get_rng_state())
    assert len(method.layers)==4 and sum(p.numel() for p in model.parameters() if p.requires_grad)==128
    a=ref.state_dict();b=model.state_dict();assert a.keys()==b.keys()
    assert all(torch.equal(a[n],b[n]) for n in a)
    assert all(p.dtype==torch.float32 for p in model.parameters() if p.requires_grad)
    assert model.model.norm.weight.dtype==torch.float16 and not model.model.norm.weight.requires_grad
    ids=torch.tensor([[2,4,7,9]])
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):
        logits=ref(ids).logits;assert torch.equal(logits,model(ids).logits)
    stream=io.BytesIO();torch.save(method,stream,pickle_module=cloudpickle);stream.seek(0)
    recovered=torch.load(stream,pickle_module=pickle,weights_only=False)
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):assert torch.equal(logits,recovered.model(ids).logits)
    with torch.autocast('cpu',dtype=torch.float16):model(ids).logits.float().square().mean().backward()
    gradients=[p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and g.dtype==torch.float32 and torch.isfinite(g).all() for g in gradients)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    method.begin_stage(1);method.after_update(1,1)
    assert all(v['changed']==0 for v in method.update_evidence[0]['from_previous'].values())
    assert not torch.cuda.is_initialized()
    return dict(passed=True,tiny_HF_layers=2,target_modules=4,trainable_coordinates=128,
        initial_state_values_and_forward_and_RNG_exact=True,all_other_weights_frozen=True,
        master_dtype='float32',forward_weight_dtype='float16',finite_backward=True,
        tiny_method_graph_roundtrip_exact=True,zero_update_recorder_exact=True,
        optimizer_steps=0,new_actual_model_requests=0,original_norm_source=original_norm_source())
