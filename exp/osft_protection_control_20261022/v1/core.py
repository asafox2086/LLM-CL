"""Original OSFT construction/re-SVD, with both per-step protection callbacks off."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
PILOT=ROOT/'exp/osft_increment_screen_20261019/v1'
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/edge_deployment_20261019/osft_protection_control_20261022/v1'
WORK=Path(__file__).parent/'private'
spec=importlib.util.spec_from_file_location('protection_control_inherited_study',PILOT/'study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sha,write=study.sha,study.write

class UnprotectedOSFT:
    def __init__(self,model):
        sys.path.insert(0,str(PORT));from osft_adapter import OSFTMethod
        original=OSFTMethod(model,'OSFT',['unused'])
        self.__dict__.update(original.__dict__)
        # Keep name, HostAdamW, activation offload, author kernels and all initial factors.
        self.skipped_projection_hooks=dict(gradient=0,parameter=0)
    def set_instruction(self,task):
        from osft_adapter import OSFTMethod
        return OSFTMethod.set_instruction(self,task)
    def begin_stage(self,stage):
        from osft_adapter import OSFTMethod
        return OSFTMethod.begin_stage(self,stage)
    def post_backward(self,scale,progress):
        self.skipped_projection_hooks['gradient']+=1
    def after_update(self,step,updates):
        self.skipped_projection_hooks['parameter']+=1
    def method_state(self,stage):
        from osft_adapter import OSFTMethod
        return OSFTMethod.method_state(self,stage)|dict(per_step_projection_package='disabled',skipped_hooks=dict(self.skipped_projection_hooks))

def cpu_gate():
    import copy,torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    sys.path.insert(0,str(PORT));from osft_adapter import OSFTMethod
    torch.set_num_threads(2);torch.manual_seed(41)
    cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    base=Qwen2ForCausalLM(cfg);ref=copy.deepcopy(base);model=copy.deepcopy(base)
    torch.manual_seed(42);original=OSFTMethod(ref,'OSFT',['unused']);rng=torch.get_rng_state().clone()
    torch.manual_seed(42);method=UnprotectedOSFT(model)
    assert torch.equal(rng,torch.get_rng_state()) and len(method.layers)==len(original.layers)==14
    assert original.name==method.name=='OSFT' and original.learning_rate==method.learning_rate==1e-5
    assert original.host_optimizer==method.host_optimizer and original.offload_activations==method.offload_activations
    assert set(ref.state_dict())==set(model.state_dict())
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in ref.state_dict().items())
    assert {n:p.requires_grad for n,p in ref.named_parameters()}=={n:p.requires_grad for n,p in model.named_parameters()}
    ref.eval();model.eval();ids=torch.tensor([[2,4,7,9]])
    with torch.no_grad():assert torch.equal(ref(ids).logits,model(ids).logits)
    model(ids).logits.float().square().mean().backward()
    gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    layer=method.layers[0]
    # A gradient known to lie in the high subspace must survive the disabled callback.
    layer.U_low.grad=layer.U_high[:,:1].expand_as(layer.U_low).clone()
    gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    parameters={n:p.clone() for n,p in model.named_parameters()}
    method.post_backward(1.,0.);method.after_update(1,1)
    assert all(torch.equal(g,dict(model.named_parameters())[n].grad) for n,g in gradients.items())
    assert all(torch.equal(p,dict(model.named_parameters())[n]) for n,p in parameters.items())
    assert method.skipped_projection_hooks==dict(gradient=1,parameter=1)
    assert all(not getattr(l,k).requires_grad for l in method.layers for k in ['U_high','S_high','V_high'])
    assert not torch.cuda.is_initialized()
    return dict(passed=True,tiny_HF_layers=2,target_modules=14,original_initial_state_and_forward_and_RNG_exact=True,
        original_trainable_names_flags_and_optimizer_settings_equal=True,high_buffers_frozen=True,
        disabled_callbacks_preserve_parameters_and_high_subspace_gradient_exact=True,finite_backward=True,
        optimizer_steps=0,new_actual_model_requests=0)
