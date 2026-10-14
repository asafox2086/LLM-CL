import sys,json,torch,numpy as np
from pathlib import Path
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r));from hyper_adapter import author_components,HyperLinear
k,revision,hashes=author_components();torch.manual_seed(42);base=torch.nn.Linear(12,8,bias=False);layer=HyperLinear(base,k);layer.lora_dropout["default"]=torch.nn.Identity();x=torch.randn(2,3,12);assert torch.equal(layer(x),base(x))
layer.cur_task=3
with torch.no_grad():layer.lora_B["default"].loraB[3].mlp.weight.fill_(.1)
y=layer(x);y.square().mean().backward();assert layer.lora_B["default"].loraB[3].mlp.weight.grad is not None and layer.lora_B["default"].loraB[2].mlp.weight.grad is None
layer.eval();layer.expert_weight=[0.]*10;layer.expert_weight[3]=1.;assert torch.allclose(layer(x),y,atol=1e-6)
u=k["map_to_poincare"](torch.tensor([[1.,2.,3.]]));v=k["map_to_poincare"](torch.tensor([[3.,2.,1.]]));assert torch.allclose(k["poincare_dist"](u,v),k["poincare_dist"](v,u));assert k["poincare_dist"](u,u)<k["poincare_dist"](u,v)
stats=[{"img_mean":np.array([0.,0.]),"img_var":np.ones(2),"txt_mean":np.array([0.,0.]),"txt_var":np.ones(2)},{"img_mean":np.array([8.,8.]),"img_var":np.ones(2),"txt_mean":np.array([.1,.1]),"txt_var":np.ones(2)}];weights=k["compute_current_stage_weights"](stats);assert all(w>.5 for w in weights)

print("PASS: verify_hyper_core.py; original author kernels verified on CPU")
