import sys,json,torch
from pathlib import Path
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r))
from osft_adapter import author_functions,OSFTLinear
k,revision,sha=author_functions();torch.manual_seed(42);base=torch.nn.Linear(12,8,bias=False);before=base.weight.detach().clone();layer=OSFTLinear(base,k);assert torch.allclose(k["reconstruct_weight_matrix"](layer.svd(),torch.float32),before,atol=2e-6)
frozen=layer.U_high.clone();x=torch.randn(4,12);loss=layer(x).square().mean();loss.backward();k["project_gradient_to_orthogonal_space"](layer.svd());assert torch.linalg.norm(layer.U_high.T@layer.U_low.grad)<1e-5;assert torch.linalg.norm(layer.V_low.grad@layer.V_high.T)<1e-5
opt=torch.optim.AdamW([layer.U_low,layer.S_low,layer.V_low],lr=.001);opt.step()
with torch.no_grad():k["project_parameter_to_orthogonal_space"](layer.svd())
assert torch.linalg.norm(layer.U_high.T@layer.U_low)<1e-5 and torch.linalg.norm(layer.V_low@layer.V_high.T)<1e-5;assert torch.equal(layer.U_high,frozen)

print("PASS: verify_osft_core.py; original author kernels verified on CPU")
