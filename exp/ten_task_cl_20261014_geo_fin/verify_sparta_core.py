import sys,json,torch
from pathlib import Path
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r));from sparta_adapter import author_components,Router
kernels,revision,hashes=author_components();torch.manual_seed(42);base=torch.nn.Linear(12,8);layer=kernels["DATAInjectedLinear"](12,8,True,r=2,r2=8);layer.linear_data=base;x=torch.randn(2,3,12);assert torch.equal(layer(x),base(x));layer(x).square().mean().backward();assert layer.data_up.weight.grad is not None and layer.data_up2.weight.grad is not None
router=Router(12,kernels);router.config.adaprompt=2;router.config.ortho_mu=0;router.allocate(1);old=router.data_k_0[:2].detach().clone();router.allocate(2);assert torch.equal(old,router.data_k_0[:2]);assert torch.linalg.norm(router.data_k_0[:2]@router.data_k_0[2:4].T)<1e-5
low,high,penalty=router(x);assert low.shape==(2,1,1) and torch.isfinite(low).all();(low.sum()+high.sum()).backward();assert torch.count_nonzero(router.data_k_0.grad[:2])==0 and torch.count_nonzero(router.data_k_0.grad[2:4])>0
print("PASS: verify_sparta_core.py; original author kernels verified on CPU")
