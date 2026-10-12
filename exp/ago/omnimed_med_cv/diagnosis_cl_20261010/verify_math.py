"""Check ported analytic formulas and the actual shared stream."""
import json
from pathlib import Path
import torch
from spectral_lora import SpectralLinear
torch.set_num_threads(2);torch.manual_seed(17)
exp=Path(__file__).resolve().parent
base=torch.nn.Linear(24,16,bias=False)
layer=SpectralLinear(base,{'dropout':0},'verification_24x16',True);layer.initialize();layer.eval()
x=torch.randn(3,5,24);y=layer(x)
raw=x.reshape(-1,24);prob=layer.gate(raw).softmax(-1);values,idx=prob.topk(2,-1);values=values/values.sum(-1,keepdim=True)
gate=torch.zeros_like(prob).scatter(1,idx,values)
res=sum(a.initial_up@a.initial_down for a in layer.adapters)*layer.scale/8
expected=torch.nn.functional.linear(raw,base.weight-res)
for i,a in enumerate(layer.adapters):expected=expected+torch.nn.functional.linear(raw,a['up'].weight@a['down'].weight)*gate[:,i:i+1]*layer.scale
error=float((y.reshape_as(expected)-expected).abs().max());assert error<2e-5,error
loss=y.square().mean()+layer.balance*.001;loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters() if p.requires_grad)
h=torch.randn(17,29,dtype=torch.float64);labels=torch.randn(17,4,dtype=torch.float64);ridge=.03
u,s,vh=torch.linalg.svd(h,full_matrices=False)
dual=vh.T@((s/(s.square()+ridge)).unsqueeze(1)*(u.T@labels))
direct=torch.linalg.solve(h.T@h+ridge*torch.eye(29,dtype=torch.float64),h.T@labels)
ridge_error=float((dual-direct).abs().max());assert ridge_error<1e-10,ridge_error
stream=[]
for name in ['seq_lora_seed42','migu_lora_seed42','sapt_lora_seed42']:
    path=exp/'runs'/name/'updates.jsonl';records=[json.loads(s) for s in path.read_text().splitlines()]
    stream.append([(r['stage'],r['epoch'],r['step'],r['questions']) for r in records])
n=min(map(len,stream));assert n>0
assert stream[0][:n]==stream[1][:n]==stream[2][:n]
result=dict(spectral_dense_equivalence_max_error=error,finite_spectral_gradients=True,
            ridge_dual_primal_max_error=ridge_error,shared_training_updates_checked=n)
(exp/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
