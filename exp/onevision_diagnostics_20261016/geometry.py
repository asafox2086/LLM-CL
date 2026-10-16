"""Exact low-rank deltaW geometry, invariant to invertible LoRA factors."""
import json,math
from pathlib import Path
import torch

def inner(a,b):
    # trace((B1 A1)^T B2 A2), without allocating a dense out*in update.
    return float(torch.trace((a['B'].double().T@b['B'].double())@(b['A'].double()@a['A'].double().T)))

def spectrum(a):
    _,rb=torch.linalg.qr(a['B'].double(),mode='reduced');_,ra=torch.linalg.qr(a['A'].double().T,mode='reduced')
    s=4*torch.linalg.svdvals(rb@ra.T);energy=s.square()
    if float(energy.sum())==0:return {'singular_values':s.tolist(),'energy95_rank':0,'entropy_effective_rank':0}
    p=energy/energy.sum();rank=int(torch.searchsorted(p.cumsum(0),torch.tensor(.95,dtype=torch.float64)))+1
    return {'singular_values':s.tolist(),'energy95_rank':rank,'entropy_effective_rank':float(torch.exp(-(p*p.clamp_min(1e-300).log()).sum()))}

def analyze(states,run):
    output={'definition':'actual deltaW=4*B*A; cosine computed by exact low-rank Gram traces','cosine':{},'spectra':{}}
    for d,state in states.items():output['spectra'][d]={key:spectrum(a) for key,a in state.items()}
    for d,a in states.items():
        output['cosine'][d]={}
        for e,b in states.items():
            numerator=sum(inner(a[k],b[k]) for k in a);na=sum(inner(a[k],a[k]) for k in a);nb=sum(inner(b[k],b[k]) for k in a)
            output['cosine'][d][e]=numerator/math.sqrt(na*nb) if na>0 and nb>0 else None
    (Path(run)/'lora_geometry.json').write_text(json.dumps(output,indent=2)+'\n')
