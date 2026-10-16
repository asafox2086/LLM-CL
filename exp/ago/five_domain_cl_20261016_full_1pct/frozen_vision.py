"""Cache immutable visual outputs only; never cache labels or trainable vision."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch
from methods import ROOT

CACHE=ROOT/'data/TenTaskCL/cache/frozen_qwen2vl_v1'
LIMIT=12*1024**3


def install(model):
    if any(parameter.requires_grad for parameter in model.visual.parameters()):
        raise ValueError('visual caching would invalidate a trainable visual module')
    original=model.visual.forward
    def forward(hidden_states,grid_thw):
        expected=int(grid_thw.prod(1).sum().item())//4
        if hidden_states.shape==(expected,model.config.hidden_size):
            return hidden_states
        return original(hidden_states,grid_thw)
    model.visual.forward=forward
    return original


def prepare(batch,row,model,original):
    if 'pixel_values' not in batch:
        return False
    CACHE.mkdir(parents=True,exist_ok=True)
    asset=ROOT/'model/Qwen2-VL-2B-Instruct/asset_manifest.json'
    model_sha=hashlib.sha256(asset.read_bytes()).hexdigest()
    images=[]
    for name in row['images']:
        path=ROOT/name
        stat=path.stat()
        images.append([name,stat.st_size,stat.st_mtime_ns])
    key=hashlib.sha256(json.dumps({'base':model_sha,'images':images,
        'grid':batch['image_grid_thw'].tolist(),'torch':torch.__version__,'precision':'fp16'},
        sort_keys=True).encode()).hexdigest()
    path=CACHE/(key+'.pt')
    if path.is_file():
        features=torch.load(path,map_location='cpu',weights_only=True)
        hit=True
    else:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            features=original(batch['pixel_values'].to(model.visual.get_dtype()),batch['image_grid_thw']).detach().cpu()
        hit=False
        temporary=CACHE/(key+f'.{os.getpid()}.partial')
        torch.save(features,temporary)
        with (CACHE/'.quota.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            ledger=CACHE/'size.json'
            try:
                current=json.loads(ledger.read_text())['bytes'] if ledger.exists() else sum(p.stat().st_size for p in CACHE.glob('*.pt'))
            except (ValueError,KeyError):
                current=sum(p.stat().st_size for p in CACHE.glob('*.pt'))
            if path.exists():
                temporary.unlink()
            elif current+temporary.stat().st_size<=LIMIT:
                current+=temporary.stat().st_size
                temporary.replace(path)
                ledger_temp=CACHE/f'.size.{os.getpid()}.tmp'
                ledger_temp.write_text(json.dumps({'bytes':current,'limit':LIMIT})+'\n')
                ledger_temp.replace(ledger)
            else:
                temporary.unlink()
    expected=int(batch['image_grid_thw'].prod(1).sum().item())//4
    if features.shape!=(expected,model.config.hidden_size) or not torch.isfinite(features).all():
        raise ValueError('invalid visual cache feature layout')
    batch['pixel_values']=features.to(model.device,dtype=model.visual.get_dtype())
    return hit
