"""Trusted full method graphs: preserve architecture, merged weights, routers and runtime."""
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import cloudpickle
import torch
from runtime_utils import rng, restore_rng, write


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''):h.update(b)
    return h.hexdigest()


def save(method, processor, run, stage, config, probe=None):
    folder=Path(run)/'checkpoints';folder.mkdir(exist_ok=True)
    path=folder/f'stage_{stage:02d}.pt'
    assert not path.exists(), 'Never overwrite a durable stage artifact'
    temporary=path.with_suffix('.pt.partial');started=time.monotonic()
    # Optimizers are deliberately recreated at every stage in this experiment.
    # Saving one here would add no stage-boundary recovery information.
    # exec injects the live builtins dictionary into reviewed author namespaces.
    # PyBind adds a process-local PyCapsule there. It is not learned state and must
    # not be serialized. Functions keep their builtin lookup; Cloudpickle restores it.
    for obj in [method]+list(method.model.modules()):
        for key in ['kernel','kernels','k']:
            ns=getattr(obj,key,None)
            if isinstance(ns,dict):ns.pop('__builtins__',None)
    payload={'format':'llmcl-method-graph-v1','method':method,'processor':processor,
             'stage':stage,'config':config,'rng':rng(),'probe':probe,
             'optimizer_lifecycle':'fresh optimizer/scheduler/scaler at each stage',
             'checkpoint_phase':'after_stage_evaluation' if probe is None else 'preflight'}
    torch.save(payload,temporary,pickle_module=cloudpickle)
    with temporary.open('rb') as f:os.fsync(f.fileno())
    temporary.replace(path)
    meta={'path':str(path.relative_to(run)),'stage':stage,'bytes':path.stat().st_size,
          'sha256':sha(path),'seconds':time.monotonic()-started,'manifest_sha256':config['manifest_sha256'],
          'method':config['method'],'full_architecture_and_runtime':True,'optimizer_saved':False}
    write(path.with_suffix('.json'),meta);write(folder/'latest.json',meta)
    # Also export small portable metadata and processor for downstream inspection.
    processor.save_pretrained(folder/'processor')
    return meta


def load(run, expected=None):
    run=Path(run);meta=json.loads((run/'checkpoints/latest.json').read_text())
    path=(run/meta['path']).resolve();assert path.is_relative_to((run/'checkpoints').resolve())
    assert sha(path)==meta['sha256']
    # Only our own trusted artifacts. Preserve original CPU/CUDA placements (virtual CUDA:0).
    obj=torch.load(path,pickle_module=pickle,weights_only=False)
    # Paths are host-local; weights and original CPU/CUDA placements stay intact.
    if hasattr(obj['method'],'io'):obj['method'].io.run=run
    if hasattr(obj['method'],'state_root'):obj['method'].state_root=run/'method_state'
    if expected:
        for k in ['method','condition','manifest_sha256','execution_source_sha256']:
            assert obj['config'][k]==expected[k], 'Checkpoint protocol/source mismatch: '+k
    restore_rng(obj['rng'])
    return obj
