"""One final parameter artifact per finished run; no optimizer or stage checkpoints."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

# Input caches, datasets, live hooks and repair-teacher models are not inference state.
OMIT={'model','processor','tokenizer','text_tokenizer','text_encoder','io','kernel','layers',
      'features','stage_features','train_ids','handles','ref_model','searcher','cached_prompt',
      'vision_original','learner','ref_adapter','_embedding_weight_snapshot'}

def portable(value):
    if value is None or isinstance(value,(bool,int,float,str)):return value
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,np.ndarray):return torch.from_numpy(value.copy())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,dict):
        if not all(isinstance(k,(str,int,float,bool,tuple)) for k in value):raise TypeError('unsupported dictionary key')
        return {k:portable(v) for k,v in value.items()}
    raise TypeError(type(value).__name__)

def runtime(obj):
    kept={};omitted={}
    for key,value in vars(obj).items():
        if key in OMIT or key in {'_parameters','_buffers','_modules'}:continue
        try:kept[key]=portable(value)
        except TypeError as exc:omitted[key]=str(exc)
    return {'values':kept,'nonportable_fields':omitted}

def save_final(method,processor,run,manifest,config):
    run=Path(run);destination=run/'final_parameters';temporary=run/'final_parameters.partial'
    assert not destination.exists() and not temporary.exists(),'Never overwrite an existing final artifact'
    temporary.mkdir()
    # Full state includes merged base weights, frozen experts and registered buffers.
    # Saving only currently requires_grad parameters would lose earlier frozen experts.
    state={name:value.detach().cpu() for name,value in method.model.state_dict().items()}
    torch.save(state,temporary/'model_state.pt')
    auxiliary={'method_state':portable(method.method_state(len(manifest['tasks']))),
               'method_runtime':runtime(method),
               'layer_runtime':{name:runtime(module) for name,module in method.model.named_modules()
                                if module.__class__.__module__ not in {'torch.nn.modules.linear','torch.nn.modules.normalization'}
                                and any(key in vars(module) for key in ['expert_weight','cur_task','gate_tau','ins_type','enabled','task_id'])}}
    torch.save(auxiliary,temporary/'method_state.pt')
    method.model.config.save_pretrained(temporary)
    processor.save_pretrained(temporary)
    (temporary/'run_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    (temporary/'task_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    sources=temporary/'runner_source';sources.mkdir()
    for p in Path(__file__).parent.glob('*.py'):(sources/p.name).write_bytes(p.read_bytes())
    files=[]
    for p in sorted(temporary.rglob('*')):
        if not p.is_file():continue
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*2**20),b''):digest.update(chunk)
        files.append({'path':str(p.relative_to(temporary)),'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
    metadata={'format':'llmcl-final-parameters-v1','method':config['method'],'condition':config['condition'],
              'completed_stages':len(manifest['tasks']),'parameter_tensor_count':len(state),
              'parameter_tensor_bytes':sum(v.numel()*v.element_size() for v in state.values()),
              'files':files,'total_file_bytes':sum(x['bytes'] for x in files),
              'optimizer_saved':False,'stage_checkpoints_saved':False,
              'base_model':config['base_model'],'base_model_files':config.get('base_model_files'),
              'reloading':'Rebuild the custom method architecture using runner_source, load model_state.pt strictly, then restore routing/runtime from method_state.pt. These are not stock Transformers weights.',
              'verification':'file SHA256 and state schema; fresh-process inference equivalence must be verified before downstream evaluation'}
    base_files=[]
    for p in sorted(Path(config['base_model']).glob('*')):
        if not p.is_file() or p.suffix not in {'.json','.safetensors'}:continue
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*2**20),b''):digest.update(chunk)
        base_files.append({'path':p.name,'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
    metadata['base_model_files']=base_files
    (temporary/'artifact_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,destination)
    return metadata
