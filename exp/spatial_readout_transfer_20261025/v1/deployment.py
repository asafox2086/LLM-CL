"""GPU construction and self-contained complete spatial-model checkpoints.

No forward/backward/optimizer/fold is performed on CPU. Meta construction has
no numerical parameter initialization; ordinary tensor I/O may use CPU.
"""
from pathlib import Path
import gc,hashlib,json

FORMAT='spatial-readout-complete-v1'
WEIGHTS={'coefficients':591534976,'bytes':1183069952}

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):value.update(b)
    return value.hexdigest()

def jd(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def functional(value):
    if isinstance(value,dict):return {k:functional(v) for k,v in value.items() if k!='_name_or_path'}
    if isinstance(value,(list,tuple)):return [functional(v) for v in value]
    return value

def aliases(items):
    groups={}
    for n,t in items:
        key=(str(t.device),t.untyped_storage().data_ptr(),t.untyped_storage().nbytes(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),str(t.dtype))
        groups.setdefault(key,[]).append(n)
    return sorted(sorted(v) for v in groups.values() if len(v)>1)

def receipt(model,storage,compact=True):
    import torch
    import interface
    core=model.core;device=next(core.parameters()).device;assert device.type=='cuda'
    assert all(p.device==device and p.dtype==torch.float16 and not p.requires_grad and p.grad is None for p in core.parameters())
    assert all(b.device==device for b in core.buffers())
    layout=storage.layout(core)
    assert layout['weights']==(WEIGHTS if compact else {'coefficients':893675552,'bytes':1787351104})
    assert layout['trainable']==0
    assert model.model is core.language_model.model and model.lm_head is core.language_model.lm_head
    assert model.visual.tower is core.vision_tower and model.visual.projector is core.multi_modal_projector
    assert model.visual.image_newline is core.image_newline
    assert core.language_model.config._attn_implementation=='sdpa'
    if compact:
        assert core.vision_tower.student.config._attn_implementation=='eager'
        assert tuple(core.vision_tower.compiled.weight.shape)==(896,12288)
        assert not any(n.endswith(('.A.weight','.B.weight')) for n in core.state_dict())
    buffers=list(core.named_buffers(remove_duplicate=False))
    for n,b in buffers:
        if n.endswith('inv_freq'):assert b.dtype==torch.float32,n
    return dict(layout=layout,digest=storage.digest(core),all_buffers_digest=storage.digest(buffers),
        parameter_alias_groups=aliases(core.named_parameters(remove_duplicate=False)),device=str(device),
        functional_config_digest=jd(functional(core.config.to_dict())),
        generation_config_digest=jd(core.generation_config.to_dict()),
        language_generation_config_digest=jd(core.language_model.generation_config.to_dict()),
        interface=interface.metadata() if compact else None,
        language_attention_classes=sorted({type(l.self_attn).__name__ for l in model.model.layers}))

def load_base(base,device,storage):
    import torch
    import onevision_bridge as bridge
    device=torch.device(device);assert device.type=='cuda'
    core,info=bridge.LlavaOnevisionForConditionalGeneration.from_pretrained(
        base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True,
        device_map={'':str(device)},low_cpu_mem_usage=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
    model=bridge.OneVisionBridge(core).requires_grad_(False).eval()
    return model,receipt(model,storage,False)

def load_student(path,device):
    import torch
    import interface
    value=torch.load(path,map_location='cpu',weights_only=True)
    tower=interface.build_student(value['config'],device)
    tower.load_state_dict(value['state'],strict=True)
    assert sum(p.numel() for p in tower.parameters())==86433024
    return tower,value['config']

def export(model,path,student_config,storage):
    import torch
    value=receipt(model,storage);path=Path(path);assert not path.exists()
    state=storage.state_cpu(model.core);assert storage.digest(state)==value['digest']
    buffers={n:b.detach().cpu() for n,b in model.core.named_buffers(remove_duplicate=False)}
    # All generation public values are preserved. HF's ordinary loading path
    # restores this hash sentinel, which to_dict intentionally omits.
    for cfg in [model.core.generation_config,model.core.language_model.generation_config]:
        assert hasattr(cfg,'_original_object_hash') and cfg._original_object_hash==hash(cfg)
    package=dict(format=FORMAT,state=state,buffers=buffers,base_config=model.core.config.to_dict(),
        student_config=student_config,interface_config=value['interface'],receipt=value,
        generation_config=model.core.generation_config.to_dict(),
        language_generation_config=model.core.language_model.generation_config.to_dict())
    torch.save(package,path);del package,state,buffers;gc.collect()
    return dict(**value,path=str(path),bytes=path.stat().st_size,sha256=sha(path),complete_graph=True,
        original_large_vision_or_teacher_required=False,no_unfolded_W_or_alignment_factors=True)

def _transfer(items,device):
    """Preserve exact tensor-storage aliases during ordinary checkpoint I/O."""
    bank={};result={}
    for n,t in items.items():
        key=(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),str(t.dtype))
        if key not in bank:bank[key]=t.to(device=device)
        result[n]=bank[key]
    return result

def _generation(values):
    from transformers import GenerationConfig
    cfg=GenerationConfig.from_dict(values)
    # Identical to Transformers4.46.3 GenerationConfig.from_pretrained and
    # from_model_config; no public parameter or _from_model_config is changed.
    cfg._original_object_hash=hash(cfg)
    assert cfg.to_dict()==values
    return cfg

def load_complete(path,device,storage):
    import torch
    from transformers import LlavaOnevisionConfig,LlavaOnevisionForConditionalGeneration
    from transformers.modeling_utils import no_init_weights
    import onevision_bridge as bridge
    import interface
    device=torch.device(device);assert device.type=='cuda'
    package=torch.load(path,map_location='cpu',weights_only=True);assert package['format']==FORMAT
    cfg=LlavaOnevisionConfig.from_dict(package['base_config'])
    cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
    with torch.device('meta'),no_init_weights():
        core=LlavaOnevisionForConditionalGeneration(cfg)
        model=bridge.OneVisionBridge(core)
        tower=interface.build_student(package['student_config'],'meta')
        part=interface.compile_interface(tower,torch.empty((896,12288),dtype=torch.float16,device='meta'),torch.empty(896,dtype=torch.float16,device='meta'))
        interface.install(model,part)
    core.load_state_dict(_transfer(package['state'],device),strict=True,assign=True)
    assert set(package['buffers'])=={n for n,_ in core.named_buffers(remove_duplicate=False)}
    for n,b in _transfer(package['buffers'],device).items():
        parent,leaf=n.rsplit('.',1);core.get_submodule(parent)._buffers[leaf]=b
    for module in core.modules():
        if hasattr(module,'original_inv_freq'):
            assert getattr(module,'rope_type','default')=='default'
            module.original_inv_freq=module.inv_freq
    core.language_model.tie_weights()
    core.generation_config=_generation(package['generation_config'])
    core.language_model.generation_config=_generation(package['language_generation_config'])
    # Rebind registered bridge aliases after assign=True, including newline.
    interface.install(model,core.vision_tower)
    model.requires_grad_(False).eval();actual=receipt(model,storage)
    for key in ['layout','digest','all_buffers_digest','parameter_alias_groups','functional_config_digest',
                'generation_config_digest','language_generation_config_digest','interface','language_attention_classes']:
        assert actual[key]==package['receipt'][key],key
    assert all(p.device==device for p in core.parameters())
    del package;gc.collect()
    return model,dict(actual,self_contained_meta_construction=True,all_buffers_explicitly_restored=True,strict_complete_reload=True)

def finish(model,before,storage,compact=True):
    after=receipt(model,storage,compact)
    assert all(before[key]==value for key,value in after.items())
    return dict(**after,complete_state_and_buffers_unchanged=True)
