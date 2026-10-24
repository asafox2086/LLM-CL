"""Reuse the frozen adapter initialization; replace only the fixed producer."""
from pathlib import Path
import ast,hashlib,sys

def make_budget(model,assets_path,budget_source,plan=None):
    import study
    tree=ast.parse(Path(budget_source).read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['rank_plan','BudgetQV']]
    assert len(selected)==2
    namespace=dict(hashlib=hashlib,sys=sys,study=study,PORT=Path(assets_path)/'exp/two_backbone_native_cl_20261017_v2')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(budget_source),'exec'),namespace)
    chosen=namespace['rank_plan']() if plan is None else plan
    return namespace['BudgetQV'](model,chosen)

def install_interface_preserving_adapters(model,interface,method):
    import torch
    import vision_interface as vi
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert trainable and len(trainable)==2*len(method.layers)
    assert all(n.endswith(('.A.weight','.B.weight')) for n in trainable)
    snapshots={n:p.detach().clone() for n,p in trainable.items()}
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []
    vi.install(model,interface)
    assert not any(p.requires_grad for p in model.parameters())
    for p in trainable.values():p.requires_grad_(True)
    now={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert set(now)==set(trainable) and all(now[n] is p and torch.equal(p,snapshots[n]) for n,p in trainable.items())
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if cuda_rng else []))
    assert not any(p.requires_grad for p in model.visual.parameters())
    return dict(adapter_parameters=len(trainable),adapter_coordinates=sum(p.numel() for p in trainable.values()),identities_bytes_and_RNG_preserved=True)

def fold(model,method):
    import torch
    from torch import nn
    records=[]
    for name in method.rank_plan:
        layer=model.get_submodule(name)
        assert hasattr(layer,'A') and hasattr(layer,'B')
        with torch.no_grad():
            original_device,original_dtype=layer.weight.device,layer.weight.dtype
            # Match the existing parent's export: CPU FP32 sum/product, followed
            # by the original FP16 storage cast. Do not use CUDA/TF32 matmul.
            base=layer.weight.detach().to(device='cpu',dtype=torch.float32)
            a=layer.A.weight.detach().to(device='cpu',dtype=torch.float32)
            b=layer.B.weight.detach().to(device='cpu',dtype=torch.float32)
            weight=(base+layer.scale*(b@a)).to(original_dtype)
            linear=nn.Linear(layer.weight.shape[1],layer.weight.shape[0],bias=layer.bias is not None,device='cpu',dtype=original_dtype)
            linear.weight.copy_(weight)
            if layer.bias is not None:linear.bias.copy_(layer.bias.detach().cpu())
            linear.to(original_device)
        parent,child=name.rsplit('.',1);setattr(model.get_submodule(parent),child,linear)
        records.append(dict(module=name,shape=list(weight.shape),dtype=str(weight.dtype),rank=method.rank_plan[name],scale=layer.scale,
                            merge_device='cpu',merge_dtype='torch.float32',output_device=str(original_device)))
    method.layers.clear()
    model.requires_grad_(False);model.eval()
    assert not any('.A.' in n or '.B.' in n for n in model.core.state_dict())
    return records

def export_package(model,student_config):
    import torch
    interface=model.core.vision_tower
    assert model.visual.tower is interface and model.visual.projector is model.core.multi_modal_projector
    return dict(state={n:t.detach().cpu().contiguous() for n,t in model.core.state_dict().items()},base_config=model.core.config.to_dict(),student_config=student_config,
                interface_config=dict(output_width=interface.affine.out_features,native_size=interface.native_size,output_grid=interface.output_grid))

def load_export(package,device='cpu'):
    import torch
    from transformers import LlavaOnevisionConfig,LlavaOnevisionForConditionalGeneration
    import onevision_bridge as bridge,vision_interface as vi
    cfg=LlavaOnevisionConfig(**package['base_config']);cfg._attn_implementation='sdpa';cfg.text_config._attn_implementation='sdpa';cfg.vision_config._attn_implementation='sdpa'
    model=bridge.OneVisionBridge(LlavaOnevisionForConditionalGeneration(cfg))
    interface=vi.VisionInterface(vi.student(package['student_config']),**package['interface_config']);vi.install(model,interface)
    with torch.no_grad():
        for parameter in model.core.parameters():parameter.data=parameter.data.to(dtype=torch.float16)
    model.core.load_state_dict(package['state'],strict=True)
    model.to(device).requires_grad_(False);model.eval()
    assert model.visual.tower is model.core.vision_tower and model.visual.projector is model.core.multi_modal_projector
    assert model.core.language_model.config._attn_implementation=='sdpa'
    return model
