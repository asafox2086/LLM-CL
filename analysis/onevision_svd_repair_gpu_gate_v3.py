"""Actual BF16 q/v numerical regression gate; no QA or model training."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    import torch
    from safetensors import safe_open
    import onevision_osft_budget_core_v3 as core
    import onevision_research_device_entry as device
    root = core.ROOT
    report = root/'summary/onevision_diagnostics_20261016/osft_budget_20261018/v3'
    out = report/'GPU_REPAIR_GATE.json'
    assert not out.exists() and not torch.cuda.is_initialized()
    cards=device.inventory();assert device.select(cards,os.getuid())==gpu
    torch.set_num_threads(4)
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    rows=[]
    with safe_open(root/'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors',framework='pt',device='cpu') as f:
        names=sorted(n for n in f.keys() if n.startswith('language_model.model.layers.')
                     and n.endswith(('.q_proj.weight','.v_proj.weight')))
        assert len(names)==48
        for name in names:
            original=f.get_tensor(name).to(dtype=torch.bfloat16,device='cuda')
            layer=core.BudgetSVDLinear(original,rank=8,protected=False)
            with torch.no_grad():
                strict=layer.current_weight(torch.float32)
                error=float((strict-original.float()).norm()/original.float().norm())
                assert error<2e-5
                torch.backends.cuda.matmul.allow_tf32=True
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    under_outer=layer.current_weight(torch.float32)
                assert torch.equal(strict,under_outer) and torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32=old_tf32
            rows.append(dict(name=name,shape=list(original.shape),dtype=str(original.dtype),
                FP32_relative_error=error,outer_autocast_and_TF32_reconstruction_exact=True,
                trainable_parameters=sum(p.numel() for p in layer.parameters() if p.requires_grad)))
            del layer,original,strict,under_outer
    assert sum(x['trainable_parameters'] for x in rows)==540672
    result=dict(passed=True,all48_native_BF16_weights=True,original_bound=2e-5,
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),root/'analysis/onevision_osft_budget_core_v3.py']},
        physical_GPU=gpu,initial_inventory=cards,cases=rows,QA_forward=0,optimizer_updates=0,
        official_multimodal_model_forward_still_pending=True,
        explanation='CPU author FP32 SVD, same author factors copied to original device; strict FP32 reconstruction. No threshold/data/hyperparameter changes.')
    report.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,actual_native_BF16_weights=48,max_error=max(x['FP32_relative_error'] for x in rows),QA=0,optimizer_updates=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);main(p.parse_args().gpu)
