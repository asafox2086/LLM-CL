"""Isolate a pre-TRAIN SVD reconstruction failure; no model/QA/optimizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    import torch
    from safetensors import safe_open
    import onevision_osft_budget_core_v2 as core
    import onevision_research_device_entry as device
    root = core.ROOT
    out = root/'summary/onevision_diagnostics_20261016/osft_budget_20261018/v2/GPU_FAILURE_PROBE.json'
    assert not out.exists() and not torch.cuda.is_initialized()
    cards = device.inventory()
    assert device.select(cards, os.getuid()) == gpu
    kernels, kernel_sha = core.author_kernels()
    model = root/'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors'
    torch.set_num_threads(4)
    initial = torch.backends.cuda.matmul.allow_tf32
    result = dict(CPU_or_GPU_QA=0, optimizer_updates=0, physical_GPU=gpu, inventory=cards,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  core_sha256=core.sha(root/'analysis/onevision_osft_budget_core_v2.py'),
                  kernel_sha256=kernel_sha, initial_allow_tf32=initial, cases=[])
    with safe_open(model, framework='pt', device='cpu') as reader:
        names = [n for n in reader.keys() if n.startswith('language_model.model.layers.')
                 and ('.layers.0.' in n or '.layers.23.' in n)
                 and n.endswith(('.q_proj.weight', '.v_proj.weight'))]
        assert len(names) == 4
        for name in sorted(names):
            cpu = reader.get_tensor(name)
            weight = cpu.cuda()
            row = dict(name=name, shape=list(weight.shape), source_dtype=str(cpu.dtype),
                       tensor_sha256=hashlib.sha256(cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(), results=[])
            for where in ['cuda', 'cpu']:
                torch.backends.cuda.matmul.allow_tf32 = False
                state = kernels['create_svd_dict'](weight if where=='cuda' else cpu, min(weight.shape)-8,
                    upcast_dtype=torch.float32, output_dtype=torch.float32)
                state = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in state.items()}
                for tf32 in [False, True]:
                    torch.backends.cuda.matmul.allow_tf32 = tf32
                    with torch.no_grad(), torch.autocast('cuda', enabled=False):
                        rebuilt = kernels['reconstruct_weight_matrix'](state, torch.float32, torch.float32)
                        error = float((rebuilt-weight.float()).norm()/weight.float().norm())
                    row['results'].append(dict(SVD_device=where, allow_tf32=tf32,
                        FP32_relative_error=error, passes_original_bound=error<2e-5))
                del state
            result['cases'].append(row)
            print(json.dumps(row), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = initial
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    main(p.parse_args().gpu)
