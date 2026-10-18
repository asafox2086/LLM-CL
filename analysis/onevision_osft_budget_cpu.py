"""Actual frozen q/v weight preflight; no QA, GPU, trainer or queue admission."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors import safe_open

import onevision_osft_budget_core as core

ROOT = core.ROOT
REPORT = ROOT/'summary/onevision_diagnostics_20261016/osft_budget_20261018/v1'


def main():
    assert not torch.cuda.is_initialized()
    original_env = os.environ.get('CUDA_VISIBLE_DEVICES')
    torch.set_num_threads(6)
    synthetic = core.checks()
    config_path = ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json'
    config = json.loads(config_path.read_text())['text_config']
    qv = []
    path = config_path.with_name('model.safetensors')
    examples = []
    with safe_open(path, framework='pt', device='cpu') as reader:
        names = sorted(name for name in reader.keys()
                       if name.startswith('language_model.model.layers.')
                       and name.endswith(('.self_attn.q_proj.weight', '.self_attn.v_proj.weight')))
        assert len(names) == 2*config['num_hidden_layers'] == 48
        for name in names:
            out, inp = reader.get_slice(name).get_shape()
            rank, dim = 8, min(out, inp)
            qv.append(dict(name=name, shape=[out, inp], trainable_parameters=rank*(out+inp),
                           frozen_S_parameters=rank, full_SVD_tensor_bytes=4*dim*(out+inp+1),
                           original_BF16_weight_bytes=2*out*inp))
        chosen = [name for name in names if '.layers.0.' in name or '.layers.23.' in name]
        assert len(chosen) == 4
        for name in chosen:
            original = reader.get_tensor(name)
            tensor_sha = hashlib.sha256(original.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            model = core.BudgetSVDLinear(original, rank=8)
            reconstructed = model.current_weight(torch.float32).detach()
            relative_error = float((reconstructed-original.float()).norm()/original.float().norm())
            assert relative_error < 2e-5
            examples.append(dict(name=name, source_tensor_sha256=tensor_sha,
                shape=list(original.shape), relative_FP32_SVD_error=relative_error,
                exact_BF16_reconstructed_coordinates=int(torch.count_nonzero(reconstructed.to(torch.bfloat16) == original)),
                total_coordinates=original.numel(), actual_tensor_bytes=model.bytes(),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    trainable = sum(row['trainable_parameters'] for row in qv)
    assert trainable == 540672
    for gpu in ['0', '1']:
        program = ("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']="+repr(gpu)+"; "
                   "sys.path.insert(0,"+repr(str(ROOT/'analysis'))+"); "
                   "import onevision_osft_budget_core as c; "
                   "c.author_kernels(); assert os.environ['CUDA_VISIBLE_DEVICES']=="+repr(gpu)+"; "
                   "assert not c.torch.cuda.is_initialized()")
        subprocess.run([sys.executable, '-c', program], check=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == original_env and not torch.cuda.is_initialized()
    result = dict(CPU_only=True, source_sha256={str(p.relative_to(ROOT)): core.sha(p) for p in
        [Path(__file__), ROOT/'analysis/onevision_osft_budget_core.py', config_path]},
        author_kernel_sha256=synthetic['author_kernel_sha256'], synthetic_mechanics=synthetic,
        actual_pretrained_weight_examples=examples, all48_shapes=qv,
        trainable_parameters=trainable, original_LoRA_trainable_parameters=540672,
        S_low_frozen_to_match_count=True, full_qv_SVD_tensor_bytes=sum(x['full_SVD_tensor_bytes'] for x in qv),
        original_qv_BF16_tensor_bytes=sum(x['original_BF16_weight_bytes'] for x in qv),
        current_function_refactor=True, import_device_0_and_1_preserved=True,
        QA_or_TEST_parsed=False, GPU_started=False, worker_or_queue_admitted=False,
        limits=['Capacity-matched control, not a new method or result on actual QA.',
                'Same trainable coordinate count does not mean same function family or independent degrees of freedom.',
                'Frozen S, fixed rank8/qv, and dropout/precision differences must be explicit in later protocol.',
                'SVD factors cost more frozen storage than the original dense q/v weights; no memory superiority.',
                'FP32 SVD approximates weights; BF16 reconstruction is not assumed bitwise original. Own S0 required.',
                'Protected/unprotected paired controls must start from EXACT same saved factors and function.',
                'No new model optimizer, visual forward or generative metric has yet been tested.'])
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'CPU_MECHANICS.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['trainable_parameters', 'full_qv_SVD_tensor_bytes',
        'original_qv_BF16_tensor_bytes', 'import_device_0_and_1_preserved', 'GPU_started', 'worker_or_queue_admitted']}))


if __name__ == '__main__':
    main()
