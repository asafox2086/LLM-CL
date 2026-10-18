"""CPU-only gradient-path control of the unchanged authored SMoLoRA forward.

Synthetic 4D inputs; no backbone, TRAIN/TEST QA, optimizer, or source mutation.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]


def build():
    import torch
    from torch import nn
    from torch.nn import functional as F
    sys.path.insert(0, str(ROOT/'exp/two_backbone_native_cl_20261017_v2'))
    from methods import AuthorSMoLinear, author_components
    components, revision, upstream_sha = author_components()
    torch.manual_seed(42)
    layer = AuthorSMoLinear(nn.Linear(4, 4, bias=False),
                           torch.randn(19, 384).tolist(), components)
    layer.weight.requires_grad_(False)
    # Remove zero-B initialization as a competing explanation for absent grads.
    for expert in layer.lora_B['default'].loraB:
        nn.init.normal_(expert.mlp.weight, mean=0, std=.1)
    before = {n: p.detach().clone() for n, p in layer.named_parameters()}
    x = torch.randn(1, 3, 4)
    output = layer(x)
    route = layer.final_router.detach().clone()
    loss = F.cross_entropy(output.reshape(-1, 4), torch.tensor([0, 1, 2]))
    loss.backward()
    gates = {}
    for name, parameter in layer.named_parameters():
        if any(key in name for key in ['lora_vu_gate', 'lora_ins_gate', 'lora_fc_A', 'lora_fc_B']):
            gates[name] = {'requires_grad': parameter.requires_grad,
                           'gradient_is_None': parameter.grad is None,
                           'gradient_norm': None if parameter.grad is None else float(parameter.grad.norm())}
    for key in ['lora_vu_gate.default.weight', 'lora_ins_gate.default.weight']:
        assert gates[key]['requires_grad'] and gates[key]['gradient_is_None']
    for key in ['lora_fc_A.default.weight', 'lora_fc_B.default.weight']:
        assert gates[key]['gradient_norm'] > 0
    expert_norms = [float(e.mlp.weight.grad.norm()) for e in layer.lora_B['default'].loraB]
    selected = [i for i, value in enumerate(route.flatten().tolist()) if value]
    assert all(expert_norms[i] > 0 if i in selected else expert_norms[i] == 0 for i in range(8))
    with torch.no_grad():
        layer(-x)
    second_route = layer.final_router.detach().clone()
    assert not torch.equal(route[..., :4], second_route[..., :4])
    assert torch.equal(route[..., 4:], second_route[..., 4:])
    assert all(torch.equal(before[n], p) for n, p in layer.named_parameters())
    path = ROOT/'code/SMoLoRA/source/PEFT_SMoLoRA/peft/tuners/smolora.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SMoLoraLinear')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    adapter = ROOT/'exp/two_backbone_native_cl_20261017_v2/methods.py'
    return {'kind': 'synthetic CPU authored-operator gradient-path evidence, NOT GPU performance',
            'seed': 42, 'torch_version': torch.__version__, 'author_revision': revision,
            'upstream_sha256': upstream_sha, 'adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(),
            'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'input_shape': list(x.shape), 'target': [0, 1, 2], 'loss': float(loss.detach()),
            'experts': 8, 'B_precondition': 'all eight B nonzero before forward; std .1',
            'gates': gates, 'selected_experts': selected, 'expert_B_gradient_norms': expert_norms,
            'route_x': route.tolist(), 'route_negative_x': second_route.tolist(),
            'parameters_unchanged': True, 'GPU_backbone_loads': 0,
            'QA_read': 0, 'optimizer_updates': 0,
            'author_forward_lines': [forward.lineno, forward.end_lineno],
            'author_forward': ast.get_source_segment(path.read_text(), forward),
            'limits': ['No full author trainer or other training objective was executed.',
                       'Not an actual OneVision task gradient or final SMoLoRA result.',
                       'Fixed gate parameters do not imply fixed input-dependent routing.',
                       'Fusion and selected experts still receive loss gradients.',
                       'No causal evidence of final F1 harm or benefit from changing the gates.',
                       'Does not apply to softmax routers or the separately learned TRAIN metric.',
                       'No active/queued trainer, parameters, source, or schedule changed.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = build()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError('Use a new evidence path; do not overwrite prior controls.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(target), 'gates': result['gates'],
                      'selected_experts': result['selected_experts'], 'GPU_performance': False}))
