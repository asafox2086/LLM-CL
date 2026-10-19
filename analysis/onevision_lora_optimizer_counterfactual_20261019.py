"""CPU-only author-optimizer counterfactual on sealed real TRAIN gradients.

This is an optimization-mechanism check, not a new forward, F1 result, or
explanation of every performance gap. Published LoRA-RITE is not our novelty.
"""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'summary/onevision_diagnostics_20261016'
REPORT = BASE / 'optimizer_geometry_20261019/v1'
UPSTREAM_SHA = '4143a782af8131a09ea281a35461ebfdec3f8c6dcd8961bbee5590882057b8aa'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def author_class():
    path = ROOT / 'code/LoRA_RITE/source/lora_rite.py'
    assert sha(path) == UPSTREAM_SHA
    spec = importlib.util.spec_from_file_location('llmcl_pinned_lora_rite', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LoRARite


def tensor_bytes(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    if hasattr(value, '__dict__'):
        return tensor_bytes(vars(value))
    return 0


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    before_env = os.environ.get('CUDA_VISIBLE_DEVICES')
    Rite = author_class()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == before_env
    assert not torch.cuda.is_initialized()
    REPORT.mkdir(parents=True, exist_ok=True)
    assert not (REPORT / 'EVIDENCE.json').exists(), 'Do not repeat the sealed audit'
    plan_path = BASE / 'two_module_prototype/CPU_PILOT_PLAN.json'
    plan = json.loads(plan_path.read_text())['spec']
    anchor_path = ROOT / plan['past_natural_checkpoint']
    assert sha(anchor_path) == plan['past_natural_sha256']
    anchor = torch.load(anchor_path, map_location='cpu', weights_only=True)['adapter']
    source_shas = {str(anchor_path.relative_to(ROOT)): sha(anchor_path),
                   str(plan_path.relative_to(ROOT)): sha(plan_path),
                   'analysis/' + Path(__file__).name: sha(__file__),
                   'code/LoRA_RITE/source/lora_rite.py': UPSTREAM_SHA}
    output = []
    start = time.monotonic()
    for task in (2, 11, 12):
        proof_path = BASE / f'two_module_prototype/T{task:02d}_control_complete.json'
        proof = json.loads(proof_path.read_text())
        assert proof['complete'] and proof['valid_training'] and proof['spec'] == plan
        raw_path = ROOT / proof['run'] / 'raw_gradient.pt'
        assert sha(raw_path) == proof['artifact_sha256']['raw_gradient.pt']
        source_shas[str(raw_path.relative_to(ROOT))] = sha(raw_path)
        source_shas[str(proof_path.relative_to(ROOT))] = sha(proof_path)
        raw = torch.load(raw_path, map_location='cpu', weights_only=True)
        assert set(raw['TRAIN_ids']) == set(plan['TRAIN_ids'][str(task)])
        gradients = {}
        offset = 0
        for name, shape in raw['layout']:
            size = math.prod(shape)
            gradients[name] = raw['gradient'][offset:offset + size].reshape(shape).clone()
            offset += size
        assert offset == raw['gradient'].numel() == 540672
        reference_updates = {}
        for optimizer_name in ('AdamW', 'author_LoRARite'):
            variants = []
            # Reference first, then both fixed scale interventions; no selection.
            for scale in (1., .25, 4.):
                pairs, parameters = {}, []
                for name, item in anchor.items():
                    a = torch.nn.Parameter(item['A'].clone() * scale)
                    b = torch.nn.Parameter(item['B'].clone() / scale)
                    assert torch.equal(b @ a, item['B'] @ item['A'])
                    a.grad = gradients[name + '.A.weight'].clone() / scale
                    b.grad = gradients[name + '.B.weight'].clone() * scale
                    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
                    pairs[name] = (a, b)
                    parameters.extend((a, b))
                assert sum(p.numel() for p in parameters) == 540672
                if optimizer_name == 'AdamW':
                    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
                    raw_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
                else:
                    # README explicitly requires clipping inside the optimizer.
                    optimizer = Rite(parameters, lr=1e-4, betas=(.9, .999),
                                     clip_unmagnified_grad=1.)
                    raw_norm = math.sqrt(sum(float(p.grad.double().square().sum()) for p in parameters))
                skips = []
                if optimizer_name == 'author_LoRARite':
                    original_skip = optimizer.helper.skip_update
                    def skip(update, threshold):
                        value = original_skip(update, threshold)
                        skips.append(bool(torch.count_nonzero(update) and not torch.count_nonzero(value)))
                        return value
                    optimizer.helper.skip_update = skip
                with torch.no_grad():
                    optimizer.step()
                updates, changed_a, changed_b = {}, 0, 0
                for name, (a, b) in pairs.items():
                    assert torch.isfinite(a).all() and torch.isfinite(b).all()
                    updates[name] = 4 * (b.detach().double() @ a.detach().double()
                                        - anchor[name]['B'].double() @ anchor[name]['A'].double())
                    changed_a += int(not torch.equal(a, anchor[name]['A'] * scale))
                    changed_b += int(not torch.equal(b, anchor[name]['B'] / scale))
                delta = math.sqrt(sum(float(v.square().sum()) for v in updates.values()))
                assert delta > 0 and math.isfinite(delta)
                if scale == 1.:
                    reference_updates[optimizer_name] = updates
                reference = reference_updates[optimizer_name]
                energy = sum(float(v.square().sum()) for v in reference.values())
                dot = sum(float((updates[n] * reference[n]).sum()) for n in updates)
                difference = math.sqrt(sum(float((updates[n] - reference[n]).square().sum()) for n in updates))
                variants.append(dict(factor_scale_A=scale, factor_scale_B=1 / scale,
                    initial_BA_bit_exact=True, raw_factor_gradient_norm=raw_norm,
                    actual_deltaW_norm=delta, relative_deltaW_change_vs_same_optimizer_scale1=difference / math.sqrt(energy),
                    cosine_to_same_optimizer_scale1=dot / (delta * math.sqrt(energy)),
                    A_layers_changed=changed_a, B_layers_changed=changed_b,
                    author_update_skip_threshold=getattr(optimizer, 'update_skipping', None),
                    author_factor_updates_skipped=sum(skips), persistent_optimizer_tensor_bytes=tensor_bytes(optimizer.state)))
            output.append(dict(task=task, TRAIN_QA=len(raw['TRAIN_ids']), optimizer=optimizer_name, variants=variants))
    result = dict(status='CPU_ACTUAL_TRAIN_GRADIENT_COUNTERFACTUAL_COMPLETE', sources_sha256=source_shas,
        published_optimizer='LoRA-RITE ICLR2025 author PyTorch reimplementation',
        author_source_unchanged=True, results=output, seconds=time.monotonic() - start,
        new_model_forwards=0, GPU_used=False, TEST_predictions_or_scores_read=False,
        same_BA_not_claim_of_native_GPU_forward_bit_identity=True,
        frozen_counterfactual_gradient_not_new_finite_TRAIN_or_TEST_trajectory=True,
        no_F1_causality_or_improvement_claim=True)
    assert not torch.cuda.is_initialized()
    (REPORT / 'EVIDENCE.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
