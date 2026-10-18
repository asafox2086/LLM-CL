"""Read-only real T02 cold/warm gradient and parameter activation comparison.

No TEST predictions, model loading, forward, optimizer or parameter changes.
Loads only small CPU LoRA/raw-gradient snapshots already saved by the runs.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'summary/onevision_diagnostics_20261016'
REPORT = BASE/'expert_cold_start_20261018/first_factor_gradient'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    start = time.monotonic()
    cold_path = BASE/'learned_reuse/learned_cosine_K3_train_complete.json'
    proof = json.loads(cold_path.read_text())
    cold = ROOT/proof['run']
    warm_attempt = BASE/'warm_start_20261018/warm_inherit_K3_v1_train_attempt.json'
    attempt = json.loads(warm_attempt.read_text())
    warm = ROOT/attempt['run']
    spec = attempt['spec']
    assert spec['cold_parent_completion_sha256'] == sha(cold_path)
    native_path = BASE/'deterministic_native/complete.json'
    assert proof['spec']['native_completion_sha256'] == sha(native_path)
    native_proof = json.loads(native_path.read_text())
    native = ROOT/native_proof['run']
    sources = {}

    def cold_file(relative):
        p = cold/relative
        assert sha(p) == proof['artifact_sha256'][relative]
        sources[str(p.relative_to(ROOT))] = sha(p)
        return p

    def warm_file(relative):
        p = warm/relative
        meta = json.loads(p.with_suffix('.json').read_text())
        assert meta['sha256'] == sha(p) and p.stat().st_size == meta['bytes']
        sources[str(p.relative_to(ROOT))] = sha(p)
        return p

    initial = native/'initial_adapter.pt'
    assert sha(initial) == native_proof['artifact_sha256']['initial_adapter.pt']
    sources[str(initial.relative_to(ROOT))] = sha(initial)
    base_adapter = torch.load(initial, map_location='cpu', weights_only=True)['adapter']
    previous = torch.load(warm_file('states/reuse_T01.pt'), map_location='cpu', weights_only=True)['pool']['e0']
    cold_trained = torch.load(cold_file('candidates/reuse_T02.pt'), map_location='cpu', weights_only=True)['adapter']
    warm_trained = torch.load(warm_file('candidates/reuse_T02.pt'), map_location='cpu', weights_only=True)['adapter']
    old_gradient = torch.load(cold_file('raw_update_gradients/reuse_T02/step1.pt'), map_location='cpu', weights_only=True)
    new_gradient = torch.load(warm_file('raw_update_gradients/reuse_T02/step1.pt'), map_location='cpu', weights_only=True)
    assert old_gradient['train_ids'] == new_gradient['train_ids'] == spec['train_ids']['2']
    assert len(old_gradient['train_ids']) == 11 and old_gradient['layout'] == new_gradient['layout']
    gate_path = warm/'initialization_gates/T02.json'
    gate = json.loads(gate_path.read_text())
    assert gate['passed'] and gate['source'] == 'e0' and gate['mode'] == 'inherit_existing_complete_AB'
    module_path = ROOT/'analysis/onevision_warm_start_trial.py'
    assert sha(module_path) == spec['source_sha256']['analysis/onevision_warm_start_trial.py']
    definition = next(n for n in ast.parse(module_path.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == 'adapter_hash')
    namespace = {'hashlib': hashlib, 'json': json}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(module_path), 'exec'), namespace)
    assert namespace['adapter_hash'](previous) == gate['initial_AB_sha256']
    assert base_adapter.keys() == previous.keys() == cold_trained.keys() == warm_trained.keys()
    groups = {}
    for name, payload in [('cold', old_gradient), ('warm', new_gradient)]:
        gradient = payload['preclip_gradient']
        assert torch.isfinite(gradient).all()
        position = 0
        split = {'A': [], 'B': []}
        layer_nonzero = {'A': 0, 'B': 0}
        for item in payload['layout']:
            part = gradient[position:position+item['count']]
            assert part.numel() == item['count']
            position += item['count']
            letter = 'A' if item['name'].endswith('.A.weight') else 'B'
            assert item['name'].endswith('.'+letter+'.weight')
            split[letter].append(part)
            layer_nonzero[letter] += int(torch.count_nonzero(part) > 0)
        assert position == gradient.numel() == 540672
        groups[name] = {'gradient_norm': {letter: float(torch.cat(parts).double().norm()) for letter, parts in split.items()},
                        'nonzero_gradient_elements': {letter: int(torch.count_nonzero(torch.cat(parts))) for letter, parts in split.items()},
                        'nonzero_gradient_layers': layer_nonzero}
    changes = {}
    for name, before, after in [('cold', base_adapter, cold_trained), ('warm', previous, warm_trained)]:
        changes[name] = {letter: {'changed_layers': sum(not torch.equal(before[k][letter], after[k][letter]) for k in before),
            'delta_norm': sum(float((after[k][letter]-before[k][letter]).double().square().sum()) for k in before)**.5}
            for letter in ['A', 'B']}
    assert all(torch.count_nonzero(base_adapter[k]['B']) == 0 for k in base_adapter)
    assert groups['cold']['nonzero_gradient_elements']['A'] == 0 and changes['cold']['A']['changed_layers'] == 0
    assert groups['cold']['gradient_norm']['B'] > 0 and changes['cold']['B']['changed_layers'] > 0
    assert groups['warm']['gradient_norm']['A'] > 0 and changes['warm']['A']['changed_layers'] > 0
    result = {'CPU_only': True, 'actual_saved_model_gradients': True, 'TEST_predictions_read': 0,
        'model_forward': 0, 'optimizer_updates_executed_by_audit': 0, 'QA_modified': False,
        'cold_run': str(cold.relative_to(ROOT)), 'warm_run': str(warm.relative_to(ROOT)),
        'target_task': 2, 'same_TRAIN_QA': 11, 'same_main_updates': 1,
        'common_first_task_native_AB_exact': json.loads((warm/'first_task_exact_gate.json').read_text())['passed'],
        'warm_initial_AB_matches_actual_prior_expert': True, 'layers': len(base_adapter),
        'gradients': groups, 'actual_parameter_changes': changes, 'source_sha256': sources,
        'first_step_formula': 'dL/dA = B^T(dL/d(deltaW)); B=0 implies dL/dA=0 under this LoRA branch',
        'limitations': ['Known LoRA initialization/chain-rule property, not new algebra.',
            'Fresh per-stage AdamW and zero weight decay here; do not generalize to nonzeroB/optimizer histories/other parameterizations.',
            'Only a fresh new expert with one update fails to train A; reused experts can have nonzeroB.',
            'Warm changes starting function and available A gradient together; this does not separate their F1 contribution.',
            'No new Answer F1 or full19/full21454 effect follows from these gradients.'],
        'elapsed_seconds': time.monotonic()-start}
    REPORT.mkdir(parents=True, exist_ok=True)
    target = REPORT/'EVIDENCE.json'
    assert not target.exists(), 'Do not repeat a completed audit as progress'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'actual_CPU_audit': True, 'gradient_norms': {k:v['gradient_norm'] for k,v in groups.items()},
                      'parameter_changes': changes, 'layers': len(base_adapter), 'F1_claim': False}))


if __name__ == '__main__':
    main()
