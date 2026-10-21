"""Read sealed PaRSP/SpaRTA records and existing PaRSP tensors; no model forward."""
import argparse
import ast
import base64
import collections
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import resource
import sys
import time
import traceback


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def controls(torch):
    u = torch.arange(1, 5, dtype=torch.float64)
    u = u / u.norm()
    P = u[:, None] @ u[None, :]
    C = torch.eye(4, dtype=torch.float64) - P
    G = torch.arange(1, 17, dtype=torch.float64).reshape(4, 4)
    mask = torch.tensor([1., 1., 0., 0.], dtype=torch.float64)
    gp = (G * mask) @ P
    pg = (G @ P) * mask
    w = torch.zeros(4, 4, dtype=torch.float64, requires_grad=True)
    w.register_hook(lambda g: g * mask)
    w.register_hook(lambda g: g @ P)
    (w * G).sum().backward()
    assert torch.equal(w.grad, gp)
    assert (gp[:, 2:] != 0).all() and (pg[:, 2:] == 0).all()
    assert torch.linalg.norm(gp @ C) < 1e-12
    assert torch.linalg.norm(pg @ C) > 0.1
    row_gp = (G * mask[:, None]) @ P
    assert (row_gp[2:] == 0).all()
    opt = torch.optim.AdamW([w], lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
    before = w.detach().clone()
    opt.step()
    adam_delta = w.detach() - before
    analytic = -.001 * gp / (gp.abs() + 1e-8)
    assert torch.allclose(adam_delta, analytic, atol=1e-17, rtol=1e-12)
    assert torch.linalg.norm(adam_delta @ C) > 1e-5
    return {
        'fixed_u': [1, 2, 3, 4], 'mask_columns': [0, 1],
        'mask_then_projection_outside_energy_fraction': float(gp[:, 2:].square().sum() / gp.square().sum()),
        'projection_then_mask_nullspace_residual_l2': float(torch.linalg.norm(pg @ C)),
        'mask_then_projection_nullspace_residual_l2': float(torch.linalg.norm(gp @ C)),
        'row_mask_zero_rows_retained': True, 'actual_hook_matches_manual_exact': True,
        'Adam_first_step_matches_closed_form': True,
        'SGD_nullspace_residual_l2': float(torch.linalg.norm((-.001 * gp) @ C)),
        'Adam_nullspace_residual_l2': float(torch.linalg.norm(adam_delta @ C)),
        'toy_optimizer_steps': 1, 'model_optimizer_steps': 0,
        'scope': 'fixed algebraic counterexample; not the actual formal gradients or F1 causality',
    }


def analyze(root, inputs, report):
    started = time.monotonic()
    check_time = lambda: time.monotonic() - started < 900
    lock = json.loads(inputs.read_text())
    assert sha(root / lock['collector_path']) == lock['collector_sha256']
    spec = importlib.util.spec_from_file_location('original_sealed_collector', root / lock['collector_path'])
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    scores = collector.collect(root, inputs)
    scores['scope'] = 'posthoc description of two additional sealed runs; not intervention or deployment proof'
    assert set(scores['methods']) == {'PaRSP', 'SpaRTA'}
    for method, info in lock['methods'].items():
        proof = json.loads(base64.b64decode(info['proof_b64']))
        run = root / proof['candidate']['run']
        snapshot = run / 'runner_source'
        for name, expected in lock['source_sha256'].items():
            assert sha(root / name) == expected, name
        # Verify the actual archived runner copies rather than relying on current code alone.
        for name in ['parsp_adapter.py', 'sparta_adapter.py', 'run.py', 'runtime_utils.py', 'host_adam.py']:
            expected = lock['source_sha256']['exp/two_backbone_native_cl_20261017_v2/' + name]
            assert sha(snapshot / name) == expected, (method, name)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    control = controls(torch)
    proof = json.loads(base64.b64decode(lock['methods']['PaRSP']['proof_b64']))
    run = root / proof['candidate']['run']
    stages, auxiliary = [], {}
    for stage in range(1, 20):
        path = run / 'auxiliary' / f'parsp_stage_{stage:02d}.json'
        record = json.loads(path.read_text())
        assert record['stage'] == stage and not record['preflight_only']
        assert record['layer_ratio'] == .38 and record['neuron_ratio'] == .4
        stages.append(record)
        auxiliary[str(path.relative_to(root))] = sha(path)
    assert all(s['historical_qa_replayed'] == 0 for s in stages)
    checkpoint = run / 'checkpoints/stage_19.pt'
    assert sha(checkpoint) == proof['checkpoint_sha256']
    base = root / 'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors'
    parent_evidence = root / 'summary/edge_deployment_20261019/cpu_update_space_20261020/v1/EVIDENCE.json'
    base_expected = json.loads(parent_evidence.read_text())['sources']['base_sha256']
    assert sha(base) == base_expected
    assert checkpoint.stat().st_size < 16 * 1024**3 and check_time()
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    assert available > max(4 * 1024**3, checkpoint.stat().st_size * 2)
    sys.path.insert(0, str(run / 'runner_source'))
    payload = torch.load(checkpoint, map_location='cpu', pickle_module=pickle, weights_only=False, mmap=True)
    assert payload['stage'] == 19 and payload['config']['method'] == 'PaRSP'
    model = payload['method'].model.core
    from safetensors import safe_open
    rows = []
    with safe_open(base, framework='pt', device='cpu') as reader, torch.inference_mode():
        for name, final in model.named_parameters():
            if not name.startswith('language_model.model.layers.') or not name.endswith('.weight'):
                continue
            pieces = name.split('.')
            if pieces[-2] not in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                continue
            assert check_time()
            layer = int(pieces[3])
            kind = pieces[-2]
            shape = reader.get_slice(name).get_shape()
            assert list(final.shape) == shape and final.ndim == 2 and final.is_floating_point()
            axis = 0 if kind in ['v_proj', 'gate_proj', 'up_proj'] else 1
            union = torch.zeros(shape[axis], dtype=torch.bool)
            selected_stages = no_mask_stages = 0
            for stage in stages:
                ffn = kind in ['gate_proj', 'up_proj', 'down_proj']
                selected = layer in stage['ffn_layers' if ffn else 'attention_layers']
                if not selected:
                    continue
                selected_stages += 1
                entries = stage['ffn_neurons' if ffn else 'attention_neurons']
                valid = sorted({int(n) for l, n in entries if int(l) == layer and 0 <= int(n) < shape[axis]})
                if kind in ['q_proj', 'k_proj'] or not valid:
                    union[:] = True
                    no_mask_stages += 1
                else:
                    union[valid] = True
            original = reader.get_tensor(name)
            delta = final.detach().double() - original.double()
            outside = delta[~union] if axis == 0 else delta[:, ~union]
            energy = float(delta.square().sum())
            outside_energy = float(outside.square().sum())
            rows.append({
                'name': name, 'shape': shape, 'base_dtype': str(original.dtype), 'final_dtype': str(final.dtype),
                'selected_stages': selected_stages, 'selected_stages_without_neuron_mask': no_mask_stages,
                'nominal_mask_axis': axis, 'nominal_union_axis_count': int(union.sum()),
                'nominal_union_coordinate_count': int(union.sum()) * shape[1-axis],
                'coordinates': delta.numel(), 'changed_coordinates': int((delta != 0).sum()),
                'outside_nominal_union_coordinates': outside.numel(),
                'outside_nominal_union_changed': int((outside != 0).sum()),
                'delta_squared_norm': energy, 'outside_nominal_union_squared_norm': outside_energy,
            })
    assert len(rows) == 168 and not torch.cuda.is_initialized()
    by_kind = {}
    for kind in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        subset = [x for x in rows if x['name'].endswith('.' + kind + '.weight')]
        by_kind[kind] = {key: sum(x[key] for x in subset) for key in [
            'coordinates', 'nominal_union_coordinate_count', 'changed_coordinates',
            'outside_nominal_union_coordinates', 'outside_nominal_union_changed',
            'delta_squared_norm', 'outside_nominal_union_squared_norm']}
    stdout = (run / 'supervisor.log').read_text(errors='replace')
    projection_lines = [line for line in stdout.splitlines() if any(tag in line for tag in ['[Force]', '[Auto]', 'nullspace', 'Null-space'])]
    # Count only literal diagnostics; do not interpret unmatched log text as zero triggers.
    diagnostics = {'supervisor_sha256': sha(run / 'supervisor.log'),
                   'literal_Force_lines': sum('[Force]' in x for x in projection_lines),
                   'literal_Auto_lines': sum('[Auto]' in x for x in projection_lines),
                   'scope': 'literal existing log diagnostics, not a complete projector classification'}
    result = {
        'source_sha256': sha(Path(__file__)), 'inputs_sha256': sha(inputs),
        'original_collector_sha256': lock['collector_sha256'], 'scores': scores,
        'matrix_control': control, 'PaRSP_final_support': {'matrices': rows, 'by_kind': by_kind,
            'checkpoint_sha256': proof['checkpoint_sha256'], 'base_sha256': base_expected,
            'auxiliary_sha256': auxiliary, 'projection_log_diagnostics': diagnostics,
            'scope': 'actual final weight differences versus union of pre-projection nominal masks; not stepwise gradients or F1 attribution'},
        'SpaRTA_recovery': {'formal_main_updates': scores['methods']['SpaRTA']['positive_finite_main_updates'],
            'period': 200, 'global_counter_initial': 0, 'global_counter_resets_at_stage': False,
            'formal_restore_trigger_count': scores['methods']['SpaRTA']['positive_finite_main_updates'] // 200,
            'preflight_calls_after_update': False, 'source_execution_chain_verified': True},
        'new_model_forwards': 0, 'new_model_optimizer_updates': 0, 'toy_optimizer_steps': 1,
        'one_existing_PaRSP_graph_deserialized_CPU_only': True, 'new_GPU_calls': 0,
        'seconds': time.monotonic() - started,
        'analysis_max_RSS_KiB_not_deployment': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'TEST_selection': False, 'goal_achieved': False,
    }
    write(report / 'EVIDENCE.json', result)
    write(report / 'COMPLETE.json', {'status': 'COMPLETE', 'source_sha256': sha(Path(__file__)),
        'inputs_sha256': sha(inputs), 'evidence_sha256': sha(report / 'EVIDENCE.json'),
        'matrices': len(rows), 'new_model_forwards': 0, 'new_model_updates': 0, 'toy_optimizer_steps': 1})
    print(json.dumps({'complete': True, 'seconds': result['seconds'], 'by_kind': by_kind,
        'toy_control': control, 'SpaRTA_restore': result['SpaRTA_recovery']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (args.report / 'EVIDENCE.json').exists() and not (args.report / 'FAILURE.json').exists()
    try:
        analyze(args.root, args.inputs, args.report)
    except Exception:
        write(args.report / 'FAILURE.json', {'source_sha256': sha(Path(__file__)), 'traceback': traceback.format_exc(),
              'no_blind_retry': True, 'original_training_and_predictions_unchanged': True})
        raise
