"""GPU1 controlled one-update pilot of two experimental modules.

Three original tiny native tasks, four fixed ablations, same old natural adapter.
This is an intervention pilot, NOT the full1313-QA/19-stage formal CL experiment.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
sys.path.insert(0, str(ROOT/'exp/onevision_diagnostics_20261016'))
from run import Experiment, freeze, hashed, write
from onevision_knowledge_intervention import Intervention, reset_seed
from onevision_research_worker import atomic_torch, digest, capture_rng, restore_rng
from onevision_capacity_probe import finish_resources
from onevision_full_input_repair import RepairedExperiment
from onevision_two_module_core import TrainLoss, accept_displacement, choose_reuse

REPORT = ROOT/'summary/onevision_diagnostics_20261016/two_module_prototype'
VARIANTS = {'control': (False, False), 'A_only': (True, False),
            'B_only': (False, True), 'A_B': (True, True)}
CASES = [(task, variant) for task in (2, 11, 12) for variant in VARIANTS]


def plan():
    manifest, data, domains, tests, generation, probes, protocol = freeze()
    reference = json.loads((ROOT/'summary/onevision_diagnostics_20261016/research_current_attempt.json').read_text())
    checkpoint = ROOT/reference['old_control_run']/'adapters/isolated_natural.pt'
    expected = reference['completed_control_sha256'][str(checkpoint.relative_to(ROOT))]
    assert digest(checkpoint) == expected
    current = {t: sorted(data[t, 'train'], key=lambda r: hashed(r, 'two-module-current'))[:8] for t in (2, 11, 12)}
    memory = sorted(data[1, 'train'], key=lambda r: hashed(r, 'two-module-old'))[:2]
    assert [len(data[t, 'train']) for t in (2, 11, 12)] == [11, 12, 6]
    sources = [Path(__file__), ROOT/'analysis/onevision_two_module_core.py',
               ROOT/'analysis/onevision_full_input_repair.py', ROOT/'analysis/onevision_capacity_probe.py',
               ROOT/'analysis/onevision_knowledge_intervention.py', ROOT/'analysis/onevision_research_worker.py',
               ROOT/'exp/onevision_diagnostics_20261016/run.py']
    spec = {'cases': CASES, 'variants': VARIANTS, 'seed': 42, 'backend': 'deterministic-default',
            'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
            'manifest_sha256': protocol['source_manifest_sha256'],
            'past_natural_checkpoint': str(checkpoint.relative_to(ROOT)), 'past_natural_sha256': expected,
            'TRAIN_ids': {str(t): [r['id'] for r in data[t, 'train']] for t in (2, 11, 12)},
            'current_probe_ids': {str(t): [r['id'] for r in current[t]] for t in current},
            'old_memory_ids': [r['id'] for r in memory],
            'target_test_ids': {str(t): [r['id'] for r in data[t, 'test']] for t in (2, 11, 12)},
            'old_test_ids': [r['id'] for r in tests['natural']],
            'old_generation_ids': [r['id'] for r in generation['natural']],
            'reuse_strengths': [1., .5, 0.], 'finite_step_fractions': [1., .5, .25, .125],
            'old_TRAIN_NLL_tolerance_nats': .02, 'trainable_parameters': 540672,
            'target_training': 'original one epoch, effective batch16, one fresh AdamW update per target,lr1e-4,weight_decay0,clip1',
            'formal_CL_ranking': False, 'full1313_training': False,
            'scope': '12 fixed independent one-update mechanism pilots; source old natural training cost347QA counted separately',
            'no_TEST_selection': True, 'success_target_later_full_CL': 'AnswerF1 AP +5pp vs same-setting SeqFT; not a promised effect'}
    return manifest, data, tests, generation, current, memory, json.loads(json.dumps(spec))


class Pilot(Intervention):
    def inputs(self, row, view='image', wrong=None, retain_pixels=False):
        batch = super().inputs(row, view, wrong, retain_pixels)
        if '_features' not in batch:
            return batch
        ids = batch['input_ids'];token = self.model.config.image_token_index
        positions = (ids[0] == token).nonzero().flatten()
        count = batch['_features'].shape[0]
        if len(positions) != count:
            assert len((wrong or row)['images']) == 1 and len(positions) > 0
            start, end = int(positions[0]), int(positions[-1])+1
            assert end-start == len(positions)
            revised = torch.cat([ids[:, :start], ids.new_full((1, count), token), ids[:, end:]], dim=1)
            assert torch.equal(ids[ids != token], revised[revised != token])
            batch['input_ids'] = revised;batch['attention_mask'] = torch.ones_like(revised)
        assert int((batch['input_ids'] == token).sum()) == count
        return batch

    def losses(self, state, current, old):
        # Preserve every RNG and the caller's train/eval mode across probes.
        py, npr = random.getstate(), np.random.get_state()
        mode = self.model.training
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                self.restore(state);self.model.eval()
                with torch.no_grad():
                    values = [float(self.loss(row)[0]) for row in current]
                    history = [float(self.loss(row)[0]) for row in old]
                self.probe_QA_forwards += len(current)+len(old)
                return (TrainLoss(sum(values)/len(values), tuple(r['id'] for r in current)),
                        {'natural': TrainLoss(sum(history)/len(history), tuple(r['id'] for r in old))})
        finally:
            random.setstate(py);np.random.set_state(npr);self.model.train(mode)


def worker(index):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    manifest, data, tests, generation, current, memory, spec = plan()
    assert spec == json.loads((REPORT/'CPU_PILOT_PLAN.json').read_text())['spec'], 'Pilot source/IDs changed after plan freeze'
    task, variant = CASES[index];use_A, use_B = VARIANTS[variant]
    name = f'T{task:02d}_{variant}'
    assert not (REPORT/(name+'_attempt.json')).exists(), 'Interrupted pilot needs explicit new run; never concatenate'
    directory = ROOT/'exp/onevision_two_module_20261017/pilot'/name/time.strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True)
    write(REPORT/(name+'_attempt.json'), {'run': str(directory.relative_to(ROOT)), 'spec': spec})
    write(directory/'protocol.json', spec)
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed();started = time.time()
    x = Pilot(manifest, directory);x.process_started = started;x.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    x.probe_QA_forwards = 0
    start = torch.load(ROOT/spec['past_natural_checkpoint'], map_location='cpu', weights_only=True)['adapter']
    x.configure(start, finetune_shared=True)
    assert sum(p.numel() for p in x.model.parameters() if p.requires_grad) == 540672
    write(directory/'model_counts.json', {'total_parameters': sum(p.numel() for p in x.model.parameters()),
          'trainable_parameters': 540672, 'retained_LoRA_modules': 1, 'rank': 8,
          'temporary_candidate_clones_billed_in_memory': True, 'adaptive_module_count_tested': False})
    # Same official native-forward gate including the known packed-token failure.
    x.model.eval()
    RepairedExperiment.forward_gate(x)
    evaluate = lambda state: x.losses(state, current[task], memory)
    current_ids = tuple(r['id'] for r in current[task]);memory_ids = {'natural': tuple(r['id'] for r in memory)}
    initial_current, initial_old = evaluate(start)
    references = {'natural': initial_old['natural'].value}
    gate = REPORT/f'T{task:02d}_initial_function_gate.json'
    gate_value = {'initial_current_TRAIN_NLL': initial_current.value,
                  'initial_old_TRAIN_NLL': references['natural'], 'start_sha256': spec['past_natural_sha256']}
    if gate.exists():
        assert json.loads(gate.read_text()) == gate_value, 'Common initial-function gate failed'
    else:
        assert variant == 'control';write(gate, gate_value)
    reference_update = None
    if variant == 'control':
        x.model.eval()
        x.evaluate('initial', {'target': data[task, 'test'], 'past_natural': tests['natural']},
                   {'target': data[task, 'test'], 'past_natural': generation['natural']})
        # Actual original trainer, same start/RNG/QA; excluded admission window.
        x.restore(start);x.cache.clear();reset_seed()
        reference_update = Experiment.train(x, 'original_trainer_reference', data[task, 'train'])
        x.restore(start);x.model.eval();gc.collect();torch.cuda.empty_cache()
    x.restore(start);x.model.eval()
    torch.cuda.reset_peak_memory_stats();x.monitor.snapshot('pilot_training_before', model=x.model)
    probe_forwards_before = x.probe_QA_forwards
    selected, selection = (choose_reuse(start, evaluate, current_ids, memory_ids, references, use_B)
                            if use_A else (start, {'chosen_strength': 1., 'disabled': True}))
    x.restore(selected);x.cache.clear();reset_seed();x.model.train()
    step_rng = capture_rng()
    named = [(n, p) for n, p in x.model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    optimizer_initial = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True);values = []
    order = sorted(data[task, 'train'], key=lambda row: hashed(row, 'training'))
    assert len(order) <= 16
    for row in order:
        loss, _ = x.loss(row);assert torch.isfinite(loss)
        (loss/len(order)).backward();values.append(float(loss.detach()))
    raw = torch.cat([p.grad.detach().float().cpu().flatten() for p in parameters])
    assert torch.isfinite(raw).all()
    if variant == 'B_only':
        control = json.loads((REPORT/f'T{task:02d}_control_complete.json').read_text())
        control_path = ROOT/control['run']/'raw_gradient.pt'
        assert digest(control_path) == control['artifact_sha256']['raw_gradient.pt']
        control_raw = torch.load(control_path, map_location='cpu', weights_only=True)['gradient']
        assert torch.equal(raw, control_raw), 'B-only raw target gradient differs from same-start control'
        write(directory/'matched_raw_gradient_gate.json', {'exact_equal': True, 'no_tolerance': True})
    atomic_torch(directory/'raw_gradient.pt', {'gradient': raw, 'layout': [(n, list(p.shape)) for n, p in named],
                                             'TRAIN_ids': [r['id'] for r in order]})
    grad = torch.nn.utils.clip_grad_norm_(parameters, 1.);assert torch.isfinite(grad)
    used_lr = optimizer.param_groups[0]['lr'];assert used_lr > 0
    optimizer.step();proposed = x.snapshot()
    if reference_update is not None:
        assert all(torch.equal(proposed[n][f], reference_update[n][f]) for n in proposed for f in ('A', 'B'))
        old_events = [json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()
                      if json.loads(s).get('label') == 'original_trainer_reference']
        assert len(old_events) == 1
        assert float(grad) == old_events[0]['gradient_l2_preclip']
        assert sum(values)/len(values) == old_events[0]['loss']
        write(directory/'original_trainer_exact_gate.json', {'passed': True, 'all_A_B_exact': True,
                                                            'loss_and_gradient_norm_exact': True})
    accepted, step = (accept_displacement(selected, proposed, evaluate, current_ids, memory_ids, references)
                      if use_B else (proposed, {'accepted_fraction': 1., 'disabled': True}))
    if step['accepted_fraction'] == 0:
        optimizer.load_state_dict(optimizer_initial)
        restore_rng(step_rng)
    x.restore(accepted);optimizer.zero_grad(set_to_none=True)
    changed = any(not torch.equal(accepted[n][f], selected[n][f]) for n in accepted for f in ('A', 'B'))
    valid_training = step['accepted_fraction'] > 0 and changed
    endpoint_current, endpoint_old = evaluate(accepted)
    x.restore(accepted)
    atomic_torch(directory/'optimizer_after.pt', optimizer.state_dict())
    del optimizer, parameters, loss
    gc.collect();torch.cuda.empty_cache();x.monitor.snapshot('pilot_training_after', model=x.model)
    training = {'task': task, 'variant': variant, 'original_target_TRAIN_QA': len(order),
                'actual_gradient_L2_preclip': float(grad), 'pre_update_loss': sum(values)/len(values),
                'actual_learning_rate_used': used_lr, 'actual_step_fraction': step['accepted_fraction'],
                'nonzero_effective_updates': int(valid_training), 'valid_training': valid_training,
                'selection': selection, 'finite_step': step, 'extra_TRAIN_probe_QA_forwards': x.probe_QA_forwards,
                'method_window_probe_QA_forwards': x.probe_QA_forwards-probe_forwards_before,
                'original_trainer_admission_QA': len(order) if variant == 'control' else 0,
                'current_probe_NLL_before': initial_current.value, 'current_probe_NLL_after': endpoint_current.value,
                'old_probe_NLL_before': references, 'old_probe_NLL_after': {t: v.value for t, v in endpoint_old.items()},
                'all_rejected_is_not_valid_CL_score': True}
    write(directory/'training.json', training)
    atomic_torch(directory/'final_adapter.pt', {'adapter': accepted, 'rank': 8, 'alpha': 32,
                                               'base_revision': '74dd0bf867a4cda7950c17663794267c60cf4b40'})
    reloaded = torch.load(directory/'final_adapter.pt', map_location='cpu', weights_only=True)['adapter']
    assert all(torch.equal(reloaded[n][f], accepted[n][f]) for n in accepted for f in ('A', 'B'))
    x.restore(reloaded);x.model.eval()
    # Target TEST is touched only after all TRAIN choices/steps are immutable.
    x.evaluate('final', {'target': data[task, 'test'], 'past_natural': tests['natural']},
               {'target': data[task, 'test'], 'past_natural': generation['natural']})
    x.monitor.snapshot('pilot_case_after', model=x.model)
    finish_resources(x, directory, 'pilot_training_before', 'pilot_training_after',
                     'one-update method window includes A/B probe computation; final inference excluded')
    artifacts = [*directory.rglob('*.pt'), *directory.rglob('*.jsonl'), *directory.rglob('*.json')]
    write(REPORT/(name+'_complete.json'), {'complete': True, 'run': str(directory.relative_to(ROOT)), 'spec': spec,
          'valid_training': valid_training, 'module_selection': selection, 'step': step, 'formal_CL_result': False,
          'artifact_sha256': {str(p.relative_to(directory)): digest(p) for p in artifacts}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--check', action='store_true')
    parser.add_argument('--case', type=int, choices=range(len(CASES)));args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        if args.check:
            *_, spec = plan();assert not torch.cuda.is_initialized()
            write(REPORT/'CPU_PILOT_PLAN.json', {'passed': True, 'GPU_used': False, 'spec': spec,
                                               'GPU_runtime_validated': False})
            print('12 fixed pilot cases and original TRAIN/TEST identity passed; actual GPU gates pending')
        else:
            assert args.case is not None
            lock = (REPORT/f'case{args.case}.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker(args.case)
    except Exception:
        write(REPORT/f'case{args.case}_needs_attention.json', {'traceback': traceback.format_exc(), 'GPU_source_not_hot_edited': True})
        raise
