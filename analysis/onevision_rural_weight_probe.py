"""Causal loss-weight control, not a proposed CL method or dataset revision."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, fcntl, hashlib, json, math, subprocess, sys, time, traceback
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'analysis'))
from onevision_basis_budget_probe import train_trajectory, weight_error
from onevision_knowledge_intervention import Intervention, reset_seed
from onevision_research_worker import atomic_torch, digest
from onevision_capacity_probe import finish_resources
from run import freeze, hashed, write
REPORT = ROOT / 'summary/onevision_diagnostics_20261016/rural_weights'
CASES = ['uniform', 'class_balanced', 'urban_high_negative']


def weights(rows, case):
    assert case in CASES and len(rows) == 6
    counts = {k: sum(r['answer'] == k for r in rows) for k in ['urban', 'rural']}
    assert counts == {'urban': 4, 'rural': 2}
    if case == 'uniform': return {r['id']: 1 / 6 for r in rows}
    if case == 'class_balanced': return {r['id']: 1 / (2 * counts[r['answer']]) for r in rows}
    high = {r['id'] for r in sorted(rows, key=lambda r: hashed(r, 'training')) if r['answer'] == 'urban'}
    high = set(sorted(high, key=lambda k: hashlib.sha256(('42:weight-negative:' + k).encode()).digest())[:2])
    return {r['id']: .25 if r['id'] in high else .125 for r in rows}


def plan():
    manifest, data, _, _, _, _, protocol = freeze()
    parent_path = ROOT / 'summary/onevision_diagnostics_20261016/basis_budget/complete.json'
    parent = json.loads(parent_path.read_text()); basis = ROOT / parent['run']
    original = json.loads((basis / 'protocol.json').read_text())
    for p, h in original['source_sha256'].items(): assert digest(ROOT / p) == h
    case = basis / 'T12_past_AB'; proof = json.loads((case / 'complete.json').read_text())
    for p, h in proof['artifact_sha256'].items(): assert digest(case / p) == h
    past = ROOT / 'exp/onevision_diagnostics_20261016/runs/20261016_164638/adapters/isolated_natural.pt'
    assert digest(past) == original['past_adapter_sha256']
    rows = data[12, 'train']; test = data[12, 'test']; assert len(test) == 100
    maps = {name: weights(rows, name) for name in CASES}
    assert all(abs(sum(w.values()) - 1) < 1e-12 for w in maps.values())
    assert sorted(maps['class_balanced'].values()) == sorted(maps['urban_high_negative'].values())
    paths = [Path(__file__), ROOT / 'analysis/onevision_basis_budget_probe.py', ROOT / 'analysis/onevision_knowledge_intervention.py',
             ROOT / 'analysis/onevision_research_worker.py', ROOT / 'analysis/onevision_capacity_probe.py', ROOT / 'exp/onevision_diagnostics_20261016/run.py']
    spec = {'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
            'manifest_sha256': protocol['source_manifest_sha256'], 'past_adapter_sha256': digest(past),
            'basis_complete_sha256': digest(parent_path), 'basis_case': str(case.relative_to(ROOT)),
            'train_ids': [r['id'] for r in rows], 'test_ids': [r['id'] for r in test], 'weights': maps,
            'cases': CASES, 'updates': 4, 'target_QA_exposures': 24, 'endpoints': [1, 4],
            'seed': 42, 'lr': 1e-4, 'weight_decay': 0, 'clip': 1, 'backend': 'deterministic-default',
            'prompt': 'unchanged original, no knowledge', 'scope': 'TRAIN-loss weighting intervention; not formal1epoch CL or full21454',
            'rule': 'all three conditions fixed before new outcomes; no test-choice of weights, labels, thresholds or checkpoints',
            'uniform_gate': 'both endpoint tensors must exactly reproduce prior basis T12 pastAB; stop on mismatch',
            'negative': 'same coefficient multiset as balanced but two TRAIN urban examples receive high weights',
            'data': 'same six original TRAIN QA and all100 native TEST; no added/deleted/rewritten QA'}
    return manifest, rows, test, past, case, spec


def weighted_train(x, rows, start, directory, mapping):
    x.configure(start, finetune_shared=True); reset_seed(); x.model.train()
    named = [(n, p) for n, p in x.model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]; opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0)
    ordered = sorted(rows, key=lambda r: hashed(r, 'training')); events = []
    x.monitor.snapshot('target_training_before', model=x.model)
    for step in range(1, 5):
        opt.zero_grad(set_to_none=True); losses = []
        for row in ordered:
            loss, _ = x.loss(row); assert torch.isfinite(loss)
            (loss * mapping[row['id']]).backward(); losses.append(float(loss.detach()))
        vector = torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for _, p in named])
        norm = torch.nn.utils.clip_grad_norm_(params, 1.); assert torch.isfinite(norm)
        opt.step()
        atomic_torch(directory / 'raw_gradients' / f'update_{step}.pt', {'gradient_preclip': vector,
                     'layout': [(n, list(p.shape)) for n, p in named], 'TRAIN_ids': [r['id'] for r in ordered], 'step': step})
        if step in [1, 4]: atomic_torch(directory / f'update_{step}.pt', {'adapter': x.snapshot(), 'updates': step})
        events.append({'update': step, 'weighted_pre_update_loss': sum(v * mapping[r['id']] for r, v in zip(ordered, losses)),
                       'unweighted_pre_update_loss': sum(losses) / len(rows), 'gradient_l2_preclip': float(norm),
                       'learning_rate_used': 1e-4, 'cumulative_QA_exposures': step * len(rows)})
        write(directory / 'training.json', {'weights': mapping, 'records': events})
    opt.zero_grad(set_to_none=True); del opt
    x.monitor.snapshot('target_training_after', model=x.model)


def worker(index):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    manifest, rows, test, past, basis_case, spec = plan(); name = CASES[index]
    if name != 'uniform':
        gate = json.loads((REPORT / 'uniform_complete.json').read_text()); assert gate['uniform_exact_gate']
        assert gate['spec'] == spec
    directory = ROOT / 'exp/onevision_rural_weights_20261017' / name / time.strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True); write(directory / 'protocol.json', spec)
    write(REPORT / (name + '_attempt.json'), {'run': str(directory.relative_to(ROOT)), 'spec': spec})
    torch.set_num_threads(6); torch.use_deterministic_algorithms(True); reset_seed(); started = time.time()
    x = Intervention(manifest, directory); x.process_started = started; x.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    x.prompt('original'); x.knowledge = None
    start = torch.load(past, map_location='cpu', weights_only=True)['adapter']
    atomic_torch(directory / 'start.pt', {'adapter': start})
    if name == 'uniform': train_trajectory(x, rows, start, directory)
    else: weighted_train(x, rows, start, directory, spec['weights'][name])
    if name == 'uniform':
        for step in [1, 4]:
            actual = torch.load(directory / f'update_{step}.pt', map_location='cpu', weights_only=True)['adapter']
            expected = torch.load(basis_case / f'update_{step}.pt', map_location='cpu', weights_only=True)['adapter']
            assert weight_error(actual, expected) == 0, 'Uniform endpoint mismatch: preserve failure, no causal scores'
    finish_resources(x, directory, 'target_training_before', 'target_training_after', 'training-only4 updates; endpoint inference excluded from training peaks')
    selected = {'TRAIN': rows, 'TEST': test}
    for step in [1, 4]:
        state = torch.load(directory / f'update_{step}.pt', map_location='cpu', weights_only=True)['adapter']
        x.configure(state, finetune_shared=False); x.model.eval()
        x.evaluate(f'endpoint_{step}', selected, selected)
    artifacts = [*directory.rglob('*.pt'), *directory.rglob('*.jsonl'), *directory.rglob('*.json')]
    write(REPORT / (name + '_complete.json'), {'complete': True, 'run': str(directory.relative_to(ROOT)), 'spec': spec,
          'uniform_exact_gate': True, 'artifact_sha256': {str(p.relative_to(directory)): digest(p) for p in artifacts}, 'full21454': False})


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); parser.add_argument('--case', type=int, choices=range(3)); args = parser.parse_args()
    if args.check:
        *_, spec = plan(); assert not torch.cuda.is_initialized()
        write(REPORT / 'CPU_PLAN_CHECK.json', {'passed': True, 'GPU_used': False, 'spec': spec}); print('CPU frozen QA/weights/references passed; GPU uniform equivalence pending'); return
    lock = (REPORT / ('coordinator.lock' if args.case is None else f'case{args.case}.lock')).open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (REPORT / 'needs_attention.json').exists()
    if args.case is not None: worker(args.case); return
    for index, name in enumerate(CASES):
        marker = REPORT / (name + '_complete.json')
        if marker.exists():
            proof = json.loads(marker.read_text()); assert proof['spec'] == plan()[-1]
            assert all(digest(ROOT / proof['run'] / p) == h for p, h in proof['artifact_sha256'].items()); continue
        while True:
            units = ['llmcl-onevision-knowledge-fit.service', 'llmcl-onevision-rural-collapse.service']
            owners = {u: int(subprocess.check_output(['systemctl', '--user', 'show', u, '-p', 'MainPID', '--value'], text=True).strip() or 0) for u in units}
            pids = subprocess.check_output(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            used, util = map(int, subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(','))
            if not any(owners.values()) and not pids and used < 500 and util < 10: break
            write(REPORT / 'queue_state.json', {'phase': 'waiting_existing_and_idle_GPU1', 'pid': os.getpid(), 'services': owners, 'GPU_PIDs': pids.splitlines()}); time.sleep(15)
        with (REPORT / (name + '_nohup.log')).open('a') as log:
            child = subprocess.Popen(['/usr/bin/nohup', str(ROOT / '.vision-env/bin/python'), '-u', str(Path(__file__)), '--case', str(index)],
                                     cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1'), stdout=log, stderr=subprocess.STDOUT)
            write(REPORT / 'queue_state.json', {'phase': 'running', 'case': name, 'pid': os.getpid(), 'worker_pid': child.pid}); code = child.wait()
        assert code == 0 and marker.exists(), 'Worker failed; no blind retries'
    write(REPORT / 'complete.json', {'complete': True, 'cases': CASES, 'full21454': False})
    write(REPORT / 'queue_state.json', {'phase': 'complete', 'pid': os.getpid()})


if __name__ == '__main__':
    try: main()
    except BlockingIOError: pass
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True); write(REPORT / 'needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()}); raise
