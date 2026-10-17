"""Deterministic replication of existing CL controls, two fixed medical anchors.

No new CL method. Wait both full-confirmation jobs; fresh process per case.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse, fcntl, hashlib, json, subprocess, sys, time, traceback
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'analysis'))
from onevision_task_domain_intervention import Mechanism, METHODS, SEQUENCES
from onevision_knowledge_intervention import reset_seed
from onevision_research_worker import atomic_torch, digest
from onevision_capacity_probe import finish_resources
from run import freeze, write
REPORT = ROOT / 'summary/onevision_diagnostics_20261016/matched_task_domain'
ANCHORS = [4, 2]
SCHEDULES = {4: SEQUENCES, 2: {'repeat_anchor': [2]*5, 'within_medical': [2, 4, 3, 6, 11],
                            'cross_short': [2, 1, 13, 15, 16], 'cross_five_domains': [2, 1, 13, 16, 18]}}
CASES = [(anchor, sequence, method) for anchor in ANCHORS for sequence in SCHEDULES[anchor] for method in METHODS]


def key(row, label): return hashlib.sha256(('42:' + label + ':' + row['id']).encode()).digest()


def plan():
    manifest, data, _, _, _, _, protocol = freeze()
    old = json.loads((ROOT / 'summary/onevision_mechanisms_20261016/task_domain_attempt.json').read_text())
    for p, h in old['spec']['source_sha256'].items(): assert digest(ROOT / p) == h
    tasks = sorted({t for sequences in SCHEDULES.values() for seq in sequences.values() for t in seq})
    selected = {t: sorted(data[t, 'train'], key=lambda r: key(r, 'task-domain'))[:8] for t in tasks}
    assert all(len(rows) == 8 for rows in selected.values())
    assert all([r['id'] for r in rows] == old['spec']['train_ids'][str(t)] for t, rows in selected.items())
    anchors = {t: sorted(data[t, 'test'], key=lambda r: key(r, 'matched-anchor-test'))[:32] for t in ANCHORS}
    current = {t: sorted(data[t, 'test'], key=lambda r: key(r, 'matched-plasticity-test'))[:8] for t in tasks}
    assert all(len(rows) == 32 for rows in anchors.values()) and all(len(rows) == 8 for rows in current.values())
    for t in tasks: assert not {r['id'] for r in selected[t]} & {r['id'] for r in data[t, 'test']}
    paths = [Path(__file__), ROOT / 'analysis/onevision_task_domain_intervention.py', ROOT / 'analysis/onevision_knowledge_intervention.py',
             ROOT / 'analysis/onevision_research_worker.py', ROOT / 'analysis/onevision_capacity_probe.py', ROOT / 'exp/onevision_diagnostics_20261016/run.py']
    spec = {'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths}, 'manifest_sha256': protocol['source_manifest_sha256'],
            'cases': CASES, 'sequences': SCHEDULES, 'methods': METHODS, 'anchors': ANCHORS,
            'TRAIN_ids': {str(t): [r['id'] for r in rows] for t, rows in selected.items()},
            'anchor_probe_ids': {str(t): [r['id'] for r in rows] for t, rows in anchors.items()},
            'plasticity_ids': {str(t): [r['id'] for r in rows] for t, rows in current.items()},
            'final_SeqFT_full_anchor_ids': {str(t): [r['id'] for r in data[t, 'test']] for t in ANCHORS},
            'seed': 42, 'backend': 'deterministic-default', 'new_QA_per_stage': 8, 'updates_per_stage': 4,
            'optimizer': 'original Mechanism.step_task unchanged; fresh AdamW lr1e-4,weight_decay0 each stage',
            'prompt': 'original unchanged', 'gate': 'every case actual stage1 A/B exactly equals canonical same-anchor stage1; fail closed',
            'evaluation': '32 fixed anchor QA at1/3/5,8 current-task held-out QA at2/3/4/5; final SeqFT also full native anchor507/228',
            'scope': '48 fixed mechanism trajectories, not formal1epoch19stage CL or full21454 method ranking',
            'limitations': ['two medical anchors, one order/seed; still not pure domain-count intervention independent of content',
                            'cross-short four domains differs in content from cross-five; compare all, no best-case selection',
                            'ER/Fisher/projection access extra history/compute; bill explicitly',
                            '32 and8-QA probes are diagnostic; only SeqFT final native-anchor evaluation is complete per-anchor',
                            'projection is an existing mechanism intervention, not a proposed published algorithm',
                            'every first-step full gradient saved by original trainer; not every historical optimizer gradient']}
    return manifest, data, selected, anchors, current, json.loads(json.dumps(spec))


def checked_checkpoint(path):
    meta = json.loads(path.with_suffix('.json').read_text()); assert digest(path) == meta['sha256']
    return torch.load(path, map_location='cpu', weights_only=True)


def worker(index):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    manifest, data, selected, anchors, current, spec = plan(); anchor, sequence, method = CASES[index]
    name = f'T{anchor:02d}_{sequence}__{method}'; pointer = REPORT / (name + '_attempt.json')
    if pointer.exists():
        old = json.loads(pointer.read_text()); assert old['spec'] == spec
        write(REPORT / (name + '_interrupted_' + str(time.time_ns()) + '.json'), old)
        write(ROOT / old['run'] / 'interrupted.json', {'incomplete': True, 'restart_from_base_not_concatenated': True})
    directory = ROOT / 'exp/onevision_matched_task_domain_20261017' / name / time.strftime('%Y%m%d_%H%M%S')
    directory.mkdir(parents=True); write(directory / 'protocol.json', spec); write(pointer, {'run': str(directory.relative_to(ROOT)), 'spec': spec})
    torch.set_num_threads(6); torch.use_deterministic_algorithms(True); reset_seed(); started = time.time()
    x = Mechanism(manifest, directory); x.process_started = started; x.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    x.prompt('original'); x.configure(x.original_initial, finetune_shared=True)
    fishers = []; tasks = SCHEDULES[anchor][sequence]; x.monitor.snapshot('case_before', model=x.model)
    for stage, task in enumerate(tasks, 1):
        write(REPORT / 'status.json', {'phase': 'training', 'case': name, 'stage': stage, 'pid': os.getpid(), 'run': str(directory.relative_to(ROOT))})
        x.monitor.stage = stage; torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        history = [r for previous in tasks[:stage-1] for r in selected[previous]]
        x.step_task(name, stage, selected[task], method, history, selected[anchor], fishers)
        state = x.snapshot(); atomic_torch(directory / 'adapters' / f'stage_{stage:02d}.pt', {'adapter': state, 'stage': stage})
        if stage == 1:
            canonical = REPORT / f'canonical_anchor{anchor}.pt'
            if not canonical.exists():
                assert sequence == 'repeat_anchor' and method == METHODS[0]
                atomic_torch(canonical, {'adapter': state, 'source_sha256': spec['source_sha256'], 'anchor': anchor})
            reference = checked_checkpoint(canonical); assert reference['source_sha256'] == spec['source_sha256']
            assert all(torch.equal(state[k][f], reference['adapter'][k][f]) for k in state for f in ['A', 'B']), 'Stage1 exact gate FAILED; no comparisons accepted'
            write(directory / 'stage1_exact_gate.json', {'passed': True, 'canonical_sha256': digest(canonical), 'no_tolerance': True})
        if method.startswith('EWC_'):
            torch.cuda.reset_peak_memory_stats(); x.monitor.snapshot(f'{name}_fisher{stage}_before', stage, model=x.model)
            fishers.append(x.fisher(selected[task])); x.monitor.snapshot(f'{name}_fisher{stage}_after', stage, model=x.model)
            atomic_torch(directory / 'fishers' / f'stage_{stage:02d}.pt', fishers[-1])
        if stage in [1, 3, 5]:
            x.model.eval(); x.evaluate(f'anchor_after{stage}', {'anchor': anchors[anchor]}, {'anchor': anchors[anchor]})
        if stage > 1:
            x.model.eval(); x.evaluate(f'plasticity_after{stage}', {'current': current[task]}, {'current': current[task]})
        if stage == 5 and method == 'SeqFT_LoRA':
            x.model.eval(); x.evaluate('final_native_anchor', {'anchor': data[anchor, 'test']}, {'anchor': data[anchor, 'test']})
    x.monitor.snapshot('case_after', model=x.model)
    finish_resources(x, directory, 'case_before', 'case_after', 'whole case includes inference; per-stage TRAIN/Fisher snapshots separate')
    artifacts = [*directory.rglob('*.pt'), *directory.rglob('*.jsonl'), *directory.rglob('*.json')]
    write(REPORT / (name + '_complete.json'), {'complete': True, 'run': str(directory.relative_to(ROOT)), 'spec': spec,
          'stage1_exact': True, 'full21454': False, 'artifact_sha256': {str(p.relative_to(directory)): digest(p) for p in artifacts}})


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true'); parser.add_argument('--case', type=int, choices=range(len(CASES))); args = parser.parse_args()
    if args.check:
        *_, spec = plan(); assert len(CASES) == 48 and not torch.cuda.is_initialized()
        write(REPORT / 'CPU_PLAN_CHECK.json', {'passed': True, 'GPU_used': False, 'spec': spec}); print('Frozen TRAIN identity, held-out32/8 IDs and48-case plan passed; actual GPU equality pending'); return
    lock = (REPORT / ('coordinator.lock' if args.case is None else f'case{args.case}.lock')).open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (REPORT / 'needs_attention.json').exists()
    if args.case is not None: worker(args.case); return
    for index, (anchor, sequence, method) in enumerate(CASES):
        name = f'T{anchor:02d}_{sequence}__{method}'; done = REPORT / (name + '_complete.json')
        if done.exists():
            proof = json.loads(done.read_text()); assert proof['spec'] == plan()[-1]
            assert all(digest(ROOT / proof['run'] / p) == h for p, h in proof['artifact_sha256'].items()); continue
        while True:
            failures = list((REPORT.parent / 'full_input_repair').glob('cap*_full_confirm_needs_attention.json'))
            if failures:
                write(REPORT / 'upstream_needs_attention.json', {'failed_proofs': [str(p.relative_to(ROOT)) for p in failures], 'GPU_started': False})
                write(REPORT / 'queue_state.json', {'phase': 'upstream_needs_attention', 'pid': os.getpid()}); return
            units = ['llmcl-onevision-deferred-full.service', 'llmcl-onevision-full-repair.service', 'llmcl-onevision-knowledge-fit.service', 'llmcl-onevision-rural-weights.service']
            owners = {u: int(subprocess.check_output(['systemctl', '--user', 'show', u, '-p', 'MainPID', '--value'], text=True).strip() or 0) for u in units}
            full_done = all((REPORT.parent / 'full_input_repair' / f'cap{cap}_full_confirm_complete.json').exists() for cap in [1, 3])
            pids = subprocess.check_output(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            used, util = map(int, subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(','))
            if full_done and not any(owners.values()) and not pids and used < 500 and util < 10: break
            write(REPORT / 'queue_state.json', {'phase': 'waiting_full_confirmations_and_idle_GPU1', 'pid': os.getpid(), 'services': owners, 'full_confirmations_complete': full_done, 'GPU_PIDs': pids.splitlines()}); time.sleep(30)
        with (REPORT / (name + '_nohup.log')).open('a') as log:
            child = subprocess.Popen(['/usr/bin/nohup', str(ROOT / '.vision-env/bin/python'), '-u', str(Path(__file__)), '--case', str(index)],
                                     cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1'), stdout=log, stderr=subprocess.STDOUT)
            write(REPORT / 'queue_state.json', {'phase': 'running', 'case': name, 'pid': os.getpid(), 'worker_pid': child.pid}); code = child.wait()
        assert code == 0 and done.exists(), 'Worker failed; inspect evidence, no blind retry'
    write(REPORT / 'complete.json', {'complete': True, 'cases': 48, 'full21454': False})
    write(REPORT / 'queue_state.json', {'phase': 'complete', 'pid': os.getpid()})


if __name__ == '__main__':
    try: main()
    except BlockingIOError: pass
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True); write(REPORT / 'needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()}); raise
