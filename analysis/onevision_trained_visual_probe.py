"""GPU1 inference-only visual ablation after generic/medical/sequential LoRA.

Fixed300 original diagnostic QA, not full test. Checkpoint choices follow task
identity/order, not best test scores. No optimizer or parameter update occurs.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'exp/onevision_diagnostics_20261016'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'analysis'))
from run import Experiment, freeze, write
from onevision_research_worker import digest
from onevision_knowledge_intervention import reset_seed

VIEWS = ('image', 'text', 'wrong')


def complete(run, case, tests):
    folder = run/'evaluations'/case; proof = folder/'durable_complete.json'
    if not proof.exists(): return False
    files = json.loads(proof.read_text())['files']
    for domain, rows in tests.items():
        for view in VIEWS:
            path = folder/f'{domain}_{view}.jsonl'
            if not path.exists() or digest(path) != files.get(path.name): return False
            actual = [json.loads(line) for line in path.read_text().splitlines()]
            if [r['id'] for r in actual] != [r['id'] for r in rows]: return False
    return True


def repair_tail(path):
    if not path.exists(): return
    lines = path.read_text().splitlines(); valid = []
    for i, line in enumerate(lines):
        try: json.loads(line); valid.append(line)
        except json.JSONDecodeError:
            if i != len(lines)-1: raise
            archive = path.parent/'interrupted'; archive.mkdir(exist_ok=True)
            (archive/(str(time.time_ns())+'_'+path.name)).write_bytes(path.read_bytes())
            path.write_text('\n'.join(valid)+ ('\n' if valid else ''))


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    torch.set_num_threads(6); reset_seed()
    manifest, data, domains, tests, generation, probes, protocol = freeze()
    tests = {d:tests[d] for d in ['natural','medical','geography']}
    generation = {d:tests[d] if d == 'medical' else generation[d] for d in tests}
    report = ROOT/'summary/onevision_mechanisms_20261016'
    reference = json.loads((ROOT/'summary/onevision_diagnostics_20261016/research_current_attempt.json').read_text())
    old = ROOT/reference['old_control_run']; sequential = ROOT/reference['run']
    checkpoints = {'generic_natural': old/'adapters/isolated_natural.pt',
                   'medical_isolated': old/'adapters/isolated_medical.pt',
                   'sequential_final': sequential/'durable_states/stage_19.pt'}
    for name, path in checkpoints.items():
        expected = (reference['completed_control_sha256'][str(path.relative_to(ROOT))]
                    if name != 'sequential_final' else json.loads(path.with_suffix('.json').read_text())['sha256'])
        assert digest(path) == expected
    files = [Path(__file__), HERE/'run.py', ROOT/'analysis/onevision_research_worker.py',
             ROOT/'analysis/onevision_knowledge_intervention.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    spec = {'source_sha256': hashes, 'checkpoint_sha256': {k:digest(v) for k,v in checkpoints.items()},
            'test_ids': {d:[r['id'] for r in rows] for d,rows in tests.items()},
            'generation_ids': {d:[r['id'] for r in rows] for d,rows in generation.items()},
            'states': ['base', *checkpoints], 'views': list(VIEWS), 'training_updates': 0,
            'scope': 'inference only;300 diagnostic QA,120 generation QA/state/view; not full21454 test',
            'checkpoint_selection': 'base, first original domain, medical domain, final sequential stage; not best test score',
            'confounds': 'Old independent adapter dropout streams and training budgets differ; no matched-training causal claim.'}
    pointer = report/'trained_vision_attempt.json'
    if pointer.exists():
        info = json.loads(pointer.read_text()); assert info['spec'] == spec; run = ROOT/info['run']
    else:
        run = ROOT/'exp/onevision_mechanisms_20261016/trained_vision'/time.strftime('%Y%m%d_%H%M%S'); run.mkdir(parents=True)
        write(pointer, {'run': str(run.relative_to(ROOT)), 'spec': spec}); write(run/'protocol.json', spec)
    x = Experiment(manifest, run); initial = x.initial
    for p in x.model.parameters(): p.requires_grad = False
    x.monitor.snapshot('inference_before', model=x.model)
    for name in spec['states']:
        case = 'vision_'+name
        if complete(run, case, tests): continue
        state = (initial if name == 'base' else torch.load(checkpoints[name],map_location='cpu',weights_only=False)['adapter'])
        x.restore(state); x.model.eval()
        write(report/'status.json', {'phase':'trained_visual_ablation','case':case,'run':str(run.relative_to(ROOT)),'pid':os.getpid()})
        folder = run/'evaluations'/case
        for domain in tests:
            for view in VIEWS: repair_tail(folder/f'{domain}_{view}.jsonl')
        x.evaluate(case,tests,generation,views=VIEWS)
        proof = {}
        for domain in tests:
            for view in VIEWS:
                path = folder/f'{domain}_{view}.jsonl'
                with path.open('rb') as stream: os.fsync(stream.fileno())
                proof[path.name] = digest(path)
        write(folder/'durable_complete.json', {'files':proof,'source_sha256':hashes})
        assert complete(run,case,tests)
    x.monitor.snapshot('inference_after',model=x.model)
    # Shared monitor uses legacy training labels; explicitly mark the workload.
    x.monitor.snapshot('training_after',model=x.model); memory = x.monitor.finish(x.model)
    memory.update(workload='inference_only',optimizer_updates=0,
                  phase_caveat='Legacy training_before/after monitor labels do not imply this run trained.')
    write(run/'memory_summary.json',memory)
    summaries = {name:json.loads((run/'evaluations'/('vision_'+name)/'summary.json').read_text()) for name in spec['states']}
    write(report/'trained_vision_complete.json', {'complete':True,'run':str(run.relative_to(ROOT)),
          'source_sha256':hashes,'formal_full_test':False,'optimizer_updates':0,'summaries':summaries})


if __name__ == '__main__':
    main()
