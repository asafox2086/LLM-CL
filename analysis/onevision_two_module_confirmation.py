"""Frozen-checkpoint inference for all four predeclared two-module ablations.

Final21454 first; sparse acquisition/current-next inference later supplies full
FWT/BWT. No training, router, checkpoint choice by TEST, or prediction splicing.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import fcntl
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
import onevision_two_module_fullstream as stream
from onevision_knowledge_intervention import reset_seed
from onevision_full_input_repair import RepairedExperiment
from onevision_research_worker import digest
from run import write

REPORT = stream.REPORT/'confirmation'


def plan():
    manifest, data, domains, _, _, _, _, training = stream.plan()
    assert training == json.loads((stream.REPORT/'CPU_FULLSTREAM_PLAN.json').read_text())['spec']
    tests = {domain: [r for t in tasks for r in data[t, 'test']] for domain, tasks in domains.items()}
    spec = {'training_spec': training, 'evaluator_sha256': digest(Path(__file__)),
            'final_TEST_QA': 21454, 'variants': list(stream.VARIANTS),
            'all_TEST_ids': {d: [r['id'] for r in rows] for d, rows in tests.items()},
            'full_metrics': '19 equal-task final AP; FWT pre-task minus base tasks2..19; BWT final minus acquired tasks1..18',
            'sparse_inference': 'States1..18: native current+next full tests, no fabricated missing cells',
            'selection': 'ALL predeclared valid variants, never TEST-selected candidate',
            'mode': 'NativeAnyRes research, separate from formal resized V2 baselines',
            'source_sha256': {**training['source_sha256'], str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}}
    assert sum(map(len, tests.values())) == 21454
    return manifest, data, tests, json.loads(json.dumps(spec))


def verify(variant, phase, spec):
    name = 'shared_initial' if phase == 'initial' else variant+'_'+phase
    path = REPORT/(name+'_complete.json')
    if not path.exists():
        return False
    proof = json.loads(path.read_text());assert proof['complete'] and proof['spec'] == spec
    run = ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_two_module_20261017/confirmation').resolve())
    for relative, sha in proof['artifact_sha256'].items():
        path = run/relative
        assert path.resolve().is_relative_to(run.resolve()) and digest(path) == sha
    return True


def native_subset(manifest, data, tasks):
    selected = {}
    task_domains = {t['task']: t['domain'] for t in manifest['tasks']}
    for task in tasks:
        for row in data[task, 'test']:
            selected.setdefault(task_domains[task], []).append(row)
    return selected


def exact_diagnostic_overlap(directory, training_run, diagnostic_ids):
    comparisons = []
    for domain, identities in diagnostic_ids.items():
        original = [json.loads(line) for line in (training_run/'evaluations/acquired_T19'/f'{domain}_image.jsonl').read_text().splitlines()]
        full = {r['id']: r for r in map(json.loads, (directory/'evaluations/final_full'/f'{domain}_image.jsonl').read_text().splitlines())}
        assert [r['id'] for r in original] == identities
        for row in original:
            actual = full[row['id']]
            assert row['nll'] == actual['nll'] and row['references'] == actual['references']
            if 'prediction' in row:
                assert row['prediction'] == actual['prediction'] and row['scores']['token_f1'] == actual['scores']['token_f1']
        comparisons.append({'domain': domain, 'NLL_exact_QA': len(original),
                            'generation_exact_QA': sum('prediction' in r for r in original)})
    assert sum(r['NLL_exact_QA'] for r in comparisons) == 500
    assert sum(r['generation_exact_QA'] for r in comparisons) == 155
    write(directory/'fresh_process_reload_gate.json', {'passed': True, 'same_frozen_checkpoint': True,
          'all_diagnostic_NLL_exact': True, 'all_diagnostic_predictions_exact': True, 'cases': comparisons})


def run(variant, phase):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    manifest, data, tests, spec = plan()
    assert spec == json.loads((REPORT/'CPU_CONFIRMATION_PLAN.json').read_text())['spec']
    if verify(variant, phase, spec):
        return
    owner = 'control' if phase == 'initial' else variant
    assert stream.verify_complete(owner, spec['training_spec']), 'Requires valid full19 training; failed variants are not scored'
    trained = json.loads((stream.REPORT/(owner+'_complete.json')).read_text());training_run = ROOT/trained['run']
    name = 'shared_initial' if phase == 'initial' else variant+'_'+phase
    pointer = REPORT/(name+'_attempt.json')
    if pointer.exists():
        info = json.loads(pointer.read_text());assert info['spec'] == spec
        directory = ROOT/info['run']
    else:
        directory = ROOT/'exp/onevision_two_module_20261017/confirmation'/name/time.strftime('%Y%m%d_%H%M%S')
        directory.mkdir(parents=True)
        write(pointer, {'run': str(directory.relative_to(ROOT)), 'spec': spec})
        write(directory/'protocol.json', spec)
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    x = stream.Stream(manifest, directory);x.probe_QA_forwards = 0
    first = training_run/('initial_adapter.pt' if phase == 'initial' else 'stages/T19/trained.pt' if phase == 'final' else 'stages/T01/trained.pt')
    x.configure(stream.checkpoint(first)['adapter'], finetune_shared=False);x.model.eval()
    assert not any(p.requires_grad for p in x.model.parameters())
    RepairedExperiment.forward_gate(x)
    states = [0] if phase == 'initial' else [19] if phase == 'final' else list(range(1, 19))
    for stage in states:
        path = training_run/'initial_adapter.pt' if stage == 0 else training_run/'stages'/f'T{stage:02d}'/'trained.pt'
        state = stream.checkpoint(path)
        assert state['spec'] == spec['training_spec']
        x.restore(state['adapter'])
        if stage == 0:
            label, selected = 'base_full', tests
        elif stage == 19:
            label, selected = 'final_full', tests
        else:
            label = f'state_T{stage:02d}_current_next';selected = native_subset(manifest, data, [stage, stage+1])
        write(REPORT/'status.json', {'phase': phase, 'variant': variant, 'state': stage, 'pid': os.getpid(),
                                   'QA': sum(map(len, selected.values())), 'optimizer_updates': 0})
        stream.evaluate(x, label, selected, selected, digest(path), spec)
    if phase == 'final':
        exact_diagnostic_overlap(directory, training_run, spec['training_spec']['diagnostic_test_ids'])
        diagnostic_ids = {i for ids in spec['training_spec']['diagnostic_test_ids'].values() for i in ids}
        rows = [json.loads(line) for d in tests for line in (directory/'evaluations/final_full'/f'{d}_image.jsonl').read_text().splitlines()]
        partitions = {'all21454': rows, 'previously_examined500': [r for r in rows if r['id'] in diagnostic_ids],
                      'other20954': [r for r in rows if r['id'] not in diagnostic_ids]}
        assert [len(r) for r in partitions.values()] == [21454, 500, 20954]
        write(directory/'cohort_scores.json', {name: {'QA': len(rows), 'Answer_F1': sum(r['scores']['token_f1'] for r in rows)/len(rows)} for name, rows in partitions.items()})
    x.monitor.snapshot('training_after', model=x.model);x.monitor.finish(x.model)
    artifacts = [*directory.rglob('*.jsonl'), *directory.rglob('*.json')]
    write(REPORT/(name+'_complete.json'), {'complete': True, 'phase': phase, 'variant': owner, 'spec': spec,
          'run': str(directory.relative_to(ROOT)), 'training_proof_sha256': digest(stream.REPORT/(owner+'_complete.json')),
          'optimizer_updates': 0, 'full_final_QA': 21454 if phase == 'final' else None,
          'artifact_sha256': {str(p.relative_to(directory)): digest(p) for p in artifacts}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--check', action='store_true')
    parser.add_argument('--variant', choices=list(stream.VARIANTS), default='control')
    parser.add_argument('--phase', choices=['initial', 'final', 'sparse'], default='final');args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        if args.check:
            manifest, data, tests, spec = plan()
            all_ids = {r['id'] for rows in tests.values() for r in rows}
            counts = []
            for stage in range(1, 19):
                selected = native_subset(manifest, data, [stage, stage+1]);ids = {r['id'] for rows in selected.values() for r in rows}
                assert ids <= all_ids and len(ids) == sum(map(len, selected.values()))
                assert {r['_task'] for rows in selected.values() for r in rows} == {stage, stage+1}
                counts.append(len(ids))
            assert not torch.cuda.is_initialized()
            write(REPORT/'CPU_CONFIRMATION_PLAN.json', {'passed': True, 'CPU_only': True, 'spec': spec,
                  'sparse_state_QA_counts': counts, 'full_final_QA': 21454, 'GPU_gates_pending': True})
            print(json.dumps({'passed': True, 'full_QA': 21454, 'sparse_states': 18, 'GPU_used': False}))
        else:
            name = 'shared_initial' if args.phase == 'initial' else args.variant+'_'+args.phase
            lock = (REPORT/(name+'.lock')).open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args.variant, args.phase)
    except Exception:
        write(REPORT/(args.variant+'_'+args.phase+'_needs_attention.json'), {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
