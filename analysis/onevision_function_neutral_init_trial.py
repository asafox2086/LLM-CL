"""Three predeclared function-neutral initialization diagnostics, not a CL method."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import fcntl
import json
from pathlib import Path
import time
import torch
import onevision_initialization_source_controls as source
from onevision_function_neutral_init_core import NeutralLoRA
from onevision_initialization_source_geometry import gradient_values
from onevision_knowledge_intervention import reset_seed, case_complete
from onevision_research_worker import atomic_torch, digest
from onevision_capacity_probe import finish_resources
from run import write

ROOT = source.ROOT
REPORT = source.parent.BASE/'function_neutral_init_20261018/v1'
FAMILY = ROOT/'exp/onevision_function_neutral_init_20261018/v1'
CASES = ['cold_compensated', 'within_compensated', 'cross_compensated']


def read(p):
    return json.loads(p.read_text())


def plan():
    old, values = source.plan()
    assert read(source.REPORT/'CPU_PLAN_CHECK.json')['spec'] == old
    checkpoint = read(source.REPORT/'sources_complete.json')
    assert source.complete('sources', None, old) and source.complete('case', 0, old)
    assert checkpoint['spec'] == old
    sources = dict(old['source_sha256'])
    names = ['onevision_function_neutral_init_core.py', 'onevision_function_neutral_init_trial.py',
             'onevision_function_neutral_init_entry.py','onevision_initialization_source_geometry.py']
    sources.update({'analysis/'+n:digest(ROOT/'analysis'/n) for n in names})
    return dict(parent_spec=old, source_sha256=sources, cases=CASES,
        source_completion_sha256=digest(source.REPORT/'sources_complete.json'),
        original_cold_completion_sha256=digest(source.REPORT/'cold_complete.json'),
        target_TRAIN_QA=26, target_updates=2, target_full_TEST_QA=507,
        reference_parameters_per_condition=540672, reference_FP32_bytes=2162688,
        trainable_parameters=540672, effective_deltaW='4*(B_t*A_t-B_ref*A_ref)',
        all3_final507=True, initial32_exact_original_cold_required=True,
        cold_two_updates_and_final507_exact_original_required=True,
        all5_original_source_controls_finish_first=True, no_TEST_selection=True), values


def references(x):
    return {k:dict(A=m.reference_A.detach().cpu().clone(), B=m.reference_B.detach().cpu().clone())
            for k,m in x.layers.items()}


def install(x):
    for key, old in list(x.layers.items()):
        m = NeutralLoRA(old)
        parent, name = key.rsplit('.', 1)
        setattr(x.model.get_submodule(parent), name, m)
        x.layers[key] = m
    assert sum(p.numel() for p in x.model.parameters() if p.requires_grad) == 540672
    assert sum(m.reference_A.numel()+m.reference_B.numel() for m in x.layers.values()) == 540672


def complete(index, spec):
    pointer = REPORT/(CASES[index]+'_complete.json')
    if not pointer.exists():
        return False
    proof = read(pointer)
    assert proof['complete'] and proof['spec'] == spec and proof['target_full_TEST_QA'] == 507
    run = ROOT/proof['run']
    assert run.resolve().is_relative_to(FAMILY.resolve())
    for rel, expected in proof['artifact_sha256'].items():
        p = run/rel
        assert p.resolve().is_relative_to(run.resolve()) and digest(p) == expected
    assert read(run/'actual_function_and_update_gate.json')['passed']
    return True


def run_trial(index):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['0', '1']
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    spec, values = plan()
    assert read(REPORT/'CPU_PLAN_CHECK.json')['spec'] == spec
    assert read(REPORT/'ADMISSION.json')['spec'] == spec
    assert all(source.complete('case', i, spec['parent_spec']) for i in range(5))
    if index:
        assert complete(0, spec), 'Actual cold implementation control must complete first'
    lock = (REPORT/'worker.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    name = CASES[index]
    assert not (REPORT/(name+'_attempt.json')).exists(), 'Preserve failures; no blind retry'
    manifest, data, native, native_proof, within, cross, probe = values
    run = FAMILY/name/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
    write(REPORT/(name+'_attempt.json'), dict(spec=spec,run=str(run.relative_to(ROOT)),pid=os.getpid()))
    write(run/'spec.json', spec)
    import onevision_research_device_entry as device
    cards = device.inventory();gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert device.select(cards, os.getuid()) == gpu
    write(run/'physical_device_execution.json',dict(physical_GPU=gpu,admission_inventory=cards,
        actual_CUDA_VISIBLE_DEVICES=str(gpu),single_research_stream=True,max_our_GPU_cards=3))
    x = source.parent.MetricExperiment(manifest, run)
    old_cold = ROOT/read(source.REPORT/'cold_complete.json')['run']
    start = torch.load(old_cold/'initial_adapter.pt', map_location='cpu', weights_only=True)['adapter']
    if index:
        source_run = ROOT/read(source.REPORT/'sources_complete.json')['run']
        start = torch.load(source_run/(name.split('_')[0]+'.pt'),map_location='cpu',weights_only=True)['adapter']
    x.configure(start, finetune_shared=True)
    install(x)
    assert source.same(x.snapshot(), start) and source.same(references(x), start)
    x.monitor.snapshot('control_before', model=x.model)
    atomic_torch(run/'initial_adapter.pt',dict(adapter=start))
    atomic_torch(run/'reference_adapter.pt',dict(adapter=references(x)))
    # Run the actual official/packed visual gate again with the NEW wrappers.
    x.model.eval();x.forward_gate()
    selected = {'medical':probe};x.evaluate('initial_probe', selected, selected)
    assert case_complete(run, 'initial_probe', selected)
    assert source.semantic(run/'evaluations/initial_probe/medical_image.jsonl') == source.semantic(
        old_cold/'evaluations/initial_probe/medical_image.jsonl')
    # Do not call configure here: it would replace the new wrappers.
    after = source.fit(x, run, 'target_T04', data[4,'train'])
    assert not source.same(after, start) and source.same(references(x), start)
    assert all(not p.requires_grad and p.grad is None
               for m in x.layers.values() for p in [m.reference_A,m.reference_B])
    gradients = []
    for step in [1,2]:
        here = run/f'raw_update_gradients/target_T04/step{step}.pt'
        raw = torch.load(here, map_location='cpu', weights_only=True)
        g = gradient_values(raw);gradients.append(g)
        assert g['preclip_B_gradient_l2'] > 0
        if index:
            assert g['preclip_A_gradient_l2'] > 0
        else:
            original = torch.load(old_cold/f'raw_update_gradients/target_T04/step{step}.pt',
                                  map_location='cpu', weights_only=True)
            assert raw.keys() == original.keys()
            assert raw['layout'] == original['layout'] and raw['train_ids'] == original['train_ids']
            assert torch.equal(raw['preclip_gradient'], original['preclip_gradient'])
    if index == 0:
        original = torch.load(old_cold/'adapters/target_T04.pt',map_location='cpu',weights_only=True)['adapter']
        assert source.same(after, original)
        def events(folder):
            return [{k:r[k] for k in ['loss','gradient_l2_preclip','learning_rate_used','qa_seen']}
                    for r in [json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
                    if r.get('event')=='train' and r.get('label')=='target_T04']
        assert events(run) == events(old_cold)
    atomic_torch(run/'function_neutral_state.pt',dict(adapter=after, reference=references(x),
        condition=name, scale=4., frozen_base_reference=spec['parent_spec']['native_initial_source']))
    full = {'medical':data[4,'test']};x.evaluate('target_full507', full, full)
    assert case_complete(run,'target_full507',full)
    if index == 0:
        assert source.semantic(run/'evaluations/target_full507/medical_image.jsonl') == source.semantic(
            old_cold/'evaluations/target_full507/medical_image.jsonl')
    assert source.same(references(x),start)
    write(run/'actual_function_and_update_gate.json',dict(passed=True,initial32_exact_original_cold=True,
        cold_actual_gradients_adapter_events_final507_exact=index==0,
        reference_unchanged=True,raw_gradients=gradients))
    x.monitor.snapshot('control_after',model=x.model)
    finish_resources(x,run,'control_before','control_after',
        'samePID includes new renderer/reload/fullTEST; pure_TRAIN windows separately reset and saved')
    files = {str(p.relative_to(run)):digest(p) for p in run.rglob('*')
             if p.is_file() and p.suffix in ['.json','.jsonl','.pt']}
    write(REPORT/(name+'_complete.json'),dict(complete=True,spec=spec,condition=name,
        run=str(run.relative_to(ROOT)),artifact_sha256=files,target_full_TEST_QA=507,
        source_history_exposures_charged=0 if index==0 else 12,source_history_updates_charged=0 if index==0 else 2,
        diagnostic_not_five_domain_method=True))
    assert complete(index,spec)


def main(check=False,index=None):
    REPORT.mkdir(parents=True,exist_ok=True)
    if check:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        spec,_ = plan()
        write(REPORT/'CPU_PLAN_CHECK.json',dict(CPU_only=True,spec=spec,
            all3_required=True,all5_originals_finish_first=True,GPU_started=False))
    else:
        assert index in range(3)
        run_trial(index)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--case-index',type=int)
    a=p.parse_args();main(a.check,a.case_index)
