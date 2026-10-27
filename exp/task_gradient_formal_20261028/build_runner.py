"""Build the independent TaskGradReuse pct01 runner from actual formal sources.

Stdlib only: copy/AST/metadata operations, never import a runner or model. The
candidate adapter is independently authored and is never copied or overwritten.
The final scientific binding separately freezes it and this builder. All copied
destination source files must be absent before this build.
"""
import argparse
import ast
import copy
import difflib
import hashlib
import json
import math
from pathlib import Path
import time


NAME = 'task_gradient_formal_20261028'
METHOD = 'TaskGradReuse'
MANIFEST = 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
REFERENCE_INDEX_SHA = 'd1fe293eece1893ddb3151dbedc3de4e6297d129a184a04e34972bf93df29cca'
TEMPLATE_SHA = {
    'run.py': 'ea347e655c8f3a20d07c86568b351ab1615868bcf9b88d209756c45d1cef1280',
    'checkpoints.py': '83ca50cc9365d36b90af8b8472d75a1f9f7c798e5bb1d440f87398c4d3e3a591',
    'pipeline.py': '2b18ce445a01d76b2f065100752c05626bdaf800dd33f47aa8c6e75ff52f9d2d',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')


def exact(raw, before, after):
    assert raw.count(before) == 1, before[:180]
    result = raw.replace(before, after, 1)
    assert result.count(after) == 1 and result.replace(after, before, 1) == raw
    return result


def between(raw, start, stop):
    assert raw.count(start) == 1 and raw.count(stop) == 1
    return raw[raw.index(start):raw.index(stop)]


def function_bytes(raw, name):
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return b''.join(raw.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])


def ast_hash(node):
    return sha(ast.dump(node, include_attributes=False).encode())


def training_loop(raw):
    main = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    return next(n for n in main.body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'stage')


def transplant_run(raw, template):
    """Precisely reconstruct the reviewed stage-I/O template from the reference."""
    helper = function_bytes(template, 'pause_at_boundary')
    paused = between(template, b'    paused = None\n', b"    (run / 'logs').mkdir")
    resume_check = between(template,
        b"        assert not resume_obj['config']['preflight_only']",
        b'        # The first interrupted stage')
    initial_pause = between(template, b'    if resume_obj:\n        # Loading, rebinding',
        b"    for stage in range(resume_obj['stage']+1")
    edits = [
        (b'from runtime_utils import input_batch, answer_loss, evaluate, make_optimizer, activation_storage, write\n',
         b'from runtime_utils import input_batch, answer_loss, evaluate, make_optimizer, activation_storage, write, restore_rng\n'),
        (b'def main():\n', helper + b'\n\ndef main():\n'),
        (b"    (run / 'logs').mkdir(parents=True, exist_ok=True)\n",
         paused + b"    (run / 'logs').mkdir(parents=True, exist_ok=True)\n"),
        (b"'checkpoint_all_stages': args.method in policy['checkpoint_all_stages_methods'],\n",
         b"'checkpoint_all_stages': True,\n              'checkpoint_retention': 'all_completed_stages',\n"),
        (b"        assert config['checkpoint_all_stages']\n",
         b"        assert config['checkpoint_all_stages']\n" + resume_check),
        (b"    write(run / 'run_status.json', {'status': 'initial_evaluation', 'completed_stage': 0})\n",
         b"    write(run / 'run_status.json', {'status': 'resuming' if resume_obj else 'initial_evaluation',\n          'completed_stage': resume_obj['stage'] if resume_obj else 0})\n"),
        (b"    if not resume_obj: monitor.snapshot('training_before', model=model)\n",
         b"    if not resume_obj: monitor.snapshot('training_before', model=model)\n" + initial_pause),
        (b"        write(run / 'run_status.json', {'status': 'stage_complete', 'completed_stage': stage})\n",
         b"        write(run / 'run_status.json', {'status': 'stage_complete', 'completed_stage': stage})\n"
         b'        if pause_at_boundary(method, processor, run, stage, config, monitor, checkpoint):\n            return\n'),
    ]
    result = raw
    for before, after in edits:
        assert after in template
        result = exact(result, before, after)
    assert result == template, 'Unexpected changes in the stage checkpoint run template'
    return result


def transplant_checkpoints(raw, template):
    result = exact(raw,
        b'def save(method, processor, run, stage, config, probe=None):\n',
        function_bytes(template, 'save') + b'\n\ndef _save(method, processor, run, stage, config, probe, state):\n')
    result = exact(result, b"'rng':rng(),'probe':probe,", b"'rng':state,'probe':probe,")
    assert result == template, 'Checkpoint changes must only isolate save-time RNG'
    assert function_bytes(raw, 'load') == function_bytes(result, 'load')
    return result


def candidate_run(raw):
    tree = ast.parse(raw)
    declaration = next(n for n in tree.body if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'METHODS' for t in n.targets))
    old = b''.join(raw.splitlines(keepends=True)[declaration.lineno - 1:declaration.end_lineno])
    result = exact(raw, old, b"METHODS = ['TaskGradReuse']\n")
    result = exact(result, b"choices=['full', 'pct01']", b"choices=['pct01']")
    return exact(result, b"    backbone = host_policy['backbone']\n",
        b"    backbone = host_policy['backbone']\n    assert backbone == 'onevision', 'This fixed candidate is LLaVA-OneVision only'\n")


def method_registry(raw):
    anchor = b'def make_method(model,name,instructions,embedding=None):\n'
    return exact(raw, anchor, anchor +
        b"    if name=='TaskGradReuse':\n"
        b'        from task_gradient_adapter import TaskGradReuseMethod\n'
        b'        return TaskGradReuseMethod(model,name,instructions,embedding)\n')


def manifest_plan(raw):
    assert sha(raw) == MANIFEST_SHA
    manifest = json.loads(raw)
    assert manifest['condition'] == 'pct01' and manifest['seed'] == 42
    assert manifest['formal_ready'] and not manifest['assets_pending']
    tasks = manifest['tasks']
    assert [t['task'] for t in tasks] == list(range(1, 20))
    assert sum(t['files']['train']['count'] for t in tasks) == 1313
    assert sum(t['files']['test']['count'] for t in tasks) == 21454
    plan = []
    for task in tasks:
        count = task['files']['train']['count']
        updates = math.ceil(count / 16)
        warmup = min(math.floor(.03 * updates), updates - 1)
        assert warmup == 0
        plan.append(dict(stage=task['task'], train_QA=count, optimizer_calls=updates,
                         warmup_updates=warmup, positive_lr_updates=updates,
                         calibration_QA=min(16, count),
                         training_order_sha256=task['training_order_sha256']))
    cells = [(stage, task) for stage in range(20)
             for task in (range(1, 20) if stage in (0, 19) else [stage, stage + 1])]
    assert len(cells) == 74
    assert sum(tasks[t - 1]['files']['test']['count'] for _, t in cells) == 80316
    assert sum(p['optimizer_calls'] for p in plan) == 93
    assert sum(p['calibration_QA'] for p in plan) == 222
    return plan


def prepare(reference, template_dir, manifest_raw):
    index_raw = (reference / 'SOURCE_SHA256.json').read_bytes()
    assert sha(index_raw) == REFERENCE_INDEX_SHA
    expected = json.loads(index_raw)
    assert len(expected) == 26 and sum(n.endswith('.py') for n in expected) == 25
    source = {}
    for name, wanted in expected.items():
        assert Path(name).name == name
        raw = (reference / name).read_bytes()
        assert sha(raw) == wanted, name
        source[name] = raw
    templates = {n: (template_dir / n).read_bytes() for n in TEMPLATE_SHA}
    assert {n: sha(raw) for n, raw in templates.items()} == TEMPLATE_SHA
    destination = dict(source)
    destination['run.py'] = candidate_run(transplant_run(source['run.py'], templates['run.py']))
    destination['methods.py'] = method_registry(source['methods.py'])
    destination['checkpoints.py'] = transplant_checkpoints(source['checkpoints.py'], templates['checkpoints.py'])
    # This is the existing explicit-pause pipeline, not a new allocator. Only its
    # source is copied; this builder never invokes pipeline.main or launch.
    destination['pipeline.py'] = templates['pipeline.py']
    plan = manifest_plan(manifest_raw)
    policy = json.loads(source['execution_policy.json'])
    assert policy['conditions']['pct01']['manifest'] == MANIFEST
    assert policy['conditions']['pct01']['manifest_sha256'] == MANIFEST_SHA
    policy.update(experiment_id=NAME, methods=[METHOD], checkpoint_all_stages_methods=[METHOD],
                  checkpoint_other_methods='all completed stages retained; final must be retained')
    policy['conditions'] = {'pct01': policy['conditions']['pct01']}
    policy['hosts'] = {n: v for n, v in policy['hosts'].items() if v['backbone'] == 'onevision'}
    for host in policy['hosts'].values():
        host['conditions'] = {'pct01': host['conditions']['pct01']}
    policy['ownership'] = {n: ['pct01:' + METHOD] for n in policy['hosts']}
    for key in ('future_full', 'full_label_exclusion', 'scope_revision_20261018'):
        policy.pop(key, None)
    destination['execution_policy.json'] = encoded(policy)
    for name, raw in destination.items():
        if name.endswith('.py'):
            compile(ast.parse(raw), name, 'exec')
    old_loop = training_loop(source['run.py'])
    new_loop = copy.deepcopy(training_loop(destination['run.py']))
    pause_tail = new_loop.body.pop()
    assert ast.dump(pause_tail, include_attributes=False) == ast.dump(
        ast.parse('if pause_at_boundary(method, processor, run, stage, config, monitor, checkpoint):\n    return').body[0],
        include_attributes=False)
    assert ast_hash(new_loop) == ast_hash(old_loop), 'Original stage training/evaluation changed'
    changed = sorted(n for n in source if destination[n] != source[n])
    assert changed == ['checkpoints.py', 'execution_policy.json', 'methods.py', 'pipeline.py', 'run.py']
    receipt = dict(schema='task-gradient-formal-source-build-v1', experiment_id=NAME,
        reference_source_index_sha256=REFERENCE_INDEX_SHA,
        reference_sources=expected, stage_io_template_sha256=TEMPLATE_SHA,
        destination_copied_sources={n: sha(raw) for n, raw in destination.items()},
        changed_files=changed,
        byte_identical_reference_files=sorted(set(source) - set(changed)),
        differences={n: ''.join(difflib.unified_diff(source[n].decode().splitlines(True),
                    destination[n].decode().splitlines(True), fromfile='reference/' + n, tofile=NAME + '/' + n))
                    for n in changed},
        original_stage_loop_ast_sha256=ast_hash(old_loop),
        copied_stage_loop_without_pause_ast_sha256=ast_hash(new_loop),
        original_stage_training_and_evaluation_AST_unchanged=True,
        runtime_loss_inputs_optimizer_evaluation_resources_byte_identical=True,
        original_checkpoints_load_byte_identical=True,
        checkpoint_save_changes='existing RNG-isolated save wrapper; full original method graph; no pruning',
        pause_resume_changes='existing stage-checkpoint template; explicit receipt-bound permit; no automatic failure retry',
        framework='shared-v2-uniform-input-and-cache', method=METHOD, condition='pct01',
        backbone='onevision', fresh_W0_required=True, prefix_or_partial_CP_reuse=False,
        manifest_path=MANIFEST, manifest_sha256=MANIFEST_SHA, stages=19,
        TRAIN_QA=1313, TEST_QA=21454, original_required_cells=74, original_scored_QA=80316,
        planned_main_optimizer_calls=93, planned_main_positive_lr_updates=93,
        planned_calibration_backwards=222, planned_main_backwards=1313,
        training_plan=plan, original_reference_modified=False,
        models_loaded=0, jobs_launched=0, new_scores=0)
    return destination, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--template-dir', type=Path)
    parser.add_argument('--manifest', type=Path, help='Existing original manifest metadata, never TRAIN/TEST rows')
    args = parser.parse_args()
    assert args.root.is_absolute() and args.root.is_dir()
    root = args.root.resolve()
    reference = (args.reference_dir or root / '.deployment/task_gradient_formal_reference_20261028').resolve()
    templates = (args.template_dir or root / 'exp/two_backbone_native_cl_20261017_v2').resolve()
    manifest = (args.manifest or root / MANIFEST).resolve()
    target = root / 'exp' / NAME
    assert target.is_dir(), 'Keep this builder and the independent adapter in the separate target directory'
    started = time.monotonic()
    destination, receipt = prepare(reference, templates, manifest.read_bytes())
    assert all(not (target / n).exists() for n in destination), 'Never overwrite an existing copied runner'
    receipt_path = target / 'BUILD_RECEIPT.json'
    assert not receipt_path.exists()
    # All transformations and metadata validation precede source creation.
    for name, raw in destination.items():
        with (target / name).open('xb') as stream:
            stream.write(raw)
    receipt.update(unix=time.time(), seconds=time.monotonic() - started,
        builder_sha256=sha(Path(__file__).read_bytes()), reference=str(reference),
        template_directory=str(templates), manifest_metadata_source=str(manifest), target=str(target),
        independent_adapter=dict(path='task_gradient_adapter.py', copied_or_modified=False,
            registry_class='TaskGradReuseMethod', final_source_binding='separate final scientific binding'),
        source_construction_only=True, candidate_model_admission=False,
        destination_py_sha256={n: sha(raw) for n, raw in destination.items() if n.endswith('.py')})
    with receipt_path.open('xb') as stream:
        stream.write(encoded(receipt))
    print(json.dumps(dict(complete=True, receipt=str(receipt_path), sha256=sha(receipt_path.read_bytes()),
                         models_loaded=0, jobs_launched=0, new_scores=0)))


if __name__ == '__main__':
    main()
