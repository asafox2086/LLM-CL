"""TRAIN-only FOREVER port preflight; no model/TEST/GPU/training execution.

Checks the unchanged author's schedule boundaries and executes the exact
author memory selector with a minimal Dataset column/selection bridge. This
is implementation evidence, not an accuracy or forgetting result.
"""
import ast
import contextlib
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_preflight_20261018'


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def author_source(relative):
    folder = ROOT/'code/FOREVER'
    expected = next(row['sha256'] for row in json.loads((folder/'files_sha256.json').read_text())
                    if row['path'] == relative)
    path = folder/'source'/relative
    raw = path.read_bytes()
    assert sha_bytes(raw) == expected, 'Pinned author source mismatch'
    return path, raw.decode('utf-8'), expected


def expression(node, namespace):
    return eval(compile(ast.Expression(node), '<unchanged-author-expression>', 'eval'), namespace)


class DatasetBridge:
    """Only the Dataset operations called by the original selector."""
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, column):
        assert column == 'output'
        # Exact canonical answer used for training; no answer normalization.
        return [row['answer'] for row in self.rows]

    def select(self, indices):
        return DatasetBridge([self.rows[i] for i in indices])


def main():
    start = time.monotonic()
    policy = json.loads((ROOT/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json').read_text())
    path = ROOT/policy['conditions']['pct01']['manifest']
    raw = path.read_bytes()
    assert sha_bytes(raw) == policy['conditions']['pct01']['manifest_sha256']
    manifest = json.loads(raw)
    assert manifest['seed'] == 42 and len(manifest['tasks']) == 19
    train = []
    data_sha = {}
    for task in manifest['tasks']:
        f = task['files']['train']
        content = (ROOT/f['path']).read_bytes()
        assert sha_bytes(content) == f['sha256']
        data_sha[f['path']] = f['sha256']
        rows = [json.loads(line) for line in content.splitlines()]
        assert len(rows) == f['count'] and all(isinstance(row['answer'], str) for row in rows)
        train.append(rows)
    assert sum(map(len, train)) == 1313
    source_path, source, source_sha = author_source('src/finetune_cl_our_qwen.py')
    _, selector_source, selector_sha = author_source('utils/load_data.py')
    parsed = ast.parse(source)
    function = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == 'train')
    defaults = {arg.arg: ast.literal_eval(value) for arg, value in
                zip(function.args.args[-len(function.args.defaults):], function.args.defaults)
                if isinstance(value, (ast.Constant, ast.List, ast.Tuple))}
    assert defaults['steps_per_day'] == 24 and defaults['memory_data_ratio'] == 2
    assert defaults['memory_epochs'] == 2
    loop = next(node for node in function.body if isinstance(node, ast.While))
    calibration = next(node for node in loop.body if isinstance(node, ast.If) and
                       ast.unparse(node.test) == 'not calibrated')
    calib_steps = next(node.value for node in calibration.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'calib_steps' for target in node.targets))
    assert ast.unparse(calib_steps) == 'min(steps_per_day, remaining_steps)'
    assert isinstance(calibration.body[-1], ast.Continue)
    assert ast.unparse(loop.test) == 'steps_done < total_steps_current'
    trigger = next(node for node in loop.body if isinstance(node, ast.If) and
                   'current_tau >=' in ast.unparse(node.test))
    # Verify the real trigger block follows the calibration/continue branch.
    assert loop.body.index(trigger) > loop.body.index(calibration)
    final_condition = next(node for node in function.body if isinstance(node, ast.If) and
                           ast.unparse(node.test) == 'memory_data is None')
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
               isinstance(n.func.value, ast.Name) and n.func.value.id == 'final_trainer_memory'
               and n.func.attr == 'train' for n in ast.walk(final_condition))
    # Boundary controls use the exact author's expressions. Unit drift and a
    # trainer consuming its requested max_steps are assumptions of this CPU
    # control, not an emulation of full HF optimization or measured model tau.
    boundary = []
    for total in [1, 2, 22, 24, 25, 48]:
        namespace = {'steps_done': 0, 'total_steps_current': total, 'steps_per_day': 24,
                     'remaining_steps': total, 'min': min}
        assert expression(loop.test, namespace)
        count = expression(calib_steps, namespace)
        namespace['steps_done'] = count
        after = expression(loop.test, namespace)
        trigger_namespace = {'memory_data': object(), 'next_trigger_idx': 0,
                             'trigger_thresholds': [float(count)], 'current_tau': float(count)}
        trigger_if_reentered = bool(expression(trigger.test, trigger_namespace))
        boundary.append({'total_updates': total, 'calibration_updates': count,
                         'while_reentered_after_calibration': after,
                         'first_threshold_reached_but_branch_unvisited': trigger_if_reentered and not after,
                         'immediate_replay_branch_reachable': trigger_if_reentered and after})
    assert all(not row['immediate_replay_branch_reachable'] for row in boundary[:4])
    assert all(row['immediate_replay_branch_reachable'] for row in boundary[4:])
    memory_function = next(node for node in ast.parse(selector_source).body
                           if isinstance(node, ast.FunctionDef) and node.name == 'load_memory_buffer')
    virtual = {os.path.join('/frozen_train', 'train', f'T{i:02d}.json'): DatasetBridge(rows)
               for i, rows in enumerate(train, 1)}
    namespace = {'Counter': Counter, 'random': random,
                 'os': SimpleNamespace(path=SimpleNamespace(join=os.path.join, exists=lambda p: p in virtual)),
                 'get_dataset_order': lambda _: [f'T{i:02d}' for i in range(1, 20)],
                 'load_dataset': lambda kind, data_files, cache_dir: {'train': virtual[data_files]},
                 'concatenate_datasets': lambda values: DatasetBridge([row for value in values for row in value.rows])}
    exec(compile(ast.Module(body=[memory_function], type_ignores=[]), str(ROOT/'code/FOREVER/source/utils/load_data.py'), 'exec'), namespace)
    profile = []
    for i, rows in enumerate(train):
        updates = math.ceil(len(rows)/16)
        assert 1 <= updates <= 22 and updates <= defaults['steps_per_day']
        item = {'task': i+1, 'TRAIN_QA': len(rows), 'main_updates': updates,
                'calibration_consumes_all_main_updates': True, 'dynamic_replay_reachable_before_main_end': False,
                'prior_TRAIN_QA': sum(map(len, train[:i]))}
        if i:
            with contextlib.redirect_stdout(io.StringIO()):
                selected = namespace['load_memory_buffer'](1, i, '/frozen_train', 2, '/no_cache', 'qwen', 42)
            identifiers = [row['id'] for row in selected.rows]
            prior = {row['id'] for task_rows in train[:i] for row in task_rows}
            assert set(identifiers).issubset(prior) and len(identifiers) == len(set(identifiers))
            target = sum(int(len(old)*.02) for old in train[:i])
            item.update(author_memory_QA=len(selected), nominal_floor_2pct_QA=target,
                        selected_fraction=len(selected)/len(prior),
                        selected_TRAIN_IDs_sha256=sha_bytes(json.dumps(identifiers).encode()),
                        exact_primary_answer_classes=[len(Counter(row['answer'] for row in old)) for old in train[:i]])
        profile.append(item)
    evidence = {'TRAIN_only': True, 'GPU_forward': 0, 'model_loaded': False, 'TEST_predictions_read': 0,
                'optimizer_updates_executed': 0, 'author_algorithm_changed': False,
                'upstream_revision': json.loads((ROOT/'code/FOREVER/source_manifest.json').read_text())['revision'],
                'source_sha256': {'src/finetune_cl_our_qwen.py': source_sha, 'utils/load_data.py': selector_sha},
                'manifest_sha256': sha_bytes(raw), 'TRAIN_sha256': data_sha,
                'shared_main_epochs': 1, 'shared_batch_size': 16, 'author_steps_per_day': 24,
                'author_memory_ratio_percent': 2, 'author_memory_epochs': 2,
                'total_main_updates': sum(row['main_updates'] for row in profile),
                'max_main_updates_per_stage': max(row['main_updates'] for row in profile),
                'all_19_stages_no_main_dynamic_trigger': True,
                'final_replay_present_in_author_source': True, 'profile': profile,
                'schedule_boundary_controls': boundary, 'elapsed_seconds': time.monotonic()-start,
                'limitations': ['Shared1epoch port preflight, not the author60/10epoch experiments.',
                  'Schedule reachability assumes a fresh phase and actual trainer consumes requested max_steps.',
                  'Dataset bridge maps exact training answer to author output column; HF dataset I/O is not executed.',
                  'Selector counts are exact for these TRAIN inputs; no TEST accuracy/forgetting claim.',
                  'No comparison of trained FOREVER versus other methods is completed.',
                  'No change to active baseline/research code, source, queue, data or parameters.']}
    REPORT.mkdir(parents=True, exist_ok=True)
    target = REPORT/'EVIDENCE.json'
    assert not target.exists(), 'Preserve first audit; use a new traceable revision for changes'
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'CPU_only': True, 'all_stages_dynamic_trigger_unreachable': True,
                      'main_updates': evidence['total_main_updates'], 'max_stage_updates': 22,
                      'last_stage_memory_QA': profile[-1]['author_memory_QA'],
                      'last_stage_nominal_2pct': profile[-1]['nominal_floor_2pct_QA'],
                      'last_stage_prior_TRAIN': profile[-1]['prior_TRAIN_QA'],
                      'boundary_checks': len(boundary), 'evidence': str(target.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
