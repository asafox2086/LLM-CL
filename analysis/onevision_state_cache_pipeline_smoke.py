"""CPU-only end-to-end plumbing test of the frozen candidate's real driver.

No real model, dataset, prediction, or experimental output is created. Toy
artifacts live only inside a temporary directory, then are removed. This is
not training equivalence, memory measurement, or evidence of Answer F1 gains.
"""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
import torch.nn.functional as F

import onevision_state_cache_trial as target
import onevision_research_device_entry as device

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1'


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj)+'\n')


class CPUOnlyTorch:
    def __getattr__(self, name):
        return getattr(torch, name)
    cuda = SimpleNamespace(reset_peak_memory_stats=lambda: None)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (REPORT/'CPU_PIPELINE_SMOKE.json').exists(), 'Do not repeat unchanged integration evidence'
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='llmcl_cache_cpu_smoke_') as dirname:
        root = Path(dirname).resolve()
        assert root.is_relative_to(Path('/tmp'))
        report = root/'summary';family = root/'exp';native = root/'native'
        report.mkdir();native.mkdir()
        initial = {f'layer{i}': {'A': torch.zeros(1), 'B': torch.zeros(1)} for i in range(48)}
        torch.save({'adapter': initial}, native/'initial_adapter.pt')
        native_proof = {'artifact_sha256': {}}
        for stage in range(1, 20):
            state = {k: {'A': torch.tensor([float(stage)]), 'B': torch.tensor([float(stage)])} for k in initial}
            p = native/f'adapters/sequential_T{stage:02d}.pt';p.parent.mkdir(exist_ok=True)
            torch.save({'adapter': state}, p)
            native_proof['artifact_sha256'][str(p.relative_to(native))] = target.digest(p)
        domains = {'natural': [1], 'medical': list(range(2, 12)), 'geography': list(range(12, 16)),
                   'finance': [16], 'traffic': [17, 18, 19]}
        full_counts = dict(zip(domains, [5000, 2094, 10004, 2233, 2123]))
        data, full, tests, generation = {}, {}, {}, {}
        def row(ident, task):
            return {'id': ident, '_task': task, 'question': 'Toy question '+ident,
                    'answer': 'synthetic', 'answers': ['synthetic'], 'images': [ident+'.png']}
        for stage in range(1, 20):
            count = 1241 if stage == 1 else 4
            data[stage, 'train'] = [row(f'TRAIN:{stage}:{i}', stage) for i in range(count)]
        assert sum(map(len, data.values())) == 1313
        for d, tasks in domains.items():
            full[d] = [row(f'TEST:{d}:{i}', tasks[i % len(tasks)]) for i in range(full_counts[d])]
            tests[d] = full[d][:100];generation[d] = tests[d][:31]
        anchors = {f'TRAIN:{t}:{i}': t for t in [1, 4, 13, 16, 18] for i in range(4)}
        proto_ids = {str(t): [r['id'] for r in data[t, 'train'][:4]] for t in range(1, 20)}
        spec = {'generation_ids': {d: [r['id'] for r in rows] for d, rows in generation.items()},
                'TRAIN_memory_arrivals': anchors, 'prototype_TRAIN_ids': proto_ids,
                'views': list(target.core.VIEWS), 'full_candidates': ['cache_geometry', 'cache_quality'],
                'CPU_TOY_TEST_ONLY': True}
        dump(report/'CPU_PLAN_CHECK.json', {'spec': spec})
        calls = {'train_stages': [], 'evaluations': [], 'resource_boundaries': [], 'full_rows': 0}

        class Monitor:
            def snapshot(self, phase, model=None):
                calls['resource_boundaries'].append(phase)
                return {'gpu_peak_allocated_bytes': 0, 'gpu_peak_reserved_bytes': 0, 'CPU_TOY_ONLY': True}

        class Model:
            def eval(self):
                pass

        class ToyExperiment(target.CacheExperiment):
            def __init__(self, manifest, run):
                self.run = run;self.original_initial = copy.deepcopy(initial)
                self.adapter = copy.deepcopy(initial);self.calling = None;self.active_key = None
                self.monitor = Monitor();self.model = Model();self.cache = {};self.metric_feature_misses = 0
            def configure(self, state, finetune_shared=False):
                self.restore(state)
            def restore(self, state):
                self.adapter = copy.deepcopy(state)
            def train(self, label, rows):
                stage = int(label[-2:]);calls['train_stages'].append(stage)
                self.adapter = {k: {'A': torch.tensor([float(stage)]), 'B': torch.tensor([float(stage)])} for k in initial}
                return copy.deepcopy(self.adapter)
            def feature(self, row):
                self.metric_feature_misses += 1
                v = torch.zeros(1792);v[int(hashlib.sha256(row['question'].encode()).hexdigest(), 16) % 1792] = 1
                return F.normalize(v, dim=0)
            def loss(self, row, view='image', wrong=None):
                self.activate_route(row)
                stage = float(self.adapter['layer0']['A'][0])
                return torch.tensor((stage-row['_task'])**2/100+.1), dict(self.last_route)
            def generate(self, row, view='image', wrong=None):
                self.activate_route(row)
                return {'prediction': 'synthetic', 'scores': {'token_f1': 100., 'rougeL': 100., 'exact_match': 100.},
                        'generation_seconds': 0.}
            def evaluate(self, label, selected, generated, views=('image',)):
                # Use the real inherited evaluator/durable completion wrapper;
                # only the language-model loss/generation above are toy stubs.
                calls['evaluations'].append(label)
                if label == 'final_full_input_router':calls['full_rows'] += sum(map(len, selected.values()))
                return super().evaluate(label, selected, generated, views)

        # Synthetic native-final reference in exactly the real evaluator schema.
        for d, rows in tests.items():
            p = native/f'evaluations/sequential_T19/{d}_image.jsonl';p.parent.mkdir(parents=True, exist_ok=True)
            with p.open('w') as f:
                gids = set(spec['generation_ids'][d])
                for row_ in rows:
                    value = float(torch.tensor((19-row_['_task'])**2/100+.1))
                    rec = {'id': row_['id'], 'native_task': row_['_task'], 'view': 'image', 'nll': value,
                           'references': row_['answers']}
                    if row_['id'] in gids:
                        rec.update(prediction='synthetic', scores={'token_f1': 100., 'rougeL': 100., 'exact_match': 100.}, generation_seconds=0.)
                    f.write(json.dumps(rec)+'\n')
        node, namespace = device.compiled_function(target, 'run_trial', 0)
        namespace.update(ROOT=root, REPORT=report, FAMILY=family, CacheExperiment=ToyExperiment,
            plan=lambda: (spec, ({}, data, domains, tests, full, native, native_proof)),
            torch=CPUOnlyTorch(), os=SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES': '0'}, getpid=os.getpid),
            reset_seed=lambda: None,
            finish_resources=lambda x, run, before, after, scope: dump(run/'memory_summary.json', {'CPU_TOY_ONLY': True}))
        complete_node = next(n for n in ast.parse(Path(target.__file__).read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == 'complete')
        exec(compile(ast.Module(body=[complete_node, node], type_ignores=[]), str(Path(__file__)), 'exec'), namespace)
        namespace['run_trial']('build')
        assert calls['train_stages'] == list(range(1, 20))
        assert calls['evaluations'] == list(target.core.VIEWS)
        proof = json.loads((report/'build_complete.json').read_text());run = root/proof['run']
        state = torch.load(run/'final_state.pt', map_location='cpu', weights_only=True)
        assert 19 in state['pool'] and 1 < len(state['pool']) <= 3
        for view in target.core.VIEWS:
            assert namespace['complete']('build', None, spec)
        for view in spec['full_candidates']:
            namespace['run_trial']('full', view)
            assert namespace['complete']('full', view, spec)
        assert calls['full_rows'] == 2*21454
        assert calls['evaluations'][-4:] == ['candidate_reload', 'final_full_input_router']*2
        assert all(f'pure_TRAIN_T{t:02d}_before' in calls['resource_boundaries'] and
                   f'pure_TRAIN_T{t:02d}_after' in calls['resource_boundaries'] for t in range(1, 20))
        evidence = {'CPU_only': True, 'real_model_or_public_QA_used': False, 'GPU_operations': 0,
            'driver_build_19_and_all4_views_completed_with_toy_components': True,
            'both_full_branches_21454_each_and_exact_reload_exercised': True,
            'all19_pure_TRAIN_boundary_calls_exercised': True,
            'snapshot_artifact_and_completion_verifiers_exercised': True,
            'source_sha256': {str(Path(__file__).relative_to(ROOT)): target.digest(Path(__file__)),
                              str(Path(target.__file__).relative_to(ROOT)): target.digest(Path(target.__file__)),
                              str(Path(device.__file__).relative_to(ROOT)): target.digest(Path(device.__file__))},
            'not_model_performance_or_GPU_equivalence_or_memory_evidence': True,
            'toy_artifacts_not_saved_in_real_experiment_or_repository': True}
    dump(REPORT/'CPU_PIPELINE_SMOKE.json', evidence)
    print(json.dumps(evidence))


if __name__ == '__main__':
    main()
