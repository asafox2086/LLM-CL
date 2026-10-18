"""Versioned, metadata-only repair of the metric feature auxiliary row.

The original frozen feature body is compiled with exactly one extra dict item:
id=row['id']. ID is used only by image-token repair logging. Question, image,
feature cache key, router and all parameter/optimizer code remain unchanged.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPAIR = ROOT/'summary/onevision_diagnostics_20261016/feature_identity_repair_20261018'
RUNS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feature_node():
    source = ROOT/'analysis/onevision_learned_reuse_trial.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MetricExperiment')
    old = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'feature')
    new = copy.deepcopy(old)
    dictionaries = [n.value for n in new.body if isinstance(n, ast.If) for n in n.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'aux' for t in n.targets)]
    assert len(dictionaries) == 1
    d = dictionaries[0]
    assert isinstance(d, ast.Dict) and [k.value for k in d.keys] == ['_task', 'question', 'images']
    d.keys.append(ast.Constant('id'))
    d.values.append(ast.Subscript(value=ast.Name(id='row', ctx=ast.Load()), slice=ast.Constant('id'), ctx=ast.Load()))
    reverted = copy.deepcopy(new)
    d2 = next(n.value for n in ast.walk(reverted) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'aux' for t in n.targets))
    d2.keys.pop(); d2.values.pop()
    assert ast.dump(reverted, include_attributes=False) == ast.dump(old, include_attributes=False)
    return old, ast.fix_missing_locations(new), sha(source)


def compiled_feature(namespace, repaired=True):
    old, new, source_sha = feature_node()
    values = dict(namespace)
    exec(compile(ast.Module(body=[new if repaired else old], type_ignores=[]), str(Path(__file__)), 'exec'), values)
    return values['feature'], source_sha


def install(gpu_gate=True):
    import onevision_learned_reuse_trial as trial
    original_feature = trial.MetricExperiment.feature
    repaired, source_sha = compiled_feature(original_feature.__globals__)
    repaired.__module__ = original_feature.__module__
    repaired.__qualname__ = original_feature.__qualname__
    trial.MetricExperiment.feature = repaired
    original_init = trial.MetricExperiment.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        receipt = {'version': 'feature_identity_v1', 'metadata_only': True,
                   'original_feature_source_sha256': source_sha,
                   'repair_source_sha256': sha(Path(__file__)),
                   'entrypoint_sha256': sha(ROOT/'analysis/onevision_identity_repair_worker.py'),
                   'algorithm_spec_and_model_inputs_unchanged': True,
                   'old_partial_predictions_reused': False, 'optimizer_updates': 0}
        if gpu_gate:
            import torch
            from run import freeze
            assert torch.cuda.current_device() == 0  # CUDA_VISIBLE_DEVICES=1
            assert __import__('os').environ['CUDA_VISIBLE_DEVICES'] == '1'
            before_rng = torch.get_rng_state().clone()
            before_cuda_rng = torch.cuda.get_rng_state().clone()
            versions = [(id(p), p._version) for p in self.model.parameters()]
            _, data, _, _, _, _, _ = freeze()
            compatible, failing = data[1, 'train'][0], data[16, 'test'][400]
            assert failing['id'] == 'FinMME:5082'
            self.feature_cache.clear()
            old = original_feature(self, compatible).clone()
            old_key = list(self.feature_cache)
            self.feature_cache.clear()
            new = repaired(self, compatible)
            assert torch.equal(old, new) and list(self.feature_cache) == old_key
            self.feature_cache.clear()
            failed_as_expected = False
            try:
                original_feature(self, failing)
            except KeyError as error:
                assert error.args == ('id',)
                failed_as_expected = True
            assert failed_as_expected, 'Positive failure control did not reproduce'
            repaired_value = repaired(self, failing)
            assert repaired_value.numel() == 1792 and torch.isfinite(repaired_value).all()
            assert [(id(p), p._version) for p in self.model.parameters()] == versions
            assert torch.equal(torch.get_rng_state(), before_rng)
            assert torch.equal(torch.cuda.get_rng_state(), before_cuda_rng)
            self.feature_cache.clear()
            receipt['GPU_gate'] = {'passed': True, 'compatible_feature_exact': True,
                'compatible_cache_key_exact': True, 'old_failure_reproduced': failing['id'],
                'repaired_finite_feature_count': 1792, 'parameter_versions_unchanged': True,
                'CPU_CUDA_RNG_exact': True, 'cache_cleared_after_gate': True,
                'additional_feature_calls': 4, 'additional_LoRA_updates': 0,
                'TEST_input_for_implementation_gate_only': True, 'TEST_answers_used': False}
        path = self.run/'feature_identity_repair_execution.json'
        path.write_text(json.dumps(receipt, indent=2)+'\n')
        RUNS.append(self.run)

    trial.MetricExperiment.__init__ = initialize
    return {'original_feature_source_sha256': source_sha, 'repair_source_sha256': sha(Path(__file__))}


def checks():
    import torch
    import torch.nn.functional as F
    from collections import OrderedDict
    from types import SimpleNamespace
    tensor = torch.arange(896).float()[None, :].repeat(2, 1)
    tokenizer = lambda text, **kw: {'input_ids': [1, 2]}
    proxy = SimpleNamespace(no_grad=torch.no_grad, cat=torch.cat, isfinite=torch.isfinite,
                            tensor=lambda value, **kw: torch.tensor(value))
    namespace = {'torch': proxy, 'F': F, 'hashlib': hashlib, 'json': json}
    original, _ = compiled_feature(namespace, False)
    revised, source_sha = compiled_feature(namespace, True)

    class Harness:
        def __init__(self, require_id):
            self.feature_cache = OrderedDict(); self.metric_feature_misses = 0
            self.require_id = require_id; self.last_aux = None
            self.model = SimpleNamespace(eval=lambda: None,
                get_input_embeddings=lambda: lambda ids: tensor[None, :len(ids[0])])
            self.processor = SimpleNamespace(tokenizer=tokenizer)

        def inputs(self, row):
            self.last_aux = row
            if self.require_id and 'id' not in row:
                raise KeyError('id')
            return {'_features': tensor}

    row = {'id': 'synthetic-only', 'question': 'q', 'images': ['synthetic-image']}
    old = Harness(False); fixed = Harness(False)
    a, b = original(old, row), revised(fixed, row)
    assert torch.equal(a, b) and list(old.feature_cache) == list(fixed.feature_cache)
    assert {k:v for k,v in fixed.last_aux.items() if k != 'id'} == old.last_aux
    assert fixed.last_aux['id'] == row['id'] and 'answer' not in fixed.last_aux
    try:
        original(Harness(True), row)
    except KeyError as error:
        assert error.args == ('id',)
    else:
        raise AssertionError('Positive failure control did not fail')
    out = revised(Harness(True), row)
    assert torch.equal(a, out) and out.numel() == 1792
    REPAIR.mkdir(parents=True, exist_ok=True)
    result = {'CPU_only': True, 'synthetic_not_model_performance': True,
              'AST_only_added_aux_id': True, 'features_and_cache_keys_exact': True,
              'original_failure_positive_control': True, 'fixed_failure_control': True,
              'question_image_task_unchanged': True, 'no_answer_in_aux': True,
              'source_sha256': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                'analysis/onevision_learned_reuse_trial.py': source_sha},
              'GPU_gate_pending': True, 'no_model_or_GPU_loaded': True}
    (REPAIR/'CPU_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(checks()))
