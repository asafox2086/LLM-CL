"""Positive/negative controls for the unchanged author memory selector."""
import ast
from collections import Counter
import contextlib
import io
import json
import os
import random
from types import SimpleNamespace

from forever_low_budget_preflight import ROOT, REPORT, DatasetBridge, author_source


def main():
    _, source, source_sha = author_source('utils/load_data.py')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == 'load_memory_buffer')
    runs = []
    for name, answers, expected in [('100_unique', [str(i) for i in range(100)], 100),
                                    ('100_same', ['same']*100, 2),
                                    ('10_unique', [str(i) for i in range(10)], 10)]:
        rows = [{'id': f'{name}:{i}', 'answer': answer} for i, answer in enumerate(answers)]
        dataset = DatasetBridge(rows)
        namespace = {'Counter': Counter, 'random': random,
                     'os': SimpleNamespace(path=SimpleNamespace(join=os.path.join, exists=lambda _: True)),
                     'get_dataset_order': lambda _: ['task'],
                     'load_dataset': lambda kind, data_files, cache_dir: {'train': dataset},
                     'concatenate_datasets': lambda values: DatasetBridge([r for v in values for r in v.rows])}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'unchanged-author-memory-selector', 'exec'), namespace)
        with contextlib.redirect_stdout(io.StringIO()):
            selected = namespace['load_memory_buffer'](1, 1, '/synthetic_control', 2, '/no_cache', 'qwen', 42)
        assert len(selected) == expected
        runs.append({'case': name, 'QA': len(rows), 'answer_classes': len(set(answers)),
                     'ratio_argument': 2, 'nominal_floor_QA': int(len(rows)*.02),
                     'actual_selected_QA': len(selected), 'expected_selected_QA': expected})
    target = REPORT/'SELECTOR_CONTROLS.json'
    assert not target.exists(), 'Preserve original controls'
    target.write_text(json.dumps({'CPU_only': True, 'synthetic_not_experiment_QA': True,
                                  'source_sha256': source_sha, 'GPU_or_model': False,
                                  'author_selector_changed': False, 'controls': runs}, indent=2)+'\n')
    print(json.dumps({'CPU_only': True, 'controls_passed': len(runs), 'cases': runs}))


if __name__ == '__main__':
    main()
