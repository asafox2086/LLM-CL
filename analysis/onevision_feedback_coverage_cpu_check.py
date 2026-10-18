"""Actual worker import/branches and same-state mechanics without a backbone."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import onevision_feedback_coverage_trial as worker


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    assert worker.FeedbackExperiment is worker.previous.FeedbackExperiment
    assert worker.core is worker.previous.core
    fixture = dict(pool={12: {'fixture': 12}, 18: {'fixture': 18}, 19: {'fixture': 19}},
        prototypes={12: torch.ones(1792), 18: -torch.ones(1792), 19: torch.zeros(1792)})
    features = torch.zeros(20, 1792)
    features[:, 0] = 1
    state = worker.make_state(fixture, [str(i) for i in range(20)], features)
    assert state['pool'] is fixture['pool'] and state['prototypes'] is fixture['prototypes']
    x = object.__new__(worker.FeedbackExperiment)
    x.feature = lambda row: features[0]
    calls = []
    x.restore = lambda value: calls.append(value)
    scores = torch.zeros(3, 20)
    scores[0] = 1
    for view in worker.VIEWS:
        x.set_feedback(state, scores, torch.zeros(1792), view)
        x.activate_route(dict(question='fixture', images=['x'], answers=['poison'], _task=999))
        first, count = dict(x.last_route), len(calls)
        x.activate_route(dict(question='fixture', images=['x'], answers=['changed'], _task=1))
        assert x.last_route == first and len(calls) == count and first['route_state'] == 12
        assert first['route_mode'] == view
    imports = []
    for gpu in ['0', '1']:
        code = 'import os,torch;import onevision_feedback_coverage_trial;assert os.environ["CUDA_VISIBLE_DEVICES"]=="'+gpu+'";assert not torch.cuda.is_initialized()'
        subprocess.run([sys.executable, '-c', code], cwd=worker.ROOT/'analysis',
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), check=True)
        imports.append(gpu)
    assert not torch.cuda.is_initialized()
    result = dict(CPU_only=True, checks=8, same_original_experiment_and_metric_objects=True,
        shared_pool_and_prototypes_not_copied=True, both_new_routes_active=True,
        gold_and_native_task_changes_do_not_change_route=True, same_state_not_reloaded=True,
        actual_worker_import_preserves_GPU_env=imports, CUDA_initialized=False, GPU_started=False,
        source_sha256={str(p.relative_to(worker.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), Path(worker.__file__)]})
    (worker.REPORT/'CPU_IMPLEMENTATION_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
