"""TRAIN-only paired backend control for the failed pre-training gate.

No TEST predictions, model updates or hyperparameter search. Compare the old
worker runtime with the sealed native reference's deterministic runtime.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import hashlib
import json
from pathlib import Path
import time


def main(gpu):
    import onevision_research_device_entry as device
    cards = device.inventory()
    assert device.select(cards, os.getuid()) == gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    import torch
    import onevision_directional_coeff_trial as worker
    from onevision_knowledge_intervention import reset_seed
    from run import hashed, write
    root = worker.ROOT
    report = root/'summary/onevision_diagnostics_20261016/directional_coeff_20261018/backend_repair'
    report.mkdir(parents=True, exist_ok=True)
    assert not (report/'PROBE_COMPLETE.json').exists()
    run = root/'exp/onevision_directional_coeff_20261018/backend_repair'/time.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True)
    spec, values = worker.plan()
    manifest, data = values[:2]
    # Fixed first hash example of every native TRAIN task, never TEST-driven.
    rows = [min(data[t, 'train'], key=lambda r: hashed(r, 'backend-control-TRAIN')) for t in range(1, 20)]
    assert len(rows) == 19 and all(r['id'] in spec['train_ids'][str(r['_task'])] for r in rows)
    write(run/'admission.json', dict(gpu=gpu, inventory=cards, TRAIN_ids=[r['id'] for r in rows],
                                   TEST_rows_read=0, optimizer_updates=0, no_frozen_source_edits=True))
    reset_seed()
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    x = worker.native.MetricExperiment(manifest, run)
    old_layers = dict(x.layers)
    results = {}

    def evaluate(label, deterministic, threads):
        torch.use_deterministic_algorithms(deterministic)
        torch.set_num_threads(threads)
        x.model.eval(); x.cache.clear()
        rr = []
        with torch.no_grad():
            for row in rows:
                batch = x.inputs(row)
                features = batch['_features'].detach().cpu().contiguous()
                loss, details = x.loss(row)
                rr.append(dict(id=row['id'], nll=float(loss), details=details,
                    input_sha256=hashlib.sha256(batch['input_ids'].cpu().numpy().tobytes()).hexdigest(),
                    visual_sha256=hashlib.sha256(features.view(torch.uint8).numpy().tobytes()).hexdigest()))
        results[label] = rr
        write(run/(label+'.json'), dict(deterministic=deterministic, threads=threads, TRAIN_only=True, rows=rr))

    evaluate('native_reference_runtime', True, 6)
    evaluate('native_old_worker_runtime', False, 4)
    torch.use_deterministic_algorithms(True); torch.set_num_threads(6)
    worker.install(x, worker.VIEWS[0])
    evaluate('direction_reference_runtime', True, 6)
    evaluate('direction_old_worker_runtime', False, 4)
    evaluate('direction_reference_repeat', True, 6)
    assert results['native_reference_runtime'] == results['direction_reference_runtime']
    assert results['direction_reference_repeat'] == results['direction_reference_runtime']
    assert results['native_old_worker_runtime'] == results['direction_old_worker_runtime']
    pairs = [('native_reference_runtime', 'native_old_worker_runtime'),
             ('direction_reference_runtime', 'direction_old_worker_runtime')]
    effects = {}
    for a, b in pairs:
        for u, v in zip(results[a], results[b]):
            assert u['id'] == v['id'] and u['input_sha256'] == v['input_sha256']
        effects[a+'__'+b] = dict(NLL_changed=sum(u['nll'] != v['nll'] for u, v in zip(results[a], results[b])),
            visual_changed=sum(u['visual_sha256'] != v['visual_sha256'] for u, v in zip(results[a], results[b])),
            max_abs_NLL=max(abs(u['nll']-v['nll']) for u, v in zip(results[a], results[b])))
    proof = dict(complete=True, run=str(run.relative_to(root)), TRAIN_QA=19, TEST_QA=0, optimizer_updates=0,
        zero_function_native_vs_direction_exact_both_runtimes=True, deterministic_repeat_exact=True,
        effects=effects, hashes={p.name:worker.digest(p) for p in run.glob('*.json')},
        source_sha256=worker.digest(Path(__file__)),
        limitation='TRAIN runtime control; repaired full500/155 initial gate still mandatory')
    write(report/'PROBE_COMPLETE.json', proof)
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    main(parser.parse_args().gpu)
