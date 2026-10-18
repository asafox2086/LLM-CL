"""Audit acquisition versus later transfer in two sealed formal V2 runs.

The collector is read-only stdlib CPU code. It neither imports a trainer nor
loads weights. The analyzer binds every reused cell to prior full-run proofs.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

RUNS = {
    'SeqFT_LoRA': 'training_20261017_161154',
    'OSFT': 'training_20261018_044103',
}
PROOFS = {'SeqFT_LoRA': 'b463d0e9fdfecc9eb456.json', 'OSFT': '5193ee4c6861655d85e8.json'}
BASE = 'summary/onevision_diagnostics_20261016/osft_acquisition_20261018'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    sys.path.insert(0, str(root/'analysis'))
    from answer_f1 import answer_f1
    output = dict(CPU_only=True, collector_sha256=sha(Path(__file__)), methods={})
    base = root/'exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01'
    for method, stamp in RUNS.items():
        run = base/method/stamp
        config = read(run/'config.json')
        matrix = read(run/'score_matrices.json')['token_f1']
        assert len(matrix) == 20 and all(len(row) == 19 for row in matrix)
        cells = {}
        for stage, task in sorted({(i, i) for i in range(1, 20)} | {(19, i) for i in range(1, 20)}):
            path = run/f'evaluations/stage_{stage:02d}/T{task:02d}_test_predictions.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            ids = [row['id'] for row in rows]
            assert len(ids) == len(set(ids)) == config['task_counts'][f'T{task:02d}']['test']
            values = [answer_f1(row['prediction'], row['references']) for row in rows]
            assert all(v == row['scores']['token_f1'] for v, row in zip(values, rows))
            value = sum(values)/len(values)
            assert math.isclose(value, matrix[stage][task-1], abs_tol=1e-10)
            rel = str(path.relative_to(run))
            cells[rel] = dict(sha256=sha(path), F1=value, QA=len(values), stage=stage, task=task)
        snapshots = [json.loads(line) for line in (run/'resources.jsonl').read_text().splitlines()]
        assert snapshots and len({s['pid'] for s in snapshots}) == 1
        counts = sorted({s['trainable_parameters'] for s in snapshots if 'trainable_parameters' in s})
        assert len(counts) == 1
        training = [json.loads(line) for line in (run/'logs/events.jsonl').read_text().splitlines()]
        assert len(training) == 93 and {x['stage'] for x in training} == set(range(1, 20))
        assert all(x['event'] == 'train_update' and math.isfinite(x['loss'])
                   and math.isfinite(x['gradient_l2_preclip']) and x['gradient_l2_preclip'] > 0
                   and all(lr > 0 for lr in x['learning_rate_used']) for x in training)
        def brief(snapshot):
            keys = ['phase', 'pid', 'unix', 'trainable_parameters', 'trainable_parameter_bytes',
                    'model_parameter_bytes', 'VmRSS_bytes', 'Pss_bytes', 'cpu_uss_bytes',
                    'gpu_allocated_bytes', 'gpu_reserved_bytes', 'gpu_process_nvml_bytes']
            return {k: snapshot.get(k) for k in keys}
        selected_keys = ['manifest_sha256', 'seed', 'epochs', 'batch_size', 'micro_batch_size',
                         'model_dtype', 'autocast_dtype', 'image_area_cap', 'visual_cache_limit_bytes',
                         'framework', 'condition', 'method', 'backbone']
        record = dict(run=str(run.relative_to(root)), cells=cells,
            config={k: config[k] for k in selected_keys}, config_sha256=sha(run/'config.json'),
            metrics=read(run/'metrics_by_score.json')['token_f1'],
            metrics_sha256=sha(run/'metrics_by_score.json'), score_matrix_sha256=sha(run/'score_matrices.json'),
            resources_sha256=sha(run/'resources.jsonl'), trainable_parameters=counts[0],
            training_log_sha256=sha(run/'logs/events.jsonl'), actual_positive_finite_updates=93,
            first_actual_LR=training[0]['learning_rate_used'],
            first_snapshot=brief(snapshots[0]), last_snapshot=brief(snapshots[-1]))
        if method == 'OSFT':
            path = run/'runner_source/osft_adapter.py'
            source = path.read_text()
            assert 'min(weight.shape)*.75' in source and 'unfreeze_rank_ratio\":.25' in source
            record['method_source_sha256'] = sha(path)
            record['actual_rank_allocation'] = 'fixed .25 bottom singular rank, all seven decoder projections'
            record['paper_adaptive_layer_allocation_executed'] = False
        output['methods'][method] = record
    return output


def analyze(root, snapshot):
    raw = read(snapshot)
    assert raw['CPU_only'] and raw['collector_sha256'] == sha(Path(__file__))
    rows, proof_hashes = {}, {}
    folder = root/'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/completed'
    for method, name in PROOFS.items():
        path = folder/name
        proof, source = read(path), raw['methods'][method]
        proof_hashes[method] = sha(path)
        assert proof['final_QA'] == 21454 and proof['rescored_cells'] == 74
        assert proof['positive_finite_training_stages'] == 19 and proof['candidate']['run'] == source['run']
        assert proof['config_sha256'] == source['config_sha256']
        assert proof['source_sha256'][source['run']+'/metrics_by_score.json'] == source['metrics_sha256']
        assert proof['source_sha256'][source['run']+'/logs/events.jsonl'] == source['training_log_sha256']
        if method == 'OSFT':
            assert proof['source_sha256'][source['run']+'/runner_source/osft_adapter.py'] == source['method_source_sha256']
        acquired, final = [], []
        for task in range(1, 20):
            for stage, target in [(task, acquired), (19, final)]:
                rel = f'evaluations/stage_{stage:02d}/T{task:02d}_test_predictions.jsonl'
                cell = source['cells'][rel]
                assert cell['sha256'] == proof['prediction_sha256'][rel]
                assert math.isfinite(cell['F1']) and cell['stage'] == stage and cell['task'] == task
                target.append(cell['F1'])
        ap, acquisition = sum(final)/19, sum(acquired)/19
        bwt = sum(b-a for a, b in zip(acquired[:-1], final[:-1]))/18
        assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in zip(final, proof['final_native19']))
        assert math.isclose(ap, proof['aggregate']['AP'], abs_tol=1e-10)
        assert math.isclose(bwt, proof['aggregate']['BWT'], abs_tol=1e-10)
        assert math.isclose(ap, acquisition+18/19*bwt, abs_tol=1e-10)
        rows[method] = dict(AP19=ap, acquired19_mean=acquisition, BWT=bwt,
            later_transfer_contribution=18/19*bwt, acquired19=acquired, final19=final,
            trainable_parameters=source['trainable_parameters'], first_actual_LR=source['first_actual_LR'],
            actual_positive_finite_updates=source['actual_positive_finite_updates'], resources=source['last_snapshot'])
    first, second = [raw['methods'][m]['config'] for m in RUNS]
    shared = [k for k in first if k != 'method']
    assert all(first[k] == second[k] for k in shared)
    old, new = rows['SeqFT_LoRA'], rows['OSFT']
    effect = {k: new[k]-old[k] for k in ['AP19', 'acquired19_mean', 'BWT', 'later_transfer_contribution']}
    assert math.isclose(effect['AP19'], effect['acquired19_mean']+effect['later_transfer_contribution'], abs_tol=1e-10)
    result = dict(CPU_only=True, analyzer_sha256=sha(Path(__file__)), snapshot_sha256=sha(snapshot),
        prior_full_proof_sha256=proof_hashes, shared_protocol_keys=shared, methods=rows,
        OSFT_minus_SeqFT=effect,
        trainable_parameter_ratio=new['trainable_parameters']/old['trainable_parameters'],
        identity='AP19 = mean_i R[i,i] + (18/19)*BWT, with 19 tasks and own-method acquired states',
        new_training_or_predictions=False,
        limits=['This is an accounting identity and observed paired method comparison, not a causal decomposition.',
                'No capacity-matched or remove-projection control, so the SVD/orthogonality contribution is not isolated.',
                'Each method has its own acquired reference; BWT is not a direct comparison on identical weights.',
                'OSFT uses fixed25% author implementation, not the paper adaptive layer-importance recipe.',
                'Different parameterization/capacity/LR mean shared framework does not mean equal trainable capacity.',
                'Resource endpoints are same-run snapshots, not isolated TRAIN peaks or normalized memory advantage.',
                'Formal FP16 common-image results are separate from native AnyRes research155.'])
    output = root/BASE
    output.mkdir(parents=True, exist_ok=True)
    (output/'EVIDENCE.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(OSFT_minus_SeqFT=effect, trainable_parameter_ratio=result['trainable_parameter_ratio'], methods=rows)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect', action='store_true')
    group.add_argument('--analyze', type=Path)
    args = parser.parse_args()
    if args.collect:
        print(json.dumps(collect(args.root)))
    else:
        analyze(args.root, args.analyze)
