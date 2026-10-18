"""TRAIN-only realized hard choices and the available routing headroom.

This is a finite 20-anchor/3-state ceiling, never a TEST/generalization oracle.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from answer_f1 import answer_f1
import onevision_answer_compatibility_core as core

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1'
SOURCE = ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    audit = read(REPORT/'TRAIN_FIT_VERIFIED.json')
    attempt = read(REPORT/'build_attempt.json')
    run = ROOT/attempt['run']
    source_proof = read(SOURCE/'build_complete.json')
    assert attempt['spec']['parent_complete_sha256'] == sha(SOURCE/'build_complete.json')
    source = ROOT/source_proof['run']
    state_path = source/'final_state.pt'
    assert sha(state_path) == attempt['spec']['fixed_pool_sha256']
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    for name, expected in audit['TRAIN_artifact_sha256'].items():
        assert sha(run/name) == expected
    feedback = torch.load(run/'feedback_state.pt', map_location='cpu', weights_only=True)
    keys, ids = sorted(state['pool']), state['memory_ids']
    assert keys == [12, 18, 19] and ids == feedback['TRAIN_ids'] and len(ids) == 20
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path) == source_proof['spec']['manifest_sha256']
    train, domains = {}, {}
    for task in read(manifest_path)['tasks']:
        info = task['files']['train']
        path = ROOT/info['path']
        assert sha(path) == info['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            train[row['id']] = row
            domains[row['id']] = task['domain']
    raw, teacher_hashes = [], {}
    for key in keys:
        path = run/f'evaluations/TRAIN_teacher_state{key}/TRAIN_memory_image.jsonl'
        assert sha(path) == feedback['teacher_sha256'][str(key)]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r['id'] for r in rows] == ids
        values = []
        for row in rows:
            assert row['references'] == train[row['id']]['answers']
            score = answer_f1(row['prediction'], row['references'])
            assert score == row['scores']['token_f1']
            values.append(score)
        raw.append(values)
        teacher_hashes[str(key)] = sha(path)
    # Match the frozen worker: Python division precedes FP32 tensor construction.
    assert torch.equal(torch.tensor([[v/100 for v in values] for values in raw],
                                    dtype=torch.float32), feedback['scores'])
    score = feedback['scores']
    allowed = core.image_mask([tuple(train[i]['images']) for i in ids])
    summaries = {}
    for label, theta in [('fixed', torch.zeros_like(feedback['theta'])), ('learned', feedback['theta'])]:
        expected = core.weights(state['memory_features'], state['memory_features'], theta, allowed)@score.T
        # Exact-score ties would need the method's geometric tie-breaker. None occur here.
        assert ((expected == expected.max(1, keepdim=True).values).sum(1) == 1).all()
        indices = expected.argmax(1)
        actual = [raw[int(k)][i] for i, k in enumerate(indices)]
        soft = (torch.softmax(expected/core.EXPERT_TEMPERATURE, 1)*score.T).sum(1)
        summaries[label] = dict(image_excluded_TRAIN_choices=[keys[int(k)] for k in indices],
            raw_hard_TRAIN_F1=sum(actual)/len(actual), soft_TRAIN_F1=float(100*soft.mean()),
            routing_counts=dict(Counter(keys[int(k)] for k in indices)))
    global_means = {str(k): sum(values)/20 for k, values in zip(keys, raw)}
    oracle = sum(max(values) for values in zip(*raw))/20
    best_global = max(global_means.values())
    informative = [i for i, values in enumerate(zip(*raw)) if len(set(values)) > 1]
    details = []
    for domain in sorted(set(domains[i] for i in ids)):
        positions = [j for j, i in enumerate(ids) if domains[i] == domain]
        means = {str(k): sum(raw[a][j] for j in positions)/len(positions) for a, k in enumerate(keys)}
        ceiling = sum(max(raw[a][j] for a in range(len(keys))) for j in positions)/len(positions)
        details.append(dict(domain=domain, TRAIN_QA=len(positions), candidate_F1=means,
                            oracle_F1=ceiling, ceiling_minus_best_fixed=ceiling-max(means.values()),
                            informative_QA=sum(i in informative for i in positions)))
    result = dict(CPU_only=True, analyzer_sha256=sha(Path(__file__)),
        source_pool_sha256=sha(state_path), TRAIN_teacher_sha256=teacher_hashes,
        TRAIN_fit_proof_sha256=sha(REPORT/'TRAIN_FIT_VERIFIED.json'),
        metric_source_sha256=sha(ROOT/'analysis/onevision_answer_compatibility_core.py'),
        TRAIN_QA=20, candidate_states=keys, informative_QA=len(informative),
        global_TRAIN_means=global_means, oracle_TRAIN_F1=oracle,
        best_fixed_TRAIN_F1=best_global, oracle_minus_best_fixed_pp=oracle-best_global,
        actual_image_excluded_training_decisions=summaries, TRAIN_domains=details,
        TEST_questions_answers_predictions_parsed=False, new_GPU_or_optimizer_steps=False,
        limits=['Finite ceiling applies only to these20 TRAIN anchors and3 frozen states.',
                'All20 score labels supervised theta; image masking is not held-out validation.',
                'No TEST routing ceiling, generalization cause, or impossibility of learned representations follows.',
                'No new settings selected or active GPU/queue source changed.'])
    destination = REPORT/'TRAIN_DECISION_CEILING.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    assert not torch.cuda.is_initialized()
    print(json.dumps({k: result[k] for k in ['global_TRAIN_means', 'oracle_TRAIN_F1',
        'oracle_minus_best_fixed_pp', 'actual_image_excluded_training_decisions', 'TRAIN_domains']}))


if __name__ == '__main__':
    main()
