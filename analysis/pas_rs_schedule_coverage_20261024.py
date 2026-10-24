"""Reconstruct RS warmup eligibility from a private, read-only metadata receipt.

This is a conditional schedule reconstruction, not per-call gradient evidence.
No Torch, tensor, checkpoint, QA, prediction or scoring code is imported.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    assert not args.output.exists(), 'Do not repeat this completed analysis.'
    raw = args.receipt.read_bytes()
    data = json.loads(raw)
    assert data['accepted_run'].endswith('/PASs_MoE/training_20261017_161725')
    config = data['config']
    assert config['batch_size'] == 16 and config['micro_batch_size'] == 1
    assert config['method'] == 'PASs_MoE' and not config['preflight_only']
    events = data['train_event_schedule']
    assert len(events) == 93
    stages = []
    for stage in range(1, 20):
        rows = [e for e in events if e['stage'] == stage]
        n = config['task_counts'][f'T{stage:02d}']['train']
        assert len(rows) == (n + 15) // 16
        zero_updates = eligible_updates = zero_micro = eligible_micro = 0
        for step, event in enumerate(rows):
            assert event['step'] == step + 1
            assert event['total_qa'] == n
            assert event['qa_seen'] == min((step + 1) * 16, n)
            assert all(v > 0 for v in event['learning_rate_used'])
            count = min(16, n - step * 16)
            # Current manifest-matched learner: stage 1 has no history;
            # later first batches have progress=0 and lambda_eff=0.
            if stage == 1 or step == 0:
                zero_updates += 1
                zero_micro += count
            else:
                eligible_updates += 1
                eligible_micro += count
        stages.append(dict(stage=stage, train_QA=n, updates=len(rows),
            conditional_zero_updates=zero_updates,
            positive_warmup_updates_upper=eligible_updates,
            conditional_zero_micro_calls=zero_micro,
            positive_warmup_micro_calls_upper=eligible_micro))
    totals = {key: sum(s[key] for s in stages) for key in stages[0] if key != 'stage'}
    assert totals['train_QA'] == 1313
    assert totals['conditional_zero_updates'] == 40
    assert totals['positive_warmup_updates_upper'] == 53
    assert totals['conditional_zero_micro_calls'] == 553
    assert totals['positive_warmup_micro_calls_upper'] == 760
    logs = data['local_log_matches']
    active = [dict(path=log['path'], **line) for log in logs
              for line in log['matches'] if '(post-backward) ACTIVE:' in line['text']]
    assert len(active) == 9
    result = dict(
        question='Does local PASs RS have uniform execution opportunity across the 19 formal stages?',
        accepted_run=data['accepted_run'], receipt_sha256=hashlib.sha256(raw).hexdigest(),
        proof_bound_artifacts=data['proof_bound_artifacts'],
        current_upstream_revision=data['current_author_revision'],
        current_upstream_sha256=data['current_author_sha256'],
        historical_upstream_SHA_bound_by_old_proof=False,
        interpretation='Conditional reconstruction using proof-bound schedule/adapter and current manifest-matched learner; not 93 historical per-call witnesses.',
        warmup='lambda_eff=lambda_stab*min(1,progress/0.3), progress=batch_start/task_size; this is a continuous ramp.',
        stages=stages, totals=totals,
        single_batch_stages=[s['stage'] for s in stages if s['updates'] == 1],
        historical_log_ACTIVE_lines=active,
        historical_log_scope='Selected current-read historical stdout lines; full stdout is not bound by the old accepted proof.',
        nonzero_regularizer_gradient_proved=False,
        RS_AnswerF1_causal_benefit_proved=False,
        new_model_runs=0, new_QA=0, new_scores=0, new_tensor_reads=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(totals=totals, historical_ACTIVE_lines=len(active),
        output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
