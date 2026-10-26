"""Once-only descriptive joins of sealed blind labels and existing scores.

No scorer, tokenizer, image, model or source QA file is opened. Raw text in the
private ledger is used only to validate its unchanged marker and blind mapping.
Public output omits questions, answer text, original IDs and judgment reasons.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

PACKET = 'ec4a0904e1575c6ced9dcc8c09e83ea2b11f9fe3ae7765f97f3bfd8b9187d379'
LEDGER = '97b0b04d7be6f5d271f50a52330efa6c918624d551885dfc6e115c7d5b22c729'
JUDGES = {
    'plain_r8_control': 'c1d109d07e81c3f9a46b4e9c0e5d478ec63022d6024e8ae71c3462c84c3d32b4',
    'research_review': 'd69441f4c9d2d91786954e564fcdfe992a29c7a56cfa6c92ca9a365aa5349a03',
}
LABELS = {'SAME_CONTENT', 'DIFFERENT_CONTENT', 'UNDETERMINED'}
GROUPS = ('UNCHANGED_RAW', 'SAME_CONTENT', 'DIFFERENT_CONTENT', 'UNDETERMINED', 'DISAGREEMENT')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path, expected):
    raw = path.read_bytes()
    assert sha(raw) == expected, str(path)
    return json.loads(raw)


def score(row, when, name):
    value = row[when][name]
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and 0 <= value <= 100
    return value


def stats(rows, denominator):
    assert denominator > 0
    result = dict(QA=len(rows), full_task_denominator=denominator)
    for name in ('token_f1', 'exact_match'):
        changes = [score(r, 'acquired', name) - score(r, 'before', name) for r in rows]
        result[name] = dict(
            before_mean=sum(score(r, 'before', name) for r in rows)/len(rows) if rows else None,
            acquired_mean=sum(score(r, 'acquired', name) for r in rows)/len(rows) if rows else None,
            higher=sum(x > 0 for x in changes), lower=sum(x < 0 for x in changes),
            tie=sum(x == 0 for x in changes),
            sum_existing_score_change=sum(changes),
            contribution_to_full_task_mean_pp=sum(changes)/denominator,
            gross_loss=sum(-x for x in changes if x < 0),
            gross_gain=sum(x for x in changes if x > 0))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.monotonic()
    packet = load(args.private_dir/'changed.blind.json', PACKET)
    ledger = load(args.private_dir/'pairs.private.json', LEDGER)
    receipt_raw = args.receipt.read_bytes()
    receipt = json.loads(receipt_raw)
    assert receipt['private_output_sha256'] == {'pair_ledger': LEDGER, 'blind_packet': PACKET}
    assert receipt['counts'] == dict(full_pairs=491, tasks=2, original_prediction_records=982,
                                   UNCHANGED_RAW=283, changed_for_blind_review=208)
    items, pairs = packet['items'], ledger['pairs']
    assert len(items) == 208 and len(pairs) == 491
    by_key = {row['key']: row for row in pairs}
    assert len(by_key) == 491 and len({row['key'] for row in items}) == 208
    judgments = {}
    for annotator, expected in JUDGES.items():
        data = load(args.private_dir/f'annotations.{annotator}.json', expected)
        assert data['schema'] == 'medical-blind-judgments-v1'
        assert data['annotator'] == annotator and data['blind_packet_sha256'] == PACKET
        rows = data['judgments']
        assert len(rows) == 208 and {r['index'] for r in rows} == set(range(208))
        judgments[annotator] = {}
        for row in rows:
            assert type(row['index']) is int and row['label'] in LABELS
            assert isinstance(row['reason'], str) and row['reason'].strip()
            if 'key' in row:
                assert row['key'] == items[row['index']]['key']
            judgments[annotator][row['index']] = row['label']
    records, cross = [], Counter()
    selected = set()
    for index, item in enumerate(items):
        pair = by_key[item['key']]
        assert pair['initial_class'] == 'AWAIT_TWO_BLIND_JUDGMENTS'
        assert item['question'] == pair['question']
        assert item['answer_A'] == pair[pair['display_mapping']['A']]['raw_answer']
        assert item['answer_B'] == pair[pair['display_mapping']['B']]['raw_answer']
        a, b = [judgments[j][index] for j in JUDGES]
        cross[a, b] += 1
        pair = dict(pair, consensus=a if a == b else 'DISAGREEMENT')
        records.append(pair); selected.add(pair['key'])
    for pair in pairs:
        if pair['key'] in selected:
            continue
        assert pair['initial_class'] == 'UNCHANGED_RAW'
        assert pair['before']['raw_answer'] == pair['acquired']['raw_answer']
        assert all(score(pair, 'before', name) == score(pair, 'acquired', name)
                   for name in ('token_f1', 'exact_match'))
        records.append(dict(pair, consensus='UNCHANGED_RAW'))
    assert len(records) == 491 and len({r['key'] for r in records}) == 491
    scopes = {}
    for scope, rows in [('T03', [r for r in records if r['task'] == 3]),
                        ('T08', [r for r in records if r['task'] == 8]), ('ALL', records)]:
        assert len(rows) == {'T03':375, 'T08':116, 'ALL':491}[scope]
        full = stats(rows, len(rows))
        groups = {g:stats([r for r in rows if r['consensus'] == g], len(rows)) for g in GROUPS}
        assert sum(g['QA'] for g in groups.values()) == len(rows)
        for name in ('token_f1', 'exact_match'):
            assert math.isclose(sum(g[name]['sum_existing_score_change'] for g in groups.values()),
                                full[name]['sum_existing_score_change'], rel_tol=0, abs_tol=1e-8)
            for group in groups.values():
                group[name]['share_of_gross_loss'] = (
                    group[name]['gross_loss']/full[name]['gross_loss'] if full[name]['gross_loss'] else None)
        if scope != 'ALL':
            task = int(scope[1:])
            cells = {x['stage']:x for x in ledger['cell_identity_checks'] if x['task'] == task}
            for when, stage in [('before', task-1), ('acquired', task)]:
                assert math.isclose(full['token_f1'][when+'_mean'], cells[stage]['sealed_F1'],
                                    rel_tol=0, abs_tol=1e-10)
        scopes[scope] = dict(full=full, groups=groups)
    public = dict(status='COMPLETE_BLIND_RELATION_AUDIT', scopes=scopes,
        judgment_cross_table=[dict(judge_1=a, judge_2=b, QA=n) for (a,b),n in sorted(cross.items())],
        agreement=dict(changed_pairs=208, same_label=sum(n for (a,b),n in cross.items() if a==b),
                       disagreements=sum(n for (a,b),n in cross.items() if a!=b)),
        evidence_sha256=dict(private_ledger=LEDGER, blind_packet=PACKET, judgments=JUDGES,
                             prepare_receipt=sha(receipt_raw), consumer_source=sha(Path(__file__).read_bytes())),
        per_ID_hashed=[dict(key=r['key'], task=r['task'], consensus=r['consensus']) for r in records],
        limits=['Two model agents compare meaning, not independent human medical correctness.',
                'Changed meaning does not identify which answer is correct or the causal learning failure.',
                'Unknowns/disagreements retained; unchanged raw answers are not certified correct.',
                'Existing exact-match values are copied, not independently re-scored.',
                'Gross-loss shares describe score differences, not causal effects or a new formal metric.',
                'Only two post-hoc tasks from one pct10 backbone/method; no new training is admitted.'],
        activity=dict(new_scores=0, tokenizations=0, model_calls=0, images_TRAIN_checkpoints_read=0,
                      semantic_reannotation=0, public_QA_or_answer_text=False),
        wall_seconds_before_output_write=time.monotonic()-started)
    with args.output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(public, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    main()
