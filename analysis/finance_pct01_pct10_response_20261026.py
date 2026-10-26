"""Read six sealed finance outputs; copy existing F1, never score or normalize.

Stdlib only. Importing this module performs no I/O. This is retrospective
analysis of two completed SeqFT runs, not a training or selection procedure.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time


POOLS = {
    'pct01': (
        'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json',
        '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777', 1313),
    'pct10': (
        'data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json',
        '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7', 13105),
}
PROOF_SHA256 = {
    'pct01': 'c0c010f18db89b088cd3ac13b92e79a76c560c13ee9f9960f0e5e2ddc8e88c69',
    'pct10': '73fdbbd139a6910225088e78cc44033824456e09dec93411ed93582ac4d67c99',
}
STAGES = (15, 16, 19)
TYPES = ('single_choice', 'multiple_choice', 'numerical', 'UNKNOWN')
GROUPS = ('ALL',) + TYPES
EXPERIMENT = 'five_domain_native_cl_20261016_sparse'
FINANCE_QA = 2233


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def relative_to(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts,
            'source path must be relative to its declared root')
    # Frozen RAM overlays legitimately link data to the original read-only
    # asset tree. Keep the lexical root-relative path; exact bytes are bound
    # by read_bound, whose ledger separately records the resolved location.
    return root / relative


def read_bound(path, expected, ledger):
    raw = path.read_bytes()
    actual = sha(raw)
    require(actual == expected, 'source SHA mismatch: ' + str(path))
    key = str(path.resolve())
    require(key not in ledger or ledger[key]['sha256'] == actual,
            'source changed within consumer')
    ledger[key] = dict(sha256=actual, bytes=len(raw))
    return raw


def json_lines(raw):
    lines = raw.splitlines()
    require(lines and all(lines), 'blank or empty JSONL')
    return [json.loads(line) for line in lines]


def existing_f1(row):
    value = row['scores']['token_f1']
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 100,
            'invalid existing F1 scalar')
    return value


def mean(values):
    return sum(values) / len(values) if values else None


def close_to_proof(value, record):
    require(record['QA'] == FINANCE_QA, 'sealed finance denominator mismatch')
    require(math.isclose(value, record['F1'], rel_tol=0, abs_tol=1e-10),
            'copied existing score mean differs from sealed proof')


def source_type(row):
    value = row.get('metadata', {}).get('question_type')
    return value if isinstance(value, str) and value in TYPES[:3] else 'UNKNOWN'


def load_condition(root, proof_path, condition, ledger):
    proof = json.loads(read_bound(proof_path, PROOF_SHA256[condition], ledger))
    candidate = proof['candidate']
    require(candidate['backbone'] == 'onevision'
            and candidate['method'] == 'SeqFT_LoRA', 'candidate identity')
    require(proof['primary_metric'] == 'existing raw Answer F1'
            and proof['rescored_cells'] == 74 and proof['final_QA'] == 21454
            and proof['positive_finite_training_stages'] == 19, 'incomplete sealed proof')
    require(candidate.get('condition', condition) == condition, 'candidate condition')
    run = relative_to(root, candidate['run'])
    config_path = run / 'config.json'
    config = json.loads(read_bound(config_path, proof['config_sha256'], ledger))
    require(proof['source_sha256'][str(config_path.relative_to(root).as_posix())]
            == proof['config_sha256'], 'proof config source chain')
    require(config['condition'] == condition and config['method'] == 'SeqFT_LoRA'
            and config['backbone'] == 'onevision' and config['seed'] == 42
            and config['epochs'] == 1 and config['test_fraction'] == 1
            and not config['preflight_only'], 'config identity')
    manifest_relative, manifest_sha, train_total = POOLS[condition]
    require(config['manifest_sha256'] == manifest_sha, 'config manifest binding')
    # The older proof has no candidate.condition or candidate.manifest fields.
    require(candidate.get('manifest_sha256', manifest_sha) == manifest_sha
            and proof.get('manifest_sha256', manifest_sha) == manifest_sha,
            'new proof manifest binding')
    require(proof['source_sha256'][manifest_relative] == manifest_sha,
            'proof manifest source chain')
    manifest = json.loads(read_bound(relative_to(root, manifest_relative), manifest_sha, ledger))
    require(manifest['experiment_id'] == EXPERIMENT and manifest['condition'] == condition
            and manifest['seed'] == 42 and manifest['formal_ready']
            and not manifest['assets_pending'], 'manifest identity')
    require([task['task'] for task in manifest['tasks']] == list(range(1, 20))
            and sum(t['files']['train']['count'] for t in manifest['tasks']) == train_total,
            'manifest task or train counts')
    task = manifest['tasks'][15]
    require(task['task'] == 16 and task['domain'] == 'finance', 'finance task mapping')
    test = task['files']['test']
    require(test['count'] == FINANCE_QA
            and Path(test['path']).name == 'T16_test.jsonl', 'finance TEST identity')
    require(proof['source_sha256'][test['path']] == test['sha256'],
            'proof finance TEST source chain')
    originals = json_lines(read_bound(relative_to(root, test['path']), test['sha256'], ledger))
    ids = [row['id'] for row in originals]
    require(len(ids) == len(set(ids)) == FINANCE_QA
            and all(isinstance(x, str) for x in ids), 'finance TEST IDs')
    for row in originals:
        require(isinstance(row['answer'], str) and isinstance(row['answers'], list)
                and row['answers'] and all(isinstance(ref, str) for ref in row['answers']),
                'source answer/reference types')
    predictions = {}
    for stage in STAGES:
        relative = f'evaluations/stage_{stage:02d}/T16_test_predictions.jsonl'
        rows = json_lines(read_bound(run / relative, proof['prediction_sha256'][relative], ledger))
        require([row['id'] for row in rows] == ids, 'prediction ID/order mismatch')
        for row, original in zip(rows, originals):
            require(row['references'] == original['answers'], 'reference mismatch')
            require(isinstance(row['prediction'], str), 'prediction is not raw string')
            require(row['generation_cap'] == task['max_new_tokens'], 'generation cap mismatch')
            require(row.get('metadata', {}) == original.get('metadata', {}),
                    'prediction metadata differs from frozen TEST')
            if 'answer' in row:
                require(row['answer'] == original['answer'], 'saved primary answer mismatch')
            existing_f1(row)
        predictions[stage] = rows
    close_to_proof(mean([existing_f1(row) for row in predictions[19]]), proof['domains']['finance'])
    if condition == 'pct10':
        close_to_proof(mean([existing_f1(row) for row in predictions[16]]),
                       proof['acquired']['domains']['finance'])
    return dict(root=root, proof=proof, run=run, config=config, task=task,
                test_sha256=test['sha256'], originals=originals, predictions=predictions)


def raw_identity(value):
    result = dict(raw_prediction_sha256=sha(value.encode('utf-8')))
    if re.fullmatch('[A-D]', value):
        result['displayable_exact_single_letter'] = value
    return result


def score_counts(rows):
    values = [existing_f1(row) for row in rows]
    return dict(QA=len(rows), existing_F1_mean=mean(values),
                existing_zero_F1=sum(value == 0 for value in values),
                existing_full_F1=sum(value == 100 for value in values))


def summarize(rows):
    result = score_counts(rows)
    counts = Counter(row['prediction'] for row in rows)
    ordered = sorted(counts, key=lambda raw: (-counts[raw], sha(raw.encode('utf-8'))))
    top = []
    for raw in ordered[:3]:
        members = [row for row in rows if row['prediction'] == raw]
        top.append(dict(**raw_identity(raw), raw_prediction_count=counts[raw],
                        **score_counts(members)))
    result.update(distinct_raw_predictions=len(counts), modal_raw=top[0] if top else None,
                  fixed_top3_raw=top,
                  exact_single_letter_A_D=score_counts([
                      row for row in rows if re.fullmatch('[A-D]', row['prediction'])]))
    return result


def compare(before, after):
    require([row['id'] for row in before] == [row['id'] for row in after], 'paired ID mismatch')
    a, b = [existing_f1(row) for row in before], [existing_f1(row) for row in after]
    differences = [y - x for x, y in zip(a, b)]
    return dict(QA=len(a), before_existing_F1_mean=mean(a), after_existing_F1_mean=mean(b),
                after_minus_before_F1_pp=mean(differences),
                higher=sum(y > x for x, y in zip(a, b)),
                lower=sum(y < x for x, y in zip(a, b)),
                tie=sum(y == x for x, y in zip(a, b)),
                zero_to_positive=sum(x == 0 and y > 0 for x, y in zip(a, b)),
                positive_to_zero=sum(x > 0 and y == 0 for x, y in zip(a, b)),
                below_full_to_full=sum(x < 100 and y == 100 for x, y in zip(a, b)),
                full_to_below_full=sum(x == 100 and y < 100 for x, y in zip(a, b)),
                identical_raw_prediction=sum(x['prediction'] == y['prediction']
                                             for x, y in zip(before, after)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root01', 'root10', 'proof01', 'proof10', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), 'output already exists; never overwrite sealed analysis')
    started = time.time()
    wall, cpu = time.monotonic(), time.process_time()
    ledger = {}
    data = {
        'pct01': load_condition(args.root01.resolve(), args.proof01.resolve(), 'pct01', ledger),
        'pct10': load_condition(args.root10.resolve(), args.proof10.resolve(), 'pct10', ledger),
    }
    a, b = data['pct01'], data['pct10']
    require(a['test_sha256'] == b['test_sha256'] and a['originals'] == b['originals'],
            'two finance TEST sources, including primary answers, differ')
    require(a['task']['instruction'] == b['task']['instruction']
            and a['task']['max_new_tokens'] == b['task']['max_new_tokens'],
            'finance instruction/generation cap differs')
    originals = a['originals']
    group_indices = {group: [index for index, row in enumerate(originals)
                            if group == 'ALL' or source_type(row) == group] for group in GROUPS}
    require(sum(len(group_indices[group]) for group in TYPES) == FINANCE_QA,
            'type groups do not partition complete finance TEST')
    unknown = Counter(sha(canonical(row.get('metadata', {}).get('question_type')))
                      for row in originals if source_type(row) == 'UNKNOWN')
    summaries, cross_ratio, acquisition, per_id = {}, {}, {}, []
    for condition in data:
        summaries[condition] = {}
        for stage in STAGES:
            summaries[condition][f'S{stage}'] = {
                group: summarize([data[condition]['predictions'][stage][index]
                                  for index in group_indices[group]]) for group in GROUPS}
        acquisition[condition] = {
            group: compare([data[condition]['predictions'][15][index] for index in group_indices[group]],
                           [data[condition]['predictions'][16][index] for index in group_indices[group]])
            for group in GROUPS}
    for stage in STAGES:
        cross_ratio[f'S{stage}'] = {
            group: compare([a['predictions'][stage][index] for index in group_indices[group]],
                           [b['predictions'][stage][index] for index in group_indices[group]])
            for group in GROUPS}
    difference_in_changes = {}
    for group in GROUPS:
        differences = []
        for index in group_indices[group]:
            change01 = existing_f1(a['predictions'][16][index]) - existing_f1(a['predictions'][15][index])
            change10 = existing_f1(b['predictions'][16][index]) - existing_f1(b['predictions'][15][index])
            differences.append(change10 - change01)
        difference_in_changes[group] = dict(QA=len(differences),
            descriptive_change10_minus_change01_F1_pp=mean(differences),
            positive=sum(x > 0 for x in differences), negative=sum(x < 0 for x in differences),
            zero=sum(x == 0 for x in differences), causal_identification=False)
    for index, original in enumerate(originals):
        record = dict(id_sha256=sha(original['id'].encode('utf-8')),
                      question_type=source_type(original),
                      source_question_type_sha256=sha(canonical(
                          original.get('metadata', {}).get('question_type'))),
                      primary_answer_sha256=sha(original['answer'].encode('utf-8')),
                      references_sha256=sha(canonical(original['answers'])))
        for condition in data:
            record[condition] = {f'S{stage}': dict(
                existing_F1=data[condition]['predictions'][stage][index]['scores']['token_f1'],
                **raw_identity(data[condition]['predictions'][stage][index]['prediction']))
                for stage in STAGES}
            record[condition]['S15_to_S16_existing_F1_pp'] = (
                record[condition]['S16']['existing_F1'] - record[condition]['S15']['existing_F1'])
        record['pct10_minus_pct01_existing_F1_pp'] = {
            f'S{stage}': record['pct10'][f'S{stage}']['existing_F1']
                         - record['pct01'][f'S{stage}']['existing_F1'] for stage in STAGES}
        record['descriptive_difference_in_changes_F1_pp'] = (
            record['pct10']['S15_to_S16_existing_F1_pp']
            - record['pct01']['S15_to_S16_existing_F1_pp'])
        per_id.append(record)
    source = Path(__file__).resolve()
    source_raw = source.read_bytes()
    ledger[str(source)] = dict(sha256=sha(source_raw), bytes=len(source_raw))
    result = dict(
        status='COMPLETE_EXISTING_OUTPUT_ANALYSIS', analysis_not_new_model_evidence=True,
        primary_state='S16 just-acquired finance', other_states=dict(
            S15='before finance task', S19='final table correspondence only'),
        backbone='onevision', method='SeqFT_LoRA', task=16, finance_QA=FINANCE_QA,
        proof_sha256=PROOF_SHA256,
        candidate_runs={condition: value['proof']['candidate']['run'] for condition, value in data.items()},
        finance_TEST_source_sha256=a['test_sha256'],
        exact_TEST_rows_answers_and_references_equal_across_pools=True,
        group_denominators={group: len(indices) for group, indices in group_indices.items()},
        unknown_question_type_value_sha256_counts=dict(sorted(unknown.items())),
        raw_string_policy='No strip, casefold, tokenization, or score recomputation; SHA is exact UTF-8.',
        top3_policy='Descending exact raw-string count, then ascending raw UTF-8 SHA256.',
        summaries=summaries, pct10_minus_pct01_paired=cross_ratio,
        within_ratio_S15_to_S16_paired=acquisition,
        descriptive_difference_in_changes=difference_in_changes, per_ID=per_id,
        limits=[
            'Retrospective full finance TEST analysis after completed sealed runs; not blind preregistration.',
            'S16 is the acquisition readout; S19 only corresponds to the final table, with no forgetting decomposition.',
            'Original question_type is used without outcome-based reclassification; unknown values remain in denominators.',
            'Modal strings can differ across groups/states/ratios; a lower modal count or more distinct strings need not mean more correct answers.',
            'Exact A-D is a limited literal-format diagnostic, not a certificate covering all valid answer formats.',
            'Existing token F1=100 and =0 are reported, not semantic correctness; multiple-choice/numerical token F1 does not certify exact set/numerical correctness.',
            'More data also changes update budget, schedule length and earlier learning history; changes and their difference are descriptive, not pure sample-size causality.',
            'Concentrated errors do not establish position bias or PIF effectiveness and do not authorize changing fixed settings.',
            'No prediction tensor/pixel revalidation, confidence interval, method selection or memory benefit is established by this consumer.',
        ],
        computation=dict(new_models=0, new_training=0, new_generations=0,
                         new_scores=0, new_tokenizations=0, new_bootstraps=0,
                         copied_existing_score_rows=6 * FINANCE_QA),
        source_files=ledger,
        timing=dict(started_unix=started, completed_UTC=datetime.now(timezone.utc).isoformat(),
                    scope='Consumer preparation through in-memory result; final JSON serialization/write excluded.',
                    wall_seconds=time.monotonic() - wall, process_CPU_seconds=time.process_time() - cpu))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a race from replacing an earlier result.
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
