"""One stdlib consumer of sealed answer rows; never score or run a model."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import time

METHODS = ('SeqFT_LoRA', 'OSFT', 'PASs_MoE')
PROOF_SHA256 = {
    'SeqFT_LoRA': 'c0c010f18db89b088cd3ac13b92e79a76c560c13ee9f9960f0e5e2ddc8e88c69',
    'OSFT': 'f8612d49ff55bcd3678365cbd193bd7dc2315c570e522a4a9edc095eddc52a5d',
    'PASs_MoE': 'a17c1000dfff6816644d3e86170fe255d6c3225eebfa911d4c0834fcabb0dc9f'}
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
PHASES = ('initial', 'acquired', 'final')
REFUSALS = ('i cannot', "i'm unable", 'i am unable', 'cannot determine', 'cannot identify',
            'not able to', '无法确定', '无法判断', '无法识别', '无法回答', '不能确定', '看不清', '无法看清')
FLAGS = ('empty', 'at_generation_cap', 'literal_refusal', 'Han_refs_Latin_only_prediction',
         'numeric_refs_no_digit_prediction', 'letter_refs_nonletter_prediction', 'very_low_token_diversity')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def features(row):
    if '_response_profile_features' in row:
        return row['_response_profile_features']
    prediction = row['prediction']; refs = row['references']; assert refs and all(isinstance(r, str) and r.strip() for r in refs)
    toks = re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', prediction.lower())
    han = lambda s: bool(re.search(r'[\u3400-\u9fff]', s))
    pure_number = lambda s: bool(re.fullmatch(r'\s*[+\-−]?(?:\d+(?:[.,]\d+)*|[.]\d+)\s*%?\s*', s))
    pure_letter = lambda s: bool(re.fullmatch(r'\s*[A-Ea-e]\s*', s))
    cap, count = row['generation_cap'], row['output_tokens']
    assert isinstance(cap, int) and isinstance(count, int) and 0 <= count <= cap
    flags = dict(empty=not prediction.strip(), at_generation_cap=count == cap,
        literal_refusal=any(phrase in prediction.lower() for phrase in REFUSALS),
        Han_refs_Latin_only_prediction=all(han(r) for r in refs) and not han(prediction) and bool(re.search('[A-Za-z]', prediction)),
        numeric_refs_no_digit_prediction=all(pure_number(r) for r in refs) and not bool(re.search(r'\d', prediction)),
        letter_refs_nonletter_prediction=all(pure_letter(r) for r in refs) and not pure_letter(prediction),
        very_low_token_diversity=len(toks) >= 12 and len(set(toks)) <= 3)
    assert set(flags) == set(FLAGS)
    result = flags, tuple(toks), tuple(tuple(re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', r.lower())) for r in refs)
    row['_response_profile_features'] = result
    return result


def summarize(rows):
    n = len(rows); assert n
    flag_counts = {key: dict(all=0, zero_F1=0, nonperfect_F1=0) for key in FLAGS}
    predictions = Counter(); references = Counter(); lengths = []; zero = full = union_zero = 0
    for row in rows:
        flags, pred, refs = features(row); f1 = row['scores']['token_f1']
        assert isinstance(f1, (int, float)) and math.isfinite(f1) and 0 <= f1 <= 100
        zero += f1 == 0; full += f1 == 100; union_zero += f1 == 0 and any(flags.values())
        predictions[pred] += 1; references[refs] += 1; lengths.append(len(pred))
        for key, value in flags.items():
            flag_counts[key]['all'] += value
            flag_counts[key]['zero_F1'] += value and f1 == 0
            flag_counts[key]['nonperfect_F1'] += value and f1 < 100
    modal, frequency = sorted(predictions.items(), key=lambda item: (-item[1], item[0]))[0]
    modal_rows = [row for row in rows if features(row)[1] == modal]
    lengths.sort()
    return dict(QA=n, existing_F1_mean=sum(row['scores']['token_f1'] for row in rows)/n,
        existing_zero_F1=zero, existing_full_F1=full, existing_partial_F1=n-zero-full,
        flags=flag_counts, zero_F1_covered_by_any_flag=union_zero,
        zero_F1_without_these_flags=zero-union_zero, flags_overlap_not_causal_error_classes=True,
        unique_normalized_predictions=len(predictions), modal_prediction_count=frequency,
        modal_prediction_token_sha256=hashlib.sha256(json.dumps(modal, ensure_ascii=False).encode()).hexdigest(),
        modal_prediction_zero_F1=sum(row['scores']['token_f1']==0 for row in modal_rows),
        modal_prediction_full_F1=sum(row['scores']['token_f1']==100 for row in modal_rows),
        modal_reference_set_count=max(references.values()),
        prediction_score_token_lengths=dict(median=(lengths[(n-1)//2]+lengths[n//2])/2,
            p95=lengths[math.ceil(.95*n)-1], maximum=lengths[-1]))


def main(root, bundle, output):
    begin = time.monotonic(); cpu = time.process_time()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120-second analysis cap')))
    signal.alarm(120)
    assert not output.exists(), 'Never repeat this consumer into the same outcome'
    binding = json.loads(bundle.read_text()); assert tuple(binding['methods']) == METHODS
    result = {}; all_rows = {}; source_hashes = {}; examples = []
    reference_ids = {}; reference_answers = {}
    for method in METHODS:
        record = binding['methods'][method]; proof = record['proof']
        assert hashlib.sha256(record['proof_raw'].encode()).hexdigest() == record['proof_sha256']
        assert record['proof_sha256'] == PROOF_SHA256[method]
        assert json.loads(record['proof_raw']) == proof
        assert proof['candidate']['method'] == method and proof['candidate']['backbone'] == 'onevision'
        assert proof['final_QA'] == 21454 and proof['rescored_cells'] == 74 and proof['positive_finite_training_stages'] == 19
        run = root / proof['candidate']['run']; assert run.is_relative_to(root)
        config = json.loads((run/'config.json').read_text()); assert sha(run/'config.json') == proof['config_sha256']
        policy = json.loads((root/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json').read_text())
        manifest_path = root/policy['conditions']['pct01']['manifest']
        assert sha(manifest_path) == config['manifest_sha256']
        manifest = json.loads(manifest_path.read_text()); assert len(manifest['tasks']) == 19
        by_phase = {phase: {domain: [] for domain in DOMAINS} for phase in PHASES}
        by_task = {}; transitions = {}
        for task in manifest['tasks']:
            tid = task['task']; domain = task['domain']; by_task[tid] = {}
            for phase, stage in (('initial', 0), ('acquired', tid), ('final', 19)):
                relative = f'evaluations/stage_{stage:02d}/T{tid:02d}_test_predictions.jsonl'; path = run/relative
                digest = sha(path); assert digest == proof['prediction_sha256'][relative]
                source_hashes[path.relative_to(root).as_posix()] = digest
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                assert len(rows) == task['files']['test']['count']
                ids = [row['id'] for row in rows]; assert len(set(ids)) == len(ids)
                if tid not in reference_ids:
                    reference_ids[tid] = ids; reference_answers[tid] = [row['references'] for row in rows]
                assert ids == reference_ids[tid] and [row['references'] for row in rows] == reference_answers[tid]
                assert all(row['generation_cap'] == task['max_new_tokens'] for row in rows)
                by_phase[phase][domain].extend(rows); by_task[tid][phase] = rows
                for flag in FLAGS:
                    selected = sorted((row for row in rows if row['scores']['token_f1']==0 and features(row)[0][flag]),
                        key=lambda row: hashlib.sha256(row['id'].encode()).digest())[:2]
                    examples.extend(dict(method=method, phase=phase, task=tid, domain=domain, flag=flag,
                        id=row['id'], prediction=row['prediction'], references=row['references'],
                        existing_F1=row['scores']['token_f1'], output_tokens=row['output_tokens'], generation_cap=row['generation_cap']) for row in selected)
            before, after = by_task[tid]['acquired'], by_task[tid]['final']
            transitions[str(tid)] = dict(domain=domain, QA=len(before),
                acquired_to_final_F1_improved=sum(b['scores']['token_f1']>a['scores']['token_f1'] for a,b in zip(before,after)),
                acquired_to_final_F1_worse=sum(b['scores']['token_f1']<a['scores']['token_f1'] for a,b in zip(before,after)),
                new_final_zero=sum(a['scores']['token_f1']>0 and b['scores']['token_f1']==0 for a,b in zip(before,after)),
                final_zero_already_zero_at_acquisition=sum(a['scores']['token_f1']==b['scores']['token_f1']==0 for a,b in zip(before,after)),
                flags_appearing_after_acquisition={flag:sum(not features(a)[0][flag] and features(b)[0][flag] for a,b in zip(before,after)) for flag in FLAGS})
        aggregate = {phase:{domain:summarize(by_phase[phase][domain]) for domain in DOMAINS} for phase in PHASES}
        assert sum(row['QA'] for row in aggregate['final'].values()) == 21454
        for domain in DOMAINS:
            assert abs(aggregate['final'][domain]['existing_F1_mean']-proof['domains'][domain]['F1']) < 1e-9
        result[method] = dict(run=proof['candidate']['run'], proof_sha256=record['proof_sha256'],
            profile=aggregate, task_profile={str(tid):{phase:summarize(rows) for phase,rows in phases.items()} for tid,phases in by_task.items()},
            acquired_to_final=transitions)
        all_rows[method] = by_phase['final']
    pair_counts = {}
    for other in ('OSFT', 'PASs_MoE'):
        pair_counts[other+'_vs_SeqFT_LoRA'] = {domain:dict(
            higher=sum(a['scores']['token_f1']>b['scores']['token_f1'] for a,b in zip(all_rows[other][domain],all_rows['SeqFT_LoRA'][domain])),
            lower=sum(a['scores']['token_f1']<b['scores']['token_f1'] for a,b in zip(all_rows[other][domain],all_rows['SeqFT_LoRA'][domain]))) for domain in DOMAINS}
    private = root/'.deployment/response_failure_profile_20261024/private_examples.json'
    write(private, dict(examples=examples, selection='First two SHA-sorted zero-F1 rows per fixed literal flag/task/method/phase; overlapping flags allowed'))
    value = dict(complete=True, methods=result, final_same_ID_pairs=pair_counts,
        source_sha256=source_hashes, consumer_sha256=sha(Path(__file__)), bundle_sha256=sha(bundle),
        private_examples_sha256=sha(private), no_QA_text_in_public_output=True,
        original_scores_copied_not_recomputed=True, new_models=0,new_QA=0,new_scores=0,new_bootstraps=0,
        wall_seconds=time.monotonic()-begin, process_CPU_seconds=time.process_time()-cpu,
        process_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        limitations=['Post-hoc sealed TEST response diagnosis; no parameter/prompt/method selection.',
          'Flags overlap and are literal observations, not causal explanations or certified semantic errors.',
          'Cap equality is a possible stop limit, not proven truncation without saved EOS token IDs.',
          'Phase alignment localizes appearance, not why it happened; rank/optimizer/knowledge causes remain unisolated.'])
    write(output, value); signal.alarm(0)
    print(json.dumps(dict(complete=True, output_sha256=sha(output), wall_seconds=value['wall_seconds'],
        final={method:{d:{key:row[key] for key in ('QA','existing_zero_F1','zero_F1_covered_by_any_flag','modal_prediction_count')}
            for d,row in record['profile']['final'].items()} for method,record in result.items()})))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root.resolve(),args.bundle,args.output)
