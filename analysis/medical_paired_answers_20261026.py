"""Prepare one private medical pair ledger and a changed-only blinded packet.

Stdlib only; no I/O at import. The two input bundles contain the already sealed
payloads. No embedded path, scorer, tokenizer, model, image, TRAIN or checkpoint
is opened. This entry never performs semantic annotation or final summarization.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


PROOF = '73fdbbd139a6910225088e78cc44033824456e09dec93411ed93582ac4d67c99'
MANIFEST_PATH = 'data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json'
MANIFEST = '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
TASK_COUNTS = {3: 375, 8: 116}
STATES = {(task, stage) for task in TASK_COUNTS for stage in (task - 1, task)}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def decoded(ref, expected, label, ledger):
    raw = base64.b64decode(ref['raw_base64'], validate=True)
    require(sha(raw) == ref['sha256'] == expected, 'sealed payload SHA: ' + label)
    if 'expected_sha256' in ref:
        require(ref['expected_sha256'] == expected, 'declared expected SHA: ' + label)
    if 'bytes' in ref:
        require(len(raw) == ref['bytes'], 'payload bytes: ' + label)
    ledger[label] = {key: ref[key] for key in
                     ('path', 'relative', 'root', 'resolved_path') if key in ref}
    ledger[label].update(sha256=expected, bytes=len(raw))
    return raw


def lines(raw):
    parts = raw.splitlines()
    require(parts and all(parts), 'empty or blank JSONL')
    return [json.loads(line) for line in parts]


def existing_score(row, name):
    value = row['scores'][name]
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value <= 100, 'invalid existing score')
    return value


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(args):
    private = args.private_dir.resolve()
    receipt = args.receipt.resolve()
    require(not private.exists() and not receipt.exists(), 'outputs must be new; no repeat/overwrite')
    require(not receipt.is_relative_to(private), 'public receipt must be outside private output')
    tick, cpu, unix = time.monotonic(), time.process_time(), time.time()
    ledger, bundles = {}, {}
    for label, path in (('medical_bundle', args.medical_bundle),
                        ('acquisition_bundle', args.acquisition_bundle)):
        raw = path.read_bytes()
        ledger[label] = dict(path=str(path.resolve()), sha256=sha(raw), bytes=len(raw))
        bundles[label] = json.loads(raw)
    medical = bundles['medical_bundle']
    require(medical['schema'] == 'private-pct10-medical-paired-inputs-v1', 'medical bundle schema')
    acquisition = bundles['acquisition_bundle']
    require(acquisition['schema'] == 'private-sealed-acquisition-bundle-v1', 'acquisition bundle schema')
    origin = acquisition['conditions']['pct10']
    proof_raw = decoded(origin['formal_proof'], PROOF, 'pct10.formal_proof', ledger)
    require(decoded(medical['formal_proof'], PROOF, 'medical.formal_proof', ledger) == proof_raw,
            'two bundle formal proofs differ')
    proof = json.loads(proof_raw)
    candidate = proof['candidate']; run = candidate['run']
    require(candidate['method'] == 'SeqFT_LoRA' and candidate['backbone'] == 'onevision'
            and candidate['condition'] == 'pct10'
            and medical['root'] == origin['root']
            and medical['candidate_run'] == origin['candidate_run'] == run, 'candidate identity')
    require(proof['primary_metric'] == 'existing raw Answer F1' and proof['rescored_cells'] == 74
            and proof['final_QA'] == 21454 and proof['positive_finite_training_stages'] == 19,
            'formal proof completeness')
    config_ref = origin['files']['config']
    require(config_ref['relative'] == run + '/config.json'
            and proof['source_sha256'][config_ref['relative']] == proof['config_sha256'], 'config chain')
    config = json.loads(decoded(config_ref, proof['config_sha256'], 'pct10.config', ledger))
    require(config['condition'] == 'pct10' and config['backbone'] == 'onevision'
            and config['method'] == 'SeqFT_LoRA' and config['manifest_sha256'] == MANIFEST
            and not config['preflight_only'], 'config identity')
    manifest_ref = origin['files']['manifest']
    require(manifest_ref['relative'] == MANIFEST_PATH
            and proof['source_sha256'][MANIFEST_PATH] == MANIFEST, 'manifest chain')
    manifest = json.loads(decoded(manifest_ref, MANIFEST, 'pct10.manifest', ledger))
    require(manifest['condition'] == 'pct10' and manifest['formal_ready']
            and not manifest['assets_pending']
            and [task['task'] for task in manifest['tasks']] == list(range(1, 20)), 'manifest identity')
    references = {}
    for row in proof['verified_cells']:
        key = row['task'], row['stage']
        require(key not in references, 'duplicate verified cell')
        references[key] = row
    require(STATES <= set(references), 'four sealed verified cells missing')
    require(len(medical['files']) == 6, 'exactly two TEST/four prediction payloads required')
    tests, predictions = {}, {}
    for ref in medical['files']:
        task = ref['task']
        require(task in TASK_COUNTS and ref['root'] == medical['root'], 'medical source scope')
        original = manifest['tasks'][task - 1]
        require(original['domain'] == 'medical', 'original task domain')
        if ref['kind'] == 'TEST':
            require(task not in tests and ref['relative'] == original['files']['test']['path'], 'TEST path/duplicate')
            expected = original['files']['test']['sha256']
            require(proof['source_sha256'][ref['relative']] == expected, 'TEST proof chain')
            tests[task] = lines(decoded(ref, expected, f'T{task:02d}.TEST', ledger))
        else:
            require(ref['kind'] == 'prediction', 'unexpected source kind')
            key = task, ref['stage']
            require(key in STATES and key not in predictions, 'prediction stage/duplicate')
            relative = f"evaluations/stage_{ref['stage']:02d}/T{task:02d}_test_predictions.jsonl"
            require(ref['prediction_relative'] == relative and ref['relative'] == run + '/' + relative,
                    'prediction source path')
            expected = proof['prediction_sha256'][relative]
            predictions[key] = lines(decoded(ref, expected, f'T{task:02d}.S{ref["stage"]:02d}', ledger))
    require(set(tests) == set(TASK_COUNTS) and set(predictions) == STATES, 'incomplete source coverage')
    pairs, blinded, seen, cell_checks = [], [], set(), []
    for task in sorted(TASK_COUNTS):
        original_task = manifest['tasks'][task - 1]
        source_rows = tests[task]
        ids = [row['id'] for row in source_rows]
        count = TASK_COUNTS[task]
        require(len(ids) == len(set(ids)) == count == original_task['files']['test']['count'], 'full TEST count/IDs')
        require(not seen.intersection(ids), 'cross-task duplicate source ID')
        seen.update(ids)
        for stage in (task - 1, task):
            records = predictions[task, stage]
            require([row['id'] for row in records] == ids, 'prediction ID/order differs from TEST')
            for source, row in zip(source_rows, records):
                require(isinstance(source['question'], str) and isinstance(source['answer'], str)
                        and isinstance(source['answers'], list) and source['answers']
                        and all(isinstance(answer, str) for answer in source['answers']), 'original QA types')
                require(row['references'] == source['answers']
                        and row['metadata'] == source['metadata']
                        and row['generation_cap'] == original_task['max_new_tokens']
                        and row['native_label'] == original_task['native_label'], 'reference/metadata/cap identity')
                require(isinstance(row['prediction'], str), 'raw prediction type')
                existing_score(row, 'token_f1'); existing_score(row, 'exact_match')
            cell = references[task, stage]
            copied_mean = sum(existing_score(row, 'token_f1') for row in records) / count
            require(cell['QA'] == count and math.isclose(copied_mean, cell['F1'], rel_tol=0, abs_tol=1e-10),
                    'existing F1 mean differs from sealed verified cell')
            cell_checks.append(dict(task=task, stage=stage, QA=count,
                existing_F1_mean=copied_mean, sealed_F1=cell['F1'], F1_mean_bound=True,
                EXACT_values_copied_from_sealed_predictions=True,
                EXACT_mean_in_verified_cells=False))
        for index, source in enumerate(source_rows):
            before, after = predictions[task, task - 1][index], predictions[task, task][index]
            source_id = source['id']
            # Different hash domains: the public/blind key must not itself
            # expose the bit that determines A/B's chronological direction.
            key = sha(b'medical-blind-key-v1\0' + source_id.encode('utf-8'))
            swap = hashlib.sha256(b'medical-answer-order-v1\0' + source_id.encode('utf-8')).digest()[0] & 1
            mapping = {'A': 'acquired' if swap else 'before', 'B': 'before' if swap else 'acquired'}
            values = {'before': before['prediction'], 'acquired': after['prediction']}
            unchanged = values['before'] == values['acquired']
            if unchanged:
                require(all(before['scores'][metric] == after['scores'][metric]
                            for metric in ('token_f1', 'exact_match')),
                        'identical raw answer/references have inconsistent sealed scores')
            record = dict(key=key, source_id=source_id, source_id_sha256=sha(source_id.encode('utf-8')),
                task=task, before_stage=task - 1, acquired_stage=task, original_order=index,
                question=source['question'], primary_answer=source['answer'], references=source['answers'],
                metadata=source['metadata'], context=source.get('context', ''),
                generation_cap=original_task['max_new_tokens'],
                before=dict(raw_answer=values['before'], token_f1=before['scores']['token_f1'],
                            exact_match=before['scores']['exact_match']),
                acquired=dict(raw_answer=values['acquired'], token_f1=after['scores']['token_f1'],
                              exact_match=after['scores']['exact_match']),
                display_mapping=mapping,
                initial_class='UNCHANGED_RAW' if unchanged else 'AWAIT_TWO_BLIND_JUDGMENTS',
                no_semantic_correctness_label_assigned=True)
            pairs.append(record)
            if not unchanged:
                sort_key = sha(b'medical-blind-sort-v1\0' + source_id.encode('utf-8'))
                blinded.append((sort_key, dict(key=key, question=source['question'],
                    answer_A=values[mapping['A']], answer_B=values[mapping['B']])))
    require(len(pairs) == len(seen) == 491, 'complete paired denominator')
    require(len({pair['key'] for pair in pairs}) == 491, 'blind-key collision')
    blind_items = [row for _, row in sorted(blinded, key=lambda item: item[0])]
    require(all(set(row) == {'key', 'question', 'answer_A', 'answer_B'} for row in blind_items), 'blind field contract')
    instructions = [
        'Judge every item once without recovering source IDs, direction, references or existing scores.',
        'SAME_CONTENT only if the two answers clearly express the same meaning.',
        'DIFFERENT_CONTENT only if the two answers clearly express different meanings; do not decide which is correct.',
        'Use UNDETERMINED for partial overlap, added/omitted information, ambiguity or insufficient context.',
        'Give a short reason per key. No image/fact checking, answer rewriting, semantic accuracy score or new F1.',
        'Do not inspect the private pair ledger or original bundles before both independent annotations are sealed.',
    ]
    private_ledger = dict(schema='private-medical-paired-ledger-v1', formal_proof_sha256=PROOF,
        candidate_run=run, pairs=pairs, cell_identity_checks=cell_checks, source_files=ledger,
        blind_direction_rule='SHA256(medical-answer-order-v1 NUL source_id), first byte low bit',
        blind_key_rule='SHA256(medical-blind-key-v1 NUL source_id)',
        blind_sort_rule='SHA256(medical-blind-sort-v1 NUL source_id)',
        original_scores_copied_not_recomputed=True, annotation_or_final_summary_performed=False)
    blind_packet = dict(schema='medical-paired-answers-blind-v1', instructions=instructions, items=blind_items)
    private.mkdir(parents=True, exist_ok=False)
    pair_path, blind_path = private / 'pairs.private.json', private / 'changed.blind.json'
    write_new(pair_path, private_ledger); write_new(blind_path, blind_packet)
    source = Path(__file__).resolve(); source_raw = source.read_bytes()
    ledger_public = {key: dict(sha256=value['sha256'], bytes=value['bytes']) for key, value in ledger.items()}
    public = dict(schema='medical-paired-prepare-receipt-v1', status='PREPARED_NOT_ANNOTATED',
        formal_proof_sha256=PROOF, manifest_sha256=MANIFEST,
        consumer_source_sha256=sha(source_raw), source_files=ledger_public,
        counts=dict(full_pairs=491, tasks=2, original_prediction_records=982,
                    UNCHANGED_RAW=len(pairs)-len(blind_items), changed_for_blind_review=len(blind_items)),
        private_output_sha256=dict(pair_ledger=sha(pair_path.read_bytes()), blind_packet=sha(blind_path.read_bytes())),
        output_bytes=dict(pair_ledger=pair_path.stat().st_size, blind_packet=blind_path.stat().st_size),
        F1_cell_means_bound_to_original_verified_cells=True,
        EXACT_scope='Existing sealed prediction values copied and checked finite/in-range; verified_cells has no EXACT mean.',
        public_QA_or_answer_text=False,
        blind_item_fields_limited_to_key_question_answer_A_answer_B=True,
        explicit_task_stage_scores_references_and_direction_fields_in_blind_items=False,
        activity=dict(new_scores=0, new_tokenizations=0, models=0, generations=0, training=0,
                      images_TRAIN_checkpoints_read=0, embedded_source_paths_opened=0,
                      semantic_annotations=0, final_summary=0),
        timing=dict(started_unix=unix, prepared_UTC=datetime.now(timezone.utc).isoformat(),
                    wall_seconds_before_receipt_write=time.monotonic()-tick,
                    CPU_seconds_before_receipt_write=time.process_time()-cpu))
    receipt.parent.mkdir(parents=True, exist_ok=True)
    write_new(receipt, public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--medical-bundle', type=Path, required=True)
    parser.add_argument('--acquisition-bundle', type=Path, required=True)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
