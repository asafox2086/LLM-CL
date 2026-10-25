"""Prepare one gold-masked TRAIN input pack for finite semantic qualification.

No model, image decoding, prediction, scoring, new split or data mutation.
The source JSONL bytes contain gold; qualification output never exposes it.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import time

AUDIT_SHA = '48f2c8294e8f6f08ae2d5be94257dc98afaa85dd505c0019320f20eb1662cb01'
PARSER_SHA = 'f14f56aae4742184092e8cad0d1b0e8ca1a30260f9511bf80833b88e9eb5cd83'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def main():
    started = time.monotonic()
    root = Path(os.environ['LLMCL_ROOT']).resolve()
    private = Path(os.environ['LLMCL_ELIGIBILITY_PRIVATE_PACK']).resolve()
    assert private.is_absolute() and not private.exists() and private.parent.is_dir()
    assert root not in private.parents, 'Private input text must stay outside the Git checkout'
    report = root / 'summary/train_answer_prior_20261025'
    receipt_path = report / 'OPTION_ELIGIBILITY_INPUTS.json'
    assert not receipt_path.exists()
    protocol_raw = (report / 'OPTION_ELIGIBILITY_PROTOCOL.md').read_bytes()
    raw = (report / 'AUDIT.json').read_bytes()
    assert digest(raw) == AUDIT_SHA
    audit = json.loads(raw)
    assert audit['selected_train_nested_exact']
    pool = audit['conditions']['pct01']
    source = next(x for x in pool['files'] if x['task'] == 16)
    assert source['count'] == 89 and source['domain'] == 'finance'
    source_path = root / source['path']
    assert source_path.name == 'T16_train.jsonl'
    blob = source_path.read_bytes()
    assert digest(blob) == source['sha256']
    lines = blob.splitlines()
    assert len(lines) == 89
    parser_path = root / 'analysis/train_answer_prior_audit_20261025.py'
    assert digest(parser_path.read_bytes()) == PARSER_SHA
    spec = importlib.util.spec_from_file_location('pinned_train_option_parser', parser_path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)  # stdlib definitions only; no main/data I/O.
    prior = {x['id_sha256']: x for x in pool['rows'] if x['domain'] == 'finance'}
    records = []
    for line in lines:
        row = json.loads(line)  # File bytes include gold, never used to qualify.
        if row['metadata']['question_type'] != 'single_choice':
            continue
        key = digest(row['id'].encode())
        previous = prior[key]
        assert previous['question_type'] == 'single_choice'
        options, flags = parser.parse_options(row['context'])
        assert options is not None and not flags and len(options) == 4
        assert [x[0] for x in options] == ['A', 'B', 'C', 'D']
        candidate = previous['no_predefined_permutation_obstacle']
        records.append(dict(id_sha256=key, source_row_sha256=digest(line),
            question=row['question'], context=row['context'],
            options=[dict(label=k, content=v) for k, v in options],
            image_identifiers_sha256=digest(json.dumps(row['images']).encode()),
            original_screen_candidate=candidate, original_screen_flags=previous['flags']))
    assert len(records) == 53 and sum(x['original_screen_candidate'] for x in records) == 46
    records.sort(key=lambda x: x['id_sha256'])
    for index, record in enumerate(records, 1):
        record['case'] = index
        assert not {'answer', 'answers', 'gold', 'prediction'} & record.keys()
    pack = dict(task=16, question_type='single_choice', records=records,
        original_gold_and_predictions_absent=True,
        intended_old_label_to_new_label={'A': 'B', 'B': 'C', 'C': 'D', 'D': 'A'},
        intended_new_position_to_original_content={'A': 'D', 'B': 'A', 'C': 'B', 'D': 'C'})
    pack_bytes = encoded(pack)
    with private.open('xb') as stream:
        stream.write(pack_bytes)
    receipt = dict(status='INPUT_PACK_ONLY_NOT_QUALIFIED_OR_MODEL_ADMISSION',
        source_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=digest(protocol_raw),
        old_audit_sha256=AUDIT_SHA, parser_sha256=PARSER_SHA,
        train_file=source['path'], train_file_sha256=source['sha256'],
        source_file_rows_parsed=89, single_choice_rows=53,
        original_candidates=46, original_fixed_obstacle_rows=7,
        source_bytes_include_gold=True, gold_used_for_new_qualification=False,
        private_pack=str(private), private_pack_sha256=digest(pack_bytes),
        private_pack_bytes=len(pack_bytes), id_hashes=[x['id_sha256'] for x in records],
        pct01_manifest_sha256=pool['manifest_sha256'],
        pct10_manifest_sha256=audit['conditions']['pct10']['manifest_sha256'],
        nested_exact_input_binding_from_prior_audit=True,
        TRAIN_files_read=1, TEST_files_read=0, images_decoded=0,
        model_loads=0, generations=0, scores=0, updates=0,
        seconds=time.monotonic() - started)
    with receipt_path.open('xb') as stream:
        stream.write(encoded(receipt))
    print(json.dumps(dict(status=receipt['status'], private_pack_sha256=digest(pack_bytes),
        receipt_sha256=digest(receipt_path.read_bytes()), seconds=receipt['seconds'],
        model_loads=0, generations=0, scores=0)))


if __name__ == '__main__':
    main()
