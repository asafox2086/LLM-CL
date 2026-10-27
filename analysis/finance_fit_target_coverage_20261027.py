"""Join existing TRAIN audit with fixed FIT IDs; no raw QA/TEST/model reading.

Only these two sealed metadata artifacts are opened. This does not rerun the
TRAIN audit. Character members/cardinality are not tokenizer-level candidates,
semantic answers, or evidence that a new training objective will work.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

AUDIT_SHA = '48f2c8294e8f6f08ae2d5be94257dc98afaa85dd505c0019320f20eb1662cb01'
BINDING_SHA = 'd149d3a002066270db2d37a77c6dedbf999505c3cd820a420036236e037a5f96'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
TRAIN_PATH = 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/T16_train.jsonl'
TYPES = ('single_choice','multiple_choice','numerical')


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read(path, expected):
    raw = path.read_bytes()
    assert sha(raw) == expected
    return json.loads(raw)


def coverage(rows):
    cardinalities = Counter(); raw_targets = Counter(); members = Counter()
    reference_counts = Counter(); flags = Counter(); options = Counter()
    defined = 0; single_reference_defined = 0; anonymous_rows = []
    for row in rows:
        target = row['safe_original_answer']
        parsed = row['original_choice_members']
        labels = row['original_option_labels']
        lexical = isinstance(target,str) and re.fullmatch('[A-D]+',target) is not None
        nonrepeating = lexical and len(set(target)) == len(target)
        consistent = nonrepeating and parsed == list(target) and row['choice_set'] == sorted(target)
        within_options = consistent and isinstance(labels,list) and set(target) <= set(labels)
        definition_available = bool(within_options)
        defined += definition_available
        single_reference_defined += definition_available and row['reference_count'] == 1
        reference_counts[str(row['reference_count'])] += 1
        options[str(row['options_count'])] += 1
        flags.update(row['flags'])
        if definition_available:
            cardinalities[str(len(set(target)))] += 1
            raw_targets[target] += 1
            members.update(target)
        anonymous_rows.append(dict(id_sha256=row['id_sha256'],answer_sha256=row['answer_sha256'],
            references_sha256=row['references_sha256'],reference_count=row['reference_count'],
            strict_AD_string=lexical,repeated_letters=bool(lexical and not nonrepeating),
            parsed_members_agree_with_raw=bool(consistent),within_original_option_labels=bool(within_options),
            character_member_definition_available=definition_available,
            cardinality=len(set(target)) if definition_available else None,
            multiple_reference_relation_unresolved=row['reference_count'] != 1,flags=row['flags']))
    return dict(QA=len(rows),character_definition_QA=defined,
        character_definition_AND_single_reference_QA=single_reference_defined,
        unresolved_definition_QA=len(rows)-defined,
        reference_count_distribution=dict(reference_counts),original_option_count_distribution=dict(options),
        original_primary_cardinality_counts=dict(sorted(cardinalities.items())),
        original_primary_letter_string_counts=dict(sorted(raw_targets.items())),
        original_primary_member_counts={letter:members[letter] for letter in 'ABCD'},
        flags=dict(sorted(flags.items())),anonymous_rows=anonymous_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit','binding','output'):p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args();assert all(x.is_absolute() for x in (a.audit,a.binding,a.output))
    assert not a.output.exists()
    started = time.monotonic()
    audit = read(a.audit,AUDIT_SHA);binding = read(a.binding,BINDING_SHA)
    condition = audit['conditions']['pct01']
    assert condition['manifest_sha256'] == binding['manifest_sha256'] == MANIFEST_SHA
    assert audit['opened_data_files'][TRAIN_PATH] == binding['train_sha256']
    assert binding['types'] == list(TYPES) and binding['FIT'] == 69
    mappings = binding['mapping'];assert len(mappings) == 69
    assert len({r['id_sha256'] for r in mappings}) == 69
    original = [r for r in condition['rows'] if r['domain'] == 'finance' and r['task'] == 16]
    assert len(original) == 89 and len({r['id_sha256'] for r in original}) == 89
    by_id = {r['id_sha256']:r for r in original};selected = []
    for entry in mappings:
        row = by_id[entry['id_sha256']]
        assert type(entry['true']) is int and 0 <= entry['true'] < len(TYPES)
        assert row['question_type'] == TYPES[entry['true']]
        selected.append(row)
    counts = Counter(r['question_type'] for r in selected)
    assert dict(counts) == binding['FIT_counts'] == dict(single_choice=40,multiple_choice=17,numerical=12)
    summaries = {kind:coverage([r for r in selected if r['question_type'] == kind]) for kind in TYPES}
    result = dict(complete=True,source_sha256=sha(Path(__file__).read_bytes()),
        inputs=dict(audit=dict(path=str(a.audit),sha256=AUDIT_SHA),binding=dict(path=str(a.binding),sha256=BINDING_SHA)),
        manifest_sha256=MANIFEST_SHA,split_sha256=binding['split_sha256'],train_sha256=binding['train_sha256'],
        existing_formal_finance_TRAIN_QA=89,fixed_finance_FIT_QA=69,type_counts=dict(counts),summaries=summaries,
        raw_QA_files_opened=0,TEST_files_opened=0,new_scores=0,new_models=0,new_generations=0,
        limitations=[
            'Only joins existing TRAIN-derived metadata. Twenty non-FIT TRAIN IDs are not treated as training exposure.',
            'Primary-target members/cardinality are described without rewriting their raw order. Numerical targets remain in the complete denominator but are not parsed as option sets.',
            'Reference count and hashes are inherited. More than one reference remains unresolved: this artifact cannot reconstruct alternative membership agreement.',
            'Character labels do not determine actual tokenizer IDs, prefix-conditioned legal next tokens or EOS behavior.',
            'Coverage does not identify a learning cause, warrant a removed auxiliary head, or authorize a new loss/training run.',
            'No TEST-derived distribution determines weights, sample selection, target counts or generation constraints.'],
        wall_seconds_before_output=time.monotonic()-started)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8',newline='\n') as h:
        json.dump(result,h,ensure_ascii=False,indent=2,allow_nan=False);h.write('\n')
    print(json.dumps(dict(output=str(a.output),sha256=sha(a.output.read_bytes()),new_scores=0)))


if __name__ == '__main__':main()
