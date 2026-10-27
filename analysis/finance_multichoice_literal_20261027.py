"""Sealed multi-choice answers: literal missing members versus wrong members.

Read-only inputs and original F1; no scoring/tokenization/model calls. All 546
financial multi-choice IDs are retained. Strict strings are raw [A-D]+; repeated
letters are a separate outcome. No stripping, sorting or answer rewriting.
Reference alternatives are never chosen by best agreement: all must be strict,
nonrepeating strings and yield the same relation for a consensus classification.
Any/all diagnostics are reported separately with their explicit denominators.
This is descriptive TEST analysis, not semantic accuracy or an EOS diagnosis.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import time

from finance_strong_baseline_acquisition_20261027 import (
    METHODS, STAGES, canonical, digest, f1, load, question_type, require, summarize,
)

HELPER_SHA = 'fd306888772f88fa53127c53047770598a842b3f205f42c2fc4c433c5350844d'
RELATIONS = ('EXACT_RAW', 'SAME_MEMBERS_OTHER_ORDER', 'MISSING_ONLY',
             'EXTRA_ONLY', 'WRONG_AND_MISSING_WITH_OVERLAP', 'DISJOINT_MEMBERS')
FLAGS = ('exact_raw', 'proper_raw_prefix', 'proper_member_subset',
         'has_wrong_member', 'has_missing_member', 'single_is_reference_member')


def strict(value):
    return re.fullmatch('[A-D]+', value) is not None


def repeated(value):
    return strict(value) and len(set(value)) != len(value)


def relation(prediction, reference):
    """Only called on strict, nonrepeating strings; no normalized score."""
    p, r = set(prediction), set(reference)
    if prediction == reference: label = 'EXACT_RAW'
    elif p == r: label = 'SAME_MEMBERS_OTHER_ORDER'
    elif p < r: label = 'MISSING_ONLY'
    elif r < p: label = 'EXTRA_ONLY'
    elif p & r: label = 'WRONG_AND_MISSING_WITH_OVERLAP'
    else: label = 'DISJOINT_MEMBERS'
    return label, dict(
        exact_raw=prediction == reference,
        proper_raw_prefix=len(prediction) < len(reference) and reference.startswith(prediction),
        proper_member_subset=p < r,
        has_wrong_member=bool(p-r),
        has_missing_member=bool(r-p),
        single_is_reference_member=len(prediction) == 1 and prediction in reference,
    )


def classify(row):
    prediction, references = row['prediction'], row['references']
    require(isinstance(prediction, str) and isinstance(references, list)
            and references and all(isinstance(r, str) for r in references), 'answer schema')
    valid = [r for r in references if strict(r) and not repeated(r)]
    base = dict(reference_count=len(references), distinct_raw_reference_count=len(set(references)),
                valid_reference_count=len(valid), invalid_reference_count=sum(not strict(r) for r in references),
                repeated_reference_count=sum(repeated(r) for r in references),
                strict_prediction=strict(prediction), repeated_prediction=repeated(prediction),
                single_prediction=re.fullmatch('[A-D]', prediction) is not None,
                raw_character_length=len(prediction), relation_pairs=0, flags=None)
    if not strict(prediction):
        return dict(base, category='PREDICTION_NONSTRICT_UNKNOWN')
    if repeated(prediction):
        return dict(base, category='PREDICTION_REPEATED_LETTER')
    pairs = [relation(prediction, r) for r in valid]
    base['relation_pairs'] = len(pairs)
    if pairs:
        base['flags'] = {flag:dict(any=any(v[flag] for _,v in pairs),
                                 all=all(v[flag] for _,v in pairs)) for flag in FLAGS}
    if len(valid) != len(references):
        category = 'REFERENCE_UNRESOLVED'
    else:
        labels = {label for label,_ in pairs}
        category = next(iter(labels)) if len(labels) == 1 else 'MIXED_REFERENCE_RELATION'
    return dict(base, category=category)


def summarize_literal(rows):
    categories = Counter(); lengths = Counter(); reference_counts = Counter()
    reference_totals = Counter(); flags = {k:Counter() for k in FLAGS}
    single_flags = {k:Counter() for k in FLAGS}
    gold_lengths = Counter(); gold_unique_counts = Counter(); single_gold_cross = Counter()
    uniform_reference = []; uniform_single_reference = []
    f1_groups = {}
    strict_count = repeated_count = single_count = eligible_flags = single_eligible = 0
    multiple_reference_QA = 0
    for row in rows:
        item = classify(row); category = item['category']; categories[category] += 1
        f1_groups.setdefault(category, []).append(f1(row))
        strict_count += item['strict_prediction']; repeated_count += item['repeated_prediction']
        single_count += item['single_prediction']; lengths[str(item['raw_character_length'])] += 1
        reference_counts[str(item['reference_count'])] += 1
        multiple_reference_QA += item['reference_count'] > 1
        for key in ('reference_count','valid_reference_count','invalid_reference_count','repeated_reference_count'):
            reference_totals[key] += item[key]
        # Reference occurrences are counted as stored, including repetitions;
        # no alternative is silently selected or deduplicated for these tables.
        for reference in row['references']:
            if strict(reference):
                gold_lengths[str(len(reference))] += 1
                gold_unique_counts[str(len(set(reference)))] += 1
                if item['single_prediction']:
                    single_gold_cross[row['prediction'],len(reference),len(set(reference))] += 1
        if item['valid_reference_count'] == item['reference_count']:
            baselines = [len(set(r))/4 for r in row['references']]
            reference_baseline = dict(min=min(baselines),max=max(baselines),mean=sum(baselines)/len(baselines))
            uniform_reference.append(reference_baseline)
            if item['single_prediction']: uniform_single_reference.append(reference_baseline)
        # Flags use all original alternatives, with none discarded. Partial valid
        # alternatives cannot supply a favorable membership/prefix observation.
        if item['flags'] is not None and item['valid_reference_count'] == item['reference_count']:
            eligible_flags += 1
            single_eligible += item['single_prediction']
            for flag,value in item['flags'].items():
                for quantifier in ('any','all'):
                    flags[flag][quantifier] += value[quantifier]
                    if item['single_prediction']: single_flags[flag][quantifier] += value[quantifier]
    require(sum(categories.values()) == len(rows), 'classification dropped QA')
    return dict(
        QA=len(rows), original_F1=summarize(rows),
        categories={key:categories[key] for key in RELATIONS+(
            'PREDICTION_NONSTRICT_UNKNOWN','PREDICTION_REPEATED_LETTER',
            'REFERENCE_UNRESOLVED','MIXED_REFERENCE_RELATION')},
        category_original_F1={key:dict(QA=len(values),mean=sum(values)/len(values),
            zero=sum(v == 0 for v in values),full=sum(v == 100 for v in values))
            for key,values in sorted(f1_groups.items())},
        strict_letter_string_QA=strict_count, repeated_letter_string_QA=repeated_count,
        strict_single_letter_QA=single_count, raw_character_length_counts=dict(sorted(lengths.items(),key=lambda x:int(x[0]))),
        original_reference_count_distribution=dict(sorted(reference_counts.items(),key=lambda x:int(x[0]))),
        multiple_reference_QA=multiple_reference_QA, reference_totals=dict(reference_totals),
        gold_strict_reference_occurrences=dict(
            raw_length_counts=dict(sorted(gold_lengths.items(),key=lambda x:int(x[0]))),
            unique_member_count_distribution=dict(sorted(gold_unique_counts.items(),key=lambda x:int(x[0]))),
            note='Reference-occurrence denominator, not QA; includes repeated-letter references separately flagged above.'),
        strict_single_output_by_gold_size=[dict(output_letter=letter,gold_raw_length=length,
            gold_unique_members=unique,reference_occurrences=n)
            for (letter,length,unique),n in sorted(single_gold_cross.items())],
        uniform_single_letter_membership_reference=dict(
            QA_denominator=len(uniform_reference),
            definition='For each fully valid-reference QA, |unique reference letters|/4; alternatives equally weighted within QA; QA equally weighted. Descriptive uniform-letter reference, not a model null or significance test.',
            alternative_mean_average=sum(x['mean'] for x in uniform_reference)/len(uniform_reference) if uniform_reference else None,
            alternative_min_average=sum(x['min'] for x in uniform_reference)/len(uniform_reference) if uniform_reference else None,
            alternative_max_average=sum(x['max'] for x in uniform_reference)/len(uniform_reference) if uniform_reference else None,
            strict_single_output_matched_QA_denominator=len(uniform_single_reference),
            strict_single_output_matched_alternative_mean_average=sum(x['mean'] for x in uniform_single_reference)/len(uniform_single_reference) if uniform_single_reference else None,
            strict_single_output_matched_alternative_min_average=sum(x['min'] for x in uniform_single_reference)/len(uniform_single_reference) if uniform_single_reference else None,
            strict_single_output_matched_alternative_max_average=sum(x['max'] for x in uniform_single_reference)/len(uniform_single_reference) if uniform_single_reference else None),
        any_all_reference_flags=dict(QA_denominator=eligible_flags,
            definition='Strict nonrepeating prediction and every original reference strict/nonrepeating; alternatives never selected.',
            counts={key:dict(any=flags[key]['any'],all=flags[key]['all']) for key in FLAGS}),
        strict_single_any_all_reference_flags=dict(QA_denominator=single_eligible,
            counts={key:dict(any=single_flags[key]['any'],all=single_flags[key]['all']) for key in FLAGS}),
    )


def paired_categories(before, after):
    require([r['id'] for r in before] == [r['id'] for r in after], 'paired ID order')
    transitions = Counter(); raw_same = 0
    for left,right in zip(before,after):
        transitions[classify(left)['category'],classify(right)['category']] += 1
        raw_same += left['prediction'] == right['prediction']
    return dict(QA=len(before), identical_raw_prediction_QA=raw_same,
        category_transitions=[dict(before=a,after=b,QA=n) for (a,b),n in sorted(transitions.items())])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','bundle','output'): parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    require(all(p.is_absolute() for p in (args.root,args.bundle,args.output)), 'absolute paths')
    require(not args.output.exists(), 'never overwrite analysis')
    start, cpu = time.monotonic(), time.process_time()
    helper = Path(__file__).with_name('finance_strong_baseline_acquisition_20261027.py')
    require(digest(helper.read_bytes()) == HELPER_SHA, 'frozen helper SHA')
    bundle_raw = args.bundle.read_bytes(); ledger = {}
    originals, data, proofs = load(args.root,json.loads(bundle_raw),ledger)
    indices = [i for i,row in enumerate(originals) if question_type(row) == 'multiple_choice']
    require(len(indices) == 546, 'complete multi-choice denominator')
    selected = {method:{stage:[data[method][stage][i] for i in indices] for stage in STAGES} for method in METHODS}
    # Fixed existing-score partition; all rows remain in ALL and one partition.
    full = [i for i,row in enumerate(selected['PASs_MoE'][16]) if f1(row) == 100]
    rest = [i for i,row in enumerate(selected['PASs_MoE'][16]) if f1(row) != 100]
    partitions = {'ALL':list(range(546)),'PAS_S16_original_F1_100':full,'PAS_S16_original_F1_not_100':rest}
    summaries = {}; stage_changes = {}; versus_seq = {}
    for name,positions in partitions.items():
        rows = {method:{stage:[selected[method][stage][i] for i in positions] for stage in STAGES} for method in METHODS}
        summaries[name] = {method:{str(stage):summarize_literal(rows[method][stage]) for stage in STAGES} for method in METHODS}
        stage_changes[name] = {method:paired_categories(rows[method][15],rows[method][16]) for method in METHODS}
        versus_seq[name] = {method:{str(stage):paired_categories(rows['SeqFT_LoRA'][stage],rows[method][stage])
            for stage in STAGES} for method in METHODS[1:]}
    result = dict(complete=True,unix=time.time(),consumer_sha256=digest(Path(__file__).read_bytes()),
        helper_sha256=HELPER_SHA,bundle_sha256=digest(bundle_raw),proof_bindings=proofs,source_files=ledger,
        task=16,domain='finance',stages=dict(before=15,acquired=16),QA=546,
        ordered_IDs_sha256=digest(canonical([originals[i]['id'] for i in indices])),
        partition_QA={name:len(values) for name,values in partitions.items()},
        summaries=summaries,before_to_acquired_categories=stage_changes,method_vs_Seq_categories=versus_seq,
        existing_F1_values_validated=3*2*2233,multi_choice_F1_values_summarized=3*2*546,
        new_scores=0,new_models=0,new_generations=0,CP_payload_reads=0,raw_QA_or_predictions_in_output=False,
        predefined_interpretation_limits=[
            'Raw [A-D]+ only; whitespace, explanation, other alphabets and empty strings remain unknown. Repeated letters are separate.',
            'Member sets describe letters only; no sorting or normalized prediction replaces any original output or F1.',
            'All original reference alternatives retained. Consensus categories and any/all flags never choose a best reference.',
            'Missing members or a literal prefix do not establish correct content, premature EOS, stopping cause or successful continuation ability.',
            'A uniform random letter is already a member with probability |gold unique letters|/4; membership majority alone does not establish content learning. This descriptive reference is not an actual model null.',
            'PAS full-F1 partition is a retrospective descriptive partition, not a new evaluation subset or training/method selection criterion.',
            'Methods differ in recipe and capacity; these observations cannot identify routing, capacity, type-conflict or loss causality.',
            'Only a competing explanation is narrowed; no loss, prompt, data, generation, scoring or closed experiment is changed.'],
        wall_seconds_before_output=time.monotonic()-start,process_CPU_seconds_before_output=time.process_time()-cpu)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(args.output),sha256=digest(args.output.read_bytes()),new_scores=0)))


if __name__ == '__main__': main()
