"""Three sealed pct01 finance S15/S16 outputs; existing raw F1 only.

Stdlib, no model/scorer/tokenizer imports. This retrospective TEST description
does not select a method, change training or certify semantic correctness.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

METHODS = ('SeqFT_LoRA','OSFT','PASs_MoE')
# Same proof identities and bundle contract as response_failure_profile_20261024.
PROOF_SHA256 = {
    'SeqFT_LoRA':'c0c010f18db89b088cd3ac13b92e79a76c560c13ee9f9960f0e5e2ddc8e88c69',
    'OSFT':'f8612d49ff55bcd3678365cbd193bd7dc2315c570e522a4a9edc095eddc52a5d',
    'PASs_MoE':'a17c1000dfff6816644d3e86170fe255d6c3225eebfa911d4c0834fcabb0dc9f'}
MANIFEST = 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
TYPES = ('single_choice','multiple_choice','numerical','UNKNOWN')
GROUPS = ('ALL',)+TYPES
STAGES = (15,16)
QA = 2233


def require(value, message):
    if not value: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def rooted(root, relative):
    path=Path(relative)
    require(not path.is_absolute() and '..' not in path.parts,'not ROOT-relative')
    return root/path  # Read-only frozen asset links may resolve outside ROOT.


def bound(path, expected, ledger):
    raw=path.read_bytes(); actual=digest(raw)
    require(actual == expected,'source SHA mismatch: '+str(path))
    key=str(path.resolve())
    require(key not in ledger or ledger[key]['sha256'] == actual,'source changed')
    ledger[key]=dict(sha256=actual,bytes=len(raw)); return raw


def lines(raw):
    parts=raw.splitlines(); require(parts and all(parts),'blank/empty JSONL')
    return [json.loads(x) for x in parts]


def f1(row):
    value=row['scores']['token_f1']
    require(isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value) and 0 <= value <= 100,'invalid sealed F1')
    return value


def strict_single(row):
    return re.fullmatch('[A-D]',row['prediction']) is not None


def question_type(row):
    metadata=row.get('metadata',{})
    value=metadata.get('question_type') if isinstance(metadata,dict) else None
    return value if isinstance(value,str) and value in TYPES[:3] else 'UNKNOWN'


def unknown_type_identity(row):
    metadata=row.get('metadata',{})
    value=metadata.get('question_type') if isinstance(metadata,dict) else {'non_mapping_metadata':metadata}
    return digest(canonical(value))


def average(values): return sum(values)/len(values) if values else None


def summarize(rows):
    values=[f1(row) for row in rows]; selected=[row for row in rows if strict_single(row)]
    count=len(rows)
    return dict(QA=count,existing_F1_mean=average(values),zero_F1=sum(v == 0 for v in values),
        full_F1=sum(v == 100 for v in values),partial_F1=sum(0 < v < 100 for v in values),
        strict_raw_A_D_count=len(selected),strict_raw_A_D_fraction=len(selected)/count if count else None,
        strict_raw_A_D_zero_F1=sum(f1(row) == 0 for row in selected),
        strict_raw_A_D_full_F1=sum(f1(row) == 100 for row in selected),
        strict_raw_letter_counts={letter:sum(row['prediction'] == letter for row in rows) for letter in 'ABCD'})


def compare(before, after):
    require([r['id'] for r in before] == [r['id'] for r in after],'paired IDs/order mismatch')
    delta=[f1(b)-f1(a) for a,b in zip(before,after)]
    gain=sum(max(v,0) for v in delta); loss=sum(max(-v,0) for v in delta)
    return dict(QA=len(delta),before_F1=average([f1(x) for x in before]),
        after_F1=average([f1(x) for x in after]),mean_difference_pp=average(delta),
        gain_count=sum(v > 0 for v in delta),loss_count=sum(v < 0 for v in delta),tie_count=sum(v == 0 for v in delta),
        gross_gain_F1_point_sum=gain,gross_loss_F1_point_sum=loss,net_F1_point_sum=sum(delta),
        gross_gain_mean_pp=gain/len(delta) if delta else None,
        gross_loss_mean_pp=loss/len(delta) if delta else None,
        zero_to_positive=sum(f1(a) == 0 and f1(b) > 0 for a,b in zip(before,after)),
        positive_to_zero=sum(f1(a) > 0 and f1(b) == 0 for a,b in zip(before,after)),
        strict_single_zero_before=sum(strict_single(x) and f1(x) == 0 for x in before),
        strict_single_zero_after=sum(strict_single(x) and f1(x) == 0 for x in after))


def intersection(method_rows):
    require(tuple(method_rows) == METHODS,'method ordering')
    sequences=list(method_rows.values()); ids=[x['id'] for x in sequences[0]]
    require(all([x['id'] for x in rows] == ids for rows in sequences),'intersection IDs')
    common_zero=[]; all_single=[]; both=[]
    per_method={method:0 for method in METHODS}
    for index,ident in enumerate(ids):
        rows=[method_rows[method][index] for method in METHODS]
        zero=all(f1(row) == 0 for row in rows); single=all(strict_single(row) for row in rows)
        if zero: common_zero.append(ident)
        if single: all_single.append(ident)
        if zero and single: both.append(ident)
        for method,row in zip(METHODS,rows): per_method[method] += zero and strict_single(row)
    return dict(QA=len(ids),all_methods_zero_F1=len(common_zero),
        all_methods_strict_single=len(all_single),all_methods_zero_AND_all_strict_single=len(both),
        among_common_zero_strict_single_by_method=per_method,
        common_zero_IDs_sha256=digest(canonical(common_zero)),
        all_strict_single_IDs_sha256=digest(canonical(all_single)),
        common_zero_AND_all_strict_single_IDs_sha256=digest(canonical(both)))


def load(root, bundle, ledger):
    require(tuple(bundle['methods']) == METHODS,'bundle method contract')
    manifest=None; originals=None; reference_ids=None; data={}; source_proofs={}
    for method in METHODS:
        record=bundle['methods'][method]; proof=record['proof']; raw=record['proof_raw'].encode()
        require(digest(raw) == record['proof_sha256'] == PROOF_SHA256[method],'proof SHA')
        require(json.loads(raw) == proof,'proof parsed/raw mismatch')
        candidate=proof['candidate']
        require(candidate['method'] == method and candidate['backbone'] == 'onevision'
                and candidate.get('condition','pct01') == 'pct01','proof identity')
        require(proof['rescored_cells'] == 74 and proof['final_QA'] == 21454
                and proof['positive_finite_training_stages'] == 19
                and proof['primary_metric'] == 'existing raw Answer F1','incomplete proof')
        run=rooted(root,candidate['run']); config_path=run/'config.json'
        config=json.loads(bound(config_path,proof['config_sha256'],ledger))
        require(proof['source_sha256'][config_path.relative_to(root).as_posix()] == proof['config_sha256'],'config proof chain')
        require(config['method'] == method and config['backbone'] == 'onevision'
                and config['condition'] == 'pct01' and config['seed'] == 42 and config['epochs'] == 1
                and config['test_fraction'] == 1 and not config['preflight_only'],'config identity')
        require(config['manifest_sha256'] == proof['source_sha256'][MANIFEST] == MANIFEST_SHA,'manifest chain')
        current=json.loads(bound(root/MANIFEST,MANIFEST_SHA,ledger))
        require(current['condition'] == 'pct01' and len(current['tasks']) == 19
                and sum(t['files']['train']['count'] for t in current['tasks']) == 1313,'manifest counts')
        if manifest is None: manifest=current
        require(manifest == current,'manifest changed between methods')
        task=manifest['tasks'][15]; test=task['files']['test']
        require(task['task'] == 16 and task['domain'] == 'finance' and test['count'] == QA,'finance task')
        require(proof['source_sha256'][test['path']] == test['sha256'],'TEST proof chain')
        source=lines(bound(rooted(root,test['path']),test['sha256'],ledger))
        ids=[row['id'] for row in source]
        require(len(ids) == len(set(ids)) == QA and all(isinstance(x,str) for x in ids),'TEST IDs')
        if originals is None: originals=source; reference_ids=ids
        require(source == originals and ids == reference_ids,'TEST content/metadata differs')
        data[method]={}
        for stage in STAGES:
            relative=f'evaluations/stage_{stage:02d}/T16_test_predictions.jsonl'
            digest_expected=proof['prediction_sha256'][relative]
            require(proof['source_sha256'][(run/relative).relative_to(root).as_posix()] == digest_expected,'prediction proof chain')
            rows=lines(bound(run/relative,digest_expected,ledger))
            require([r['id'] for r in rows] == ids,'prediction ID order')
            for row,original in zip(rows,source):
                require(isinstance(row['prediction'],str),'raw prediction string')
                require(row['references'] == original['answers'] and isinstance(original['answers'],list)
                        and original['answers'] and all(isinstance(x,str) for x in original['answers']),'references')
                require(row.get('metadata',{}) == original.get('metadata',{}),'question type metadata')
                require(row['generation_cap'] == task['max_new_tokens'],'cap')
                require(type(row['output_tokens']) is int and 0 <= row['output_tokens'] <= row['generation_cap'],'output token count')
                if 'answer' in row: require(row['answer'] == original['answer'],'primary answer')
                f1(row)
            data[method][stage]=rows
        source_proofs[method]=dict(proof_sha256=record['proof_sha256'],run=candidate['run'],
                                  config_sha256=proof['config_sha256'])
    return originals,data,source_proofs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','bundle','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve(); output=args.output.resolve()
    require(args.root.is_absolute() and args.bundle.is_absolute() and args.output.is_absolute(),'absolute paths required')
    require(not output.exists(),'never overwrite an existing analysis')
    wall,cpu=time.monotonic(),time.process_time(); ledger={}
    raw=args.bundle.read_bytes(); originals,data,proofs=load(root,json.loads(raw),ledger)
    groups={g:[i for i,row in enumerate(originals) if g == 'ALL' or question_type(row) == g] for g in GROUPS}
    require(sum(len(groups[g]) for g in TYPES) == QA,'type partition dropped rows')
    summaries={}; acquisition={}; versus_seq={}; common={}; decomposition={}
    for group,indices in groups.items():
        selected={method:{stage:[data[method][stage][i] for i in indices] for stage in STAGES} for method in METHODS}
        summaries[group]={method:{str(stage):summarize(selected[method][stage]) for stage in STAGES} for method in METHODS}
        acquisition[group]={method:compare(selected[method][15],selected[method][16]) for method in METHODS}
        versus_seq[group]={method:{str(stage):compare(selected['SeqFT_LoRA'][stage],selected[method][stage]) for stage in STAGES}
            for method in METHODS[1:]}
        common[group]={str(stage):intersection({method:selected[method][stage] for method in METHODS}) for stage in STAGES}
        decomposition[group]={}
        for method in METHODS[1:]:
            before=versus_seq[group][method]['15']['mean_difference_pp']; after=versus_seq[group][method]['16']['mean_difference_pp']
            own=acquisition[group][method]['mean_difference_pp']; seq=acquisition[group]['SeqFT_LoRA']['mean_difference_pp']
            value=after-before if before is not None else None
            require(value is None or math.isclose(value,own-seq,abs_tol=1e-10,rel_tol=0),'difference decomposition')
            decomposition[group][method]=dict(acquired_method_minus_Seq=after,before_method_minus_Seq=before,
                difference_of_stage_changes_pp=value,descriptive_not_causal=True)
    unknown=Counter(unknown_type_identity(row)
                    for row in originals if question_type(row) == 'UNKNOWN')
    result=dict(complete=True,consumer_sha256=digest(Path(__file__).read_bytes()),bundle_sha256=digest(raw),
        proof_bindings=proofs,source_files=ledger,stages=dict(before=15,acquired=16),task=16,domain='finance',
        denominator=QA,question_type_denominators={g:len(v) for g,v in groups.items()},unknown_type_sha256_counts=dict(unknown),
        summaries=summaries,method_vs_Seq=versus_seq,before_to_acquired=acquisition,
        stage_change_decomposition=decomposition,three_method_intersections=common,
        ordered_TEST_IDs_sha256=digest(canonical([r['id'] for r in originals])),
        existing_score_values_consumed=len(METHODS)*len(STAGES)*QA,new_models=0,new_scores=0,new_generations=0,
        CP_payload_reads=0,raw_QA_or_predictions_in_output=False,strict_single_definition='raw fullmatch [A-D], no stripping or normalization',
        limitations=['Sealed TEST-only retrospective description; not a method/prompt selection procedure.',
            'F1=100 is full original token F1, not independently certified semantic correctness; zero is likewise a score observation.',
            'Multi-choice token F1 is not set accuracy and numerical token F1 is not numerical equivalence.',
            'More/fewer strict single-letter zero-F1 outputs do not identify option-position bias or its cause.',
            'Methods differ in training/adapter recipe; before/acquired comparisons do not isolate routing, capacity or data causes.'],
        wall_seconds_before_output=time.monotonic()-wall,process_CPU_seconds_before_output=time.process_time()-cpu)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(output),sha256=digest(output.read_bytes()),new_scores=0,existing_score_values_consumed=result['existing_score_values_consumed'])))


if __name__=='__main__': main()
