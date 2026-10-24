"""One read-only financial label/options/context contract check; no scoring."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PROOFS = {'SeqFT_LoRA':'c0c010f18db89b088cd3ac13b92e79a76c560c13ee9f9960f0e5e2ddc8e88c69',
          'OSFT':'f8612d49ff55bcd3678365cbd193bd7dc2315c570e522a4a9edc095eddc52a5d',
          'PASs_MoE':'a17c1000dfff6816644d3e86170fe255d6c3225eebfa911d4c0834fcabb0dc9f'}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):return re.findall(r'[a-z0-9]+|[\u3400-\u9fff]',text.lower())


def options(context):
    # Same strict input-only parser as the earlier audit; no output-to-letter mapping.
    matches=list(re.finditer(r'(?m)^([A-Z])\s*:\s*',context))
    if len(matches)<2 or context[:matches[0].start()].strip():return None
    letters=[m.group(1) for m in matches]
    if letters!=[chr(ord('A')+i) for i in range(len(letters))]:return None
    result={m.group(1):context[m.end():matches[i+1].start() if i+1<len(matches) else len(context)].strip() for i,m in enumerate(matches)}
    values=[tuple(norm(t)) for t in result.values()]
    if any(not v for v in values) or len(set(values))!=len(values):return None
    return result


METADATA_READER = r'''from pathlib import Path
import ast,hashlib,json,sys
root=Path('/data2/liyapeng_grp/program/LLMCL');bundle=json.loads(sys.stdin.read());result={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
for method,record in bundle.items():
 proof=record['proof'];run=root/proof['candidate']['run']
 config_path=run/'config.json';assert sha(config_path)==proof['config_sha256']
 config=json.loads(config_path.read_text());assert config['method']==method and config['backbone']=='onevision'
 path=run/'runner_source/runtime_utils.py';assert sha(path)==proof['source_sha256'][str(path.relative_to(root))]
 fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='input_batch')
 assert hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()==record['expected_input_batch_AST_sha256']
 stages={}
 for stage in (0,16,19):
  relative=f'evaluations/stage_{stage:02d}/T16_test_predictions.jsonl';p=run/relative
  assert sha(p)==proof['prediction_sha256'][relative]
  rows=[json.loads(line) for line in p.read_text().splitlines()]
  stages[str(stage)]=dict(prediction_sha256=proof['prediction_sha256'][relative],rows=[dict(id=r['id'],
   references_sha256=digest(r['references']),metadata_options_sha256=digest(r.get('metadata',{}).get('options')),
   retrieved_chunk_indexes=r.get('retrieved_chunk_indexes'),prompt_tokens=r.get('prompt_tokens')) for r in rows])
 result[method]=dict(runtime_source_sha256=sha(path),input_batch_AST_sha256=record['expected_input_batch_AST_sha256'],
  config_sha256=proof['config_sha256'],manifest_sha256=config['manifest_sha256'],stages=stages)
assert 'torch' not in sys.modules
print(json.dumps(result))
'''


def main(root,output):
    begin=time.monotonic();assert not output.exists() and 'torch' not in sys.modules
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('120-second contract cap')))
    signal.alarm(120)
    index=json.loads((root/'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json').read_text())
    current=root/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py'
    fn=next(n for n in ast.parse(current.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='input_batch')
    expected=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()
    bundle={};proof_sources={}
    for method,expected_proof in PROOFS.items():
        entry=next(x for x in index['methods'] if x['method']==method and x['backbone']=='onevision')
        path=root/entry['proof_file'];assert sha(path)==expected_proof
        proof=json.loads(path.read_text());assert proof['final_QA']==21454
        assert proof['candidate']['method']==method and proof['candidate']['backbone']=='onevision'
        bundle[method]=dict(proof=proof,expected_input_batch_AST_sha256=expected)
        proof_sources[method]=dict(path=entry['proof_file'],sha256=expected_proof)
    policy=json.loads((root/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json').read_text())
    manifest_path=root/policy['conditions']['pct01']['manifest'];manifest=json.loads(manifest_path.read_text())
    task=manifest['tasks'][15];assert task['task']==16 and task['domain']=='finance'
    data={};sources={}
    for split,count in (('train',89),('test',2233)):
        info=task['files'][split];path=root/info['path'];assert sha(path)==info['sha256']
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows)==count==info['count'] and len({r['id'] for r in rows})==count
        data[split]=rows;sources[split]=dict(path=info['path'],sha256=info['sha256'],count=count)
    wanted={r['id'] for rows in data.values() for r in rows};assert len(wanted)==2322
    import pyarrow.parquet as pq
    originals={};rawfiles=[]
    for path in sorted((root/'data/FinMME/raw/data').glob('*.parquet')):
        parquet=pq.ParquetFile(path);rawfiles.append(dict(path=path.relative_to(root).as_posix(),file_bytes=path.stat().st_size,rows=parquet.metadata.num_rows,columns_read=['id','question_text','options','answer']))
        for batch in parquet.iter_batches(batch_size=128,columns=['id','question_text','options','answer']):
            for row in batch.to_pylist():
                key='FinMME:'+str(row['id'])
                if key in wanted:
                    assert key not in originals;originals[key]=row
    assert rawfiles and set(originals)==wanted
    source_counts={};problems=[]
    for split,rows in data.items():
        counts=Counter();labels=Counter();types=Counter();max_context=0
        for row in rows:
            original=originals[row['id']];context=row.get('context','');max_context=max(max_context,len(context))
            original_answer=original['answer'] or ''
            expected_answers=original_answer if isinstance(original_answer,list) else [original_answer]
            comparisons=dict(question=row['question']==original['question_text'],context=context==(original['options'] or ''),
                metadata_options=row['metadata']['options']==original['options'],answer=row['answer']==expected_answers[0],
                all_answers=row['answers']==expected_answers)
            for key,ok in comparisons.items():
                counts[key+'_matches_raw']+=ok
                if not ok:problems.append(dict(id=row['id'],split=split,reason=key))
            counts['empty_context']+=not bool(context);counts['context_over_6000']+=len(context)>6000
            parsed=options(context);counts['strict_options_parseable']+=parsed is not None
            single=bool(re.fullmatch('[A-Z]',row['answer']))
            counts['single_letter_gold']+=single;counts['not_single_letter_gold']+=not single
            if single:labels[row['answer']]+=1
            if parsed is not None and single:
                valid=row['answer'] in parsed;counts['single_letter_gold_with_parsed_options']+=1
                counts['single_letter_gold_not_in_parsed_options']+=not valid
                if not valid:problems.append(dict(id=row['id'],split=split,reason='gold_not_in_parsed_options'))
            types[str(row['metadata'].get('question_type'))]+=1
        source_counts[split]=dict(QA=len(rows),counts=dict(counts),single_letter_gold_distribution=dict(labels),
            original_question_type_counts=dict(types),maximum_context_characters=max_context,
            ID_order_sha256=digest([r['id'] for r in rows]))
    cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8','-i',str(root/'.deployment/old_pct01_id_ed25519'),'-p','8666','liyapeng_grp@10.254.29.8',
         '/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python','-c',METADATA_READER]
    # A script sent through stdin keeps shell quoting independent of data and code.
    remote_code=METADATA_READER.replace("bundle=json.loads(sys.stdin.read())",'bundle=json.loads('+repr(json.dumps(bundle))+')')
    cmd=cmd[:-2]+['-']
    remote=subprocess.run(cmd,input=remote_code.encode(),capture_output=True,timeout=75)
    assert remote.returncode==0,remote.stderr.decode();observed=json.loads(remote.stdout)
    input_checks={};test=data['test'];test_ids=[r['id'] for r in test]
    for method,record in observed.items():
        assert record['manifest_sha256']==sha(manifest_path)
        stages={}
        for stage,proof_rows in record['stages'].items():
            saved=proof_rows['rows'];assert [r['id'] for r in saved]==test_ids
            counts=Counter()
            for row,snapshot in zip(test,saved):
                context=row.get('context','');indices=snapshot['retrieved_chunk_indexes']
                references_match=snapshot['references_sha256']==digest(row['answers'])
                metadata_match=snapshot['metadata_options_sha256']==digest(row['metadata']['options'])
                counts['references_match']+=references_match;counts['metadata_options_match']+=metadata_match
                if not references_match:problems.append(dict(id=row['id'],method=method,stage=stage,reason='saved_reference_mismatch'))
                if not metadata_match:problems.append(dict(id=row['id'],method=method,stage=stage,reason='saved_options_metadata_mismatch'))
                if indices is None:counts['missing_retrieval_metadata']+=1;continue
                assert isinstance(indices,list) and all(isinstance(i,int) for i in indices)
                if len(context)<=6000:
                    counts['short_context_saved_no_retrieval']+=not indices
                    if indices:problems.append(dict(id=row['id'],method=method,stage=stage,reason='unexpected_retrieval'))
                    retained=context if not indices else None
                else:
                    chunks=[context[start:start+1000] for start in range(0,len(context),800)]
                    assert all(0<=i<len(chunks) for i in indices)
                    retained='\n'.join(chunks[i] for i in indices);counts['long_context_reconstructed_from_saved_indices']+=1
                parsed=options(context)
                if parsed is not None:
                    counts['strict_options_checked_for_retention']+=1
                    all_present=retained is not None and all(text in retained for text in parsed.values())
                    counts['all_strict_option_texts_retained']+=all_present
                    if not all_present:problems.append(dict(id=row['id'],method=method,stage=stage,reason='strict_option_text_missing'))
                    headers=list(re.finditer(r'(?m)^([A-Z])\s*:\s*',context))
                    spans=[context[m.start():headers[i+1].start() if i+1<len(headers) else len(context)].strip() for i,m in enumerate(headers)]
                    assert len(spans)==len(parsed)
                    bindings=retained is not None and all(span in retained for span in spans)
                    counts['all_strict_label_content_spans_preserved_verbatim']+=bindings
                    if not bindings:problems.append(dict(id=row['id'],method=method,stage=stage,reason='strict_label_content_span_not_preserved_verbatim'))
                counts['positive_recorded_prompt_tokens']+=isinstance(snapshot['prompt_tokens'],int) and snapshot['prompt_tokens']>0
            stages[stage]=dict(QA=len(saved),counts=dict(counts),prediction_sha256=proof_rows['prediction_sha256'])
        input_checks[method]=dict(runtime_source_sha256=record['runtime_source_sha256'],input_batch_AST_sha256=record['input_batch_AST_sha256'],
            config_sha256=record['config_sha256'],manifest_sha256=record['manifest_sha256'],stages=stages)
    private=root/'.deployment/finance_input_contract_20261024'
    private.mkdir(parents=True,exist_ok=True)
    details=private/'anomalies.json';assert not details.exists();details.write_text(json.dumps(problems,indent=2)+'\n')
    value=dict(complete=True,source_checks=source_counts,existing_prediction_input_checks=input_checks,
        frozen_data_sources=sources,manifest_sha256=sha(manifest_path),proof_sources=proof_sources,
        current_raw_parquet_files=rawfiles,current_selected_four_column_rows_sha256=digest(originals),
        anomaly_count=len(problems),anomaly_ID_reason_record_sha256=sha(details),
        missing_saved_retrieval_records=sum(s['counts'].get('missing_retrieval_metadata',0) for r in input_checks.values() for s in r['stages'].values()),
        historical_full_tokenized_input_equality_proven=False,actual_TRAIN_prompt_record_available=False,
        consumer_sha256=sha(Path(__file__)),current_input_batch_AST_sha256=expected,
        wall_seconds=time.monotonic()-begin,models_run=0,QA_generations=0,new_scores=0,new_training_updates=0,
        historical_raw_parquet_full_file_SHA_not_reverified=True,
        limits=['Current raw four-column projection agrees or disagrees with immutable frozen QA; no historical raw-file hash is claimed.',
                'Saved retrieval metadata and source-level prompt construction do not prove historical full tokenized-input equality; prompt IDs and strings were not saved.',
                'TRAIN prompt inclusion is a static-source/context-length check, not replay of actual past tokenizer calls.',
                'A missing saved retrieval record is unknown, not passed. Non-preserved verbatim label/content spans signal a contract needing inspection, not necessarily semantic option loss.',
                'Strict parser is incomplete; unparseable/non-single-letter records are separate, not errors.',
                'Gold distribution does not establish why the model chooses a letter; no TEST-based recipe or score modification.'])
    assert 'torch' not in sys.modules;output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');signal.alarm(0)
    print(json.dumps(dict(complete=True,evidence_sha256=sha(output),source_checks=source_counts,anomaly_count=len(problems),wall_seconds=value['wall_seconds']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root.resolve(),args.output)
