"""Consume four complete position readouts once; stdlib only, no QA or scorer.

Metadata and file hashes pass before any raw response is parsed. The public
OUTCOME contains only hashed IDs, response hashes, strict A-D labels and counts.
No model, tokenizer, gold, prompt, image or checkpoint payload is opened here.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import traceback


PHASES = [('pct01',15),('pct01',16),('pct10',15),('pct10',16)]
NAMES = [f'{condition}_S{stage}' for condition,stage in PHASES]
FILES = {'predictions.jsonl','MODEL_STATE.json','SOURCE_BINDING.json','RESOURCE_SUMMARY.json'}
DECISION = 'f2088a3fbed06d1dd2e6a328ece573fb1b74746c98aaaf1dfc60d4b764ad9cda'
WORKER = '6dc74f36d072029f6a0b054c4e671181d78fd269bac9c381407afd6e3a5b8bd1'
INVERSE = dict(zip('ABCD','DABC'))
ZERO = ('new_scores','optimizer_updates','backwards','TEST_QA','CPU_model_passes')


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,
                      separators=(',',':'),allow_nan=False).encode('utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def valid_sha(value):
    assert isinstance(value,str) and len(value) == 64
    assert all(x in '0123456789abcdef' for x in value)
    return value


def finite(value):
    assert isinstance(value,(int,float)) and not isinstance(value,bool)
    assert math.isfinite(value) and value >= 0
    return value


def ref_read(ref, certificate):
    path = Path(ref['path'])
    assert path.is_absolute() and path.is_file()
    raw = path.read_bytes()
    assert digest(raw) == ref['sha256']
    certificate[ref['sha256']] = {'bytes':len(raw)}
    return json.loads(raw)


def text_ref(ref, certificate):
    path = Path(ref['path'])
    assert path.is_absolute() and path.is_file()
    raw = path.read_bytes()
    assert digest(raw) == ref['sha256']
    certificate[ref['sha256']] = {'bytes':len(raw)}


def write(path, value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def ledger_ref(ledger, ref):
    entry = ledger[ref['path']]
    assert entry['sha256'] == ref['sha256'] and entry['bytes'] > 0


def metadata(admission_path, output):
    """No predictions JSON, source QA, images or model payloads are decoded."""
    certificate = {}
    admission_raw = admission_path.read_bytes()
    admission_sha = digest(admission_raw)
    admission = json.loads(admission_raw)
    assert admission['schema'] == 'finance-position-run-v1'
    assert output == Path(admission['output_path']).resolve()
    assert set(admission['sources']) == {'worker.py','controller.py','queue_adapter.py','verify.py'}
    source_dir = Path(admission.get('source_dir',str(Path(admission['root'])/'exp/finance_position_diagnostic_20261026')))
    assert source_dir.is_absolute()
    for name,expected in admission['sources'].items():
        valid_sha(expected)
        assert sha(source_dir/name) == expected
    assert sha(Path(__file__)) == admission['sources']['verify.py']
    assert admission['sources']['worker.py'] == WORKER
    binding = ref_read(admission['binding'],certificate)
    assert binding['schema'] == 'finance-position-diagnostic-v1'
    assert binding['worker_sha256'] == WORKER
    assert binding['protocol'] == admission['protocol']
    text_ref(binding['protocol'],certificate)
    decision = ref_read(binding['decision'],certificate)
    assert binding['decision']['sha256'] == DECISION
    ids = decision['accepted_id_hashes']
    assert len(ids) == len(set(ids)) == 45
    for key in ids:
        valid_sha(key)
    assert digest(canonical(ids)) == decision['accepted_ids_sha256']
    assert decision['counts'] == {'accepted':45,'excluded_fixed':7,'uncertain':1}
    receipt = ref_read(binding['inputs_receipt'],certificate)
    assert binding['inputs_receipt']['sha256'] == decision['evidence_sha256']['OPTION_ELIGIBILITY_INPUTS.json']
    assert receipt['private_pack_sha256'] == binding['masked_pack']['sha256']
    # The pack has question text. Its sealed digest is checked via each worker's
    # source ledger; the consumer deliberately does not open it or TRAIN JSONL.
    transform = ref_read(admission['transform_binding'],certificate)
    assert transform['decision_sha256'] == DECISION
    assert transform['private_pack_sha256'] == binding['masked_pack']['sha256']
    assert transform['strict_inverse_all'] and transform['actual_transformed_input_count'] == 45
    assert [x['id_sha256'] for x in transform['cases']] == ids
    transforms = {x['id_sha256']:{k:v for k,v in x.items() if k != 'id_sha256'} for x in transform['cases']}
    inventory = ref_read(binding['source_inventory'],certificate)
    states = {x['condition']:x for x in inventory['states']}
    proofs = {c:ref_read(binding['formal_proofs'][c],certificate) for c in ('pct01','pct10')}
    supervisor_path = output/'SUPERVISOR_COMPLETE.json'
    supervisor = read(supervisor_path)
    assert supervisor['admission_sha256'] == admission_sha
    assert supervisor['no_retry'] is True and supervisor['active_worker'] is None
    assert len(supervisor['phases']) == 4 and isinstance(supervisor['GPU'],int)
    assert supervisor['new_generations'] == 360
    assert supervisor['new_scores'] == supervisor['new_training_updates'] == 0
    assert not (output/'SUPERVISOR_FAILURE.json').exists()
    supervisor_started = read(output/'SUPERVISOR_STARTED.json')
    assert supervisor_started['admission_sha256'] == admission_sha
    assert supervisor_started['identity'] == supervisor['identity']
    phase_meta, common_layout, gpu, total_seconds = {}, None, None, 0.
    for (condition,stage), exit_row, name in zip(PHASES,supervisor['phases'],NAMES):
        assert (exit_row['condition'],exit_row['stage']) == (condition,stage)
        assert exit_row['returncode'] == 0 and exit_row['identity_absent'] is True
        assert exit_row['timeout'] is False
        phase_exit = read(output/('PHASE_'+name+'_EXIT.json'))
        assert phase_exit == {k:v for k,v in exit_row.items() if k != 'complete_sha256'}
        phase_start = read(output/('PHASE_'+name+'_STARTED.json'))
        assert (phase_start['condition'],phase_start['stage']) == (condition,stage)
        assert phase_start['identity'] == exit_row['identity'] and phase_start['GPU'] == supervisor['GPU']
        assert phase_start['deadline_unix']-phase_start['unix'] == 1200
        directory = output/name
        assert directory.is_dir() and not (directory/'FAILURE.json').exists()
        complete_path = directory/'COMPLETE.json'
        assert sha(complete_path) == exit_row['complete_sha256']
        complete = read(complete_path)
        assert complete['status'] == 'COMPLETE_POSITION_READOUT_ONLY'
        assert (complete['condition'],complete['stage']) == (condition,stage)
        assert complete['worker_sha256'] == WORKER
        assert complete['binding_sha256'] == admission['binding']['sha256']
        assert complete['protocol_sha256'] == binding['protocol']['sha256']
        assert complete['decision_sha256'] == DECISION and complete['ordered_45_ids'] == ids
        assert complete['new_generations'] == 90 and all(complete[k] == 0 for k in ZERO)
        assert complete['full_parameters_and_registered_buffers_unchanged'] is True
        assert complete['TRAIN_diagnostic_not_heldout_or_causal'] is True
        assert complete['quality_decision_deferred_until_all_four_states'] is True
        seconds = finite(complete['seconds']); assert seconds <= 1200
        total_seconds += seconds
        assert set(complete['files']) == FILES
        for filename,expected in complete['files'].items():
            assert sha(directory/filename) == expected
        started = read(directory/'STARTED.json')
        assert (started['condition'],started['stage']) == (condition,stage)
        assert started['worker_sha256'] == WORKER and started['binding_sha256'] == admission['binding']['sha256']
        assert started['limit_seconds'] == 1200
        identity = exit_row['identity']
        assert all(started[k] == identity[k] for k in ('pid','uid','pgid'))
        assert int(identity['startticks']) > 0 and identity['pid'] == identity['pgid']
        assert str(started['CUDA_VISIBLE_DEVICES']) == str(supervisor['GPU'])
        assert complete['GPU']['visible'] == str(supervisor['GPU']) and complete['GPU']['virtual'] == 0
        if gpu is None:
            gpu = complete['GPU']
        assert complete['GPU'] == gpu
        model = read(directory/'MODEL_STATE.json')
        assert model['unchanged'] is True and model['before'] == model['after']
        assert model['original_dtype_and_unfolded_graph'] is True
        assert model['generation_before'] == model['generation_after']
        assert model['original_structure'] == {'qv_modules':48,'adapter_tensors':96,
            'adapter_coefficients':540672,'rank':8,'scale':4,'dropout':.1,'unfolded_original_graph':True}
        layout = {key:model['before'][key] for key in ('weights','registered_buffers','parameter_flags')}
        if common_layout is None:
            common_layout = layout
        assert layout == common_layout
        for key in ('parameters_sha256','all_registered_buffers_sha256'):
            valid_sha(model['before'][key])
        source = read(directory/'SOURCE_BINDING.json')
        assert source['binding'] == binding
        assert source['source_inventory_observation_only_for_intermediate_payload'] is True
        ledger = source['source_files']
        for key in ('protocol','decision','inputs_receipt','masked_pack','source_inventory'):
            ledger_ref(ledger,binding[key])
        ledger_ref(ledger,binding['formal_proofs'][condition])
        state = states[condition]
        assert state['root'] == binding['data_roots'][condition]
        original = Path(state['root'])/state['run']
        config_ref = {'path':str(original/'config.json'),'sha256':state['config_sha256']}
        ledger_ref(ledger,config_ref)
        config = ref_read(config_ref,certificate)
        assert config['method'] == 'SeqFT_LoRA' and config['backbone'] == 'onevision'
        assert config['condition'] == condition and not config['preflight_only']
        assert config['epochs'] == 1 and config['seed'] == 42
        assert config['model_dtype'] == config['autocast_dtype'] == 'torch.float16'
        proof = proofs[condition]
        assert proof['candidate']['run'] == state['run'] and proof['config_sha256'] == state['config_sha256']
        observed = state['stages'][str(stage)]
        for kind in ('stages','checkpoints'):
            filename = f'{kind}/stage_{stage:02d}.json'
            ref = {'path':str(original/filename),'sha256':observed[filename]['sha256']}
            ledger_ref(ledger,ref)
            assert ref_read(ref,certificate) == observed[filename]['value']
        marker_key = str(Path(state['run'])/f'stages/stage_{stage:02d}.json')
        assert proof['source_sha256'][marker_key] == observed[f'stages/stage_{stage:02d}.json']['sha256']
        meta = observed[f'checkpoints/stage_{stage:02d}.json']['value']
        assert meta['stage'] == stage and meta['method'] == 'SeqFT_LoRA'
        assert meta['full_architecture_and_runtime'] is True and meta['optimizer_saved'] is False
        assert meta['path'] == f'checkpoints/stage_{stage:02d}.pt'
        assert complete['checkpoint_sha256'] == source['checkpoint_payload_SHA_checked_once_this_phase'] == meta['sha256']
        assert complete['checkpoint_bytes'] == model['checkpoint_bytes'] == meta['bytes'] == observed['checkpoint_stat']['bytes']
        assert complete['config_sha256'] == state['config_sha256']
        assert complete['manifest_sha256'] == config['manifest_sha256'] == meta['manifest_sha256']
        assert complete['formal_proof_sha256'] == binding['formal_proofs'][condition]['sha256']
        manifest_entries = [(path,item) for path,item in ledger.items()
                            if path.endswith('/manifest.json') and item['sha256'] == config['manifest_sha256']]
        assert len(manifest_entries) == 1
        manifest = ref_read({'path':manifest_entries[0][0],'sha256':config['manifest_sha256']},certificate)
        task = manifest['tasks'][15]
        assert task['task'] == 16 and task['domain'] == 'finance' and task['max_new_tokens'] == 512
        train = task['files']['train']
        ledger_ref(ledger,{'path':str(Path(state['root'])/train['path']),'sha256':train['sha256']})
        port = Path(state['root'])/'exp'/Path(state['run']).parts[1]
        for filename,expected in config['execution_source_sha256'].items():
            assert Path(filename).name == filename
            assert sha(port/filename) == sha(original/'runner_source'/filename) == expected
            certificate[expected] = {'kind':'formal_python_source'}
        assert set(source['image_files']) == {str(Path(root)/image)
            for root in binding['data_roots'].values() for image in binding['image_sha256']}
        for path,item in source['image_files'].items():
            matches = [image for image in binding['image_sha256'] if path.endswith('/'+image)]
            assert len(matches) == 1 and item['sha256'] == binding['image_sha256'][matches[0]]
            assert item['bytes'] > 0
        resource = read(directory/'RESOURCE_SUMMARY.json')
        assert resource['unit'] == 'bytes' and resource['rows'] > 0
        assert resource['no_cross_process_memory_subtraction'] is True
        for value in resource['observed_peak_bytes'].values():
            if value is not None:
                finite(value)
        phase_meta[name] = {'complete':complete,'complete_sha256':sha(complete_path),
                            'directory':directory,'task':task,'source':source}
    assert total_seconds <= 4800
    return admission_sha, admission, binding, ids, transforms, phase_meta, certificate, total_seconds


def prompt_digest(ids):
    assert isinstance(ids,list) and ids and all(isinstance(x,int) and x >= 0 for x in ids)
    meta = canonical(['input_ids','torch.int64',[1,len(ids)]])
    return digest(len(meta).to_bytes(8,'little')+meta+struct.pack('<'+'q'*len(ids),*ids))


def parse_predictions(phase_meta, binding, ids, transforms):
    """Called only after all four states and supervisor pass metadata gates."""
    results, input_reference, feature_evidence = {}, {}, {}
    for (condition,stage),name in zip(PHASES,NAMES):
        item = phase_meta[name]
        lines = (item['directory']/'predictions.jsonl').read_bytes().splitlines()
        assert len(lines) == 90 and all(lines)
        records = [json.loads(line) for line in lines]
        result, total_tokens = {}, 0
        for index,key in enumerate(ids):
            pair = records[2*index:2*index+2]
            labels, raw_hashes = {}, {}
            for view,row in zip(('original','cyclic'),pair):
                assert (row['condition'],row['stage'],row['index'],row['id_sha256'],row['view']) == (condition,stage,index+1,key,view)
                assert digest(row['source_id'].encode()) == key
                assert row['source_bytes_include_gold'] is True and row['gold_used_for_generation_or_selection'] is False
                assert row['scores_computed'] == 0 and not row['input_details']['retrieved_chunk_indexes']
                assert row['generation_cap'] == 512 and row['greedy'] is True
                assert row['num_beams'] == 1 and row['use_cache'] is True
                assert row['output_tokens'] == len(row['output_token_ids']) <= 512
                total_tokens += row['output_tokens']; finite(row['generation_seconds'])
                assert row['transform'] == transforms[key] and row['transform']['strict_inverse_exact'] is True
                assert row['context_sha256'] == transforms[key][view+'_context_sha256']
                assert row['instruction_sha256'] == digest(item['task']['instruction'].encode())
                assert all(binding['image_sha256'][image] == value for image,value in row['image_sha256'].items())
                assert row['image_identifiers_sha256'] == digest(json.dumps(list(row['image_sha256'])).encode())
                for field in ('source_train_line_sha256','common_QA_canonical_sha256',
                              'question_sha256','context_sha256','instruction_sha256','raw_input_sha256'):
                    valid_sha(row[field])
                prepared = row['prepared_tensors']
                assert row['prepared_prompt_sha256'] == prepared['input_ids']['sha256'] == prompt_digest(row['prepared_prompt_token_ids'])
                assert prepared['input_ids']['dtype'] == 'torch.int64'
                assert prepared['input_ids']['shape'] == [1,len(row['prepared_prompt_token_ids'])]
                assert all(value['device'] == 'cuda:0' for value in prepared.values())
                for value in prepared.values():
                    valid_sha(value['sha256'])
                assert row['visual_features_sha256'] == prepared.get('pixel_values',{}).get('sha256')
                if row['image_sha256']:
                    assert 'pixel_values' in prepared and 'image_grid_thw' in prepared
                else:
                    assert 'pixel_values' not in prepared
                invariant = {k:row[k] for k in ('source_train_line_sha256','common_QA_canonical_sha256',
                    'question_sha256','context_sha256','instruction_sha256','image_identifiers_sha256',
                    'image_sha256','raw_input_sha256','prepared_prompt_sha256',
                    'pad_token_id','eos_token_id')}
                invariant['prepared_layout'] = {name:{k:value[k] for k in ('shape','dtype','device')}
                                                for name,value in prepared.items()}
                ref_key = (key,view)
                if ref_key not in input_reference:
                    input_reference[ref_key] = invariant
                assert invariant == input_reference[ref_key], 'Input/prompt/image identity or prepared layout differs across states'
                # Frozen caches were populated by their original training runs.
                # Cross-trajectory feature bytes are observed, not equated by
                # clearing caches or imposed as an extra model-result gate.
                feature_evidence.setdefault(key,{}).setdefault(view,{})[name] = {
                    field:prepared.get(field,{}).get('sha256')
                    for field in ('pixel_values','image_grid_thw')}
                raw = row['raw_prediction']; assert isinstance(raw,str)
                label = raw.strip()
                labels[view] = label if label in ('A','B','C','D') else None
                raw_hashes[view] = digest(raw.encode())
            for field in ('source_id','source_train_line_sha256','common_QA_canonical_sha256','question_sha256',
                          'instruction_sha256','image_identifiers_sha256','image_sha256','visual_features_sha256'):
                assert pair[0][field] == pair[1][field]
            for field in ('pixel_values','image_grid_thw'):
                assert pair[0]['prepared_tensors'].get(field) == pair[1]['prepared_tensors'].get(field)
            original,cyclic = labels['original'],labels['cyclic']
            both = original is not None and cyclic is not None
            status = ('both_mapped' if both else 'original_only' if original is not None
                      else 'cyclic_only' if cyclic is not None else 'neither')
            result[key] = {'id_sha256':key,'raw_prediction_sha256':raw_hashes,
                'strict_labels':labels,'cyclic_original_content_label':INVERSE[cyclic] if cyclic else None,
                'coverage':status,'content_same':original == INVERSE[cyclic] if both else None,
                'same_raw_letter':original == cyclic if both else None}
        assert total_tokens == item['complete']['output_tokens']
        results[name] = result
    feature_matches = {'reference_phase':NAMES[0], 'full_QA_denominator':45,
        'same_original_cyclic_pair_visual_in_every_phase':True,
        'cross_state_layout_shape_dtype_device_equal':True,
        'cross_state_feature_bytes_equal_is_not_a_gate':True,
        'per_id':[{'id_sha256':key,'views':{view:{
            'feature_sha256_by_phase':feature_evidence[key][view],
            'matches_reference_by_phase':{name:feature_evidence[key][view][name] == feature_evidence[key][view][NAMES[0]]
                                          for name in NAMES},
            'all_four_feature_signatures_equal':all(feature_evidence[key][view][name] == feature_evidence[key][view][NAMES[0]]
                                                    for name in NAMES)}
            for view in ('original','cyclic')}} for key in ids]}
    feature_matches['all_four_signatures_equal_QA_count'] = sum(
        all(row['views'][view]['all_four_feature_signatures_equal'] for view in ('original','cyclic'))
        for row in feature_matches['per_id'])
    return results, feature_matches


def phase_counts(rows):
    counts = Counter(x['coverage'] for x in rows)
    both = [x for x in rows if x['coverage'] == 'both_mapped']
    return {'full_denominator':len(rows),
        **{k:counts[k] for k in ('both_mapped','original_only','cyclic_only','neither')},
        'content_same':sum(x['content_same'] for x in both),
        'content_changed':sum(not x['content_same'] for x in both),
        'same_raw_letter':sum(x['same_raw_letter'] for x in both),
        'different_raw_letter':sum(not x['same_raw_letter'] for x in both),
        'original_A_D_counts':{k:sum(x['strict_labels']['original'] == k for x in rows) for k in 'ABCD'},
        'cyclic_A_D_counts':{k:sum(x['strict_labels']['cyclic'] == k for x in rows) for k in 'ABCD'}}


def comparison(first,second,ids):
    eligible = [key for key in ids if first[key]['content_same'] is not None and second[key]['content_same'] is not None]
    transitions = Counter(('stable' if first[key]['content_same'] else 'changed')+'_to_'+
                          ('stable' if second[key]['content_same'] else 'changed') for key in eligible)
    a = {key for key in ids if first[key]['content_same'] is not None}
    b = {key for key in ids if second[key]['content_same'] is not None}
    return {'full_denominator':len(ids),'all_four_outputs_valid':len(eligible),
        'only_first_both_valid':len(a-b),'only_second_both_valid':len(b-a),
        'neither_state_both_valid':len(set(ids)-(a|b)),
        'eligible_id_sha256':eligible,
        'transitions':{key:transitions[key] for key in ('stable_to_stable','stable_to_changed','changed_to_stable','changed_to_changed')},
        'first_content_changed':sum(not first[key]['content_same'] for key in eligible),
        'second_content_changed':sum(not second[key]['content_same'] for key in eligible)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert args.admission.is_absolute() and args.output.is_absolute()
    output = args.output.resolve(); assert output.is_dir()
    assert not any((output/name).exists() for name in ('VERIFY_STARTED.json','VERIFY_FAILURE.json','VERIFIED.json','OUTCOME.json'))
    start = time.monotonic()
    admission_sha,admission,binding,ids,transforms,phases,certificate,model_seconds = metadata(args.admission,output)
    write(output/'VERIFY_STARTED.json',{'unix':time.time(),'pid':os.getpid(),
        'admission_sha256':admission_sha,'consumer_sha256':sha(Path(__file__)),
        'all_four_metadata_gates_passed':True,'planned_raw_responses':360,'new_scores':0,
        'no_model_tokenizer_QA_image_or_checkpoint_payload_reads':True})
    try:
        results,feature_matches = parse_predictions(phases,binding,ids,transforms)
        all_eight = [key for key in ids if all(results[name][key]['content_same'] is not None for name in NAMES)]
        outcome = {'status':'COMPLETE_DESCRIPTIVE_TRAIN_POSITION_DIAGNOSTIC',
            'ordered_id_sha256':ids,'raw_responses':360,'full_QA_denominator':45,
            'phases':{name:{'counts':phase_counts(list(results[name].values())),
                           'per_id':[results[name][key] for key in ids]} for name in NAMES},
            'S15_to_S16':{condition:comparison(results[condition+'_S15'],results[condition+'_S16'],ids)
                          for condition in ('pct01','pct10')},
            'pct01_to_pct10':{str(stage):comparison(results[f'pct01_S{stage}'],results[f'pct10_S{stage}'],ids)
                              for stage in (15,16)},
            'all_eight_outputs_valid':{'full_denominator':45,'eligible':len(all_eight),
                'eligible_id_sha256':all_eight,
                'phase_counts':{name:phase_counts([results[name][key] for key in all_eight]) for name in NAMES},
                'S15_to_S16':{condition:comparison(results[condition+'_S15'],results[condition+'_S16'],all_eight)
                              for condition in ('pct01','pct10')},
                'pct01_to_pct10':{str(stage):comparison(results[f'pct01_S{stage}'],results[f'pct10_S{stage}'],all_eight)
                                  for stage in (15,16)}},
            'new_scores':0,'gold_correctness_evaluated':False,
            'cross_state_feature_matches':{
                'full_QA_denominator':45,
                'all_four_signatures_equal_QA_count':feature_matches['all_four_signatures_equal_QA_count'],
                'different_QA_count':45-feature_matches['all_four_signatures_equal_QA_count'],
                'differences_are_cross_trajectory_interpretation_limits_not_a_result_gate':True},
            'two_invalid_outputs_count_as_stable':False,
            'parse_rule':'raw_prediction.strip() strictly in uppercase A/B/C/D; no rescue',
            'cyclic_inverse':INVERSE,'stable_content_does_not_mean_correct':True,
            'single_seed_single_cycle_explored_TRAIN_subset_not_causal_or_heldout':True}
        # Recheck small sealed records and prediction bytes without a second
        # parse/consumption before committing the unique public result.
        assert sha(args.admission) == admission_sha
        for name,item in phases.items():
            assert sha(item['directory']/'COMPLETE.json') == item['complete_sha256']
            for filename,expected in item['complete']['files'].items():
                assert sha(item['directory']/filename) == expected
        write(output/'OUTCOME.json',outcome)
        write(output/'VERIFIED.json',{'status':'VERIFIED_ONCE_NO_QUALITY_SCORING',
            'admission_sha256':admission_sha,'consumer_sha256':admission['sources']['verify.py'],
            'binding_sha256':admission['binding']['sha256'],
            'supervisor_complete_sha256':sha(output/'SUPERVISOR_COMPLETE.json'),
            'phase_complete_sha256':{name:item['complete_sha256'] for name,item in phases.items()},
            'phase_file_sha256':{name:item['complete']['files'] for name,item in phases.items()},
            'public_source_certificate':{'source_sha256':admission['sources'],
                'bound_metadata_sha256':certificate,'transform_binding_sha256':admission['transform_binding']['sha256'],
                'decision_sha256':DECISION,'formal_proof_sha256':{k:v['sha256'] for k,v in binding['formal_proofs'].items()}},
            'outcome_sha256':sha(output/'OUTCOME.json'),'raw_responses_consumed_once':360,
            'cross_state_feature_matches':feature_matches,
            'new_scores':0,'new_models':0,'model_seconds_sum':model_seconds,
            'model_seconds_overlap_not_added_again':True,'consumer_seconds':time.monotonic()-start,
            'no_QA_image_checkpoint_payload_or_tokenizer_reads':True})
    except BaseException:
        write(output/'VERIFY_FAILURE.json',{'unix':time.time(),'traceback':traceback.format_exc(),
            'admission_sha256':admission_sha,'no_automatic_reconsume':True,'new_scores':0})
        raise


if __name__ == '__main__':
    main()
