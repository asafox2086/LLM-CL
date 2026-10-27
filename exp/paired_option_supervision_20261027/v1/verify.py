"""Verify four complete GPU phases, then score 600 original answers once.

Stdlib only. No model/processor import, deserialization, generation or TEST.
The original runtime tokens/score definitions are compiled unchanged from AST.
"""
import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

ARMS = ('repeat','paired')
ORDER = [(arm,phase) for arm in ARMS for phase in ('train','eval')]
DOMAINS = ('natural','medical','geography','finance','traffic')
LIMITS = {'train':5400,'eval':1800}
WORKER = 'bad0ffd6873890202f27e1b8d8688c6d042f5a4ce01f0e0de118d6c701c70481'
HELPERS = 'f4bb0df32897a075330443f375e302e2bf71162b8ef1918a6023e62ca006a0ba'
ADAPTATION = '251a2d490f9ce29069de141686a16da549c28de82f7c1ce58fab2df8da31252a'
MANIFEST = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
SPLIT = 'd8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
TRAIN_AST = '2b31ff9e06234e967ea42d3cb6d594c82050054c7c2c058d77f61bf8100dda7f'
SCORER_FILE = '4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240'
SCORER_DEFINITIONS = '60830848dd0ed604359d7a34c64dfa31eaf736b724bb6258f8a795676426162a'
RECEIPT_KEYS = ('layout','digest','all_buffers_digest','parameter_alias_groups',
    'buffer_alias_groups','buffer_manifest','functional_config_digest',
    'generation_config_digest','language_generation_config_digest',
    'language_attention_classes','vision_attention_classes','method')


def read(path):
    return json.loads(Path(path).read_bytes())


def process_identity(pid):
    try:
        path = Path('/proc')/str(pid)
        fields = (path/'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z': return None
        return dict(pid=int(pid), startticks=fields[19], uid=path.stat().st_uid, pgid=int(fields[2]))
    except (OSError, ValueError): return None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite(value):
    assert isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    return value


def hash_string(value):
    assert isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value)


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


class Files:
    """Stream each immutable checkpoint once; never unpickle any tensor."""
    def __init__(self):
        self.hashes,self.stamps = {},{}

    def checked(self,path,expected=None):
        path = Path(path)
        st = path.stat(); stamp = (st.st_size,st.st_mtime_ns)
        if path in self.stamps:
            assert self.stamps[path] == stamp,'Previously checked artifact changed'
            value = self.hashes[str(path)]
        else:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    h.update(block)
            value = h.hexdigest()
            assert (path.stat().st_size,path.stat().st_mtime_ns) == stamp
            self.stamps[path],self.hashes[str(path)] = stamp,value
        assert expected is None or value == expected,str(path)
        return path

    def json(self,path,expected=None):
        return read(self.checked(path,expected))

    def unchanged(self):
        assert all((p.stat().st_size,p.stat().st_mtime_ns) == stamp for p,stamp in self.stamps.items())


def inside(root,name):
    p = Path(name)
    assert not p.is_absolute() and '..' not in p.parts
    return root/p


def receipt(value,arm,deployed):
    mode = value['method']
    explicit = not deployed
    assert mode['arm'] == 'shared' and mode['deployed'] is deployed and mode['explicit_AB'] is explicit
    assert mode['split'] == [4,2,2] and mode['scale'] == 4 and mode['dropout'] == .1
    assert mode['soft_temperature'] == 1 and mode['extra_router_parameters'] == 0
    assert mode['training_coordinates'] == 540672
    expected_plan = {f'model.layers.{i}.self_attn.{p}_proj':8 for i in range(24) for p in ('q','v')}
    assert mode['rank_plan'] == expected_plan
    assert mode['routing'] == 'none'
    assert value['layout']['weights'] == {'coefficients':894216224 if explicit else 893675552,
        'bytes':1789513792 if explicit else 1787351104}
    assert value['layout']['ordinary_buffers']['bytes'] == 9032 and value['layout']['trainable'] == 0
    for key in ('digest','all_buffers_digest','functional_config_digest',
                'generation_config_digest','language_generation_config_digest'):
        hash_string(value[key])
    assert all(x['dtype'] == 'torch.float32' for n,x in value['buffer_manifest'].items() if n.endswith('inv_freq'))


def complete_metadata(root,run,files,binding_path):
    supervisor = files.json(run/'SUPERVISOR_COMPLETE.json')
    assert not (run/'SUPERVISOR_FAILURE.json').exists()
    assert supervisor['no_retry'] is True and supervisor['active_worker'] is None
    assert supervisor['new_generations'] == 634 and supervisor['new_scores'] == supervisor['TEST_QA'] == 0
    assert supervisor['FIT_backwards'] == 2426 and supervisor['optimizer_updates'] == 174
    assert supervisor['cumulative_model_budget_seconds'] == 14400
    assert len(supervisor['phases']) == 4
    started = files.json(run/'SUPERVISOR_STARTED.json')
    assert started['identity'] == supervisor['identity'] and started['GPU'] == supervisor['GPU']
    assert started['source_sha256'] == supervisor['source_sha256']
    assert started['protocol'] == supervisor['protocol']
    assert process_identity(supervisor['identity']['pid']) != supervisor['identity']
    assert started['no_retry'] and started['existing_allocator_supplied_GPU']
    source_dir = Path(supervisor['source_dir'])
    assert source_dir == root/'exp/paired_option_supervision_20261027/v1'
    sources = supervisor['source_sha256']
    assert set(sources) == {'controller.py','worker.py','helpers.py'}
    assert sources['worker.py'] == WORKER and sources['helpers.py'] == HELPERS
    for name,h in sources.items():
        files.checked(source_dir/name,h)
    binding = files.json(binding_path)
    assert started['binding_sha256'] == supervisor['binding_sha256'] == files.hashes[str(binding_path)]
    assert binding['schema'] == 'paired-option-supervision-v1' and binding['limits'] == LIMITS
    assert binding['sources'] == {'worker.py':WORKER, 'helpers.py':HELPERS}
    assert binding['protocol'] == supervisor['protocol']
    spec = importlib.util.spec_from_file_location('verified_paired_option_helpers',source_dir/'helpers.py')
    helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
    assert helpers.bind_sources(root,source_dir,binding) == LIMITS
    files.checked(Path(supervisor['protocol']['path']),supervisor['protocol']['sha256'])
    manifest_path = root/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    split_path = root/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
    manifest,split = files.json(manifest_path,MANIFEST),files.json(split_path,SPLIT)
    tasks = manifest['tasks']
    assert [t['task'] for t in tasks] == list(range(1,20)) and manifest['condition'] == 'pct01'
    assert manifest['seed'] == 42 and sum(t['files']['train']['count'] for t in tasks) == 1313
    assert set(t['domain'] for t in tasks) == set(DOMAINS)
    fit_ids,dev_ids = {},{}
    for task in tasks:
        stage = task['task']
        fit_ids[stage],dev_ids[stage] = split['fit_ids'][str(stage)],split['dev_ids'][str(stage)]
        assert len(set(fit_ids[stage]+dev_ids[stage])) == len(fit_ids[stage])+len(dev_ids[stage])
        assert len(fit_ids[stage])+len(dev_ids[stage]) == task['files']['train']['count']
    assert sum(map(len,fit_ids.values())) == 1213 and sum(map(len,dev_ids.values())) == 100
    assert len({i for v in dev_ids.values() for i in v}) == 100
    assert all(sum(len(dev_ids[t['task']]) for t in tasks if t['domain'] == d) == 20 for d in DOMAINS)
    original,fit,dev = {},{},{}
    for task in tasks:
        entry = task['files']['train']; path = files.checked(inside(root,entry['path']),entry['sha256'])
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        by_id = {r['id']:r for r in rows}
        assert len(rows) == len(by_id) == entry['count']
        assert set(fit_ids[task['task']]+dev_ids[task['task']]) == set(by_id)
        fit[task['task']] = [by_id[i] for i in fit_ids[task['task']]]
        dev[task['task']] = [by_id[i] for i in dev_ids[task['task']]]
        assert not set(original).intersection(by_id)
        original.update(by_id)
    ledger = {}
    selected,cyclic,option_proofs = helpers.eligibility(root,manifest,fit,dev,binding,ledger)
    for path,h in ledger.items(): files.checked(Path(path),h)
    options = dict(selected=selected,cyclic=cyclic,proofs=option_proofs,original=original)
    expected_binding = dict(manifest_path=str(manifest_path),manifest_sha256=MANIFEST,
        split_path=str(split_path),split_sha256=SPLIT,FIT_QA=1213,DEV_QA=100,original_TRAIN_QA=1313,TEST_QA=0,
        eligible_FIT=33,eligible_DEV=12,scope_sha256=helpers.SCOPE_SHA,
        decision_sha256=helpers.DECISION_SHA,binding_sha256=files.hashes[str(binding_path)])
    phases,identities,previous_exit = {},set(),None
    for (arm,phase),row in zip(ORDER,supervisor['phases']):
        assert (row['arm'],row['phase']) == (arm,phase)
        assert row['returncode'] == 0 and row['identity_absent'] is True and row['timeout'] is False
        identity = row['identity']; key = (identity['pid'],int(identity['startticks']))
        assert key not in identities and identity['pid'] == identity['pgid']
        identities.add(key)
        assert process_identity(identity['pid']) != identity
        directory = run/arm
        assert Path(row['run_dir']) == directory
        outer_start = files.json(run/f'PHASE_{arm}_{phase}_STARTED.json')
        outer_exit = files.json(run/f'PHASE_{arm}_{phase}_EXIT.json')
        assert outer_exit == {k:v for k,v in row.items() if k not in ('started_sha256','complete_sha256','checkpoint')}
        assert outer_start['identity'] == identity and outer_start['GPU'] == supervisor['GPU']
        assert outer_start['arm'] == arm and outer_start['phase'] == phase
        assert outer_start['limit_seconds'] == LIMITS[phase]
        command = outer_start['command']
        assert command.count('--binding') == 1 and Path(command[command.index('--binding')+1]) == binding_path
        if previous_exit is not None:
            assert outer_start['unix'] >= previous_exit
        previous_exit = row['exit_unix']
        actual_start = files.json(directory/(phase+'_STARTED.json'),row['started_sha256'])
        completed = files.json(directory/(phase+'_COMPLETE.json'),row['complete_sha256'])
        assert not (directory/(phase+'_FAILURE.json')).exists()
        assert actual_start['identity'] == completed['identity'] == identity
        assert actual_start['arm'] == completed['arm'] == arm and actual_start['phase'] == completed['phase'] == phase
        assert actual_start['GPU'] == completed['GPU'] == supervisor['GPU']
        assert actual_start['limit_seconds'] == LIMITS[phase]
        assert 0 <= finite(completed['seconds']) <= LIMITS[phase]
        assert completed['new_scores'] == completed['TEST_QA'] == 0
        assert completed['new_generations'] == (212 if phase == 'train' else 105)
        assert completed['FIT_backwards'] == (1213 if phase == 'train' else 0)
        assert completed['optimizer_updates'] == (87 if phase == 'train' else 0)
        assert completed['data_binding'] == expected_binding
        assert files.json(directory/'INPUT_BINDING.json') == expected_binding
        dependencies = files.json(directory/'SOURCE_BINDING.json')
        assert completed['source_sha256'] == dependencies
        assert dependencies[str(source_dir/'worker.py')] == WORKER
        assert dependencies[str(source_dir/'helpers.py')] == HELPERS
        assert dependencies[str(root/'exp/conditional_shared_budget_20261026/v1/adaptation.py')] == ADAPTATION
        for path,h in dependencies.items():
            files.checked(Path(path),h)
        for name,h in completed['files'].items():
            files.checked(inside(directory,name),h)
        if phase == 'train':
            assert completed['before_QA'] == completed['acquired_QA'] == 100 and completed['final_QA'] == 0
            assert completed['coordinates'] == 540672 and completed['parameters'] == 96
            assert completed['frozen_unchanged'] and completed['buffers_unchanged']
            assert completed['adapter_before'] != completed['adapter_after']
            assert completed['training_ast_sha256'] == TRAIN_AST
            assert completed['supervised_forwards'] == 1246 and completed['acquired_cyclic_QA'] == 12
            receipt(completed['training_end'],arm,False)
            cp = completed['checkpoint']; receipt(cp,arm,True)
            assert cp['self_contained'] and cp['no_teacher_or_optimizer']
            assert Path(cp['path']) == directory/'complete.pt'
            assert cp['bytes'] == (directory/'complete.pt').stat().st_size
            files.checked(directory/'complete.pt',cp['sha256'])
            assert row['checkpoint'] == {k:cp[k] for k in ('path','bytes','sha256')}
        else:
            training = phases[arm,'train']
            assert row['checkpoint'] == {k:training['checkpoint'][k] for k in ('path','bytes','sha256')}
            assert completed['complete_state_and_buffers_unchanged']
            for key in RECEIPT_KEYS:
                assert completed['reloaded'][key] == completed['invariant'][key] == training['checkpoint'][key],key
            assert completed['reloaded']['strict_complete_reload'] and completed['reloaded']['self_contained_meta_construction']
            assert completed['resource_proof']['new_generations'] == 5 and completed['resource_proof']['fixed_new_tokens'] == 32
            assert completed['resource_proof']['window_order'] == 'fresh_reload_before_any_DEV'
            assert completed['resource_proof']['actual_device'] == f"cuda:{supervisor['GPU']}"
            assert completed['prediction_proof']['new_generations'] == 100
            assert completed['prediction_proof']['new_scores'] == 0
        phases[arm,phase] = completed
    assert len(identities) == 4
    return supervisor,tasks,split,fit_ids,dev_ids,phases,options


def training_evidence(run,tasks,fit_ids,phases,files,options,split):
    reference = None
    for arm in ARMS:
        directory = run/arm; completed = phases[arm,'train']
        initial = files.json(directory/'ADAPTER_INITIAL.json')
        assert initial['coordinates'] == 540672 and initial['parameters'] == 96
        assert initial['rank_plan'] == completed['training_end']['method']['rank_plan']
        assert len(initial['initialization']) == 48
        for row in initial['initialization']:
            assert row['rank'] == 8 and row['scale'] == 4 and row['dropout'] == .1 and row['B_zero']
            assert row['prefix_r8_exact'] and row['alpha'] == 32
        fit = files.json(directory/'FIT_INPUTS.json')
        plan = completed['training_plan']; assert len(plan) == 19
        expected = []
        for task,p in zip(tasks,plan):
            stage = task['task']; ids = sorted(fit_ids[stage],key=lambda i:hashlib.sha256(f'42:{stage}:{i}'.encode()).digest())
            expected += [(stage,i) for i in ids]
            assert p == dict(stage=stage,QA=len(ids),updates=math.ceil(len(ids)/16),
                order_sha256=digest(''.join(i+'\n' for i in ids).encode()))
        assert [(r['task'],r['id']) for r in fit] == expected
        assert all(r['phase'] == 'TRAIN_ORIGINAL' for r in fit)
        groups = files.json(directory/'SUPERVISION_GROUPS.json')
        extras = files.json(directory/'EXTRA_VIEW_INPUTS.json')
        assert files.json(directory/'OPTION_INPUTS.json') == options['proofs']
        assert len(groups) == 1213 and len(extras) == 33
        assert [(g['task'],g['id']) for g in groups] == expected
        assert [r['id'] for r in extras] == [g['id'] for g in groups if g['eligible']]
        extra_by_id = {r['id']:r for r in extras}; assert len(extra_by_id) == 33
        for index,(group,first) in enumerate(zip(groups,fit)):
            original = options['original'][group['id']]
            eligible = group['id'] in options['selected']['fit']
            assert group['group_index'] == index and group['eligible'] is eligible
            assert group['id_sha256'] == digest(group['id'].encode())
            assert group['backward_complete'] and group['group_loss'] == 'mean_view_CE'
            assert group['original_minibatch_group_denominator'] and group['loss_scale'] == 1.
            views = group['views']; assert len(views) == (2 if eligible else 1)
            for position,view in enumerate(views):
                assert view['index'] == position
                assert view['view'] == ('original' if position == 0 else arm)
                row = original if position == 0 or arm == 'repeat' else options['cyclic'][group['id']]
                input_record = first if position == 0 else extra_by_id[group['id']]
                assert input_record['id'] == group['id'] and input_record['task'] == group['task']
                assert input_record['phase'] == ('TRAIN_ORIGINAL' if position == 0 else 'TRAIN_SECOND')
                assert view['raw_input_sha256'] == input_record['raw_input_sha256']
                assert view['transport_sha256'] == input_record['transport_sha256']
                assert view['prompt_tokens'] == input_record['prompt_tokens'] and type(view['target_tokens']) is int
                assert 0 < view['target_tokens'] <= 513
                assert view['answer_sha256'] == digest(row['answer'].encode())
                assert view['references_sha256'] == digest(json.dumps(row['answers'],ensure_ascii=False,sort_keys=True,separators=(',', ':'),allow_nan=False).encode())
                hash_string(view['prepared_sha256'])
                if eligible:
                    assert group['task'] == 16 and not input_record['retrieved_chunk_indexes']
                    task = tasks[15]
                    images = [dict(path=n,sha256=split['image_sha256'][n]) for n in row['images']]
                    assert images == input_record['image_SHA256'] == first['image_SHA256']
                    contract = dict(images=images,instruction=task['instruction'],question=row['question'],
                        original_context=row.get('context',''),resolved_context=row.get('context',''),
                        retrieved_chunk_indexes=[],image_shapes=input_record['image_shapes'],common_image_area_cap=401408)
                    assert view['raw_input_sha256'] == digest(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',', ':')).encode())
            if eligible and arm == 'repeat':
                assert all(views[0][k] == views[1][k] for k in ('prepared_sha256','raw_input_sha256','transport_sha256','answer_sha256','references_sha256'))
        expected_counts = dict(groups=1213,supervised_forwards=1246,explicit_backwards=1213,eligible_groups=33,
            first_target_tokens=sum(g['views'][0]['target_tokens'] for g in groups),
            extra_target_tokens=sum(v['target_tokens'] for g in groups for v in g['views'][1:]))
        assert completed['group_supervision'] == expected_counts
        events = [json.loads(line) for line in files.checked(directory/'training/logs/events.jsonl').read_bytes().splitlines()]
        assert len(events) == 87
        for p in plan:
            stage,n = p['stage'],p['updates']
            records = [r for r in events if r['stage'] == stage]
            assert [r['step'] for r in records] == list(range(1,n+1))
            warmup = min(math.floor(.03*n),n-1)
            assert warmup == 0  # All original FIT stages are below the warmup threshold.
            for event in records:
                assert event['event'] == 'train_update' and event['warmup_updates'] == warmup
                assert event['learning_rate_used'] and any(lr > 0 for lr in event['learning_rate_used'])
                assert all(finite(lr) >= 0 for lr in event['learning_rate_used'])
                assert finite(event['loss']) >= 0 and finite(event['gradient_l2_preclip']) > 0
                stage_groups = [g for g in groups if g['task'] == stage]
                start = (event['step']-1)*16; minibatch = stage_groups[start:start+16]
                assert minibatch and event['qa_seen'] == start+len(minibatch) and event['total_qa'] == len(stage_groups)
                assert all(g['stage_fraction'] == start/len(stage_groups) for g in minibatch)
                assert event['prompt_tokens'] == sum(v['prompt_tokens'] for g in minibatch for v in g['views'])
                assert event['target_tokens'] == sum(v['target_tokens'] for g in minibatch for v in g['views'])
                grad = files.json(directory/f'grad_S{stage:02}_{event["step"]:03}.json')
                assert grad == dict(stage=stage,step=event['step'],parameters=96,gradient_bytes=2162688,finite=True)
            validity = files.json(directory/f'training/validity/stage_{stage:02}.json')
            assert validity['passed'] and validity['actual_parameters_changed']
            assert validity['updates'] == validity['positive_lr_updates'] == n
            proof = files.json(directory/f'STAGE_{stage:02}_COMPLETE.json')
            assert proof['stage'] == stage and proof['updates'] == n and proof['frozen_unchanged']
            assert proof['optimizer_state_bytes'] == 4325760 and proof['all_optimizer_state_GPU']
            assert proof['bytes'] == 2162688
            files.checked(directory/f'adapter_S{stage:02}.pt',proof['snapshot_sha256'])
        shared = dict(initialization=initial['initialization'],rank_plan=initial['rank_plan'],
            adapter_before=completed['adapter_before'],original_base=completed['original_base'],
            FIT_inputs=fit,plan=plan,source=completed['source_sha256'])
        if reference is None:
            reference = shared
        assert shared == reference,'Arms differ in original W0, initialization, FIT transport/order or inherited sources'


def read_answers(run,tasks,dev_ids,phases,files,options):
    """Validate every existing answer/reference before invoking any scorer."""
    task_by_id = {i:t for t in tasks for i in dev_ids[t['task']]}
    expected_ids = [i for t in tasks for i in dev_ids[t['task']]]
    all_rows,input_reference,resource_reference = {},{},None
    for arm in ARMS:
        directory = run/arm
        for name,phase in (('before','DEV_BEFORE'),('acquired','DEV_ACQUIRED'),('final','DEV_FINAL')):
            rows = files.json(directory/(name+'_predictions.json'))
            assert len(rows) == 100 and [r['id'] for r in rows] == expected_ids
            for row in rows:
                task = task_by_id[row['id']]
                assert row['phase'] == phase and row['task'] == task['task'] and row['domain'] == task['domain']
                if name != 'final':
                    assert row['stage'] == task['task']  # phase disambiguates before vs acquired.
                assert row['generation_cap'] == task['max_new_tokens']
                assert isinstance(row['prediction'],str) and row['references']
                assert row['references'] == options['original'][row['id']]['answers']
                assert all(isinstance(ref,str) and ref for ref in row['references'])
                assert isinstance(row['output_tokens'],list) and len(row['output_tokens']) <= row['generation_cap']
                invariant = {k:row[k] for k in ('task','domain','references','generation_cap',
                    'raw_input_sha256','transport_sha256','prepared_sha256','input_shape','input_tensor_shapes','image_SHA256')}
                if row['id'] not in input_reference:
                    input_reference[row['id']] = invariant
                assert invariant == input_reference[row['id']],'Paired DEV input/reference differs'
            all_rows[arm,name] = rows
        readouts = files.json(directory/'READOUT_EVENTS.json')
        expected_readouts = []
        for t in tasks:
            expected_readouts.extend((t['task'],p) for p in ('DEV_BEFORE','DEV_ACQUIRED'))
            if t['task'] == 16: expected_readouts.append((16,'DEV_ACQUIRED_CYCLIC'))
        assert len(readouts) == 39
        assert [(e['stage'],e['phase']) for e in readouts] == expected_readouts
        for event in readouts:
            count = 12 if event['phase'] == 'DEV_ACQUIRED_CYCLIC' else len(dev_ids[event['stage']])
            assert event['new_generations'] == count
            assert event['task_has_no_DEV'] == (count == 0)
            assert event['state_and_buffers_unchanged'] and event['training_RNG_restored']
        cyclic = files.json(directory/'acquired_cyclic_predictions.json')
        expected_cyclic = [i for i in dev_ids[16] if i in options['selected']['dev']]
        assert len(cyclic) == 12 and [r['id'] for r in cyclic] == expected_cyclic
        for row in cyclic:
            source = options['cyclic'][row['id']]
            assert row['phase'] == 'DEV_ACQUIRED_CYCLIC' and row['stage'] == row['task'] == 16
            assert row['domain'] == 'finance' and row['generation_cap'] == tasks[15]['max_new_tokens']
            assert row['references'] == source['answers'] and isinstance(row['prediction'],str)
            assert isinstance(row['output_tokens'],list) and len(row['output_tokens']) <= row['generation_cap']
            assert not row['retrieved_chunk_indexes']
            assert row['image_SHA256'] == input_reference[row['id']]['image_SHA256']
            contract = dict(images=row['image_SHA256'],instruction=tasks[15]['instruction'],question=source['question'],
                original_context=source['context'],resolved_context=source['context'],retrieved_chunk_indexes=[],
                image_shapes=row['image_shapes'],common_image_area_cap=401408)
            assert row['raw_input_sha256'] == digest(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',', ':')).encode())
            key = ('cyclic',row['id'])
            invariant = {k:row[k] for k in ('task','domain','references','generation_cap',
                'raw_input_sha256','transport_sha256','prepared_sha256','input_shape','input_tensor_shapes','image_SHA256')}
            if key not in input_reference: input_reference[key] = invariant
            assert input_reference[key] == invariant
        all_rows[arm,'cyclic'] = cyclic
        windows = files.json(directory/'memory.json')
        assert len(windows) == 5 and [w['domain'] for w in windows] == list(DOMAINS)
        window_bindings = []
        for domain,w in zip(DOMAINS,windows):
            selected = min((i for i in expected_ids if task_by_id[i]['domain'] == domain),key=lambda i:hashlib.sha256(i.encode()).digest())
            assert w['id'] == selected and w['new_tokens'] == 32
            assert w['raw_input_sha256'] == input_reference[selected]['raw_input_sha256']
            assert w['prepared_sha256'] == input_reference[selected]['prepared_sha256']
            assert 0 <= finite(w['baseline_allocated']) <= finite(w['peak_allocated'])
            assert finite(w['peak_reserved']) >= w['peak_allocated']
            window_bindings.append({k:w[k] for k in ('id','domain','raw_input_sha256','prepared_sha256','new_tokens')})
        if resource_reference is None:
            resource_reference = window_bindings
        assert window_bindings == resource_reference
    return all_rows


def scorer(root,files):
    path = files.checked(root/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py',SCORER_FILE)
    raw = path.read_bytes()
    nodes = [n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
    tree = ast.Module(body=nodes,type_ignores=[])
    # Source bytes are invariant across Python3.10/3.12 FunctionDef.type_params.
    # Compile each interpreter's own parsed AST; never compare its schema dump.
    selected = b'\n'.join(b''.join(raw.splitlines(keepends=True)[n.lineno-1:n.end_lineno]) for n in nodes)
    assert len(nodes) == 2 and digest(selected) == SCORER_DEFINITIONS
    namespace = {'re':re,'Counter':Counter}
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
    return namespace['score']


def summarize(scored,tasks,dev_ids):
    result = {}
    for arm in ARMS:
        by_phase = {}
        for phase in ('before','acquired','final'):
            rows = scored[arm,phase]
            by_task,by_domain = {},{}
            for task in tasks:
                selected = [r for r in rows if r['task'] == task['task']]
                assert len(selected) == len(dev_ids[task['task']])
                by_task[str(task['task'])] = dict(QA=len(selected),F1=sum(r['F1'] for r in selected)/len(selected) if selected else None,
                    domain=task['domain'],empty_task=not bool(selected))
            for domain in DOMAINS:
                selected = [r for r in rows if r['domain'] == domain]
                assert len(selected) == 20
                by_domain[domain] = dict(QA=len(selected),F1=sum(r['F1'] for r in selected)/len(selected))
            by_phase[phase] = dict(QA=100,domains=by_domain,domain_macro_F1=sum(x['F1'] for x in by_domain.values())/5,
                tasks=by_task,per_id=rows)
        changes = {}
        for task in tasks:
            a,b = by_phase['before']['tasks'][str(task['task'])],by_phase['acquired']['tasks'][str(task['task'])]
            changes[str(task['task'])] = dict(QA=a['QA'],before_F1=a['F1'],acquired_F1=b['F1'],
                acquired_minus_before=None if a['QA'] == 0 else b['F1']-a['F1'])
        result[arm] = dict(**by_phase,acquisition=dict(tasks=changes,
            domain_macro_change=by_phase['acquired']['domain_macro_F1']-by_phase['before']['domain_macro_F1'],
            domains={d:by_phase['acquired']['domains'][d]['F1']-by_phase['before']['domains'][d]['F1'] for d in DOMAINS}),
            phase_times_are_task_specific=True,final_is_actual_deployment_graph=True)
    return result


def strict_letter(text):
    value = text.strip()
    return value if value in ('A','B','C','D') else None


def option_readout(answers,options):
    inverse = dict(zip('BCDA','ABCD'))
    result = {}
    for arm in ARMS:
        original = {r['id']:r for r in answers[arm,'acquired']}
        counts = dict(QA=12,original_mappable=0,cyclic_mappable=0,both_mappable=0,
            both_unmappable=0,only_one_mappable=0,same_content=0,changed_content=0,
            stable_strict_correct=0,stable_strict_wrong=0,original_strict_correct=0,cyclic_strict_correct=0)
        per_id = []
        for cyclic in answers[arm,'cyclic']:
            row = original[cyclic['id']]; gold = options['original'][row['id']]['answer']
            assert gold in ('A','B','C','D')
            a,b = strict_letter(row['prediction']),strict_letter(cyclic['prediction'])
            mapped = inverse[b] if b is not None else None
            both = a is not None and mapped is not None
            same = both and a == mapped
            counts['original_mappable'] += int(a is not None); counts['cyclic_mappable'] += int(b is not None)
            counts['both_mappable'] += int(both)
            counts['both_unmappable'] += int(a is None and mapped is None)
            counts['only_one_mappable'] += int((a is None) != (mapped is None))
            counts['same_content'] += int(same); counts['changed_content'] += int(both and not same)
            counts['stable_strict_correct'] += int(same and a == gold)
            counts['stable_strict_wrong'] += int(same and a != gold)
            counts['original_strict_correct'] += int(a == gold)
            counts['cyclic_strict_correct'] += int(mapped == gold)
            per_id.append(dict(id_sha256=digest(row['id'].encode()),original_label=a,cyclic_label=b,
                inverse_mapped_label=mapped,both_mappable=both,same_content=same,
                original_strict_correct=a == gold,cyclic_strict_correct=mapped == gold,
                original_prediction_sha256=digest(row['prediction'].encode()),
                cyclic_prediction_sha256=digest(cyclic['prediction'].encode())))
        assert len(per_id) == 12
        assert counts['both_mappable']+counts['both_unmappable']+counts['only_one_mappable'] == 12
        assert counts['same_content']+counts['changed_content'] == counts['both_mappable']
        assert counts['stable_strict_correct']+counts['stable_strict_wrong'] == counts['same_content']
        result[arm] = dict(counts=counts,per_id=per_id,
            same_content_fraction_of_both_mappable=counts['same_content']/counts['both_mappable'] if counts['both_mappable'] else None,
            strict_mapping='prediction.strip() must be exactly A/B/C/D; cyclic inverse B/C/D/A -> A/B/C/D',
            cyclic_F1_calls=0,invalid_pairs_never_count_as_stable=True,
            scope='12 fixed explored DEV, S16 training graph, not held-out generalization or final deployed-graph invariance')
    return result


def prefix_observation(run,tasks,answers,files):
    """Descriptive existing artifacts, never an additional acceptance condition."""
    stages = []
    for stage in range(1,16):
        digest_by_arm = {a:files.json(run/a/f'STAGE_{stage:02}_COMPLETE.json')['adapter_digest'] for a in ARMS}
        outputs = {}
        for phase in ('before','acquired'):
            left = [r for r in answers['repeat',phase] if r['task'] == stage]
            right = [r for r in answers['paired',phase] if r['task'] == stage]
            assert [r['id'] for r in left] == [r['id'] for r in right]
            outputs[phase] = dict(QA=len(left),same_prediction=sum(x['prediction']==y['prediction'] for x,y in zip(left,right)),
                same_output_tokens=sum(x['output_tokens']==y['output_tokens'] for x,y in zip(left,right)))
        stages.append(dict(stage=stage,adapter_digest=digest_by_arm,
            adapter_digest_equal=len(set(digest_by_arm.values()))==1,outputs=outputs))
    a = [r for r in answers['repeat','before'] if r['task']==16]
    b = [r for r in answers['paired','before'] if r['task']==16]
    assert [r['id'] for r in a] == [r['id'] for r in b]
    return dict(stages=stages,S16_before=dict(QA=len(a),same_prediction=sum(x['prediction']==y['prediction'] for x,y in zip(a,b)),
        same_output_tokens=sum(x['output_tokens']==y['output_tokens'] for x,y in zip(a,b))),
        no_forced_prefix_equality_gate=True,additional_model_or_score_calls=0,
        interpretation='Differences before the S16 intervention are reported, not suppressed or declared scientific failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True)
    args = parser.parse_args(); root,run = args.root.resolve(),args.run.resolve()
    assert args.root.is_absolute() and args.run.is_absolute() and root.is_dir() and run.is_dir()
    assert not any((run/name).exists() for name in ('VERIFY_STARTED.json','VERIFY_FAILURE.json','VERIFIED.json','OUTCOME.json'))
    began = time.monotonic(); files = Files()
    assert args.binding.is_absolute() and args.binding.is_file()
    supervisor,tasks,split,fit_ids,dev_ids,phases,options = complete_metadata(root,run,files,args.binding)
    training_evidence(run,tasks,fit_ids,phases,files,options,split)
    answers = read_answers(run,tasks,dev_ids,phases,files,options)
    mapping = option_readout(answers,options)
    prefix = prefix_observation(run,tasks,answers,files)
    score = scorer(root,files)
    own = files.checked(Path(__file__)); own_sha = files.hashes[str(own)]
    write(run/'VERIFY_STARTED.json',dict(unix=time.time(),pid=os.getpid(),verifier_sha256=own_sha,
        all_four_phases_exit0_and_integrity_passed=True,planned_scoring_calls=600,
        strict_cyclic_mapping_readouts=24,cyclic_F1_calls=0,
        new_models=0,new_generations=0,exclusive_no_automatic_retry=True))
    try:
        scored,calls = {},0
        for arm in ARMS:
            for phase in ('before','acquired','final'):
                rows = []
                for row in answers[arm,phase]:
                    value = score(row['prediction'],row['references']); calls += 1
                    assert set(value) == {'token_f1','exact_match','rougeL'}
                    assert all(0 <= finite(v) <= 100 for v in value.values())
                    rows.append(dict(id_sha256=digest(row['id'].encode()),task=row['task'],domain=row['domain'],
                        prediction_sha256=digest(row['prediction'].encode()),F1=value['token_f1']))
                scored[arm,phase] = rows
        assert calls == 600
        quality = summarize(scored,tasks,dev_ids)
        resources = {}
        for arm in ARMS:
            train,evaluation = phases[arm,'train'],phases[arm,'eval']
            windows = files.json(run/arm/'memory.json')
            resources[arm] = dict(training_seconds=train['seconds'],eval_seconds=evaluation['seconds'],
                training_monitor_includes_212_stage_readouts=True,
                training_memory=files.json(run/arm/'TRAIN_MEMORY.json'),
                actual_training_end_layout=train['training_end']['layout'],
                actual_final_layout=train['checkpoint']['layout'],checkpoint_file_bytes=train['checkpoint']['bytes'],
                checkpoint_sha256=train['checkpoint']['sha256'],deployment=train['deployment'],
                windows=windows,unit='bytes',MiB_divisor=1048576,
                fixed_window_max_peak_allocated=max(w['peak_allocated'] for w in windows),
                fixed_window_max_peak_reserved=max(w['peak_reserved'] for w in windows),
                scope='Five fixed raw DEV inputs, 32 generated tokens; not full-TEST worst case or a speed claim')
        comparisons = {}
        for left,right in (('paired','repeat'),):
            comparisons[left+'_minus_'+right] = {phase:dict(
                domain_macro_F1=quality[left][phase]['domain_macro_F1']-quality[right][phase]['domain_macro_F1'],
                domains={d:quality[left][phase]['domains'][d]['F1']-quality[right][phase]['domains'][d]['F1'] for d in DOMAINS})
                for phase in ('before','acquired','final')}
        files.unchanged()
        outcome = dict(status='COMPLETE_PAIRED_OPTION_EXPLORATORY_RESULT',quality=quality,comparisons=comparisons,
            strict_option_readout=mapping,preintervention_prefix=prefix,
            resources=resources,scored_existing_answers=600,strict_cyclic_mapping_readouts=24,cyclic_F1_calls=0,
            new_generations_by_verifier=0,model_generations=634,FIT_backwards=2426,optimizer_updates=174,
            supervised_forwards=2492,
            no_per_domain_acceptance_gate=True,no_test_or_hyperparameter_search=True,
            limits=['1213 FIT/100 explored DEV, one seed; not the formal 10% setting.',
                'Before/acquired aggregate multiple task-specific model times; empty task DEV retained as null.',
                'Both final quality/resource readouts use their actual folded shared-r8 graph.',
                'Paired minus repeat compares a matched-exposure, two-view supervision package; no first permutation-augmentation claim.',
                'Strict same-content counts differ from raw F1 and semantic medical/financial correctness; unmappable pairs stay in the full denominator.',
                'No automatic claim of significance, equivalence, strong-baseline superiority or Ours.'])
        write(run/'OUTCOME.json',outcome)
        write(run/'VERIFIED.json',dict(status='VERIFIED_ONCE',verifier_sha256=own_sha,
            supervisor_complete_sha256=files.hashes[str(run/'SUPERVISOR_COMPLETE.json')],
            source_and_artifact_sha256=files.hashes,scorer_source_sha256=SCORER_FILE,
            scorer_definitions_source_sha256=SCORER_DEFINITIONS,
            outcome_sha256=digest((run/'OUTCOME.json').read_bytes()),scoring_calls=600,
            cyclic_F1_calls=0,strict_cyclic_mapping_readouts=24,
            models_started=0,new_generations=0,new_TEST_scores=0,
            all_four_phases_real_exit0=True,original_W0_initialization_and_FIT_equal=True,
            complete_checkpoints_streamed_not_loaded=True,seconds=time.monotonic()-began))
    except BaseException:
        write(run/'VERIFY_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),
            no_automatic_retry=True,scoring_calls_completed=calls if 'calls' in locals() else 0))
        raise


if __name__ == '__main__':
    main()
