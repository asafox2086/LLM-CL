"""Verify six complete GPU phases, then score 900 existing answers once.

Stdlib only. No model/processor import, deserialization, generation or TEST.
The original runtime tokens/score definitions are compiled unchanged from AST.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

ARMS = ('shared','uniform','soft')
ORDER = [(arm,phase) for arm in ARMS for phase in ('train','eval')]
DOMAINS = ('natural','medical','geography','finance','traffic')
LIMITS = {'train':5400,'eval':1800}
WORKER = '912d5e08edb2642268a483f98f01da7440f7fed730e7b2e91d32609127e190c8'
ADAPTATION = '251a2d490f9ce29069de141686a16da549c28de82f7c1ce58fab2df8da31252a'
MANIFEST = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
SPLIT = 'd8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
TRAIN_AST = '2b31ff9e06234e967ea42d3cb6d594c82050054c7c2c058d77f61bf8100dda7f'
SCORER_FILE = '4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240'
SCORER_AST = 'ad62d7fd6f426fc0396d797e30eddb79e868c018534507755881c7aee07ce779'
RECEIPT_KEYS = ('layout','digest','all_buffers_digest','parameter_alias_groups',
    'buffer_alias_groups','buffer_manifest','functional_config_digest',
    'generation_config_digest','language_generation_config_digest',
    'language_attention_classes','vision_attention_classes','method')


def read(path):
    return json.loads(Path(path).read_bytes())


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
    explicit = not deployed or arm == 'soft'
    assert mode['arm'] == arm and mode['deployed'] is deployed and mode['explicit_AB'] is explicit
    assert mode['split'] == [4,2,2] and mode['scale'] == 4 and mode['dropout'] == .1
    assert mode['soft_temperature'] == 1 and mode['extra_router_parameters'] == 0
    assert mode['training_coordinates'] == 540672
    expected_plan = {f'model.layers.{i}.self_attn.{p}_proj':8 for i in range(24) for p in ('q','v')}
    assert mode['rank_plan'] == expected_plan
    assert mode['routing'] == {'shared':'none','uniform':'fixed_half','soft':'post_dropout_A_energy'}[arm]
    assert value['layout']['weights'] == {'coefficients':894216224 if explicit else 893675552,
        'bytes':1789513792 if explicit else 1787351104}
    assert value['layout']['ordinary_buffers']['bytes'] == 9032 and value['layout']['trainable'] == 0
    for key in ('digest','all_buffers_digest','functional_config_digest',
                'generation_config_digest','language_generation_config_digest'):
        hash_string(value[key])
    assert all(x['dtype'] == 'torch.float32' for n,x in value['buffer_manifest'].items() if n.endswith('inv_freq'))


def complete_metadata(root,run,files):
    supervisor = files.json(run/'SUPERVISOR_COMPLETE.json')
    assert not (run/'SUPERVISOR_FAILURE.json').exists()
    assert supervisor['no_retry'] is True and supervisor['active_worker'] is None
    assert supervisor['new_generations'] == 915 and supervisor['new_scores'] == supervisor['TEST_QA'] == 0
    assert supervisor['FIT_backwards'] == 3639 and supervisor['optimizer_updates'] == 261
    assert supervisor['cumulative_model_budget_seconds'] == 21600
    assert len(supervisor['phases']) == 6
    started = files.json(run/'SUPERVISOR_STARTED.json')
    assert started['identity'] == supervisor['identity'] and started['GPU'] == supervisor['GPU']
    assert started['source_sha256'] == supervisor['source_sha256']
    assert started['protocol'] == supervisor['protocol']
    assert started['no_retry'] and started['existing_allocator_supplied_GPU']
    source_dir = Path(supervisor['source_dir'])
    assert source_dir == root/'exp/conditional_shared_budget_20261026/v1'
    sources = supervisor['source_sha256']
    assert set(sources) == {'controller.py','worker.py','adaptation.py'}
    assert sources['worker.py'] == WORKER and sources['adaptation.py'] == ADAPTATION
    for name,h in sources.items():
        files.checked(source_dir/name,h)
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
    expected_binding = dict(manifest_path=str(manifest_path),manifest_sha256=MANIFEST,
        split_path=str(split_path),split_sha256=SPLIT,FIT_QA=1213,DEV_QA=100,original_TRAIN_QA=1313,TEST_QA=0)
    phases,identities,previous_exit = {},set(),None
    for (arm,phase),row in zip(ORDER,supervisor['phases']):
        assert (row['arm'],row['phase']) == (arm,phase)
        assert row['returncode'] == 0 and row['identity_absent'] is True and row['timeout'] is False
        identity = row['identity']; key = (identity['pid'],int(identity['startticks']))
        assert key not in identities and identity['pid'] == identity['pgid']
        identities.add(key)
        directory = run/arm
        assert Path(row['run_dir']) == directory
        outer_start = files.json(run/f'PHASE_{arm}_{phase}_STARTED.json')
        outer_exit = files.json(run/f'PHASE_{arm}_{phase}_EXIT.json')
        assert outer_exit == {k:v for k,v in row.items() if k not in ('started_sha256','complete_sha256','checkpoint')}
        assert outer_start['identity'] == identity and outer_start['GPU'] == supervisor['GPU']
        assert outer_start['arm'] == arm and outer_start['phase'] == phase
        assert outer_start['limit_seconds'] == LIMITS[phase]
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
        assert completed['new_generations'] == (200 if phase == 'train' else 105)
        assert completed['FIT_backwards'] == (1213 if phase == 'train' else 0)
        assert completed['optimizer_updates'] == (87 if phase == 'train' else 0)
        assert completed['data_binding'] == expected_binding
        assert files.json(directory/'INPUT_BINDING.json') == expected_binding
        dependencies = files.json(directory/'SOURCE_BINDING.json')
        assert completed['source_sha256'] == dependencies
        assert dependencies[str(source_dir/'worker.py')] == WORKER
        assert dependencies[str(source_dir/'adaptation.py')] == ADAPTATION
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
    assert len(identities) == 6
    return supervisor,tasks,split,fit_ids,dev_ids,phases


def training_evidence(run,tasks,fit_ids,phases,files):
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
        assert all(r['phase'] == 'TRAIN' for r in fit)
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


def read_answers(run,tasks,dev_ids,phases,files):
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
                assert all(isinstance(ref,str) and ref for ref in row['references'])
                assert isinstance(row['output_tokens'],list) and len(row['output_tokens']) <= row['generation_cap']
                invariant = {k:row[k] for k in ('task','domain','references','generation_cap',
                    'raw_input_sha256','transport_sha256','prepared_sha256','input_shape','input_tensor_shapes','image_SHA256')}
                if row['id'] not in input_reference:
                    input_reference[row['id']] = invariant
                assert invariant == input_reference[row['id']],'Paired DEV input/reference differs'
            all_rows[arm,name] = rows
        readouts = files.json(directory/'READOUT_EVENTS.json')
        assert len(readouts) == 38
        assert [(e['stage'],e['phase']) for e in readouts] == [(t['task'],p) for t in tasks for p in ('DEV_BEFORE','DEV_ACQUIRED')]
        for event in readouts:
            assert event['new_generations'] == len(dev_ids[event['stage']])
            assert event['task_has_no_DEV'] == (not bool(dev_ids[event['stage']]))
            assert event['state_and_buffers_unchanged'] and event['training_RNG_restored']
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
    nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
    tree = ast.Module(body=nodes,type_ignores=[])
    assert len(nodes) == 2 and digest(ast.dump(tree,include_attributes=False).encode()) == SCORER_AST
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args = parser.parse_args(); root,run = args.root.resolve(),args.run.resolve()
    assert args.root.is_absolute() and args.run.is_absolute() and root.is_dir() and run.is_dir()
    assert not any((run/name).exists() for name in ('VERIFY_STARTED.json','VERIFY_FAILURE.json','VERIFIED.json','OUTCOME.json'))
    began = time.monotonic(); files = Files()
    supervisor,tasks,split,fit_ids,dev_ids,phases = complete_metadata(root,run,files)
    training_evidence(run,tasks,fit_ids,phases,files)
    answers = read_answers(run,tasks,dev_ids,phases,files)
    score = scorer(root,files)
    own = files.checked(Path(__file__)); own_sha = files.hashes[str(own)]
    write(run/'VERIFY_STARTED.json',dict(unix=time.time(),pid=os.getpid(),verifier_sha256=own_sha,
        all_six_phases_exit0_and_integrity_passed=True,planned_scoring_calls=900,
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
        assert calls == 900
        quality = summarize(scored,tasks,dev_ids)
        resources = {}
        for arm in ARMS:
            train,evaluation = phases[arm,'train'],phases[arm,'eval']
            windows = files.json(run/arm/'memory.json')
            resources[arm] = dict(training_seconds=train['seconds'],eval_seconds=evaluation['seconds'],
                training_monitor_includes_200_stage_readouts=True,
                training_memory=files.json(run/arm/'TRAIN_MEMORY.json'),
                actual_training_end_layout=train['training_end']['layout'],
                actual_final_layout=train['checkpoint']['layout'],checkpoint_file_bytes=train['checkpoint']['bytes'],
                checkpoint_sha256=train['checkpoint']['sha256'],deployment=train['deployment'],
                windows=windows,unit='bytes',MiB_divisor=1048576,
                fixed_window_max_peak_allocated=max(w['peak_allocated'] for w in windows),
                fixed_window_max_peak_reserved=max(w['peak_reserved'] for w in windows),
                scope='Five fixed raw DEV inputs, 32 generated tokens; not full-TEST worst case or a speed claim')
        comparisons = {}
        for left,right in (('soft','uniform'),('soft','shared'),('uniform','shared')):
            comparisons[left+'_minus_'+right] = {phase:dict(
                domain_macro_F1=quality[left][phase]['domain_macro_F1']-quality[right][phase]['domain_macro_F1'],
                domains={d:quality[left][phase]['domains'][d]['F1']-quality[right][phase]['domains'][d]['F1'] for d in DOMAINS})
                for phase in ('before','acquired','final')}
        files.unchanged()
        outcome = dict(status='COMPLETE_THREE_ARM_EXPLORATORY_RESULT',quality=quality,comparisons=comparisons,
            resources=resources,scored_existing_answers=900,new_generations_by_verifier=0,
            model_generations=915,FIT_backwards=3639,optimizer_updates=261,
            no_per_domain_acceptance_gate=True,no_test_or_hyperparameter_search=True,
            limits=['1213 FIT/100 explored DEV, one seed; not the formal 10% setting.',
                'Before/acquired aggregate multiple task-specific model times; empty task DEV retained as null.',
                'Final quality/resources use the actual folded shared/uniform or explicit-bank soft graph.',
                'Soft versus uniform tests a conditional-update package; no independent novel-router claim.',
                'No automatic claim of significance, equivalence, strong-baseline superiority or Ours.'])
        write(run/'OUTCOME.json',outcome)
        write(run/'VERIFIED.json',dict(status='VERIFIED_ONCE',verifier_sha256=own_sha,
            supervisor_complete_sha256=files.hashes[str(run/'SUPERVISOR_COMPLETE.json')],
            source_and_artifact_sha256=files.hashes,scorer_source_sha256=SCORER_FILE,scorer_AST_sha256=SCORER_AST,
            outcome_sha256=digest((run/'OUTCOME.json').read_bytes()),scoring_calls=900,
            models_started=0,new_generations=0,new_TEST_scores=0,
            all_six_phases_real_exit0=True,original_W0_initialization_and_FIT_equal=True,
            complete_checkpoints_streamed_not_loaded=True,seconds=time.monotonic()-began))
    except BaseException:
        write(run/'VERIFY_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),
            no_automatic_retry=True,scoring_calls_completed=calls if 'calls' in locals() else 0))
        raise


if __name__ == '__main__':
    main()
