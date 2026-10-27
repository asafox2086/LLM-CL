"""One post-completion local T16 verification and 80 original F1 scores.

Stdlib only: no model, processor, checkpoint deserialization, or generation.
Historical prefix costs remain separate from the four new model phases.
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

ARMS=('aligned','random')
ORDER=[(a,p) for a in ARMS for p in ('train','eval')]
DOMAINS=('natural','medical','geography','finance','traffic')
LIMITS={'train':1800,'eval':900}
SOURCE_NAMES={'controller.py','worker.py','adaptation.py','calibration.py','verify.py','prefix.py'}
MANIFEST='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
SPLIT='d8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
TRAIN_AST='2b31ff9e06234e967ea42d3cb6d594c82050054c7c2c058d77f61bf8100dda7f'
SCORER_FILE='4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240'
PREFIX_COMPLETE='ae5847e12d2773b54f5129dc59cd22921795c83556bc72a25d6b120e04af9de0'
PREFIX_CHECKPOINT='efb578fe8bd65430284d5d5e305ad0eb55946d06ef623265f304db3ba2ab69fe'
RECEIPT_KEYS=('layout','digest','all_buffers_digest','parameter_alias_groups','buffer_alias_groups',
    'buffer_manifest','functional_config_digest','generation_config_digest','language_generation_config_digest',
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

def same_live(who):
    try:
        p=Path('/proc')/str(who['pid']);fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return (fields[0]!='Z' and fields[19]==str(who['startticks']) and p.stat().st_uid==who['uid']
                and int(fields[2])==who['pgid'])
    except (OSError,IndexError,ValueError):return False

def merge_records(records,modules):
    assert len(records)==48 and {r['module'] for r in records}==set(modules)
    for row in records:
        assert row['scale']==4 and row['merge_dtype']=='torch.float32' and row['output_dtype']=='torch.float16'
        assert row['autocast_enabled'] is False and row['matmul_allow_tf32'] is False
        for key in ('A_digest','B_digest','base_before_digest','merged_base_digest'):hash_string(row[key])
        for key in ('rounding_max_abs','rounding_rms','update_max_abs'):assert finite(row[key])>=0

def scorer(root,files):
    path = files.checked(root/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py',SCORER_FILE)
    nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
    tree = ast.Module(body=nodes,type_ignores=[])
    assert len(nodes)==2
    # Raw source is pinned. Do not compare Python-version-dependent type_params fields.
    namespace = {'re':re,'Counter':Counter}
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),namespace)
    return namespace['score']


def plan_names():
    return {f'model.layers.{i}.self_attn.{p}_proj':8 for i in range(24) for p in ('q','v')}


def mode(value,arm,deployed,active=False):
    assert value['arm']==arm and value['deployed'] is deployed and value['explicit_AB'] is (not deployed)
    assert value['rank_plan']==plan_names() and value['rank']==8 and value['alpha']==32
    assert value['scale']==4 and value['dropout']==.1 and value['A_row_norm']==1/math.sqrt(3)
    assert value['base_dtype']=='torch.float16' and value['A_B_dtype']=='torch.float32'
    assert value['structural_AB_coordinates']==(0 if deployed else 540672)
    assert value['structural_AB_parameter_tensors']==(0 if deployed else 96)
    assert value['active_training_coordinates']==(540672 if active else 0)
    assert value['active_training_parameter_tensors']==(96 if active else 0)
    assert value['extra_router_parameters']==0 and value['cumulative_delta_rank_is_not_fixed_r8']


def receipt(value,arm,deployed):
    mode(value['method'],arm,deployed,False)
    assert value['layout']['weights']==dict(coefficients=893675552 if deployed else 894216224,
        bytes=1787351104 if deployed else 1789513792)
    assert value['layout']['ordinary_buffers']['bytes']==9032 and value['layout']['trainable']==0
    for key in ('digest','all_buffers_digest','functional_config_digest','generation_config_digest',
                'language_generation_config_digest'):hash_string(value[key])
    assert all(x['dtype']=='torch.float32' for n,x in value['buffer_manifest'].items() if n.endswith('inv_freq'))


def prefix_evidence(binding,reused,files):
    """Small sealed lineage only; model processes already hashed/loaded old CP.

    Preserve that distinction: do not claim another full CP hash here.
    """
    expected=binding['prefix'];old=Path(expected['run'])
    assert old.is_absolute() and expected['GPU']==reused['GPU']==1 and expected['uid']==reused['uid']==1084
    assert reused['run']==str(old)
    assert expected['complete_sha256']==reused['complete_sha256']==PREFIX_COMPLETE
    assert expected['checkpoint_sha256']==reused['checkpoint_sha256']==PREFIX_CHECKPOINT
    for key in ('checkpoint_bytes','prefix_state_sha256','metadata_sha256','generation_evidence'):
        assert expected[key]==reused[key]
    assert reused['trusted_private_checkpoint_owner_verified']
    assert reused['full_checkpoint_SHA_checked_without_deserialization']
    complete=files.json(old/'prefix_COMPLETE.json',PREFIX_COMPLETE)
    state=files.json(old/'PREFIX_STATE.json',expected['prefix_state_sha256'])
    meta=files.json(old/'checkpoints/stage_15.json',expected['metadata_sha256'])
    assert files.json(old/'checkpoints/latest.json',expected['metadata_sha256'])==meta
    assert complete['checkpoint']==state['checkpoint']==meta==reused['original_validation']['checkpoint']
    assert meta['path']=='checkpoints/stage_15.pt' and meta['stage']==15
    assert meta['sha256']==PREFIX_CHECKPOINT and meta['bytes']==expected['checkpoint_bytes']
    assert (old/meta['path']).stat().st_size==meta['bytes']
    assert complete['files']['PREFIX_STATE.json']==expected['prefix_state_sha256']
    assert complete['files']['checkpoints/stage_15.json']==expected['metadata_sha256']
    original=reused['original_validation']
    assert original['model_digest']==state['model_digest'] and original['buffers_digest']==state['buffers_digest']
    assert original['original_exit0_and_identities_gone'] and original['original_source_files_verified']
    assert original['historical_cost']['FIT_backwards']==959 and original['historical_cost']['optimizer_updates']==69
    assert original['historical_cost']['generations']==120
    assert (complete['FIT_backwards'],complete['optimizer_updates'],complete['new_generations'])==(959,69,120)
    assert complete['new_scores']==complete['TEST_QA']==0
    files.json(old/'prefix_STARTED.json',original['old_prefix_STARTED_sha256'])
    files.json(old.parent/'PHASE_prefix_prefix_EXIT.json',original['old_prefix_EXIT_sha256'])
    # This is the old prefix's lineage, not the failed global direction run.
    files.json(old.parent/'SUPERVISOR_FAILURE.json',original['old_parent_FAILURE_sha256'])
    evidence=files.json(Path(expected['generation_evidence']['path']),expected['generation_evidence']['sha256'])
    assert evidence['stage']==15 and evidence['checkpoint_sha256']==PREFIX_CHECKPOINT
    assert set(evidence['configs'])=={'outer','language'}
    for value in evidence['configs'].values():
        assert digest(json.dumps(value['public'],sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())==value['public_sha256']
        assert value['from_model_config'] and value['nondefault_model_generation']=={}
        assert value['derived_equals_public'] and value['derived_public']==value['public']
    return state,evidence


def metadata(root,run,binding_path,binding,files):
    start=files.json(run/'SUPERVISOR_STARTED.json');end=files.json(run/'SUPERVISOR_COMPLETE.json')
    assert not (run/'SUPERVISOR_FAILURE.json').exists() and not same_live(end['identity'])
    assert start['identity']==end['identity'] and start['GPU']==end['GPU']==1
    assert start['protocol']==end['protocol']==binding['protocol']
    assert start['source_sha256']==end['source_sha256']==binding['sources']
    assert set(binding['sources'])==SOURCE_NAMES and binding['schema']=='task-direction-local-v1'
    assert binding['manifest_sha256']==MANIFEST and binding['split_sha256']==SPLIT and binding['training_ast_sha256']==TRAIN_AST
    assert start['binding_path']==end['binding_path']==str(binding_path)
    assert start['binding_sha256']==end['binding_sha256']==files.hashes[str(binding_path)]
    assert start['phase_limits']==LIMITS and end['cumulative_model_budget_seconds']==5400
    assert start['no_retry'] and end['no_retry'] and end['active_worker'] is None
    assert start['planned_scoring_calls']==end['planned_scoring_calls']==80
    assert end['no_automatic_consumer'] and end['prefix_history_is_not_new_execution']
    assert (end['new_generations'],end['FIT_backwards'],end['calibration_backwards'],end['optimizer_updates'])==(90,138,32,10)
    assert end['new_scores']==end['TEST_QA']==0
    source_dir=root/'exp/task_direction_local_20261027/v1'
    assert Path(start['source_dir'])==Path(end['source_dir'])==source_dir
    for name,h in binding['sources'].items():files.checked(source_dir/name,h)
    files.checked(Path(binding['protocol']['path']),binding['protocol']['sha256'])
    reused=files.json(run/'REUSED_PREFIX.json',end['reused_prefix_sha256'])
    assert start['reused_prefix_sha256']==end['reused_prefix_sha256']
    assert start['reused_prefix']==end['reused_prefix']==reused
    state,evidence=prefix_evidence(binding,reused,files)
    manifest_path=root/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    split_path=root/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
    manifest=files.json(manifest_path,MANIFEST);split=files.json(split_path,SPLIT)
    assert [t['task'] for t in manifest['tasks']]==list(range(1,20))
    task=manifest['tasks'][15];assert task['task']==16 and task['domain']=='finance'
    fit_ids=split['fit_ids']['16'];dev_ids=split['dev_ids']['16']
    assert len(fit_ids)==69 and len(dev_ids)==20 and len(set(fit_ids+dev_ids))==89
    train=task['files']['train'];raw=files.checked(inside(root,train['path']),train['sha256']).read_bytes()
    rows=[json.loads(line) for line in raw.splitlines()];assert len(rows)==train['count']==89
    by_id={r['id']:r for r in rows};assert set(by_id)==set(fit_ids+dev_ids)
    ordered=sorted(fit_ids,key=lambda i:hashlib.sha256(f'42:16:{i}'.encode()).digest())
    plan=dict(stage=16,QA=69,updates=5,order_sha256=digest(''.join(i+'\n' for i in ordered).encode()))
    assert binding['training_plan']==[dict(plan,calibration_QA=16)]
    expected_binding=dict(manifest_path=str(manifest_path),manifest_sha256=MANIFEST,
        split_path=str(split_path),split_sha256=SPLIT,FIT_QA=69,DEV_QA=20,
        original_TRAIN_QA=1313,stage=16,TEST_QA=0,binding_sha256=files.hashes[str(binding_path)])
    deps={str(source_dir/n):binding['sources'][n] for n in ('worker.py','adaptation.py','calibration.py','prefix.py')}
    deps.update({str(inside(root,n)):h for n,h in binding['inherited_sources'].items()})
    for path,h in deps.items():files.checked(Path(path),h)
    phases={};seen=set();previous_exit=None
    assert len(end['phases'])==4
    for (arm,phase),item in zip(ORDER,end['phases']):
        assert (item['arm'],item['phase'])==(arm,phase)
        assert item['returncode']==0 and item['identity_absent'] and not item['timeout']
        who=item['identity'];assert who['pid']==who['pgid'] and not same_live(who)
        ident=(who['pid'],str(who['startticks']));assert ident not in seen;seen.add(ident)
        directory=run/arm;assert Path(item['run_dir'])==directory
        outer=files.json(run/f'PHASE_{arm}_{phase}_STARTED.json')
        exited=files.json(run/f'PHASE_{arm}_{phase}_EXIT.json')
        assert exited=={k:v for k,v in item.items() if k not in ('started_sha256','complete_sha256','checkpoint')}
        assert outer['identity']==who and outer['GPU']==1 and outer['limit_seconds']==LIMITS[phase]
        assert (outer['arm'],outer['phase'])==(arm,phase)
        if previous_exit is not None:assert outer['unix']>=previous_exit
        previous_exit=item['exit_unix']
        started=files.json(directory/(phase+'_STARTED.json'),item['started_sha256'])
        completed=files.json(directory/(phase+'_COMPLETE.json'),item['complete_sha256'])
        assert not (directory/(phase+'_FAILURE.json')).exists()
        assert started['identity']==completed['identity']==who and started['GPU']==completed['GPU']==1
        assert (started['arm'],started['phase'])==(arm,phase)==(completed['arm'],completed['phase'])
        assert started['limit_seconds']==LIMITS[phase] and 0<=finite(completed['seconds'])<=LIMITS[phase]
        assert completed['source_sha256']==files.json(directory/'SOURCE_BINDING.json')==deps
        assert completed['data_binding']==files.json(directory/'INPUT_BINDING.json')==expected_binding
        assert completed['new_scores']==completed['TEST_QA']==0 and completed['stage']==16
        assert completed['new_generations']==(20 if phase=='train' else 25)
        assert completed['FIT_backwards']==(69 if phase=='train' else 0)
        assert completed['optimizer_updates']==(5 if phase=='train' else 0)
        for n,h in completed['files'].items():files.checked(inside(directory,n),h)
        if phase=='train':
            assert completed['before_QA']==20 and completed['acquired_QA']==completed['final_QA']==0
            assert completed['prefix_validation']==reused and completed['calibration_backwards']==16
            assert completed['new_common_seed']==42
            receipt(completed['training_end'],arm,False);receipt(completed['checkpoint'],arm,True)
            cp=completed['checkpoint'];assert cp['self_contained'] and cp['no_teacher_or_optimizer']
            assert Path(cp['path'])==directory/'complete.pt' and (directory/'complete.pt').stat().st_size==cp['bytes']
            files.checked(directory/'complete.pt',cp['sha256'])
        else:
            cp=phases[arm,'train']['checkpoint']
            assert completed['final_QA']==20 and completed['complete_state_and_buffers_unchanged']
            for key in RECEIPT_KEYS:assert completed['reloaded'][key]==completed['invariant'][key]==cp[key]
            assert completed['reloaded']['strict_complete_reload'] and completed['reloaded']['self_contained_meta_construction']
            resource=completed['resource_proof'];prediction=completed['prediction_proof']
            assert resource['new_generations']==5 and resource['fixed_new_tokens']==32 and resource['actual_device']=='cuda:1'
            assert resource['window_order']=='fresh_reload_before_any_DEV'
            assert prediction['new_generations']==20 and prediction['task']==16
            assert prediction['input_scope']=='original20_finance_DEV' and prediction['training_RNG_restored']
            assert prediction['complete_state_and_buffers_unchanged']
        assert item['checkpoint']=={k:cp[k] for k in ('path','bytes','sha256')}
        phases[arm,phase]=completed
    return end,manifest,split,by_id,fit_ids,dev_ids,plan,phases,reused,state,evidence


def training(run,phases,fit_ids,plan,reused,state,evidence,files):
    reference=None
    for arm in ARMS:
        d=run/arm;t=phases[arm,'train'];initial=files.json(d/'ADAPTER_INITIAL.json')
        assert initial['coordinates']==540672 and initial['parameters']==96 and initial['rank_plan']==plan_names()
        mode(initial['mode'],arm,False,True);assert len(initial['initialization'])==48
        for r in initial['initialization']:
            assert r['rank']==8 and r['alpha']==32 and r['scale']==4 and r['dropout']==.1
            assert r['B_zero'] and r['prefix_r8_exact'];hash_string(r['A_digest']);hash_string(r['B_digest'])
        base=t['shared_prefix'];assert base['prefix']==reused and base['new_common_seed']==42
        assert base['strict_loaded_prefix_core_digest']==state['model_digest']
        assert base['strict_loaded_prefix_buffers_digest']==state['buffers_digest']
        assert base['old_RNG_restored_by_loader_then_deliberately_replaced']
        assert base['new_training_updates']==base['new_generations']==base['new_scores']==0
        assert base['reused_prefix_FIT']==959 and base['reused_prefix_updates']==69
        assert base['shared_fold']['arm']=='shared' and not base['shared_fold']['explicit_bank']
        assert len(base['shared_fold']['folded_modules'])==48
        dense=base['common_dense'];assert dense['layout']['weights']==dict(coefficients=893675552,bytes=1787351104)
        assert dense['layout']['ordinary_buffers']['bytes']==9032 and dense['all_buffers_digest']==state['buffers_digest']
        assert base['generation']['public_sha256']=={k:v['public_sha256'] for k,v in evidence['configs'].items()}
        assert base['generation']['original_public_fields_unchanged']
        fit=files.json(d/'FIT_INPUTS.json')
        ids=sorted(fit_ids,key=lambda i:hashlib.sha256(f'42:16:{i}'.encode()).digest())
        assert [(r['task'],r['id'],r['phase']) for r in fit]==[(16,i,'TRAIN') for i in ids]
        assert t['training_plan']==[plan] and t['training_ast_sha256']==TRAIN_AST
        assert t['coordinates']==540672 and t['parameters']==96
        assert t['non_target_frozen_unchanged'] and t['buffers_unchanged']
        assert t['local_initial_fold_B_zero'] and t['prior_shared_update_folded_into_common_base']
        assert t['adapter_before']!=t['adapter_after']
        events=[json.loads(line) for line in files.checked(d/'training/logs/events.jsonl').read_bytes().splitlines()]
        assert len(events)==5 and [r['step'] for r in events]==list(range(1,6))
        for r in events:
            assert r['event']=='train_update' and r['stage']==16 and r['warmup_updates']==0
            assert any(finite(lr)>0 for lr in r['learning_rate_used']) and finite(r['loss'])>=0
            assert finite(r['gradient_l2_preclip'])>0
            assert files.json(d/f'grad_S16_{r["step"]:03}.json')==dict(stage=16,step=r['step'],parameters=96,gradient_bytes=2162688,finite=True)
        validity=files.json(d/'training/validity/stage_16.json')
        assert validity['passed'] and validity['actual_parameters_changed'] and validity['updates']==validity['positive_lr_updates']==5
        snap=files.json(d/'STAGE_16_COMPLETE.json');assert snap['stage']==16 and snap['updates']==5
        assert snap['optimizer_state_bytes']==4325760 and snap['all_optimizer_state_GPU'] and snap['frozen_unchanged']
        assert snap['bytes']==2162688;files.checked(d/'adapter_S16.pt',snap['snapshot_sha256'])
        event=files.json(d/'DIRECTION_S16.json');assert t['direction_events']==[event]
        fold,cal,reset=event['fold'],event['calibration'],event['reset']
        assert event['stage']==fold['stage']==cal['stage']==reset['stage']==16
        assert fold['parameter_objects_preserved'];merge_records(fold['folded_modules'],plan_names())
        assert all(r['B_zero_after'] and r['update_max_abs']==0 for r in fold['folded_modules'])
        assert cal['QA']==cal['backwards']==16 and cal['optimizer_updates']==0 and cal['gradient_bytes']==88080384
        assert cal['ordered_IDs_sha256']==digest(''.join(i+'\n' for i in fit_ids[:16]).encode())
        assert [(r['task'],r['id'],r['phase']) for r in cal['input_records']]==[(16,i,'DIRECTION_CALIBRATION') for i in fit_ids[:16]]
        assert len(cal['answer_losses'])==16 and all(finite(v)>=0 for v in cal['answer_losses'])
        assert cal['QA_equal_weight'] and cal['current_task_only'] and cal['model_state_unchanged'] and cal['RNG_restored']
        assert cal['gradient_dtype']=='torch.float32' and cal['forward_autocast']=='torch.float16'
        assert reset['arm']==arm and reset['new_branch_zero_for_every_dropout_mask'] and reset['first_step_A_gradient_zero']
        assert len(reset['modules'])==48 and {r['module'] for r in reset['modules']}==set(plan_names())
        for r in reset['modules']:
            assert r['gradient_shape']==[896 if r['module'].endswith('q_proj') else 128,896]
            assert r['gradient_dtype']==r['svd_dtype']=='torch.float32' and r['svd_full_matrices'] is False
            assert r['B_zero'] and r['A_row_norm_target']==1/math.sqrt(3)
            for k in ('gradient_digest','A_digest','B_digest'):hash_string(r[k])
            assert 0<=finite(r['A_row_norm_max_error'])<=1e-4 and 0<=finite(r['basis_orthogonality_max_error'])<=1e-4
            assert len(r['singular_values_top_r'])==8 and all(finite(x)>=0 for x in r['singular_values_top_r'])
            assert finite(r['total_squared_singular_energy'])>=0 and finite(r['top_r_squared_singular_energy'])>=0
            assert 0<=finite(r['top_r_energy_fraction'])<=1.00001
            assert r['main_RNG_used_for_random_direction'] is False
            if arm=='aligned':assert r['private_random_seed'] is None and r['direction']=='task_gradient_top_right_singular_vectors'
            else:
                assert r['private_random_seed']==int(hashlib.sha256(f'42:task-direction:16:{r["module"]}'.encode()).hexdigest()[:16],16)
                assert r['direction']=='random_orthogonal_rows'
        mode(reset['metadata'],arm,False,True);assert fold['metadata']==reset['metadata']
        ignored={'active_training_coordinates','active_training_parameter_tensors'}
        assert {k:v for k,v in reset['metadata'].items() if k not in ignored}=={k:v for k,v in t['training_end']['method'].items() if k not in ignored}
        merge_records(t['deployment']['folded_modules'],plan_names())
        assert not t['deployment']['explicit_bank'] and t['deployment']['metadata']==t['checkpoint']['method']
        common=dict(dense=dense,loaded_core=base['strict_loaded_prefix_core_digest'],
            loaded_buffers=base['strict_loaded_prefix_buffers_digest'],generation=base['generation'],
            initial_AB=initial['initialization'],adapter_before=t['adapter_before'],FIT_inputs=fit,plan=t['training_plan'],
            calibration_inputs=cal['input_records'])
        if reference is None:reference=common
        assert common==reference,'Shared prefix, folded W15, initialization, calibration or FIT inputs differ'


def answers(run,phases,manifest,split,by_id,dev_ids,files):
    answer_rows={};bindings={};before_reference=None;resources=None
    task=manifest['tasks'][15]
    for arm in ARMS:
        d=run/arm
        for name,phase in (('before','DEV_BEFORE'),('final','DEV_FINAL')):
            rows=files.json(d/(name+'_predictions.json'))
            assert len(rows)==20 and [r['id'] for r in rows]==dev_ids
            for r in rows:
                original=by_id[r['id']]
                assert r['task']==16 and r['domain']=='finance' and r['stage']==16 and r['phase']==phase
                assert r['references']==original['answers'] and isinstance(r['prediction'],str)
                assert r['generation_cap']==task['max_new_tokens'] and isinstance(r['output_tokens'],list)
                assert len(r['output_tokens'])<=r['generation_cap']
                entry={k:r[k] for k in ('raw_input_sha256','transport_sha256','prepared_sha256','input_shape','input_tensor_shapes','image_SHA256')}
                assert entry['image_SHA256']==[dict(path=n,sha256=split['image_sha256'][n]) for n in original['images']]
                if r['id'] not in bindings:bindings[r['id']]=entry
                assert bindings[r['id']]==entry
            answer_rows[arm,name]=rows
            if name=='before':
                current=[(r['id'],r['prediction'],r['output_tokens']) for r in rows]
                if before_reference is None:before_reference=current
                assert current==before_reference,'Before outputs differ despite the common folded W15'
        event=files.json(d/'READOUT_EVENTS.json')
        assert len(event)==1 and event[0]['phase']=='DEV_BEFORE' and event[0]['stage']==16 and event[0]['new_generations']==20
        assert event[0]['state_and_buffers_unchanged'] and event[0]['training_RNG_restored']
        ev=phases[arm,'eval']['prediction_proof']['events']
        assert len(ev)==1 and ev[0]['stage']==16 and ev[0]['phase']=='DEV_ACQUIRED' and ev[0]['new_generations']==20
        assert ev[0]['state_and_buffers_unchanged'] and ev[0]['training_RNG_restored']
        windows=files.json(d/'memory.json');assert len(windows)==5
        reference=[]
        for domain,w in zip(DOMAINS,windows):
            ids=[i for t in manifest['tasks'] if t['domain']==domain for i in split['dev_ids'][str(t['task'])]]
            selected=min(ids,key=lambda i:hashlib.sha256(i.encode()).digest())
            assert w['domain']==domain and w['id']==selected and w['new_tokens']==32
            assert 0<=finite(w['baseline_allocated'])<=finite(w['peak_allocated'])<=finite(w['peak_reserved'])
            if selected in bindings:
                assert w['raw_input_sha256']==bindings[selected]['raw_input_sha256'] and w['prepared_sha256']==bindings[selected]['prepared_sha256']
            reference.append({k:w[k] for k in ('id','domain','raw_input_sha256','prepared_sha256','new_tokens')})
        if resources is None:resources=reference
        assert reference==resources
    return answer_rows


def paired(left,right):
    assert [(r['task'],r['id_sha256']) for r in left]==[(r['task'],r['id_sha256']) for r in right]
    changed=[dict(id_sha256=a['id_sha256'],before=a['F1'],after=b['F1'],delta=b['F1']-a['F1'])
             for a,b in zip(left,right) if a['F1']!=b['F1']]
    return dict(QA=len(left),mean_difference_pp=sum(b['F1']-a['F1'] for a,b in zip(left,right))/len(left),
        higher=sum(b['F1']>a['F1'] for a,b in zip(left,right)),lower=sum(b['F1']<a['F1'] for a,b in zip(left,right)),
        tie=sum(b['F1']==a['F1'] for a,b in zip(left,right)),changed=changed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('root','run','binding'):parser.add_argument('--'+n,type=Path,required=True)
    a=parser.parse_args();assert all(p.is_absolute() for p in (a.root,a.run,a.binding))
    root,run=a.root,a.run;began=time.monotonic();files=Files()
    assert not any((run/n).exists() for n in ('VERIFY_STARTED.json','VERIFY_FAILURE.json','VERIFIED.json','OUTCOME.json'))
    binding=files.json(a.binding)
    supervisor,manifest,split,by_id,fit_ids,dev_ids,plan,phases,reused,state,evidence=metadata(root,run,a.binding,binding,files)
    training(run,phases,fit_ids,plan,reused,state,evidence,files)
    pending=answers(run,phases,manifest,split,by_id,dev_ids,files)
    score=scorer(root,files);own=files.checked(Path(__file__));own_sha=files.hashes[str(own)]
    write(run/'VERIFY_STARTED.json',dict(unix=time.time(),pid=os.getpid(),verifier_sha256=own_sha,
        all_four_phases_exit0_and_integrity_passed=True,planned_scoring_calls=80,
        new_models=0,new_generations=0,exclusive_no_automatic_retry=True))
    calls=0
    try:
        scored={};quality={}
        for arm in ARMS:
            quality[arm]={}
            for source,label in (('before','before'),('final','acquired')):
                rows=[]
                for r in pending[arm,source]:
                    values=score(r['prediction'],r['references']);calls+=1
                    assert set(values)=={'token_f1','exact_match','rougeL'} and all(0<=finite(v)<=100 for v in values.values())
                    rows.append(dict(id_sha256=digest(r['id'].encode()),task=16,domain='finance',F1=values['token_f1'],
                                     prediction_sha256=digest(r['prediction'].encode())))
                scored[arm,label]=rows
                quality[arm][label]=dict(QA=20,original_F1_mean=sum(r['F1'] for r in rows)/20,per_id=rows)
            quality[arm]['acquired_minus_before']=paired(scored[arm,'before'],scored[arm,'acquired'])
        assert calls==80
        comparison={label:paired(scored['random',label],scored['aligned',label]) for label in ('before','acquired')}
        assert comparison['before']['higher']==comparison['before']['lower']==0
        resources={arm:dict(training_seconds=phases[arm,'train']['seconds'],eval_seconds=phases[arm,'eval']['seconds'],
            training_monitor_includes_20_before_readouts=True,training_memory=files.json(run/arm/'TRAIN_MEMORY.json'),
            final_layout=phases[arm,'train']['checkpoint']['layout'],checkpoint_file_bytes=phases[arm,'train']['checkpoint']['bytes'],
            checkpoint_sha256=phases[arm,'train']['checkpoint']['sha256'],windows=files.json(run/arm/'memory.json'),unit='bytes') for arm in ARMS}
        files.unchanged()
        outcome=dict(status='COMPLETE_SAME_START_T16_FUNCTIONAL_CONTRAST',quality=quality,
            aligned_minus_random=comparison,resources=resources,unique_existing_answers_scored=80,
            new_model_generations=90,new_FIT_backwards=138,new_calibration_backwards=32,new_optimizer_updates=10,
            new_total_backwards=170,reused_prefix=reused,prefix_history_is_not_new_execution=True,
            no_automatic_continuation=True,formal_main_table_result=False,
            limits=['One shared-folded S15 start, T16 69 FIT/20 explored DEV, one seed; no five-domain quality macro.',
                'Acquired is the actual S16 folded graph after fresh-process reload; no separate S19 final or forgetting claim.',
                'Same-start aligned/random tests current-task direction information under common fold/reset/calibration.',
                'B0 first-step A gradient is zero; SVD top-energy is a construction property, not a quality result.',
                'Five-window resource values are not full-TEST worst-case memory or a strong-baseline replacement proof.',
                'No significance, equivalence, capacity-root-cause or Ours claim follows automatically.'])
        write(run/'OUTCOME.json',outcome)
        write(run/'VERIFIED.json',dict(status='VERIFIED_ONCE',verifier_sha256=own_sha,scoring_calls=80,
            supervisor_complete_sha256=files.hashes[str(run/'SUPERVISOR_COMPLETE.json')],source_and_artifact_sha256=files.hashes,
            scorer_source_sha256=SCORER_FILE,outcome_sha256=digest((run/'OUTCOME.json').read_bytes()),
            all_four_phases_real_exit0=True,same_folded_prefix_and_initial_AB_and_before_outputs=True,
            new_models=0,new_generations=0,new_TEST_scores=0,old_prefix_CP_deserialized=False,
            seconds=time.monotonic()-began))
    except BaseException:
        write(run/'VERIFY_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),
            scoring_calls_completed=calls,no_automatic_retry=True));raise


if __name__=='__main__':main()
