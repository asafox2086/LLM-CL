"""Verify the one-shot full-TRAIN QV confirmation; score21454 final TEST answers once.

No Torch import, checkpoint deserialization, model execution or historical
rescoring. Overall macro F1 is primary; no per-domain nondecrease gate.
"""
from pathlib import Path
import ast
import collections
import hashlib
import importlib.util
import json
import math
import os
import re
import time

from common import ROOT, RUN, ASSET, LIMITS, read, sha, write, load_formal, shard_for, DOMAIN_TEST_COUNTS

DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
WEIGHTS = {'coefficients': 893675552, 'bytes': 1787351104}
GRADIENT_BYTES = 17596416
OPTIMIZER_BYTES = 35193216


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def digest_string(value):
    assert isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def keyed(rows):
    result = {row['id']: row for row in rows}
    assert len(result) == len(rows)
    return result


def expected_plan():
    names = [f'model.layers.{i}.self_attn.v_proj' for i in range(24)]
    chosen = set(sorted(names, key=lambda n: hashlib.sha256(('42:coordinate-remainder:' + n).encode()).hexdigest())[:6])
    return {f'model.layers.{i}.self_attn.{kind}_proj': 65 + int(f'model.layers.{i}.self_attn.{kind}_proj' in chosen)
            for i in range(24) for kind in ('q', 'v')}


def receipt(value, gpu):
    assert value['layout']['weights'] == WEIGHTS and value['layout']['trainable'] == 0
    assert value['layout']['ordinary_buffers']['bytes'] == 9032
    assert value['device'] == f'cuda:{gpu}' and value['response'] is None
    for key in ('digest', 'all_buffers_digest', 'functional_config_digest', 'generation_config_digest', 'language_generation_config_digest'):
        digest_string(value[key])
    assert not any(name.endswith(('.mlp.beta', '.mlp.response_sign', '.mlp.calibration_scale'))
                   for name in value['buffer_manifest'])
    assert all(x['dtype'] == 'torch.float32' for n, x in value['buffer_manifest'].items() if n.endswith('inv_freq'))


def same_receipt(left, right, *, device=True):
    keys = ('layout', 'digest', 'all_buffers_digest', 'buffer_manifest', 'parameter_alias_groups',
            'buffer_alias_groups', 'functional_config_digest', 'generation_config_digest',
            'language_generation_config_digest', 'language_attention_classes', 'vision_attention_classes', 'response')
    for key in keys:
        assert left[key] == right[key], key
    if device:
        assert left['device'] == right['device']


def same_function(left_path, right_path, name):
    def find(path):
        return next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(find(left_path), include_attributes=False) == ast.dump(find(right_path), include_attributes=False), name



def rows_digest(rows):
    return hashlib.sha256(''.join(f"{t['task']}:{r['id']}\n" for t,r in rows).encode()).hexdigest()


def main():
    tick=time.monotonic()
    assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists()
    assert not list(RUN.glob('*_FAILURE.json'))
    ready=read(RUN/'READY.json');rh=sha(RUN/'READY.json')
    for field in ('sources','prior_files'):
        for n,h in ready[field].items():assert sha(RUN/n)==h,n
    for n,h in ready['environment'].items():assert sha(n)==h,n
    for n,h in ready['runtime_files'].items():
        if not n.endswith('.safetensors'):assert sha(ASSET/n)==h,n
    assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
    prior=read(RUN/'prior_qv_VERIFIED.json');prior_ready=read(RUN/'prior_qv_READY.json')
    assert prior['complete'] and prior['ready_sha256']==sha(RUN/'prior_qv_READY.json')
    assert prior['TEST_QA']==0 and prior['new_scores']==100 and prior['matched_GPU_QV_only_control']
    assert prior['quality']['qv_final']['macro']>=max(prior['quality'][a+'_final']['macro'] for a in ('dynamic','static'))
    assert ready['selected_candidate']=='bare_wide_QV' and ready['selection_before_TEST'] is True
    old_train=read(RUN/'prior_qv_train_COMPLETE.json');old_initial=read(RUN/'prior_qv_ADAPTER_INITIAL.json')
    assert sha(RUN/'prior_qv_train_COMPLETE.json')==prior['source_COMPLETE_sha256']['train_qv']
    assert sha(RUN/'prior_qv_ADAPTER_INITIAL.json')==old_train['files']['qv_ADAPTER_INITIAL.json']
    calibration=read(RUN/'calibration_COMPLETE.json')
    assert sha(RUN/'calibration_COMPLETE.json')==prior_ready['prior_files']['calibration_COMPLETE.json']
    old_run=Path(ready['prior_qv_run'])
    for n in ('control_adaptation.py','control_deployment.py'):
        assert sha(old_run/n)==prior_ready['sources'][n] and sha(RUN/n)==sha(old_run/n)
    assert sha(old_run/'readouts.py')==prior_ready['sources']['readouts.py']
    for name in ('_batch','_generate','resource_windows'):
        same_function(RUN/'readouts.py',old_run/'readouts.py',name)
    pas=read(RUN/'prior_PAS_VERIFIED.json');pas_complete=read(RUN/'prior_PAS_COMPLETE.json');pas_memory=read(RUN/'prior_PAS_memory.json')
    assert pas['complete'] and pas['actual_successful_exit'] and pas['new_scores']==0
    assert sha(RUN/'prior_PAS_COMPLETE.json')==pas['COMPLETE_sha256']
    assert sha(RUN/'prior_PAS_memory.json')==pas['memory_sha256']==pas_complete['memory_sha256']

    expected_limits={'train':3600,'resource':600,**{f'test_{i}':14400 for i in range(4)}}
    assert LIMITS==expected_limits
    all_exit=read(RUN/'ALL_EXIT.json');assert all_exit['complete'] and all_exit['ready_sha256']==rh
    assert set(all_exit['source_COMPLETE_sha256'])==set(LIMITS)
    for n,v in dict(new_generations=21459,new_scores=0,TRAIN_backwards=1313,optimizer_updates=93,TEST_QA=21454).items():assert all_exit[n]==v
    ends,starts,exits={},{},{}
    for phase,limit in LIMITS.items():
        start=read(RUN/(phase+'_STARTED.json'));end=read(RUN/(phase+'_COMPLETE.json'));ex=read(RUN/(phase+'_EXIT.json'))
        who=start['identity'];assert who==end['identity'] and who['pid']==who['pgid']==ex['pid'] and who['uid']==os.getuid()
        proc=Path('/proc')/str(who['pid'])/'stat'
        if proc.exists():assert proc.read_text().rsplit(') ',1)[1].split()[19]!=str(who['startticks'])
        assert ex['returncode']==0 and start['ready_sha256']==end['ready_sha256']==rh
        assert start['limit']==limit and finite(end['seconds']) and 0<end['seconds']<=limit
        assert finite(start['unix']) and finite(ex['unix']) and ex['unix']>=start['unix']+end['seconds']
        assert start['GPU']==ready['phase_GPUs'][phase] and start['GPU_uuid']==ready['GPU_uuids'][str(start['GPU'])]
        assert sha(RUN/(phase+'_COMPLETE.json'))==all_exit['source_COMPLETE_sha256'][phase]
        for n,h in end['files'].items():assert sha(RUN/n)==h,n
        ends[phase],starts[phase],exits[phase]=end,start,ex
    assert len({x['boot'] for x in starts.values()})==1
    assert starts['train']['GPU']==0 and starts['resource']['GPU']==1
    assert starts['resource']['GPU_uuid']==pas['GPU_uuid'] and pas['physical_GPU']==1
    assert starts['resource']['unix']>=exits['train']['unix']
    assert all(starts[f'test_{i}']['unix']>=exits['resource']['unix'] and starts[f'test_{i}']['GPU']==i for i in range(4))
    assert max(starts[f'test_{i}']['unix'] for i in range(4))<min(exits[f'test_{i}']['unix'] for i in range(4))
    assert sum(x['seconds'] for x in ends.values())<=61800

    manifest,data,plan=load_formal(ready,include_test=True)
    assert plan==ready['training_plan'] and len(plan)==19 and sum(x['updates'] for x in plan)==93
    train_rows=[(t,r) for t in manifest['tasks'] for r in data[t['task'],'train']]
    test_rows=[(t,r) for t in manifest['tasks'] for r in data[t['task'],'test']]
    assert len(train_rows)==1313 and len(test_rows)==21454
    train_ids={r['id'] for t,r in train_rows};gold={r['id']:(t,r) for t,r in test_rows};assert len(gold)==21454 and train_ids.isdisjoint(gold)
    port=ASSET/'exp/two_backbone_native_cl_20261017_v2'
    original_main=next(n for n in ast.parse((port/'run.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop=next(n for n in original_main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='stage')
    cutoff=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='wanted' for t in n.targets))
    prefix=ast.Module(body=loop.body[:cutoff-1],type_ignores=[])
    assert hashlib.sha256(ast.dump(prefix,include_attributes=False).encode()).hexdigest()==ready['training_ast_sha256']==prior_ready['training_ast_sha256']

    tr=ends['train'];assert tr['arm']=='qv' and tr['new_generations']==0 and tr['TRAIN_backwards']==1313 and tr['optimizer_updates']==93 and tr['coordinates']==4399104
    assert tr['training_plan']==plan and tr['training_ast_sha256']==ready['training_ast_sha256']
    assert tr['frozen_unchanged'] and tr['buffers_unchanged'] and tr['adapter_before']!=tr['adapter_after']
    assert tr['paired_initialization_exact'] and tr['original_full_TRAIN_order_exact'] and tr['no_response_module']
    needed={'qv_ADAPTER_INITIAL.json','qv_TRAIN_INPUTS.json','qv_TRAIN_MEMORY.json','qv_initial.pt','qv_complete.pt','qv_training/logs/events.jsonl'}
    for x in plan:
        sid=x['stage'];needed.update({f'qv_STAGE_{sid:02}_COMPLETE.json',f'qv_adapter_S{sid:02}.pt',f'qv_training/validity/stage_{sid:02}.json'})
        needed.update(f'qv_grad_S{sid:02}_{i:03}.json' for i in range(1,x['updates']+1))
    assert needed<=set(tr['files'])
    assert len(list(RUN.glob('qv_grad_S*.json')))==93 and len(list(RUN.glob('qv_STAGE_*_COMPLETE.json')))==19 and len(list(RUN.glob('qv_adapter_S*.pt')))==19
    for field in ('original_base','initial_checkpoint','initial_reload','initial_invariant','checkpoint'):receipt(tr[field],0)
    same_receipt(tr['original_base'],old_train['original_base'],device=False)
    assert tr['original_base']['digest']==calibration['initial_state_digest'] and tr['original_base']['all_buffers_digest']==calibration['all_buffers_digest'] and tr['original_base']['layout']==calibration['layout']
    same_receipt(tr['original_base'],tr['initial_checkpoint']);same_receipt(tr['initial_checkpoint'],tr['initial_reload']);same_receipt(tr['initial_reload'],tr['initial_invariant'])
    assert tr['initial_invariant']['complete_state_and_buffers_unchanged']
    for suffix,field in [('initial','initial_checkpoint'),('complete','checkpoint')]:
        cp=tr[field];name='qv_'+suffix+'.pt';assert cp['sha256']==tr['files'][name] and (RUN/name).stat().st_size==cp['bytes']
        assert Path(cp['path']).resolve()==(RUN/name).resolve()
        assert cp['complete_graph'] and not cp['external_base_weights_required'] and cp['no_adapter_optimizer_or_teacher'] and cp['alias_preserving_cpu_snapshot']
    for field in ('self_contained_meta_construction','all_buffers_explicitly_restored','strict_complete_reload','mixed_parameter_dtypes_preserved'):assert tr['initial_reload'][field]
    init=read(RUN/'qv_ADAPTER_INITIAL.json');ranks=expected_plan()
    assert init['coordinates']==4399104 and init['parameters']==96 and init['rank_plan']==ranks==old_initial['rank_plan']
    assert init['initialization']==old_initial['initialization'] and len(init['initialization'])==48 and init['paired_initialization_exact'] and init['no_response_module']
    assert {x['module']:x['rank'] for x in init['initialization']}==ranks
    assert all(x['B_zero'] and x['scale']==4 and x['dropout']==.1 for x in init['initialization'])
    assert len(tr['fold'])==48 and {x['module']:x['rank'] for x in tr['fold']}==ranks
    assert all(x['scale']==4 and x['dtype']=='torch.float16' and x['merge_dtype']=='torch.float32' and x['device']=='cuda:0' and not x['autocast_enabled'] and not x['matmul_allow_tf32'] for x in tr['fold'])
    events=[json.loads(line) for line in (RUN/'qv_training/logs/events.jsonl').read_bytes().splitlines()];assert len(events)==93
    for x in plan:
        sid,updates=x['stage'],x['updates'];stage=read(RUN/f'qv_STAGE_{sid:02}_COMPLETE.json');valid=read(RUN/f'qv_training/validity/stage_{sid:02}.json')
        assert stage['stage']==sid and stage['updates']==updates and stage['bytes']==GRADIENT_BYTES
        assert stage['optimizer_state_bytes']==OPTIMIZER_BYTES and stage['all_optimizer_state_GPU'] and stage['frozen_unchanged'];digest_string(stage['adapter_digest'])
        assert valid['passed'] and valid['actual_parameters_changed'] and valid['updates']==valid['positive_lr_updates']==updates
        assert valid['parameters_before_sha256']!=valid['parameters_after_sha256']
        for i in range(1,updates+1):assert read(RUN/f'qv_grad_S{sid:02}_{i:03}.json')==dict(parameters=96,finite=True,gradient_bytes=GRADIENT_BYTES)
        ev=[e for e in events if e['stage']==sid];assert len(ev)==updates;warmup=min(math.floor(.03*updates),updates-1)
        for k,e in enumerate(ev):
            assert e['event']=='train_update' and e['step']==k+1 and e['qa_seen']==min((k+1)*16,x['QA']) and e['total_qa']==x['QA'] and e['warmup_updates']==warmup and finite(e['loss'])
            lr=1e-4*(.5*(1+math.cos(math.pi*(k-warmup)/max(1,updates-warmup))) if k>=warmup else k/max(1,warmup))
            assert len(e['learning_rate_used'])==1 and e['learning_rate_used'][0]>0 and math.isclose(e['learning_rate_used'][0],lr,rel_tol=1e-12,abs_tol=1e-15)
            assert finite(e['gradient_l2_preclip']) and e['gradient_l2_preclip']>=0 and finite(e['gradient_l2_postclip']) and 0<=e['gradient_l2_postclip']<=1.00001
            assert e['visual_cache_cpu_bytes']==e['visual_cache_cpu_limit']==0

    spec=importlib.util.spec_from_file_location('formal_raw_retrieval',port/'retrieval.py');retrieval=importlib.util.module_from_spec(spec);spec.loader.exec_module(retrieval)
    from PIL import Image
    image_sizes={};raw_cache={};images=dict(ready['TRAIN_image_sha256'])
    for n,h in ready['TEST_image_sha256'].items():assert n not in images or images[n]==h;images[n]=h
    def raw(task,row):
        if row['id'] in raw_cache:return raw_cache[row['id']]
        context=row.get('context','');indexes=[]
        if len(context)>6000:
            chunks=[context[i:i+1000] for i in range(0,len(context),800)];indexes=sorted(retrieval.retrieve(row['question'],chunks,5));context='\n'.join(chunks[i] for i in indexes)
        shapes=[];refs=[]
        for n in row['images']:
            if n not in image_sizes:
                with Image.open(ASSET/n) as im:w,h=im.size
                before=[w,h]
                if w*h>401408:
                    ratio=math.sqrt(401408/(w*h));w,h=max(1,int(w*ratio)),max(1,int(h*ratio))
                image_sizes[n]=dict(original=before,common_input=[w,h])
            shapes.append(image_sizes[n]);refs.append(dict(path=n,sha256=images[n]))
        contract=dict(images=refs,instruction=task['instruction'],question=row['question'],original_context=row.get('context',''),resolved_context=context,retrieved_chunk_indexes=indexes,image_shapes=shapes,common_image_area_cap=401408)
        result=(hashlib.sha256(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),contract);raw_cache[row['id']]=result;return result
    def bound(x,task,row):
        h,contract=raw(task,row)
        assert x['id']==row['id'] and x['task']==task['task'] and x['domain']==task['domain']
        assert x['raw_input_sha256']==h and x['image_SHA256']==contract['images'] and x['image_shapes']==contract['image_shapes']
        assert x['common_image_area_cap']==401408 and not x['visual_cache_hit'];digest_string(x['transport_sha256'])
    inputs=read(RUN/'qv_TRAIN_INPUTS.json');expected_order=[(t,r) for t in manifest['tasks'] for r in sorted(data[t['task'],'train'],key=lambda r:hashlib.sha256(f"42:{t['task']}:{r['id']}".encode()).digest())]
    assert len(inputs)==1313 and [x['id'] for x in inputs]==[r['id'] for t,r in expected_order]
    for x,(task,row) in zip(inputs,expected_order):assert x['phase']=='TRAIN';bound(x,task,row)

    for phase in ('resource',*[f'test_{i}' for i in range(4)]):
        e=ends[phase];gpu=starts[phase]['GPU'];receipt(e['reloaded'],gpu);receipt(e['invariant'],gpu)
        same_receipt(tr['checkpoint'],e['reloaded'],device=False);same_receipt(e['reloaded'],e['invariant']);assert e['invariant']['complete_state_and_buffers_unchanged']
        assert e['TRAIN_backwards']==e['optimizer_updates']==0 and e['checkpoint_source_sha256']==tr['checkpoint']['sha256']
        for flag in ('self_contained_meta_construction','all_buffers_explicitly_restored','strict_complete_reload','mixed_parameter_dtypes_preserved'):assert e['reloaded'][flag]
    resource=ends['resource'];assert resource['new_generations']==5 and resource['forced_new_tokens']==160 and resource['TEST_QA']==0 and resource['same_PAS_resource_inputs']
    assert set(resource['files'])=={'resource_memory.json'};memory=read(RUN/'resource_memory.json');assert len(memory)==5
    held=read(ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json')
    assert sha(ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json')==ready['train_split_sha256']
    train_by_id={r['id']:(t,r) for t,r in train_rows};comparison=[]
    for domain,x,old in zip(DOMAINS,memory,pas_memory):
        candidates=[rid for stage,ids in held['dev_ids'].items() for rid in ids if train_by_id[rid][0]['domain']==domain]
        rid=min(candidates,key=lambda rid:hashlib.sha256(rid.encode()).digest());task,row=train_by_id[rid]
        assert x['domain']==domain and x['id']==rid and x['new_tokens']==32
        for k in ('id','domain','raw_input_sha256','transport_sha256','prepared_sha256','input_shape','input_tensor_shapes','new_tokens'):assert x[k]==old[k],k
        h,contract=raw(task,row);assert x['raw_input_sha256']==h and x['image_SHA256']==contract['images']
        assert all(type(x[k]) is int and x[k]>0 for k in ('baseline_allocated','peak_allocated','peak_reserved')) and x['baseline_allocated']<=x['peak_allocated']<=x['peak_reserved']
        comparison.append(dict(domain=domain,PAS_peak_allocated=old['peak_allocated'],QV_peak_allocated=x['peak_allocated'],reduction_percent=100*(old['peak_allocated']-x['peak_allocated'])/old['peak_allocated']))

    predictions={}
    for shard in range(4):
        phase=f'test_{shard}';end=ends[phase];expected=[(t,r) for t,r in test_rows if shard_for(t['task'],r['id'])==shard]
        assert end['shard']==shard and end['TEST_QA']==end['new_generations']==len(expected) and end['new_scores']==0
        assert end['assigned_ID_sha256']==rows_digest(expected)
        counts={str(t['task']):sum(tt['task']==t['task'] for tt,r in expected) for t in manifest['tasks']};assert end['task_counts']==counts
        proof=end['test_proof'];assert proof['new_generations']==len(expected) and proof['new_scores']==proof['NLL_requests']==proof['backwards']==proof['optimizer_updates']==0
        assert proof['batch_size']==proof['num_beams']==1 and not proof['do_sample'] and proof['original_caps'] and proof['no_resume']
        assert {str(t['task']):proof['task_counts'].get(str(t['task']),0) for t in manifest['tasks']}==counts
        name=f'test_{shard}_predictions.jsonl';assert set(end['files'])=={name}
        rows=[json.loads(line) for line in (RUN/name).read_bytes().splitlines()]
        assert len(rows)==len(expected) and [x['id'] for x in rows]==[r['id'] for t,r in expected]
        for x,(task,row) in zip(rows,expected):
            assert x['id'] not in predictions and x['phase']=='TEST_FINAL';bound(x,task,row)
            assert x['references']==row['answers'] and x['generation_cap']==task['max_new_tokens'] and isinstance(x['prediction'],str)
            assert isinstance(x['output_tokens'],list) and len(x['output_tokens'])<=x['generation_cap'] and all(type(t) is int and t>=0 for t in x['output_tokens'])
            assert x['input_shape']==[1,x['prompt_tokens']]==x['input_tensor_shapes']['input_ids'] and x['input_tensor_shapes']['attention_mask']==x['input_shape']
            digest_string(x['prepared_sha256']);predictions[x['id']]=x
    assert set(predictions)==set(gold) and len(predictions)==21454
    nodes=[n for n in ast.parse((port/'runtime_utils.py').read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')];assert {n.name for n in nodes}=={'tokens','score'}
    tree=ast.Module(body=nodes,type_ignores=[]);scorer_hash=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest();assert scorer_hash==prior['scorer_ast_sha256']
    scope=dict(re=re,Counter=collections.Counter);exec(compile(tree,'<original formal Answer F1>','exec'),scope)
    anchors=ready['formal_anchors'];assert anchors['PASs_MoE']['macro']==46.619976395158304 and anchors['OSFT']['macro']==45.93743821034059
    assert len(anchors)>=8 and all(finite(x['macro']) and finite(x['AP19']) and x['backbone'] in ('onevision','qwen') for x in anchors.values())
    assert anchors['PASs_MoE']['backbone']==anchors['OSFT']['backbone']=='onevision'
    anchor_index=read(RUN/'FORMAL_ANCHORS.json');assert anchors==anchor_index['anchors'] and anchor_index['copied_scores_only_no_rescoring']
    assert 'FORMAL_ANCHORS.json' in ready['prior_files']
    for anchor in anchors.values():
        name=anchor['proof_file'];assert name in ready['prior_files'] and ready['prior_files'][name]==anchor_index['proofs'][name]['sha256']
    same_backbone={n:a for n,a in anchors.items() if a['backbone']=='onevision'}
    assert same_backbone
    write('VERIFY_STARTED.json',dict(unix=time.time(),ready_sha256=rh,new_scores=21454,scorer_ast_sha256=scorer_hash,all_six_workers_exited=True,old_scores_recomputed=0))
    per_id=[];domain_scores={d:[] for d in DOMAINS};task_scores={t['task']:[] for t in manifest['tasks']}
    for task,row in test_rows:
        x=predictions[row['id']];value=scope['score'](x['prediction'],x['references'])['token_f1'];assert finite(value) and 0<=value<=100
        domain_scores[task['domain']].append(value);task_scores[task['task']].append(value);per_id.append(dict(id=row['id'],task=task['task'],domain=task['domain'],F1=value))
    domains={d:dict(F1=sum(v)/len(v),QA=len(v)) for d,v in domain_scores.items()};assert {d:x['QA'] for d,x in domains.items()}==DOMAIN_TEST_COUNTS
    tasks=[dict(task=t['task'],domain=t['domain'],F1=sum(task_scores[t['task']])/len(task_scores[t['task']]),QA=len(task_scores[t['task']])) for t in manifest['tasks']]
    quality=dict(domains=domains,tasks=tasks,macro=sum(x['F1'] for x in domains.values())/5,AP19=sum(x['F1'] for x in tasks)/19,per_id=per_id,FWT=None,BWT=None)
    threshold=max(a['macro'] for a in anchors.values());same_threshold=max(a['macro'] for a in same_backbone.values())
    assert threshold==anchor_index['all_sealed_global_threshold'] and same_threshold==anchor_index['same_backbone_threshold']
    quality_pass=quality['macro']>threshold;same_pass=quality['macro']>same_threshold
    resource_lower=all(x['QV_peak_allocated']<x['PAS_peak_allocated'] for x in comparison)
    result=dict(complete=True,ready_sha256=rh,all_exit_sha256=sha(RUN/'ALL_EXIT.json'),source_COMPLETE_sha256=all_exit['source_COMPLETE_sha256'],scorer_ast_sha256=scorer_hash,
        selected_candidate='bare_wide_QV',selection_prior_VERIFIED_sha256=sha(RUN/'prior_qv_VERIFIED.json'),quality=quality,formal_anchors=anchors,
        primary_metric='equal_macro_of_five_domain_QA_mean_Answer_F1',primary_threshold=threshold,primary_quality_pass=quality_pass,
        same_backbone_gate=dict(backbone='onevision',threshold=same_threshold,passed=same_pass,anchor_names=list(same_backbone)),
        all_sealed_global_gate=dict(threshold=threshold,passed=quality_pass,anchor_names=list(anchors),user_objective_primary=True),per_domain_floor_used=False,
        strongest_baseline_comparisons={n:dict(macro_difference=quality['macro']-a['macro'],AP19_difference=quality['AP19']-a['AP19'] if 'AP19' in a else None) for n,a in anchors.items()},
        memory=memory,PAS_memory_comparison=comparison,all_five_measured_peaks_lower_than_original_PAS=resource_lower,complete_checkpoint=tr['checkpoint'],
        fewer_complete_weight_coefficients_than_original_PAS=WEIGHTS['coefficients']<pas_complete['layout']['weights']['coefficients'],
        fewer_original_dtype_weight_bytes_than_PAS=WEIGHTS['bytes']<pas_complete['layout']['weights']['bytes'],
        same_dtype_optimal_deployment_superiority_proven=False,algorithm_novelty_proven=False,
        global_quality_and_PAS_resource_condition=quality_pass and resource_lower and WEIGHTS['coefficients']<pas_complete['layout']['weights']['coefficients'] and WEIGHTS['bytes']<pas_complete['layout']['weights']['bytes'],
        strongest_Qwen_actual_GPU_memory_comparison_measured=False,user_joint_goal_complete_proven=False,
        future_new_sealed_references_may_only_tighten_publication_comparison=True,selection_cannot_change_after_TEST=True,
        phase_seconds={n:e['seconds'] for n,e in ends.items()},total_GPU_model_seconds=sum(e['seconds'] for e in ends.values()),controller_seconds=all_exit['controller_seconds'],
        TRAIN_backwards=1313,optimizer_updates=93,new_generations=21459,new_scores=21454,TEST_QA=21454,TEST_cells=19,original_74_cells_measured=False,
        initial_or_intermediate_TEST_generations=0,FWT=None,BWT=None,all_TEST_IDs_exactly_once=True,prior_scores_recomputed=0,
        generation_topology='four fixed task-ID SHA256 shards; independent batch1 greedy, seed42; cross-GPU bitwise equality not tested or guaranteed',
        interpretation='One fixed fresh full-TRAIN ordinary wide-QV recipe selected before TEST. Aggregate point comparison, all domain/task/ID results retained. Actual original mixed-dtype PAS five-window resource comparison; no pure architecture, optimized same-dtype, latency, significance or new-algorithm claim.',verification_seconds=time.monotonic()-tick)
    write('VERIFIED.json',result)
    print(json.dumps(dict(VERIFIED_sha256=sha(RUN/'VERIFIED.json'),macro=quality['macro'],AP19=quality['AP19'],primary_quality_pass=quality_pass,new_scores=21454,total_GPU_model_seconds=result['total_GPU_model_seconds'])))

if __name__=='__main__':main()
