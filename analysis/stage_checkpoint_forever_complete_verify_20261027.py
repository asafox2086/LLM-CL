"""Single completed retained-stage FOREVER run, separate opt-in score cache.

No Torch, checkpoint deserialization, launcher, publisher or table mutation.
The original FOREVER replay checks and original raw-F1 loop are SHA-bound AST
reuse. Only version provenance and checkpoint retention contracts are adapted.
Default inspection runs metadata/replay validation, never the scoring function.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback
import types

HERE = Path(__file__).resolve()
NATIVE = 'analysis/stage_checkpoint_native_complete_verify_20261027.py'
NATIVE_SHA = '1434c424dcc7d003494da33054bf6ef823c3b49d7e3710805ff360856bf80cc2'
LEGACY = {
    'pct01': ('analysis/forever_multimodal_formal_verify_v3_20261019.py',
              '5f51ab1cbfc871ed1c9492a042235c71d4830517c98f3b34da6e4180bd4bc55a'),
    'pct10': ('analysis/pct10_forever_complete_verify_20261026.py',
              '00e9f9f97e86b53a029e2441db7df1d1b08a3dfa0cf672ea26cba25282721ca0'),
}
FRAMES = {c: 'exp/two_backbone_'+c+'_forever_checkpoint_20261027' for c in LEGACY}
WAITER = 'analysis/forever_checkpoint_queue_20261027.py'
WAITER_SHA = '237f1790dcc36f5b880b5e0643cbb2f016cd2810cfc2edd4c2bc1f2c3664de09'
READY = 'summary/stage_checkpoint_dispatch_20261027/READY.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def module(path, expected):
    assert sha(path) == expected, str(path)
    spec = importlib.util.spec_from_file_location('_private_'+Path(path).stem, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def provenance(root, run, binding, host, files, native):
    condition = binding['condition']; spec = dict(native.SPECS[condition], frame=FRAMES[condition])
    config = files.json(run/'config.json'); frame = root/spec['frame']
    assert binding['frame'] == spec['frame'] and config['condition'] == condition
    assert config['method'] == 'FOREVER' and config['experiment_id'] == frame.name
    assert run.parent == frame/'runs'/config['backbone']/condition/'FOREVER'
    assert config['manifest_sha256'] == spec['manifest_sha'] and config['rolling_checkpoint']
    assert config['checkpoint_all_stages'] and config['checkpoint_retention'] == 'all_completed_stages'
    assert config['save_final_parameters'] and not config['preflight_only']
    assert files.json(run/'run_status.json')['status'] == 'completed'
    sources = binding['sources']; assert config['execution_source_sha256'] == sources
    assert {p.name for p in frame.glob('*.py')} == set(sources)
    for name, digest in sources.items():
        assert Path(name).name == name
        files.checked(frame/name, digest); files.checked(run/'runner_source'/name, digest)
    policy = files.json(frame/'execution_policy.json', binding['policy_sha256'])
    assert policy['conditions'][condition]['manifest_sha256'] == spec['manifest_sha']
    _, build = files.ref(binding['build'])
    if condition == 'pct10':
        assert build['complete'] and build['inputs']['manifest_sha256'] == spec['manifest_sha']
        targets = [r for r in build['runners'] if Path(r['target']).name == frame.name]
        assert len(targets) == 1; target = targets[0]
        assert target['destination_py_sha256'] == sources
        assert files.json(frame/'BUILD_RECEIPT.json') == target
        _, ready = files.ref(binding['ready'])
        assert binding['ready']['path'] == READY
        assert ready['dynamic_sources'] == sources
        assert ready['dynamic_policy_sha256'] == binding['policy_sha256']
        files.checked(frame/'BUILD_RECEIPT.json', ready['build_receipt_sha256'])
        files.checked(root/'exp/stage_checkpoint_dispatch_20261027/dispatch.py', ready['original_dispatch_sha256'])
        queue_path, queue = files.ref(binding['completed_queue_snapshot'])
        job = queue['jobs'][config['backbone']+':FOREVER']
        current_path = root/'summary/two_backbone_pct10_20261025'/('queue_old.json' if host == 'old2080' else 'queue_new.json')
        assert queue_path != current_path
        current = read(current_path)['jobs'][config['backbone']+':FOREVER']
        assert job['status'] in ('completed_unverified', 'completed_verified')
        assert current['status'] in ('completed_unverified', 'completed_verified')
        assert current['run_dir'] == job['run_dir'] == run.relative_to(root).as_posix()
        _, exit = files.ref(binding['terminal_exit'])
        assert exit == job['last_exit'] == current['last_exit']
        assert exit['returncode'] == 0 and exit['phase'] == 'training' and exit['run_dir'] == job['run_dir']
    else:
        assert config['backbone'] == 'qwen' and host == 'new3090'
        assert Path(build['target']).name == frame.name and build['training_settings_unchanged']
        assert {n:h for n,h in build['destination_sha256'].items() if n.endswith('.py')} == sources
        assert build['destination_sha256']['execution_policy.json'] == binding['policy_sha256']
        _, launcher = files.ref(binding['launcher_binding'])
        files.checked(root/WAITER, WAITER_SHA)
        # Reuse the actual waiter's source-matching contract, including the
        # inherited mathematical sources except its explicit I/O-only changes.
        tree = ast.parse((root/WAITER).read_bytes())
        fun = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'source_check')
        ns = dict(ROOT=root, Path=Path, sha=lambda p: str(files.checked(p) and sha(p)), read=read,
                  __file__=str(root/WAITER), HELPERS=('analysis/forever_multimodal_core_20261019.py',
                  'analysis/forever_multimodal_replay_v2_20261019.py'), MANIFEST_SHA=spec['manifest_sha'],
                  IO_CHANGES={'run.py','rolling_io.py','pipeline.py','execution_policy.json'})
        exec(compile(ast.Module(body=[fun], type_ignores=[]), str(root/WAITER), 'exec'), ns)
        assert ns['source_check'](launcher, frame) == sources
        _, gate = files.ref(binding['source_matched_gate'])
        assert gate['passed'] and gate['execution_source_sha256'] == sources
        assert gate['fresh_formal_model_required'] and gate['only_existing_Qwen_preflight_and_reload']
        assert gate['inherited_admission_sha256'] == launcher['inherited_formal_admission']['sha256']
        gate_run = Path(gate['run']); assert gate_run.is_relative_to(frame/'runs') and gate_run != run
        for name, digest in gate['proof_sha256'].items(): files.checked(gate_run/name, digest)
        assert set(gate['proof_sha256']) == {'FOREVER_REPLAY_GPU_GATE.json','gpu_preflight.json','checkpoint_reload_proof.json'}
        pre, reload = read(gate_run/'gpu_preflight.json'), read(gate_run/'checkpoint_reload_proof.json')
        assert pre['passed'] and pre['task_ids'] == list(range(1,20))
        assert reload['passed'] and reload['fresh_process'] and reload['finite_backward']
        for value in (pre, reload):
            assert value['execution_source_sha256'] == sources and value['manifest_sha256'] == spec['manifest_sha']
        _, complete = files.ref(binding['training_complete'])
        assert complete['run'] == str(run) and complete['not_yet_verified_result']
        assert binding['terminal_exits']
        exit_paths, previous_unix = set(), -1
        for index, descriptor in enumerate(binding['terminal_exits']):
            path, exit = files.ref(descriptor)
            assert exit['returncode'] == 0 and exit['run'] == str(run) and type(exit['pid']) is int
            assert path.name == ('training' if index == 0 else 'resume')+'_EXIT_%d.json'%exit['pid']
            assert path not in exit_paths and exit['unix'] > previous_unix
            exit_paths.add(path); previous_unix = exit['unix']
        # No identity_absent field exists in this producer: do not invent one.
    manifest = files.json(root/spec['manifest'], spec['manifest_sha'])
    assert len(manifest['tasks']) == 19 and sum(t['files']['train']['count'] for t in manifest['tasks']) == spec['train']
    assert sum(t['files']['test']['count'] for t in manifest['tasks']) == 21454
    return spec, config, manifest


def retention_and_pauses(run, config, spec, manifest, files):
    latest = files.json(run/'checkpoints/latest.json')
    assert files.json(run/'checkpoints/committed_latest.json') == latest and latest['stage'] == 19
    metadata, retained, missing = {}, [], []
    for stage in range(1,20):
        meta = files.json(run/f'checkpoints/stage_{stage:02d}.json'); metadata[stage] = meta
        assert meta['stage'] == stage and meta['method'] == 'FOREVER' and meta['full_architecture_and_runtime']
        assert meta['manifest_sha256'] == spec['manifest_sha'] and meta['path'] == f'checkpoints/stage_{stage:02d}.pt'
        proof = files.json(run/f'stages/stage_{stage:02d}.json')
        assert proof['complete'] and proof['stage'] == stage and proof['weights_saved']
        commit = files.json(run/f'rolling_checkpoint_commits/stage_{stage:02d}.json')
        assert commit['stage'] == stage and commit['committed_checkpoint'] == meta
        assert commit['all_completed_stage_weights_retained'] and not commit['removed_older_weight_files']
        assert commit['earlier_checkpoint_metadata_retained']
        payload = run/meta['path']
        if payload.exists(): assert payload.stat().st_size == meta['bytes']; retained.append(stage)
        else: missing.append(stage)
    assert metadata[19] == latest and 19 in retained
    history = run/'pause_history'; pauses = {}; previous = -1
    for path in sorted(history.glob('pause_S*.json')):
        value = files.json(path); digest = sha(path); stage = value['completed_stage']
        assert type(stage) is int and previous < stage < 19; previous = stage
        assert path.name.startswith(f'pause_S{stage:02d}_') and value['status'] == 'paused_checkpoint'
        assert value['full_stage_complete'] and value['manifest_sha256'] == spec['manifest_sha']
        assert value['execution_source_sha256'] == config['execution_source_sha256']
        assert value['external_helper_sha256'] == config['external_helper_sha256']
        assert value['config_sha256'] == sha(run/'config.json')
        if stage == 0:
            metadata[0] = files.json(run/'checkpoints/stage_00.json')
            proof = files.json(run/'stages/stage_00.json'); commit = files.json(run/'rolling_checkpoint_commits/stage_00.json')
            assert proof['complete'] and proof['weights_saved'] and proof['stage'] == 0
            assert commit['stage'] == 0 and commit['committed_checkpoint'] == metadata[0]
        assert value['checkpoint'] == metadata[stage]
        assert value['checkpoint_metadata_sha256'] == sha(run/f'checkpoints/stage_{stage:02d}.json')
        request_path = run/value['request_path']; assert request_path.resolve().is_relative_to(history.resolve())
        request = files.json(request_path, value['request_sha256'])
        assert request['request_id'] == value['request_id'] and request['reason'] == value['reason']
        pauses[digest] = dict(stage=stage, receipt=path.relative_to(run).as_posix(), resume=None)
    for path in history.glob('resume_*.json'):
        value = files.json(path); digest = value['pause_receipt_sha256']
        assert path.name == 'resume_'+sha(path)+'.json' and digest in pauses and pauses[digest]['resume'] is None
        pauses[digest]['resume'] = path.relative_to(run).as_posix()
    assert all(x['resume'] for x in pauses.values())
    assert len(list(history.glob('request_*.json'))) == len(pauses)
    branches = list((run/'interrupted_branches').glob('resume_*/RESUME_BRANCH.json'))
    assert len(branches) == len(pauses)
    assert sorted(files.json(p)['committed_stage'] for p in branches) == sorted(x['stage'] for x in pauses.values())
    for path in branches:
        value = files.json(path); assert value['restored_checkpoint_metadata'] == metadata[value['committed_stage']]
        assert value['extra_interrupted_compute_preserved_not_counted_as_new_learning']
        assert value['old_resource_scopes_preserved_and_not_joined_across_processes']
    rows = []
    for stage in range(1,20):
        part = [json.loads(x) for x in files.checked(run/f'logs/stage_{stage:02d}.events.jsonl').read_text().splitlines()]
        n = math.ceil(manifest['tasks'][stage-1]['files']['train']['count']/16)
        assert len(part) == n and [x['step'] for x in part] == list(range(1,n+1))
        assert all(x['event'] == 'train_update' and x['stage'] == stage for x in part); rows.extend(part)
    assert rows == [json.loads(x) for x in files.checked(run/'logs/events.jsonl').read_text().splitlines()]
    assert len(rows) == spec['updates']
    return dict(pauses=list(pauses.values()), pause_count=len(pauses), retained_payloads_at_inspection=retained,
                missing_intermediate_payloads=missing, missing_reason_not_inferred=True,
                final_full_checkpoint_required=True, final_SHA_checked_by_original_score_verifier=True,
                intermediate_process_identity_not_recorded_not_certified=True,
                cross_process_training_memory_peak_not_certified=True)


def extension(root, run, host, binding, spec, base, files):
    relative, expected = LEGACY[binding['condition']]; path = files.checked(root/relative, expected)
    tree = ast.parse(path.read_bytes()); condition = binding['condition']
    if condition == 'pct01':
        old = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify')
        start = next(i for i,n in enumerate(old.body) if isinstance(n, ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'run')
        nodes = old.body[start:]; changed = 0; body = []
        for node in nodes:
            if isinstance(node, ast.For) and ast.unparse(node.iter) == "admission['training_source_sha256'].items()":
                changed += 1; continue  # source_check above binds old math plus new I/O version.
            if isinstance(node, ast.Assert) and "glob('*.pt')" in ast.unparse(node.test) and 'sorted(' in ast.unparse(node.test):
                changed += 1; continue  # Explicit all-stage retention/pause contract above.
            body.append(node)
        assert changed == 2
        fun = ast.parse('def legacy_extension(relative, host):\n pass').body[0]; fun.body = body
        helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'checked']
        ns = dict(ROOT=root, FRAME=root/spec['frame'], REPORT=root/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3',
                  BASE_SHA=base['BASE_COLLECTOR_SHA'], PROFILE_SHA='99a01a7cb53b88deb4e1099e5f7618a5d9bbefcde06b66fdef24e87f4bd82d05',
                  Path=Path, json=json, math=math, hashlib=hashlib, sha=sha, __file__=str(HERE),
                  original=root/spec['base'], own_sha=sha(HERE), result={})
        compiled = ast.fix_missing_locations(ast.Module(body=helpers+[fun],type_ignores=[]))
        exec(compile(compiled,str(path),'exec'),ns)
        def checked(path, hashes, expected=None):
            files.checked(path, expected)
            hashes[Path(path).relative_to(root).as_posix()] = sha(path)
            return path
        ns['checked'] = checked
        value = ns['legacy_extension'](run.relative_to(root).as_posix(),host)
        files.hashes.update(value.pop('extension_source_and_artifact_sha256'))
        value.pop('shared_V2_proof'); return value
    names = {'selector','replay','extension'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    class Adapt(ast.NodeTransformer):
        retention = routes = 0
        def visit_Assert(self, node):
            if "glob('*.pt')" in ast.unparse(node.test) and 'sorted(' in ast.unparse(node.test): self.retention += 1; return None
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value == 'exp/two_backbone_pct10_20261025':
                self.routes += 1; return ast.copy_location(ast.Constant('exp/stage_checkpoint_dispatch_20261027'),node)
            return node
    change=Adapt(); compiled=ast.fix_missing_locations(change.visit(ast.Module(body=nodes,type_ignores=[])))
    assert (change.retention,change.routes) == (1,1)
    import random
    from collections import Counter
    from types import SimpleNamespace
    ns=dict(Path=Path, ROOT=root, ast=ast, hashlib=hashlib, json=json, os=os, math=math,
        random=random, Counter=Counter, SimpleNamespace=SimpleNamespace, sha=sha, read=read,
        FRAME_REL=spec['frame'], READY_REL=READY, READY_SHA=binding['ready']['sha256'],
        LEGACY_SHA='2d9d5be87d81df80eab6ddf0095ce399c9a73d9c8c29909a5022d2364fdeeed8',
        REPLAY_SHA='8024931496a032ea666e44376d79614d7c223b17c73ff11a8074494f87cd3683',
        DAYS=(1,2,4,7,15,30,60,90,120))
    exec(compile(compiled,str(path),'exec'),ns)
    return ns['extension'](root,run,host,types.SimpleNamespace(**base),files)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True); ap.add_argument('--run',required=True)
    ap.add_argument('--host',choices=['old2080','new3090'],required=True)
    ap.add_argument('--binding',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--write',action='store_true'); args=ap.parse_args()
    root=args.root.resolve(); assert args.root.is_absolute() and HERE.is_relative_to(root)
    assert args.binding.is_absolute() and args.binding.is_relative_to(root)
    assert args.output.is_absolute() and args.output.is_relative_to(root) and 'completion_observer' not in str(args.output)
    native=module(root/NATIVE,NATIVE_SHA); native.METHODS=['FOREVER']; files=native.Files(root)
    files.checked(HERE); files.checked(root/NATIVE,NATIVE_SHA)
    binding=files.json(args.binding); run=native.rooted(root,args.run)
    spec,config,manifest=provenance(root,run,binding,args.host,files,native)
    retention=retention_and_pauses(run,config,spec,manifest,files)
    if binding['condition']=='pct01': assert len(binding['terminal_exits']) == retention['pause_count']+1
    calls=[0]; base,adapted=native.inherited(root,args.binding,spec,binding['sources'],calls)
    candidate=base['candidate'](args.run,args.host)
    extra=extension(root,run,args.host,binding,spec,base,files); assert calls[0]==0
    key=hashlib.sha256((args.host+':'+args.run).encode()).hexdigest()[:20]
    folder=args.output/(binding['condition']+'_'+key)
    fingerprint=dict(candidate=candidate,binding_sha256=sha(args.binding),consumer_sha256=sha(HERE),
        metadata_sha256=hashlib.sha256(json.dumps(files.hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    if (folder/'COMPLETE.json').exists():
        complete=read(folder/'COMPLETE.json'); assert complete['fingerprint']==fingerprint
        assert sha(folder/'VERIFIED.json')==complete['proof_sha256']
        print(json.dumps(dict(cached=True,new_scores=0,proof=str(folder/'VERIFIED.json')))); return
    if not args.write:
        print(json.dumps(dict(inspection_only=True,scoring_started=False,intended_cache=str(folder),
            planned_score_calls=80316,fingerprint=fingerprint,pause_retention=retention))); return
    folder.mkdir(parents=True,exist_ok=True)
    # Fixed per-run claim prevents changing --output to repeat scoring. It is
    # separate from scientific RUNs, old observers, proofs and main tables.
    claim=root/'summary/stage_checkpoint_dispatch_20261027/forever_score_claims'/(binding['condition']+'_'+key+'.json')
    claim.parent.mkdir(parents=True,exist_ok=True)
    native.write(claim,dict(consumer_sha256=sha(HERE),cache=str(folder),fingerprint=fingerprint,unix=time.time()))
    native.write(folder/'VERIFY_STARTED.json',dict(fingerprint=fingerprint,unix=time.time(),planned_score_calls=80316,no_retry=True))
    started=time.monotonic()
    try:
        shared=base[spec['function']](args.run,args.host)
        assert calls[0]==80316 and shared['rescored_cells']==74 and shared['final_QA']==21454
        assert base['candidate'](args.run,args.host)==candidate; files.unchanged()
        shared['FOREVER_in_scope']=True
        proof=dict(status='STAGE_CHECKPOINT_FOREVER_COMPLETE_VERIFIED',condition=binding['condition'],
            candidate=candidate,fingerprint=fingerprint,shared_original_proof=shared,FOREVER_in_scope=True,
            forever_extension=extra,pause_and_retention=retention,source_sha256=files.hashes,
            original_forever_verifier_sha256=LEGACY[binding['condition']][1],adapted_base_AST_sha256=adapted,
            scoring_calls=calls[0],current_optimizer_updates=spec['updates'],positive_LR_updates=spec['positive'],
            CPU_only=True,models_started=0,new_generations=0,old_proofs_or_tables_modified=False,
            seconds=time.monotonic()-started)
        native.write(folder/'VERIFIED.json',proof)
        native.write(folder/'COMPLETE.json',dict(fingerprint=fingerprint,proof_sha256=sha(folder/'VERIFIED.json'),
            scoring_calls=calls[0],unix=time.time()))
        print(json.dumps(dict(proof=str(folder/'VERIFIED.json'),scoring_calls=calls[0])))
    except BaseException:
        native.write(folder/'FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),
            scoring_calls_completed=calls[0],no_retry=True)); raise


if __name__=='__main__': main()
