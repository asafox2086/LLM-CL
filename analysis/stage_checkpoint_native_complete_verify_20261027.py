"""One completed stage-checkpoint native run; separate proof/cache, no service.

The pinned original pct01/pct10 score and schedule functions are compiled in an
isolated namespace. Only provenance substitutions and a transparent score-call
counter are inserted. No model imports/loading, launches, SSH or table writes.
Requires a SHA-bound BUILD, READY/admission and immutable completed-queue snapshot.
Pause provenance certifies archived controls/CP metadata, not unrecorded PIDs.
"""
import argparse
import ast
import functools
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve()
METHODS = ['SeqFT_LoRA', 'CL_MoE', 'SMoLoRA', 'PASs_MoE', 'PaRSP', 'OSFT', 'SpaRTA']
DOMAINS = ['natural', 'medical', 'geography', 'finance', 'traffic']
SPECS = {
    'pct01': dict(frame='exp/two_backbone_native_checkpoint_20261027',
        base='analysis/v2_verified_baseline_report.py',
        base_sha='d7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a',
        function='verify', manifest='data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json',
        manifest_sha='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777', train=1313, updates=93, positive=93),
    'pct10': dict(frame='exp/two_backbone_pct10_native_checkpoint_20261027',
        base='analysis/pct10_verified_baseline_report_20261025.py',
        base_sha='2b2a3202fa049c911324081aacb31d0e7c7508b1020a2727893f1f85ab15c376',
        function='_verify_uncached', manifest='data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json',
        manifest_sha='02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7', train=13105, updates=828, positive=822),
}
METRIC_SHA = '16c59998413d12b820d6f57269f9e1f45ac78e08595d20bf1f4fbc967fd94c4b'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def rooted(root, relative):
    path = Path(relative)
    assert not path.is_absolute() and path.parts and '..' not in path.parts
    return root/path  # Frozen assets may be symlinks outside ROOT.


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


class Files:
    def __init__(self, root): self.root, self.hashes, self.stamps = root, {}, {}

    def checked(self, path, expected=None):
        path = Path(path); value = sha(path)
        assert expected is None or value == expected, str(path)
        key = path.relative_to(self.root).as_posix()
        assert key not in self.hashes or self.hashes[key] == value
        self.hashes[key] = value
        stat = path.stat(); self.stamps[path] = (stat.st_size, stat.st_mtime_ns)
        return path

    def json(self, path, expected=None): return read(self.checked(path, expected))

    def ref(self, value):
        path = rooted(self.root, value['path'])
        return path, self.json(path, value['sha256'])

    def unchanged(self):
        assert all((p.stat().st_size, p.stat().st_mtime_ns) == stamp for p, stamp in self.stamps.items())


def provenance(root, run, binding_path, host, files):
    binding = files.json(binding_path); condition = binding['condition']; spec = SPECS[condition]
    frame = root/spec['frame']; config = files.json(run/'config.json')
    assert run.resolve().is_relative_to((frame/'runs').resolve())
    assert run.parent == frame/'runs'/config['backbone']/condition/config['method']
    assert config['method'] in METHODS and config['backbone'] in ('qwen', 'onevision')
    assert binding['frame'] == spec['frame'] and config['condition'] == condition
    assert config['experiment_id'] == frame.name and config['manifest_sha256'] == spec['manifest_sha']
    assert config['checkpoint_all_stages'] is True and config['checkpoint_retention'] == 'all_completed_stages'
    assert config['save_final_parameters'] is True and not config['preflight_only']
    sources = binding['sources']
    assert config['execution_source_sha256'] == sources
    assert {p.name for p in frame.glob('*.py')} == set(sources)
    for name, expected in sources.items():
        assert Path(name).name == name
        files.checked(frame/name, expected); files.checked(run/'runner_source'/name, expected)
    policy = files.json(frame/'execution_policy.json', binding['policy_sha256'])
    assert policy['conditions'][condition]['manifest_sha256'] == spec['manifest_sha']
    build_path, build = files.ref(binding['build']); ready_path, ready = files.ref(binding['ready'])
    if condition == 'pct10':
        assert build['complete'] and build['inputs']['manifest_sha256'] == spec['manifest_sha']
        targets = [r for r in build['runners'] if Path(r['target']).name == frame.name]
        assert len(targets) == 1
        target = targets[0]
        assert target['destination_py_sha256'] == sources
        assert target['destination_sha256']['execution_policy.json'] == binding['policy_sha256']
        assert files.json(frame/'BUILD_RECEIPT.json') == target
        assert ready['native_sources'] == sources and ready['native_policy_sha256'] == binding['policy_sha256']
        # READY's build_receipt_sha256 is FOREVER-specific. Native provenance
        # comes from the total BUILD plus READY.native_sources/native_policy.
        files.checked(root/'exp/stage_checkpoint_dispatch_20261027/dispatch.py', ready['original_dispatch_sha256'])
    else:
        assert Path(build['target']).name == frame.name and build['training_settings_unchanged'] is True
        assert {k: v for k, v in build['destination_sha256'].items() if k.endswith('.py')} == sources
        assert build['destination_sha256']['execution_policy.json'] == binding['policy_sha256']
        assert build_path == frame/'BUILD_RECEIPT.json'
        assert ready['pct01_runner']['sources'] == sources
        assert ready['pct01_runner']['policy_sha256'] == binding['policy_sha256']
        files.checked(root/'analysis/baseline_pct01_checkpoint_capacity_20261027.py', ready['capacity_sha256'])
    queue_path, queue = files.ref(binding['completed_queue_snapshot'])
    if condition == 'pct10':
        live_queue_path = root/'summary/two_backbone_pct10_20261025'/('queue_old.json' if host == 'old2080' else 'queue_new.json')
        job = queue['jobs'][config['backbone']+':'+config['method']]
        assert job['status'] in ('completed_unverified', 'completed_verified')
        assert job['last_exit']['returncode'] == 0 and job['last_exit']['phase'] == 'training'
        assert job['last_exit']['run_dir'] == job['run_dir']
        exit_path, exit = files.ref(binding['terminal_exit'])
        assert exit == job['last_exit']
        current = read(live_queue_path)['jobs'][config['backbone']+':'+config['method']]
        assert current['status'] in ('completed_unverified', 'completed_verified')
        assert current['run_dir'] == job['run_dir'] and current['last_exit'] == job['last_exit']
    else:
        live_queue_path = root/'summary/two_backbone_native_cl_20261017_v2'/(config['host']+'_run_status.json')
        assert queue['host'] == config['host']
        job = queue['jobs']['pct01:'+config['method']]
        assert job['status'] == 'completed' and job['completed_stage'] == 19
        assert job.get('capacity_last_exit_code') == 0, 'Final actual wait status absent; None is not exit0'
        assert queue['capacity_wrapper'] == 'analysis/baseline_pct01_checkpoint_capacity_20261027.py'
        current = read(live_queue_path)['jobs']['pct01:'+config['method']]
        assert current['status'] == 'completed' and current['run_dir'] == job['run_dir']
        assert current.get('capacity_last_exit_code') == 0
    assert queue_path != live_queue_path, 'Bind a stable snapshot, not the allocator-owned changing queue'
    assert job['run_dir'] == run.relative_to(root).as_posix()
    assert job['method'] == config['method'] and job['condition'] == condition
    expected_host = 'old2080' if config['host'].startswith('old') else 'new3090'
    assert host == expected_host
    manifest = files.json(root/spec['manifest'], spec['manifest_sha'])
    assert len(manifest['tasks']) == 19 and sum(t['files']['train']['count'] for t in manifest['tasks']) == spec['train']
    assert sum(t['files']['test']['count'] for t in manifest['tasks']) == 21454
    assert sum(math.ceil(t['files']['train']['count']/16) for t in manifest['tasks']) == spec['updates']
    return binding, config, spec, manifest


def checkpoints_and_pauses(run, config, spec, files):
    latest = files.json(run/'checkpoints/latest.json')
    assert latest['stage'] == 19 and latest['full_architecture_and_runtime']
    metadata, retained, missing = {}, [], []
    for stage in range(1, 20):
        meta = files.json(run/'checkpoints'/('stage_%02d.json'%stage)); metadata[stage] = meta
        assert meta['stage'] == stage and meta['method'] == config['method']
        assert meta['manifest_sha256'] == spec['manifest_sha'] and meta['full_architecture_and_runtime']
        assert meta['path'] == 'checkpoints/stage_%02d.pt'%stage
        proof = files.json(run/'stages'/('stage_%02d.json'%stage))
        assert proof['complete'] and proof['stage'] == stage and proof['weights_saved']
        weight = run/meta['path']
        if weight.exists():
            assert weight.stat().st_size == meta['bytes']; retained.append(stage)
        else: missing.append(stage)
    assert metadata[19] == latest and 19 in retained, 'Final complete checkpoint must exist'
    history = run/'pause_history'; pauses = {}; previous = -1
    for path in sorted(history.glob('pause_S*.json')):
        receipt = files.json(path); receipt_sha = files.hashes[path.relative_to(files.root).as_posix()]
        stage = receipt['completed_stage']; assert type(stage) is int and previous < stage < 19
        previous = stage
        assert path.name.startswith('pause_S%02d_'%stage) and receipt['full_stage_complete']
        assert receipt['manifest_sha256'] == spec['manifest_sha'] and receipt['execution_source_sha256'] == config['execution_source_sha256']
        assert receipt['config_sha256'] == sha(run/'config.json') and receipt['next_stage'] == stage+1
        if stage == 0: metadata[0] = files.json(run/'checkpoints/stage_00.json')
        meta = metadata[stage]; assert receipt['checkpoint'] == meta
        assert receipt['checkpoint_metadata_sha256'] == sha(run/'checkpoints'/('stage_%02d.json'%stage))
        request_path = run/receipt['request_path']
        assert request_path.resolve().is_relative_to(history.resolve())
        request = files.json(request_path, receipt['request_sha256'])
        assert request['request_id'] == receipt['request_id'] and isinstance(request['reason'], str) and request['reason'].strip()
        pauses[receipt_sha] = dict(stage=stage, pause_receipt=path.relative_to(run).as_posix(), resume=None)
    for path in history.glob('resume_*.json'):
        request = files.json(path); digest = request['pause_receipt_sha256']
        assert path.name == 'resume_'+sha(path)+'.json'
        assert digest in pauses and pauses[digest]['resume'] is None
        pauses[digest]['resume'] = path.relative_to(run).as_posix()
    assert all(p['resume'] is not None for p in pauses.values()), 'Completed run contains an unresumed pause'
    assert len(list(history.glob('request_*.json'))) == len(pauses)
    events = [json.loads(s) for s in files.checked(run/'logs/events.jsonl').read_text(encoding='utf8').splitlines()]
    updates = [x for x in events if x.get('event') == 'train_update']
    assert len(updates) == spec['updates']
    for stage in range(1, 20):
        records = [x for x in updates if x['stage'] == stage]
        assert [x['step'] for x in records] == list(range(1, len(records)+1))
    return dict(pauses=list(pauses.values()), pause_count=len(pauses),
        certified='Source/config/CP metadata and archived explicit control chain; no unrecorded intermediate process identities inferred',
        all_stage_checkpoint_metadata_retained=True, retained_intermediate_payloads_at_verification=retained,
        intermediate_payloads_absent=missing, missing_intermediate_reason_not_inferred=True,
        intermediate_payloads_rehashed=0, final_payload_checked_by_original_verifier=True,
        native_cross_process_memory_growth_not_certified=bool(pauses))


def inherited(root, binding_path, spec, sources, calls):
    path = root/spec['base']; assert sha(path) == spec['base_sha']
    tree = ast.parse(path.read_bytes())
    names = {'sha', 'eligible', 'candidate', spec['function']}
    if spec['function'] == '_verify_uncached': names.update(('read', 'rooted'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    class Adapt(ast.NodeTransformer):
        paths = maps = counters = 0
        def visit_Constant(self, node):
            if node.value == 'summary/two_backbone_pct10_20261025/OLD_TRANSPORT_RECEIPT.json':
                self.paths += 1
                return ast.copy_location(ast.Constant(binding_path.relative_to(root).as_posix()), node)
            return node
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'frozen_sources':
                self.maps += 1; assert isinstance(node.value, ast.DictComp)
                node.value = ast.Name(id='BOUND_SOURCES', ctx=ast.Load())
            return self.generic_visit(node)
        def visit_Expr(self, node):
            signature = ast.dump(ast.parse('spec.loader.exec_module(module)').body[0], include_attributes=False)
            if ast.dump(node, include_attributes=False) == signature:
                self.counters += 1
                return [node, ast.parse('module.answer_f1 = count_score(module.answer_f1)').body[0]]
            return self.generic_visit(node)
    adapt = Adapt(); body = ast.fix_missing_locations(adapt.visit(ast.Module(body=nodes, type_ignores=[])))
    assert (adapt.paths, adapt.maps, adapt.counters) == ((2, 1, 1) if spec['function'] == '_verify_uncached' else (0, 0, 1))
    def count_score(function):
        @functools.wraps(function)
        def counted(*args, **kwargs):
            calls[0] += 1
            return function(*args, **kwargs)
        return counted
    namespace = dict(ROOT=root, FRAME=root/spec['frame'], SOURCE_SHA=sha(HERE), __file__=str(HERE),
        Path=Path, json=json, hashlib=hashlib, math=math, importlib=importlib,
        METHODS=METHODS, BACKBONES=['qwen', 'onevision'], DOMAINS=DOMAINS,
        MANIFEST=root/spec['manifest'], MANIFEST_SHA=spec['manifest_sha'], METRIC_SHA=METRIC_SHA,
        BASE_COLLECTOR_SHA=SPECS['pct01']['base_sha'], TRANSPORT_SHA=sha(binding_path),
        BOUND_SOURCES=sources, count_score=count_score)
    exec(compile(body, str(path), 'exec'), namespace)
    return namespace, hashlib.sha256(ast.dump(body, include_attributes=False).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--run', required=True, help='ROOT-relative completed native training RUN')
    ap.add_argument('--host', choices=['old2080', 'new3090'], required=True)
    ap.add_argument('--binding', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='Independent new cache parent; never old observer folders')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args(); root = args.root.resolve(); run = rooted(root, args.run)
    assert args.root.is_absolute() and root.is_dir() and HERE.is_relative_to(root)
    assert args.binding.is_absolute() and args.binding.is_relative_to(root)
    assert args.output.is_absolute() and args.output.is_relative_to(root)
    assert 'completion_observer' not in str(args.output) and 'completion_observer_native' not in str(args.output)
    assert read(run/'run_status.json')['status'] == 'completed' and read(run/'run_status.json')['completed_stage'] == 19
    files = Files(root); files.checked(HERE)
    binding, config, spec, manifest = provenance(root, run, args.binding, args.host, files)
    files.checked(root/spec['base'], spec['base_sha'])
    files.checked(root/'analysis/answer_f1.py', METRIC_SHA)
    pause_proof = checkpoints_and_pauses(run, config, spec, files)
    calls = [0]; base, adapted_ast = inherited(root, args.binding, spec, binding['sources'], calls)
    candidate = base['candidate'](args.run, args.host)
    assert calls[0] == 0
    key = hashlib.sha256((args.host+':'+args.run).encode()).hexdigest()[:20]
    folder = args.output/(binding['condition']+'_'+key)
    fingerprint = dict(candidate=candidate, binding_sha256=sha(args.binding), consumer_sha256=sha(HERE),
        provenance_metadata_sha256=hashlib.sha256(json.dumps(files.hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
    if (folder/'COMPLETE.json').exists():
        complete = read(folder/'COMPLETE.json'); assert complete['fingerprint'] == fingerprint
        assert sha(folder/'VERIFIED.json') == complete['proof_sha256']
        print(json.dumps(dict(cached=True, proof=str(folder/'VERIFIED.json'), new_scores=0)))
        return
    if not args.write:
        print(json.dumps(dict(inspection_only=True, scoring_started=False, fingerprint=fingerprint,
            intended_cache=str(folder), planned_score_calls=80316, pause_provenance=pause_proof)))
        return
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'VERIFY_STARTED.json').exists(), 'Incomplete/failed verification is never automatically retried'
    write(folder/'VERIFY_STARTED.json', dict(unix=time.time(), fingerprint=fingerprint, planned_score_calls=80316,
        no_automatic_retry=True, preserved_old_proofs=True))  # O_EXCL, before any metric call.
    began = time.monotonic()
    try:
        proof = base[spec['function']](args.run, args.host)
        assert calls[0] == 80316 and proof['rescored_cells'] == 74 and proof['final_QA'] == 21454
        assert proof['candidate'] == candidate
        assert base['candidate'](args.run, args.host) == candidate
        files.unchanged()
        verified = dict(status='STAGE_CHECKPOINT_NATIVE_COMPLETE_VERIFIED', fingerprint=fingerprint,
            condition=binding['condition'], shared_original_proof=proof, base_source_sha256=spec['base_sha'],
            adapted_function_AST_sha256=adapted_ast, provenance_source_sha256=files.hashes,
            source_adaptations='FRAME/ROOT; pct10 transport+source map; transparent call counter only',
            pause_and_retention=pause_proof, scoring_calls=calls[0], train_QA=spec['train'],
            current_optimizer_updates=spec['updates'], positive_LR_updates=spec['positive'],
            models_started=0, new_generations=0, old_proofs_or_tables_modified=False,
            seconds=time.monotonic()-began)
        write(folder/'VERIFIED.json', verified)
        write(folder/'COMPLETE.json', dict(unix=time.time(), fingerprint=fingerprint,
            proof_sha256=sha(folder/'VERIFIED.json'), status='COMPLETE_VERIFIED', scoring_calls=calls[0]))
        print(json.dumps(dict(proof=str(folder/'VERIFIED.json'), proof_sha256=sha(folder/'VERIFIED.json'), scoring_calls=calls[0])))
    except BaseException:
        write(folder/'FAILURE.json', dict(unix=time.time(), traceback=traceback.format_exc(),
            scoring_calls_completed=calls[0], no_automatic_retry=True))
        raise


if __name__ == '__main__': main()
