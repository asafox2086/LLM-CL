"""Local metadata-only completion verifier for seven native pct10 methods.

Adapted from v2_verified_baseline_report.py (BASE_COLLECTOR_SHA). The original
candidate/verify/select_results checks are retained below with explicit pct10
configuration. No model import/loading, subprocess, SSH, training or generation.
Default discover is read-only; scoring/cache writes require --write. Consumers
use VERIFIED_INDEX.json, never partially evaluated runs or this program's stdout
to populate scientific tables. This program never writes a Markdown table.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
import traceback

assert os.environ.get('LLMCL_ROOT'), 'LLMCL_ROOT is required'
ROOT = Path(os.environ['LLMCL_ROOT']).resolve()
assert ROOT.is_absolute() and ROOT.is_dir()
FRAME = ROOT/'exp/two_backbone_pct10_native_20261025'
REPORT = ROOT/'summary/two_backbone_pct10_20261025/completion_observer_native'
MANIFEST = ROOT/'data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json'
MANIFEST_SHA = '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
METRIC_SHA = '16c59998413d12b820d6f57269f9e1f45ac78e08595d20bf1f4fbc967fd94c4b'
BASE_COLLECTOR_SHA = 'd7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a'
TRANSPORT_SHA = '9c30b2912af3f29b6be836a0ef7e8f6ab1b1b3e574ca073389e8926c141892f4'
METHODS = ['SeqFT_LoRA','CL_MoE','SMoLoRA','PASs_MoE','PaRSP','OSFT','SpaRTA']
BACKBONES = ['qwen','onevision']
DOMAINS = ['natural','medical','geography','finance','traffic']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


SOURCE_SHA = sha(Path(__file__))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def rooted(relative):
    """Preserve the manifest's ROOT-relative lexical path (assets may symlink)."""
    value = Path(relative)
    assert not value.is_absolute() and '..' not in value.parts
    return ROOT/value


def eligible(status):
    return status.get('status') == 'completed' and status.get('completed_stage') == 19


def candidate(relative, host):
    run = rooted(relative).resolve()
    assert run.is_relative_to(FRAME/'runs')
    status = read(run/'run_status.json')
    assert eligible(status)
    config = read(run/'config.json')
    backbone, method = config['backbone'], config['method']
    assert backbone in BACKBONES and method in METHODS and config['condition'] == 'pct10'
    assert run.parent == FRAME/'runs'/backbone/'pct10'/method
    assert not config['preflight_only'] and config['manifest_sha256'] == MANIFEST_SHA
    # A worker writing completed just before exit is insufficient. Require the
    # originating queue: its coordinator must already have accepted training0.
    # Archive-without-queue admission needs a separate real exit receipt.
    queue_path = ROOT/'summary/two_backbone_pct10_20261025'/('queue_old.json' if host == 'old2080' else 'queue_new.json')
    assert queue_path.is_file(), 'Originating queue with real training exit0 is required'
    job = read(queue_path)['jobs'][backbone+':'+method]
    assert job['status'] in ('completed_unverified','completed_verified')
    assert job['run_dir'] == str(run.relative_to(ROOT))
    assert job['last_exit']['returncode'] == 0 and job['last_exit']['phase'] == 'training'
    assert job['last_exit']['run_dir'] == job['run_dir']
    assert sha(MANIFEST) == MANIFEST_SHA
    manifest = read(MANIFEST)
    latest = read(run/'checkpoints/latest.json')
    checkpoint = (run/latest['path']).resolve()
    assert checkpoint.is_relative_to(run/'checkpoints')
    paths = [run/name for name in ['run_status.json','config.json','metrics_by_score.json','domain_results.json',
             'checkpoints/latest.json','logs/events.jsonl']]+[checkpoint, Path(__file__)]
    paths += [ROOT/'summary/primary_metric_20261017/metric_lock.json', ROOT/'analysis/answer_f1.py', MANIFEST,
              ROOT/'summary/two_backbone_pct10_20261025/OLD_TRANSPORT_RECEIPT.json']
    for name in config['execution_source_sha256']:
        assert Path(name).name == name
        paths += [FRAME/name, run/'runner_source'/name]
    for task in manifest['tasks']:
        paths += [rooted(task['files'][s]['path']) for s in ['train','test']]
    for stage in range(20):
        if stage:
            paths += [run/f'stages/stage_{stage:02d}.json', run/f'validity/stage_{stage:02d}.json']
        for tid in (range(1,20) if stage in (0,19) else [stage,stage+1]):
            p = run/f'evaluations/stage_{stage:02d}/T{tid:02d}_test_predictions.jsonl'
            paths += [p, p.with_suffix('.scores.json')]
    versions = [(str(p.relative_to(ROOT)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    signature = hashlib.sha256(json.dumps(versions, separators=(',',':')).encode()).hexdigest()
    return {'host': host, 'run': str(run.relative_to(ROOT)), 'backbone': backbone, 'method': method,
            'signature': signature, 'collector_sha256': SOURCE_SHA, 'condition':'pct10', 'manifest_sha256':MANIFEST_SHA}


def discover(host):
    results = []
    queue_path = ROOT/'summary/two_backbone_pct10_20261025'/('queue_old.json' if host == 'old2080' else 'queue_new.json')
    if not queue_path.is_file():
        return {'host':host, 'collector_sha256':SOURCE_SHA, 'candidates':[], 'scoring_started':False,
                'reason':'Originating queue absent; real training exit0 is unproven; no run admitted'}
    queue = read(queue_path)['jobs']
    for backbone in BACKBONES:
        for method in METHODS:
            if queue[backbone+':'+method]['status'] not in ('completed_unverified','completed_verified'):
                continue
            for p in sorted((FRAME/'runs'/backbone/'pct10'/method).glob('training_*/run_status.json')):
                if eligible(read(p)):
                    results.append(candidate(str(p.parent.relative_to(ROOT)), host))
    return {'host':host, 'collector_sha256':SOURCE_SHA, 'candidates':results, 'scoring_started':False}


def _verify_uncached(relative, host):
    """Inherited raw-F1 verifier; called only after the exclusive start receipt."""
    before = candidate(relative, host)
    run = ROOT/relative
    source_hashes = {}
    def checked(path, expected=None):
        h = sha(path)
        assert expected is None or h == expected, str(path)
        source_hashes[str(path.relative_to(ROOT))] = h
        return path
    transport = read(checked(ROOT/'summary/two_backbone_pct10_20261025/OLD_TRANSPORT_RECEIPT.json', TRANSPORT_SHA))
    config = read(checked(run/'config.json'))
    assert config['experiment_id'] == FRAME.name and config['seed'] == 42 and config['epochs'] == 1
    assert not config['preflight_only'] and config['primary_score'] == 'token_f1'
    assert config['train_fraction'] == .10 and config['test_fraction'] == 1.0
    assert config['framework'] == 'shared-v2-uniform-input-and-cache' and config['image_area_cap'] == 401408
    frozen_sources = {Path(name).name:h for name,h in transport['files_sha256'].items()
                      if Path(name).parent.as_posix() == FRAME.relative_to(ROOT).as_posix() and name.endswith('.py')}
    assert config['execution_source_sha256'] == frozen_sources and frozen_sources
    assert {p.name for p in FRAME.glob('*.py')} == set(frozen_sources)
    for name, expected in config['execution_source_sha256'].items():
        checked(FRAME/name, expected)
        checked(run/'runner_source'/name, expected)
    lock = read(checked(ROOT/'summary/primary_metric_20261017/metric_lock.json'))
    assert not lock['option_mapping'] and not lock['additional_prefix_or_answer_extraction']
    assert lock['metric_sha256'] == METRIC_SHA and lock['score_field'] == 'token_f1'
    metric = checked(ROOT/'analysis/answer_f1.py', METRIC_SHA)
    spec = importlib.util.spec_from_file_location('locked_pct10_completion_f1', metric)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mp = checked(MANIFEST, MANIFEST_SHA)
    assert config['manifest_sha256'] == MANIFEST_SHA
    manifest = read(mp)
    assert manifest['condition'] == 'pct10' and manifest['seed'] == 42
    assert len(manifest['tasks']) == 19 and [t['task'] for t in manifest['tasks']] == list(range(1,20))
    assert sum(t['files']['train']['count'] for t in manifest['tasks']) == 13105
    assert sum(math.ceil(t['files']['train']['count']/16) for t in manifest['tasks']) == 828
    gold = {}
    for task in manifest['tasks']:
        checked(rooted(task['files']['train']['path']), task['files']['train']['sha256'])
        entry = task['files']['test']
        gold[task['task']] = [json.loads(s) for s in checked(rooted(entry['path']), entry['sha256']).read_text(encoding='utf-8').splitlines()]
        assert len(gold[task['task']]) == entry['count']
    assert sum(map(len,gold.values())) == 21454
    events = [json.loads(s) for s in checked(run/'logs/events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len([e for e in events if e.get('event') == 'train_update']) == 828
    training_schedule = []
    for stage in range(1,20):
        durable = read(checked(run/f'stages/stage_{stage:02d}.json'))
        validity = read(checked(run/f'validity/stage_{stage:02d}.json'))
        assert durable['complete'] and durable['stage'] == stage
        assert validity['passed'] and validity['actual_parameters_changed']
        assert validity['parameters_before_sha256'] != validity['parameters_after_sha256']
        updates = [e for e in events if e.get('event') == 'train_update' and e['stage'] == stage]
        expected = math.ceil(manifest['tasks'][stage-1]['files']['train']['count']/16)
        # Exact original training_validity.warmup_updates formula. HF cosine
        # starts with all-zero LR only when warmup>0; this is a valid call.
        warmup = min(math.floor(.03*expected), expected-1)
        expected_positive = expected-int(warmup > 0)
        assert len(updates) == validity['updates'] == expected
        assert [e['step'] for e in updates] == list(range(1,expected+1))
        assert all(e['warmup_updates'] == warmup for e in updates)
        positive_count = 0
        for e in updates:
            lr_used = e['learning_rate_used']
            assert isinstance(lr_used,list) and lr_used
            assert all(math.isfinite(lr) and lr >= 0 for lr in lr_used)
            has_positive_lr = any(lr > 0 for lr in lr_used)
            if warmup > 0 and e['step'] == 1:
                assert all(lr == 0 for lr in lr_used)
            else:
                assert has_positive_lr
            positive_count += int(has_positive_lr)
        assert positive_count == validity['positive_lr_updates'] == expected_positive
        assert all(math.isfinite(e['loss']) and math.isfinite(e['gradient_l2_preclip']) and
                   e['gradient_l2_preclip'] > 0 for e in updates)
        training_schedule.append({'stage':stage,'optimizer_calls':expected,'warmup_updates':warmup,
                                  'expected_positive_lr_updates':expected_positive,
                                  'expected_zero_lr_updates':int(warmup > 0),
                                  'actual_positive_lr_updates':positive_count})
    assert sum(s['optimizer_calls'] for s in training_schedule) == 828
    assert sum(s['actual_positive_lr_updates'] for s in training_schedule) == 822
    assert sum(s['expected_zero_lr_updates'] for s in training_schedule) == 6
    cells, predictions = {}, {}
    scored_QA = 0
    for stage in range(20):
        for tid in (range(1,20) if stage in (0,19) else [stage,stage+1]):
            p = checked(run/f'evaluations/stage_{stage:02d}/T{tid:02d}_test_predictions.jsonl')
            rows = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
            assert len(rows) == len(gold[tid])
            scores = []
            for row, original in zip(rows,gold[tid]):
                assert row['id'] == original['id'] and row['references'] == original['answers']
                value = module.answer_f1(row['prediction'], original['answers'])
                assert math.isfinite(value) and value == row['scores']['token_f1']
                scores.append(value)
            scored_QA += len(scores)
            value = sum(scores)/len(scores)
            saved = read(checked(p.with_suffix('.scores.json')))
            assert saved['count'] == len(rows) and abs(saved['token_f1']-value) < 1e-12
            cells[stage,tid] = value
            predictions[str(p.relative_to(run))] = source_hashes[str(p.relative_to(ROOT))]
    assert len(cells) == 74 and scored_QA == 80316
    final = [cells[19,t] for t in range(1,20)]
    aggregate = {'AP':sum(final)/19, 'FWT':sum(cells[t-1,t]-cells[0,t] for t in range(2,20))/18,
                 'BWT':sum(cells[19,t]-cells[t,t] for t in range(1,19))/18}
    saved = read(checked(run/'metrics_by_score.json'))['token_f1']
    assert all(abs(value-saved[k]) < 1e-12 for k,value in aggregate.items())
    domains, acquired_domains = {}, {}
    original = read(checked(run/'domain_results.json'))['token_f1']
    assert set(t['domain'] for t in manifest['tasks']) == set(DOMAINS)
    for domain in DOMAINS:
        tt = [t for t in manifest['tasks'] if t['domain'] == domain]
        n = sum(len(gold[t['task']]) for t in tt)
        value = sum(cells[19,t['task']]*len(gold[t['task']]) for t in tt)/n
        assert abs(value-original[domain]['final']) < 1e-12
        domains[domain] = {'F1':value,'QA':n}
        acquired_domains[domain] = {'F1':sum(cells[t['task'],t['task']]*len(gold[t['task']]) for t in tt)/n,'QA':n}
    domain_macro = sum(v['F1'] for v in domains.values())/5
    assert abs(domain_macro-saved['domain_macro_final']) < 1e-12
    acquired = {'AP19':sum(cells[t,t] for t in range(1,20))/19,
                'domain_macro_F1':sum(v['F1'] for v in acquired_domains.values())/5,
                'domains':acquired_domains, 'native19':[cells[t,t] for t in range(1,20)],
                'multiple_task_specific_model_times':True, 'new_model_requests':0, 'additional_scoring':0}
    latest = read(checked(run/'checkpoints/latest.json'))
    assert latest['stage'] == 19 and latest['method'] == config['method']
    checkpoint = (run/latest['path']).resolve()
    assert checkpoint.is_relative_to(run/'checkpoints') and checkpoint.stat().st_size == latest['bytes']
    checked(checkpoint, latest['sha256'])  # Stream only; never unpickle/load a model.
    assert candidate(relative,host) == before, 'Completed artifacts changed during verification'
    assert sha(Path(__file__)) == SOURCE_SHA
    return {'CPU_only':True,'candidate':before,'source_sha256':source_hashes,
            'base_collector_sha256':BASE_COLLECTOR_SHA,'condition':'pct10','manifest_sha256':MANIFEST_SHA,
            'primary_metric':'existing raw Answer F1','rescored_cells':74,'scored_QA':scored_QA,'final_QA':21454,
            'train_QA':13105,'current_optimizer_updates':828,'current_positive_lr_updates':822,
            'current_zero_lr_updates':6,'training_schedule':training_schedule,'positive_finite_training_stages':19,
            'aggregate':aggregate,'domains':domains,'domain_macro_final':domain_macro,'acquired':acquired,
            'verified_cells':[{'stage':s,'task':t,'F1':v,'QA':len(gold[t])} for (s,t),v in cells.items()],
            'final_native19':final,'checkpoint_sha256':latest['sha256'],'checkpoint_bytes':latest['bytes'],
            'config_sha256':source_hashes[str((run/'config.json').relative_to(ROOT))],
            'prediction_sha256':predictions,'no_training_or_prediction_started':True,
            'FOREVER_in_scope':False,'Markdown_tables_written':False}


def cache_paths(c):
    key = hashlib.sha256((c['host']+':'+c['run']).encode()).hexdigest()[:20]
    folder = REPORT/'completed'
    return folder/(key+'.json'), folder/(key+'.VERIFY_STARTED.json'), folder/(key+'.COMPLETE.json')


def verify(relative, host):
    """At most one scoring invocation per host/run; failure never auto-retries."""
    c = candidate(relative,host)
    proof_path, start_path, complete_path = cache_paths(c)
    if proof_path.exists() or complete_path.exists():
        receipt = read(complete_path)
        assert sha(proof_path) == receipt['proof_sha256']
        proof = read(proof_path)
        assert proof['candidate'] == c and receipt['candidate'] == c, 'Cached artifacts changed; no rescore'
        assert receipt['status'] == 'COMPLETE_VERIFIED' and proof['scored_QA'] == 80316
        return proof
    start_path.parent.mkdir(parents=True,exist_ok=True)
    # O_EXCL closes concurrent invocation races; leave this marker on failure.
    with start_path.open('x',encoding='utf-8') as stream:
        json.dump({'candidate':c,'pid':os.getpid(),'unix':time.time(),'planned_scored_QA':80316,
                   'failure_policy':'No automatic retry or repeated scoring'},stream,indent=2)
        stream.write('\n')
    start = time.monotonic()
    try:
        proof = _verify_uncached(relative,host)
        assert proof['candidate'] == c
        write(proof_path,proof)
        write(complete_path,{'status':'COMPLETE_VERIFIED','candidate':c,'unix':time.time(),
                            'proof_sha256':sha(proof_path),'elapsed_seconds':time.monotonic()-start})
        return proof
    except Exception:
        write(start_path.with_suffix('.FAILURE.json'),{'candidate':c,'unix':time.time(),
              'traceback':traceback.format_exc(),'no_automatic_retry':True,'models_started':0})
        raise


def select_results(proofs):
    groups, selected, common = {}, {}, {}
    for proof in proofs:
        c = proof['candidate']
        assert proof['CPU_only'] and proof['rescored_cells'] == 74 and proof['final_QA'] == 21454
        assert proof['condition'] == 'pct10' and proof['manifest_sha256'] == MANIFEST_SHA
        assert proof['scored_QA'] == 80316 and proof['current_optimizer_updates'] == 828
        assert proof['current_positive_lr_updates'] == 822 and proof['current_zero_lr_updates'] == 6
        assert c['method'] in METHODS and c['backbone'] in BACKBONES
        groups.setdefault((c['backbone'],c['method']),[]).append(proof)
    for key, items in groups.items():
        first = items[0]
        for other in items[1:]:
            assert other['candidate']['run'] == first['candidate']['run'], 'Multiple completed runs; never select TEST-best'
            for field in ['config_sha256','checkpoint_sha256','prediction_sha256','aggregate','domains','acquired']:
                assert other[field] == first[field], 'Divergent copies; explicit provenance review required'
        selected[key] = first
        for rel,h in first['source_sha256'].items():
            if ((rel.startswith(str(FRAME.relative_to(ROOT))+'/') and '/runs/' not in rel)
                    or rel.startswith('data/FiveDomainCL/') or rel == 'analysis/answer_f1.py'):
                assert rel not in common or common[rel] == h, 'Shared protocol/source mismatch: '+rel
                common[rel] = h
    return selected


def refresh(host, only_run=None):
    assert sha(Path(__file__)) == SOURCE_SHA, 'No active collector hot edits'
    candidates = [candidate(only_run,host)] if only_run else discover(host)['candidates']
    for c in candidates:
        verify(c['run'],host)
    # A host's index contains exactly its discoverable completed runs; per-run
    # invocation also checks its existing proofs but never scores other runs.
    proofs = []
    for path in sorted((REPORT/'completed').glob('*.COMPLETE.json')):
        receipt = read(path)
        p = path.with_name(path.name.replace('.COMPLETE.json','.json'))
        assert sha(p) == receipt['proof_sha256']
        proof = read(p)
        assert proof['candidate'] == receipt['candidate']
        assert proof['candidate']['collector_sha256'] == SOURCE_SHA
        if proof['candidate']['host'] == host:
            assert candidate(proof['candidate']['run'],host) == proof['candidate'], 'Previously verified run changed/disappeared'
        proofs.append(proof)
    selected = select_results(proofs)
    index = {'CPU_only':True,'collector_sha256':SOURCE_SHA,'base_collector_sha256':BASE_COLLECTOR_SHA,
             'condition':'pct10','manifest_sha256':MANIFEST_SHA,'verified_method_backbone_pairs':len(selected),
             'optimizer_calls_per_complete_run':828,'positive_lr_updates_per_complete_run':822,
             'zero_lr_initial_updates_per_complete_run':6,
             'required_native_pairs':14,'total_planned_pairs_including_FOREVER':16,'methods_in_scope':METHODS,
             'FOREVER_in_scope':False,'TEST_best_run_selection':False,'primary_AP19_unchanged':True,
             'Markdown_tables_written':False,'models_started':0,'missing_results_are_not_zero':True,
             'methods':[{'backbone':b,'method':m,'proof_file':str(cache_paths(p['candidate'])[0].relative_to(ROOT)),
                         'proof_sha256':sha(cache_paths(p['candidate'])[0]),'aggregate':p['aggregate'],
                         'domains':p['domains'],'domain_macro_final':p['domain_macro_final'],'acquired':p['acquired'],
                         'checkpoint_sha256':p['checkpoint_sha256'],'checkpoint_bytes':p['checkpoint_bytes']}
                        for (b,m),p in sorted(selected.items())]}
    write(REPORT/'VERIFIED_INDEX.json',index)
    write(REPORT/'observer_status.json',{'pid':os.getpid(),'unix':time.time(),'host':host,
          'verified_native_pairs':len(selected),'required_native_pairs':14,'models_started':0,
          'Markdown_tables_written':False,'FOREVER_excluded_pending_independent_verification':True})
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',choices=['old2080','new3090'],required=True)
    parser.add_argument('--mode',choices=['discover','verify','once','watch'],default='discover')
    parser.add_argument('--run',help='ROOT-relative completed native training run')
    parser.add_argument('--write',action='store_true',help='Allow one-time original F1 verification and JSON proof/index writes')
    args = parser.parse_args()
    if args.mode == 'discover':
        assert not args.run and not args.write
        print(json.dumps(discover(args.host)))
        return
    assert args.write, 'Verification requires explicit --write; default discovery never scores'
    assert (args.mode == 'verify') == bool(args.run), '--run belongs only to --mode verify'
    REPORT.mkdir(parents=True,exist_ok=True)
    assert not (REPORT/'needs_attention.json').exists(), 'Preserve failure; no automatic verification retry'
    with (REPORT/'observer.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                count = refresh(args.host,args.run)
                print(json.dumps({'verified_native_pairs':count,'required_native_pairs':14,'models_started':0}),flush=True)
                if args.mode != 'watch' or count == 14:
                    break
                time.sleep(60)
        except Exception:
            write(REPORT/'needs_attention.json',{'pid':os.getpid(),'unix':time.time(),
                  'traceback':traceback.format_exc(),'no_automatic_retry':True,'models_started':0})
            raise


if __name__ == '__main__':
    main()
