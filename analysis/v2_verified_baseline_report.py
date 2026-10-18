"""CPU-only completion verifier/publisher for all ten V2 methods, two backbones.

Never launches training, loads a checkpoint, selects a best run, or imputes
missing cells. Old-host RPC transfers JSON evidence only, never model/data.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT = ROOT/'summary/two_backbone_pct01_20261017/completion_observer_20261018'
TABLE = ROOT/'summary/two_backbone_pct01_20261017/MAIN_TABLE.md'
METHODS = ['SeqFT_LoRA','ER_LoRA','CL_MoE','SMoLoRA','PASs_MoE','PaRSP','OSFT','SpaRTA','FV_guided_Training','ALER']
LABELS = dict(zip(METHODS, ['SeqFT-LoRA','ER-LoRA','CL-MoE','SMoLoRA','PASs-MoE','PaRSP','OSFT','SpaRTA','FV-guided Training','ALER']))
BACKBONES = ['qwen','onevision']
MARKERS = {'Qwen2-VL-2B-Instruct':'qwen', 'LLaVA-OneVision-Qwen2-0.5B-OV':'onevision'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


SOURCE_SHA = sha(Path(__file__))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def eligible(status):
    return status.get('status') == 'completed' and status.get('completed_stage') == 19


def candidate(relative, host):
    run = (ROOT/relative).resolve()
    assert run.is_relative_to(FRAME/'runs')
    status = json.loads((run/'run_status.json').read_text())
    assert eligible(status)
    config = json.loads((run/'config.json').read_text())
    backbone, method = config['backbone'], config['method']
    assert backbone in BACKBONES and method in METHODS and config['condition'] == 'pct01'
    assert run.parent == FRAME/'runs'/backbone/'pct01'/method
    manifest = json.loads((ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json').read_text())
    latest = json.loads((run/'checkpoints/latest.json').read_text())
    checkpoint = (run/latest['path']).resolve()
    assert checkpoint.is_relative_to(run/'checkpoints')
    paths = [run/name for name in ['run_status.json','config.json','metrics_by_score.json','domain_results.json',
             'checkpoints/latest.json','logs/events.jsonl']]+[checkpoint, Path(__file__)]
    paths += [ROOT/'summary/primary_metric_20261017/metric_lock.json', ROOT/'analysis/answer_f1.py',
              ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json']
    for name in config['execution_source_sha256']:
        assert Path(name).name == name
        paths += [FRAME/name, run/'runner_source'/name]
    for task in manifest['tasks']:
        paths += [ROOT/task['files'][s]['path'] for s in ['train','test']]
    for stage in range(20):
        if stage:
            paths += [run/f'stages/stage_{stage:02d}.json', run/f'validity/stage_{stage:02d}.json']
        for tid in (range(1,20) if stage in (0,19) else [stage,stage+1]):
            p = run/f'evaluations/stage_{stage:02d}/T{tid:02d}_test_predictions.jsonl'
            paths += [p, p.with_suffix('.scores.json')]
    versions = [(str(p.relative_to(ROOT)), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    signature = hashlib.sha256(json.dumps(versions, separators=(',',':')).encode()).hexdigest()
    return {'host': host, 'run': str(run.relative_to(ROOT)), 'backbone': backbone, 'method': method,
            'signature': signature, 'collector_sha256': SOURCE_SHA}


def discover(host):
    results = []
    for backbone in BACKBONES:
        for method in METHODS:
            for p in sorted((FRAME/'runs'/backbone/'pct01'/method).glob('training_*/run_status.json')):
                if eligible(json.loads(p.read_text())):
                    results.append(candidate(str(p.parent.relative_to(ROOT)), host))
    return {'host': host, 'collector_sha256': SOURCE_SHA, 'candidates': results}


def verify(relative, host):
    before = candidate(relative, host)
    run = ROOT/relative
    source_hashes = {}
    def checked(path, expected=None):
        h = sha(path)
        assert expected is None or h == expected, str(path)
        source_hashes[str(path.relative_to(ROOT))] = h
        return path
    config = json.loads(checked(run/'config.json').read_text())
    assert config['experiment_id'] == FRAME.name and config['seed'] == 42 and config['epochs'] == 1
    assert config['framework'] == 'shared-v2-uniform-input-and-cache' and config['image_area_cap'] == 401408
    for name, expected in config['execution_source_sha256'].items():
        checked(FRAME/name, expected)
        checked(run/'runner_source'/name, expected)
    lock = json.loads(checked(ROOT/'summary/primary_metric_20261017/metric_lock.json').read_text())
    assert not lock['option_mapping'] and not lock['additional_prefix_or_answer_extraction']
    metric = checked(ROOT/'analysis/answer_f1.py', lock['metric_sha256'])
    spec = importlib.util.spec_from_file_location('locked_v2_completion_f1', metric)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mp = checked(ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json', config['manifest_sha256'])
    manifest = json.loads(mp.read_text())
    assert len(manifest['tasks']) == 19 and sum(t['files']['train']['count'] for t in manifest['tasks']) == 1313
    gold = {}
    for task in manifest['tasks']:
        checked(ROOT/task['files']['train']['path'], task['files']['train']['sha256'])
        entry = task['files']['test']
        gold[task['task']] = [json.loads(s) for s in checked(ROOT/entry['path'], entry['sha256']).read_text().splitlines()]
        assert len(gold[task['task']]) == entry['count']
    assert sum(map(len,gold.values())) == 21454
    events = [json.loads(s) for s in checked(run/'logs/events.jsonl').read_text().splitlines()]
    for stage in range(1,20):
        durable = json.loads(checked(run/f'stages/stage_{stage:02d}.json').read_text())
        validity = json.loads(checked(run/f'validity/stage_{stage:02d}.json').read_text())
        assert durable['complete'] and durable['stage'] == stage
        assert validity['passed'] and validity['actual_parameters_changed']
        assert validity['parameters_before_sha256'] != validity['parameters_after_sha256']
        updates = [e for e in events if e.get('event') == 'train_update' and e['stage'] == stage]
        expected = math.ceil(manifest['tasks'][stage-1]['files']['train']['count']/16)
        assert len(updates) == validity['updates'] == validity['positive_lr_updates'] == expected
        assert all(math.isfinite(e['loss']) and math.isfinite(e['gradient_l2_preclip']) and
                   e['gradient_l2_preclip'] > 0 and any(lr > 0 for lr in e['learning_rate_used']) for e in updates)
    cells, predictions = {}, {}
    for stage in range(20):
        for tid in (range(1,20) if stage in (0,19) else [stage,stage+1]):
            p = checked(run/f'evaluations/stage_{stage:02d}/T{tid:02d}_test_predictions.jsonl')
            rows = [json.loads(s) for s in p.read_text().splitlines()]
            assert len(rows) == len(gold[tid])
            scores = []
            for row, original in zip(rows,gold[tid]):
                assert row['id'] == original['id'] and row['references'] == original['answers']
                value = module.answer_f1(row['prediction'], original['answers'])
                assert math.isfinite(value) and value == row['scores']['token_f1']
                scores.append(value)
            value = sum(scores)/len(scores)
            saved = json.loads(checked(p.with_suffix('.scores.json')).read_text())
            assert saved['count'] == len(rows) and abs(saved['token_f1']-value) < 1e-12
            cells[stage,tid] = value
            predictions[str(p.relative_to(run))] = source_hashes[str(p.relative_to(ROOT))]
    assert len(cells) == 74
    final = [cells[19,t] for t in range(1,20)]
    aggregate = {'AP':sum(final)/19, 'FWT':sum(cells[t-1,t]-cells[0,t] for t in range(2,20))/18,
                 'BWT':sum(cells[19,t]-cells[t,t] for t in range(1,19))/18}
    saved = json.loads(checked(run/'metrics_by_score.json').read_text())['token_f1']
    assert all(abs(value-saved[k]) < 1e-12 for k,value in aggregate.items())
    domains = {}
    original = json.loads(checked(run/'domain_results.json').read_text())['token_f1']
    for domain in dict.fromkeys(t['domain'] for t in manifest['tasks']):
        tt = [t for t in manifest['tasks'] if t['domain'] == domain]
        n = sum(len(gold[t['task']]) for t in tt)
        value = sum(cells[19,t['task']]*len(gold[t['task']]) for t in tt)/n
        assert abs(value-original[domain]['final']) < 1e-12
        domains[domain] = {'F1':value,'QA':n}
    domain_macro = sum(v['F1'] for v in domains.values())/5
    assert abs(domain_macro-saved['domain_macro_final']) < 1e-12
    latest = json.loads(checked(run/'checkpoints/latest.json').read_text())
    assert latest['stage'] == 19 and latest['method'] == config['method']
    checkpoint = (run/latest['path']).resolve()
    assert checkpoint.is_relative_to(run/'checkpoints') and checkpoint.stat().st_size == latest['bytes']
    checked(checkpoint, latest['sha256'])  # Stream the trusted artifact; never unpickle/load it.
    assert candidate(relative,host) == before, 'Completed artifacts changed during verification'
    assert sha(Path(__file__)) == SOURCE_SHA
    return {'CPU_only':True,'candidate':before,'source_sha256':source_hashes,
            'primary_metric':'existing raw Answer F1','rescored_cells':74,'final_QA':21454,
            'positive_finite_training_stages':19,'aggregate':aggregate,'domains':domains,
            'domain_macro_final':domain_macro,'final_native19':final,'checkpoint_sha256':latest['sha256'],
            'config_sha256':source_hashes[str((run/'config.json').relative_to(ROOT))],
            'prediction_sha256':predictions,'no_training_or_prediction_started':True}


def select_results(proofs):
    groups = {}
    for proof in proofs:
        c = proof['candidate']
        assert proof['CPU_only'] and proof['rescored_cells'] == 74 and proof['final_QA'] == 21454
        groups.setdefault((c['backbone'],c['method']),[]).append(proof)
    selected = {}
    common = {}
    for key, items in groups.items():
        first = items[0]
        for other in items[1:]:
            # Only identical copies of the same run may be deduplicated.
            assert other['candidate']['run'] == first['candidate']['run'], 'Multiple completed runs; explicit provenance review required'
            for field in ['config_sha256','checkpoint_sha256','prediction_sha256','aggregate','domains']:
                assert other[field] == first[field], 'Divergent copies; never select TEST-best run'
        selected[key] = first
        for rel,h in first['source_sha256'].items():
            if ((rel.startswith(str(FRAME.relative_to(ROOT))+'/') and '/runs/' not in rel)
                    or rel.startswith('data/FiveDomainCL/') or rel == 'analysis/answer_f1.py'):
                assert rel not in common or common[rel] == h, 'Shared cross-host protocol/source mismatch: '+rel
                common[rel] = h
    return selected


def rendered_table(text, selected):
    output, active, found = [], None, set()
    by_label = {label:method for method,label in LABELS.items()}
    for line in text.splitlines():
        if line.startswith('| **'):
            active = next((b for marker,b in MARKERS.items() if marker in line), None)
        elif line.startswith('|') and active:
            label = line.split('|')[1].strip()
            if label in by_label:
                key = (active,by_label[label])
                found.add(key)
                if key in selected:
                    p = selected[key]
                    values = [p['domains'][d]['F1'] for d in ['natural','medical','geography','finance','traffic']]
                    values += [p['aggregate'][m] for m in ['AP','FWT','BWT']]
                    expected = [f'{v:.3f}' for v in values]
                    old = [s.strip() for s in line.split('|')[2:-1]]
                    assert len(old) == 8 and all(a in ('','—') or a == b for a,b in zip(old,expected)), 'Existing published row differs; review before overwriting'
                    line = '| '+label+' | '+' | '.join(expected)+' |'
        output.append(line)
    assert found == {(b,m) for b in BACKBONES for m in METHODS}, 'Do not overwrite an unrecognized table layout'
    assert set(selected).issubset(found)
    return '\n'.join(output)+'\n'


def rpc(host, mode, relative=None):
    if host == 'new3090':
        return discover(host) if mode == 'discover' else verify(relative,host)
    remote_root = '/data2/liyapeng_grp/program/LLMCL'
    args = [remote_root+'/.vision-env/bin/python', remote_root+'/analysis/v2_verified_baseline_report.py',
            '--host',host,'--mode',mode]
    if relative is not None:
        args += ['--run',relative]
    command = 'CUDA_VISIBLE_DEVICES="" '+shlex.join(args)
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
              '-i',str(ROOT/'.deployment/old_pct01_id_ed25519'),'-p','8666','liyapeng_grp@10.254.29.8',command],
              text=True,capture_output=True,timeout=180 if mode == 'verify' else 20)
    if result.returncode:
        if result.returncode == 255:
            raise ConnectionError(result.stderr[-1000:])
        raise RuntimeError('Remote verification failed, preserve artifacts: '+result.stderr[-3000:])
    return json.loads(result.stdout)


def refresh():
    assert sha(Path(__file__)) == SOURCE_SHA, 'No active reporter hot edits'
    candidates = []
    for host in ['new3090','old2080']:
        response = rpc(host,'discover')
        assert response['collector_sha256'] == SOURCE_SHA
        candidates += response['candidates']
    folder = REPORT/'completed'
    folder.mkdir(parents=True,exist_ok=True)
    changed = False
    for c in candidates:
        name = hashlib.sha256((c['host']+':'+c['run']).encode()).hexdigest()[:20]+'.json'
        path = folder/name
        if path.exists():
            cached = json.loads(path.read_text())
            assert cached['candidate'] == c, 'Previously completed artifacts changed; no silent re-audit'
            continue
        proof = rpc(c['host'],'verify',c['run'])
        assert proof['candidate'] == c
        write(path,proof)
        changed = True
    proofs = [json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))]
    assert {(p['candidate']['host'],p['candidate']['run']) for p in proofs} == {(c['host'],c['run']) for c in candidates}, 'Completed run disappeared; review provenance'
    selected = select_results(proofs)
    if changed or not (REPORT/'VERIFIED_INDEX.json').exists():
        before = TABLE.read_bytes()
        rendered = rendered_table(before.decode(),selected)
        index = {'CPU_only':True,'collector_sha256':SOURCE_SHA,'verified_method_backbone_pairs':len(selected),
                 'required_pairs':20,'TEST_best_run_selection':False,'primary_AP19_unchanged':True,
                 'methods':[{'backbone':b,'method':m,'proof_file':str((folder/(hashlib.sha256((p['candidate']['host']+':'+p['candidate']['run']).encode()).hexdigest()[:20]+'.json')).relative_to(ROOT)),
                             'aggregate':p['aggregate'],'domains':p['domains']} for (b,m),p in sorted(selected.items())]}
        write(REPORT/'VERIFIED_INDEX.json',index)
        assert TABLE.read_bytes() == before, 'Concurrent table edit; do not overwrite'
        if rendered.encode() != before:
            tmp = TABLE.with_suffix('.md.observer.tmp')
            tmp.write_text(rendered)
            tmp.replace(TABLE)
    write(REPORT/'observer_status.json',{'pid':os.getpid(),'CPU_only':True,'verified_pairs':len(selected),
          'required_pairs':20,'new_pairs_verified_this_poll':changed,'last_checked_unix':time.time(),
          'no_GPU_or_training_started':True})
    return len(selected)


def checks():
    tests = [not eligible({}),not eligible({'status':'evaluation','completed_stage':19}),
             not eligible({'status':'completed','completed_stage':18}),
             eligible({'status':'completed','completed_stage':19}),rendered_table(TABLE.read_text(),{}) == TABLE.read_text()]
    example = {'CPU_only':True,'rescored_cells':74,'final_QA':21454,
               'candidate':{'backbone':'onevision','method':'SeqFT_LoRA','run':'same'},
               'config_sha256':'x','checkpoint_sha256':'y','prediction_sha256':{},
               'aggregate':{'AP':1},'domains':{},'source_sha256':{}}
    tests.append(len(select_results([example,json.loads(json.dumps(example))])) == 1)
    for field,value in [('run','different'),('aggregate',{'AP':99})]:
        other = json.loads(json.dumps(example))
        if field == 'run':
            other['candidate']['run'] = value
        else:
            other[field] = value
        try:
            select_results([example,other])
        except AssertionError:
            tests.append(True)
        else:
            tests.append(False)
    assert all(tests)
    return len(tests)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',choices=['new3090','old2080'],default='new3090')
    parser.add_argument('--mode',choices=['discover','verify','check','once','watch'],required=True)
    parser.add_argument('--run')
    args = parser.parse_args()
    if args.mode == 'discover':
        print(json.dumps(discover(args.host)))
    elif args.mode == 'verify':
        assert args.run
        print(json.dumps(verify(args.run,args.host)))
    elif args.mode == 'check':
        print(json.dumps({'CPU_only':True,'checks':checks(),'GPU_started':False}))
    else:
        REPORT.mkdir(parents=True,exist_ok=True)
        assert not (REPORT/'needs_attention.json').exists(), 'Review prior failure; never blindly restart'
        handle = (REPORT/'observer.lock').open('a')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                try:
                    count = refresh()
                except (ConnectionError,subprocess.TimeoutExpired) as error:
                    # An observation timeout is not a stopped training job.
                    write(REPORT/'transport_wait.json',{'error':str(error),'no_GPU_restart':True,'time':time.time()})
                    if args.mode == 'once':
                        raise SystemExit(2)
                else:
                    if args.mode == 'once' or count == 20:
                        break
                time.sleep(60)
        except Exception:
            write(REPORT/'needs_attention.json',{'pid':os.getpid(),'traceback':traceback.format_exc(),
                                                'no_training_restarted':True})
            raise
