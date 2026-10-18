"""Read-only outcome of the sealed 19-stage initialization treatment.

No model forward, optimizer, new predictions, or TEST-dependent choices.
All five domains and all 19 diagnostic task effects are retained.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from answer_f1 import answer_f1
from onevision_acquisition_budget_bridge import interval

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'summary/onevision_diagnostics_20261016'
REPORT = BASE/'warm_start_20261018/first19_outcome'
DOMAINS = ['natural', 'medical', 'geography', 'finance', 'traffic']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(1)
    assert not (REPORT/'EVIDENCE.json').exists(), 'Do not repeat completed audits as progress'
    inputs = {}

    def read(p):
        inputs[str(p.relative_to(ROOT))] = sha(p)
        return json.loads(p.read_text())

    verified = read(BASE/'warm_start_20261018/VERIFIED_EVIDENCE.json')
    assert verified['warm_train_complete'] and verified['raw_Answer_F1'] and not verified['formal_V2']
    metric = read(ROOT/'summary/primary_metric_20261017/metric_lock.json')
    assert sha(ROOT/'analysis/answer_f1.py') == metric['metric_sha256']
    variants = ['learned_cosine_K3', 'warm_inherit_K3_v1']
    proofs, runs, records, pools, windows, training = {}, {}, {}, {}, {}, {}
    for variant, folder in zip(variants, ['learned_reuse', 'warm_start_20261018']):
        p = BASE/folder/(variant+'_train_complete.json')
        proof = read(p)
        assert proof['complete'] and proof['train_qa'] == 1313 and proof['stages'] == 19
        assert proof['experts'] == 3 and proof['diagnostic_generated'] == 155 and proof['diagnostic_qa'] == 500
        assert verified['completion_sha256'][str(p.relative_to(ROOT))] == sha(p)
        for rel, expected in proof['spec']['source_sha256'].items():
            assert sha(ROOT/rel) == expected, rel
        proofs[variant], runs[variant] = proof, ROOT/proof['run']

        def artifact(rel):
            p = runs[variant]/rel
            assert sha(p) == proof['artifact_sha256'][rel], rel
            inputs[str(p.relative_to(ROOT))] = sha(p)
            return p

        events = [json.loads(s) for s in artifact('events.jsonl').read_text().splitlines()]
        assert len(events) == 93 and all(e['learning_rate_used'] > 0 and math.isfinite(e['loss'])
                                        and math.isfinite(e['gradient_l2_preclip']) for e in events)
        assert sum(e['qa_seen']-(events[i-1]['qa_seen'] if i and events[i-1]['label']==e['label'] else 0)
                   for i,e in enumerate(events)) == 1313
        training[variant] = {'positive_updates':len(events), 'TRAIN_QA':1313,
                             'logged_main_train_seconds':sum(e['seconds'] for e in events)}
        windows[variant] = [json.loads(artifact(f'TRAIN_windows/reuse_T{t:02d}.json').read_text()) for t in range(1,20)]
        pools[variant] = torch.load(artifact('latest.pt'), map_location='cpu', weights_only=True)['pool']
        records[variant] = {}
        for domain in DOMAINS:
            rows = [json.loads(s) for s in artifact(f'evaluations/final_diagnostic/{domain}_image.jsonl').read_text().splitlines()]
            assert [r['id'] for r in rows] == proof['spec']['diagnostic_ids'][domain]
            assert len(rows) == 100
            for row in rows:
                assert math.isfinite(row['nll'])
                if 'prediction' in row:
                    assert answer_f1(row['prediction'],row['references']) == row['scores']['token_f1']
            assert [r['id'] for r in rows if 'prediction' in r] == [i for i in proof['spec']['diagnostic_ids'][domain]
                                if i in set(proof['spec']['generation_ids'][domain])]
            records[variant][domain] = rows
        if variant == variants[1]:
            assert read(artifact('first_task_exact_gate.json'))['passed']
            receipt = read(artifact('feature_identity_repair_execution.json'))
            assert receipt['GPU_gate']['passed'] and receipt['old_partial_predictions_reused'] is False
            for stage in range(1,20):
                assert read(artifact(f'initialization_gates/T{stage:02d}.json'))['passed']
                assert read(artifact(f'geometry_gates/S{stage:02d}.json'))['passed']

    cold, warm = variants
    spec = proofs[warm]['spec']
    assert spec['train_ids'] == proofs[cold]['spec']['train_ids']
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = read(manifest_path)
    assert sha(manifest_path) == spec['manifest_sha256']
    needed = {i for ids in spec['diagnostic_ids'].values() for i in ids}
    gold = {}
    for task in manifest['tasks']:
        p = ROOT/task['files']['test']['path']
        assert sha(p) == task['files']['test']['sha256']
        inputs[str(p.relative_to(ROOT))] = sha(p)
        for s in p.read_text().splitlines():
            row = json.loads(s)
            if row['id'] in needed:
                gold[row['id']] = row
    assert set(gold) == needed
    domains, tasks = [], {t:[] for t in range(1,20)}
    route_exact = 0
    for domain in DOMAINS:
        left, right = records[cold][domain], records[warm][domain]
        assert [r['id'] for r in left] == [r['id'] for r in right]
        values, images = [], []
        scores = {v:[] for v in variants}
        pred_changes = 0
        for a,b in zip(left,right):
            assert a['references'] == b['references'] == gold[a['id']]['answers']
            fields = ['id','native_task','route_mode','route_expert','route_cosine','route_threshold']
            assert {k:a[k] for k in fields} == {k:b[k] for k in fields}
            route_exact += 1
            if 'prediction' not in a:
                assert 'prediction' not in b
                continue
            x,y = a['scores']['token_f1'],b['scores']['token_f1']
            scores[cold].append(x);scores[warm].append(y)
            values.append(y-x);images.append(gold[a['id']]['images'])
            pred_changes += a['prediction'] != b['prediction']
            tasks[a['native_task']].append((x,y))
        domains.append({'domain':domain,'NLL_QA':100,'generated_QA':len(values),
            'cold_F1':sum(scores[cold])/len(values),'warm_F1':sum(scores[warm])/len(values),
            'paired_difference':interval(values,images),'changed_generated_answers':pred_changes,
            'cold_NLL':sum(r['nll'] for r in left)/100,'warm_NLL':sum(r['nll'] for r in right)/100})
    assert route_exact == 500 and sum(d['generated_QA'] for d in domains) == 155
    task_rows = [{'task':t,'generated_QA':len(v),'cold_F1':sum(x for x,y in v)/len(v),
                 'warm_F1':sum(y for x,y in v)/len(v)} for t,v in tasks.items()]
    assert all(tasks.values())
    aggregates = {}
    for v,key in [(cold,'cold_F1'),(warm,'warm_F1')]:
        aggregates[v] = {'AP19':sum(r[key] for r in task_rows)/19,'domain_macro':sum(r[key] for r in domains)/5}
        expected = next(r for r in verified['results'] if r['variant']==v and r['phase']=='train')
        assert aggregates[v]['AP19'] == expected['AP']
    pool_exact = {e:all(torch.equal(pools[cold][e][k][l],pools[warm][e][k][l])
                       for k in pools[cold][e] for l in ['A','B']) for e in sorted(pools[cold])}
    resources = {v:{'pure_TRAIN_CUDA_peak_allocated_bytes':max(r['cuda_peak_allocated_bytes'] for r in windows[v]),
                    'pure_TRAIN_CUDA_peak_reserved_bytes':max(r['cuda_peak_reserved_bytes'] for r in windows[v]),
                    'training_summary':verified['resources'][v+'/train']} for v in variants}
    result = {'CPU_only':True,'model_forwards':0,'new_optimizer_updates':0,'new_predictions':0,
              'TEST_selects_conditions':False,'scope':'same500NLL/155generated diagnostic; NOT full21454',
              'domains':domains,'tasks':task_rows,'aggregate':aggregates,'routing_exact_rows':route_exact,
              'final_expert_AB_bitwise_equal':pool_exact,'training':training,'resources':resources,
              'input_sha256':inputs,'source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                  'analysis/answer_f1.py':sha(ROOT/'analysis/answer_f1.py'),
                  'analysis/onevision_acquisition_budget_bridge.py':sha(ROOT/'analysis/onevision_acquisition_budget_bridge.py')},
              'limits':['155 fixed generated QA do not establish full21454 or all-domain population gains.',
                        'Fixed-weight image-bootstrap intervals do not replace repeated training seeds.',
                        'Complete inheritance changes function and optimization access together.',
                        'Full goal and +5pp complete-domain-macro target remain unproven.',
                        'Different admission/warmup histories preclude declaring a memory advantage from before/after differences.']}
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['aggregate','routing_exact_rows','final_expert_AB_bitwise_equal']}))
    print(json.dumps({'domain_effects':[{k:d[k] for k in ['domain','generated_QA','paired_difference','changed_generated_answers']} for d in domains]}))


if __name__ == '__main__':
    main()
