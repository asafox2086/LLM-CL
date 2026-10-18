"""Read ONLY the complete, preregistered hard-quality diagnostic cell.

The whole four-view pilot may still be running. A per-view durable completion
is required; no partial rows, view selection, GPU calls, or queue mutations.
This never marks the four-view pilot or full21454 confirmation complete.
"""
import hashlib
import json
import math
from pathlib import Path

from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'summary/onevision_diagnostics_20261016'
STUDY = BASE/'warm_quality_20261018'
REPORT = STUDY/'predeclared_hard_case'
DOMAINS = ['natural','medical','geography','finance','traffic']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    attempt_path = STUDY/'pilot_attempt.json'
    attempt = json.loads(attempt_path.read_text())
    run, spec = ROOT/attempt['run'], attempt['spec']
    complete = run/'evaluations/quality_hard/durable_complete.json'
    if not complete.exists():
        print(json.dumps({'condition_complete':False,'reason':'Wait for complete quality_hard durable proof; no partial scores read'}))
        return
    assert not (REPORT/'VERIFIED_EVIDENCE.json').exists(), 'Do not repeat completed evidence'
    inputs = {}

    def record(p):
        inputs[str(p.relative_to(ROOT))] = sha(p)
        return p

    def read(p):
        return json.loads(record(p).read_text())

    record(attempt_path)
    assert spec == read(STUDY/'CPU_PLAN_CHECK.json')['spec']
    assert spec['full_candidates'] == ['quality_hard'] and spec['no_TEST_view_selection']
    for rel, expected in spec['source_sha256'].items():
        assert sha(record(ROOT/rel)) == expected
    parent_path = BASE/'warm_start_20261018/warm_inherit_K3_v1_train_complete.json'
    parent = read(parent_path)
    assert sha(parent_path) == spec['parent_completion_sha256'] and parent['complete']
    parent_run = ROOT/parent['run']
    assert sha(record(parent_run/'latest.pt')) == spec['parent_state_sha256']
    lock = read(ROOT/'summary/primary_metric_20261017/metric_lock.json')
    assert sha(record(ROOT/'analysis/answer_f1.py')) == lock['metric_sha256']
    assert not lock['option_mapping'] and not lock['additional_prefix_or_answer_extraction']
    gate = read(run/'original_exact_gate.json')
    assert gate['passed'] and gate['no_tolerance'] and gate['QA']==500 and gate['generated']==155
    calibration = read(run/'calibration_summary.json')
    assert calibration['actual_changed'] and calibration['TRAIN_QA']==43 and calibration['NLL_forwards']==172
    assert calibration['parameters']==16 and calibration['CPU_head_steps']==400
    assert calibration['calibration_sha256'] == sha(record(run/'calibration.pt'))
    assert read(run/'calibration.json')['sha256'] == calibration['calibration_sha256']
    receipt = read(run/'feature_identity_repair_execution.json')
    assert receipt['GPU_gate']['passed'] and receipt['old_partial_predictions_reused'] is False
    for label in ['original_hard','quality_hard']:
        proof = read(run/f'evaluations/{label}/durable_complete.json')
        assert proof['complete']
        assert set(proof['files']) == {d+'_image.jsonl' for d in DOMAINS}
        for name, expected in proof['files'].items():
            assert sha(record(run/f'evaluations/{label}'/name)) == expected
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = read(manifest_path)
    assert sha(manifest_path) == spec['parent_spec']['manifest_sha256']
    gold, train_ids = {}, set()
    for task in manifest['tasks']:
        for split in ['train','test']:
            p = record(ROOT/task['files'][split]['path'])
            assert sha(p) == task['files'][split]['sha256']
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if split=='train':train_ids.add(r['id'])
                else:gold[r['id']] = r['answers']
    assert len(train_ids)==1313 and set(spec['calibration_TRAIN_ids']) <= train_ids
    domains, by_task = {}, {t:[] for t in range(1,20)}
    for domain in DOMAINS:
        rows = [json.loads(s) for s in (run/f'evaluations/quality_hard/{domain}_image.jsonl').read_text().splitlines()]
        original = [json.loads(s) for s in (run/f'evaluations/original_hard/{domain}_image.jsonl').read_text().splitlines()]
        p = parent_run/f'evaluations/final_diagnostic/{domain}_image.jsonl'
        assert sha(record(p)) == parent['artifact_sha256'][str(p.relative_to(parent_run))]
        old = [json.loads(s) for s in p.read_text().splitlines()]
        semantic = lambda rs:[{k:v for k,v in r.items() if k!='generation_seconds'} for r in rs]
        assert semantic(original)==semantic(old)
        assert [r['id'] for r in rows] == spec['parent_spec']['diagnostic_ids'][domain]
        assert len(rows)==100
        values=[]
        for row in rows:
            assert row['references']==gold[row['id']] and math.isfinite(row['nll'])
            assert row['route_policy']=='quality_hard'
            if 'prediction' in row:
                value=answer_f1(row['prediction'],row['references'])
                assert value==row['scores']['token_f1']
                values.append(value);by_task[row['native_task']].append(value)
        assert [r['id'] for r in rows if 'prediction' in r] == [i for i in spec['parent_spec']['diagnostic_ids'][domain]
                        if i in set(spec['parent_spec']['generation_ids'][domain])]
        domains[domain]={'generated':len(values),'F1':sum(values)/len(values)}
    assert sum(r['generated'] for r in domains.values())==155 and all(by_task.values())
    current={'cell':'11','generated':155,'domains':{d:domains[d]['F1'] for d in DOMAINS},
             'AP':sum(sum(v)/len(v) for v in by_task.values())/19,
             'domain_macro':sum(r['F1'] for r in domains.values())/5,'FWT':None,'BWT':None}
    prior=read(STUDY/'FOUR_CELL_EVIDENCE.json')
    assert prior['CPU_only'] and prior['raw_Answer_F1'] and not prior['formal_V2']
    assert prior['TEST_selects_anything'] is False
    for rel,expected in prior['source_sha256'].items():
        assert sha(record(ROOT/rel))==expected
    rows={r['cell']:r for r in prior['rows']['diagnostic155'] if r['cell'] in ['00','10','01']}
    assert set(rows)=={'00','10','01'} and all(r['generated']==155 for r in rows.values())
    rows['11']=current
    effects={key:{'B_given_cold':rows['01'][key]-rows['00'][key],
                  'B_given_warm':rows['11'][key]-rows['10'][key],
                  'A_plus_B_vs_cold':rows['11'][key]-rows['00'][key],
                  'interaction':rows['11'][key]-rows['10'][key]-rows['01'][key]+rows['00'][key]}
             for key in ['AP','domain_macro']}
    result={'CPU_only':True,'GPU_operations':0,'model_or_optimizer_loads':0,
            'scope':'Complete predeclared quality_hard 500NLL/155generated only; four-view pilot may still run',
            'condition_complete':True,'four_view_pilot_marked_complete_by_this_report':False,
            'full21454_marked_complete_by_this_report':False,'TEST_view_selection':False,
            'rows':[rows[c] for c in ['00','10','01','11']],'effects':effects,
            'calibration':calibration,'original_reload':gate,'input_sha256':inputs,
            'report_source_sha256':sha(Path(__file__)),
            'limits':['All original auxiliary views, controls and full21454 jobs remain required.',
                      '155 diagnostic scores cannot establish complete21454 or +5pp full-domain-macro target.',
                      'Four-cell interaction is descriptive at one seed, not proof of statistical synergy.',
                      'Full FWT/BWT remain missing; no memory advantage from 16 parameters alone.']}
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'VERIFIED_EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 预定质量硬调用条件：完整诊断格（全四视图仍待）','',
           '只接受该条件500NLL/155生成的durable完整证明，不把其他视图或21454测试标记为完成。','',
           '| 格 | 自然 | 医学 | 遥感 | 金融 | 交通 | 域宏平均 | AP |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['rows']:
        lines.append('| '+row['cell']+' | '+' | '.join(f"{row['domains'][d]:.3f}" for d in DOMAINS)
                     +f" | {row['domain_macro']:.3f} | {row['AP']:.3f} |")
    lines+=['','[来源SHA、实际校准及四格差分](VERIFIED_EVIDENCE.json)','']
    lines+=['- '+s for s in result['limits']]
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'predeclared_hard_condition_complete':True,'cell11':current,'effects':effects,
                      'whole_pilot_or_full_not_marked_complete':True}))


if __name__=='__main__':
    main()
