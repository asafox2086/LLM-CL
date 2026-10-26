"""One stdlib derivation from two sealed aggregate-only bundles; never score.

CLI: --bundle SIX_JSON_BUNDLE --cells01 SEALED_38_SCORES --output RESULT.json
The Markdown sibling is created once. No embedded path is opened: all payloads
are decoded from the two supplied metadata bundles, then checked against the
original formal proof hashes. Importing this file performs no I/O.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import time


PROOFS = {
    'pct01': 'c0c010f18db89b088cd3ac13b92e79a76c560c13ee9f9960f0e5e2ddc8e88c69',
    'pct10': '73fdbbd139a6910225088e78cc44033824456e09dec93411ed93582ac4d67c99',
}
POOLS = {
    'pct01': ('data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json',
              '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777', 1313),
    'pct10': ('data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json',
              '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7', 13105),
}
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
METHODS_SHA = 'e5bd8f51690b006274ed5adcfa4ef3b1a2c3b0f1d79476d948b71d1cfec41c5b'
VALIDITY_SHA = '0ec302d2fa4e8bb1a87b41c861c3aea78bf473e3df805b87d4dd450d21a2f735'
SCORE_KEYS = {'rougeL', 'token_f1', 'exact_match', 'count', 'exact_match_correct'}


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def number(value, low=0, high=100):
    check(isinstance(value, (int, float)) and not isinstance(value, bool)
          and math.isfinite(value) and low <= value <= high, 'invalid sealed scalar')
    return value


def positive_int(value):
    check(isinstance(value, int) and not isinstance(value, bool) and value > 0,
          'invalid positive count')
    return value


def relative(name):
    path = PurePosixPath(name)
    check(isinstance(name, str) and not path.is_absolute() and '..' not in path.parts,
          'invalid embedded relative path')
    return path.as_posix()


def payload(ref, expected, ledger, label):
    raw = base64.b64decode(ref['raw_base64'], validate=True)
    check(sha(raw) == ref['sha256'] == expected, 'payload SHA chain: ' + label)
    if 'expected_sha256' in ref:
        check(ref['expected_sha256'] == expected, 'expected payload SHA: ' + label)
    if 'bytes' in ref:
        check(ref['bytes'] == len(raw), 'payload byte count: ' + label)
    ledger[label] = {key: ref[key] for key in
                     ('root', 'relative', 'path', 'resolved_path') if key in ref}
    ledger[label].update(sha256=expected, bytes=len(raw))
    return json.loads(raw)


def source_sha(proof, path, expected):
    check(proof['source_sha256'][relative(path)] == expected,
          'original proof source SHA: ' + path)


def same_scalar(left, right, message):
    check(math.isclose(left, right, rel_tol=0, abs_tol=1e-10), message)


def load_condition(bundle, condition, ledger):
    item = bundle['conditions'][condition]
    proof = payload(item['formal_proof'], PROOFS[condition], ledger, condition + '.formal_proof')
    candidate = proof['candidate']
    run = relative(candidate['run'])
    check(item['candidate_run'] == run and candidate['backbone'] == 'onevision'
          and candidate['method'] == 'SeqFT_LoRA', 'candidate identity')
    check(proof['primary_metric'] == 'existing raw Answer F1'
          and proof['rescored_cells'] == 74 and proof['final_QA'] == 21454
          and proof['positive_finite_training_stages'] == 19, 'formal completeness')
    check(candidate.get('condition', condition) == condition, 'candidate condition')
    config_ref = item['files']['config']
    check(relative(config_ref['relative']) == run + '/config.json', 'config source path')
    source_sha(proof, config_ref['relative'], proof['config_sha256'])
    config = payload(config_ref, proof['config_sha256'], ledger, condition + '.config')
    check(config['condition'] == condition and config['method'] == 'SeqFT_LoRA'
          and config['backbone'] == 'onevision' and config['seed'] == 42
          and config['epochs'] == 1 and config['batch_size'] == 16
          and config['micro_batch_size'] == 1 and config['test_fraction'] == 1
          and config['train_fraction'] == (.01 if condition == 'pct01' else .1)
          and not config['preflight_only'], 'shared training config')
    check(config['execution_source_sha256']['methods.py'] == METHODS_SHA
          and config['execution_source_sha256']['training_validity.py'] == VALIDITY_SHA,
          'shared parameterization and scheduler-source identity')
    runner = PurePosixPath(run).parts[0:2]
    port = '/'.join(runner)
    for filename, expected in (('methods.py', METHODS_SHA), ('training_validity.py', VALIDITY_SHA)):
        source_sha(proof, port + '/' + filename, expected)
        source_sha(proof, run + '/runner_source/' + filename, expected)
    manifest_path, manifest_sha, train_total = POOLS[condition]
    check(config['manifest_sha256'] == manifest_sha
          and proof.get('manifest_sha256', manifest_sha) == manifest_sha
          and candidate.get('manifest_sha256', manifest_sha) == manifest_sha,
          'manifest identity in config/proof')
    manifest_ref = item['files']['manifest']
    check(relative(manifest_ref['relative']) == manifest_path, 'fixed manifest path')
    source_sha(proof, manifest_path, manifest_sha)
    manifest = payload(manifest_ref, manifest_sha, ledger, condition + '.manifest')
    check(manifest['experiment_id'] == 'five_domain_native_cl_20261016_sparse'
          and manifest['condition'] == condition and manifest['seed'] == 42
          and manifest['formal_ready'] and not manifest['assets_pending'], 'manifest contract')
    tasks = manifest['tasks']
    check([t['task'] for t in tasks] == list(range(1, 20)), 'all original tasks required')
    check(sum(positive_int(t['files']['train']['count']) for t in tasks) == train_total,
          'TRAIN count')
    check(sum(positive_int(t['files']['test']['count']) for t in tasks) == 21454,
          'complete TEST count')
    check(set(t['domain'] for t in tasks) == set(DOMAINS), 'original five domains')
    for task in tasks:
        test = task['files']['test']
        source_sha(proof, test['path'], test['sha256'])
        check(config['task_counts'][task['task_id']]['train'] == task['files']['train']['count']
              and config['task_counts'][task['task_id']]['test'] == test['count'], 'config task counts')
    metric_ref = item['files']['metrics']
    check(relative(metric_ref['relative']) == run + '/metrics_by_score.json', 'metrics path')
    expected = proof['source_sha256'][metric_ref['relative']]
    metrics = payload(metric_ref, expected, ledger, condition + '.metrics')
    same_scalar(number(metrics['token_f1']['AP']), number(proof['aggregate']['AP']),
                'sealed aggregate metric identity')
    same_scalar(number(metrics['token_f1']['domain_macro_final']), number(proof['domain_macro_final']),
                'sealed final domain macro identity')
    # Final metrics are identity checks only; no retention/forgetting decomposition.
    return dict(proof=proof, config=config, manifest=manifest, root=item['root'], run=run)


def cells01(extra, data, ledger):
    check(extra['schema'] == 'private-pct01-acquisition-cells-v1', 'pct01 cell bundle schema')
    check(extra['root'] == data['root'] and extra['candidate_run'] == data['run']
          and extra['formal_proof']['sha256'] == PROOFS['pct01'], 'pct01 cell provenance')
    check(len(extra['files']) == 38, 'exactly 38 pre/acquired cells required')
    cells = {}
    for ref in extra['files']:
        task, stage, when = ref['task'], ref['stage'], ref['when']
        check(isinstance(task, int) and not isinstance(task, bool) and task in range(1, 20), 'cell task')
        check(when in ('before', 'acquired') and stage == task - (when == 'before'), 'cell stage')
        check(ref['root'] == data['root'], 'cell root identity')
        expected_path = data['run'] + f'/evaluations/stage_{stage:02d}/T{task:02d}_test_predictions.scores.json'
        check(relative(ref['relative']) == expected_path, 'aggregate score source path')
        expected = data['proof']['source_sha256'][expected_path]
        value = payload(ref, expected, ledger, f'pct01.cell.{task}.{when}')
        check(set(value) == SCORE_KEYS, 'actual sealed aggregate score schema')
        count = positive_int(value['count'])
        check(count == data['manifest']['tasks'][task - 1]['files']['test']['count'], 'score QA count')
        check(isinstance(value['exact_match_correct'], int)
              and 0 <= value['exact_match_correct'] <= count, 'existing exact count')
        key = stage, task
        check(key not in cells, 'duplicate pre/acquired cell')
        cells[key] = dict(F1=number(value['token_f1']), QA=count,
                          source='original sealed aggregate token_f1/count', source_sha256=expected)
    wanted = {(stage, task) for task in range(1, 20) for stage in (task - 1, task)}
    check(set(cells) == wanted, 'missing pre/acquired cells')
    return cells


def cells10(data):
    records = data['proof']['verified_cells']
    check(len(records) == 74, 'pct10 original 74 verified cells')
    cells = {}
    for row in records:
        stage, task = row['stage'], row['task']
        check(isinstance(stage, int) and isinstance(task, int)
              and 0 <= stage <= 19 and 1 <= task <= 19, 'verified cell index')
        key = stage, task
        check(key not in cells, 'duplicate verified cell')
        count = positive_int(row['QA'])
        check(count == data['manifest']['tasks'][task - 1]['files']['test']['count'], 'verified QA count')
        cells[key] = dict(F1=number(row['F1']), QA=count,
                          source='sealed pct10 verified_cells', formal_proof_sha256=PROOFS['pct10'])
    check(all((stage, task) in cells for task in range(1, 20) for stage in (task - 1, task)),
          'missing pct10 pre/acquired cell')
    return cells


def summary(rows):
    total = sum(row['TEST_QA'] for row in rows)
    before = sum(row['before_F1'] * row['TEST_QA'] for row in rows) / total
    acquired = sum(row['acquired_F1'] * row['TEST_QA'] for row in rows) / total
    return dict(TEST_QA=total, tasks=[row['task'] for row in rows], before_F1=before,
                acquired_F1=acquired, acquisition_change_pp=acquired - before,
                multiple_task_specific_model_times=True)


def direction_counts(rows, field):
    return dict(higher=sum(row[field] > 0 for row in rows),
                lower=sum(row[field] < 0 for row in rows),
                tie=sum(row[field] == 0 for row in rows), denominator=len(rows))


def fmt(value):
    return f'{value:.6f}'


def text(value):
    return str(value).replace('|', '\\|').replace('\r', ' ').replace('\n', ' ')


def markdown(result):
    lines = ['# 全19任务的新习得：固定1%／10%学习配置比较', '',
        '只读两条已完整封存的OneVision／SeqFT-LoRA轨迹，0新评分／生成／训练。'
        '前值为R[t−1,t]，刚习得为R[t,t]；变化为后减前。所有原分、负变化和完整分母保留。', '',
        '两组源参数化同为q/v-r8、alpha32、dropout0.1；不是相同的已学数值子空间。'
        '完整参数及可学坐标沿用已有测量声明，本消费者没有读权重或重新测量。', '',
        '|任务|域|原native标签|TEST QA|1% TRAIN/更新|1%前|1%刚习得|1%变化|10% TRAIN/更新|10%前|10%刚习得|10%变化|刚习得10−1|变化10−1|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['tasks']:
        a, b = row['pct01'], row['pct10']
        label = row['native_label'] if row['native_label'] is not None else '—'
        lines.append('|'+ '|'.join([f"T{row['task']:02d}", text(row['domain']), text(label),
            str(row['TEST_QA']), f"{a['TRAIN_QA']}/{a['single_epoch_optimizer_calls']}",
            fmt(a['before_F1']), fmt(a['acquired_F1']), fmt(a['acquisition_change_pp']),
            f"{b['TRAIN_QA']}/{b['single_epoch_optimizer_calls']}", fmt(b['before_F1']),
            fmt(b['acquired_F1']), fmt(b['acquisition_change_pp']),
            fmt(row['pct10_minus_pct01_acquired_F1_pp']),
            fmt(row['pct10_minus_pct01_acquisition_change_pp'])])+'|')
    lines += ['', '|原域|TEST QA|1%前|1%刚习得|1%变化|10%前|10%刚习得|10%变化|刚习得10−1|变化10−1|',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for domain in DOMAINS:
        a, b = result['conditions']['pct01']['domains'][domain], result['conditions']['pct10']['domains'][domain]
        lines.append('|'+ '|'.join([domain,str(a['TEST_QA']),fmt(a['before_F1']),fmt(a['acquired_F1']),
            fmt(a['acquisition_change_pp']),fmt(b['before_F1']),fmt(b['acquired_F1']),
            fmt(b['acquisition_change_pp']),fmt(b['acquired_F1']-a['acquired_F1']),
            fmt(b['acquisition_change_pp']-a['acquisition_change_pp'])])+'|')
    for key,label in (('AP19','19任务等权'),('domain_macro','五域等权宏')):
        a,b=result['conditions']['pct01'][key],result['conditions']['pct10'][key]
        lines.append('|'+ '|'.join([label,'—',fmt(a['before_F1']),fmt(a['acquired_F1']),
            fmt(a['acquisition_change_pp']),fmt(b['before_F1']),fmt(b['acquired_F1']),
            fmt(b['acquisition_change_pp']),fmt(b['acquired_F1']-a['acquired_F1']),
            fmt(b['acquisition_change_pp']-a['acquisition_change_pp'])])+'|')
    counts = result['cross_ratio_task_counts']
    lines += ['', f"刚习得水平：{counts['acquired']['higher']}高／{counts['acquired']['lower']}低／"
              f"{counts['acquired']['tie']}平；阶段内变化：{counts['change']['higher']}高／"
              f"{counts['change']['lower']}低／{counts['change']['tie']}平，分母均为19个原任务。", '',
        '域内按原TEST QA数加权，五域宏等权；AP19按19任务等权。域级前／刚习得均汇集多个任务各自时点，'
        '不是同一模型时刻的整域测试。TRAIN/更新由原manifest单轮ceil(TRAIN/16)派生，'
        '更新是optimizer调用数，不能全部改称正LR更新。', '',
        '局限：相同参数化容量不等于相同已学方向；数据、更新数、调度及先前课程共同变化。'
        '起点更低也可造成更大阶段增量，增量不能代替刚习得绝对水平。'
        '本结果不识别纯数据量因果，不涵盖另一骨干／其它方法，不选择新算法，'
        '不研究保持／遗忘，不证明方法创新或资源优势。', '',
        f"完整JSON及来源账本：{text(result['output_json_name'])}；只复制／派生既有聚合分数，0新评分。"]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--cells01', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    report = output.with_suffix('.md')
    check(output.suffix.lower() == '.json' and not output.exists() and not report.exists(),
          'JSON and Markdown outputs must both be new')
    tick, cpu, unix = time.monotonic(), time.process_time(), time.time()
    ledger = {}
    inputs = {}
    for label,path in (('bundle',args.bundle),('cells01_bundle',args.cells01)):
        raw=path.read_bytes()
        ledger[label]=dict(path=str(path.resolve()),sha256=sha(raw),bytes=len(raw))
        inputs[label]=json.loads(raw)
    bundle=inputs['bundle']
    check(bundle['schema']=='private-sealed-acquisition-bundle-v1'
          and set(bundle['conditions'])=={'pct01','pct10'},'two-condition bundle schema')
    data={condition:load_condition(bundle,condition,ledger) for condition in ('pct01','pct10')}
    cells={'pct01':cells01(inputs['cells01_bundle'],data['pct01'],ledger),
           'pct10':cells10(data['pct10'])}
    rows=[]
    for task in range(1,20):
        a,b=(data[c]['manifest']['tasks'][task-1] for c in ('pct01','pct10'))
        for field in ('task','task_id','domain','modality','instruction','native_label_field',
                      'native_label','max_new_tokens'):
            check(a[field]==b[field],'task identity differs: '+field)
        ta,tb=a['files']['test'],b['files']['test']
        check(ta['sha256']==tb['sha256'] and ta['count']==tb['count'],'TEST SHA/count differ')
        row=dict(task=task,task_id=a['task_id'],domain=a['domain'],native_label_field=a['native_label_field'],
                 native_label=a['native_label'],TEST_QA=ta['count'],TEST_identity=dict(
                     sha256=ta['sha256'],count=ta['count'],
                     source_paths={'pct01':ta['path'],'pct10':tb['path']},
                     ID_and_reference_identity_inherited_from_same_file_SHA=True,
                     TEST_rows_re_read=False))
        for condition in data:
            original=data[condition]['manifest']['tasks'][task-1]
            n=original['files']['train']['count']
            pre,post=cells[condition][task-1,task],cells[condition][task,task]
            row[condition]=dict(TRAIN_QA=n,single_epoch_optimizer_calls=math.ceil(n/16),
                before_stage=task-1,acquired_stage=task,before_F1=pre['F1'],acquired_F1=post['F1'],
                acquisition_change_pp=post['F1']-pre['F1'],
                before_cell_source=pre,acquired_cell_source=post)
        row['pct10_minus_pct01_acquired_F1_pp']=row['pct10']['acquired_F1']-row['pct01']['acquired_F1']
        row['pct10_minus_pct01_acquisition_change_pp']=row['pct10']['acquisition_change_pp']-row['pct01']['acquisition_change_pp']
        rows.append(row)
    conditions={}
    for condition in data:
        merged=[dict(row[condition],task=row['task'],domain=row['domain'],TEST_QA=row['TEST_QA']) for row in rows]
        domains={domain:summary([row for row in merged if row['domain']==domain]) for domain in DOMAINS}
        pre=sum(row['before_F1'] for row in merged)/19
        post=sum(row['acquired_F1'] for row in merged)/19
        macro_pre=sum(domains[d]['before_F1'] for d in DOMAINS)/5
        macro_post=sum(domains[d]['acquired_F1'] for d in DOMAINS)/5
        conditions[condition]=dict(TRAIN_QA=sum(row['TRAIN_QA'] for row in merged),
            single_epoch_optimizer_calls=sum(row['single_epoch_optimizer_calls'] for row in merged),
            domains=domains,AP19=dict(before_F1=pre,acquired_F1=post,acquisition_change_pp=post-pre),
            domain_macro=dict(before_F1=macro_pre,acquired_F1=macro_post,acquisition_change_pp=macro_post-macro_pre))
    proof=data['pct10']['proof']; acquired=proof['acquired']; derived=conditions['pct10']
    same_scalar(derived['AP19']['acquired_F1'],number(acquired['AP19']),'pct10 acquired AP19 recheck')
    same_scalar(derived['domain_macro']['acquired_F1'],number(acquired['domain_macro_F1']),'pct10 acquired five-domain macro recheck')
    check(len(acquired['native19'])==19,'pct10 acquired native19 length')
    for row,sealed in zip(rows,acquired['native19']):
        same_scalar(row['pct10']['acquired_F1'],number(sealed),'pct10 acquired task scalar')
    for domain in DOMAINS:
        sealed=acquired['domains'][domain]; value=derived['domains'][domain]
        check(sealed['QA']==value['TEST_QA'],'pct10 acquired domain QA')
        same_scalar(value['acquired_F1'],number(sealed['F1']),'pct10 acquired domain recheck')
    check(derived['single_epoch_optimizer_calls']==proof['current_optimizer_updates'], 'pct10 update budget binding')
    schedule={row['stage']:row for row in proof['training_schedule']}
    check(set(schedule)==set(range(1,20)),'pct10 sealed training stages')
    for row in rows:
        check(schedule[row['task']]['optimizer_calls']==row['pct10']['single_epoch_optimizer_calls'],
              'pct10 per-task optimizer calls')
    source=Path(__file__).resolve(); source_raw=source.read_bytes()
    ledger['consumer_source']=dict(path=str(source),sha256=sha(source_raw),bytes=len(source_raw))
    result=dict(status='COMPLETE_SEALED_AGGREGATE_DERIVATION',output_json_name=output.name,
        backbone='onevision',method='SeqFT_LoRA',conditions=conditions,tasks=rows,
        cross_ratio_task_counts=dict(acquired=direction_counts(rows,'pct10_minus_pct01_acquired_F1_pp'),
                                    change=direction_counts(rows,'pct10_minus_pct01_acquisition_change_pp')),
        source_sha256=ledger,formal_proof_sha256=PROOFS,
        parameterization=dict(source_methods_sha256=METHODS_SHA,rank=8,alpha=32,dropout=.1,
            targets=['q_proj','v_proj'],adapter_parameter_dtype='torch.float32',
            rank_and_targets_source='Pinned methods.py, not fields in the run configs',
            same_parameterization_function_class=True,same_learned_numeric_U_V_subspace_not_established=True,
            previously_reported_trainable_coordinates_each=540672,
            previously_reported_complete_coefficients_each=894216224,
            count_scope='Prior published measurements supplied by root; no tensor layout or weight payload read by this consumer'),
        budget_source='Original manifest TRAIN counts, one epoch, batch16/micro1; ceil counts optimizer calls, not all-positive LR claims',
        weights=dict(domain_within='Original TEST QA counts',domain_macro='Equal five domains',AP19='Equal 19 tasks'),
        limits=['Descriptive after completed runs, not blind preregistration.',
                'Pre/acquired domain aggregates use multiple task-specific model times, not one joint model state.',
                'More data changes updates, schedule and prior curriculum; differences are not pure data-size causality.',
                'A larger change from a lower starting score does not itself imply better acquired performance.',
                'OneVision/SeqFT only, not both backbones or all methods; no new algorithm or method selection.',
                'No retention/forgetting analysis, bootstrap, geometry/norm audit, or memory/quality advantage claim.'],
        activity=dict(new_scores=0,new_generations=0,new_training=0,model_or_checkpoint_payload_reads=0,
                      QA_or_prediction_rows_read=0,embedded_payload_paths_opened=0,new_bootstraps=0,
                      original_pct01_aggregate_score_JSONs_decoded=38,
                      pct10_verified_cells_inherited=74),
        elapsed=dict(started_unix=unix,prepared_UTC=datetime.now(timezone.utc).isoformat(),
                     wall_seconds_before_output_write=time.monotonic()-tick,
                     CPU_seconds_before_output_write=time.process_time()-cpu))
    rendered=markdown(result)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    with report.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(rendered)


if __name__=='__main__':
    main()
