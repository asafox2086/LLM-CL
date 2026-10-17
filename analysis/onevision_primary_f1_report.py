"""Reassess completed diagnostics using the sole user-selected Answer F1.

Raw token_f1 is retained exactly. No option/answer extraction or GPU inference.
"""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from answer_f1 import answer_f1,VERSION

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def average(rows):
    assert rows
    return sum(r['answer_f1'] for r in rows)/len(rows)


def main():
    sources={}
    def read(path):
        sources[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest=read(manifest_path)
    assert sha(manifest_path)=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    original={}
    for task in manifest['tasks']:
        f=task['files']['test'];path=ROOT/f['path'];assert sha(path)==f['sha256'];sources[f['path']]=f['sha256']
        original.update({r['id']:dict(r,_task=task['task'],_domain=task['domain']) for r in map(json.loads,path.read_text().splitlines())})

    def rows(path,digest,expected_ids=None,task=None):
        assert sha(path)==digest
        data=[json.loads(line) for line in path.read_text().splitlines()]
        assert len({r['id'] for r in data})==len(data)
        if expected_ids is not None:assert [r['id'] for r in data]==expected_ids
        result=[]
        for row in data:
            assert row['references']==original[row['id']]['answers']
            if 'prediction' not in row:continue
            if task is not None and original[row['id']]['_task']!=task:continue
            value=answer_f1(row['prediction'],row['references'])
            assert value==row['scores']['token_f1']
            result.append({'id':row['id'],'answer_f1':value})
        assert sha(path)==digest
        sources[str(path.relative_to(ROOT))]=digest
        return result

    def delta(a,b):
        assert [r['id'] for r in a]==[r['id'] for r in b] and a
        values=[y['answer_f1']-x['answer_f1'] for x,y in zip(a,b)]
        groups=defaultdict(list)
        for row,value in zip(a,values):groups[tuple(original[row['id']]['images'])].append(value)
        clusters=list(groups.values());rng=random.Random(42);samples=[]
        for _ in range(2000):
            chosen=[clusters[rng.randrange(len(clusters))] for _ in clusters]
            samples.append(sum(map(sum,chosen))/sum(map(len,chosen)))
        samples.sort()
        return {'QA':len(a),'before':average(a),'after':average(b),'delta_pp':sum(values)/len(values),
                'image_cluster_95CI':[samples[49],samples[1949]],
                'quality_increased_QA':sum(v>0 for v in values),'quality_decreased_QA':sum(v<0 for v in values),
                'quality_unchanged_QA':sum(v==0 for v in values),
                'sum_positive_QA_points':sum(v for v in values if v>0),
                'sum_negative_QA_points':sum(v for v in values if v<0),
                'scope':'Question-level quality changes, not binary accuracy/retention; bootstrap conditional on fixed weights, uncorrected.'}

    proof=read(REPORT/'deterministic_native/complete.json');assert proof['complete'];run=ROOT/proof['run']
    cache={}
    def native(label,domain,task):
        key=(label,domain)
        if key not in cache:
            path=run/'evaluations'/label/f'{domain}_image.jsonl'
            cache[key]=rows(path,proof['artifact_sha256'][str(path.relative_to(run))])
        return [r for r in cache[key] if original[r['id']]['_task']==task]
    task_results=[];all_acquired=[];all_final=[]
    for t in manifest['tasks']:
        tid,d=t['task'],t['domain']
        cohorts={name:native(label,d,tid) for name,label in {
            'base':'base_initial','before_task':f'pre_T{tid:02d}',
            'isolated':f'matched_native_T{tid:02d}','acquired':f'sequential_T{tid:02d}','final':'sequential_T19'}.items()}
        assert all([r['id'] for r in v]==[r['id'] for r in cohorts['base']] for v in cohorts.values())
        result={'task':tid,'domain':d,'train_QA':t['files']['train']['count'],'generated_QA':len(cohorts['base']),
                'Answer_F1':{name:average(v) for name,v in cohorts.items()},
                'acquired_to_final':delta(cohorts['acquired'],cohorts['final']),
                'isolated_to_acquired':delta(cohorts['isolated'],cohorts['acquired'])}
        task_results.append(result);all_acquired+=cohorts['acquired'];all_final+=cohorts['final']
    assert len(all_final)==155
    domain_results=[]
    for d in dict.fromkeys(t['domain'] for t in manifest['tasks']):
        a=[r for r in all_acquired if original[r['id']]['_domain']==d]
        b=[r for r in all_final if original[r['id']]['_domain']==d]
        domain_results.append({'domain':d,'acquired_to_final':delta(a,b)})
    native_metrics={'AP':sum(t['Answer_F1']['final'] for t in task_results)/19,
                    'FWT':sum(t['Answer_F1']['before_task']-t['Answer_F1']['base'] for t in task_results[1:])/18,
                    'BWT':sum(t['Answer_F1']['final']-t['Answer_F1']['acquired'] for t in task_results[:-1])/18,
                    'scope':'Fixed155 diagnostic generated QA across19native tasks; sparse measured cells, NOT full21454 formal results.',
                    'domain_macro_final':sum(d['acquired_to_final']['after'] for d in domain_results)/5}
    knowledge={};clinical={}
    for case in ['rank1_correct','rank1_permuted','rank8_correct','rank8_permuted']:
        p=read(REPORT/'knowledge_fit'/f'{case}_complete.json');assert p['complete'];directory=ROOT/p['run']
        knowledge[case]={}
        for group in ['medical','medical_KG']:
            endpoints={}
            for step in [0,8,64]:
                path=directory/f'evaluations/clinical_update_{step:02d}'/f'{group}_image.jsonl'
                endpoints[step]=rows(path,p['artifact_sha256'][str(path.relative_to(directory))],p['spec']['clinical_ids'][group])
                clinical[case,step,group]=endpoints[step]
            knowledge[case][group]={'QA':len(endpoints[0]),'Answer_F1':{str(k):average(v) for k,v in endpoints.items()},
                                     'step0_to64':delta(endpoints[0],endpoints[64])}
    knowledge_pairs=[]
    for rank in [1,8]:
        for group in ['medical','medical_KG']:
            knowledge_pairs.append({'rank':rank,'group':group,'permuted_to_correct_step64':delta(
                clinical[f'rank{rank}_permuted',64,group],clinical[f'rank{rank}_correct',64,group])})
    rural=[]
    for case in ['uniform','class_balanced','urban_high_negative']:
        p=read(REPORT/'rural_weights'/f'{case}_complete.json');assert p['complete'];directory=ROOT/p['run'];endpoints={}
        for step in [1,4]:
            path=directory/f'evaluations/endpoint_{step}/TEST_image.jsonl'
            endpoints[step]=rows(path,p['artifact_sha256'][str(path.relative_to(directory))],p['spec']['test_ids'])
        rural.append({'case':case,'QA':100,'Answer_F1':{str(k):average(v) for k,v in endpoints.items()},'one_to_four_updates':delta(endpoints[1],endpoints[4])})
    closed=read(REPORT/'FULL_COMPLETED_DOMAINS_AUDIT.json');full=[]
    for d in ['natural','medical','geography','finance']:
        path=ROOT/'exp/onevision_capacity_20261016/cap1_full_confirm/20261017_091853/evaluations/final_full_input_router'/f'{d}_image.jsonl'
        values=rows(path,closed['source_sha256'][str(path.relative_to(ROOT))])
        assert len(values)==sum(t['files']['test']['count'] for t in manifest['tasks'] if t['domain']==d)
        full.append({'domain':d,'full_test_QA':len(values),'Answer_F1':average(values),'scope':'Completed domain only; traffic still excluded while live.'})
    result={'primary_metric':'Answer F1','metric_version':VERSION,'same_as_existing_score_field':'token_f1',
            'option_mapping':False,'extra_normalization_or_new_inference':False,
            'native_19_tasks':task_results,'native_diagnostic_metrics':native_metrics,
            'native_diagnostic_quality_changes':delta(all_acquired,all_final),'native_domains':domain_results,
            'knowledge':knowledge,'knowledge_correct_vs_permuted':knowledge_pairs,'rural_weights':rural,
            'full_confirmation_closed_domains':full,'all21454_complete':False,'sources':sources,
            'metric_sha256':sha(ROOT/'analysis/answer_f1.py'),'reporter_sha256':sha(Path(__file__)),
            'limits':['Answer F1 is lexical content overlap, not full semantic correctness. No option normalization or semantic judge.',
                      'Old EM/ROUGE files preserved for traceability, not co-primary metrics or new tuning objectives.',
                      'Native155 cohort is diagnostic; knowledge medical100 andKG267 overlap12, never pooled as367 independent QA.',
                      'Fact-fit64 updates512 exposures are auxiliary interventions, not formal1epoch CL.',
                      'Existing histories and unmatched domain order do not establish pure task-count/domain-count causality.',
                      'QA bootstrap intervals are conditional on fixed fitted weights, single seed/order, no multiple-comparison correction.']}
    for name,digest in sources.items():assert sha(ROOT/name)==digest
    (REPORT/'PRIMARY_F1_REASSESSMENT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Answer F1唯一主指标：已有诊断的重新审视','',
           '用户决定：沿用原始token_f1，改称Answer F1；不做选项映射、不加前缀去除、不用语义裁判。原预测/训练/评分文件不改。以下均重新核验原文件SHA、金标及逐题分数完全相同。','',
           '## 1. 顺序学习：质量上升和下降是否相抵？','',
           '| 域 | 固定生成QA | 刚学完累计F1 | 最终F1 | 差值pp | 质量上升/下降/持平题数 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for d in domain_results:
        v=d['acquired_to_final'];lines.append(f"| {d['domain']} | {v['QA']} | {v['before']:.3f} | {v['after']:.3f} | {v['delta_pp']:+.3f} | {v['quality_increased_QA']}/{v['quality_decreased_QA']}/{v['quality_unchanged_QA']} |")
    total=result['native_diagnostic_quality_changes'];lines+=['',f"全部155题：{total['before']:.3f}→{total['after']:.3f}；{total['quality_increased_QA']}题上升、{total['quality_decreased_QA']}题下降、{total['quality_unchanged_QA']}题持平。刚学完值组合不同任务各自习得时点，不是一个模型快照。",'',
           f"19原生任务诊断AP={native_metrics['AP']:.3f}，FWT={native_metrics['FWT']:.3f}，BWT={native_metrics['BWT']:.3f}。都基于同一个Answer F1，仅诊断155题，不进入完整21454题主表。",'',
           '## 2. 记住极小知识库是否足够？','',
           '| 条件 | 医学100题：0→64步F1 | 原生KG267题：0→64步F1 |',
           '| --- | ---: | ---: |']
    for case in knowledge:
        a=knowledge[case]['medical']['Answer_F1'];b=knowledge[case]['medical_KG']['Answer_F1']
        lines.append(f"| {case} | {a['0']:.3f}→{a['64']:.3f} | {b['0']:.3f}→{b['64']:.3f} |")
    lines+=['','正确/错配同rank同预算的配对差值与区间见JSON。不能因某一rank或某个分组点估计高就选择它。医学100与KG267重叠12，不合并计数。事实回忆已另有干预有效性证据，但“写入事实→目标问答普遍提高”的链条仍未成立。','',
            '## 3. 遥感坍缩与部分改进在F1下还成立吗？','',
            '| 固定损失条件 | 1步F1 | 4步F1 |', '| --- | ---: | ---: |']
    for d in rural:lines.append(f"| {d['case']} | {d['Answer_F1']['1']:.3f} | {d['Answer_F1']['4']:.3f} |")
    lines+=['','这些单词标签任务的F1数值与旧二值评分相同，因此已证明的坍缩/类别平衡局部作用并未因换主指标消失。仍不证明视觉完全失效或类别比例是唯一成因。','',
            '## 4. 完整确认仅已结束的四域','', '| 域 | 完整QA | Answer F1 |','| --- | ---: | ---: |']
    for d in full:lines.append(f"| {d['domain']} | {d['full_test_QA']} | {d['Answer_F1']:.3f} |")
    lines+=['','交通在跑的尾部不读、不填0；上表不计算全21454 AP/FWT/BWT。研究nativeAnyRes/统一lr1e-4与正式双骨干V2设置分别报告，不跨协议排名。','',
            '## 下一步判断','',
            '已验证：有些退化在连续F1下仍存在，极小知识写入不能保证所有目标组获益。未验证：多域数量相对少任务/预算是主要原因。既定确定性双锚六机制控制仍是核心缺口；不以本次重评分替代该GPU实验。','']+['- '+v for v in result['limits']]
    (REPORT/'PRIMARY_F1_REASSESSMENT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'native_metrics':native_metrics,'quality_changes':total,'knowledge_pairs':knowledge_pairs,'rural':rural,'source_count':len(sources)},ensure_ascii=False))


if __name__=='__main__':main()
