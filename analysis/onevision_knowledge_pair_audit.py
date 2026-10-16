"""Read-only, paired image-cluster uncertainty for completed knowledge cases."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_knowledge_intervention import case_complete
from onevision_research_worker import digest
from run import freeze
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def paired_cluster(values,keys,repetitions=10000):
    assert len(values)==len(keys)>0 and all(np.isfinite(values))
    groups={k:[] for k in keys}
    for value,key in zip(values,keys):groups[key].append(value)
    sums=np.array([sum(v) for v in groups.values()],dtype=float)
    sizes=np.array([len(v) for v in groups.values()],dtype=float)
    rng=np.random.default_rng(42);indices=rng.integers(0,len(groups),size=(repetitions,len(groups)))
    samples=sums[indices].sum(1)/sizes[indices].sum(1)
    return {'mean':float(np.mean(values)),'image_cluster_percentile95':np.quantile(samples,[.025,.975]).tolist(),
            'QA':len(values),'image_clusters':len(groups),'bootstrap_repetitions':repetitions}


def main():
    manifest,data,domains,tests,generation,probes,protocol=freeze()
    attempt=json.loads((ROOT/'summary/onevision_mechanisms_20261016/knowledge_attempt.json').read_text())
    run=ROOT/attempt['run'];selected={'medical':tests['medical'],'medical_KG':data[11,'test']}
    families=['context_base','context_shared','residual_rank1','residual_rank2','continue_rank8']
    pairs=[(family+'_correct',family+'_'+control) for family in families
           for control in (['none','unrelated','permuted'] if family.startswith('context') else ['unrelated','permuted'])]
    pairs += [(family+'_'+kind,'context_shared_none') for family in families[2:]
              for kind in ['correct','unrelated','permuted']]
    cache={};sources={};rows=[]
    def load(name):
        if name in cache:return cache[name]
        if not (run/'evaluations'/name/'durable_complete.json').exists():return None
        assert case_complete(run,name,selected)
        output={}
        for group,expected in selected.items():
            p=run/'evaluations'/name/(group+'_image.jsonl');sources[str(p.relative_to(ROOT))]=digest(p)
            records=[json.loads(line) for line in p.read_text().splitlines()]
            assert [x['id'] for x in records]==[x['id'] for x in expected]
            output[group]=records
        cache[name]=output;return output
    for treatment,control in pairs:
        a,b=load(treatment),load(control)
        if a is None or b is None:continue
        for group,expected in selected.items():
            keys=[tuple(x['images']) for x in expected]
            metrics={}
            for metric in ['exact_match','token_f1','rougeL','nll','prediction_characters']:
                def value(record):
                    if metric=='prediction_characters':return len(record['prediction'])
                    return record['nll'] if metric=='nll' else record['scores'][metric]
                delta=[value(x)-value(y) for x,y in zip(a[group],b[group])]
                metrics[metric]=paired_cluster(delta,keys)
            rows.append({'treatment':treatment,'control':control,'group':group,'difference':'treatment minus control','metrics':metrics})
    # Verify constant-within-cluster outcomes give an exact zero interval.
    zero=paired_cluster([0.,0.,0.],['a','a','b'],100)
    assert zero['mean']==0 and zero['image_cluster_percentile95']==[0,0]
    single=paired_cluster([1.,3.],['a','a'],100)
    assert single['mean']==2 and single['image_cluster_percentile95']==[2,2]
    forward=paired_cluster([1.,3.,-2.],['a','a','b'],100)
    reverse=paired_cluster([-1.,-3.,2.],['a','a','b'],100)
    assert np.allclose(reverse['image_cluster_percentile95'],[-v for v in forward['image_cluster_percentile95'][::-1]])
    result={'run':attempt['run'],'audit_sha256':digest(Path(__file__)),'rows':rows,'source_sha256':sources,
            'CI_definition':'Paired bootstrap of image-path groups; resample groups then QA-weighted ratio, seed42,10000 draws.',
            'limitations':['Exploratory single-seed diagnostic intervals, no multiple-comparison correction or population-wide claim.',
                           'Image paths define clusters; distinct files may still depict related patients/scenes.',
                           'Medical100 and KG267 overlap12 QA; groups never pooled.',
                           'Only completed negative controls admitted; no test-guided changes to prompts, training or capacity.',
                           'A positive generated-score difference is needed alongside content-specific controls; NLL alone is insufficient.',
                           'Prediction length in Unicode characters, not tokens; heterogeneous language and answer types remain.']}
    (REPORT/'knowledge_pair_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 知识干预的配对负对照与不确定性','',
           '只读已完成逐题输出。差值=相关知识−对应控制；EM/F1越大越好，NLL越小越好。'
           '按原图像路径组配对bootstrap一万次，区间是探索性95%分位区间，未做多重比较校正。', '',
           '| 相关知识组 | 控制 | 评测组 | QA/图像组 | ΔEM [95%] | ΔF1 [95%] | ΔNLL [95%] | Δ回答字符数 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(m):
        lo,hi=m['image_cluster_percentile95'];return f'{m["mean"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'
    for row in rows:
        m=row['metrics'];n=m['nll']
        lines.append(f'| {row["treatment"]} | {row["control"]} | {row["group"]} | {n["QA"]}/{n["image_clusters"]} | {fmt(m["exact_match"])} | {fmt(m["token_f1"])} | {fmt(n)} | {m["prediction_characters"]["mean"]:+.2f} |')
    lines+=['','相关知识优于“无上下文”不能单独证明事实正确性的作用；还需同时检查低相关与错配控制。'
            '同一图像多题不是独立样本，医学100与KG267亦不能合并为367题。区间不能替代更多种子／独立资料验证。', '', '限制：','']
    lines+=['- '+x for x in result['limitations']]
    (REPORT/'KNOWLEDGE_PAIRED_EVIDENCE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed_pair_groups':len(rows),'completed_cases':len(cache)}))


if __name__=='__main__':main()
