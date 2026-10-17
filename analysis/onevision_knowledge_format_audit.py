"""CPU-only fixed-prefix sensitivity and auxiliary fact-recall audit.

No generated text/weights/data/main scores are changed. A single literal wrapper
rule is fixed across all controls, never selected separately against test gold.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_knowledge_intervention import case_complete
from onevision_knowledge_pair_audit import paired_cluster
from onevision_research_worker import digest
from run import freeze,score
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def read(path):return json.loads(path.read_text())
def records(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def auxiliary(row):
    text=row['prediction'].strip();wrapped=text.lower().startswith('the answer is ')
    if wrapped:text=text[len('the answer is '):].strip()
    refs=row.get('references') or [row['reference']]
    return score(text,refs),wrapped


def summarize(rows):
    aux=[auxiliary(row) for row in rows]
    return {'QA':len(rows),'original_EM_correct':sum(r['scores']['exact_match']==100 for r in rows),
            'fixed_prefix_aux_EM_correct':sum(a['exact_match']==100 for a,_ in aux),
            'prefix_count':sum(w for _,w in aux),'mean_NLL':sum(r['nll'] for r in rows)/len(rows),
            'original_token_F1':sum(r['scores']['token_f1'] for r in rows)/len(rows),
            'fixed_prefix_aux_token_F1':sum(a['token_f1'] for a,_ in aux)/len(rows)}


def main():
    _,data,_,tests,_,_,protocol=freeze()
    clinical_proof=ROOT/'summary/onevision_mechanisms_20261016/knowledge_complete.json'
    clinical=ROOT/read(clinical_proof)['run']
    encoding_proof=REPORT/'capacity/knowledge_sanity_complete.json'
    encoding_info=read(encoding_proof);encoding=ROOT/encoding_info['run']
    sources={str(Path(__file__).relative_to(ROOT)):digest(Path(__file__)),
             str(clinical_proof.relative_to(ROOT)):digest(clinical_proof),
             str(encoding_proof.relative_to(ROOT)):digest(encoding_proof)}
    selected={'medical':tests['medical'],'medical_KG':data[11,'test']}
    families=['context_base','context_shared','residual_rank1','residual_rank2','continue_rank8']
    names=[f'{family}_{kind}' for family in families for kind in (['none','correct','unrelated','permuted'] if family.startswith('context') else ['correct','unrelated','permuted'])]
    cases={};clinical_summary=[]
    for name in names:
        assert case_complete(clinical,name,selected)
        cases[name]={}
        for group,expected in selected.items():
            path=clinical/f'evaluations/{name}/{group}_image.jsonl';rows=records(path)
            proof=read(path.parent/'durable_complete.json')
            assert digest(path)==proof['files'][path.name]
            assert [r['id'] for r in rows]==[r['id'] for r in expected]
            sources[str(path.relative_to(ROOT))]=digest(path);cases[name][group]=rows
            clinical_summary.append({'case':name,'group':group,**summarize(rows)})
    assert len(cases)==17
    comparisons=[]
    for family in families:
        treatment=family+'_correct'
        for control in (['none','unrelated','permuted'] if family.startswith('context') else ['unrelated','permuted']):
            control=family+'_'+control
            for group,expected in selected.items():
                a,b=cases[treatment][group],cases[control][group]
                keys=[tuple(r['images']) for r in expected];statistics={}
                for kind in ['original','fixed_prefix_aux']:
                    def value(row):return row['scores']['exact_match'] if kind=='original' else auxiliary(row)[0]['exact_match']
                    statistics[kind]=paired_cluster([value(x)-value(y) for x,y in zip(a,b)],keys)
                comparisons.append({'treatment':treatment,'control':control,'group':group,'EM_pp':statistics})
    encoding_summary=[]
    for path in sorted((encoding/'evaluations').glob('*/*_text.jsonl')):
        assert digest(path)==encoding_info['artifact_sha256'][str(path.relative_to(encoding))]
        sources[str(path.relative_to(ROOT))]=digest(path);rows=records(path)
        for template in ['original','paraphrase']:
            part=[r for r in rows if r['template']==template];assert len(part)==16
            encoding_summary.append({'case':path.parent.name,'cohort':path.stem.removesuffix('_text'),
                                     'template':template,**summarize(part)})
    result={'manifest_sha256':protocol['source_manifest_sha256'],'source_sha256':sources,
            'rule':'remove one leading literal The answer is (case-insensitive); apply unchanged original scorer; no other wrapper or semantic parsing',
            'clinical_cases':clinical_summary,'clinical_paired':comparisons,'encoding_fact_recall':encoding_summary,
            'limitations':['post-hoc auxiliary metric, not a replacement benchmark score or training objective',
                          'medical100 and nativeKG267 overlap12; never pool as367 independent QA',
                          'same16 fact original/paraphrase are recall, not32 independent facts or downstream generalization',
                          'low-relevance controls remain anatomy and context lengths differ',
                          'zero exact recall does not prove no partial/semantic knowledge; fixed prefix cannot solve aliases',
                          'original auxiliary training8 updates/64 exposures; no claim sufficient budget or successful knowledge write',
                          'image-cluster percentile intervals10000/seed42, no multiple-comparison correction']}
    (REPORT/'KNOWLEDGE_FORMAT_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    out=['# 知识干预：格式敏感性与事实写入检查','',
         '17/17原知识干预、事实回忆输出均通过完成证明与SHA验证；CPU只读，无新训练。固定单一The answer is前缀规则只作事后辅助，不修改主分数或数据。所有上下文对照使用原实验统一both提示，不混用其他prompt的基座结果。','',
         '## 医学100题：原分数上升未必是知识收益','',
         '| 干预 | 原EM正确 | 固定前缀辅助正确 | 前缀输出数 | NLL |', '| --- | ---: | ---: | ---: | ---: |']
    for row in clinical_summary:
        if row['group']=='medical':out.append(f"| {row['case']} | {row['original_EM_correct']}/100 | {row['fixed_prefix_aux_EM_correct']}/100 | {row['prefix_count']} | {row['mean_NLL']:.3f} |")
    out+=['','基座none→correct原EM为0→9，但去固定前缀的辅助正确数为24→11。共享LoRA none→correct原EM8→21，辅助15→21；低相关同为21，错配为22。正确事实带来的特异性生成收益仍未被这些对照确证。不能把原EM差全部当知识收益，也不能反过来认定知识一定无用。','',
          '## 原生KG267题：同样保留负对照与格式限制','', '| 干预 | 原EM正确 | 固定前缀辅助正确 | 前缀输出数 |', '| --- | ---: | ---: | ---: |']
    for row in clinical_summary:
        if row['group']=='medical_KG':out.append(f"| {row['case']} | {row['original_EM_correct']}/267 | {row['fixed_prefix_aux_EM_correct']}/267 | {row['prefix_count']} |")
    out+=['','## 正确事实与负对照的配对区间','',
          '| 正确事实干预 | 对照 | 组 | 原EM差95%区间（pp） | 辅助EM差95%区间（pp） |',
          '| --- | --- | --- | --- | --- |']
    for row in comparisons:
        a=row['EM_pp']['original'];b=row['EM_pp']['fixed_prefix_aux']
        def show(v):return f"{v['mean']:+.2f} [{v['image_cluster_percentile95'][0]:+.2f}, {v['image_cluster_percentile95'][1]:+.2f}]"
        out.append(f"| {row['treatment']} | {row['control']} | {row['group']} | {show(a)} | {show(b)} |")
    out+=['','## 事实原模板：去前缀仍未得到严格回忆成功','',
          '| 干预 | 参照事实 | 原EM | 辅助EM | NLL |', '| --- | --- | ---: | ---: | ---: |']
    for row in encoding_summary:
        if row['template']=='original':out.append(f"| {row['case']} | {row['cohort']} | {row['original_EM_correct']}/16 | {row['fixed_prefix_aux_EM_correct']}/16 | {row['mean_NLL']:.3f} |")
    out+=['','rank1／rank2／rank8正确事实LoRA对原16训练模板，原评分及仅去前缀评分均0/16；对应同事实固定改写也均0/16。未证明写入成功不等于证明知识不存在：别名、中文长答案的部分语义回忆及训练预算仍是解释。至少“只差这个英文前缀”不能解释其严格回忆失败。上下文正确事实shared原模板为4/16、改写为2/16，说明本测试也非必然全零。',
          '', '下一步优先核验训练干预本身是否足量且能拟合原事实，再讨论对医学原生任务的特异收益；不修改旧事实池，不按测试答案挑事实／rank／预算，不先设计新方法。',
          '', '原医学100题和KG267题有12题交集；原模板与改写共享16事实，均不得视作独立扩样。完整逐条件统计、配对区间及文件SHA见[JSON](KNOWLEDGE_FORMAT_AUDIT.json)。']
    (REPORT/'KNOWLEDGE_FORMAT_AUDIT.md').write_text('\n'.join(out)+'\n')
    print(json.dumps({'clinical_cases':len(cases),'paired_comparisons':len(comparisons),'fact_recall_cells':len(encoding_summary),'verified':True}))


if __name__=='__main__':main()
