"""Read completed inference controls; distinguish visual signal from decisions.

All candidate labels were fixed from TRAIN. This does not select/tune a model,
replace primary generation scores, or treat 100 nativeQA as full five-domain test.
"""
import collections,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_knowledge_pair_audit import paired_cluster
from onevision_research_worker import digest
from run import freeze,score
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def read(p):return json.loads(p.read_text())
def records(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def margin(row,wrapped=False):
    prefix='The answer is ' if wrapped else '';c=row['candidates']
    return c[prefix+'rural']['sum_nll_with_EOS']-c[prefix+'urban']['sum_nll_with_EOS']


def main():
    complete=REPORT/'rural_collapse/complete.json'
    if not complete.exists():print('Pending: actual GPU study completion proof not available; no results published.');return
    proof=read(complete);spec=proof['spec'];run=ROOT/proof['run'];sources={str(complete.relative_to(ROOT)):digest(complete),str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}
    for p,h in spec['source_sha256'].items():assert digest(ROOT/p)==h
    for rel,h in proof['artifact_sha256'].items():assert digest(run/rel)==h;sources[str((run/rel).relative_to(ROOT))]=h
    for meta in spec['checkpoints'].values():assert digest(ROOT/meta['path'])==meta['sha256']
    _,data,_,_,_,_,_=freeze();groups={'TRAIN':data[12,'train'],'TEST':data[12,'test']}
    assert len({(r['question'],r.get('context'),r['_task']) for part in groups.values() for r in part})==1
    output=[];loaded={};visual=[];replication=[]
    for case in spec['checkpoints']:
        for split,original in groups.items():
            keys=[tuple(r['images']) for r in original]
            for view in spec['views']:
                p=run/f'{case}_{split}_{view}.jsonl';rows=records(p);loaded[case,split,view]=rows
                assert [r['id'] for r in rows]==spec['ids'][split]==[r['id'] for r in original]
                assert all(r['references']==o['answers'] for r,o in zip(rows,original))
                assert all(r['visual_tokens']==0 if view=='text' else r['visual_tokens']>0 for r in rows)
                if view=='wrong':assert all(r['wrong_image_id']==spec['wrong_image_ids'][split][r['id']] and r['wrong_image_id']!=r['id'] for r in rows)
                margins=[margin(r) for r in rows];wrapped=[margin(r,True) for r in rows]
                aux=0
                for row in rows:
                    text=row['prediction'].strip()
                    if text.lower().startswith('the answer is '):text=text[len('the answer is '):].strip()
                    aux+=score(text,row['references'])['exact_match']==100
                if view=='text':assert max(margins)-min(margins)<1e-5,'Identical text input candidate margins vary; inspect numerical/input gate'
                correct=lambda m,o:('urban' if m>0 else 'rural')==o['answer']
                output.append({'checkpoint':case,'split':split,'view':view,'QA':len(rows),
                               'primary_EM_correct':sum(r['scores']['exact_match']==100 for r in rows),'fixed_prefix_aux_correct':aux,
                               'gold_mean_NLL':float(np.mean([r['gold_nll'] for r in rows])),
                               'candidate_bare_correct':sum(correct(m,o) for m,o in zip(margins,original)),
                               'candidate_wrapped_correct':sum(correct(m,o) for m,o in zip(wrapped,original)),
                               'bare_margin_mean':float(np.mean(margins)),'bare_margin_std':float(np.std(margins)),
                               'bare_margin_min':min(margins),'bare_margin_max':max(margins),
                               'bare_margin_ties':sum(m==0 for m in margins),'prediction_counts':dict(collections.Counter(r['prediction'] for r in rows))})
            image,wrong,text=[loaded[case,split,v] for v in ['image','wrong','text']]
            signed=lambda row,o:margin(row)*(1 if o['answer']=='urban' else -1)
            visual.append({'checkpoint':case,'split':split,
                           'wrong_minus_image_gold_NLL':paired_cluster([b['gold_nll']-a['gold_nll'] for a,b in zip(image,wrong)],keys),
                           'image_minus_wrong_target_signed_margin':paired_cluster([signed(a,o)-signed(b,o) for a,b,o in zip(image,wrong,original)],keys),
                           'image_minus_wrong_primary_EM_pp':paired_cluster([a['scores']['exact_match']-b['scores']['exact_match'] for a,b in zip(image,wrong)],keys),
                           'text_minus_image_gold_NLL':paired_cluster([b['gold_nll']-a['gold_nll'] for a,b in zip(image,text)],keys)})
    basis_root=ROOT/read(REPORT/'basis_budget/complete.json')['run']
    for start in ['random_A_zero_B','past_AB']:
        old_case=basis_root/f'T12_{start}';old_proof=read(old_case/'complete.json')
        for update in [1,4]:
            p=old_case/f'evaluations/endpoint_{update}/native_image.jsonl'
            assert digest(p)==old_proof['artifact_sha256'][str(p.relative_to(old_case))];sources[str(p.relative_to(ROOT))]=digest(p)
            old=records(p);new=loaded[f'{start}_update{update}','TEST','image']
            assert [r['id'] for r in old]==[r['id'] for r in new]
            exact=all(a['prediction']==b['prediction'] and a['scores']==b['scores'] for a,b in zip(old,new))
            difference=max(abs(a['nll']-b['gold_nll']) for a,b in zip(old,new))
            assert exact and difference<.015,'Same-checkpoint image reference not reproduced; do not mix interpretations'
            replication.append({'start':start,'updates':update,'QA':len(new),'predictions_and_scores_exact':exact,'max_NLL_error':difference})
    trajectory=[]
    for start in ['random_A_zero_B','past_AB']:
        for split,original in groups.items():
            a,b=[loaded[f'{start}_update{step}',split,'image'] for step in [1,4]]
            ma,mb=[np.array([margin(r) for r in rows]) for rows in [a,b]]
            trajectory.append({'start':start,'split':split,
                               'bare_margin_shift_4_minus_1':paired_cluster((mb-ma).tolist(),[tuple(r['images']) for r in original]),
                               'bare_margin_shift_std':float(np.std(mb-ma)),
                               'bare_margin_correlation_1_4':float(np.corrcoef(ma,mb)[0,1]) if np.std(ma)>0 and np.std(mb)>0 else None,
                               'bare_margin_class_means':{label:{'update1':float(np.mean([m for m,o in zip(ma,original) if o['answer']==label])),
                                                               'update4':float(np.mean([m for m,o in zip(mb,original) if o['answer']==label]))} for label in ['urban','rural']},
                               'same_weight_endpoint_TRAIN_or_TEST_NLL_4_minus_1':paired_cluster([y['gold_nll']-x['gold_nll'] for x,y in zip(a,b)],[tuple(r['images']) for r in original]),
                               'primary_EM_pp_4_minus_1':paired_cluster([y['scores']['exact_match']-x['scores']['exact_match'] for x,y in zip(a,b)],[tuple(r['images']) for r in original])})
    result={'sources_sha256':sources,'summaries':output,'visual_paired':visual,'endpoint_changes':trajectory,'image_reference_replication':replication,
            'candidate_definition':'logP(urban)-logP(rural) over full candidate sequence inclEOS; positives urban, ties rural; auxiliary two-candidate decision, not free-generation accuracy or calibrated visual understanding',
            'scope':'all original native T12 TEST100 and frozen TRAIN6; only five fixed checkpoints, no optimizer',
            'limitations':['post-hoc follow-up after observed collapse, single seed/task/trajectory',
                          'same questions/contexts/instructions; variable image margin indicates image-conditioned distribution, not semantic visual reasoning',
                          'wrong paths/pixels distinct but may share class or semantics; text view also changes token positions',
                          'conditional candidate choices are auxiliary; no test answers used to select weights/router/budget',
                          'mean per-QA NLL and complete candidate sequence likelihood differ in weighting; raw lengths preserved',
                          'bootstrap image-path clusters10000 seed42, no multiple-comparison correction']}
    (REPORT/'RURAL_COLLAPSE_EVIDENCE.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    out=['# 原权重视觉／训练拟合对照：完成证据','',result['scope'],
         '', '全部输出／参数／源码SHA核验；四个既有终点的原图TEST生成与原基底预算结果逐题复核，400条prediction和scores必须完全相同，NLL误差单列。相同文字输入的候选margin常量门槛通过。无新训练。','',
         '| 权重 | split | 视图 | 主EM | 固定前缀辅助 | 两裸候选辅助 | gold NLL | 裸候选margin均值／标准差 |',
         '| --- | --- | --- | ---: | ---: | ---: | ---: | --- |']
    for row in output:
        out.append(f"| {row['checkpoint']} | {row['split']} | {row['view']} | {row['primary_EM_correct']}/{row['QA']} | {row['fixed_prefix_aux_correct']}/{row['QA']} | {row['candidate_bare_correct']}/{row['QA']} | {row['gold_mean_NLL']:.4f} | {row['bare_margin_mean']:+.4f} / {row['bare_margin_std']:.4f} |")
    out+=['','两候选概率只来自TRAIN固定的rural/urban，按完整字符串含EOS计算；不能替代原自由生成分数，不是医学／金融等其他领域的泛化成绩。margin跨图变化本身不证明正确视觉理解；需要与确定性错图对照，保留区间。','',
          '## 原图相对错图：配对差','', '| 权重 | split | 错图−原图gold NLL [95%] | 原图−错图目标带符号margin [95%] |', '| --- | --- | --- | --- |']
    def show(v):return f"{v['mean']:+.4f} [{v['image_cluster_percentile95'][0]:+.4f}, {v['image_cluster_percentile95'][1]:+.4f}]"
    for row in visual:out.append(f"| {row['checkpoint']} | {row['split']} | {show(row['wrong_minus_image_gold_NLL'])} | {show(row['image_minus_wrong_target_signed_margin'])} |")
    out+=['','## 相同终点权重下训练拟合与测试变化','', '| 起点 | split | NLL4−NLL1 [95%] | 主EM4−EM1 pp [95%] |', '| --- | --- | --- | --- |']
    for row in trajectory:out.append(f"| {row['start']} | {row['split']} | {show(row['same_weight_endpoint_TRAIN_or_TEST_NLL_4_minus_1'])} | {show(row['primary_EM_pp_4_minus_1'])} |")
    out+=['','这里TRAIN NLL为真实终点1／4下重新评测，区别于原训练日志的pre-update loss。仅6条训练QA，配对区间不是独立训练seed方差。若训练NLL下降、测试退化可定位目标拟合与泛化差，但不足以唯一归因于样本比例；没有对标签比例或优化器做新的因果干预。',
          '', '## 坍缩不等于图像信号消失',
          '', '旧AB的4步TEST输出全部urban，但原图相对错图仍降低gold NLL，目标带符号margin也更好。必须区分图像对答案分布仍有信息与自由生成决策是否正确。',
          '', '1→4步裸候选margin的整体偏移、逐图差值标准差、跨图相关性及按金标类均值见JSON。正值偏urban；这是事后机制定位，不是校准器或测试调参。不能仅凭整体偏移唯一归因于训练类比例，或断言视觉能力完全不变。完整字符串含EOS且两裸候选长度不同，wrapped候选的分数另外保留，不能择优替换主指标。',
          '', '输出频率、bare/wrapped候选、全部配对区间、参考复现误差与SHA见[完整JSON](RURAL_COLLAPSE_EVIDENCE.json)。']
    (REPORT/'RURAL_COLLAPSE_EVIDENCE.md').write_text('\n'.join(out)+'\n')
    print(json.dumps({'completed':True,'cases':len(spec['checkpoints']),'summary_cells':len(output),'replicated_original_generations':400}))


if __name__=='__main__':main()
