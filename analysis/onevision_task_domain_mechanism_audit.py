"""CPU-only paired transitions, exposure accounting and saved gradient audit.

Historical default-backend runs remain exploratory, never relabeled matched.
"""
import json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_task_domain_intervention import METHODS,SEQUENCES,project
from onevision_knowledge_intervention import case_complete
from onevision_research_worker import digest
from onevision_knowledge_pair_audit import paired_cluster
from run import freeze
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def transitions(first,last):
    old=[x['scores']['exact_match']==100 for x in first];new=[x['scores']['exact_match']==100 for x in last]
    return {'correct_both':sum(a and b for a,b in zip(old,new)),
            'correct_then_wrong':sum(a and not b for a,b in zip(old,new)),
            'wrong_then_correct':sum(not a and b for a,b in zip(old,new)),
            'wrong_both':sum(not a and not b for a,b in zip(old,new))}


def main():
    torch.set_num_threads(4)
    # A genuine CPU optimizer counterexample: Euclidean gradient orthogonality
    # alone is not a first-order old-loss guarantee for Adam's preconditioner.
    toy=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));toy.grad=torch.tensor([2.,1.],dtype=torch.float64)
    optimizer=torch.optim.AdamW([toy],lr=1e-4,weight_decay=0);optimizer.step()
    assert float(torch.dot(toy.detach(),torch.tensor([1.,-2.],dtype=torch.float64)))>0
    adam_counterexample={'new_gradient':[2,1],'old_gradient':[1,-2],'gradient_dot':0,
                         'actual_CPU_Adam_first_delta':toy.detach().tolist(),'old_gradient_dot_actual_delta':float((toy[0]-2*toy[1]).detach())}
    del optimizer,toy
    manifest,data,domains,tests,*_=freeze();anchor=[x for x in tests['medical'] if x['_task']==4]
    keys=[tuple(x['images']) for x in anchor];assert len(anchor)==23
    info_path=ROOT/'summary/onevision_mechanisms_20261016/task_domain_attempt.json'
    info=json.loads(info_path.read_text());run=ROOT/info['run'];sources={str(info_path.relative_to(ROOT)):digest(info_path)}
    for relative,sha in info['spec']['source_sha256'].items():
        assert digest(ROOT/relative)==sha;sources[relative]=sha
    sources['analysis/onevision_knowledge_pair_audit.py']=digest(ROOT/'analysis/onevision_knowledge_pair_audit.py')
    cases={};rows=[]
    def read(path):sources[str(path.relative_to(ROOT))]=digest(path);return json.loads(path.read_text())
    for sequence in SEQUENCES:
        for method in METHODS:
            case=sequence+'__'+method;done=read(run/'cases'/(case+'_complete.json'))
            assert done['complete'] and done['source_sha256']==info['spec']['source_sha256']
            scores={}
            for stage in [1,3,5]:
                label=f'{case}_after{stage}';assert case_complete(run,label,{'anchor':anchor})
                path=run/'evaluations'/label/'anchor_image.jsonl';sources[str(path.relative_to(ROOT))]=digest(path)
                scores[stage]=[json.loads(line) for line in path.read_text().splitlines()]
                assert [x['id'] for x in scores[stage]]==[x['id'] for x in anchor]
            cases[case]=scores
            exposures={'new_QA':0,'replay_QA':0,'reference_gradient_QA':0,'fisher_QA':0,'optimizer_updates':0,'target_tokens_including_replay':0,'negative_anchor_dots':0,'projection_flags':0}
            raw_audits=[]
            for stage in range(1,6):
                training=read(run/'training'/case/f'stage_{stage:02d}.json');events=training['events'];assert len(events)==4
                expected=set(info['spec']['train_ids'][str(SEQUENCES[sequence][stage-1])])
                assert {qid for event in events for qid in event['train_ids']}==expected
                for event in events:
                    assert event['lr_used']>0
                    exposures['new_QA']+=len(event['train_ids']);exposures['replay_QA']+=len(event['replay_ids'])
                    exposures['reference_gradient_QA']+=event['reference_extra_QA'];exposures['optimizer_updates']+=1
                    exposures['target_tokens_including_replay']+=event['target_tokens']
                    dot=event['anchor_gradient_dot'];exposures['negative_anchor_dots']+=dot is not None and dot<0
                    exposures['projection_flags']+=event['projection_applied']
                    assert event['projection_applied']==(method=='anchor_gradient_projection' and dot is not None and dot<0)
                exposures['fisher_QA']+=training['fisher_extra_qa_next']
                if method.startswith('anchor_gradient_') and stage>1:
                    path=run/'raw_step_gradients'/case/f'stage_{stage:02d}.pt';meta=read(path.with_suffix('.json'));assert digest(path)==meta['sha256']
                    sources[str(path.relative_to(ROOT))]=digest(path)
                    saved=torch.load(path,map_location='cpu',weights_only=True);current=saved['pre_intervention_gradient'];reference=saved['reference_gradient']
                    assert torch.isfinite(current).all() and torch.isfinite(reference).all()
                    projected,dot=project(current,reference);before=float(torch.dot(current.double(),reference.double())/(current.double().norm()*reference.double().norm()))
                    after=float(torch.dot(projected.double(),reference.double())/(projected.double().norm()*reference.double().norm()))
                    # Long flat FP32 CPU reductions lose precision here; compare
                    # the FP64 reference, without changing training or tolerance.
                    expected_norm=float((projected if method=='anchor_gradient_projection' else current).double().norm())
                    assert math.isclose(dot,events[0]['anchor_gradient_dot'],rel_tol=1e-5,abs_tol=1e-7)
                    assert math.isclose(expected_norm,events[0]['gradient_l2_preclip'],rel_tol=1e-5,abs_tol=1e-6)
                    actual_gradient=projected if method=='anchor_gradient_projection' else current
                    clipped=actual_gradient.double()*min(1.,1./(expected_norm+1e-6))
                    adam_delta=-1e-4*clipped/(clipped.abs()+1e-8)
                    stage_states=[]
                    for number in [stage-1,stage]:
                        checkpoint=run/'adapters'/case/f'stage_{number:02d}.pt';checksum=read(checkpoint.with_suffix('.json'))['sha256']
                        assert digest(checkpoint)==checksum;sources[str(checkpoint.relative_to(ROOT))]=checksum
                        stage_states.append(torch.load(checkpoint,map_location='cpu',weights_only=True)['adapter'])
                    deltas=[]
                    for name,shape in saved['parameter_layout']:
                        key,factor,_=name.rsplit('.',2);assert factor in ['A','B']
                        assert list(stage_states[0][key][factor].shape)==shape
                        deltas.append((stage_states[1][key][factor].double()-stage_states[0][key][factor].double()).flatten())
                    stage_delta=torch.cat(deltas);assert stage_delta.numel()==reference.numel()
                    raw_audits.append({'stage':stage,'pre_cosine':before,'reconstructed_projection_cosine':after,
                                       'expected_norm_matches_saved_optimizer_gradient_norm':True,'saved_reference_TRAIN_ids':saved['reference_train_ids'],
                                       'hypothetical_SGD_first_old_linear_change':float(torch.dot(-1e-4*clipped,reference.double())),
                                       'reconstructed_Adam_first_old_linear_change':float(torch.dot(adam_delta,reference.double())),
                                       'actual_four_update_stage_delta_old_linear_term':float(torch.dot(stage_delta,reference.double()))})
            assert exposures['new_QA']==40 and exposures['optimizer_updates']==20
            expected_extra=(32 if method=='ER_LoRA' else 128 if method.startswith('anchor_gradient_') else 20 if method.startswith('EWC_') else 0)
            assert exposures['replay_QA']+exposures['reference_gradient_QA']+exposures['fisher_QA']==expected_extra
            delta={metric:paired_cluster([b['nll']-a['nll'] if metric=='nll' else b['scores'][metric]-a['scores'][metric]
                                         for a,b in zip(scores[1],scores[5])],keys) for metric in ['nll','exact_match','token_f1']}
            rows.append({'sequence':sequence,'method':method,'S1_to_S5_transitions':transitions(scores[1],scores[5]),
                         'within_case_S5_minus_S1':delta,'exposures':exposures,'first_minibatch_gradient_audits':raw_audits})
    contrasts=[]
    for sequence in SEQUENCES:
        for treatment,control in [('ER_LoRA','SeqFT_LoRA'),('anchor_gradient_projection','anchor_gradient_noop'),('EWC_lambda100','EWC_lambda1')]:
            a=cases[sequence+'__'+treatment];b=cases[sequence+'__'+control]
            metrics={}
            for metric in ['nll','exact_match','token_f1']:
                def value(row):return row['nll'] if metric=='nll' else row['scores'][metric]
                endpoint=[value(x)-value(y) for x,y in zip(a[5],b[5])]
                difference=[value(x5)-value(x1)-value(y5)+value(y1) for x1,x5,y1,y5 in zip(a[1],a[5],b[1],b[5])]
                metrics[metric]={'endpoint':paired_cluster(endpoint,keys),'difference_in_changes':paired_cluster(difference,keys)}
            contrasts.append({'sequence':sequence,'treatment':treatment,'control':control,'metrics':metrics})
    sequence_contrasts=[]
    for method in METHODS:
        for treatment,control in [('within_medical','repeat_anchor'),('cross_short','within_medical'),
                                  ('cross_five_domains','within_medical'),('cross_five_domains','cross_short')]:
            a=cases[treatment+'__'+method];b=cases[control+'__'+method];metrics={}
            for metric in ['nll','exact_match','token_f1']:
                def value(row):return row['nll'] if metric=='nll' else row['scores'][metric]
                delta=[value(x5)-value(x1)-value(y5)+value(y1) for x1,x5,y1,y5 in zip(a[1],a[5],b[1],b[5])]
                metrics[metric]=paired_cluster(delta,keys)
            sequence_contrasts.append({'method':method,'treatment':treatment,'control':control,'difference_in_changes':metrics})
    result={'run':info['run'],'audit_sha256':digest(Path(__file__)),'source_sha256':sources,'rows':rows,'contrasts':contrasts,
            'sequence_contrasts':sequence_contrasts,'Adam_counterexample':adam_counterexample,
            'scope':'24 completed default-backend controls,23 original anchor QA; not full-test CL metrics or strict causal proof',
            'limitations':['Stage1 weights differ across cases despite same seed; subtracting S1 does NOT remove this numerical confound.',
                           'One weakly acquired Organ anchor, single order/seed; few initial correct answers make retention fractions unstable.',
                           'Task/domain interventions also change content, target lengths and visual inputs.',
                           'ER has extra SGD replay; Fisher/projection have different extra compute. Only projection/noop share reference-read counts.',
                           'Raw gradient audit covers first minibatch at stages2-5 only; reconstructed projection is not a saved postprojection vector.',
                           'Read-only CPU norms use FP64 after flat FP32 reduction mismatch; saved training gradients and original model reproducibility gates unchanged.',
                           'Reconstructed first Adam step is an idealized fresh-optimizer formula; actual four-step displacement linear term is Taylor evidence, not measured loss or a saved first-step update.',
                           'Bootstrap uses image-path groups, exploratory95% intervals, no multiple-testing correction.',
                           'Correct-to-wrong transition is reference-EM loss, not a proof of semantic/clinical knowledge deletion.',
                           'Sparse anchor endpoints cannot provide full benchmark AP/FWT/BWT or historical-peak ForgetRate.']}
    (REPORT/'task_domain_mechanism_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    def interval(value):
        lo,hi=value['image_cluster_percentile95'];return f'{value["mean"]:+.3f} [{lo:+.3f},{hi:+.3f}]'
    lines=['# 任务／领域控制：逐题转移、额外计算与梯度干预','',result['scope'], '',
           '同一23题的S1→S5拆分：早期正确后错误、早期错误后正确分别记录。净分上升不等于没有旧题退化；分母很小，不将两三题的保持率泛化。', '',
           '| 序列 | 机制 | 始终正确 | 正确→错误 | 错误→正确 | 始终错误 | ΔEM [95%] | ΔNLL [95%] |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        t=row['S1_to_S5_transitions'];d=row['within_case_S5_minus_S1']
        lines.append(f'| {row["sequence"]} | {row["method"]} | {t["correct_both"]} | {t["correct_then_wrong"]} | {t["wrong_then_correct"]} | {t["wrong_both"]} | {interval(d["exact_match"])} | {interval(d["nll"])} |')
    lines+=['','## 额外数据读取与计算','',
            '每个方法均为40次当前QA暴露、20更新。ER额外32次回放SGD；EWC额外20次Fisher QA；投影及no-op均额外128次锚QA梯度计算。'
            '后两者的读取次数相同，可以检查投影操作，但默认数值与原型小样本混杂仍存在。40次重复锚只含8个不同QA，不能写成40个新样本。', '',
            '| 序列 | 干预−控制 | ΔEM变化差 [95%] | ΔNLL变化差 [95%] |','|---|---|---:|---:|']
    for c in contrasts:
        lines.append(f'| {c["sequence"]} | {c["treatment"]}−{c["control"]} | {interval(c["metrics"]["exact_match"]["difference_in_changes"])} | {interval(c["metrics"]["nll"]["difference_in_changes"])} |')
    lines+=['','变化差=(干预S5−S1)−(控制S5−S1)。它缓解起始分数不一致的展示问题，不能修复起始权重不一致或非线性训练路径差异。', '',
            '## 固定任务数下的序列对比','',
            '以下为SeqFT；其余机制的全部同类对比见JSON。跨短序列不含交通长答案，但内容／视觉输入仍不同，不能将差异纯归为领域数量。', '',
            '| 序列−参照 | ΔEM变化差 [95%] | ΔNLL变化差 [95%] |','|---|---:|---:|']
    for contrast in sequence_contrasts:
        if contrast['method']=='SeqFT_LoRA':
            metrics=contrast['difference_in_changes'];lines.append(f'| {contrast["treatment"]}−{contrast["control"]} | {interval(metrics["exact_match"])} | {interval(metrics["nll"])} |')
    lines+=['',
            '## 原始TRAIN梯度的核验','',
            '仅阶段2–5第一个minibatch有原始当前／参考梯度：CPU重算点积、投影及实际优化器记录的梯度范数，所有32个检查通过。'
            'JSON保留各阶段原余弦与重建投影余弦、负点积／投影频率、目标token数与额外QA读取。重建向量不是历史保存的投影后向量，负余弦本身也不是最终遗忘的因果证明。', '',
            '补充优化器限制：真实CPU Adam的二维反例确认，梯度点积为0时其参数更新仍可能使旧梯度一阶项为正。'
            '因此原投影控制只能称Euclidean原始梯度投影，不能声称Adam实际更新必保旧损失。'
            '实际四步参数位移的一阶项与首步Adam公式重建均记录，前者不是实测loss变化，后者不是历史保存的单步位移。', '', '限制：','']
    lines+=['- '+x for x in result['limitations']]
    (REPORT/'TASK_DOMAIN_MECHANISM_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete_cases':len(rows),'raw_gradient_checks':sum(len(r['first_minibatch_gradient_audits']) for r in rows),
                      'SeqFT_transitions':{r['sequence']:r['S1_to_S5_transitions'] for r in rows if r['method']=='SeqFT_LoRA'}}))


if __name__=='__main__':main()
