"""CPU-only audit: matched native acquisition/retention and rural collapse.

Primary scores stay frozen. The separate rural prefix check uses only its two
TRAIN labels and a fixed literal wrapper, never test-answer substring matching.
"""
import collections,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis'))
from onevision_task_domain_mechanism_audit import transitions
from run import freeze,score
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def em(xs):return {'correct':sum(r['scores']['exact_match']==100 for r in xs),'total':len(xs)}
def avg(xs):return sum(r['nll'] for r in xs)/len(xs)


def prefix_prediction(text,labels):
    text=text.strip().lower().rstrip('.!?。！').strip()
    if text.startswith('the answer is '):text=text[len('the answer is '):].strip()
    return text if text in labels else None


def main():
    manifest,data,domains,tests,_,_,protocol=freeze()
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
             'exp/five_domain_native_cl_20261016_sparse/runtime_utils.py':sha(ROOT/'exp/five_domain_native_cl_20261016_sparse/runtime_utils.py')}
    proof_path=REPORT/'deterministic_native/complete.json';proof=read(proof_path);run=ROOT/proof['run']
    sources[str(proof_path.relative_to(ROOT))]=sha(proof_path)
    def checked(p,completion=proof,root=run):
        expected=completion['artifact_sha256'][str(p.relative_to(root))]
        assert sha(p)==expected;sources[str(p.relative_to(ROOT))]=expected;return rows(p)
    gate=run/'full_T01_reproduction_gate.json'
    assert read(gate)['exact_equal'];assert sha(gate)==proof['artifact_sha256'][str(gate.relative_to(run))]
    native=[]
    for task in manifest['tasks']:
        tid,domain=task['task'],task['domain']
        selected={}
        for name,label in [('base','base_initial'),('pre',f'pre_T{tid:02d}'),('isolated',f'matched_native_T{tid:02d}'),('acquired',f'sequential_T{tid:02d}'),('final','sequential_T19')]:
            selected[name]=[r for r in checked(run/f'evaluations/{label}/{domain}_image.jsonl') if r['native_task']==tid and 'scores' in r]
        reference=[r['id'] for r in selected['base']]
        assert reference and all([r['id'] for r in x]==reference for x in selected.values())
        scores={}
        for name,xs in selected.items():
            auxiliary_correct=prefix_count=0
            for row in xs:
                text=row['prediction'].strip()
                if text.lower().startswith('the answer is '):
                    prefix_count+=1;text=text[len('the answer is '):].strip()
                auxiliary_correct+=score(text,row['references'])['exact_match']==100
            scores[name]={'EM':em(xs),'mean_NLL_on_generated_QA':avg(xs),
                          'fixed_literal_prefix_auxiliary_correct':auxiliary_correct,'prefix_count':prefix_count}
        native.append({'task':tid,'domain':domain,'train_QA':len(data[tid,'train']),
                       'scores':scores,
                       'pre_to_acquired':transitions(selected['pre'],selected['acquired']),
                       'acquired_to_final':transitions(selected['acquired'],selected['final']),
                       'isolated_to_acquired':transitions(selected['isolated'],selected['acquired'])})
    assert sum(x['scores']['base']['EM']['total'] for x in native)==155
    total={name:{'correct':sum(r['scores'][name]['EM']['correct'] for r in native),'total':155,
                 'fixed_literal_prefix_auxiliary_correct':sum(r['scores'][name]['fixed_literal_prefix_auxiliary_correct'] for r in native),
                 'prefix_count':sum(r['scores'][name]['prefix_count'] for r in native)} for name in ['base','pre','isolated','acquired','final']}
    retention={k:sum(x['acquired_to_final'][k] for x in native) for k in native[0]['acquired_to_final']}
    rural_train=data[12,'train'];rural_test=data[12,'test'];labels=set(r['answer'].lower() for r in rural_train)
    assert labels=={'rural','urban'} and len(rural_train)==6 and len(rural_test)==100
    lookup={r['id']:r for r in rural_test};basis_root=ROOT/read(REPORT/'basis_budget/complete.json')['run']
    rural=[];preds={}
    for start in ['random_A_zero_B','past_A_zero_B','past_AB']:
        case=basis_root/f'T12_{start}';completion=read(case/'complete.json')
        p=case/'training.json';assert sha(p)==completion['artifact_sha256']['training.json']
        training=read(p);case_rows=[]
        for update in [1,4]:
            xs=checked(case/f'evaluations/endpoint_{update}/native_image.jsonl',completion,case)
            assert [r['id'] for r in xs]==[r['id'] for r in rural_test]
            preds[start,update]=xs;confusion={a:collections.Counter() for a in sorted(labels)}
            prefix_correct=0;unparsed=0
            for row in xs:
                target=lookup[row['id']]['answer'].lower();parsed=prefix_prediction(row['prediction'],labels)
                confusion[target][parsed or '<unparsed>']+=1;unparsed+=parsed is None;prefix_correct+=parsed==target
            case_rows.append({'updates':update,'primary_EM':em(xs),'mean_test_NLL':avg(xs),
                              'predictions':dict(collections.Counter(r['prediction'] for r in xs)),
                              'auxiliary_literal_prefix_correct':prefix_correct,'auxiliary_unparsed':unparsed,
                              'auxiliary_confusion':{k:dict(v) for k,v in confusion.items()}})
        rural.append({'start':start,'training_records':training['records'],'endpoints':case_rows,
                      'primary_transitions_1_to_4':transitions(preds[start,1],preds[start,4])})
    result={'manifest_sha256':protocol['source_manifest_sha256'],'sources':sources,'native_tasks':native,
            'native_generated_QA_totals':total,'native_acquired_to_final_transitions':retention,
            'rural':{'TRAIN_labels':dict(collections.Counter(r['answer'] for r in rural_train)),
                     'TEST_labels':dict(collections.Counter(r['answer'] for r in rural_test)),
                     'TRAIN_unique_images':len({tuple(r['images']) for r in rural_train}),
                     'TEST_unique_images':len({tuple(r['images']) for r in rural_test}),
                     'identical_question_all_TRAIN_TEST':len({r['question'] for r in rural_train+rural_test})==1,
                     'cases':rural,'auxiliary_rule':'lowercase, strip terminal punctuation; remove literal leading the answer is; accept only original TRAIN labels rural/urban; main EM unchanged'},
            'limitations':['155 generated QA vs500 NLL diagnostics, not full21454',
                          'native prior-history comparisons include more historical training exposures; not domain-count-only causal effect',
                          'training step losses are BEFORE the corresponding update, not endpoint1/4 matched-weight TRAIN evaluation',
                          'rural collapse is real, majority TRAIN bias is a hypothesis; no balancing or visual-causal intervention yet',
                          'literal-prefix auxiliary scoring is post-hoc and is not semantic judging or new primary accuracy']}
    (REPORT/'ACQUISITION_COLLAPSE_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    out=['# 初始习得、后续覆盖与多数答案坍缩审计','',
         'CPU只读；全部被读取预测及完成证明SHA核验，确定性完整T1逐元素复现门槛通过。原评分、训练数据、图像、权重保持不变。','',
         '## 确定性19原生任务：生成诊断155题','',
         '| 状态 | 原始答对／总数 | 仅去固定英文前缀的辅助正确数 | 含该前缀的输出数 |','| --- | ---: | ---: | ---: |']
    for name,v in total.items():out.append(f"| {name} | {v['correct']}/{v['total']} | {v['fixed_literal_prefix_auxiliary_correct']}/{v['total']} | {v['prefix_count']} |")
    out+=['','pre和acquired是每个任务各自的时点汇总，**不是一个统一模型快照**；isolated为19个独立模型，final为单一最终模型。所有当前任务共享冻结QA／训练步数，但历史训练量不同。不能据此把正迁移完全归于多领域而排除额外训练。',
          '', '固定规则仅删除一次开头The answer is，然后调用原评分器。基座同155题原EM为1，辅助为31；最终模型原／辅助均47。原始差46题，格式敏感性检查下差16题，提示不能把所有原EM增量直接解释为知识或视觉能力学习；这不是对训练增益的严格因果百分比拆分。规则在此为事后辅助，不替代主结果，不构成通用语义正确性判断。',
          '', f"acquired→final逐题转移：保留{retention['correct_both']}，丢失{retention['correct_then_wrong']}，新增{retention['wrong_then_correct']}，始终错误{retention['wrong_both']}。这是稀疏生成诊断的汇总，不是完整FWT/BWT。",'',
          '| 原生任务 | TRAIN | base | pre | isolated | acquired | final | 习得后丢失／新增 |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in native:
        s=row['scores'];tr=row['acquired_to_final'];nums=[f"{s[n]['EM']['correct']}/{s[n]['EM']['total']}" for n in ['base','pre','isolated','acquired','final']]
        out.append(f"| T{row['task']:02d} {row['domain']} | {row['train_QA']} | "+' | '.join(nums)+f" | {tr['correct_then_wrong']}/{tr['wrong_then_correct']} |")
    out+=['','## 遥感rural：训练loss下降，但输出坍缩为多数答案','',
          '冻结TRAIN仅6题（urban4/rural2），原TEST100题（urban44/rural56）；所有问题相同，图片不同。完整旧AB起点由1更新76/100降为4更新44/100，后者100题全部输出urban；等于只猜TRAIN多数类别的测试成绩。当前结果直接证明输出坍缩，**尚不能证明训练标签比例就是唯一原因**，也不意味着模型没有读取图像。',
          '', '| 起点 | 更新 | 原EM | 固定前缀辅助正确数 | 未解析 | 测试NLL |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for case in rural:
        for e in case['endpoints']:
            out.append(f"| {case['start']} | {e['updates']} | {e['primary_EM']['correct']}/100 | {e['auxiliary_literal_prefix_correct']}/100 | {e['auxiliary_unparsed']} | {e['mean_test_NLL']:.4f} |")
    out+=['','辅助规则仅去除固定英文前缀The answer is及末尾标点，使用TRAIN的rural/urban标签字典，不按测试答案搜索匹配；属于事后诊断，不能替换主EM。单更新随机A／旧A零B的0分不能直接等同于完全不会判断图像。',
          '', '旧AB训练记录loss为0.5561→0.3295→0.2616→0.2354，每个数均在对应更新前记录；它们不是终点1／4同权重的训练集评测，不能据此直接估计终点泛化差。下一步应在原冻结权重上补测6条TRAIN及原图／错图／无图，观察固定标签的输出概率和生成变化，不改训练池、不新增方法。',
          '', '## 可以排除与仍待验证','',
          '- “LoRA从未更新”不成立：先前因子审计已确认正LR和实际deltaW；本次观察到生成行为明确改变。不能据此证明参数更新都有效。',
          '- “所有0分都代表缺少知识”不成立：至少rural存在字面答案前缀导致原EM归零的情况；该任务不代表所有医学／金融题。',
          '- “多训练必定更好”不成立：同一旧AB轨迹出现生成坍缩与测试NLL上升。多数答案偏置、图像区分退化、概率阈值变化仍需分开。',
          '- “后续任务只带来遗忘”不能预设：原生对照同时存在保留、丢失与新增，需按任务报告，不能用总分遮蔽转移。',
          '', '完整计数／原始输出频率／混淆表／训练loss及梯度／文件SHA见[机器可读审计](ACQUISITION_COLLAPSE_AUDIT.json)。']
    (REPORT/'ACQUISITION_COLLAPSE_AUDIT.md').write_text('\n'.join(out)+'\n')
    print(json.dumps({'totals':total,'retention':retention,'rural':[{k:v for k,v in c.items() if k!='training_records'} for c in rural]},ensure_ascii=False))


if __name__=='__main__':main()
