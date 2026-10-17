"""Prevent interpreting a zero regularizer gradient as proof of harmful updates.

Read existing saved-gradient analysis, revalidate relevant artifacts, distinguish
local Taylor predictions from measured losses/accuracy. No GPU or training.
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'analysis'))
from onevision_research_worker import digest
REPORT=ROOT/'summary/onevision_diagnostics_20261016'


def main():
    path=REPORT/'task_domain_mechanism_audit.json';audit=json.loads(path.read_text())
    producer=ROOT/'analysis/onevision_task_domain_mechanism_audit.py'
    assert digest(producer)==audit['audit_sha256']
    run=ROOT/audit['run'];sources={str(path.relative_to(ROOT)):digest(path),str(producer.relative_to(ROOT)):digest(producer),
                                 str(Path(__file__).relative_to(ROOT)):digest(Path(__file__))}
    rows=[]
    for case in audit['rows']:
        if not case['method'].startswith('anchor_gradient_'):continue
        key=case['sequence']+'__'+case['method']
        for relative in [f'raw_step_gradients/{key}/stage_02.pt',f'adapters/{key}/stage_01.pt',f'adapters/{key}/stage_02.pt']:
            p=run/relative;expected=json.loads(p.with_suffix('.json').read_text())['sha256'];assert digest(p)==expected
            assert audit['source_sha256'][str(p.relative_to(ROOT))]==expected
            sources[str(p.relative_to(ROOT))]=expected
        first=next(v for v in case['first_minibatch_gradient_audits'] if v['stage']==2)
        assert first['expected_norm_matches_saved_optimizer_gradient_norm']
        rows.append({'sequence':case['sequence'],'control':case['method'],
                     'new_old_TRAIN_gradient_cosine':first['pre_cosine'],
                     'idealized_fresh_Adam_old_TRAIN_first_order_term':first['reconstructed_Adam_first_old_linear_change'],
                     'actual_four_update_displacement_old_TRAIN_first_order_term':first['actual_four_update_stage_delta_old_linear_term'],
                     'not_measured':'old loss at finite update or generated accuracy'})
    assert len(rows)==8
    result={'sources_sha256':sources,'cases':rows,'positive_gradient_cosines':sum(r['new_old_TRAIN_gradient_cosine']>0 for r in rows),
            'negative_Adam_Taylor_terms':sum(r['idealized_fresh_Adam_old_TRAIN_first_order_term']<0 for r in rows),
            'negative_four_update_Taylor_terms':sum(r['actual_four_update_displacement_old_TRAIN_first_order_term']<0 for r in rows),
            'accuracy_or_finite_NLL_causality_proven':False,
            'interpretation':'Latest-centered EWC first derivative0 is verified algebra; observed first transitions here are locally compatible, so zero protection alone does not prove harmful transfer.',
            'limitations':['These gradients belong to noop/projection own checkpoints, not EWC checkpoints.',
                          'Taylor terms are local diagnostics; finite nonlinear/BF16 loss can differ.',
                          'Original default-backend trajectories have stage1 numerical differences and one23-QA anchor.',
                          'All counts describe controls, not independent seeds or population-level significance.']}
    (REPORT/'FIRST_STEP_CAUSALITY_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    out=['# 首步保护项为零，不等于首步正在遗忘','',
         '这是K01的必要反例边界检查。既有旧TRAIN梯度、优化器范数和实际阶段位移分析的生产脚本／源文件SHA重新核验，无GPU或新训练。','',
         '| 序列 | 控制 | 新／旧梯度cos | 理想fresh Adam首步旧损失一阶项 | 实际4步位移旧损失一阶项 |',
         '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        out.append(f"| {row['sequence']} | {row['control']} | {row['new_old_TRAIN_gradient_cosine']:+.4f} | {row['idealized_fresh_Adam_old_TRAIN_first_order_term']:+.4f} | {row['actual_four_update_displacement_old_TRAIN_first_order_term']:+.4f} |")
    out+=['','8/8新旧TRAIN梯度cos为正；8/8理想Adam首步和实际四步位移的一阶项为负，局部近似均指向旧TRAIN损失下降。因此这些初始转移不能作为“EWC首步项为零导致遗忘”的正证据，反而要求保留可能的正迁移。',
          '', '这不是实测有限步旧NLL下降，也不是生成准确率证明；不能把noop／projection的梯度代入其他EWC条件。尚未测得的因果不被补全。K01的数学启用条件继续成立，但“为何最终掉分”必须另由匹配的权重／损失干预验证。',
          '', '原研究中的较后阶段可出现梯度冲突，但原默认后端、内容／token预算和单锚限制仍在，不能直接据此建立领域数的唯一因果解释。完整记录见[JSON](FIRST_STEP_CAUSALITY_AUDIT.json)。']
    (REPORT/'FIRST_STEP_CAUSALITY_AUDIT.md').write_text('\n'.join(out)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['positive_gradient_cosines','negative_Adam_Taylor_terms','negative_four_update_Taylor_terms','accuracy_or_finite_NLL_causality_proven']}))


if __name__=='__main__':main()
