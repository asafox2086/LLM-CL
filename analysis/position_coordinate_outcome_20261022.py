"""Read accepted allocation-control aggregates and resource metadata once; no QA/model."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/position_coordinate_control_20261022/v1'
WORK=ROOT/'exp/position_coordinate_control_20261022/v1/private'
SCREEN=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())

def main():
    begin=time.monotonic()
    assert not (REPORT/'OUTCOME.json').exists() and not (REPORT/'FINDINGS.md').exists()
    v=load(REPORT/'VERIFIED.json');c=load(REPORT/'COMPLETE.json');ready=load(REPORT/'READY.json')
    assert v['complete'] and v['quality']==c['quality'] and v['complete_sha256']==sha(REPORT/'COMPLETE.json')
    assert v['ready_sha256']==c['ready_sha256']==sha(REPORT/'READY.json')
    assert v['source_sha256']==sha(ROOT/'exp/position_coordinate_control_20261022/v1/common.py')
    assert v['mechanism_only_no_full_candidates'] and v['TEST_QA']==0 and v['updates']==87
    refs=load(SCREEN/'SCREEN_COMPLETE.json')['all_four_results'];q=v['quality'];pairs=v['paired']['comparisons']
    assert all(math.isclose(q['domain_macro']-refs[k]['domain_macro'],pairs[k]['macro_difference_pp'],abs_tol=1e-10,rel_tol=0) for k in ['LoRA_all','LoRA_qv'])
    counts={k:{kind:sum(t[kind] for t in p['QA_comparison'].values()) for kind in ['higher','lower','equal']} for k,p in pairs.items()}
    assert all(sum(v.values())==100 for v in counts.values())
    resources=[json.loads(s) for s in (WORK/'resources.jsonl').read_text().splitlines()]
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()];assert len(events)==87
    phase=lambda n:next(x for x in resources if x['phase']==n)
    checkpoints=[load(p) for p in sorted((WORK/'checkpoints').glob('stage_??.json'))]
    assert [p['stage'] for p in checkpoints]==list(range(1,20)) and checkpoints[-1]==v['checkpoint']==c['checkpoint']
    costs=dict(model_RUN_seconds=phase('run_finished')['unix']-phase('process_before_model')['unix'],
        training_interval_seconds=phase('training_after')['unix']-phase('training_before')['unix'],
        update_body_seconds=sum(e['seconds'] for e in events),checkpoint_metadata_seconds=sum(p['seconds'] for p in checkpoints),
        logical_checkpoint_bytes=sum(p['bytes'] for p in checkpoints),update_GPU_allocated_recorded_peak_bytes=max(e['gpu_peak_allocated_bytes'] for e in events),
        memory_summary=c['memory_summary'],CPU_admission_seconds=load(REPORT/'ADMISSION_LOG.json')['seconds'],
        CPU_verification_seconds=v['CPU_verification_seconds'],
        scope='RUN excludes early imports/source checks. TRAIN interval includes first18 saves, excludes final save; save metadata excludes outer second SHA and processor export. All are overlapping, not added. Logical bytes are not physical IO/deployment. No controlled speed comparison.')
    generic_all=4399104-168*8**2
    matrix_geometry=dict(wide=load(REPORT/'RANK_PLAN.json'),all_r8=dict(trainable_coordinates=4399104,
        summed_generic_rank_manifold_dimension=generic_all,summed_zero_B_matrix_map_Jacobian_dimension=24*8*(896+128+128+896+4864+4864+896)),
        scope='Exact-arithmetic per-matrix parameterization maps, conditional on full-row-rank A and generic factors. Sums are not whole-model logit Jacobian rank, numerical rank, effective freedom, useful learning or F1 contribution.')
    unit=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show','llmcl-onevision-coordinate-control-20261022.service',
        '-p','ActiveState','-p','MainPID','-p','ExecMainStatus','-p','Restart','-p','KillMode'],text=True).splitlines() if '=' in line)
    assert unit['ActiveState']=='inactive' and unit['MainPID']=='0' and unit['ExecMainStatus']=='0'
    queue=load(REPORT/'queue/state.json');assert queue['phase']=='complete_verified'
    inputs=[REPORT/'VERIFIED.json',REPORT/'COMPLETE.json',REPORT/'READY.json',REPORT/'RANK_PLAN.json',SCREEN/'SCREEN_COMPLETE.json',WORK/'resources.jsonl',WORK/'logs/events.jsonl']
    out=dict(complete=True,quality=q,old_reference_aggregates={k:refs[k] for k in ['LoRA_all','LoRA_qv']},paired=v['paired'],
        QA_total_counts=counts,costs=costs,matrix_parameterization_geometry=matrix_geometry,
        finance_macro_contribution_pp=pairs['LoRA_all']['domain_difference_pp']['finance']/5,
        other_four_domains_macro_contribution_pp=sum(x for d,x in pairs['LoRA_all']['domain_difference_pp'].items() if d!='finance')/5,
        service_terminal=unit,mechanism_only_no_full_candidates=True,current_configuration_stopped=True,
        source_sha256=sha(__file__),input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        new_model_or_QA_or_GPU_requests=0,not_new_algorithm_or_significance_or_equivalence=True,
        report_CPU_seconds=time.monotonic()-begin)
    (REPORT/'OUTCOME.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 同坐标预算分配对照：更高DEV点估计，位置必要性仍未证实','',
        '原四组qv-r8与all-r8同时改变位置和训练坐标。本次只新增一组宽q/v、精确匹配all的4399104坐标及乘数4，复用原两参照；基座新鲜训练原1213FIT/19阶段/87更新，仅最终100DEV。原完整LoRA_all TEST负结果不改，本实验不解释该TEST差距。',
        '', '| 配置 | 训练坐标 | 更新模块 | DEV五域均分 |','|---|---:|---:|---:|',
        f"| 原q/v r8 | 540672 | 48 | {refs['LoRA_qv']['domain_macro']:.9f} |",
        f"| 原七类 r8 | 4399104 | 168 | {refs['LoRA_all']['domain_macro']:.9f} |",
        f"| 宽q/v r65，6个v r66 | 4399104 | 48 | {q['domain_macro']:.9f} |",'',
        '| 域（各20QA） | 宽q/v | 原all | 差pp | 原q/v r8 | 差pp |','|---|---:|---:|---:|---:|---:|']
    for d in ['natural','medical','geography','finance','traffic']:
        lines.append(f"| {d} | {q['domains'][d]['F1']:.9f} | {refs['LoRA_all']['domains'][d]['F1']:.9f} | {pairs['LoRA_all']['domain_difference_pp'][d]:+.9f} | {refs['LoRA_qv']['domains'][d]['F1']:.9f} | {pairs['LoRA_qv']['domain_difference_pp'][d]:+.9f} |")
    lines+=['']
    for key,label in [('LoRA_all','主要对比：宽q/v减all'),('LoRA_qv','次要对比：宽q/v减原q/v-r8')]:
        p=pairs[key];interval=p['exploratory_group_interval_pp']['macro'];n=counts[key]
        lines.append(f"{label} {p['macro_difference_pp']:+.9f}pp，100QA高/低/平={n['higher']}/{n['lower']}/{n['equal']}；固定已知组探索性百分位区间[{interval[0]:.9f}, {interval[1]:.9f}]pp。")
    lines+=['',f"对all，金融仅3题更高、0题更低，贡献宏+{out['finance_macro_contribution_pp']:.9f}pp；其余四域合计{out['other_four_domains_macro_contribution_pp']:+.9f}pp。自然/遥感/交通均下降，不是五域全面优势。两个宏区间均跨0；已探索DEV、单seed、小样本、未知跨组依赖，均不能声称稳定/显著/等效或总体泛化。",'',
        '在这100题和固定优化recipe上，集中于q/v也取得更高宏点估计。因此原all对q/v-r8的点估计增益不能单独作为额外120位置必要性的证据。这不证明仅坐标数解释全部收益，也不证明位置无用：宽度、坐标分配、有限步优化几何仍不同，金融贡献集中。rank65/66只来自预算算术，不是最优rank；alpha=4r固定乘数，不是额外scale搜索。',
        '',f"同训练坐标也不是同更新自由度。条件精确算术下，逐矩阵rank-r参数化的一般流形维数之和：宽q/v {matrix_geometry['wide']['nominal_generic_rank_manifold_dimension']}、all-r8 {generic_all}；零B时逐矩阵ΔW映射Jacobian像维数之和：宽q/v {matrix_geometry['wide']['nominal_zero_B_Jacobian_image_dimension']}、all-r8 {matrix_geometry['all_r8']['summed_zero_B_matrix_map_Jacobian_dimension']}。这不是实测数值秩/整模logit Jacobian/功能有效维数，不能从这些维数预测F1或判断参数无用。",'',
        f"新增模型RUN {costs['model_RUN_seconds']:.3f}秒；TRAIN区间{costs['training_interval_seconds']:.3f}（含前18保存），更新主体累计{costs['update_body_seconds']:.3f}，19保存元数据{costs['checkpoint_metadata_seconds']:.3f}秒，逻辑权重累计{costs['logical_checkpoint_bytes']}bytes。保存元数据排除外层第二次SHA和processor保存，这些仍在RUN内；计时重叠不双加。CPU新增准入{costs['CPU_admission_seconds']:.3f}秒、独立核验/固定配对{costs['CPU_verification_seconds']:.3f}秒单列。原四组探索成本见原screen报告，不抹去。",
        f"更新段记录CUDA allocated峰{costs['update_GPU_allocated_recorded_peak_bytes']}bytes；其余同PID内存及RUN观测分列OUTCOME。不是受控速度比较或部署收益。新增主FIT1213曝光/87更新，另38NLL路径+38至多4token门槛生成+100DEV生成；基座/科学源/原AnswerF1与全部参考不改。最终研究checkpoint并非部署重载证明。",'',
        '按已冻结机制协议，本v1完成正常退出，不列完整候选、不自动1313/21454、不加第二scale/rank/位置变体，也不把新分数代回旧四组名单。原完整确认及原正式基线记录不改。更高DEV点估计只改变必要性证据的解释，不是新算法或完整方法成功；端侧仍仅背景。']
    (REPORT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(outcome_sha256=sha(REPORT/'OUTCOME.json'),wide_DEV_macro=q['domain_macro'],
        primary_difference_pp=pairs['LoRA_all']['macro_difference_pp'],counts=counts,costs={k:v for k,v in costs.items() if k!='memory_summary'},no_full_candidates=True)))
if __name__=='__main__':main()
