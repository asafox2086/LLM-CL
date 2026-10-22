"""Consume accepted OSFT protection-control aggregates once; zero model/QA requests."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/osft_protection_control_20261022/v1'
WORK=ROOT/'exp/osft_protection_control_20261022/v1/private'
SCREEN=ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
DOMAINS=('natural','medical','geography','finance','traffic')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())

def main():
    begin=time.monotonic()
    assert not (REPORT/'OUTCOME.json').exists() and not (REPORT/'FINDINGS.md').exists()
    v=load(REPORT/'VERIFIED.json');c=load(REPORT/'COMPLETE.json');ready=load(REPORT/'READY.json')
    assert v['complete'] and v['quality']==c['quality'] and v['complete_sha256']==sha(REPORT/'COMPLETE.json')
    assert v['ready_sha256']==c['ready_sha256']==sha(REPORT/'READY.json')
    assert v['source_sha256']==sha(ROOT/'exp/osft_protection_control_20261022/v1/common.py')
    assert v['mechanism_only_no_full_candidates'] and v['TEST_QA']==0 and v['updates']==87
    assert ready['trainable_coordinates']==113766144 and ready['per_step_projection_package']=='disabled'
    assert c['skipped_projection_hooks']==dict(gradient=1213,parameter=87)
    ref=load(SCREEN/'SCREEN_COMPLETE.json')['all_four_results']['OSFT_all'];q=v['quality']
    pair=v['paired']['comparisons']['OSFT_all'];delta=pair['macro_difference_pp']
    assert list(v['paired']['comparisons'])==['OSFT_all']
    assert math.isclose(q['domain_macro']-ref['domain_macro'],delta,abs_tol=1e-10,rel_tol=0)
    assert v['paired']['replicates']==10000 and v['paired']['seed']==20261022
    counts={kind:sum(t[kind] for t in pair['QA_comparison'].values()) for kind in ['higher','lower','equal']}
    assert sum(counts.values())==100
    resources=[json.loads(s) for s in (WORK/'resources.jsonl').read_text().splitlines()]
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()];assert len(events)==87
    phase=lambda n:next(x for x in resources if x['phase']==n)
    checkpoints=[load(p) for p in sorted((WORK/'checkpoints').glob('stage_??.json'))]
    assert [p['stage'] for p in checkpoints]==list(range(1,20)) and checkpoints[-1]==v['checkpoint']==c['checkpoint']
    costs=dict(model_RUN_seconds=phase('run_finished')['unix']-phase('process_before_model')['unix'],
        training_interval_seconds=phase('training_after')['unix']-phase('training_before')['unix'],
        update_body_seconds=sum(e['seconds'] for e in events),checkpoint_metadata_seconds=sum(p['seconds'] for p in checkpoints),
        logical_checkpoint_bytes=sum(p['bytes'] for p in checkpoints),update_GPU_allocated_recorded_peak_bytes=max(e['gpu_peak_allocated_bytes'] for e in events),
        memory_summary=c['memory_summary'],CPU_admission_seconds=load(REPORT/'CPU_GATE.json')['seconds'],
        CPU_verification_seconds=v['CPU_verification_seconds'],
        scope='RUN excludes early imports/source checks. TRAIN interval includes first18 saves, excludes final save; save metadata excludes outer second SHA and processor export. Times overlap and are not added. Logical bytes are not physical IO/deployment. No controlled speed comparison.')
    unit=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show','llmcl-onevision-osft-protection-control-20261022.service',
        '-p','ActiveState','-p','MainPID','-p','ExecMainStatus','-p','Restart','-p','KillMode'],text=True).splitlines() if '=' in line)
    assert unit['ActiveState']=='inactive' and unit['MainPID']=='0' and unit['ExecMainStatus']=='0'
    queue=load(REPORT/'queue/state.json');assert queue['phase']=='complete_verified'
    interval=pair['exploratory_group_interval_pp']['macro']
    direction='higher' if delta>0 else 'lower' if delta<0 else 'same'
    spans_zero=interval[0]<=0<=interval[1]
    unresolved=spans_zero or delta==0
    action=('unresolved_keep_package_by_default_then_close' if unresolved else
            'retain_conditional_possible_cost_evidence_then_close' if delta>0 else
            'retain_package_in_fixed_recipe_then_close')
    inputs=[REPORT/'VERIFIED.json',REPORT/'COMPLETE.json',REPORT/'READY.json',REPORT/'INTERVENTION.json',
            REPORT/'CPU_GATE.json',SCREEN/'SCREEN_COMPLETE.json',WORK/'resources.jsonl',WORK/'logs/events.jsonl',
            WORK/'TRAIN_COMPLETE.json',*sorted((WORK/'checkpoints').glob('stage_??.json'))]
    out=dict(complete=True,quality=q,original_OSFT_aggregate=ref,paired=v['paired'],QA_total_counts=counts,
        skipped_projection_hooks=c['skipped_projection_hooks'],costs=costs,action=action,
        point_estimate_direction=direction,exploratory_interval_spans_zero=spans_zero,net_effect_unresolved=unresolved,
        service_terminal=unit,mechanism_only_no_full_candidates=True,current_configuration_stopped=True,
        source_sha256=sha(__file__),input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        new_model_or_QA_or_GPU_requests=0,not_new_algorithm_or_significance_or_equivalence=True,
        report_CPU_seconds=time.monotonic()-begin)
    lines=['# OSFT逐步保护包：固定训练配置下的净效果','',
        '问题是原OSFT的逐步保护包是否贡献最终回答质量。只新增一组同时禁用梯度/参数投影；七类位置、25%尾部USV初始化、113766144坐标/168模块、LR1e-5、HostAdamW/offload、逐阶段重SVD及原TRAIN AST均保留。原1213FIT/19阶段87有限正LR→100已探索DEV，复用原OSFT参照，不重训参照、不读官方TEST。',
        '', '| 域（各20QA） | 原OSFT保护包 | 去保护包 | 差pp | QA高/低/平 |','|---|---:|---:|---:|---|']
    for d in DOMAINS:
        n=pair['QA_comparison'][d]
        lines.append(f"| {d} | {ref['domains'][d]['F1']:.9f} | {q['domains'][d]['F1']:.9f} | {pair['domain_difference_pp'][d]:+.9f} | {n['higher']}/{n['lower']}/{n['equal']} |")
    lines+=['',f"五域宏：原{ref['domain_macro']:.9f}，去保护包{q['domain_macro']:.9f}，差{delta:+.9f}pp；100QA高/低/平={counts['higher']}/{counts['lower']}/{counts['equal']}。固定已知组10000次/seed20261022探索性百分位区间[{interval[0]:.9f}, {interval[1]:.9f}]pp。",
        'DEV已多次探索、单训练seed、各域20QA且未知跨组关联未保证，区间不表示训练seed不确定性或保证覆盖；不能称显著、稳定、等效或总体泛化。T7/8/9/12无DEV，不称评价全部19任务。', '',
        '关闭投影仍冻结高分量张量，但低因子可以进入高子空间并改变其作用，可达集合扩大；不是相同空间下纯优化器几何对照。后续各自重SVD可产生不同高基，只保持同一规则。一次关闭两回调，只能检验保护包整体净效果，不能拆分梯度/参数投影作用或唯一归因Adam。末态DEV不能区分习得与遗忘，本实验不解释或挽救已结束完整LoRA_all TEST差距。', '',
        ('探索区间跨0或点差为0，保护包净作用仍未解。操作上默认保留保护包并结束本v1，是研究决策，不是已证实其净收益；点估计方向单列OUTCOME，区间也不是显著性检验。' if unresolved else
         '新条件宏点估计更高，保留保护包在这个固定DEV上可能有净代价的条件证据；不直接删机制或进入完整TEST。' if delta>0 else
         '新条件宏点估计更低，在所试固定配置下继续保留保护包并结束去投影分支；不是保护机制普遍必要的定理。' if delta<0 else
         '宏点估计相同，问题仍未解决，结束本次控制；不把相同点估计当功能等效。'),
        '按预定出口，本v1无论正负都正常结束，不列完整候选、不自动1313/21454、不追加投影四格/初始化/rank/LR搜索；不因单次点估计或TEST追分扩大条件。', '',
        f"实际跳过1213次微样本后梯度回调请求、87次参数回调请求；不是内部算子或FLOPs计数。原独立verifier已核验全部ID/全原参考/原AnswerF1、预测/CP/配置/门槛SHA、19有效变化阶段/87有限正LR及106梯度文件SHA。本报告只消费封存聚合与资源元数据，0新模型/QA/优化/GPU。", '',
        f"新增模型RUN {costs['model_RUN_seconds']:.3f}秒，TRAIN区间{costs['training_interval_seconds']:.3f}（含前18保存、排最终保存），更新主体累计{costs['update_body_seconds']:.3f}，19保存元数据{costs['checkpoint_metadata_seconds']:.3f}秒；逻辑权重累计{costs['logical_checkpoint_bytes']}bytes。保存元数据排除外层第二次SHA和processor保存，二者仍在RUN内；计时重叠不相加。CPU准入{costs['CPU_admission_seconds']:.3f}秒、独立核验/唯一配对{costs['CPU_verification_seconds']:.3f}秒另计。",
        f"更新段记录CUDA allocated峰{costs['update_GPU_allocated_recorded_peak_bytes']}bytes，其余同PID内存分列OUTCOME。不是受控训练加速或部署成本证明。主FIT1213曝光/87更新，另38NLL路径+38至多4token门槛生成+100DEV生成；原screen/完整确认/同预算对照全部探索成本各自保留。", '',
        '普通去投影不是新算法，113766144训练坐标不是有效自由度或推理压缩，研究checkpoint不是部署重载证明。端侧仅作背景；正式8/16基线口径、Ours及未测部署列保持原记录。']
    (REPORT/'OUTCOME.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (REPORT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(outcome_sha256=sha(REPORT/'OUTCOME.json'),DEV_macro=q['domain_macro'],difference_pp=delta,
        interval_pp=interval,QA_counts=counts,action=action,costs={k:v for k,v in costs.items() if k!='memory_summary'})))
if __name__=='__main__':main()
