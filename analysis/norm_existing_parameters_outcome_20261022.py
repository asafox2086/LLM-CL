"""Consume sealed norm-only aggregates once; no QA, model, or bootstrap rerun."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/norm_existing_parameters_20261022/v1'
WORK=ROOT/'exp/norm_existing_parameters_20261022/v1/private'
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
    assert v['source_sha256']==sha(ROOT/'exp/norm_existing_parameters_20261022/v1/common.py')
    for path in [SCREEN/'SCREEN_COMPLETE.json',REPORT/'CPU_GATE.json',REPORT/'INTERVENTION.json']:
        assert sha(path)==ready['evidence'][path.relative_to(ROOT).as_posix()]
    assert sha(WORK/'logs/events.jsonl')==c['events_sha256']
    assert sha(WORK/'gain_updates.json')==c['update_evidence_sha256']
    assert v['TEST_QA']==0 and v['updates']==87 and not v['automatic_full_confirmation']
    assert ready['trainable_coordinates']==43008 and ready['method']=='RMSNorm_only'
    refs=load(SCREEN/'SCREEN_COMPLETE.json')['all_four_results'];q=v['quality']
    assert list(v['paired']['comparisons'])==['LoRA_qv','OSFT_all']
    assert v['paired']['replicates']==10000 and v['paired']['seed']==20261022
    threshold=max(refs[name]['domain_macro'] for name in ['LoRA_qv','OSFT_all'])
    passed=q['domain_macro']>threshold
    assert v['fixed_anchor_threshold']==threshold and v['possible_followup_fresh_PAS_reference']==passed
    counts={}
    for name,p in v['paired']['comparisons'].items():
        assert math.isclose(p['macro_difference_pp'],q['domain_macro']-refs[name]['domain_macro'],abs_tol=1e-10,rel_tol=0)
        counts[name]={kind:sum(t[kind] for t in p['QA_comparison'].values()) for kind in ['higher','lower','equal']}
        assert sum(counts[name].values())==100
    updates=load(WORK/'gain_updates.json');assert updates==v['actual_forward_update_evidence'] and len(updates)==87
    stages=[]
    for plan in ready['training_plan']:
        rows=[r for r in updates if r['stage']==plan['stage']]
        stages.append(dict(stage=plan['stage'],updates=len(rows),
            master_changed_updates=sum(r['from_previous']['master']['changed']>0 for r in rows),
            forward_gain_changed_updates=sum(r['from_previous']['forward']['changed']>0 for r in rows),
            from_stage_start=rows[-1]['from_stage_start']))
    actual=dict(stage_details=stages,final_gain_change=v['final_gain_change'],
        total_master_changed_updates=sum(s['master_changed_updates'] for s in stages),
        total_forward_gain_changed_updates=sum(s['forward_gain_changed_updates'] for s in stages),
        stages_without_forward_gain_change=[s['stage'] for s in stages if s['from_stage_start']['forward']['changed']==0],
        not_whole_function_or_output_or_F1_causal_proof=True)
    assert c['config']['frozen_digest_before_TRAIN']==c['config']['frozen_digest_after_TRAIN']
    resources=[json.loads(s) for s in (WORK/'resources.jsonl').read_text().splitlines()]
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()];assert len(events)==87
    phase=lambda name:next(x for x in resources if x['phase']==name)
    checkpoints=[load(p) for p in sorted((WORK/'checkpoints').glob('stage_??.json'))]
    assert [p['stage'] for p in checkpoints]==list(range(1,20)) and checkpoints[-1]==v['checkpoint']==c['checkpoint']
    costs=dict(model_RUN_seconds=phase('run_finished')['unix']-phase('process_before_model')['unix'],
        training_interval_seconds=phase('training_after')['unix']-phase('training_before')['unix'],
        update_body_seconds=sum(e['seconds'] for e in events),checkpoint_metadata_seconds=sum(p['seconds'] for p in checkpoints),
        logical_checkpoint_bytes=sum(p['bytes'] for p in checkpoints),update_GPU_allocated_recorded_peak_bytes=max(e['gpu_peak_allocated_bytes'] for e in events),
        memory_summary=c['memory_summary'],CPU_admission_seconds=load(REPORT/'CPU_GATE.json')['seconds'],
        CPU_verification_seconds=v['CPU_verification_seconds'],
        scope='RUN excludes early imports/source checks. TRAIN interval includes first18 saves, excludes final save and terminal frozen-weight digest; RUN includes these. Save metadata excludes second SHA/processor export. Times overlap, not added. Logical bytes not physical IO/deployment; no controlled speed comparison.')
    unit=dict(line.split('=',1) for line in subprocess.check_output(['systemctl','--user','show','llmcl-onevision-norm-only-20261022.service',
        '-p','ActiveState','-p','MainPID','-p','ExecMainStatus','-p','Restart','-p','KillMode'],text=True).splitlines() if '=' in line)
    assert unit['ActiveState']=='inactive' and unit['MainPID']=='0' and unit['ExecMainStatus']=='0'
    assert load(REPORT/'queue/state.json')['phase']=='complete_verified'
    action='retain_viability_only_consider_separate_fresh_FIT_PAS_reference' if passed else 'close_fixed_recipe_no_LR_search'
    inputs=[REPORT/'VERIFIED.json',REPORT/'COMPLETE.json',REPORT/'READY.json',REPORT/'INTERVENTION.json',REPORT/'CPU_GATE.json',
        SCREEN/'SCREEN_COMPLETE.json',WORK/'resources.jsonl',WORK/'logs/events.jsonl',WORK/'TRAIN_COMPLETE.json',WORK/'gain_updates.json',
        *sorted((WORK/'checkpoints').glob('stage_??.json'))]
    out=dict(complete=True,quality=q,anchors={name:refs[name] for name in ['LoRA_qv','OSFT_all']},paired=v['paired'],
        QA_total_counts=counts,primary_passed=passed,fixed_anchor_threshold=threshold,action=action,actual_gain_updates=actual,
        costs=costs,service_terminal=unit,current_configuration_stopped=True,automatic_full_confirmation=False,
        source_sha256=sha(__file__),input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in inputs},
        new_model_or_QA_or_GPU_requests=0,not_new_algorithm_or_significance_or_equivalence=True,report_CPU_seconds=time.monotonic()-begin)
    lines=['# 已有RMSNorm增益：固定配方的功能替代可行性','',
        '问题是原1%预算下，仅更新已有层内归一化增益是否有竞争力。只新增一个48处/43008权重的配方；最终norm、视觉/projector及其他所有权重冻结。原1213FIT/19阶段87更新→100已经探索DEV，LR1e-4沿Seq配方，不搜LR/位置，不匹配实际功能步长。不是新算法或纯参数空间因果实验。',
        '', '| 域（各20QA） | 原q/v LoRA | 原OSFT | 仅已有RMSNorm |','|---|---:|---:|---:|']
    for d in DOMAINS:lines.append(f"| {d} | {refs['LoRA_qv']['domains'][d]['F1']:.9f} | {refs['OSFT_all']['domains'][d]['F1']:.9f} | {q['domains'][d]['F1']:.9f} |")
    lines+=['',f"五域宏：q/v {refs['LoRA_qv']['domain_macro']:.9f}，OSFT {refs['OSFT_all']['domain_macro']:.9f}，仅RMSNorm {q['domain_macro']:.9f}。固定primary严格高于两锚中较高者{threshold:.9f}：{passed}。"]
    for name,p in v['paired']['comparisons'].items():
        n=counts[name];interval=p['exploratory_group_interval_pp']['macro']
        lines.append(f"对{name}宏差{p['macro_difference_pp']:+.9f}pp，100QA高/低/平={n['higher']}/{n['lower']}/{n['equal']}；整已知组10000次/seed20261022探索区间[{interval[0]:.9f}, {interval[1]:.9f}]pp。逐域配对见OUTCOME，不能按域选路由。")
    lines+=['', 'DEV已多次探索、单seed、每域20QA，未知跨组关联未保证；不是新未见验证，不称总体显著/稳定/等效。T7/8/9/12无DEV，不称独立评价全部19任务。', '',
        ('越过两个固定锚，仅保留该配方的可行性候选。尚未对比fresh FIT-only最强PASs，不能宣布替代最强方法；未来是否做这一独立对照需另行判断用途和预算。本次不自动PASs/1313/21454。' if passed else
         '未越过两个固定锚，按协议结束这一个配方，不搜LR/层、不据TEST救分。阴性只限制当前固定预算与数值配方，不能宣称已有参数普遍不够或新增LoRA/专家必然必要。'), '',
        f"FP32主副本实际变化的更新{actual['total_master_changed_updates']}/87；实际forward FP16增益变化的更新{actual['total_forward_gain_changed_updates']}/87。阶段首尾forward增益无变化的阶段{actual['stages_without_forward_gain_change']}。最终master/forward完整初终差与各阶段计数见OUTCOME；初始/最终私有张量已由原CPU verifier核验dtype/转换及SHA。", 
        '保持原FP16 forward计算，FP32仅为优化主副本。舍入可能使master变化而实际增益不变；这不是新增精度实验，也不靠19个master-validity推断19次函数更新。实际增益变化不是整体输出或F1贡献的因果证明。其他冻结参数TRAIN前后全量digest相同。没有数据量对照，不作1%特有机制或训练不充分的确定归因。', '',
        f"新增模型RUN {costs['model_RUN_seconds']:.3f}秒；TRAIN区间{costs['training_interval_seconds']:.3f}含前18保存，更新主体累计{costs['update_body_seconds']:.3f}，19保存元数据{costs['checkpoint_metadata_seconds']:.3f}秒，逻辑权重累计{costs['logical_checkpoint_bytes']}bytes。最终保存、终态冻结权重digest与外层SHA/processor导出计入RUN；重叠计时不相加。准入{costs['CPU_admission_seconds']:.3f}秒、独立核验/两个唯一配对{costs['CPU_verification_seconds']:.3f}秒另计。",
        f"主FIT1213曝光/87更新；额外38NLL+38最多4token门槛生成+100DEV。更新段CUDA allocated记录峰{costs['update_GPU_allocated_recorded_peak_bytes']}bytes，仅本运行记录，非受控加速或端侧优势。旧参照和此前探索成本各自保留。", '',
        '原科学源/READY不改。本先导不改正式主表，Ours/未测部署保持原记录。43008仅训练坐标数，不是有效维数或推理压缩；端侧仅科研背景。']
    (REPORT/'OUTCOME.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (REPORT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(outcome_sha256=sha(REPORT/'OUTCOME.json'),DEV_macro=q['domain_macro'],primary_passed=passed,
        action=action,actual_gain_updates={k:value for k,value in actual.items() if k!='stage_details'},costs={k:value for k,value in costs.items() if k!='memory_summary'})))
if __name__=='__main__':main()
