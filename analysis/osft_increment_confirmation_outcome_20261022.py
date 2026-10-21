"""Consume the one sealed full confirmation; never load model, QA, or predictions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
ENTRY=ROOT/'exp/osft_increment_confirmation_20261022/v1'
sys.path.insert(0,str(ENTRY))
import common
REPORT=common.REPORT
WORK=common.WORK
sha,write=common.sha,common.write

def main():
    started=time.monotonic()
    assert (REPORT/'VERIFIED.json').exists(), 'Wait for the original independent verifier, no partial results'
    assert not (REPORT/'OUTCOME.json').exists() and not (REPORT/'FINDINGS.md').exists(), 'One report; preserve prior result'
    verified=json.loads((REPORT/'VERIFIED.json').read_text())
    complete=json.loads((REPORT/'COMPLETE.json').read_text())
    ready=json.loads((REPORT/'READY.json').read_text())
    assert verified['complete'] and verified['completion_sha256']==sha(REPORT/'COMPLETE.json')
    assert verified['ready_sha256']==complete['ready_sha256']==sha(REPORT/'READY.json')
    assert verified['source_sha256']==sha(ENTRY/'common.py')
    assert verified['quality']==complete['quality']
    assert ready['anchors']==common.ANCHORS
    quality=verified['quality'];assert quality['TEST_QA']==21454 and len(quality['tasks'])==19
    assert quality['FWT'] is None and quality['BWT'] is None
    assert complete['TRAIN_QA']==1313 and complete['updates']==93
    assert complete['config']['trainable_parameters']==4399104
    anchors=ready['anchors'];threshold=max(a['macro'] for a in anchors.values())
    assert threshold==ready['primary_threshold']==46.619976395158304
    primary=quality['domain_macro']>threshold;assert primary==verified['primary_pass']
    differences={d:{a:v['F1']-x['domains'][d] for a,x in anchors.items()} for d,v in quality['domains'].items()}
    all_domains=all(delta>0 for dd in differences.values() for delta in dd.values())
    macro_delta={a:quality['domain_macro']-x['macro'] for a,x in anchors.items()}
    ap_delta={a:quality['AP19']-x['AP19'] for a,x in anchors.items()}
    resources=[json.loads(s) for s in (WORK/'resources.jsonl').read_text().splitlines()]
    events=[json.loads(s) for s in (WORK/'logs/events.jsonl').read_text().splitlines()]
    assert len(events)==93
    resource_phase=lambda p:next(r for r in resources if r['phase']==p)
    before=resource_phase('training_before');after=resource_phase('training_after')
    checkpoints=[json.loads(p.read_text()) for p in sorted((WORK/'checkpoints').glob('stage_??.json'))]
    assert [c['stage'] for c in checkpoints]==list(range(1,20))
    assert checkpoints[-1]==complete['checkpoint']==verified['checkpoint']
    costs=dict(model_RUN_observed_seconds=resource_phase('run_finished')['unix']-resource_phase('process_before_model')['unix'],
        training_before_to_after_seconds=after['unix']-before['unix'],
        summed_update_body_seconds=sum(e['seconds'] for e in events),
        saved_stage_file_bytes_sum=sum(c['bytes'] for c in checkpoints),
        summed_checkpoint_metadata_seconds=sum(c['seconds'] for c in checkpoints),
        training_update_recorded_GPU_allocated_peak_bytes=max(e['gpu_peak_allocated_bytes'] for e in events),
        memory_summary=complete['memory_summary'],
        scope='RUN begins before model, excludes earlier imports/source checks. Training interval includes first18 saves, excludes final save. Save metadata excludes outer second SHA/processor save; these are inside RUN. Times overlap and cannot be added. Byte sum is logical checkpoint size, not physical disk IO or deployment package. No controlled speed comparison.')
    provenance={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [REPORT/'VERIFIED.json',REPORT/'COMPLETE.json',REPORT/'READY.json',
        WORK/'config.json',WORK/'resources.jsonl',WORK/'logs/events.jsonl',*sorted((WORK/'checkpoints').glob('stage_??.json'))]}
    out=dict(complete=True,quality=quality,anchors=anchors,primary_threshold=threshold,primary_pass=primary,
        all_five_domains_above_both_anchors=all_domains,domain_differences_pp=differences,
        macro_differences_pp=macro_delta,AP19_differences_pp=ap_delta,training_coordinates=4399104,
        original_OSFT_training_coordinates=113766144,not_effective_dimension_or_inference_compression=True,
        costs=costs,main_TRAIN_QA=1313,main_positive_updates=93,
        additional_native_gate_NLL_paths=38,additional_native_gate_short_generations=38,
        final_TEST_generations=21454,exploration_costs_separate_report='summary/edge_deployment_20261019/osft_increment_screen/v1/RESULTS_20261022.md',
        proof_scope='final_only_19_tasks_not_74_cells',source_sha256=sha(__file__),input_sha256=provenance,
        new_model_or_optimization_or_GPU_requests=0,single_seed_not_significance_or_equivalence=True,
        ordinary_LoRA_not_new_algorithm=True,deployment_not_measured=True)
    out['CPU_report_seconds']=time.monotonic()-started
    lines=['# LoRA_all完整1%最终确认','','本报告只在原VERIFIED后消费封存结果；0新增模型/优化/生成/GPU，不读取QA或逐题预测。',
        '',f"主要五域宏均分：{quality['domain_macro']:.9f}；预定门槛{threshold:.9f}；{'通过' if primary else '未通过'}。",
        f"对PASs {macro_delta['PASs_MoE']:+.9f}pp，对OSFT {macro_delta['OSFT']:+.9f}pp。",
        f"各域分别超过两参照：{all_domains}。单seed点估计不证明显著/稳定/等效；均值阳性不能掩盖单域下降。",'',
        '| 域 | QA | LoRA_all F1 | PASs F1 | 差pp | OSFT F1 | 差pp |','|---|---:|---:|---:|---:|---:|---:|']
    for d in ['natural','medical','geography','finance','traffic']:
        q=quality['domains'][d];dd=differences[d]
        lines.append(f"| {d} | {q['QA']} | {q['F1']:.9f} | {anchors['PASs_MoE']['domains'][d]:.9f} | {dd['PASs_MoE']:+.9f} | {anchors['OSFT']['domains'][d]:.9f} | {dd['OSFT']:+.9f} |")
    lines+=['',f"伴随AP19={quality['AP19']:.9f}，对PASs {ap_delta['PASs_MoE']:+.9f}pp，对OSFT {ap_delta['OSFT']:+.9f}pp。原全部19任务见OUTCOME.json；AP19不替代主要量。FWT/BWT空，最终19格不冒充原74格基线。",'',
        '完整1313 TRAIN/19阶段/93正LR来自基座新鲜训练，r8七类位置/alpha32/dropout.1/LR1e-4/seed42及原TRAIN AST/FP16基座与外层autocast/FP32 LoRA张量不改。不是FIT权重续训。四组探索的额外费用见原screen报告，未入选条件未全训/TEST。',
        '',f"记录模型RUN {costs['model_RUN_observed_seconds']:.3f}秒，training_before→after {costs['training_before_to_after_seconds']:.3f}秒（含前18保存），更新主体累计{costs['summed_update_body_seconds']:.3f}秒；19次保存元数据计时{costs['summed_checkpoint_metadata_seconds']:.3f}秒，逻辑权重字节累计{costs['saved_stage_file_bytes_sum']}。这些计时重叠，不相加。外层二次SHA和processor保存包含在RUN内，未计入保存元数据seconds；早期imports/源码检查排除。",
        f"更新段记录的CUDA allocated峰{costs['training_update_recorded_GPU_allocated_peak_bytes']}bytes；同PID内存/整RUN观测峰等分列OUTCOME，不是部署指标或严格同设备受控速度对照。完整checkpoint未验证部署功能重载；4399104训练坐标不是有效维数或推理压缩。额外原门槛38NLL路径+38个至多4token短生成另计。",'',
        '结果只检验此固定普通LoRA配置的最终回答质量。位置/总坐标仍混杂，跨OSFT另有LR/投影/参数化差异；不证明这些机制可删、rank最优或1%特有因果。不根据本TEST改路由/层/rank/提示/步数或追加备选，不择优重跑。端侧只是科研背景。']
    write(REPORT/'OUTCOME.json',out)
    (REPORT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(outcome_sha256=sha(REPORT/'OUTCOME.json'),domain_macro=quality['domain_macro'],AP19=quality['AP19'],
        primary_pass=primary,all_domains_above_both=all_domains,CPU_report_seconds=out['CPU_report_seconds'])))
if __name__=='__main__':main()
