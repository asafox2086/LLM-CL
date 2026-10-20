"""Read-only publication of a sealed ten-QA CPU function probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/cpu_function_space_20261020/v2'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish():
    target=REPORT/'VERIFIED_RESULTS.json'
    if target.exists():return
    c=json.loads((REPORT/'COMPLETE.json').read_text());e=json.loads((REPORT/'EVIDENCE.json').read_text());r=json.loads((REPORT/'READY.json').read_text())
    assert c['status']=='CPU_TRAIN_FUNCTION_SPACE_COMPLETE' and c['evidence_sha256']==sha(REPORT/'EVIDENCE.json')
    assert c['source_sha256']==e['source_sha256']==r['source_sha256']==sha(ROOT/'analysis/cpu_function_space_20261020_v2.py')
    assert e['protocol_sha256']==r['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
    assert e['CPU_only'] and not e['official_TEST_read'] and not e['DEV_answers_used']
    assert e['teacher_has_trained_on_all_probe_QA'] and e['first_CPU_original_folded_NLL_exact']
    assert e['model_forward_calls']==11 and e['new_optimizer_updates']==0
    records=e['records'];assert len(records)==10 and len(e['calibrators'])==48
    wanted={x['id']:x for x in r['selected']};assert {x['id'] for x in records}==set(wanted)
    all_names=None
    for x in records:
        assert all(x[k]==wanted[x['id']][k] for k in ['role','group','domain','task'])
        assert math.isfinite(x['NLL']) and x['target_tokens']>0
        names={y['name'] for y in x['observations']};assert len(names)==48
        if all_names is None:all_names=names
        else:assert names==all_names
        for y in x['observations']:
            assert y['tokens']==x['target_tokens']
            for key in ['reference_energy','weight_rank8_error']+(['function_rank8_error'] if x['role']=='probe' else []):
                assert math.isfinite(y[key]) and y[key]>=0
    probes=[]
    for x in records:
        if x['role']!='probe':continue
        total=math.fsum(y['reference_energy'] for y in x['observations']);assert total>0
        probes.append({'id':x['id'],'domain':x['domain'],'reference_energy':total,
            'weight_rank8_relative_error':math.sqrt(math.fsum(y['weight_rank8_error'] for y in x['observations'])/total),
            'function_rank8_relative_error':math.sqrt(math.fsum(y['function_rank8_error'] for y in x['observations'])/total)})
    assert len(probes)==5 and len({x['domain'] for x in probes})==5
    output={'status':'TEN_QA_CPU_FUNCTION_SPACE_INDEPENDENTLY_VERIFIED','evidence_sha256':sha(REPORT/'EVIDENCE.json'),
            'report_source_sha256':sha(__file__),'probe_results':probes,
            'domain_macro_relative_error':{k:math.fsum(x[k] for x in probes)/5 for k in ['weight_rank8_relative_error','function_rank8_relative_error']},
            'CPU_only':True,'AnswerF1_measured':False,'deployed_compressed_model_measured':False,
            'calibration_function_error_scope':'pooled per layer; no per-calibration-QA function-error artifact'}
    lines=['# CPU功能空间小测：完整结果','',
           '同一OSFT最终权重、相同rank8，比较权重SVD与按5个校准QA拟合的输出子空间。下面是5个不同图像组探针，teacher训练见过它们；不是未见任务评价或Answer F1。','',
           '| 探针领域 | 权重rank8相对输出误差 | 功能rank8相对输出误差 |','| --- | ---: | ---: |']
    for x in probes:lines.append(f"| {x['domain']} | {100*x['weight_rank8_relative_error']:.3f}% | {100*x['function_rank8_relative_error']:.3f}% |")
    m=output['domain_macro_relative_error'];lines += [f"| 五域等权均值 | {100*m['weight_rank8_relative_error']:.3f}% | {100*m['function_rank8_relative_error']:.3f}% |",'',
        '每题先汇总48矩阵误差平方／该题参考输出能量，再开平方；最后五域等权。不把某一层或长答案的优势当整个领域F1。全部逐题逐层值见EVIDENCE。', '',
        '**能说明：** 同秩的两种空间在这些实际输入上的局部作用差异，权重能量覆盖不等于功能误差。**不能说明：** 完整模型精度、自由生成、未见任务泛化、参数可删、部署变小或推理加速；只压ΔW不会缩小完整基座。','',
        '校准功能误差只保存逐层合并值；未保存逐校准题的该项误差，不补造。探针的两种误差均逐题／逐层完整。5题探针只能给点观察，不宣称总体显著性，不据此填Ours或选择官方TEST超参。']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(lines)+'\n')
    target.write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        if (REPORT/'FAILURE.json').exists():break
        if (REPORT/'COMPLETE.json').exists():publish();break
        if not a.watch:break
        time.sleep(30)
