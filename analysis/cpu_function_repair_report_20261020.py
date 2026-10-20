"""Independently summarize a sealed engineering gate, never a quality result."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'summary/edge_deployment_20261019/cpu_function_space_20261020/v2'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=D/'repair_gate';c=json.loads((p/'COMPLETE.json').read_text());e=json.loads((p/'EVIDENCE.json').read_text())
    assert c['status']==e['status']=='CPU_CONTEXT_REPAIR_ALL168_EXACT'
    assert c['evidence_sha256']==sha(p/'EVIDENCE.json')
    assert c['source_sha256']==e['source_sha256']==sha(ROOT/'analysis/cpu_function_context_gate_20261020.py')
    assert e['CPU_only'] and not e['QA_access'] and not e['full_model_forward'] and e['optimizer_updates']==0
    rows=e['rows'];assert len(rows)==168 and len({x['name'] for x in rows})==168
    assert all(x['original_vs_matching_context_exact'] for x in rows)
    assert all(math.isfinite(x['max_abs_outside_context_error']) for x in rows)
    new_exact=sum(x['original_vs_matching_context_exact'] for x in rows)
    old_bad=sum(not x['original_vs_outside_context_exact'] for x in rows)
    result={'status':'REPAIR_GATE_INDEPENDENTLY_VERIFIED','source_sha256':sha(__file__),
            'gate_evidence_sha256':sha(p/'EVIDENCE.json'),'operators':168,'new_exact':new_exact,
            'old_mismatched':old_bad,'CPU_only':True,'AnswerF1_or_deployment_gain':False,
            'scope':'Same original CPU FP16 autocast; not GPU-path equivalence or whole-model NLL gate.'}
    (p/'INDEPENDENT_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# 原CPU上下文精确性：工程修复已核验','',
          '| 对照 | 原168线性算子逐值比较 |','| --- | ---: |',
          f'| 同原CPU autocast上下文重建 | {new_exact}/168相同 |',
          f'| 旧外部重建上下文 | {old_bad}/168不相同 |','',
          '这是同一checkpoint、同一原FP16 autocast与固定合成线性输入的结果，无QA／训练／GPU。它解释了旧折叠不是原CPU算子的等价替换；不是原GPU精确性证明，不能代替v2真实整模NLL门槛，更不是新方法质量或端侧收益。',
          '',f"实际CPU两数值线程，门槛耗时{e['seconds']:.1f}秒；修复额外计算应计入研究开销，不包装为性能改善。",'',
          '原v1失败证据与K59无autocast系数数据保留。v2仍测原同10QA／48处／两种rank8，teacher见过这些训练题，不称未见任务泛化。']
    (p/'VERIFIED_REPAIR.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
