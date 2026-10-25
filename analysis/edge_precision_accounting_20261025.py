"""Integer storage sensitivity of sealed representations; no model execution."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'summary/edge_resource_comparison_20261025/v1/PRECISION_ACCOUNTING.json'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists()
    p = ROOT/'summary/strong_pas_resource_20261025/v1/COMPLETE.json'
    q = ROOT/'summary/strong_qwen_resource_20261025/v1/COMPLETE.json'
    pv = ROOT/'summary/strong_pas_resource_20261025/v1/VERIFIED.json'
    qv = ROOT/'summary/strong_qwen_resource_20261025/v1/VERIFIED.json'
    a, b = json.loads(p.read_bytes()), json.loads(q.read_bytes())
    assert sha(pv) == 'c987ec2ac68980944a1ebdc515fab08a792667f923d142f6f8489d3725753d08'
    assert sha(qv) == 'dd97300219ecbfaeb5052c571c68a4fd61fc3dd8455ab24eff5aa8627ca05e7c'
    assert sha(p) == json.loads(pv.read_bytes())['COMPLETE_sha256']
    assert sha(q) == json.loads(qv.read_bytes())['COMPLETE_sha256']
    source=ROOT/'.deployment/pas_deploy_sources_20261022/CoIN/peft/tuners/coinmoelora.py'
    assert sha(source) == '75d46bcd7850ec3c03d13968f1e4180098d62a1da7ab905df6f9fa1d054a4b6a'
    baseline = dict(coefficients=893675552, bytes=1787351104)
    rows={}
    for name, end in [('PAS',a), ('Qwen_SeqLoRA',b)]:
        weights=end['layout']['weights']
        two_bytes=2*weights['coefficients']
        premium=weights['bytes']-two_bytes
        assert premium >= 0 and premium % 2 == 0
        rows[name]=dict(original_neural_coefficients=weights['coefficients'],
            original_neural_bytes=weights['bytes'], hypothetical_two_byte_neural_payload=two_bytes,
            persistent_precision_premium_bytes=premium, persistent_precision_premium_MiB=premium/2**20,
            equivalent_FP32_coefficient_count_if_other_weights_FP16=premium//2,
            hypothetical_two_byte_excess_over_LLaVA_neural_bytes=two_bytes-baseline['bytes'])
    assert rows['PAS']['persistent_precision_premium_bytes'] == 281542656
    assert rows['Qwen_SeqLoRA']['persistent_precision_premium_bytes'] == 2179072
    value=dict(source_proofs={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in (p,q,pv,qv)},
        source_code=dict(path='CoIN/peft/tuners/coinmoelora.py',sha256=sha(source),
                         observed='gate_router=a input follows A dtype; score uses same dtype; masked_fill uses -1e9. No converted model was attempted.'),
        folded_LLaVA=baseline, fixed_representations=rows,
        PAS_required_masks_bytes_separate=a['required_mask_bytes'],
        hypothetical_Qwen_folded_FP16_base=dict(coefficients=2208985600,bytes=4417971200),
        conditions='Two-byte neural payload preserves every neural coefficient, without pruning/sharing/reencoding. Required masks and ordinary buffers are separate. This is a conditional storage accounting, not a model conversion or universal deployment lower bound.',
        limitations='No measured common-dtype GPU peak or quality. Precision premium must not be subtracted from observed GPU peaks. Existing five-window peaks remain original-dtype implementation measurements.',
        action='No additional model or cast-only PAS probe. Await full ARC quality before deciding any further experiment.',
        new_models=0,new_QA=0,new_scores=0,checkpoint_payload_bytes_read=0)
    with OUT.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(output_sha256=sha(OUT),storage_precision_premium_MiB={n:x['persistent_precision_premium_MiB'] for n,x in rows.items()},new_models=0)))


if __name__ == '__main__':
    main()
