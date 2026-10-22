"""One integer-only accounting of implicit full-complement representations."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/implicit_core_feasibility_20261022/v1'
INPUT=ROOT/'summary/edge_deployment_20261019/update_geometry_literature_20261022/PROTECTED_BLOCK_COUNTS.json'
EXPLICIT=ROOT/'summary/edge_deployment_20261019/subspace_parameterization_feasibility_20261022/STORAGE.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (REPORT/'COUNTS.json').exists()
    prior=json.loads(INPUT.read_text());explicit=json.loads(EXPLICIT.read_text());rows=[]
    assert sha(INPUT)==explicit['source_sha256']
    for name,x in prior['by_position'].items():
        m,n,h,r,L=x['m'],x['n'],x['preserved_rank'],x['trainable_rank'],x['layers']
        p,q=m-h,n-h;assert min(p,q)==r
        core=p*q
        # geqrf arrays keep R in their upper triangles, reflectors below it.
        # Never add an extra stored R matrix; high B is formed as a temporary.
        qr_arrays=h*(m+n)
        tau=2*h
        original=(h+r)*(m+n+1)
        original_trainable=r*(m+n+1)
        direct=qr_arrays+tau+h+core  # S_high retained; B=R_u diag(S_h) R_v^T temporary.
        offset=qr_arrays+tau+core   # W_start FP16 separately; no S_high/R copies required.
        assert direct==m*n+h*h+3*h and offset==m*n+h*h+2*h
        row=dict(position=name,layers=L,shape=[m,n],core_shape=[p,q],
            original_USV_FP32_bytes=L*4*original,original_trainable_gradient_FP32_bytes=L*4*original_trainable,
            original_two_Adam_moments_CPU_FP32_bytes=L*8*original_trainable,
            direct_state_FP32_bytes=L*4*direct,offset_state_FP32_bytes=L*4*offset,
            offset_anchor_FP16_bytes=L*2*m*n,core_gradient_FP32_bytes=L*4*core,
            core_two_Adam_moments_CPU_FP32_bytes=L*8*core,
            direct_high_B_temporary_FP32_bytes=4*h*h,
            one_full_weight_FP32_bytes=4*m*n,one_full_weight_FP16_bytes=2*m*n)
        row['original_state_plus_gradient_bytes']=row['original_USV_FP32_bytes']+row['original_trainable_gradient_FP32_bytes']
        row['direct_state_plus_gradient_bytes']=row['direct_state_FP32_bytes']+row['core_gradient_FP32_bytes']
        row['offset_state_plus_gradient_bytes']=row['offset_state_FP32_bytes']+row['offset_anchor_FP16_bytes']+row['core_gradient_FP32_bytes']
        row['original_state_gradient_moments_bytes']=row['original_state_plus_gradient_bytes']+row['original_two_Adam_moments_CPU_FP32_bytes']
        row['direct_state_gradient_moments_bytes']=row['direct_state_plus_gradient_bytes']+row['core_two_Adam_moments_CPU_FP32_bytes']
        row['offset_state_gradient_moments_bytes']=row['offset_state_plus_gradient_bytes']+row['core_two_Adam_moments_CPU_FP32_bytes']
        rows.append(row)
    keys=[k for k in rows[0] if k.endswith('_bytes') and not k.startswith(('one_full','direct_high_B'))]
    totals={k:sum(row[k] for row in rows) for k in keys}
    assert totals['original_USV_FP32_bytes']==explicit['FP32_tensor_bytes']['original_all_USV_elements']
    assert totals['original_trainable_gradient_FP32_bytes']==4*prior['totals']['raw_coordinates']
    assert totals['core_gradient_FP32_bytes']==4*prior['totals']['full_core_coordinates']
    out=dict(kind='new_implicit_representation_integer_accounting_not_runtime_measurement',by_position=rows,totals_bytes=totals,
        direct_state_ratio=totals['direct_state_FP32_bytes']/totals['original_USV_FP32_bytes'],
        offset_state_ratio=(totals['offset_state_FP32_bytes']+totals['offset_anchor_FP16_bytes'])/totals['original_USV_FP32_bytes'],
        offset_GPU_state_gradient_ratio=totals['offset_state_plus_gradient_bytes']/totals['original_state_plus_gradient_bytes'],
        offset_mixed_device_state_gradient_moments_ratio=totals['offset_state_gradient_moments_bytes']/totals['original_state_gradient_moments_bytes'],
        scope='Target matrices only. FP32 QR arrays/tau/core and original FP32 factors; optional frozen anchor FP16. State+gradient counts available during a stage; HostAdamW moments counted on CPU separately. Mixed-device sum is byte bookkeeping, never a CPU/GPU peak. Excludes shared backbone/bias, optimizer scalar/transfers, activations/checkpointing/offload, reconstructed dense matrices, QR/SVD/ormqr workspace, allocator and I/O. Per-matrix visible temporary sizes are not multiplied into a claimed actual training peak.',
        reused_coordinate_counts_not_new_discovery=True,initial_forward_alignment_not_proved=True,
        no_READY_or_worker=True,new_model_or_QA_or_tensor_or_operator_or_GPU_requests=0,
        source_sha256=sha(__file__),input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [INPUT,EXPLICIT,REPORT/'QUESTION.md']})
    (REPORT/'COUNTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['by_position','input_sha256']}))
if __name__=='__main__':main()
