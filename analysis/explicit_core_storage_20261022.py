"""Shape-only feasibility accounting, no tensor/model/QA allocation or performance claim."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'summary/edge_deployment_20261019/update_geometry_literature_20261022/PROTECTED_BLOCK_COUNTS.json'
REPORT=ROOT/'summary/edge_deployment_20261019/subspace_parameterization_feasibility_20261022'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (REPORT/'STORAGE.json').exists()
    v=json.loads(SOURCE.read_text());rows=[]
    for name,x in v['by_position'].items():
        m,n,h,k,L=x['m'],x['n'],x['preserved_rank'],x['trainable_rank'],x['layers']
        assert min(m,n)==h+k and k==min(m,n)//4
        original=L*(h+k)*(m+n+1)
        core=L*(m-h)*(n-h)
        high_factor=L*h*(m+n+1)
        complement=L*(m*(m-h)+n*(n-h))
        high_dense=L*m*n
        rows.append(dict(position=name,layers=L,shape=[m,n],full_core_shape=[m-h,n-h],
          original_all_USV_elements=original,original_trainable_elements=x['raw_coordinates'],
          complete_core_trainable_elements=core,explicit_complement_basis_elements=complement,
          high_factor_elements=high_factor,high_dense_elements=high_dense,
          core_with_high_factors_elements=high_factor+complement+core,
          core_with_dense_high_elements=high_dense+complement+core,
          per_position_cheaper_explicit_core_elements=min(high_factor,high_dense)+complement+core,
          tail_s_only_trainable_elements=L*k))
    sums={k:sum(x[k] for x in rows) for k in rows[0] if k.endswith('_elements')}
    assert sums['original_trainable_elements']==113766144 and sums['complete_core_trainable_elements']==71245824
    assert sums['tail_s_only_trainable_elements']==28416
    fp32={k:v*4 for k,v in sums.items()}
    moments=dict(original_USV_two_moments_bytes=2*4*sums['original_trainable_elements'],
        complete_core_two_moments_bytes=2*4*sums['complete_core_trainable_elements'],
        tail_s_only_two_moments_bytes=2*4*sums['tail_s_only_trainable_elements'])
    out=dict(kind='shape_only_explicit_representation_planning_not_runtime_measurement',
       source=SOURCE.relative_to(ROOT).as_posix(),source_sha256=sha(SOURCE),by_position=rows,
       totals_elements=sums,FP32_tensor_bytes=fp32,two_Adam_moment_tensor_bytes=moments,
       complete_core_min_explicit_weight_ratio=sums['per_position_cheaper_explicit_core_elements']/sums['original_all_USV_elements'],
       trainable_coordinate_reduction_fraction=1-sums['complete_core_trainable_elements']/sums['original_trainable_elements'],
       scope='Seven target positions only. Explicit full complements with either original high factors or dense high; per-position minimum among these two constructions, not a universal memory lower bound. Parameters/buffers FP32 as original OSFT factors. Two same-dtype Adam moments counted separately (original HostAdamW retains them on CPU). No combined GPU/RAM peak; excludes shared model, gradients, activations, SVD/QR/workspaces, allocator, scalar counters and transient moment transfers. Other implicit representations may differ.',
       reused_coordinate_counts_not_new_discovery=True,
       source_analysis_sha256=sha(__file__),new_model_or_QA_or_tensor_or_GPU_requests=0)
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'STORAGE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(totals=sums,FP32_bytes=fp32,moments=moments,
       explicit_weight_ratio=out['complete_core_min_explicit_weight_ratio'],
       coordinate_reduction_fraction=out['trainable_coordinate_reduction_fraction'])))
if __name__=='__main__':main()
