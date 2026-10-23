"""One fixed CPU-only weight approximation comparison; no model or QA paths."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_key]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import datetime, hashlib, json, math, resource, struct, sys, time

ROOT=Path('/data1/liyapeng/program/LLMCL')
OUT=ROOT/'summary/final_weight_reduction_20261023/kron_layout_priority_v1'
BASE=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors'
HEADER_SHA='874dbcdff7fd946c2f1d74b0d46059e598b03bdc2b93853f65603360a984c5a5'
RANK=200
K=69

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def write_new(name, obj):
    with (OUT/name).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())

def main():
    wall=time.monotonic();cpu=time.process_time()
    assert not any((OUT/n).exists() for n in ('STARTED.json','EVIDENCE.json','FAILURE.json'))
    declaration=json.loads((OUT/'RUN_DECLARATION.json').read_text())
    assert declaration['analysis_sha256']==sha_bytes(Path(__file__).read_bytes())
    assert declaration['question_sha256']==sha_bytes((OUT/'QUESTION.md').read_bytes())
    write_new('STARTED.json',dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
        startticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        analysis_sha256=declaration['analysis_sha256'],question_sha256=declaration['question_sha256']))
    import numpy as np
    results=[]
    with BASE.open('rb') as f:
        hsize=struct.unpack('<Q',f.read(8))[0];hbytes=f.read(hsize)
        assert sha_bytes(hbytes)==HEADER_SHA
        header=json.loads(hbytes);data_start=8+hsize
        for kind in ('gate_proj','up_proj','down_proj'):
            key='language_model.model.layers.0.mlp.'+kind+'.weight'
            entry=header[key]
            assert entry['dtype']=='F16'
            shape=entry['shape'];expected=[896,4864] if kind=='down_proj' else [4864,896]
            assert shape==expected,(key,shape)
            begin,end=entry['data_offsets'];assert end-begin==2*4864*896
            f.seek(data_start+begin);raw=f.read(end-begin);assert len(raw)==end-begin
            W=np.frombuffer(raw,dtype='<f2').reshape(shape).astype(np.float64)
            if kind=='down_proj': W=W.T.copy()
            assert W.shape==(4864,896) and np.isfinite(W).all()
            norm_sq=float(np.sum(W*W,dtype=np.float64));assert norm_sq>0
            row=dict(key=key,source_shape=shape,analysis_shape=list(W.shape),source_tensor_sha256=sha_bytes(raw),
                source_tensor_bytes=len(raw),ordinary_rank=RANK,separation_terms=K,weight_norm_sq=norm_sq,
                ordinary_factor_coefficients=RANK*(4864+896),kron_factor_coefficients=K*(38*7+128*128))
            assert row['kron_factor_coefficients']<=row['ordinary_factor_coefficients']
            for label,matrix,keep in (
                ('ordinary',W,RANK),
                ('kron',W.reshape(38,128,7,128).transpose(0,2,1,3).reshape(266,16384),K)):
                t=time.monotonic();c=time.process_time()
                singular=np.linalg.svd(matrix,compute_uv=False,full_matrices=False)
                total=float(np.dot(singular,singular));tail=float(np.dot(singular[keep:],singular[keep:]))
                assert abs(total-norm_sq)<=1e-10*norm_sq and math.isfinite(tail)
                row[label]=dict(relative_frobenius_residual=math.sqrt(tail/norm_sq),tail_norm_sq=tail,
                    wall_seconds=time.monotonic()-t,cpu_seconds=time.process_time()-c)
                del singular
            row['kron_strictly_better']=row['kron']['relative_frobenius_residual']<row['ordinary']['relative_frobenius_residual']
            for method in ('ordinary','kron'):
                row[method+'_hypothetical_FP16_factor_bytes']=2*row[method+'_factor_coefficients']
            results.append(row)
            print(json.dumps(dict(key=key,kron_strictly_better=row['kron_strictly_better'],
                ordinary_residual=row['ordinary']['relative_frobenius_residual'],kron_residual=row['kron']['relative_frobenius_residual'])),flush=True)
            del W,raw,matrix
    wins=sum(r['kron_strictly_better'] for r in results)
    report=dict(scope='fixed layer0/natural layout/r200 versus K69 priority only; not model function, F1, trainability or novelty',
        source_header_sha256=HEADER_SHA,source_header_bytes=hsize,current_tensor_snapshots_only=True,
        declared_full_base_sha256_reverified=False,run_declaration_sha256=sha_bytes((OUT/'RUN_DECLARATION.json').read_bytes()),
        analysis_sha256=declaration['analysis_sha256'],question_sha256=declaration['question_sha256'],
        numpy_version=np.__version__,analysis_dtype='float64',source_dtype='F16',numeric_threads_requested=2,
        no_torch_or_model_QA_GPU=True,results=results,kron_strict_wins=wins,
        decision='retain_only_functional_feasibility_discussion_no_model_admission' if wins==3 else 'close_fixed_K69_natural_layout_priority_point',
        mixed_result=0<wins<3,full_matrix_rank_not_computed=True,spectra_not_retained=True,
        elapsed_wall_seconds=time.monotonic()-wall,elapsed_cpu_seconds=time.process_time()-cpu,
        max_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,goal_achieved=False)
    assert 'torch' not in sys.modules
    write_new('EVIDENCE.json',report)
    print(json.dumps(dict(completed=True,decision=report['decision'],wall_seconds=report['elapsed_wall_seconds'],
        cpu_seconds=report['elapsed_cpu_seconds'],evidence_sha256=sha_bytes((OUT/'EVIDENCE.json').read_bytes()))),flush=True)

if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if not (OUT/'FAILURE.json').exists():
            write_new('FAILURE.json',dict(type=type(exc).__name__,message=str(exc),scope='CPU-only matrix analysis; stop without retry'))
        raise
