"""One repair gate: preserve the existing CPU autocast semantics on export."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'summary/edge_deployment_20261019/cpu_function_space_20261020/v2/repair_gate'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(n,v):
    D.mkdir(parents=True,exist_ok=True)
    (D/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    ready=json.loads((D/'READY.json').read_text());assert ready['source_sha256']==sha(__file__)
    assert not (D/'STARTED.json').exists()
    write('STARTED.json',{'pid':os.getpid(),'CPU_only':True});start=time.monotonic()
    teacher=ROOT/'exp/edge_teacher_transfer_20261019/onevision_osft_final'
    cp=teacher/'checkpoints/stage_19.pt';expected='f93481daa528bdca4cd6bdb68ca6ec522698dfaf4703cb277cf5548b19aba541'
    assert sha(cp)==expected;sys.path.insert(0,str(teacher/'runner_source'))
    obj=torch.load(cp,map_location='cpu',pickle_module=pickle,weights_only=False)
    g=torch.Generator().manual_seed(42);rows=[]
    with torch.inference_mode():
        for name,m in obj['method'].model.core.named_modules():
            if type(m).__name__!='OSFTLinear':continue
            x=torch.randn(2,m.in_features,generator=g,dtype=torch.float16)*.1
            no_context=m.kernel['reconstruct_weight_matrix'](m.svd(),torch.float32,m.output_dtype).detach()
            with torch.autocast('cpu',dtype=torch.float16):
                reference=m(x)
                matching=m.kernel['reconstruct_weight_matrix'](m.svd(),torch.float32,m.output_dtype).detach()
                repaired=F.linear(x,matching,m.bias)
                old_fold=F.linear(x,no_context,m.bias)
            same=torch.equal(reference,repaired)
            rows.append({'name':name,'original_vs_matching_context_exact':same,
                'original_vs_outside_context_exact':torch.equal(reference,old_fold),
                'max_abs_outside_context_error':float((reference.float()-old_fold.float()).abs().max()),
                'weights_exact_across_contexts':torch.equal(no_context,matching),
                'no_context_weight_sha256':hashlib.sha256(no_context.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                'matching_context_weight_sha256':hashlib.sha256(matching.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
            assert same,name
            print('MATCHED_OPERATOR',len(rows),name,flush=True)
    assert len(rows)==168 and not torch.cuda.is_initialized()
    assert any(not x['weights_exact_across_contexts'] for x in rows)
    write('EVIDENCE.json',{'status':'CPU_CONTEXT_REPAIR_ALL168_EXACT','source_sha256':sha(__file__),
        'checkpoint_sha256':expected,'rows':rows,'CPU_only':True,'QA_access':False,'full_model_forward':False,
        'synthetic_linear_calls':504,'optimizer_updates':0,'seconds':time.monotonic()-start,
        'scope':'Repair engineering only; same existing FP16 autocast, not precision tuning or F1 evidence.'})
    write('COMPLETE.json',{'status':'CPU_CONTEXT_REPAIR_ALL168_EXACT','source_sha256':sha(__file__),
                           'evidence_sha256':sha(D/'EVIDENCE.json'),'operators':168})


if __name__=='__main__':
    try:main()
    except Exception:
        write('FAILURE.json',{'source_sha256':sha(__file__),'traceback':traceback.format_exc(),'no_blind_retry':True})
        raise
