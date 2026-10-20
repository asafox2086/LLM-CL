"""Fixed-condition two-matrix numerical attribution control, not precision tuning."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/edge_deployment_20261019/cpu_update_space_20261020/v1/zero_training_control'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(name,value):
    (REPORT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    ready=json.loads((REPORT/'READY.json').read_text())
    assert ready['source_sha256']==sha(__file__) and ready['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
    assert not (REPORT/'STARTED.json').exists()
    write('STARTED.json',{'pid':os.getpid(),'CPU_only':True})
    start=time.monotonic();teacher=ROOT/'exp/edge_teacher_transfer_20261019/onevision_osft_final'
    cp=teacher/'checkpoints/stage_19.pt'
    assert sha(cp)=='f93481daa528bdca4cd6bdb68ca6ec522698dfaf4703cb277cf5548b19aba541'
    sys.path.insert(0,str(teacher/'runner_source'))
    obj=torch.load(cp,map_location='cpu',pickle_module=pickle,weights_only=False)
    core=obj['method'].model.core
    source=json.loads((REPORT.parent/'EVIDENCE.json').read_text())
    actual={x['name']:x for x in source['OSFT_layers']}
    layers=['language_model.model.layers.0.self_attn.q_proj','language_model.model.layers.23.mlp.down_proj']
    rows=[]
    with safe_open(ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors',framework='pt',device='cpu') as reader,torch.inference_mode():
        for name in layers:
            m=core.get_submodule(name);w0=reader.get_tensor(name+'.weight');w=w0.clone()
            trained=m.kernel['reconstruct_weight_matrix'](m.svd(),torch.float32,m.output_dtype).detach()
            delta=trained.double()-w0.double();expected=actual[name]
            assert abs(float(delta.square().sum())-expected['energy'])<1e-9
            history=[]
            for step in range(1,20):
                svd=m.kernel['create_svd_dict'](w,int(min(w.shape)*.75),upcast_dtype=torch.float32,output_dtype=torch.float32)
                w=m.kernel['reconstruct_weight_matrix'](svd,torch.float32,m.output_dtype).detach()
                assert w.dtype==w0.dtype==torch.float16
                z=w.double()-w0.double();energy=float(z.square().sum())
                history.append({'decompositions':step,'zero_training_energy':energy,
                                'zero_to_actual_norm_ratio':(energy/expected['energy'])**.5})
            s=torch.linalg.svdvals(z);ze=float(z.square().sum())
            assert abs(float(s.square().sum())-ze)<1e-9
            row={'name':name,'shape':list(w.shape),'zero_training_history':history,
                 'actual_energy':expected['energy'],'zero_energy':ze,
                 'zero_to_actual_norm_ratio':(ze/expected['energy'])**.5,
                 'actual_vs_zero_cosine':float((z*delta).sum())/(ze*expected['energy'])**.5 if ze else None,
                 'actual_rank8_retention':expected['rank_energy']['8']/expected['energy'],
                 'zero_rank8_retention':float(s[:8].square().sum())/ze if ze else None,
                 'final_zero_singular_values':s.tolist()}
            rows.append(row);print(name,json.dumps({k:v for k,v in row.items() if k not in ['zero_training_history','final_zero_singular_values']}),flush=True)
    result={'status':'CPU_ZERO_TRAINING_TWO_MATRIX_COMPLETE','CPU_only':True,'QA_access':False,
            'model_forward':False,'optimizer_updates':0,'source_sha256':sha(__file__),
            'actual_evidence_sha256':sha(REPORT.parent/'EVIDENCE.json'),'protocol_sha256':sha(REPORT/'PROTOCOL.md'),
            'rows':rows,'seconds':time.monotonic()-start,'cuda_initialized':torch.cuda.is_initialized(),
            'limit':'CPU author SVD control, not a bit-exact replay of original GPU decomposition/training; no causal learned-energy percentage.'}
    assert not result['cuda_initialized'];write('EVIDENCE.json',result)
    write('COMPLETE.json',{'status':result['status'],'source_sha256':sha(__file__),'evidence_sha256':sha(REPORT/'EVIDENCE.json')})


if __name__=='__main__':
    try:main()
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        write('FAILURE.json',{'source_sha256':sha(__file__),'traceback':traceback.format_exc(),'no_blind_retry':True})
        raise
