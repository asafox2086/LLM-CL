"""Reconstruct one sealed PAS static fold and preserve a complete model package."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import weakref

SCHEMA='pas-static-final-confirmation-v1'
CP_SHA='583ab5d02e187fbd6a54bd4a05a019e33101deb8af2d9533795253140721441a'
CAL_SHA='c55d1a97b19fb839c4f8cc6056876180ff9b8831035501c70b68589293641e84'
FOLD_DIGEST='3ea480aec723afb0d8660853c274312b9aab0580bc7674f96d35d6f7180c7683'
CONFIG_SHA='79edcca96dd315260816932438b7c0ec57835f5761d2ec0b88d8ecdb5c837fb6'
MANIFEST_SHA='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(8*1024**2),b''):h.update(value)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def validate(binding):
    b=read(binding);assert b['schema']==SCHEMA
    code=Path(__file__).parent
    assert set(b['sources'])=={'export.py','evaluate.py','verify.py','deployment.py'}
    for name,h in b['sources'].items():assert sha(code/name)==h,name
    for prefix in ['protocol','CALIBRATION','probe_RESULT','probe_verified','storage_helper','split','pas_memory','readouts','batcher','metric_lock','pas_resource_worker']:
        p=Path(b[prefix+'_path']);assert p.is_absolute() and sha(p)==b[prefix+'_sha256'],prefix
    assert read(b['metric_lock_path'])['score_field']=='token_f1'
    assert b['CALIBRATION_sha256']==CAL_SHA and b['export_timeout_seconds']==600 and b['evaluation_timeout_seconds']>0
    assert Path(b['source_root']).is_absolute() and Path(b['source_dir']).is_absolute() and Path(b['original_run']).is_absolute()
    root=Path(b['source_root']);run=Path(b['original_run']);port=Path(b['source_dir'])
    assert sha(run/'config.json')==b['config_sha256']==CONFIG_SHA
    config=read(run/'config.json');assert config['method']=='PASs_MoE' and config['backbone']=='onevision' and config['condition']=='pct01'
    assert config['manifest_sha256']==b['manifest_sha256']==MANIFEST_SHA
    assert sha(b['manifest_path'])==MANIFEST_SHA
    for name,h in config['execution_source_sha256'].items():assert sha(port/name)==sha(run/'runner_source'/name)==h,name
    for name,h in b['runtime_files'].items():assert sha(root/name)==h,name
    meta=read(run/'checkpoints/latest.json');assert meta['stage']==19 and meta['sha256']==CP_SHA==b['checkpoint_sha256'] and meta['bytes']==3051292733
    assert (run/meta['path']).stat().st_size==meta['bytes']
    cal=read(b['CALIBRATION_path']);probe=read(b['probe_RESULT_path'])
    verified=read(b['probe_verified_path']);assert verified['RESULT_sha256']==b['probe_RESULT_sha256']
    assert verified['verifier_sha256']=='4b4d7e92af72d7a95831ccd59127a2b93783b0f45a257f8f6311664f80a9eb6a'
    assert probe['calibration_sha256']==CAL_SHA and probe['checkpoint_sha256']==CP_SHA
    assert probe['arms']['constant_folded']['state_digest']==FOLD_DIGEST and len(cal['constants'])==168
    assert all(v['mask']==[1.,1.] and len(v['c'])==2 and abs(sum(v['c'])-1)<1e-12 for v in cal['constants'].values())
    return b,root,run,port,config,cal,probe

def identity():
    s=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    return dict(pid=os.getpid(),startticks=s[19],uid=os.getuid(),pgid=os.getpgrp())

def idle_gpu(b):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    q=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    assert q[0].strip()==b['GPU_uuid'] and int(q[1])<500 and int(q[2])<10
    assert not subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    b,root,run,port,config,cal,probe=validate(a.binding);assert a.output.is_absolute() and not a.output.exists()
    if not a.execute:print(json.dumps(dict(metadata_validated=True,model_loaded=False)));return
    idle_gpu(b);a.output.mkdir(parents=True,exist_ok=False);out=a.output;who=identity();assert who['pid']==who['pgid'];tick=time.monotonic()
    write(out/'STARTED.json',dict(identity=who,binding_sha256=sha(a.binding),unix=time.time(),limit_seconds=600))
    def expired(*_):raise TimeoutError('PAS fixed static export600 deadline expired')
    signal.signal(signal.SIGALRM,expired);signal.alarm(600)
    try:
        import torch
        torch.set_num_threads(8);torch.cuda.set_device(0);assert torch.cuda.device_count()==1
        sys.path.insert(0,str(port));import checkpoints
        payload=checkpoints.load(run,expected=config);assert payload['stage']==19 and payload['probe'] is None and payload['format']=='llmcl-method-graph-v1'
        method=payload['method'];model=method.model.eval();processor=payload['processor'];del payload
        model.requires_grad_(False);storage=module('pas_final_storage',b['storage_helper_path'])
        deployment=module('pas_final_deployment',Path(__file__).parent/'deployment.py')
        assert storage.digest(model.core)==probe['arms']['dynamic']['state_digest']
        before_generation={key:dict(public=value.to_dict(),semantics=deployment.generation_semantics(value)) for key,value in [('outer',model.core.generation_config),('language',model.core.language_model.generation_config)]}
        layers={name:value for name,value in model.core.named_modules() if hasattr(value,'_current_E')};assert set(layers)==set(cal['constants']) and len(layers)==168
        refs=[weakref.ref(value) for value in layers.values()]
        for name,m in layers.items():
            assert m._expert_mask.detach().cpu().tolist()==[1.,1.] and m._current_E()==2
            adapter=m.active_adapter
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                w=m.weight.detach().float().clone()
                for e,c in enumerate(cal['constants'][name]['c']):
                    if m._expert_mask[e]>.5:w.add_(m.lora_B[adapter].loraB[e].weight.float()@m.As_weight(e).float(),alpha=m.scaling[adapter]*c)
                linear=torch.nn.Linear(m.in_features,m.out_features,bias=m.bias is not None,device=m.weight.device,dtype=m.weight.dtype)
                linear.weight.copy_(w.to(m.weight.dtype))
                if m.bias is not None:linear.bias.copy_(m.bias)
                linear.requires_grad_(False);linear.eval()
            parent,child=name.rsplit('.',1);setattr(model.core.get_submodule(parent),child,linear)
        method.layers=[];method.learner=None;del m,layers;gc.collect();torch.cuda.empty_cache()
        assert all(ref() is None for ref in refs) and not any(hasattr(value,'_current_E') for value in model.core.modules())
        assert storage.digest(model.core)==FOLD_DIGEST
        assert storage.layout(model.core)==probe['arms']['constant_folded']['layout']
        receipt=deployment.export(model,out/'model.pt',storage,response_installed=False)
        with (out/'model.pt').open('rb') as stream:os.fsync(stream.fileno())
        assert receipt['digest']==FOLD_DIGEST and receipt['external_base_weights_required'] is False
        processor.save_pretrained(out/'processor')
        processor_files={str(p.relative_to(out/'processor')):sha(p) for p in (out/'processor').rglob('*') if p.is_file()}
        assert processor_files
        for key,value in [('outer',model.core.generation_config),('language',model.core.language_model.generation_config)]:
            assert before_generation[key]==dict(public=value.to_dict(),semantics=deployment.generation_semantics(value))
        write(out/'EXPORT.json',dict(checkpoint=receipt,processor_files=processor_files,source_checkpoint_sha256=CP_SHA,CALIBRATION_sha256=CAL_SHA,fold_digest=FOLD_DIGEST,old_PAS_modules_released=True,training_learner_removed=True,generation_before=before_generation,new_calibration=0,new_generations=0,optimizer_updates=0,backwards=0))
        write(out/'COMPLETE.json',dict(complete=True,identity=who,binding_sha256=sha(a.binding),EXPORT_sha256=sha(out/'EXPORT.json'),seconds=time.monotonic()-tick))
        signal.alarm(0)
    except BaseException:write(out/'FAILURE.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
