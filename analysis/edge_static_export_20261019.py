"""CPU-only canonical export audit of three sealed OneVision baselines.

These are ordinary baseline deployment optimizations, not a proposed algorithm.
No TEST files are opened; no model F1 is inferred from local operator fidelity.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'exp/edge_static_export_20261019/v1'
REPORT=ROOT/'summary/edge_deployment_20261019/static_export/v1'
RUNS={
 'SeqFT_LoRA':'training_20261017_161154',
 'OSFT':'training_20261018_044103',
 'PaRSP':'training_20261018_040230',
}
EXPECTED={
 'SeqFT_LoRA':'911d8937d74bb2525bcabd630e9fc28528e3aa1cc8413b293a49c90e3ad6b5a5',
 'OSFT':'f93481daa528bdca4cd6bdb68ca6ec522698dfaf4703cb277cf5548b19aba541',
 'PaRSP':'20c89257ceb48e9dce314dba243e05527565615d8269dac4f19dc258bd6f3f18',
}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()


def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def tensor_bytes(values):
 seen=set();total=0
 for x in values:
  storage=x.untyped_storage();key=(storage.data_ptr(),storage.nbytes())
  if key not in seen:seen.add(key);total+=storage.nbytes()
 return total


def container_tensors(value):
 import torch
 if torch.is_tensor(value):yield value
 elif isinstance(value,dict):
  for v in value.values():yield from container_tensors(v)
 elif isinstance(value,(tuple,list)):
  for v in value:yield from container_tensors(v)


def digest_parameters(model):
 h=hashlib.sha256()
 for name,p in model.named_parameters():
  h.update(json.dumps([name,list(p.shape),str(p.dtype)]).encode())
  h.update(p.detach().contiguous().numpy().tobytes())
 return h.hexdigest()


def cpu_info():
 return {'lscpu':subprocess.check_output(['lscpu'],text=True),
         'loadavg':Path('/proc/loadavg').read_text(),'torch_threads':2,
         'server_CPU_proxy_only':True,'mobile_device_measured':False}


def audit(method):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import torch
 import torch.nn.functional as F
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
 assert not torch.cuda.is_initialized()
 run=ROOT/'exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01'/method/RUNS[method]
 meta=json.loads((run/'checkpoints/latest.json').read_text());checkpoint=run/meta['path']
 assert meta['stage']==19 and sha(checkpoint)==meta['sha256']==EXPECTED[method]
 sys.path.insert(0,str(run/'runner_source'))
 tick=time.monotonic()
 payload=torch.load(checkpoint,map_location='cpu',pickle_module=pickle,weights_only=False)
 assert payload['stage']==19 and payload['config']['backbone']=='onevision'
 model=payload['method'].model;model.eval();core=model.core
 assert not any(p.device.type!='cpu' for p in core.parameters())
 report={'method':method,'checkpoint':str(checkpoint.relative_to(ROOT)),
   'checkpoint_sha256':meta['sha256'],'checkpoint_bytes_NOT_deployment':meta['bytes'],
   'analysis_source_sha256':sha(__file__),'CPU':cpu_info(),
   'load_checkpoint_seconds':time.monotonic()-tick,
   'model_parameter_buffer_storage_bytes_before':tensor_bytes(list(core.parameters())+list(core.buffers())),
   'non_model_explicit_training_tensor_attributes':{},'local_operator_checks':[],
   'microbenchmarks':[],'new_model_training':False,'TEST_access':False,
   'not_an_algorithm_innovation':True,'full_quality_verified':False}
 for name,value in vars(payload['method']).items():
  if name=='model':continue
  ts=list(container_tensors(value))
  if ts:report['non_model_explicit_training_tensor_attributes'][name]=tensor_bytes(ts)
 # Predetermined first decoder layer, query and down projections, is only a
 # CPU operator microbenchmark. No layer is selected using TEST performance.
 targets=['language_model.model.layers.0.self_attn.q_proj',
          'language_model.model.layers.0.mlp.down_proj']
 with torch.inference_mode():
  for name in targets:
   module=core.get_submodule(name)
   if type(module).__name__=='OSFTLinear':
    w=module.kernel['reconstruct_weight_matrix'](module.svd(),torch.float32,module.output_dtype).detach()
   elif hasattr(module,'A') and hasattr(module,'B'):
    w=(module.weight.float()+module.scale*(module.B.weight.float()@module.A.weight.float())).to(module.weight.dtype).detach()
   else:w=module.weight.detach()
   for length in [1,128]:
    x=torch.randn(length,w.shape[1],dtype=w.dtype)*.1
    dense=lambda:F.linear(x,w,module.bias)
    reference=lambda:module(x)
    r,d=reference(),dense()
    error=float((r.float()-d.float()).abs().max())
    exact=torch.equal(r,d)
    # No tolerance is used to turn an approximate merge into an exact claim.
    report['local_operator_checks'].append(dict(module=name,tokens=length,exact=exact,
       max_abs_error=error,requires_full_quality_reassessment=not exact))
    reference();dense()
    times={'original':[],'static':[]}
    for iteration in range(7):
     pairs=[('original',reference),('static',dense)]
     if iteration%2:pairs.reverse()
     for label,fn in pairs:
      start=time.perf_counter();fn();times[label].append(time.perf_counter()-start)
    report['microbenchmarks'].append(dict(module=name,tokens=length,
        original_seconds=times['original'],static_seconds=times['static'],
        median_original=statistics.median(times['original']),median_static=statistics.median(times['static']),
        microbenchmark_only_not_TTFT=True))
  # Standard inference export applied equally, not solely to the high-scoring
  # method. SeqFT merging is marked approximate if any local output changes.
  replacements=[]
  for name,module in list(core.named_modules()):
   kind=type(module).__name__
   if kind=='OSFTLinear':
    w=module.kernel['reconstruct_weight_matrix'](module.svd(),torch.float32,module.output_dtype).detach()
   elif hasattr(module,'A') and hasattr(module,'B') and hasattr(module,'scale'):
    w=(module.weight.float()+module.scale*(module.B.weight.float()@module.A.weight.float())).to(module.weight.dtype).detach()
   else:continue
   x=torch.randn(2,w.shape[1],dtype=w.dtype)*.1
   target=torch.nn.Linear(w.shape[1],w.shape[0],bias=module.bias is not None,device='cpu',dtype=w.dtype)
   target.weight=torch.nn.Parameter(w,requires_grad=False)
   if module.bias is not None:target.bias=torch.nn.Parameter(module.bias.detach().clone(),requires_grad=False)
   old,new=module(x),target(x)
   exact=torch.equal(old,new)
   if kind=='OSFTLinear':assert exact,'Static reconstructed weight must preserve this CPU operator exactly'
   replacements.append(dict(module=name,kind=kind,exact_on_fixed_inputs=exact,
       max_abs_error=float((old.float()-new.float()).abs().max())))
   par,child=name.rsplit('.',1);setattr(core.get_submodule(par),child,target)
  report['replacements']=replacements
  # PaRSP's original trainable weights may be FP32. Keep their dtype here;
  # reload must preserve it, otherwise the separate reload gate will reject.
  report['canonical_parameter_dtypes']={str(t.dtype):sum(p.numel() for p in core.parameters() if p.dtype==t.dtype)
                                      for t in core.parameters()}
  report['model_parameter_buffer_storage_bytes_after']=tensor_bytes(list(core.parameters())+list(core.buffers()))
  report['canonical_parameter_sha256']=digest_parameters(core)
  package=WORK/method/'deployment';assert not package.exists(),'No export overwrite'
  package.mkdir(parents=True)
  core.save_pretrained(package,safe_serialization=True,max_shard_size='2GB')
  payload['processor'].save_pretrained(package)
  report['package']=str(package.relative_to(ROOT))
  report['package_files']={str(p.relative_to(package)):{'bytes':p.stat().st_size,'sha256':sha(p)}
                           for p in sorted(package.rglob('*')) if p.is_file()}
  report['actual_package_bytes']=sum(x['bytes'] for x in report['package_files'].values())
  report['all_original_checkpoint_files_unchanged']=checkpoint.stat().st_size==meta['bytes']
  report['cuda_initialized']=torch.cuda.is_initialized();assert not report['cuda_initialized']
  report['operator_scope_not_full_AnswerF1']=True
  write(REPORT/f'{method}_EXPORTED.json',report)
  print(json.dumps(dict(method=method,exported=True,package_bytes=report['actual_package_bytes'],
       replaced=len(replacements),local_exact=sum(r['exact_on_fixed_inputs'] for r in replacements))),flush=True)


def reload(method):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import torch
 from transformers import LlavaOnevisionForConditionalGeneration,AutoProcessor
 torch.set_num_threads(2)
 before=json.loads((REPORT/f'{method}_EXPORTED.json').read_text());package=ROOT/before['package']
 for rel,v in before['package_files'].items():assert sha(package/rel)==v['sha256']
 # dtype=None preserves the default FP32 model and casts half tensors. We use
 # auto for the all-half OSFT/merged-SeqFT exports; mixed PaRSP remains a gate.
 m=LlavaOnevisionForConditionalGeneration.from_pretrained(package,torch_dtype='auto',local_files_only=True)
 proc=AutoProcessor.from_pretrained(package,local_files_only=True)
 digest=digest_parameters(m)
 same=digest==before['canonical_parameter_sha256']
 result=dict(method=method,parameter_digest_exact=same,actual_digest=digest,
    expected_digest=before['canonical_parameter_sha256'],CPU_only=True,
    cuda_initialized=torch.cuda.is_initialized(),processor_reloaded=proc is not None,
    full_quality_verified=False,scope='ordinary HF CPU reload, not AnswerF1 or latency')
 write(REPORT/f'{method}_RELOAD.json',result)
 assert same,'Reload dtype/parameter change; preserve evidence, do not claim equivalent deployment'
 print(json.dumps(result),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--method',choices=list(RUNS));p.add_argument('--reload',action='store_true');p.add_argument('--all',action='store_true');a=p.parse_args()
 REPORT.mkdir(parents=True,exist_ok=True)
 if a.all:
  # All declared methods are attempted; a non-identical PaRSP reload is kept
  # as a negative result and does not silently change its precision recipe.
  results={}
  for method in RUNS:
   command=[sys.executable,'-u',str(Path(__file__).resolve()),'--method',method]
   r=subprocess.run(command,cwd=ROOT)
   results[method]={'export_returncode':r.returncode}
   if r.returncode==0:
    r=subprocess.run(command+['--reload'],cwd=ROOT)
    results[method]['reload_returncode']=r.returncode
  write(REPORT/'CONTROLLER_FINISHED.json',dict(attempted_all_declared_methods=True,results=results,
       no_GPU_or_training_started=True,not_complete_F1_or_deployment_profile=True))
 else:
  try:
   if a.reload:reload(a.method)
   else:audit(a.method)
  except Exception:
   write(REPORT/f'{a.method}_{"RELOAD" if a.reload else "EXPORT"}_FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True))
   raise


if __name__=='__main__':main()
