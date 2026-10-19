"""Five fixed TRAIN requests: CPU dense/uniform/author-DivPrune cost probe.

This is a proxy performance pilot, not mobile measurement or TEST Answer F1.
Standard OSFT export itself still needs GPU-to-export quality validation.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/cpu_visual_probe/v1'
EXPORT=ROOT/'summary/edge_deployment_20261019/static_export/v1/OSFT_EXPORTED.json'
VIEWS=['dense','uniform50','author_divprune50']


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()


def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def author_selector():
 import torch
 folder=ROOT/'code/DivPrune';meta=json.loads((folder/'source_manifest.json').read_text())
 rel='LLaVA/llava/model/llava_arch.py';path=folder/'source'/rel
 assert meta['revision']=='799e2d950aa01ba7860907f5a6d86061f885dca6'
 assert sha(path)==meta['files'][rel]
 cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LlavaMetaForCausalLM')
 funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['pairwise_cosine_similarity','DivPrune']]
 assert len(funcs)==2
 env={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=funcs,type_ignores=[])),str(path),'exec'),env)
 obj=types.SimpleNamespace()
 for n in funcs:setattr(obj,n.name,types.MethodType(env[n.name],obj))
 return obj


def input_function():
 # Execute the unchanged pure processor part of the frozen formal helper.
 # Cut before visual caching/bridge transport; vision must be recomputed and timed.
 from PIL import Image
 run=ROOT/'exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/OSFT/training_20261018_044103'
 source=run/'runner_source/runtime_utils.py'
 config=json.loads((run/'config.json').read_text())
 assert sha(source)==config['execution_source_sha256']['runtime_utils.py']
 tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='input_batch')
 cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Import) and any(x.name=='onevision_bridge' for x in n.names))
 assert cut>0
 fn.body=fn.body[:cut]+[ast.Return(ast.Tuple(elts=[ast.Name(id='batch',ctx=ast.Load()),ast.Name(id='image_shapes',ctx=ast.Load())],ctx=ast.Load()))]
 sys.path.insert(0,str(run/'runner_source'));import retrieval
 if hasattr(retrieval,'ROOT'):retrieval.ROOT=ROOT
 env={'Image':Image,'math':math,'ROOT':ROOT,'retrieve':retrieval.retrieve}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),env)
 return env['input_batch']


def checks():
 import torch
 torch.set_num_threads(2);torch.manual_seed(42)
 a=author_selector()
 for n in [10,33]:
  x=torch.randn(n,16);indices,_=a.DivPrune(x,n,threshold_ratio=.5)
  assert len(indices)==round(n*.5) and indices.unique().numel()==len(indices)
  assert torch.isfinite(x).all()
 # Known degenerate ties are documented, not silently fixed in author code.
 x=torch.ones(10,16);indices,_=a.DivPrune(x,10,threshold_ratio=.5)
 write(REPORT/'CPU_CHECK.json',dict(random_shape_checks=2,author_source_unmodified=True,
       identical_feature_tie_unique=int(indices.unique().numel()),
       identical_feature_tie_requested=len(indices),actual_unique_indices_gate=True,
       same_ratio_uniform_control=True,GPU_started=False))


def run(view):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 import torch,psutil
 from transformers import LlavaOnevisionForConditionalGeneration,AutoProcessor
 from transformers.generation.streamers import BaseStreamer
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
 assert not torch.cuda.is_initialized()
 source=json.loads(EXPORT.read_text());package=ROOT/source['package']
 assert json.loads((EXPORT.parent/'OSFT_RELOAD.json').read_text())['parameter_digest_exact']
 for rel,value in source['package_files'].items():assert sha(package/rel)==value['sha256']
 process=psutil.Process();before_load=process.memory_full_info()._asdict();tick=time.monotonic()
 model=LlavaOnevisionForConditionalGeneration.from_pretrained(package,torch_dtype='auto',local_files_only=True).eval()
 processor=AutoProcessor.from_pretrained(package,local_files_only=True)
 load_seconds=time.monotonic()-tick
 prepare=input_function();selector=author_selector()
 manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
 manifest=json.loads(manifest_path.read_text());selected=[]
 for task_id in [1,2,12,16,17]:
  task=manifest['tasks'][task_id-1];info=task['files']['train'];path=ROOT/info['path'];assert sha(path)==info['sha256']
  rows=[json.loads(s) for s in path.read_text().splitlines()]
  row=min(rows,key=lambda r:hashlib.sha256(('edge-profile42:'+r['id']).encode()).digest())
  selected.append((task,row))
 results=[]
 for task,row in selected:
  peak=[process.memory_info().rss];stop=threading.Event()
  def monitor():
   while not stop.wait(.02):peak[0]=max(peak[0],process.memory_info().rss)
  thread=threading.Thread(target=monitor,daemon=True);thread.start()
  start=time.monotonic()
  with torch.inference_mode():
   batch,shapes=prepare(row,task,processor,'cpu')
   ids=batch['input_ids'];original_text=ids[ids!=model.config.image_token_index].clone()
   features=model.get_image_features(batch['pixel_values'].to(next(model.vision_tower.parameters()).dtype),
       batch['image_sizes'],model.config.vision_feature_layer,model.config.vision_feature_select_strategy)
   features,_=model.pack_image_features(features,batch['image_sizes'],model.image_newline,model.config.vision_aspect_ratio)
   encode_end=time.monotonic();original_visual=len(features)
   special=(features==model.image_newline.to(features.dtype).unsqueeze(0)).all(dim=1)
   content=(~special).nonzero().flatten();structural=special.nonzero().flatten()
   select_start=time.monotonic()
   if view=='dense':chosen=torch.arange(len(features))
   else:
    keep=int(round(len(content)*.5));assert keep>0
    if view=='uniform50':chosen_content=torch.linspace(0,len(content)-1,keep).round().long()
    else:
     f=features[content].float();assert torch.isfinite(f).all() and (f.norm(dim=1)>0).all()
     chosen_content,_=selector.DivPrune(f,len(content),threshold_ratio=.5)
     assert chosen_content.unique().numel()==len(chosen_content),'Author tie/duplicate selection; stop and preserve'
    chosen=torch.cat([content[chosen_content],structural]).sort().values
   selected_features=features[chosen];select_end=time.monotonic()
   token=model.config.image_token_index;positions=(ids[0]==token).nonzero().flatten()
   assert len(positions)>0 and int(positions[-1]-positions[0]+1)==len(positions)
   first,last=int(positions[0]),int(positions[-1])+1
   ids=torch.cat([ids[:,:first],torch.full((1,len(selected_features)),token,dtype=ids.dtype),ids[:,last:]],dim=1)
   assert torch.equal(original_text,ids[ids!=token])
   assert all(i in chosen.tolist() for i in structural.tolist())
   embeds=model.get_input_embeddings()(ids)
   mask=(ids==token).unsqueeze(-1).expand_as(embeds)
   assert int(mask.sum())==selected_features.numel()
   embeds=embeds.masked_scatter(mask,selected_features.to(embeds.dtype))
   class Clock(BaseStreamer):
    def __init__(self):self.calls=0;self.tokens=[];self.times=[]
    def put(self,value):
     self.calls+=1
     if self.calls==1:return
     self.times.append(time.monotonic());self.tokens.extend(value.reshape(-1).tolist())
    def end(self):pass
   clock=Clock()
   output=model.language_model.generate(input_ids=ids,inputs_embeds=embeds,attention_mask=torch.ones_like(ids),
       max_new_tokens=8,min_new_tokens=8,do_sample=False,use_cache=True,streamer=clock,
       pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
   done=time.monotonic()
  stop.set();thread.join();peak[0]=max(peak[0],process.memory_info().rss)
  assert len(clock.tokens)==len(clock.times)==8
  record=dict(id=row['id'],task=task['task_id'],domain=task['domain'],image_shapes=shapes,
     visual_tokens_before=original_visual,visual_tokens_after=len(selected_features),structural_tokens_preserved=len(structural),
     text_tokens_unchanged=True,preprocess_and_vision_seconds=encode_end-start,selection_seconds=select_end-select_start,
     TTFT_seconds=clock.times[0]-start,decode_tokens_per_second=7/(clock.times[-1]-clock.times[0]),
     total_seconds=done-start,output_tokens=clock.tokens,forced8token_speed_probe_not_AnswerF1=True,
     peak_RSS_generation_bytes=peak[0],after_request_memory=process.memory_full_info()._asdict(),
     system_loadavg=Path('/proc/loadavg').read_text())
  results.append(record);write(REPORT/(view+'_partial.json'),dict(view=view,complete=False,requests=results))
  print(json.dumps({'view':view,'id':row['id'],'visual_before':original_visual,'visual_after':len(selected_features),
      'TTFT':record['TTFT_seconds'],'selection':record['selection_seconds']}),flush=True)
  del batch,ids,embeds,features,selected_features,output
 result=dict(complete=True,view=view,requests=results,TRAIN_requests=5,TEST_requests=0,
      CPU_model=subprocess.check_output(['lscpu'],text=True),threads=2,CPU_server_proxy=True,mobile_measured=False,
      load_seconds=load_seconds,before_load_memory=before_load,package_bytes=source['actual_package_bytes'],
      trained_checkpoint_sha256=source['checkpoint_sha256'],export_proof_sha256=sha(EXPORT),
      manifest_sha256=sha(manifest_path),source_sha256=sha(__file__),cuda_initialized=torch.cuda.is_initialized(),
      quality_not_verified=True,no_main_table_metrics_publication=True,
      scope='CPU cost feasibility; fixed half content tokens plus preserved newline; no model-weight updates')
 assert not result['cuda_initialized'];write(REPORT/(view+'_COMPLETE.json'),result)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--view',choices=VIEWS);p.add_argument('--all',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
 REPORT.mkdir(parents=True,exist_ok=True)
 if a.check:checks()
 elif a.all:
  for view in VIEWS:
   code=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--view',view],cwd=ROOT).returncode
   if code:write(REPORT/'CONTROLLER_FAILURE.json',dict(view=view,returncode=code,no_blind_retry=True));sys.exit(code)
  write(REPORT/'ALL_COMPLETE.json',dict(views=VIEWS,requests_per_view=5,TRAIN_only=True,not_AnswerF1=True))
 else:
  try:run(a.view)
  except Exception:
   write(REPORT/(a.view+'_FAILURE.json'),dict(traceback=traceback.format_exc(),no_blind_retry=True,partial_is_not_complete=True));raise
