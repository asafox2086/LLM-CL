"""Separate source-approved asset acquisition and one CPU admission. Never activates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request
from core import ROOT,PORT,REPORT,BASE,MODEL_ID,REVISION,ATTENTION,CASES,BUDGET,TRAINABLE,study,sha,write
from common import SOURCES,data

ASSETS=('config.json','generation_config.json','preprocessor_config.json','processor_config.json',
 'special_tokens_map.json','tokenizer_config.json','tokenizer.json','vocab.json','merges.txt','added_tokens.json','chat_template.json','model.safetensors')

def approved():
 review=json.loads((REPORT/'SOURCE_REVIEW.json').read_text())
 assert review['decision']=='accept_source_for_assets_and_single_CPU_gate_only'
 for n,h in review['sources'].items():assert sha(ROOT/n)==h,n
 assert set(review['sources'])=={(Path(__file__).parent/n).relative_to(ROOT).as_posix() for n in SOURCES}
 assert review['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
 assert review['implementation_spec_sha256']==sha(REPORT/'IMPLEMENTATION_SPEC.md')
 assert review['repair_review_sha256']==sha(REPORT.parent/'native_response_reuse_pilot_v1/COMPATIBILITY_REPAIR_REVIEW.json')
 return review

def acquire():
 raise RuntimeError('v2 reuses verified assets; no second download')

def cpu_gate():
 import copy,tempfile,torch
 from transformers import Idefics3Config,Idefics3ForConditionalGeneration,get_cosine_schedule_with_warmup
 from core import make_method,compile_final,restore_compiled,save_tensor_file,cpu_tree,digest,observed_optimizer,layout
 sys.path.insert(0,str(PORT));from runtime_utils import make_optimizer
 assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
 torch.set_num_threads(2);torch.manual_seed(42)
 cfg=Idefics3Config(image_token_id=63,scale_factor=2,vision_config=dict(hidden_size=32,intermediate_size=64,
  num_hidden_layers=1,num_attention_heads=4,num_channels=3,image_size=8,patch_size=4),
  text_config=dict(model_type='llama',vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
   num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,tie_word_embeddings=False))
 base=Idefics3ForConditionalGeneration._from_config(cfg,torch_dtype=torch.float32,attn_implementation=dict(ATTENTION)).eval()
 assert base.model.vision_model.config._attn_implementation=='eager'
 assert base.model.text_model.config._attn_implementation=='sdpa'
 batch=dict(input_ids=torch.tensor([[1,63,4,5]]),pixel_values=torch.randn(1,1,3,8,8),
  pixel_attention_mask=torch.ones(1,1,8,8,dtype=torch.bool),use_cache=False)
 with torch.no_grad():parent=base(**batch).logits
 outcomes={};qv=None
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)
  for case in CASES:
   model=copy.deepcopy(base);method=make_method(model,case,root/case/'method_state');model.eval()
   with torch.no_grad():assert torch.equal(model(**batch).logits,parent)
   common=digest([p for n,p in model.named_parameters() if '.A.' in n or '.B.' in n])
   if qv is None:qv=common
   assert qv==common
   before=digest([p for p in model.parameters() if p.requires_grad]);frozen=[p for p in model.parameters() if not p.requires_grad];stable=digest(frozen)
   reference=None
   if case=='E':
    from types import MethodType
    reference=make_method(copy.deepcopy(base),'E',root/'author_reference/method_state')
    for layer in reference.expert_layers:
     # The reference bypasses Hosted observation/offload contexts and calls the author forward directly.
     author_forward=type(layer).__mro__[1].forward;layer.forward=MethodType(author_forward,layer)
    assert digest(model.state_dict().values())==digest(reference.model.state_dict().values())
   for stage in (1,2):
    method.begin_stage(stage);opt=observed_optimizer(method,make_optimizer,root/case);scheduler=get_cosine_schedule_with_warmup(opt,0,2)
    if reference is not None:
     reference.learner.before_train(stage,reference.model)
     for layer in reference.expert_layers:layer.gate_tau=1.5
     ropt=make_optimizer(reference);rsched=get_cosine_schedule_with_warmup(ropt,0,2)
    for step in (1,2):
     opt.zero_grad(set_to_none=True);model.train();method.recording(True)
     rng=torch.get_rng_state()
     loss=model(**batch).logits.float().square().mean();method.recording(False);loss.backward();method.post_backward(1.,(step-1)/2)
     if reference is not None:
      after=torch.get_rng_state();torch.set_rng_state(rng);ropt.zero_grad(set_to_none=True);reference.model.train()
      reference.learner.enable_record(True);rloss=reference.model(**batch).logits.float().square().mean();reference.learner.enable_record(False)
      rloss.backward();reference.learner.post_backward(reference.model,cl_task_progress=(step-1)/2)
      assert torch.equal(loss,rloss) and torch.equal(after,torch.get_rng_state())
      right=dict(reference.model.named_parameters())
      for n,p in model.named_parameters():
       assert (p.grad is None)==(right[n].grad is None)
       if p.grad is not None:assert torch.equal(p.grad,right[n].grad)
     assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
     if stage==step==1:
      path=root/case/'absent'/'gradients/first.pt';grads={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
      save_tensor_file(path,grads);saved=torch.load(path,weights_only=True)
      assert saved.keys()==grads.keys() and all(saved[n].dtype==g.dtype and torch.equal(saved[n],g) for n,g in grads.items())
     torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();scheduler.step();method.after_update(step,2)
     if reference is not None:
      torch.nn.utils.clip_grad_norm_(reference.model.parameters(),1.);ropt.step();rsched.step()
      for layer in reference.expert_layers:layer.gate_tau=.7+.4*(1+__import__('math').cos(__import__('math').pi*step/2))
      assert all(torch.equal(p,dict(reference.model.named_parameters())[n]) for n,p in model.named_parameters())
     evidence=json.loads((root/case/'optimizer'/f'S{stage:02d}_{step:03d}.json').read_text())
     coords=sum(p.numel() for p in model.parameters() if p.requires_grad)
     assert evidence['parameter_IDs_match'] and evidence['coordinates']==coords and evidence['gradient_bytes']==4*coords and evidence['two_moment_bytes']==8*coords
    opt.zero_grad(set_to_none=True);method.end_stage(stage)
    if reference is not None:
     ropt.zero_grad(set_to_none=True);state_root=root/'author_reference/method_state';state_root.mkdir(parents=True,exist_ok=True)
     reference.learner._cl_state_root=str(state_root);reference.learner._save_cl_state=lambda task:None
     reference.learner.after_train(stage,reference.model)
     for name in ('_I_agg','_A_ref','_B_ref'):
      a=getattr(method.learner,name);b=getattr(reference.learner,name)
      assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)
   assert before!=digest([p for p in model.parameters() if p.requires_grad]) and stable==digest(frozen)
   final=compile_final(method,dtype=torch.float32);final.eval()
   with torch.no_grad():expected=final(**batch).logits
   payload=dict(kind='native_reuse_compiled_v1',case=case,config=final.config.to_dict(),weights=cpu_tree(final.state_dict()),
    state_digest=digest(final.state_dict().values()),gate_tau=[l.gate_tau for l in method.expert_layers])
   path=root/case/'compiled.pt';save_tensor_file(path,payload);reload=restore_compiled(path,dtype=torch.float32)
   with torch.no_grad():assert torch.equal(expected,reload(**batch).logits)
   assert layout(final)==layout(reload) and layout(final)['trainable_coordinates']==0
   outcomes[case]=dict(two_stages_four_updates=True,shared_qv_initial_digest=common,initial_parent_logits_exact=True,
    missing_directory_gradient_reload_exact=True,persistent_optimizer_commits=4,compiled_graph_reload_exact=True,
    callbacks=method.records,original_author_E_math_callbacks_exact=(case=='E'),not_real_carrier_BF16_or_task_quality=True)
 assert not torch.cuda.is_initialized();return dict(passed=True,cases=outcomes,CUDA_uninitialized=True,CPU_dtype='float32')

def admit():
 approved();assert not (REPORT/'CPU_GATE_STARTED.json').exists() and not (REPORT/'READY.json').exists()
 assets=json.loads((REPORT/'ASSET_RECEIPT.json').read_text())
 for n,h in assets['base_files'].items():assert sha(ROOT/n)==h
 assert 'torch' not in sys.modules
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
 write(REPORT/'CPU_GATE_STARTED.json',dict(single_attempt=True,CUDA_hidden_before_Torch=True))
 tick=time.monotonic();gate=cpu_gate()
 from transformers import AutoProcessor,GenerationConfig
 from runtime import policy
 processor=AutoProcessor.from_pretrained(BASE,local_files_only=True)
 native_policy=policy(processor,GenerationConfig.from_pretrained(BASE,local_files_only=True))
 gate['seconds']=time.monotonic()-tick;gate['native_policy']=native_policy;write(REPORT/'CPU_GATE.json',gate)
 path,_,_,_,_,_,plan,_=data();s=study()
 prior_report=REPORT.parent/'native_response_reuse_pilot_v1'
 prior=[prior_report/n for n in ('CPU_GATE_STARTED.json','CPU_GATE_FAILURE.json','COMPATIBILITY_REPAIR_REVIEW.json','SOURCE_REVIEW.json')]
 prior.extend(ROOT/'exp/native_response_reuse_20261024/v1'/n for n in SOURCES)
 deps=[PORT/n for n in ('run.py','runtime_utils.py','methods.py','pass_adapter.py','host_adam.py','training_validity.py','resource_monitor.py','retrieval.py')]
 deps.extend([ROOT/'exp/osft_increment_screen_20261019/v1/study.py',ROOT/'analysis/osft_increment_paired_analysis_20261022.py',
  ROOT/'analysis/onevision_research_device_entry.py',ROOT/'code/PASs_MoE/files_sha256.json',ROOT/'code/PASs_MoE/source_manifest.json',
  ROOT/'code/PASs_MoE/source/CoIN/peft/tuners/coinmoelora.py',ROOT/'code/PASs_MoE/source/cl_learner/moelora.py'])
 deps.extend(prior)
 # Bind actual installed native implementation files, not a guessed package version.
 import inspect,transformers
 from transformers.models.idefics3 import modeling_idefics3,processing_idefics3
 from transformers.models.llama import modeling_llama
 from transformers.models.siglip import modeling_siglip
 from transformers import modeling_utils
 modules=(modeling_idefics3,processing_idefics3,modeling_llama,modeling_siglip,modeling_utils)
 environment={str(Path(inspect.getfile(mod)).resolve()):sha(inspect.getfile(mod)) for mod in modules}
 image_source=str(Path(inspect.getfile(type(processor.image_processor))).resolve());environment[image_source]=sha(image_source)
 evidence=[REPORT/n for n in ('PROTOCOL.md','IMPLEMENTATION_SPEC.md','NECESSITY_REVIEW.json','SOURCE_REVIEW.json','ASSET_RECEIPT.json','CPU_GATE.json')]
 value=dict(attention=dict(ATTENTION),cases=list(CASES),case_budgets=BUDGET,aggregate_budget_seconds=36000,trainable_coordinates=TRAINABLE,
  sources={(Path(__file__).parent/n).relative_to(ROOT).as_posix():sha(Path(__file__).parent/n) for n in SOURCES},
  evidence={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},dependencies={p.relative_to(ROOT).as_posix():sha(p) for p in deps},
  base_files=assets['base_files'],base_weight_coefficients=assets['serialized_weight_coefficients'],manifest_sha256=sha(path),
  split_sha256=sha(s.REPORT/'SPLIT.json'),training_ast_sha256=s.training_ast()[1],training_plan=plan,native_policy=native_policy,
  interpreter=str(Path(sys.executable).resolve()),transformers_version=transformers.__version__,environment=environment,
  no_model_activation=True,not_real_BF16_quality_proof=True)
 write(REPORT/'READY.json',value)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['assets','admit']);args=ap.parse_args()
 if args.action=='assets':acquire()
 else:admit()
