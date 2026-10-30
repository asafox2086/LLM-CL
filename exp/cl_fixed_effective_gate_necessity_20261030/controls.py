"""DRAFT implementation controls only: five frozen TRAIN IDs, original native <=2 tokens.
No CE/F1, training, new coefficient fitting or full dynamic TEST replay.
"""
import copy,hashlib
import torch
from transformers import StoppingCriteria,StoppingCriteriaList
CONTROL_INDICES=(0,16,32,48,64)
class FirstTwo(StoppingCriteria):
 def __init__(self,n):self.n=n
 def __call__(self,input_ids,scores,**kwargs):return torch.full((input_ids.shape[0],),input_ids.shape[1]-self.n>=2,dtype=torch.bool,device=input_ids.device)
def kwargs(model,processor,task):
 kw=dict(do_sample=False,num_beams=1,max_new_tokens=task['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
 cfg=copy.deepcopy(model.core.language_model.generation_config);cfg.update(**kw);assert cfg.get_generation_mode().value=='greedy_search' and cfg.max_time is None
 return kw
def run_controls(*,controller,method,processor,rows,rt,fv):
 model=method.model;assert not model.training and not torch.is_grad_enabled();records=[]
 for index in CONTROL_INDICES:
  task,row=rows[index];safe={k:row[k] for k in ['id','question','images','context'] if k in row};method.set_instruction(task['task'])
  batch,details=rt.input_batch(safe,task,processor,model.device)
  if hasattr(method,'vision_original'):
   fv.prepare(batch,safe,model,method.vision_original);method.prepare_row(safe,batch)
  else:raise RuntimeError('Original CL feature preparation path missing')
  kw=kwargs(model,processor,task);n=batch['input_ids'].shape[1];config_before=model.core.language_model.generation_config.to_dict()
  before=rt.rng();controller.begin('original')
  with torch.autocast('cuda',dtype=model.dtype):a=model.generate(**batch,**kw,stopping_criteria=StoppingCriteriaList([FirstTwo(n)]))
  after=rt.rng();original_calls=controller.step;controller.install();controller.begin('passthrough');rt.restore_rng(before)
  try:
   with torch.autocast('cuda',dtype=model.dtype):b=model.generate(**batch,**kw,stopping_criteria=StoppingCriteriaList([FirstTwo(n)]))
   assert torch.equal(a,b) and controller.step==original_calls and controller.step==a.shape[1]-n and 1<=controller.step<=2
   assert model.core.language_model.generation_config.to_dict()==config_before
   controller.end_control()
  finally:rt.restore_rng(after);controller.uninstall()
  records.append(dict(calibration_index=index,qa_id_sha256=hashlib.sha256(safe['id'].encode()).hexdigest(),task=task['task'],domain=task['domain'],prediction_forwards=original_calls,
   original_and_passthrough_token_ids=a[0,n:].cpu().tolist(),all_168_projection_full_output_byte_digests_equal=True,whole_native_tokens_bitwise_equal=True,new_scores=0))
 return records
