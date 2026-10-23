"""Native Idefics3 input/label semantics. No samples or models load on import."""
import hashlib
import json
import math
import os
import time
from pathlib import Path
from core import ROOT,PORT,write

def policy(processor,generation_config):
 stops=generation_config.eos_token_id
 stops=[stops] if isinstance(stops,int) else list(stops)
 assert stops and all(isinstance(i,int) for i in stops)
 messages=[{'role':'user','content':[{'type':'text','text':'LLMCL synthetic interface probe.'}]}]
 prefix=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 marker='LLMCL_SYNTHETIC_ASSISTANT_BOUNDARY'
 full=processor.apply_chat_template(messages+[{'role':'assistant','content':[{'type':'text','text':marker}]}],
  tokenize=False,add_generation_prompt=False)
 assert full.startswith(prefix) and full.count(marker)==1
 suffix=full.split(marker,1)[1]
 matches=[(i,processor.tokenizer.convert_ids_to_tokens(i)) for i in stops
  if processor.tokenizer.convert_ids_to_tokens(i) in suffix]
 assert len(matches)==1, 'Native assistant terminator is not uniquely resolved'
 stop_id,stop_text=matches[0]
 assert processor.tokenizer(stop_text,add_special_tokens=False)['input_ids']==[stop_id]
 return dict(generation_stop_ids=stops,assistant_stop_id=stop_id,assistant_stop_text=stop_text,
  tokenizer_eos_id=processor.tokenizer.eos_token_id,pad_id=processor.tokenizer.pad_token_id,
  native_template_SHA256=hashlib.sha256(json.dumps(dict(processor=getattr(processor,'chat_template',None),
   tokenizer=getattr(processor.tokenizer,'chat_template',None)),ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
  different_text_boundary_and_generation_stop_allowed=True,not_runtime_model_proof=True)

def tensor_hash(batch):
 import torch
 h=hashlib.sha256()
 for n,t in sorted(batch.items()):
  if not isinstance(t,torch.Tensor):continue
  v=t.detach().cpu().contiguous()
  h.update(str((n,tuple(v.shape),str(v.dtype))).encode())
  a=v.reshape(-1).view(torch.uint8).numpy().reshape(-1)
  for start in range(0,a.size,8*1024**2):h.update(a[start:start+8*1024**2].tobytes())
 return h.hexdigest()

class NativeRuntime:
 def __init__(self,processor,model,root,policy_value,context_limit=8192,input_dtype=None,image_SHA256=None):
  import torch
  self.processor,self.model,self.root,self.policy=processor,model,Path(root),policy_value
  self.context_limit=context_limit;self.input_records=[];self.requests=[];self.native_gates=[]
  self.input_dtype=torch.bfloat16 if input_dtype is None else input_dtype
  self.capture_logits=False;self.last_logits_SHA256=None
  self.phase='UNASSIGNED'
  self.image_SHA256=image_SHA256
 def input_batch(self,row,task,processor,device):
  import torch
  from PIL import Image
  from retrieval import retrieve
  context=row.get('context','');retrieval=[]
  if len(context)>6000:
   chunks=[context[s:s+1000] for s in range(0,len(context),800)]
   retrieval=sorted(retrieve(row['question'],chunks,5));context='\n'.join(chunks[i] for i in retrieval)
  messages=[];images=[];shapes=[];hashes=[]
  for name in row.get('images',[]):
   p=ROOT/name;image_sha=__import__('core').sha(p)
   assert self.image_SHA256 is not None and self.image_SHA256[name]==image_sha,'Original image SHA changed'
   hashes.append(dict(path=name,sha256=image_sha))
   with Image.open(p) as image:
    im=image.convert('RGB');before=im.size
    if im.width*im.height>401408:
     ratio=math.sqrt(401408/(im.width*im.height))
     im=im.resize((max(1,int(im.width*ratio)),max(1,int(im.height*ratio))),Image.Resampling.BICUBIC)
    assert im.width*im.height<=401408
    images.append(im);shapes.append(dict(original=list(before),common_input=list(im.size)))
   messages.append({'type':'image'})
  question=row['question']
  if '<image>' in question:
   assert question.count('<image>')==len(images);question=question.replace('<image>','')
  text=task['instruction']+('\nContext:\n'+context if context else '')+'\nQuestion: '+question
  messages=[{'role':'user','content':messages+[{'type':'text','text':text}]}]
  prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  batch=dict(processor(text=[prompt],images=images or None,return_tensors='pt',truncation=False))
  for k,t in batch.items():
   assert isinstance(t,torch.Tensor),k
   if t.is_floating_point():batch[k]=t.to(dtype=self.input_dtype)
  assert batch['input_ids'].shape[0]==1
  assert batch['input_ids'].shape[1]<self.context_limit,'Input would exceed native context'
  prepared=tensor_hash(batch)
  details=dict(prompt_tokens=int(batch['input_ids'].shape[1]),prepared_input_SHA256=prepared,
   image_SHA256=hashes,image_shapes=shapes,retrieved_chunk_indexes=retrieval,common_image_area_cap=401408,
   visual_tokens=int((batch['input_ids']==self.model.config.image_token_id).sum()))
  assert self.phase in ('TRAIN','NATIVE_GATE','DEV')
  self.input_records.append(dict(id=row['id'],task=task['task'],phase=self.phase,**details))
  batch={k:t.to(device) for k,t in batch.items()}
  batch['_messages']=messages;batch['_prompt']=prompt
  return batch,details
 def target(self,batch,answer):
  processor=self.processor;stop=self.policy['assistant_stop_text']
  assert stop not in answer,'Answer contains a native control token'
  full=processor.apply_chat_template(batch['_messages']+[{'role':'assistant','content':[{'type':'text','text':answer}]}],
   tokenize=False,add_generation_prompt=False)
  prompt=batch['_prompt'];assert full.startswith(prompt)
  suffix=full[len(prompt):];assert suffix.count(stop)==1
  suffix=suffix[:suffix.index(stop)+len(stop)]
  a=processor.tokenizer(prompt,add_special_tokens=False,truncation=False)['input_ids']
  b=processor.tokenizer(prompt+suffix,add_special_tokens=False,truncation=False)['input_ids']
  assert b[:len(a)]==a,'Native prompt/assistant token boundary is not a prefix'
  target=b[len(a):]
  assert target and target[-1]==self.policy['assistant_stop_id']
  assert len(target)-1<=512,'Target would exceed original target-token policy; no truncation'
  assert batch['input_ids'].shape[1]+len(target)<=self.context_limit,'Full answer would exceed native context'
  return target
 def answer_loss(self,model,batch,answer,processor):
  import torch
  target=self.target(batch,answer)
  labels=torch.tensor([target],device=batch['input_ids'].device,dtype=torch.long)
  length=batch['input_ids'].shape[1]
  ids=torch.cat([batch['input_ids'],labels],dim=1)
  payload={k:v for k,v in batch.items() if not k.startswith('_') and k not in ('input_ids','attention_mask')}
  out=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True,**payload)
  hidden=out.last_hidden_state[:,length-1:-1,:]
  logits=model.lm_head(hidden).float()
  if self.capture_logits:self.last_logits_SHA256=tensor_hash({'logits':logits})
  loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
  self.requests.append(dict(kind='answer_NLL',target_tokens=len(target)))
  return loss,len(target)
 def generate(self,batch,cap):
  import torch
  assert batch['input_ids'].shape[1]+cap<=self.context_limit,'Generation cap would exceed native context'
  kwargs={k:v for k,v in batch.items() if not k.startswith('_')}
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   out=self.model.generate(**kwargs,do_sample=False,num_beams=1,max_new_tokens=cap,use_cache=True,
    eos_token_id=self.policy['generation_stop_ids'],pad_token_id=self.policy['pad_id'])
  response=out[0,batch['input_ids'].shape[1]:]
  self.requests.append(dict(kind='generation',output_tokens=len(response),cap=cap))
  return self.processor.tokenizer.decode(response,skip_special_tokens=True),len(response)
 def gate(self,row,task,stage):
  import torch
  self.phase='NATIVE_GATE'
  self.model.eval();batch,details=self.input_batch(row,task,self.processor,self.model.device)
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):loss,length=self.answer_loss(self.model,batch,row['answer'],self.processor)
  assert torch.isfinite(loss)
  _,n=self.generate(batch,4)
  self.native_gates.append(dict(stage=stage,id=row['id'],loss=float(loss),target_tokens=length,
   short_generation_tokens=n,prepared_input_SHA256=details['prepared_input_SHA256']))
 def evaluate(self,manifest,dev):
  self.phase='DEV'
  self.model.eval()
  for task in manifest['tasks']:
   k=task['task']
   if not dev[k]:continue
   path=self.root/'evaluations/stage_19'/f'T{k:02d}_dev_predictions.jsonl'
   assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
   temp=path.with_suffix('.jsonl.partial')
   with temp.open('w',encoding='utf-8') as f:
    for row in dev[k]:
     batch,details=self.input_batch(row,task,self.processor,self.model.device)
     tick=time.monotonic();prediction,n=self.generate(batch,task['max_new_tokens'])
     # Only the single case verifier calculates scores, after COMPLETE.
     record=dict(id=row['id'],prediction=prediction,references=row['answers'],generation_cap=task['max_new_tokens'],
      output_tokens=n,generation_seconds=time.monotonic()-tick,**details)
     f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
    os.fsync(f.fileno())
   temp.replace(path)
