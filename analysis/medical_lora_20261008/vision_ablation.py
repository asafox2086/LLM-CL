"""Paired image-presence and resolution probes; never modify checkpoints."""
import csv,hashlib,io,json,sys,time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
from cv_data import MODEL
from common import Scorer
from run_olora import create_method
RUN=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
data=torch.load(ROOT/'exp/CV_result/shared/data.pt',map_location='cpu',weights_only=False)['medical_vqa_rad']['test']
# 60 fixed questions, 10 per organ x answer-type stratum; all six cells have >=10.
rows=[]
for organ in ['CHEST','HEAD','ABD']:
 for typ in ['CLOSED','OPEN']:
  group=[r for r in data if r['organ']==organ and r['answer_type']==typ]
  group.sort(key=lambda r:hashlib.sha256(('medical-image42\n'+r['instance_id']).encode()).hexdigest())
  assert len(group)>=10;rows.extend(group[:10])
# Assign a deterministic image from a different held-out image and same organ.
wrong={}
for row in rows:
 candidates=sorted({r['image'] for r in data if r['organ']==row['organ'] and r['image']!=row['image']})
 index=int(hashlib.sha256(row['instance_id'].encode()).hexdigest(),16)%len(candidates)
 wrong[row['instance_id']]=candidates[index]
config=json.loads((RUN/'seq_lora/config.json').read_text());torch.set_num_threads(4)
model=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
method=create_method(model,config);scorer=Scorer();summary=[]
protocol={'selection':'fixed SHA256 within organ x OPEN/CLOSED; 10 each, 60 total','instance_ids':[r['instance_id'] for r in rows],
 'image_permutation':wrong,'stages':[0,9],'method':'seq_lora','modes':['actual128','wrong128','no_image','actual512'],
 'decode':'greedy max_new_tokens256 batch4 at128, batch1 at512, no prompt truncation','optimizer_updates':0,
 'scope':'60-question stratified paired diagnostic, not full benchmark; high-resolution inference at unchanged weights is not high-resolution retraining',
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'vision_ablation_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
for stage in [0,9]:
 if stage:
  state=torch.load(RUN/'seq_lora/checkpoints'/f'stage_{stage:02d}'/'completed.pt',map_location='cpu',weights_only=False);method.load(io.BytesIO(state['method']))
 model.eval()
 for mode in protocol['modes']:
  path=OUT/f'vision_stage{stage}_{mode}.jsonl'
  if path.exists():
   saved=[json.loads(line) for line in path.read_text().splitlines()];assert [r['instance_id'] for r in saved]==protocol['instance_ids']
  else:
   proc=AutoProcessor.from_pretrained(MODEL,local_files_only=True,min_pixels=4*28*28,max_pixels=(512 if mode=='actual512' else 128)*28*28)
   saved=[]
   with path.with_suffix('.tmp').open('w') as handle:
    batch_size=1 if mode=='actual512' else 4
    for start in range(0,len(rows),batch_size):
     selected=rows[start:start+batch_size];texts=[];images=[]
     for r in selected:
      content=([{'type':'image'}] if mode!='no_image' else [])+[{'type':'text','text':r['prompt']}]
      texts.append(proc.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True))
      if mode!='no_image':
       image=wrong[r['instance_id']] if mode=='wrong128' else r['image']
       with Image.open(ROOT/'CV_data'/image) as im:images.append(im.convert('RGB').copy())
     packed=proc(text=texts,images=images or None,padding=True,return_tensors='pt').to(model.device)
     assert int(packed['attention_mask'].sum(1).max())<=768
     with torch.inference_mode():
      generated=model.generate(**packed,do_sample=False,num_beams=1,max_new_tokens=256,pad_token_id=proc.tokenizer.pad_token_id,eos_token_id=proc.tokenizer.eos_token_id)
     ids=generated[:,packed['input_ids'].shape[1]:].cpu().tolist();answers=proc.batch_decode(ids,skip_special_tokens=True)
     for r,tokens,pred in zip(selected,ids,answers):
      if proc.tokenizer.eos_token_id in tokens:tokens=tokens[:tokens.index(proc.tokenizer.eos_token_id)+1]
      record={'stage':stage,'mode':mode,'instance_id':r['instance_id'],'image':r['image'],'used_image':wrong[r['instance_id']] if mode=='wrong128' else (r['image'] if mode!='no_image' else None),
       'organ':r['organ'],'answer_type':r['answer_type'],'prompt':r['prompt'],'references':r['references'],'prediction':pred,'scores':scorer.score(pred,r['references']),
       'generated_token_ids':tokens,'hit_limit':len(tokens)==256 and tokens[-1]!=proc.tokenizer.eos_token_id}
      saved.append(record);handle.write(json.dumps(record,ensure_ascii=False)+'\n')
     handle.flush()
     print(json.dumps({'stage':stage,'mode':mode,'done':len(saved)}),flush=True)
   path.with_suffix('.tmp').replace(path)
  original={r['instance_id']:r for r in [json.loads(line) for line in (RUN/'seq_lora/predictions'/f'stage_{stage:02d}'/'medical_vqa_rad.jsonl').read_text().splitlines()]}
  for group in ['all','CLOSED','OPEN','CHEST','HEAD','ABD']:
   chosen=[r for r in saved if group=='all' or r['answer_type']==group or r['organ']==group]
   summary.append({'stage':stage,'mode':mode,'group':group,'n':len(chosen),
    'rougeL':sum(r['scores']['rougeL'] for r in chosen)/len(chosen),'EM':sum(r['scores']['exact_match'] for r in chosen)/len(chosen),
    'agreement_with_original_prediction':sum(r['prediction']==original[r['instance_id']]['prediction'] for r in chosen)/len(chosen),
    'generation_limit_hits':sum(r['hit_limit'] for r in chosen)})
with (OUT/'vision_ablation_scores.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]),lineterminator='\n');w.writeheader();w.writerows(summary)
print('Visual probe complete: 480 answers.',flush=True)
