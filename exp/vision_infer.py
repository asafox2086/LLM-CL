"""Local image question answering with the downloaded Qwen2-VL-2B model."""
import argparse
import json
import time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration

ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--image',type=Path);p.add_argument('--question',default='Describe the image briefly.')
 p.add_argument('--device',default='cpu');p.add_argument('--max-new-tokens',type=int,default=32)
 p.add_argument('--demo',action='store_true');p.add_argument('--output',type=Path)
 a=p.parse_args();torch.set_num_threads(8)
 directory=ROOT/'model/Qwen2-VL-2B-Instruct'
 status=json.loads((directory/'download_status.json').read_text())
 if status['status']!='completed' or not status['all_sha256_verified']:raise RuntimeError('Model download has not passed integrity checks')
 processor=AutoProcessor.from_pretrained(directory,local_files_only=True,min_pixels=4*28*28,max_pixels=128*28*28)
 dtype=torch.float32 if a.device=='cpu' else torch.float16
 started=time.monotonic()
 model=Qwen2VLForConditionalGeneration.from_pretrained(directory,local_files_only=True,torch_dtype=dtype,attn_implementation='sdpa',low_cpu_mem_usage=True).to(a.device).eval()
 if a.demo:
  examples=[]
  for dataset in ['natural/vqav2','medical/vqa_rad']:
   row=json.loads((ROOT/'CV_data'/dataset/'test.jsonl').read_text().splitlines()[0])
   examples.append({'dataset':dataset,'id':row['id'],'image':str(ROOT/'CV_data'/row['image']),'question':row['question'],'references':row['references']})
 else:
  if not a.image:p.error('--image is required without --demo')
  examples=[{'image':str(a.image.resolve()),'question':a.question}]
 records=[]
 for row in examples:
  image=Image.open(row['image']).convert('RGB')
  messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']}]}]
  prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  inputs=processor(text=[prompt],images=[image],return_tensors='pt').to(a.device)
  begin=time.monotonic()
  with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False)
  answer=processor.batch_decode(tokens[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
  record={**row,'prediction':answer,'seconds':time.monotonic()-begin,'image_grid_thw':inputs.image_grid_thw.cpu().tolist(),
          'image_tokens':int((inputs.input_ids==model.config.image_token_id).sum()),'generated_token_ids':tokens[0,inputs.input_ids.shape[1]:].cpu().tolist()}
  records.append(record);print(json.dumps(record,ensure_ascii=False),flush=True)
 result={'model':'Qwen/Qwen2-VL-2B-Instruct','device':a.device,'dtype':str(dtype),'max_pixels':128*28*28,
         'max_new_tokens':a.max_new_tokens,'wall_seconds':time.monotonic()-started,'scope':'Two-image loading/inference smoke test, not an accuracy benchmark.','examples':records}
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
