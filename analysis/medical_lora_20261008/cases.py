"""Publish a fixed, explicitly failure-selected diagnostic case set and image metadata."""
import csv,hashlib,json,shutil,sys
from pathlib import Path
from collections import Counter
from statistics import mean
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
from common import Scorer,normalize_answer
RUN=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run']);DATA=torch.load(ROOT/'exp/CV_result/shared/data.pt',map_location='cpu',weights_only=False)
METHODS=['seq_lora','migu_lora','olora','sapt_lora'];scorer=Scorer()
read=lambda method,stage,task:{r['instance_id']:r for r in [json.loads(l) for l in (RUN/method/'predictions'/f'stage_{stage:02d}'/(task+'.jsonl')).read_text().splitlines()]}
cases=[]
for task,stage,chosen in [('medical_mts_dialog_note',6,['mts_test_122','mts_test_67','mts_test_152']),('medical_iu_xray_impression',7,['CXR1233','CXR2139'])]:
 base=read('seq_lora',0,task)
 for ident in chosen:
  cases.append({'task':task,'instance_id':ident,'prompt':base[ident]['prompt'],'references':base[ident]['references'],
   'selection':'deliberately selected compression/omission examples; not representative averages',
   'methods':{m:{str(s):read(m,s,task)[ident] for s in [0,stage,9]} for m in METHODS}})
vision='medical_vqa_rad';final={m:read(m,9,vision) for m in METHODS};base=read('seq_lora',0,vision);(OUT/'assets').mkdir(exist_ok=True)
for organ in ['CHEST','HEAD','ABD']:
 for typ in ['CLOSED','OPEN']:
  pool=[r for r in DATA[vision]['test'] if r['organ']==organ and r['answer_type']==typ]
  pool.sort(key=lambda r:(sum(final[m][r['instance_id']]['scores']['exact_match']==100 for m in METHODS),r['instance_id']))
  row=pool[0];ident=row['instance_id'];src=ROOT/'CV_data'/row['image'];dst=OUT/'assets'/src.name;shutil.copy2(src,dst)
  cases.append({'task':vision,'instance_id':ident,'prompt':row['prompt'],'references':row['references'],'organ':organ,'answer_type':typ,
   'image':str(dst.relative_to(OUT)),'image_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'grid':row['image_grid_thw'],
   'selection':'within each organ x answer type, fewest final exact matches across methods, then ID; illustrative failures, not random',
   'methods':{m:{str(s):read(m,s,vision)[ident] for s in [0,8,9]} for m in METHODS}})
(OUT/'cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False)+'\n')
metadata=[];seen=set()
for task in ['natural_vqav2','medical_vqa_rad']:
 for split in ['train','test']:
  for r in DATA[task][split]:
   key=(task,split,r['image'])
   if key in seen:continue
   seen.add(key)
   with Image.open(ROOT/'CV_data'/r['image']) as im:
    w,h=im.size;mode=im.mode
   grid=r['image_grid_thw'];loww=grid[2]*14;lowh=grid[1]*14
   metadata.append({'task':task,'split':split,'image':r['image'],'width':w,'height':h,'mode':mode,'processed_width':loww,'processed_height':lowh,
    'merged_visual_tokens':r['image_tokens'],'retained_pixel_area_ratio':loww*lowh/(w*h)})
with (OUT/'image_resolution.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(metadata[0]),lineterminator='\n');w.writeheader();w.writerows(metadata)
priors=[]
for task in ['medical_mts_dialog_note','medical_iu_xray_impression','medical_vqa_rad','natural_vqav2']:
 train=DATA[task]['train'];counts=Counter(normalize_answer(r.get('training_answer',r['references'][0])) for r in train)
 answer=counts.most_common(1)[0][0];test=DATA[task]['test'];ss=[scorer.score(answer,r['references']) for r in test]
 priors.append({'task':task,'train_most_frequent_answer':answer,'train_frequency':counts[answer]/len(train),'test_ROUGE_L':mean(r['rougeL'] for r in ss),
  'test_EM':mean(r['exact_match'] for r in ss),'test_token_F1':mean(r['token_f1'] for r in ss),'n':len(test)})
with (OUT/'constant_answer_baselines.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(priors[0]),lineterminator='\n');w.writeheader();w.writerows(priors)
print('Published 11 illustrative cases and original-image resolution audit.')
