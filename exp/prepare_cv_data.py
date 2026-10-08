"""Prepare a compact VQAv2 pilot and original VQA-RAD with image-disjoint splits."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import zipfile
import requests
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'CV_data'

def digest(value):return hashlib.sha256(value).hexdigest()
def ordered(value):return digest(('cv_pilot_seed42\n'+str(value)).encode())
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');p.replace(path)
def jsonl(path,rows):
 path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
def zipped(path):
 with zipfile.ZipFile(path) as z:
  if z.testzip() is not None:raise ValueError('Corrupt archive '+str(path))
  return json.loads(z.read(z.namelist()[0]))
def verify_image(path):
 with Image.open(path) as image:image.verify()
 with Image.open(path) as image:return image.size

def prepare_medical():
 dest=BASE/'medical/vqa_rad';images=dest/'images';images.mkdir(exist_ok=True)
 with zipfile.ZipFile(dest/'raw/images.zip') as archive:
  if archive.testzip() is not None:raise ValueError('Corrupt VQA-RAD archive')
  for name in archive.namelist():
   if name.endswith('/'):continue
   path=images/Path(name).name
   path.write_bytes(archive.read(name));verify_image(path)
 raw=json.loads((dest/'raw/VQA_RAD_Dataset_Public.json').read_text())
 # Union connected image/case groups: neither an image nor its known case crosses splits.
 parents={}
 def find(x):
  parents.setdefault(x,x)
  if parents[x]!=x:parents[x]=find(parents[x])
  return parents[x]
 def union(a,b):parents[find(a)]=find(b)
 for r in raw:
  union('image:'+r['image_name'],'case:'+r['image_case_url'] if r.get('image_case_url') else 'image:'+r['image_name'])
 groups={find('image:'+r['image_name']) for r in raw};groups=sorted(groups,key=ordered)
 n=len(groups);mapping={g:'train' if i<int(n*.7) else 'dev' if i<int(n*.8) else 'test' for i,g in enumerate(groups)}
 splits={s:[] for s in ['train','dev','test']}
 for i,r in enumerate(raw):
  split=mapping[find('image:'+r['image_name'])];image_path=images/r['image_name']
  if not image_path.is_file():raise ValueError('Missing VQA-RAD image')
  row={'id':f'vqarad_{i:04d}','source_qid':r['qid'],'domain':'medical','dataset':'VQA-RAD',
       'image':str(image_path.relative_to(BASE)),'image_id':r['image_name'],'case_url':r.get('image_case_url'),
       'question':r['question'],'answer':str(r['answer']),'references':[str(r['answer'])],
       'answer_type':r['answer_type'],'question_type':r['question_type'],'organ':r['image_organ'],'split':split}
  splits[split].append(row)
 for split,rows in splits.items():jsonl(dest/f'{split}.jsonl',rows)
 save(dest/'manifest.json',{'source':'https://osf.io/89kps/','original_qa_count':len(raw),'image_count':len(list(images.iterdir())),
  'group_count':n,'split_policy':'Custom seed42 70/10/20 split by connected image/case groups; NOT published standard question-level split.',
  'splits':{s:{'qa':len(r),'images':len({x['image'] for x in r}),'sha256':digest((dest/f'{s}.jsonl').read_bytes())} for s,r in splits.items()},
  'raw_sha256':{p.name:digest(p.read_bytes()) for p in (dest/'raw').iterdir() if p.is_file()}})
 print('medical prepared', {s:len(r) for s,r in splits.items()},flush=True)
 return splits

def prepare_natural_annotations():
 dest=BASE/'natural/vqav2';splits={s:[] for s in ['train','dev','test']}
 for source,limits in [('train',[('train',1000),('dev',200)]),('val',[('test',500)])]:
  questions=zipped(dest/f'raw/{source}_questions.zip')
  annotations=zipped(dest/f'raw/{source}_annotations.zip')
  qmap={q['question_id']:q for q in questions['questions']}
  # One deterministically selected question per image for the pilot.
  unique={}
  for a in annotations['annotations']:
   img=a['image_id']
   if img not in unique or a['question_id']<unique[img]['question_id']:unique[img]=a
  candidates=sorted(unique.values(),key=lambda a:ordered(a['image_id']))
  offset=0
  for split,count in limits:
   for a in candidates[offset:offset+count]:
    image_name=f'COCO_{source}2014_{a["image_id"]:012d}.jpg'
    image_path=dest/'images'/image_name
    splits[split].append({'id':f'vqav2_{a["question_id"]}','domain':'natural','dataset':'VQAv2',
      'image':str(image_path.relative_to(BASE)),'image_id':a['image_id'],'question_id':a['question_id'],
      'question':qmap[a['question_id']]['question'],'answer':a['multiple_choice_answer'],
      'references':[r['answer'] for r in a['answers']],'answer_annotations':a['answers'],
      'answer_type':a['answer_type'],'question_type':a['question_type'],'split':split,
      'source_split':source+'2014','image_url':f'https://s3.amazonaws.com/images.cocodataset.org/{source}2014/{image_name}'})
   offset+=count
  save(dest/f'raw/{source}_license.json',{'questions':questions.get('license'),'annotations':annotations.get('license'),'info':annotations.get('info')})
 for split,rows in splits.items():jsonl(dest/f'{split}.jsonl',rows)
 return splits

def download_image(row):
 path=BASE/row['image'];path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():
  try:verify_image(path);return
  except Exception:pass
 partial=path.with_suffix('.jpg.part')
 for attempt in range(5):
  session=requests.Session();session.trust_env=False
  try:
   with session.get(row['image_url'],stream=True,timeout=(10,45)) as response:
    response.raise_for_status()
    with partial.open('wb') as handle:
     for block in response.iter_content(128*1024):handle.write(block)
   verify_image(partial);partial.replace(path);return
  except (requests.RequestException,OSError) as e:
   if attempt==4:raise RuntimeError(f'{row["id"]}: {e}')
   time.sleep(attempt+1)
  finally:session.close()

def validate(splits):
 image_sets={s:{r['image'] for r in rows} for s,rows in splits.items()}
 for a,b in [('train','dev'),('train','test'),('dev','test')]:
  assert not image_sets[a]&image_sets[b],(a,b)
 images={}
 for rows in splits.values():
  for r in rows:
   p=BASE/r['image']
   if r['image'] not in images:
    size=verify_image(p);images[r['image']]={'width':size[0],'height':size[1],'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())}
   assert r['question'].strip() and r['references']
 return images

if __name__=='__main__':
 signal.signal(signal.SIGHUP,signal.SIG_IGN)
 status=BASE/'preparation_status.json'
 try:
  save(status,{'status':'running','phase':'medical','pid':os.getpid()})
  medical=prepare_medical();med_images=validate(medical)
  save(BASE/'medical/vqa_rad/image_manifest.json',med_images)
  save(status,{'status':'running','phase':'natural_annotations','pid':os.getpid()})
  # A separate resumable curl may still be finishing this archive.
  deadline=time.monotonic()+900
  while not zipfile.is_zipfile(BASE/'natural/vqav2/raw/train_annotations.zip'):
   if time.monotonic()>deadline:raise RuntimeError('Train annotation download not complete')
   time.sleep(5)
  natural=prepare_natural_annotations();rows=[r for v in natural.values() for r in v]
  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
   for done,_ in enumerate(pool.map(download_image,rows),1):
    if done%50==0:
     save(status,{'status':'running','phase':'natural_images','pid':os.getpid(),'downloaded':done,'total':len(rows)})
     print('natural images',done,'/',len(rows),flush=True)
  nat_images=validate(natural);save(BASE/'natural/vqav2/image_manifest.json',nat_images)
  dest=BASE/'natural/vqav2'
  save(dest/'manifest.json',{'source':'https://visualqa.org/download.html','image_source':'https://cocodataset.org/',
    'scope':'1700-question/image pilot subset; complete original train/val annotations retained under raw.',
    'split_policy':'Seed42 image-disjoint subset; train/dev from official train2014; test from official val2014. NOT official VQAv2 test.',
    'splits':{s:{'qa':len(r),'images':len({x['image'] for x in r}),'sha256':digest((dest/f'{s}.jsonl').read_bytes())} for s,r in natural.items()},
    'raw_sha256':{p.name:digest(p.read_bytes()) for p in (dest/'raw').iterdir() if p.is_file()}})
  save(status,{'status':'completed','natural_images':len(nat_images),'natural_qa':len(rows),'medical_images':len(med_images),
               'medical_qa':sum(len(r) for r in medical.values()),'all_images_decoded':True,'image_disjoint_splits':True})
 except BaseException as e:
  save(status,{'status':'failed','error':str(e)});raise
