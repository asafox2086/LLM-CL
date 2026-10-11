import argparse,concurrent.futures,hashlib,json,os,re,struct,threading,time
from pathlib import Path
from urllib.parse import quote
import requests

ROOT=Path(__file__).resolve().parent.parent
WORK=ROOT/'data'/'.multimodal_asset_cache'; WORK.mkdir(exist_ok=True)
LOCK=threading.Lock()
MODELS=[
 {'repo':'Qwen/Qwen3.5-0.8B-Base','folder':'Qwen3.5-0.8B-Base','stage':'pretrained','nominal_b':0.8},
 {'repo':'Qwen/Qwen3.5-2B-Base','folder':'Qwen3.5-2B-Base','stage':'pretrained','nominal_b':2},
 {'repo':'HuggingFaceTB/SmolVLM-500M-Base','folder':'SmolVLM-500M-Base','stage':'pretrained','nominal_b':0.5},
 {'repo':'HuggingFaceTB/SmolVLM-Base','folder':'SmolVLM-Base','stage':'pretrained','nominal_b':2.25},
 {'repo':'Qwen/Qwen3-VL-2B-Instruct','folder':'Qwen3-VL-2B-Instruct','stage':'instruction_tuned_backbone','nominal_b':2},
]

def digest(path,git=False):
 h=hashlib.sha1() if git else hashlib.sha256()
 if git:h.update(f'blob {path.stat().st_size}\0'.encode())
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()

def metadata(kind,repo,revision=None):
 path=WORK/(repo.replace('/','__')+'.json')
 if path.exists():
  cached=json.loads(path.read_text())
  if revision is None or cached['sha']==revision:return cached
 api=f'https://huggingface.co/api/{kind}/{repo}'+(f'/revision/{revision}' if revision else '')
 r=requests.get(api+'?blobs=true',timeout=60);r.raise_for_status();m=r.json()
 path.write_text(json.dumps(m,ensure_ascii=False));return m

def selected(kind,spec,files):
 if kind=='models':return [f for f in files if '/' not in f['rfilename'] and (f['rfilename'].endswith(('.safetensors','.json','.txt','.model','.md','.py')) or f['rfilename'] in ['LICENSE','LICENSE.md']) and f['rfilename']!='.gitattributes']
 patterns=spec.get('include',['*']);import fnmatch
 return [f for f in files if any(fnmatch.fnmatch(f['rfilename'],p) for p in patterns) and not f['rfilename'].endswith('.py') and f['rfilename']!='.gitattributes']

def one_file(kind,spec,meta,f,base):
 name=f['rfilename'];path=base/name;path.parent.mkdir(parents=True,exist_ok=True)
 assert base.resolve() in path.resolve().parents
 size=f['size'];expected=f.get('lfs',{}).get('sha256');gitsha=f.get('blobId') if not expected else None
 if not path.is_file() or path.stat().st_size!=size:
  partial=path.with_name(path.name+'.partial')
  for attempt in range(8):
   try:
    offset=partial.stat().st_size if partial.exists() else 0
    if offset==size:partial.replace(path);break
    if offset>size:partial.unlink();offset=0
    prefix='datasets/' if kind=='datasets' else ''
    endpoint='https://huggingface.co' if attempt%2==0 else 'https://hf-mirror.com'
    url=f'{endpoint}/{prefix}{spec["repo"]}/resolve/{meta["sha"]}/{quote(name)}?download=true&asset_time={int(time.time())}'
    headers={'Range':f'bytes={offset}-'} if offset else {}
    with requests.get(url,headers=headers,stream=True,timeout=(30,120)) as response:
     response.raise_for_status()
     if offset and response.status_code!=206:offset=0
     if offset:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
     with partial.open('ab' if offset else 'wb') as stream:
      for block in response.iter_content(4*1024**2):
       if block:stream.write(block)
    assert partial.stat().st_size==size
    partial.replace(path);break
   except Exception as e:
    print('RETRY',spec['repo'],name,attempt,type(e).__name__,flush=True)
    if attempt==7:raise
    time.sleep(min(2**attempt,20))
 sha=digest(path)
 if expected:assert sha==expected,(spec['repo'],name,'LFS SHA256 mismatch')
 if gitsha:assert digest(path,True)==gitsha,(spec['repo'],name,'Git blob mismatch')
 print('VERIFIED',spec['repo'],name,size,flush=True)
 return dict(file=name,bytes=size,sha256=sha,upstream_sha256=expected,git_blob_id=gitsha)

def model_headers(base,files):
 params=0;dtypes={}
 for f in files:
  if not f['file'].endswith('.safetensors'):continue
  with (base/f['file']).open('rb') as stream:
   n=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(n))
  for name,info in header.items():
   if name=='__metadata__':continue
   count=1
   for dim in info['shape']:count*=dim
   params+=count;dtypes[info['dtype']]=dtypes.get(info['dtype'],0)+count
 return dict(stored_tensor_parameters=params,stored_tensor_b=params/1e9,tensor_dtypes=dtypes)

def run(kind,spec):
 meta=metadata(kind,spec['repo'],spec.get('revision'));assert not meta.get('gated'),spec['repo']
 base=ROOT/('model' if kind=='models' else 'data')/spec['folder']
 if kind=='datasets':base=base/'raw'
 base.mkdir(parents=True,exist_ok=True)
 files=selected(kind,spec,meta['siblings']);assert files
 existing_manifest=base.parent/'asset_manifest.json' if kind=='datasets' else base/'asset_manifest.json'
 if existing_manifest.exists():
  previous=json.loads(existing_manifest.read_text())
  if previous.get('revision')==meta['sha'] and {f['file'] for f in previous['files']}=={f['rfilename'] for f in files} and all((base/f['file']).is_file() and (base/f['file']).stat().st_size==f['bytes'] for f in previous['files']):
   print('REUSE_ALREADY_VERIFIED',spec['repo'],flush=True);return previous
 print('BEGIN',spec['repo'],len(files),sum(f['size'] for f in files),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  verified=list(pool.map(lambda f:one_file(kind,spec,meta,f,base),files))
 info=dict(spec,kind=kind,revision=meta['sha'],source_url=f'https://huggingface.co/'+('datasets/' if kind=='datasets' else '')+spec['repo'],verified_on='2026-10-11',files=verified,download_status='complete',license=meta.get('cardData',{}).get('license'),bytes=sum(f['bytes'] for f in verified))
 if kind=='models':info.update(model_headers(base,verified));assert 0.45<=info['stored_tensor_b']<=3.0,info['stored_tensor_b']
 (base.parent/'asset_manifest.json' if kind=='datasets' else base/'asset_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
 print('COMPLETE',spec['repo'],info['bytes'],flush=True)
 return info

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--models',action='store_true');p.add_argument('--datasets',action='store_true');args=p.parse_args()
 if args.models:
  pinned=json.loads((ROOT/'model/multimodal_manifest.json').read_text())['models']
  for spec in MODELS:
   spec['revision']=next(m['revision'] for m in pinned if m['repo']==spec['repo'])
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: infos=list(pool.map(lambda s:run('models',s),MODELS))
  (WORK/'downloaded_models.json').write_text(json.dumps(infos,ensure_ascii=False,indent=2))
 if args.datasets:
  specs=json.loads((ROOT/'data/multimodal_dataset_specs.json').read_text())
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:infos=list(pool.map(lambda s:run('datasets',s),specs))
  (WORK/'downloaded_datasets.json').write_text(json.dumps(infos,ensure_ascii=False,indent=2))
