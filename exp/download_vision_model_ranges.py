"""Verified, resumable parallel HTTP ranges for the official Qwen2-VL snapshot."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import requests
from download_vision_model import DEST,sha,save,fetch

CHUNK=32*1024*1024

def download_range(job):
 meta,index,start,end=job
 folder=DEST/'.chunks'/meta['Path'];folder.mkdir(parents=True,exist_ok=True)
 path=folder/f'{index:05d}.part';size=end-start+1
 if path.exists() and path.stat().st_size==size:return size
 for attempt in range(6):
  offset=path.stat().st_size if path.exists() else 0
  if offset>size:raise ValueError('Oversized chunk')
  session=requests.Session();session.trust_env=False
  url=f"https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct/resolve/{meta['Revision']}/{meta['Path']}"
  try:
   with session.get(url,headers={'Range':f'bytes={start+offset}-{end}'},stream=True,timeout=(15,60)) as response:
    response.raise_for_status()
    if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start+offset}-{end}/{meta["Size"]}':
     raise ValueError('Server returned unexpected range')
    with path.open('ab') as handle:
     for block in response.iter_content(1024*1024):handle.write(block)
     handle.flush();os.fsync(handle.fileno())
   if path.stat().st_size==size:return size
  except (requests.RequestException,OSError) as e:
   print('retry',meta['Path'],index,str(e)[:100],flush=True)
  finally:session.close()
  time.sleep(1+attempt)
 raise RuntimeError(f'Incomplete range {meta["Path"]}/{index}')

if __name__=='__main__':
 signal.signal(signal.SIGHUP,signal.SIG_IGN)
 files=[f for f in json.loads((DEST/'source_api.json').read_text())['Data']['Files'] if f['Type']=='blob' and f['Path']!='.gitattributes']
 save('download_status.json',{'status':'running','pid':os.getpid(),'strategy':'16 parallel resumable 32MiB ranges','expected_bytes':sum(f['Size'] for f in files)})
 try:
  for f in files:
   if f['Size']<100*1024*1024:fetch(f)
  large=[f for f in files if f['Size']>=100*1024*1024]
  jobs=[]
  for f in large:
   path=DEST/f['Path']
   if path.exists() and path.stat().st_size==f['Size'] and sha(path)==f['Sha256']:continue
   # Reuse already transferred prefix from the previous single-connection downloader.
   prefix=path.with_suffix(path.suffix+'.part')
   folder=DEST/'.chunks'/f['Path'];folder.mkdir(parents=True,exist_ok=True)
   if prefix.exists():
    with prefix.open('rb') as handle:
     for i in range((prefix.stat().st_size+CHUNK-1)//CHUNK):
      block=handle.read(CHUNK);part=folder/f'{i:05d}.part'
      if not part.exists():part.write_bytes(block)
   for i,start in enumerate(range(0,f['Size'],CHUNK)):jobs.append((f,i,start,min(f['Size']-1,start+CHUNK-1)))
  done=0
  with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
   for result in pool.map(download_range,jobs):
    done+=1
    save('download_status.json',{'status':'running','pid':os.getpid(),'completed_chunks':done,'total_chunks':len(jobs),'expected_bytes':sum(f['Size'] for f in files)})
    if done%10==0:print('chunks',done,'/',len(jobs),flush=True)
  for f in large:
   path=DEST/f['Path']
   if path.exists() and path.stat().st_size==f['Size'] and sha(path)==f['Sha256']:continue
   temporary=path.with_suffix(path.suffix+'.assembling')
   with temporary.open('wb') as target:
    for i in range((f['Size']+CHUNK-1)//CHUNK):
     with (DEST/'.chunks'/f['Path']/f'{i:05d}.part').open('rb') as source:
      for b in iter(lambda:source.read(8*1024*1024),b''):target.write(b)
    target.flush();os.fsync(target.fileno())
   if temporary.stat().st_size!=f['Size'] or sha(temporary)!=f['Sha256']:raise ValueError('Final checksum mismatch')
   temporary.replace(path);print('verified',f['Path'],flush=True)
  save('download_manifest.json',{'model_id':'Qwen/Qwen2-VL-2B-Instruct','source':'Qwen official ModelScope repository','files':files})
  save('download_status.json',{'status':'completed','files':len(files),'bytes':sum(f['Size'] for f in files),'all_sha256_verified':True})
 except BaseException as e:
  save('download_status.json',{'status':'failed','error':str(e)});raise
