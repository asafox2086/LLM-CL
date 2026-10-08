"""Fetch immutable official ModelScope files; resume partial transfers and verify SHA256."""
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'model/Qwen2-VL-2B-Instruct'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(name,value):
 p=DEST/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)

def fetch(meta):
 path=DEST/meta['Path'];partial=path.with_suffix(path.suffix+'.part')
 if path.exists() and path.stat().st_size==meta['Size'] and sha(path)==meta['Sha256']:return meta
 url=f"https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct/resolve/{meta['Revision']}/{meta['Path']}"
 for attempt in range(5):
  command=['curl','--noproxy','*','-sSL','--fail','--connect-timeout','20','--max-time','1800','--retry','2','-C','-',url,'-o',str(partial)]
  result=subprocess.run(command)
  if result.returncode==0 and partial.stat().st_size==meta['Size']:
   if sha(partial)!=meta['Sha256']:raise ValueError('Checksum mismatch '+meta['Path'])
   partial.replace(path);print('verified',meta['Path'],meta['Size'],flush=True);return meta
  time.sleep(2)
 raise RuntimeError('Download failed: '+meta['Path'])

if __name__=='__main__':
 signal.signal(signal.SIGHUP,signal.SIG_IGN)
 files=[f for f in json.loads((DEST/'source_api.json').read_text())['Data']['Files'] if f['Type']=='blob' and f['Path']!='.gitattributes']
 save('download_status.json',{'status':'running','pid':os.getpid(),'model':'Qwen/Qwen2-VL-2B-Instruct','expected_bytes':sum(f['Size'] for f in files)})
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(fetch,files))
  save('download_manifest.json',{'model_id':'Qwen/Qwen2-VL-2B-Instruct','source':'Qwen official ModelScope repository','files':verified})
  save('download_status.json',{'status':'completed','files':len(verified),'bytes':sum(f['Size'] for f in files),'all_sha256_verified':True})
 except BaseException as e:
  save('download_status.json',{'status':'failed','error':str(e)});raise
