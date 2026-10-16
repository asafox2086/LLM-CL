"""Direct IPv4 mirror download; pinned official metadata, SHA verification, no proxy."""
import concurrent.futures,hashlib,json,socket,time
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
original=socket.getaddrinfo
socket.getaddrinfo=lambda host,port,family=0,type=0,proto=0,flags=0:original(host,port,socket.AF_INET,type,proto,flags)

def download(f):
    s=requests.Session();s.trust_env=False
    path=BASE/f['rfilename'];partial=path.with_suffix(path.suffix+'.partial')
    for attempt in range(6):
        try:
            if not path.exists() or path.stat().st_size!=f['size']:
                offset=partial.stat().st_size if partial.exists() else 0
                url=f"https://hf-mirror.com/llava-hf/llava-onevision-qwen2-0.5b-ov-hf/resolve/{META['sha']}/{f['rfilename']}?download=true&t={time.time_ns()}"
                with s.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(20,90)) as response:
                    response.raise_for_status()
                    if response.status_code!=206:offset=0
                    with partial.open('ab' if offset else 'wb') as out:
                        for block in response.iter_content(4*1024**2):out.write(block)
                assert partial.stat().st_size==f['size']
                partial.replace(path)
            h=hashlib.sha256();g=hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
            with path.open('rb') as inp:
                for block in iter(lambda:inp.read(8*1024**2),b''):h.update(block);g.update(block)
            expected=f.get('lfs',{}).get('sha256')
            assert h.hexdigest()==expected if expected else g.hexdigest()==f['blobId']
            print('VERIFIED',path.name,path.stat().st_size,flush=True)
            return {'file':path.name,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'upstream_lfs_sha256':expected}
        except Exception as exc:
            print('RETRY',path.name,attempt,type(exc).__name__,flush=True)
            if attempt==5:raise
            time.sleep(3)

if __name__=='__main__':
    META=json.loads((BASE/'upstream_metadata.json').read_text())
    assert META['sha']=='74dd0bf867a4cda7950c17663794267c60cf4b40'
    files=[f for f in META['siblings'] if '/' not in f['rfilename'] and f['rfilename'].endswith(('.json','.txt','.safetensors','.md'))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:verified=list(pool.map(download,files))
    (BASE/'download_integrity.json').write_text(json.dumps({'repo':META['id'],'revision':META['sha'],'proxy':False,'files':verified,'complete':True},indent=2)+'\n')
