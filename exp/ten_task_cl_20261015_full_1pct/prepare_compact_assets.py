"""Download the original RSVQA-LR release and AQUA paintings directly, without a proxy."""
import concurrent.futures
import hashlib
import json
import socket
import zipfile
from pathlib import Path
import requests
from prepare_more_assets import semart

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/RSVQA_LR'

def main():
    session=requests.Session();session.trust_env=False
    # The server resolver returns 0.0.0.0 for Zenodo. Resolve only this public
    # host in this process; retain HTTPS hostname/certificate verification.
    dns=session.get('https://dns.alidns.com/resolve',params={'name':'zenodo.org','type':'A'},timeout=20)
    dns.raise_for_status();addresses=[a['data'] for a in dns.json()['Answer'] if a['type']==1]
    original=socket.getaddrinfo
    def lookup(host,port,*args,**kwargs):
        if host=='zenodo.org':
            return [entry for ip in addresses for entry in original(ip,port,socket.AF_INET,socket.SOCK_STREAM)]
        return original(host,port,*args,**kwargs)
    socket.getaddrinfo=lookup
    raw=DATA/'raw';raw.mkdir(parents=True,exist_ok=True)
    response=session.get('https://zenodo.org/api/records/6344334',timeout=30);response.raise_for_status()
    record=response.json();(raw/'zenodo_record.json').write_text(json.dumps(record,indent=2)+'\n')
    def acquire(info):
        p=raw/info['key'];expected=info['checksum'].split(':')[1]
        if p.exists() and hashlib.md5(p.read_bytes()).hexdigest()==expected:return
        partial=p.with_suffix(p.suffix+'.partial');offset=partial.stat().st_size if partial.exists() else 0
        if info['size']>32*2**20:
            chunks=raw/(info['key']+'.chunks');chunks.mkdir(exist_ok=True)
            step=8*2**20;starts=list(range(0,info['size'],step))
            def part(start):
                stop=min(start+step,info['size']);target=chunks/str(start)
                if target.exists() and target.stat().st_size==stop-start:return
                if offset>=stop:
                    with partial.open('rb') as source:source.seek(start);target.write_bytes(source.read(stop-start))
                    return
                with session.get(info['links']['self'],headers={'Range':f'bytes={start}-{stop-1}'},timeout=90) as resp:
                    resp.raise_for_status();assert resp.status_code==206 and len(resp.content)==stop-start
                    assert resp.headers['Content-Range'].startswith(f'bytes {start}-{stop-1}/')
                    target.write_bytes(resp.content)
            with concurrent.futures.ThreadPoolExecutor(6) as pool:list(pool.map(part,starts))
            with partial.open('wb') as f:
                for start in starts:f.write((chunks/str(start)).read_bytes())
            assert hashlib.md5(partial.read_bytes()).hexdigest()==expected
            partial.replace(p);print(json.dumps({'downloaded':info['key'],'bytes':info['size']}),flush=True)
            return
        with session.get(info['links']['self'],headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=90) as resp:
            resp.raise_for_status()
            with partial.open('ab' if offset and resp.status_code==206 else 'wb') as f:
                for chunk in resp.iter_content(2**20):f.write(chunk)
        assert partial.stat().st_size==info['size'],info['key']
        assert hashlib.md5(partial.read_bytes()).hexdigest()==expected,info['key']
        partial.replace(p);print(json.dumps({'downloaded':info['key'],'bytes':info['size']}),flush=True)
    # Annotation files first, so their native splits/counts can be audited early.
    entries=sorted(record['files'],key=lambda x:x['size'])
    with concurrent.futures.ThreadPoolExecutor(4) as pool:list(pool.map(acquire,entries))
    dest=DATA/'images';dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(raw/'Images_LR.zip') as z:
        assert z.testzip() is None
        for member in z.infolist():
            if member.is_dir():continue
            target=dest/Path(member.filename).name
            if not target.exists():target.write_bytes(z.read(member))
    (DATA/'full_assets_ready.json').write_text(json.dumps({'ready':True,'record':'https://zenodo.org/records/6344334',
         'network':'direct; no proxy; process-local verified DNS lookup','checksums':'all source MD5 and archive CRC validated',
         'files':[{k:x[k] for k in ['key','size','checksum']} for x in record['files']]},indent=2)+'\n')
    semart()

if __name__=='__main__':main()
