"""One immutable ID/order ledger shared by all methods and both hosts."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=ROOT/'data/FiveDomainCL'/HERE.name


def main():
    lock={'experiment_id':HERE.name,'seed':42,'method_independent':True,'conditions':{}}
    assert not (DEST/'split_lock.json').exists()
    for c in ['full','pct01']:
        path=DEST/c/'manifest.json';m=json.loads(path.read_text());files={};ids={}
        for t in m['tasks']:
            for s,f in t['files'].items():
                p=ROOT/f['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
                rows=[json.loads(x) for x in p.read_text().splitlines()]
                if s=='train':rows.sort(key=lambda x:hashlib.sha256(f"42:{t['task']}:{x['id']}".encode()).digest())
                ids[f"{t['task_id']}/{s}"]=[x['id'] for x in rows]
                files[f['path']]=f['sha256'];p.chmod(0o444)
        idpath=DEST/c/'frozen_ids.json';idpath.write_text(json.dumps(ids,ensure_ascii=False)+'\n');idpath.chmod(0o444)
        lock['conditions'][c]={'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                             'frozen_ids_sha256':hashlib.sha256(idpath.read_bytes()).hexdigest(),'files':files}
        path.chmod(0o444)
    p=DEST/'split_lock.json';p.write_text(json.dumps(lock,indent=2)+'\n');p.chmod(0o444)
    print(json.dumps({c:{k:v for k,v in x.items() if k!='files'} for c,x in lock['conditions'].items()}))


if __name__=='__main__':main()
