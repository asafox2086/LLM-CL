import json,hashlib,zipfile
from pathlib import Path
OUT=Path(__file__).resolve().parent
assert json.loads((OUT/'status.json').read_text())['status']=='completed'
assert json.loads((OUT/'adaptation_status.json').read_text())['status']=='completed'
updates=[json.loads(x) for x in (OUT/'adaptation_updates.jsonl').read_text().splitlines()]
assert len(updates)==423
for seed in [42,43,44]:assert [r['step'] for r in updates if r['seed']==seed]==list(range(1,142))
for script,protocol in [('train_readout_lora.py','adaptation_protocol.json'),('evaluate.py','status.json'),('prepare.py','protocol.json')]:
    assert hashlib.sha256((OUT/script).read_bytes()).hexdigest()==json.loads((OUT/protocol).read_text())['script_sha256'],script
allowed={'.py','.json','.jsonl','.md','.csv','.log','.png','.pdf'}
excluded={'data','images','features128','adaptation_checkpoints','__pycache__','runtime'}
files=[p for p in OUT.rglob('*') if p.is_file() and not set(p.relative_to(OUT).parts)&excluded and p.suffix in allowed and p.name!='file_sha256.json']
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(OUT/'file_sha256.json').write_text(json.dumps(manifest,indent=2))
files.append(OUT/'file_sha256.json')
with zipfile.ZipFile(OUT/'review.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,arcname='omnimed_20261009/'+str(p.relative_to(OUT)))
print('PACKAGE',len(files),'files',len(updates),'completed optimizer updates',flush=True)
