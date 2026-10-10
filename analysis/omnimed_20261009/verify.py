import json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sel=json.loads((OUT/'selection.json').read_text());manifest=json.loads((OUT/'mirror_download_manifest.json').read_text());checks=[]
for r in manifest:
    p=OUT/'data'/r['path'];sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert p.stat().st_size==r['size'] and sha==r['sha256'],p
    checks.append(dict(path=r['path'],size=r['size'],sha256=sha))
overlap=[]
for p in sel['probes']:
    shas={s:{r['image_sha256'] for r in rs} for s,rs in p['splits'].items()}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:assert not shas[a]&shas[b]
    overlap.append(dict(source=p['source'],overlap=0,counts={s:len(x) for s,x in shas.items()}))
old=json.loads((ROOT/'analysis/research_grounding_20261008/selection.json').read_text())
priortrain={r['image_metadata']['sha256'] for unit in old['train_units'] for r in unit}
pool=json.loads((OUT/'pool.json').read_text());assert not priortrain & {r['image_sha256'] for r in pool}
result=dict(parquet_files_verified=len(checks),parquet_bytes=sum(x['size'] for x in checks),pool_qa=len(pool),pool_images=len({r['image_sha256'] for r in pool}),custom_split_overlap=overlap,prior_vqarad_training_sha_overlap=0,original_image_sample_audit=json.loads((OUT/'image_transport_audit.json').read_text()))
(OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
