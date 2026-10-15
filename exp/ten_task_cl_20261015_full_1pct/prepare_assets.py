"""Direct, resumable retrieval of official LLaVA-Med annotations and PMC figures."""
import base64
import concurrent.futures
import hashlib
import io
import json
import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/LLaVA_Med'
SESSION = requests.Session()
SESSION.trust_env = False
REV = 'main'
FILES = ['data/instruct/llava_med_instruct_60k.json',
         'data/eval/llava_med_eval_qa50_qa.jsonl',
         'data/llava_med_image_urls.jsonl', 'data/llava_med_test_image_urls.jsonl']


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    partial = dest.with_suffix(dest.suffix+'.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    with SESSION.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                     stream=True, timeout=(15, 40)) as response:
        response.raise_for_status()
        mode = 'ab' if offset and response.status_code == 206 else 'wb'
        with partial.open(mode) as f:
            for blob in response.iter_content(1024*1024):
                f.write(blob)
    partial.replace(dest)


def annotations():
    global REV
    info = SESSION.get('https://api.github.com/repos/microsoft/LLaVA-Med/commits/main', timeout=20)
    info.raise_for_status()
    REV = info.json()['sha']
    evidence = []
    for name in FILES:
        url = f'https://media.githubusercontent.com/media/microsoft/LLaVA-Med/{REV}/{name}'
        dest = DATA / 'raw' / Path(name).name
        # Validate exact content against the official Git-LFS SHA, not only JSON parsing.
        pointer = SESSION.get(f'https://api.github.com/repos/microsoft/LLaVA-Med/contents/{name}',
                              params={'ref': REV}, timeout=20)
        pointer.raise_for_status()
        text = base64.b64decode(pointer.json()['content']).decode()
        expected = next(x.split('sha256:')[1] for x in text.splitlines() if x.startswith('oid sha256:'))
        size=int(next(x.split()[1] for x in text.splitlines() if x.startswith('size ')))
        download_ranges(url,dest,size)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert digest == expected
        evidence.append({'file': str(dest.relative_to(ROOT)), 'url': url, 'sha256': digest,
                         'bytes': dest.stat().st_size})
    (DATA / 'source_manifest.json').write_text(json.dumps({'source': 'microsoft/LLaVA-Med',
        'revision': REV, 'network': 'direct; trust_env=False', 'venue': 'NeurIPS 2023 Datasets and Benchmarks',
        'files': evidence, 'images_complete': False}, indent=2)+'\n')


def download_ranges(url,dest,size):
    if dest.exists():return
    dest.parent.mkdir(parents=True,exist_ok=True)
    block=4*1024*1024
    legacy=dest.with_suffix(dest.suffix+'.partial')
    chunks=dest.parent/(dest.name+'.chunks');chunks.mkdir(exist_ok=True)
    def get(start):
        end=min(size,start+block);p=chunks/str(start)
        if p.exists() and p.stat().st_size==end-start:return p
        if legacy.exists() and legacy.stat().st_size>=end:
            with legacy.open('rb') as f:f.seek(start);data=f.read(end-start)
        else:
            with SESSION.get(url,headers={'Range':f'bytes={start}-{end-1}'},timeout=(15,60)) as response:
                response.raise_for_status()
                assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {start}-')
                data=response.content
        assert len(data)==end-start
        p.write_bytes(data);return p
    starts=list(range(0,size,block))
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        for i,_ in enumerate(pool.map(get,starts),1):
            print(json.dumps({'event':'annotation_chunks','file':dest.name,'chunks_done':i,'chunks_total':len(starts)}),flush=True)
    with dest.open('wb') as f:
        for start in starts:f.write((chunks/str(start)).read_bytes())


def acquire_figures():
    train = json.loads((DATA/'raw/llava_med_instruct_60k.json').read_text())
    test = [json.loads(x) for x in (DATA/'raw/llava_med_eval_qa50_qa.jsonl').read_text().splitlines()]
    needed = {Path(x['image']).stem for x in train + test}
    mapping = {}
    for file in ['llava_med_image_urls.jsonl', 'llava_med_test_image_urls.jsonl']:
        for line in (DATA/'raw'/file).open():
            row = json.loads(line)
            if row['pair_id'] in needed:
                mapping[row['pair_id']] = row
    assert needed <= mapping.keys(), f'Missing official figure URLs for {len(needed-mapping.keys())} images'
    groups = defaultdict(list)
    for key in sorted(needed):
        groups[mapping[key]['pmc_tar_url']].append(mapping[key])
    (DATA/'images').mkdir(exist_ok=True)

    def job(item):
        url, rows = item
        missing = [row for row in rows if not (DATA/'images'/f"{row['pair_id']}.jpg").exists()]
        if not missing:
            return {'url': url, 'status': 'already_present'}
        archive = DATA/'raw/pmc_articles'/Path(url).name
        try:
            download(url, archive)
            with tarfile.open(archive, 'r:*') as t:
                members = {m.name: m for m in t if m.isfile()}
                for row in missing:
                    source = row['image_file_path']
                    member = members.get(source)
                    if member is None:
                        matches = [v for k,v in members.items() if k.endswith('/'+Path(source).name)]
                        assert len(matches) == 1, source
                        member = matches[0]
                    blob = t.extractfile(member).read()
                    with Image.open(io.BytesIO(blob)) as image:
                        image.verify()
                    # Keep original image bytes and native content, no JPEG recompression.
                    (DATA/'images'/f"{row['pair_id']}.jpg").write_bytes(blob)
            return {'url': url, 'status': 'complete', 'images': len(missing)}
        except Exception as exc:
            return {'url': url, 'status': 'error', 'error': str(exc)[:500]}

    (DATA/'download_progress.jsonl').touch(exist_ok=True)
    failures = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, result in enumerate(pool.map(job, groups.items()), 1):
            failures += result['status'] == 'error'
            with (DATA/'download_progress.jsonl').open('a') as f:
                f.write(json.dumps({'unix': time.time(), **result})+'\n')
            if index % 25 == 0:
                print(json.dumps({'event': 'llavamed_images', 'articles_done': index,
                                  'total_articles': len(groups), 'errors': failures}), flush=True)
    assert failures == 0, f'{failures} source article downloads failed; retry resumes, never exclude unavailable samples'
    assert all((DATA/'images'/f'{key}.jpg').exists() for key in needed)
    p=DATA/'source_manifest.json';m=json.loads(p.read_text());m.update(images_complete=True, image_count=len(needed))
    p.write_text(json.dumps(m,indent=2)+'\n')


if __name__ == '__main__':
    DATA.mkdir(parents=True, exist_ok=True)
    annotations()
    acquire_figures()
