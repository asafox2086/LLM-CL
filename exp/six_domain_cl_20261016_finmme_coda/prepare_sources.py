"""Acquire pinned author releases directly; no environment/Clash proxy is used.

Only English CODA-LM Train/Test are required. Official Val and Mini are not
training or evaluation inputs. Original Parquet files and embedded image bytes
are preserved, including author-drawn regional bounding boxes.
"""
import concurrent.futures
import hashlib
import json
import socket
import threading
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SPECS = [
    ('FinMME', 'luojunyu/FinMME', lambda p: p.startswith('data/') and p.endswith('.parquet')),
    ('CODA_LM', 'KaiChen1998/coda-lm-llava-format',
     lambda p: p.startswith(('English/Train-', 'English/Test-')) and p.endswith('.parquet')),
]
LOCAL = threading.local()
ORIGINAL_DNS = socket.getaddrinfo
DNS_CACHE = {}
DNS_LOCK = threading.Lock()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    return LOCAL.session


def dns(host, port, *args, **kwargs):
    try:
        entries = ORIGINAL_DNS(host, port, *args, **kwargs)
        if entries and all(e[4][0] not in ('0.0.0.0', '::') for e in entries):
            return entries
    except socket.gaierror:
        pass
    with DNS_LOCK:
        if host not in DNS_CACHE:
            # Resolver is contacted directly; system DNS and hosts stay unchanged.
            r = session().get('https://dns.alidns.com/resolve',
                              params={'name': host, 'type': 'A'}, timeout=30)
            r.raise_for_status()
            DNS_CACHE[host] = [x['data'] for x in r.json().get('Answer', []) if x['type'] == 1]
        addresses = DNS_CACHE[host]
    if not addresses:
        raise socket.gaierror('No public A record for ' + host)
    return [entry for ip in addresses
            for entry in ORIGINAL_DNS(ip, port, socket.AF_INET, socket.SOCK_STREAM)]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def acquire(base, repo, revision, info):
    path = base / 'raw' / info['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = info['lfs']['oid']
    assert len(expected) == 64 and set(expected) <= set('0123456789abcdef')
    if path.exists() and path.stat().st_size == info['size'] and digest(path) == expected:
        return
    chunks = path.parent / (path.name + '.chunks')
    chunks.mkdir(exist_ok=True)
    step = 8 * 2**20
    url = f'https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{info["path"]}'

    def part(start):
        stop = min(start + step, info['size'])
        target = chunks / str(start)
        if target.exists() and target.stat().st_size == stop - start:
            return
        for attempt in range(5):
            try:
                with session().get(url, headers={'Range': f'bytes={start}-{stop-1}'},
                                   timeout=(30, 90)) as r:
                    r.raise_for_status()
                    assert r.status_code == 206, (r.status_code, info['path'])
                    assert r.headers['Content-Range'].startswith(f'bytes {start}-{stop-1}/')
                    assert len(r.content) == stop - start
                    tmp = target.with_suffix('.partial')
                    tmp.write_bytes(r.content)
                    tmp.replace(target)
                return
            except Exception as e:
                if attempt == 4:
                    raise
                print(json.dumps({'retry': info['path'], 'offset': start, 'error': str(e)[:180]}), flush=True)
                time.sleep(min(2**attempt, 15))

    starts = list(range(0, info['size'], step))
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        list(pool.map(part, starts))
    partial = path.with_suffix(path.suffix + '.partial')
    with partial.open('wb') as f:
        for start in starts:
            f.write((chunks / str(start)).read_bytes())
    assert partial.stat().st_size == info['size']
    assert digest(partial) == expected, ('source SHA256 mismatch', info['path'])
    partial.replace(path)
    for start in starts:
        (chunks / str(start)).unlink()
    chunks.rmdir()
    print(json.dumps({'downloaded': str(path.relative_to(ROOT)), 'bytes': info['size'],
                      'sha256': expected}), flush=True)


def main():
    socket.getaddrinfo = dns
    for dataset, repo, selected in SPECS:
        base = ROOT / 'data' / dataset
        base.mkdir(parents=True, exist_ok=True)
        pinned=base/'source_download_manifest.json'
        if pinned.exists():
            prior=json.loads(pinned.read_text())
            assert prior['repository']==repo
            revision=prior['revision']
        else:
            r = session().get('https://hf-mirror.com/api/datasets/' + repo, timeout=30)
            r.raise_for_status()
            revision = r.json()['sha']
        r = session().get(f'https://hf-mirror.com/api/datasets/{repo}/tree/{revision}',
                          params={'recursive': 'true', 'limit': 1000}, timeout=30)
        r.raise_for_status()
        assert 'next' not in r.links, 'Pagination must be handled before selecting source files'
        files = [x for x in r.json() if x['type'] == 'file' and selected(x['path'])]
        assert files
        card = session().get(f'https://hf-mirror.com/datasets/{repo}/raw/{revision}/README.md', timeout=30)
        card.raise_for_status()
        (base / 'SOURCE_README.md').write_text('\n'.join(line.rstrip() for line in card.text.splitlines()).rstrip()+'\n', encoding='utf-8')
        audit = {'dataset': dataset, 'repository': repo, 'revision': revision,
                 'network': 'direct HTTPS; trust_env=False; process-local DNS fallback only',
                 'files': [{k: x[k] for k in ('path', 'size', 'lfs')} for x in files]}
        (base / 'source_download_manifest.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps({'dataset': dataset, 'files': len(files),
                          'expected_bytes': sum(x['size'] for x in files)}), flush=True)
        for info in files:
            acquire(base, repo, revision, info)
        import pyarrow.parquet as pq
        counts = {}
        for info in files:
            role = 'train' if '/Train-' in info['path'] or '/train-' in info['path'] else 'test'
            counts[role] = counts.get(role, 0) + pq.ParquetFile(base / 'raw' / info['path']).metadata.num_rows
        audit.update(ready=True, source_rows=counts,
                     readiness_scope='Complete source Parquet SHA256 verified; semantic/image decoding audit pending')
        (base / 'source_files_ready.json').write_text(json.dumps(audit, indent=2) + '\n')
        print(json.dumps({'source_files_ready': dataset, 'rows': counts}), flush=True)


if __name__ == '__main__':
    main()
