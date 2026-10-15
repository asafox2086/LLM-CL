"""Materialize complete original GeoChat/SemArt assets; never select by QA type."""
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from PIL import Image
from prepare_assets import download

ROOT = Path(__file__).resolve().parents[2]


class MultipartReader(io.RawIOBase):
    def __init__(self, paths):
        self.files = [p.open('rb') for p in paths]
        self.sizes = [p.stat().st_size for p in paths]
        self.position = 0
        self.length = sum(self.sizes)

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.length+offset
        return self.position

    def read(self, size=-1):
        size = self.length-self.position if size < 0 else min(size, self.length-self.position)
        cursor, output, remaining = self.position, [], size
        for stream, n in zip(self.files, self.sizes):
            if cursor >= n:
                cursor -= n
                continue
            stream.seek(cursor)
            block = stream.read(min(remaining, n-cursor))
            output.append(block)
            remaining -= len(block)
            cursor = 0
            if not remaining:
                break
        self.position += size-remaining
        return b''.join(output)


def geochat():
    d = ROOT/'data/GeoChat_Instruct'
    paths = [d/'raw'/f'images_parta{x}' for x in 'abc']
    expected = [38654705664, 38654705664, 24218986482]
    if not all(p.exists() and p.stat().st_size == n for p,n in zip(paths, expected)):
        raise RuntimeError('GeoChat original archive is still transferring')
    expected_sha = ['247edd3873b3dc57b64ece898a417876890c2f270e9e09f28fec9eac93d333b5',
                    'ad43ffad4317c3c26b3a2e44634d29fd31d286663a1cb3fede05f5fffb732736',
                    'cfcc43f35b794ca147978f6bfec154003d7137344905252279fcc306e6ef9e70']
    proof=d/'full_source_verification.json'
    if not proof.exists():
        for p, digest in zip(paths, expected_sha):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==digest
        assert hashlib.sha256((d/'raw/GeoChat_Instruct.json').read_bytes()).hexdigest()=='fb5266c927874f86a72e71eb909ed1a831bbfd138c8ecb7a170de307d4946972'
        proof.write_text(json.dumps({'archive_sizes':expected,'archive_sha256':expected_sha,'annotation_verified':True},indent=2))
    rows=json.loads((d/'raw/GeoChat_Instruct.json').read_text())
    needed={Path(x['image']).name for x in rows if x.get('image')}
    with zipfile.ZipFile(MultipartReader(paths)) as z:
        names={Path(x).name:x for x in z.namelist() if not x.endswith('/')}
        assert needed<=names.keys()
        for i,name in enumerate(sorted(needed),1):
            p=d/'raw/images'/name
            if not p.exists():
                b=z.read(names[name])
                with Image.open(io.BytesIO(b)) as im:im.verify()
                p.write_bytes(b)
            if i%1000==0:print(json.dumps({'dataset':'GeoChat','images_done':i,'total':len(needed)}),flush=True)
    (d/'full_assets_ready.json').write_text(json.dumps({'images':len(needed),'all_original_question_types':True}))


def semart():
    d=ROOT/'data/AQUA'
    download('https://researchdata.aston.ac.uk/380/1/SemArt.zip',d/'raw/SemArt.zip')
    dest=d/'selected_images';dest.mkdir(exist_ok=True)
    needed={x['image'] for role in ['train','test','val'] for x in json.loads((d/'raw'/f'{role}.json').read_text())}
    with zipfile.ZipFile(d/'raw/SemArt.zip') as z:
        names={Path(x).name:x for x in z.namelist() if not x.endswith('/')}
        assert needed<=names.keys()
        for name in sorted(needed):
            p=dest/name
            if not p.exists():
                b=z.read(names[name])
                with Image.open(io.BytesIO(b)) as im:im.verify()
                p.write_bytes(b)
        for name in ['semart_train.csv','semart_val.csv','semart_test.csv']:
            if name in names:(d/'raw'/name).write_bytes(z.read(names[name]))
    (d/'full_assets_ready.json').write_text(json.dumps({'images':len(needed),'full_original_splits':True}))


if __name__=='__main__':
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        futures=[executor.submit(f) for f in [geochat,semart]]
        for f in futures:f.result()
