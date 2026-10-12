"""Prepare only fixed experiment assets, using direct connections and original QA."""
import concurrent.futures
import hashlib
import io
import json
import re
import struct
import subprocess
import threading
import time
import unicodedata
import zipfile
import zlib
from collections import Counter
from pathlib import Path
import requests
from prepare_data import ROOT, OUT, sha

URL = 'https://researchdata.aston.ac.uk/380/1/SemArt.zip'
ART_SIZE = 3327241783
LOCAL_PART = ROOT/'data/AQUA/raw/SemArt.zip.partial'
thread_local = threading.local()


def session():
    if not hasattr(thread_local,'session'):
        thread_local.session = requests.Session()
        thread_local.session.trust_env = False
    return thread_local.session


def remote_range(offset, length):
    if LOCAL_PART.is_file() and LOCAL_PART.stat().st_size >= offset+length:
        with LOCAL_PART.open('rb') as stream:
            stream.seek(offset)
            return stream.read(length)
    for attempt in range(5):
        try:
            response = session().get(URL+f'?llmcl_range={offset}',
                headers={'Range':f'bytes={offset}-{offset+length-1}'}, timeout=(20,90))
            response.raise_for_status()
            if response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                raise ValueError('server did not honor range; refusing full-archive transfer')
            if len(response.content)!=length:
                raise ValueError('incomplete range')
            return response.content
        except Exception as error:
            print('DIRECT_RANGE_RETRY',offset,length,type(error).__name__,flush=True)
            if attempt==4:
                raise
            time.sleep(min(2*(attempt+1),10))


class RemoteZip(io.RawIOBase):
    def __init__(self):
        self.position = 0
    def seekable(self):
        return True
    def readable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence==0 else self.position+offset if whence==1 else ART_SIZE+offset
        return self.position
    def read(self, size=-1):
        size = ART_SIZE-self.position if size<0 else min(size,ART_SIZE-self.position)
        if size<=0:
            return b''
        if size>8*1024**2:
            raise ValueError('unbounded ZIP metadata read')
        data = remote_range(self.position,size)
        self.position += len(data)
        return data


def prepare_art():
    selected = set()
    for split in ['train','dev','test']:
        for line in (OUT/f'T10_{split}.jsonl').open():
            selected.add(json.loads(line)['metadata']['original_image'])
    destination = ROOT/'data/AQUA/selected_images'
    destination.mkdir(exist_ok=True)
    archive_file = ROOT/'data/AQUA/raw/SemArt.zip'
    with zipfile.ZipFile(archive_file if archive_file.is_file() else RemoteZip()) as archive:
        mapping = {Path(info.filename).name:info for info in archive.infolist()}
    missing = selected-mapping.keys()
    if missing:
        raise ValueError(f'missing original image members: {sorted(missing)[:10]}')
    def fetch_image(name):
        path = destination/name
        info = mapping[name]
        if path.is_file() and path.stat().st_size==info.file_size and zlib.crc32(path.read_bytes())==info.CRC:
            return name,sha(path)
        if archive_file.is_file():
            with zipfile.ZipFile(archive_file) as archive:
                data = archive.read(info.filename)
        else:
            length = min(ART_SIZE-info.header_offset,30+len(info.filename.encode())+2048+info.compress_size)
            packed = remote_range(info.header_offset,length)
            if packed[:4]!=b'PK\x03\x04':
                raise ValueError('invalid ZIP local header')
            name_length,extra_length = struct.unpack_from('<HH',packed,26)
            start = 30+name_length+extra_length
            if len(packed)<start+info.compress_size:
                packed = remote_range(info.header_offset,start+info.compress_size)
            payload = packed[start:start+info.compress_size]
            data = zlib.decompress(payload,-15) if info.compress_type==zipfile.ZIP_DEFLATED else payload
        if len(data)!=info.file_size or zlib.crc32(data)!=info.CRC:
            raise ValueError('original image size/CRC mismatch')
        temporary = path.with_suffix(path.suffix+'.partial')
        temporary.write_bytes(data)
        temporary.replace(path)
        return name,hashlib.sha256(data).hexdigest()
    checksums = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name,digest in pool.map(fetch_image,sorted(selected)):
            checksums[name] = digest
            if len(checksums)%25==0:
                print('ART_SELECTED_IMAGES',len(checksums),len(selected),flush=True)
    (OUT/'art_selected_images_sha256.json').write_text(json.dumps(checksums,indent=2)+'\n')
    print('ART_SELECTED_READY',len(selected),flush=True)


def norm(value):
    return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',value).casefold())


def words(value):
    return re.findall(r'[a-z0-9]+',value.lower())


def pdf_member(title,mapping):
    key=norm(title)
    if key in mapping:
        return mapping[key],'normalized exact official filename'
    candidates=[value for candidate,value in mapping.items() if candidate.startswith(key)]
    if len(candidates)!=1:
        raise ValueError(f'no unique official PDF exhibit for contract: {title}')
    # Some official filenames append a second agreement name to the same SEC exhibit.
    return candidates[0],'unique full contract-title prefix in official PDF exhibit filename'


def retrieve(question, texts, top=2):
    # Document-local BM25. Does not accept answers or evidence spans.
    tokens = [words(text) for text in texts]
    query = set(words(question))
    average = sum(map(len,tokens))/max(1,len(tokens))
    import math
    frequency = {word:sum(word in set(text) for text in tokens) for word in query}
    scores = []
    for index,text in enumerate(tokens):
        counts = Counter(text)
        score = 0.0
        for word in query:
            tf = counts[word]
            idf = math.log(1+(len(tokens)-frequency[word]+0.5)/(frequency[word]+0.5))
            score += idf*tf*2.5/(tf+1.5*(0.25+0.75*len(text)/max(average,1)))
        scores.append((score,index))
    return [index for _,index in sorted(scores,key=lambda value:(-value[0],value[1]))[:top]]


def prepare_law():
    directory = ROOT/'data/CUAD/experiment_pdf'
    pages = ROOT/'data/CUAD/experiment_pages'
    directory.mkdir(exist_ok=True)
    pages.mkdir(exist_ok=True)
    with zipfile.ZipFile(ROOT/'data/CUAD/raw/CUAD_v1_official.zip') as archive:
        mapping = {norm(Path(name).stem):name for name in archive.namelist() if name.lower().endswith('.pdf')}
        text_cache = {}
        for split in ['train','dev','test']:
            path = OUT/f'T06_{split}.jsonl'
            rows = [json.loads(line) for line in path.open()]
            for index,row in enumerate(rows):
                title = row['metadata']['document_title']
                key = norm(title)
                member,match_rule=pdf_member(title,mapping)
                document_id = hashlib.sha256(title.encode()).hexdigest()[:24]
                pdf = directory/(document_id+'.pdf')
                if not pdf.exists():
                    pdf.write_bytes(archive.read(member))
                if key not in text_cache:
                    text_path=directory/(document_id+'.pages.txt')
                    if text_path.is_file():
                        output=text_path.read_text()
                    else:
                        output = subprocess.check_output(['pdftotext','-layout','-enc','UTF-8',str(pdf),'-']).decode()
                        text_path.write_text(output)
                    texts = output.split('\f')
                    if texts and not texts[-1].strip():
                        texts.pop()
                    if not texts:
                        raise ValueError('empty contract pages')
                    text_cache[key] = texts
                indexes = retrieve(row['question'],text_cache[key],2)
                images = []
                for page in sorted(indexes):
                    stem = pages/f'{document_id}_{page+1:04d}'
                    image = stem.with_suffix('.png')
                    if not image.is_file():
                        subprocess.run(['pdftoppm','-f',str(page+1),'-l',str(page+1),'-singlefile',
                                        '-scale-to','1600','-png',str(pdf),str(stem)],check=True,
                                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    images.append(str(image.relative_to(ROOT)))
                row['images'] = images
                row['metadata'].update({'selected_pages_1based':[page+1 for page in sorted(indexes)],
                     'page_selection':'BM25 question-only, top 2, document order',
                     'original_pdf_member':member,'pdf_title_match_rule':match_rule,'requires_pdf_render':False,
                     'original_pdf_sha256':sha(pdf)})
                if index%200==0:
                    print('LAW_PAGES',split,index,len(rows),flush=True)
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    print('LAW_SELECTED_READY',len(text_cache),'contracts',flush=True)


def prepare_minilm():
    repo = 'sentence-transformers/all-MiniLM-L6-v2'
    model_dir = ROOT/'model/all-MiniLM-L6-v2'
    model_dir.mkdir(exist_ok=True)
    meta_path = model_dir/'source_api.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        response = session().get('https://hf-mirror.com/api/models/'+repo,timeout=(20,90))
        response.raise_for_status()
        meta = response.json()
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
    revision = meta['sha']
    filenames = ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','model.safetensors']
    available = {entry['rfilename'] for entry in meta['siblings']}
    records = []
    for name in filenames:
        if name not in available:
            raise ValueError(f'missing expected safe model file: {name}')
        path = model_dir/name
        if not path.is_file():
            response = session().get(f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}',stream=True,timeout=(20,120))
            response.raise_for_status()
            temp = path.with_suffix(path.suffix+'.partial')
            with temp.open('wb') as stream:
                for block in response.iter_content(1024**2):
                    stream.write(block)
            temp.replace(path)
        records.append({'file':name,'sha256':sha(path),'bytes':path.stat().st_size})
    (OUT/'instruction_encoder_manifest.json').write_text(json.dumps({'repo':repo,'revision':revision,'purpose':'SMoLoRA author instruction encoder, not CL backbone','files':records},indent=2)+'\n')
    print('MINILM_READY',revision,flush=True)


def finalize():
    manifest_path = ROOT/'data/TenTaskCL/final_task_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    from PIL import Image
    for task in manifest['tasks']:
        for split,info in task['files'].items():
            path = ROOT/info['path']
            rows = [json.loads(line) for line in path.open()]
            assert len(rows)==manifest['counts_per_task'][split]
            assert [row['id'] for row in rows]==task['sample_ids'][split]
            for row in rows:
                if task['modality']=='image' and not row['images']:
                    raise ValueError(f'missing image input {row["id"]}')
                for name in row['images']:
                    with Image.open(ROOT/name) as image:
                        image.verify()
            info['sha256'] = sha(path)
        task['ready_for_formal_training'] = True
    manifest['formal_ready'] = True
    manifest['assets_pending'] = []
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'selection_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('ALL_TEN_TASK_INPUTS_READY',flush=True)


if __name__ == '__main__':
    prepare_law()
    prepare_art()
    prepare_minilm()
    finalize()
