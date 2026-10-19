"""Measure exact within-image processed-view duplication in all frozen TRAIN.

No model, TEST, question/answer editing, selection or performance claim.
QA-weighted and unique-source-image views are both reported.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/train_view_census/v1'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(name,value):
    temp=REPORT/(name+'.tmp');target=REPORT/name
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(target)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    from edge_cpu_visual_probe_20261019 import input_function
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    REPORT.mkdir(parents=True,exist_ok=True)
    assert not (REPORT/'STARTED.json').exists(),'No blind retry'
    write('STARTED.json',dict(pid=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],source_sha256=sha(__file__)))
    meta_path=ROOT/'summary/edge_deployment_20261019/static_export/v1/OSFT_EXPORTED.json'
    meta=json.loads(meta_path.read_text());package=ROOT/meta['package']
    for rel,info in meta['package_files'].items():
        if not rel.endswith(('.safetensors','.bin','.pt')):assert sha(package/rel)==info['sha256']
    processor=AutoProcessor.from_pretrained(package,local_files_only=True)
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest=json.loads(manifest_path.read_text());all_rows=[]
    for task in manifest['tasks']:
        info=task['files']['train'];p=ROOT/info['path'];assert sha(p)==info['sha256']
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        all_rows.extend((task,row) for row in rows)
    assert len(all_rows)==1313 and len(manifest['tasks'])==19
    cache={};domains={};tasks={};start=time.perf_counter()
    def image_record(name):
        if name in cache:return cache[name]
        path=(ROOT/name).resolve();assert path.is_relative_to(ROOT.resolve())
        source_sha=sha(path)
        with Image.open(path) as image:
            rgb=image.convert('RGB');original=list(rgb.size)
            if rgb.width*rgb.height>401408:
                ratio=math.sqrt(401408/(rgb.width*rgb.height))
                rgb=rgb.resize((max(1,int(rgb.width*ratio)),max(1,int(rgb.height*ratio))),Image.Resampling.BICUBIC)
            assert rgb.width*rgb.height<=401408
            processed=processor.image_processor(images=[rgb],return_tensors='pt')
            pixels=processed['pixel_values'];assert pixels.ndim==5 and len(pixels)==1
            hashes=[hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for t in pixels[0]]
            unique=[];seen={};inverse=[]
            for i,h in enumerate(hashes):
                if h not in seen:seen[h]=len(unique);unique.append(i)
                else:assert torch.equal(pixels[0,i],pixels[0,unique[seen[h]]])
                inverse.append(seen[h])
            r=dict(path=name,source_sha256=source_sha,original_size=original,common_size=list(rgb.size),
                views=len(hashes),unique_views=len(unique),first_indices=unique,inverse=inverse,
                pixel_shape=list(pixels.shape),pixel_dtype=str(pixels.dtype),view_sha256=hashes)
        cache[name]=r
        return r
    # Verify the hand-extracted pure image path against the actual full helper,
    # on the five predeclared TRAIN requests before using it on all images.
    prepare=input_function();gates=[]
    for task_id in [1,2,12,16,17]:
        candidates=[(t,r) for t,r in all_rows if t['task_id']==f'T{task_id:02d}']
        task,row=min(candidates,key=lambda pair:hashlib.sha256(('edge-profile42:'+pair[1]['id']).encode()).digest())
        batch,_=prepare(row,task,processor,'cpu');assert len(row['images'])==1
        r=image_record(row['images'][0]);got=[hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest() for x in batch['pixel_values'][0]]
        assert got==r['view_sha256']
        gates.append(dict(id=row['id'],pure_image_path_exact_to_frozen_helper=True))
    for index,(task,row) in enumerate(all_rows):
        items=[image_record(name) for name in row.get('images',[])];assert items
        raw=sum(x['views'] for x in items);unique=sum(x['unique_views'] for x in items)
        for table,key in [(domains,task['domain']),(tasks,task['task_id'])]:
            a=table.setdefault(key,dict(QA=0,image_references=0,raw_views=0,exact_unique_within_each_image_views=0,QA_with_duplicates=0,source_image_paths=set()))
            a['QA']+=1;a['image_references']+=len(items);a['raw_views']+=raw
            a['exact_unique_within_each_image_views']+=unique;a['QA_with_duplicates']+=int(raw>unique)
            a['source_image_paths'].update(row['images'])
        if index%100==0:write('progress.json',dict(TRAIN_QA=index+1,unique_images=len(cache),MODEL_FORWARDS=0,TEST_QA=0))
    for table in [domains,tasks]:
        for a in table.values():
            refs=a.pop('source_image_paths');images=[cache[p] for p in sorted(refs)]
            a.update(unique_source_images=len(images),unique_source_raw_views=sum(x['views'] for x in images),
                unique_source_reused_views=sum(x['unique_views'] for x in images),
                unique_source_images_with_duplicates=sum(x['views']>x['unique_views'] for x in images))
    assert sum(v['QA'] for v in domains.values())==1313
    assert not torch.cuda.is_initialized()
    write('COMPLETE.json',dict(complete=True,TRAIN_QA=1313,tasks=19,domains=domains,native_tasks=tasks,
        unique_source_images=len(cache),images=[cache[p] for p in sorted(cache)],
        source_sha256=sha(__file__),manifest_sha256=sha(manifest_path),processor_export_sha256=sha(meta_path),
        official_helper_pixel_gates=gates,elapsed_seconds=time.perf_counter()-start,
        MODEL_FORWARDS=0,TEST_QA=0,CUDA_initialized=False,no_answers_or_questions_changed=True,
        no_cross_request_caching_claim=True,not_latency_or_quality=True,
        not_a_property_caused_by_low_data=True,not_all_benchmarks_or_backbones=True))


if __name__=='__main__':
    try:run()
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True);write('FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
