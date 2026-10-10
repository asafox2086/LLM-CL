import json,hashlib,collections,random,re
from pathlib import Path
import pyarrow.parquet as pq
OUT=Path(__file__).resolve().parent
def dump(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def signature(r):return (r['image_path'],r['question'],r['gt_answer'],tuple(r.get('option_'+c,'') for c in 'ABCD'))
official=[r for p in (OUT/'data'/'official').rglob('*.json') for r in json.loads(p.read_text())]
lookup={signature(r):r for r in official}
rows=[]; images={};miss=[]
folder=OUT/'images';folder.mkdir(exist_ok=True)
for p in sorted((OUT/'data').glob('*.parquet')):
    for batch in pq.ParquetFile(p).iter_batches(batch_size=128):
        for r in batch.to_pylist():
            sig=(r['image_path'],r['problem'],r['answer'],tuple(r['choice_'+c.lower()] for c in 'ABCD'))
            if sig not in lookup:miss.append(sig[:3]);continue
            original=lookup[sig]; data=r['image']['bytes'];sha=hashlib.sha256(data).hexdigest()
            im=folder/(sha+'.img')
            if not im.exists():im.write_bytes(data)
            a=dict(original,image=str(im),image_sha256=sha,choices=[r['choice_'+c.lower()] for c in 'ABCD'])
            matches=[i for i,x in enumerate(a['choices']) if x.strip().casefold()==a['gt_answer'].strip().casefold()]
            assert len(matches)==1,(a['question_id'],matches)
            a['answer_index']=matches[0]; rows.append(a);images[sha]=str(im)
    print('read',p.name,len(rows),flush=True)
assert not miss,('Unmatched mirror records',len(miss),miss[:3])
assert len({r['question_id'] for r in rows})==len(rows)
dump('pool.json',rows)
def order(rs,salt):return sorted(rs,key=lambda r:hashlib.sha256((salt+r['question_id']).encode()).hexdigest())
primary=[]
for category,n in [('Disease Diagnosis',512),('Anatomy Identification',128),('Modality Recognition',128),('Other Biological Attributes',128),('Lesion Grading',128)]:
    by=collections.defaultdict(list)
    for r in order([r for r in rows if r['question_type']==category],'primary42'):by[r['dataset']].append(r)
    selected=[];seen=set();counts=collections.Counter()
    while len(selected)<n:
        added=False
        for source in sorted(by):
            while by[source]:
                r=by[source].pop(0)
                if r['image_sha256'] in seen:continue
                selected.append(r);seen.add(r['image_sha256']);counts[source]+=1;added=True;break
            if len(selected)>=n:break
        if not added:break
    primary+=selected
    print('selected',category,len(selected),dict(counts),flush=True)

byimage=collections.defaultdict(dict)
for r in order(rows,'matched42'):byimage[r['image_sha256']].setdefault(r['question_type'],r)
matched=[]
for sha,kinds in sorted(byimage.items(),key=lambda x:hashlib.sha256(('paired42'+x[0]).encode()).hexdigest()):
    if 'Disease Diagnosis' in kinds and ('Anatomy Identification' in kinds or 'Modality Recognition' in kinds):
        matched.append([kinds['Disease Diagnosis'],kinds.get('Anatomy Identification',kinds.get('Modality Recognition'))])
    if len(matched)==128:break

# Separate supervised readout pilot. These are custom splits, not benchmark scores.
# Use exact labels, <=4 common classes per source; no CT slices in this pilot.
sources=collections.defaultdict(dict)
for r in order(rows,'probe42'):
    if r['question_type']=='Disease Diagnosis' and not re.search(r'computed|resonance|(^|\W)ct($|\W)|mri',r['modality_type'],re.I):
        sources[r['dataset']].setdefault(r['image_sha256'],r)
candidates=[]
for source,ims in sources.items():
    groups=collections.defaultdict(list)
    for r in ims.values():groups[r['gt_answer'].strip().casefold()].append(r)
    classes=[label for label,rs in sorted(groups.items(),key=lambda kv:(-len(kv[1]),kv[0])) if len(rs)>=15][:4]
    if len(classes)>=2:candidates.append((sum(min(64,len(groups[c])) for c in classes),source,groups,classes))
probes=[];probe_shas=set()
for _,source,groups,classes in sorted(candidates,key=lambda x:(-x[0],x[1]))[:3]:
    split={s:[] for s in ['train','dev','test']}; names=sorted(classes)
    for label in names:
        available=[r for r in order(groups[label],'probe_split42') if r['image_sha256'] not in probe_shas][:64]
        for r in available:probe_shas.add(r['image_sha256'])
        n=len(available);a=int(n*.6);b=int(n*.8)
        for s,rs in [('train',available[:a]),('dev',available[a:b]),('test',available[b:])]:
            for r in rs:split[s].append(dict(r,probe_label=names.index(label)))
    probes.append(dict(source=source,classes=names,splits=split))

selection=dict(primary=primary,matched_pairs=matched,probes=probes)
dump('selection.json',selection)
dump('protocol.json',dict(dataset='OmniMedVQA, CVPR 2024',official_archive='https://huggingface.co/datasets/foreverbeliever/OmniMedVQA',
    image_transport='mtybilly/OmniMedVQA question_type/test, all 16 shards; every question/options/answer/image_path must match official metadata exactly',
    pool_n=len(rows),pool_unique_images=len(images),official_metadata_n=len(official),
    primary_n=len(primary),primary_counts=dict(collections.Counter(r['question_type'] for r in primary)),
    checkpoint='base Qwen2-VL-2B-Instruct plus existing VQA-RAD SFT_r8 seeds42/43/44; no OmniMedVQA tuning of these models',
    visual_tokens=128,primary_metric='candidate letter accuracy; casefold greedy first answer token accuracy separately',
    image_control='same questions and options, remove image',matched_n=len(matched),
    readout='frozen mean pooled 1536-dimensional merger output; ridge classifier, lambda selected on custom dev; balanced per-class deterministic image SHA splits; label-shuffle control',
    hypothesis_scope='A successful readout shows recoverable dataset-label information in frozen features. It does not establish that the language model already recognizes clinically sufficient findings.',
    limitations=['Repacked test subset, not full official benchmark','Patient IDs unavailable; SHA split cannot guarantee patient separation','Synthetic questions based on source classification labels; some ask non-image-determinable diagnoses/stages','Readout is fitted with extra labeled examples; adaptation benefit is not an equal-data algorithm comparison'],
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print('PREPARED',len(primary),len(matched),[(p['source'],p['classes'],{s:len(rs) for s,rs in p['splits'].items()}) for p in probes],flush=True)
