"""Read native release metadata and export traceable README examples; no split mutation."""
import collections
import importlib.util
import json
import math
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/TenTaskCL'/HERE.name
SAMPLES=ROOT/'sample'/HERE.name
spec=importlib.util.spec_from_file_location('native_data', HERE/'prepare_data.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
OUT.mkdir(parents=True,exist_ok=True);SAMPLES.mkdir(parents=True,exist_ok=True)
records=[]

def add(name,train,test,sample,rule,extra=None):
    rec={'dataset':name,'full_train':train,'full_test':test,'pct01_train':math.ceil(.01*train),
         'split_rule':rule,'example':sample,'notes':extra or {}}
    images=sample.get('images',[])
    rec['sample_image']=None
    if images and (ROOT/images[0]).is_file():
        src=ROOT/images[0]
        with Image.open(src) as im:ext={ 'JPEG':'.jpg','PNG':'.png','WEBP':'.webp'}.get(im.format,'.img')
        dst=SAMPLES/(name.replace('/','_')+ext);shutil.copyfile(src,dst)
        rec['sample_image']=str(dst.relative_to(ROOT))
    records.append(rec)
    print(name,train,test,rec['pct01_train'],flush=True)

def count_pq(ds,split):
    return sum(pq.ParquetFile(p).metadata.num_rows for p in (ROOT/'data'/ds/'raw').rglob(split+'-*.parquet'))

squad={r:list(n.native_squad(r)) for r in ['train','test']}
add('SQuAD11',len(squad['train']),len(squad['test']),squad['train'][0],'official train / dev')
del squad
for ds in ['TextVQA','VQA_RAD']:
    add(ds,count_pq(ds,'train'),count_pq(ds,'test' if ds=='VQA_RAD' else 'validation'),
        next(n.native_parquet(ds,'train')),'official train / '+('test' if ds=='VQA_RAD' else 'validation'))
med=n.native_medquad()
add('MedQuAD',len(med['train']),len(med['test']),med['train'][0],'research document-group 80/20',
    json.loads((OUT/'medquad_unavailable_labels.json').read_text()))
slake={r:json.loads((ROOT/'data/SLAKE/raw'/f'{r}.json').read_text()) for r in ['train','test','validation']}
add('SLAKE',len(slake['train']),len(slake['test']),next(n.native_slake('train')),'official bilingual train / test',
    {'unused_validation':len(slake['validation'])})
geo=n.native_geoquestions()
add('GeoQuestions1089',len(geo['train']),len(geo['test']),geo['train'][0],'research original-query-group 80/20',
    json.loads((OUT/'geoquestions_unavailable_labels.json').read_text()))
rs={role:list(n.native_rsvqa(role)) for role in ['train','test']}
val=json.loads((ROOT/'data/RSVQA_LR/raw/LR_split_val_questions.json').read_text())['questions']
add('RSVQA-LR',len(rs['train']),len(rs['test']),rs['train'][0],'official LR active-mask train / test',
    {'unused_validation':sum(x.get('active',True) for x in val),'original_total_qa':77232,
     'native_type_counts':dict(collections.Counter(x['metadata']['question_type'] for role in rs.values() for x in role)),
     'source':'https://zenodo.org/records/6344334','images_complete':(ROOT/'data/RSVQA_LR/full_assets_ready.json').exists()})
del rs
fin={r:list(n.native_finqa(r)) for r in ['train','test']}
add('FinQA',len(fin['train']),len(fin['test']),fin['train'][0],'official train / test')
mme=n.native_mme()
add('MME-Finance',len(mme['train']),len(mme['test']),mme['train'][0],'research chart-group 80/20',{'configuration':'English MMfin.tsv, all original types'})
aq={r:json.loads((ROOT/'data/AQUA/raw'/f'{r}.json').read_text()) for r in ['train','test','val']}
ac={r:dict(collections.Counter('knowledge' if x['need_external_knowledge'] else 'visual' for x in rows)) for r,rows in aq.items()}
aqsample=next((x for x in n.native_aqua('train',False) if (ROOT/x['images'][0]).is_file()),next(n.native_aqua('train',False)))
add('AQUA',len(aq['train']),len(aq['test']),aqsample,'official train / test; native modality flag',ac)
records[-1]['pct01_train']=sum(math.ceil(.01*ac['train'][k]) for k in ['knowledge','visual'])
records[-1]['text_example']=next(n.native_aqua('train',True))
payload={'audited_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'count_unit':'QA pairs, not images or conversations',
         'status':'metadata audit; not a complete-assets or frozen-split declaration','records':records}
(OUT/'readme_dataset_audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print('AUDIT',OUT/'readme_dataset_audit.json',flush=True)
