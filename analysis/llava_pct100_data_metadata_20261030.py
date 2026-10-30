"""Read frozen full/1%/10% metadata and existing JSONL bytes; no resplit/model/score."""
from pathlib import Path
import json,hashlib,math
R=Path('/data2/liyapeng_grp/program/LLMCL')
REL={'original_full':'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/full/manifest.json','authorized_full':'data/FiveDomainCL/five_domain_native_cl_20261016_sparse_labeled_20261017/full/manifest.json','pct01':'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json','pct10':'data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json'}
EXPECTED={'original_full':'3dac40b318c4bc7c792eeebc4e5667b172cfdf42ae835b92f4fe56998ef44504','authorized_full':'cf737555dab3646c9916dc63a66f1d52285eebbeed4aa414e807ae5b08a0d952','pct01':'36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777','pct10':'02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
cache={}
def file_record(info,split,root):
 p=root/info['path'];st=p.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
 if key not in cache:
  h=hashlib.sha256();n=0;ids=set()
  with p.open('rb') as f:
   for line in f:
    h.update(line);n+=1
    if split=='train':
     ident=json.loads(line)['id'];assert ident not in ids;ids.add(ident)
  after=p.stat();assert key==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
  cache[key]=(h.hexdigest(),n,ids)
 actual,n,ids=cache[key];assert actual==info['sha256'] and n==info['count'],str(p)
 return dict(path=str(p),resolved_path=str(p.resolve()),sha256=actual,count=n,bytes=st.st_size),ids
manifests={};rows={};idsets={};totals={};allinput={}
for condition,rel in REL.items():
 root=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_') if condition=='pct10' else R
 p=root/rel;m=read(p);assert sha(p)==EXPECTED[condition] and len(m['tasks'])==19 and m['seed']==42
 manifests[condition]=dict(path=str(p),sha256=sha(p),source_manifest_sha256=m['source_manifest_sha256'],native_task_order=m['native_task_order']);rows[condition]=[];idsets[condition]={};alltrain=set()
 for i,t in enumerate(m['tasks'],1):
  assert t['task']==i and t['task_id']==f'T{i:02d}'
  tr,ids=file_record(t['files']['train'],'train',root);te,_=file_record(t['files']['test'],'test',root)
  assert not alltrain&ids;alltrain.update(ids);idsets[condition][i]=ids
  order=sorted(ids,key=lambda ident:hashlib.sha256(f'42:{i}:{ident}'.encode()).digest());assert hashlib.sha256(''.join(x+chr(10) for x in order).encode()).hexdigest()==t['training_order_sha256']
  rows[condition].append(dict(task=i,task_id=t['task_id'],domain=t['domain'],native_label=t['native_label'],instruction=t['instruction'],max_new_tokens=t['max_new_tokens'],train=tr,test=te,training_order_sha256=t['training_order_sha256'],one_epoch_batch16_updates=math.ceil(tr['count']/16)))
 totals[condition]=dict(train=sum(x['train']['count'] for x in rows[condition]),test=sum(x['test']['count'] for x in rows[condition]),one_epoch_batch16_updates=sum(x['one_epoch_batch16_updates'] for x in rows[condition]))
for i in range(1,20):
 reference=rows['pct01'][i-1]
 for condition in REL:
  x=rows[condition][i-1];assert all(x[k]==reference[k] for k in ['task','task_id','domain','native_label','instruction','max_new_tokens'])
  assert x['test']['sha256']==reference['test']['sha256'] and x['test']['count']==reference['test']['count']
 assert idsets['pct01'][i]<=idsets['pct10'][i]<=idsets['authorized_full'][i]<=idsets['original_full'][i]
removed=set().union(*(idsets['original_full'][i]-idsets['authorized_full'][i] for i in range(1,20)));assert removed=={'SLAKE:1622'}
assert all((idsets['original_full'][i]-idsets['authorized_full'][i])==({'SLAKE:1622'} if i==4 else set()) for i in range(1,20))
# The old exclusion file contains TRAIN only; no TEST answer is parsed here.
p=Path(rows['original_full'][3]['train']['path']);h=hashlib.sha256();excluded_count=0
for line in p.open('rb'):
 if json.loads(line)['id']=='SLAKE:1622':excluded_count+=1
 else:h.update(line)
assert excluded_count==1 and h.hexdigest()==rows['authorized_full'][3]['train']['sha256']
for i in range(19):
 if i!=3:assert rows['original_full'][i]['train']['sha256']==rows['authorized_full'][i]['train']['sha256']
assert totals['original_full']['train']==131019 and totals['authorized_full']['train']==131018 and totals['pct01']['train']==1313 and totals['pct10']['train']==13105
assert all(x['test']==21454 for x in totals.values())
auth=R/'summary/two_backbone_native_cl_20261017_v2/FULL_MISSING_LABEL_EXCLUSION.json';a=read(auth);assert a['excluded_id']=='SLAKE:1622' and a['derived_manifest_sha256']==EXPECTED['authorized_full'] and a['user_authorization']=='只有这一个有问题？那就把这个直接踢掉就可以了'
parent=R/'data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/full/manifest.json'
if parent.exists():assert sha(parent)==manifests['original_full']['source_manifest_sha256']
out=dict(schema='llava-pct100-existing-data-metadata-v1',manifests=manifests,totals=totals,tasks=rows,TEST19_per_file_actual_SHA_identical_across_conditions=True,TEST_answers_parsed=False,new_splits=0,new_exclusions=0,excluded_original_id='SLAKE:1622',excluded_ID_in_pct01=any('SLAKE:1622' in x for x in idsets['pct01'].values()),excluded_ID_in_pct10=any('SLAKE:1622' in x for x in idsets['pct10'].values()),authorized_exclusion_evidence=dict(path=str(auth),sha256=sha(auth),recorded_human_quote=a['user_authorization'],reason='Original empty gold TRAIN answer, onlyT04; exact retained raw line bytes verified.',user_quote_is_existing_historical_receipt_not_a_new_authorization=True),source_parent_manifest=dict(path=str(parent),exists_now=parent.exists(),original_recorded_sha256=manifests['original_full']['source_manifest_sha256'],current_actual_sha256=sha(parent) if parent.exists() else None),eligible_full_variant_reuses_all_other37_QA_files=True,TRAIN_ID_nesting_verified=True,current_1pct_10pct_immutable=True,models_started=0,new_generations=0,new_scores=0,CP_payloads_read=0)
print(json.dumps(out,ensure_ascii=False))
