"""Fixed stratified MMLU test probes, read-only w.r.t. training data."""
import hashlib
import json
import os
from pathlib import Path
import requests
import pyarrow.parquet as pq
from common import ROOT,write_json

OUT=ROOT/'data/knowledge_probe';OUT.mkdir(parents=True,exist_ok=True)
s=requests.Session();s.trust_env=False
revision='c30699e8356da336a370243923dbaf21066bb9fe'  # immutable evaluation protocol
tree=s.get(f'https://hf-mirror.com/api/datasets/cais/mmlu/tree/{revision}/all',timeout=30);tree.raise_for_status()
entry=next(x for x in tree.json() if x['path'].startswith('all/test-'))
path=OUT/'mmlu_test.parquet'
if not path.exists():
 response=s.get(f"https://hf-mirror.com/datasets/cais/mmlu/resolve/{revision}/{entry['path']}",timeout=60)
 response.raise_for_status();path.write_bytes(response.content)
sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==entry['lfs']['oid']
rows=pq.read_table(path).to_pylist()
medical={'anatomy','clinical_knowledge','college_biology','college_medicine','medical_genetics','professional_medicine'}
subjects=sorted({row['subject'] for row in rows});assert len(subjects)==57
selected=[]
for subject in subjects:
 candidates=[(idx,row) for idx,row in enumerate(rows) if row['subject']==subject]
 candidates.sort(key=lambda pair:hashlib.sha256(('llmcl-mmlu-probe42\n'+str(pair[0])).encode()).hexdigest())
 count=16 if subject in medical else 2
 for index,row in candidates[:count]:
  selected.append({'instance_id':f'mmlu_test_{index}','subject':subject,'group':'medical' if subject in medical else 'general',
    'question':row['question'],'choices':row['choices'],'answer':int(row['answer']),'source_row':index})
assert len(selected)==198
content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected)
(OUT/'probes.jsonl').write_text(content)
manifest={'source':'https://huggingface.co/datasets/cais/mmlu','author_source':'https://github.com/hendrycks/test',
 'transport':'HF mirror; pinned commit and verified LFS SHA256','revision':revision,'source_path':entry['path'],
 'source_sha256':sha,'selection_sha256':hashlib.sha256(content.encode()).hexdigest(),'selection_seed':'llmcl-mmlu-probe42',
 'selection':'sha256 ordering of original test row index within each subject; 2 general / 16 medical',
 'subjects':subjects,'medical_subjects':sorted(medical),'general_count':102,'medical_count':96,
 'scope':'198-question zero-shot fixed pilot probe, not full MMLU; evaluate only, no training/dev selection'}
write_json(OUT/'manifest.json',manifest)
write_json(ROOT/'summary_cv/knowledge_probe_protocol.json',manifest)
print(json.dumps(manifest,ensure_ascii=False),flush=True)
