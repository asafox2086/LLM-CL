"""K-COMP-style 80/10/10 question split; preserve each prepared JSONL line exactly."""
import json,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'data/MedQuAD';SOURCE=BASE/'open_ended/all.jsonl';DEST=BASE/'paper_split_seed42'
content=SOURCE.read_bytes();lines=content.splitlines(keepends=True);ids=[json.loads(line)['id'] for line in lines]
assert len(set(ids))==len(ids)
indices=list(range(len(lines)));random.Random(42).shuffle(indices)
train_n=int(.8*len(lines));dev_n=(len(lines)-train_n)//2
splits={'train':indices[:train_n],'dev':indices[train_n:train_n+dev_n],'test':indices[train_n+dev_n:]}
assert set(splits['train']).isdisjoint(splits['dev']) and set(splits['train']).isdisjoint(splits['test']) and set(splits['dev']).isdisjoint(splits['test'])
DEST.mkdir(exist_ok=True);files={}
for split,selection in splits.items():
 data=b''.join(lines[i] for i in selection);file=DEST/(split+'.jsonl')
 if file.exists():assert file.read_bytes()==data,'Refuse to change an existing locked split'
 else:file.write_bytes(data)
 files[split]={'file':str(file.relative_to(BASE)),'rows':len(selection),'sha256':hashlib.sha256(data).hexdigest(),'source_line_indices':selection}
manifest={'dataset':'MedQuAD','source_file':'open_ended/all.jsonl','source_sha256':hashlib.sha256(content).hexdigest(),'source_rows':len(lines),'split_seed':42,'ratios':[.8,.1,.1],'unit':'question-answer pair','paper_protocol':'K-COMP, NAACL 2025, Appendix A.2','paper_url':'https://aclanthology.org/2025.naacl-long.351/','reproduction_difference':'Paper documents random 80/10/10. Paper RNG seed and exact row list not verified; Python Random(42) and this row ordering are our explicit reproducibility choices, not an assertion of identical paper samples. No claim of patient/page/disease disjointness.','content_policy':'Byte-identical existing QA lines; original XML and original QA content are not rewritten.','files':files}
(DEST/'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('MEDQUAD_PAPER_STYLE_SPLIT_READY',{s:r['rows'] for s,r in files.items()},flush=True)
