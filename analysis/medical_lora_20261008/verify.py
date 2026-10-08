"""Verify audit provenance, intervention pairs, scores, captions and raw images."""
import csv,hashlib,json,math,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
from common import Scorer
scorer=Scorer();checks={};source=json.loads((OUT/'provenance.json').read_text());RUN=Path(source['source_run'])
for rel,sha in source['raw_prediction_files_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha
assert source['prediction_records']==72000;checks['original_predictions_unchanged']=72000
for method in ['seq_lora','migu_lora','olora','sapt_lora']:
 config=json.loads((RUN/method/'config.json').read_text())
 for rel,sha in config['implementation'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==sha,rel
checks['original_training_implementation_unchanged']=True
checkpoint_hashes={}
for method in ['seq_lora','migu_lora','olora','sapt_lora']:
 for stage in range(10):
  p=RUN/method/'checkpoints'/f'stage_{stage:02d}'/'completed.pt'
  checkpoint_hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'checkpoint_fingerprints.json').write_text(json.dumps(checkpoint_hashes,indent=2)+'\n')
checks['source_checkpoints_fingerprinted']=len(checkpoint_hashes)

for protocol,script in [('provenance.json','audit.py'),('routing_protocol.json','routing.py'),('gradient_protocol.json','gradients.py'),('mask_protocol.json','masks.py'),('intervention_protocol.json','interventions.py'),('vision_ablation_protocol.json','vision_ablation.py')]:
 obj=json.loads((OUT/protocol).read_text());assert obj['code_sha256']==hashlib.sha256((OUT/script).read_bytes()).hexdigest(),script
checks['diagnostic_code_fingerprints']=6
ids=json.loads((OUT/'vision_ablation_protocol.json').read_text())['instance_ids'];summary=list(csv.DictReader((OUT/'vision_ablation_scores.csv').open()))
for stage in [0,9]:
 for mode in ['actual128','wrong128','no_image','actual512']:
  records=[json.loads(line) for line in (OUT/f'vision_stage{stage}_{mode}.jsonl').read_text().splitlines()];assert [r['instance_id'] for r in records]==ids
  for r in records:
   assert r['stage']==stage and r['mode']==mode
   if mode=='wrong128':assert r['used_image']!=r['image']
   expected=scorer.score(r['prediction'],r['references']);assert all(math.isclose(v,r['scores'][k],abs_tol=1e-8) for k,v in expected.items())
  s=next(r for r in summary if r['stage']==str(stage) and r['mode']==mode and r['group']=='all')
  assert math.isclose(float(s['EM']),sum(r['scores']['exact_match'] for r in records)/60,abs_tol=1e-8)
checks['paired_visual_predictions_recomputed']=480
image_manifest=json.loads((ROOT/'CV_data/medical/vqa_rad/image_manifest.json').read_text())
checked_images=set()
for stage in [0,9]:
 for mode in ['actual128','wrong128','no_image','actual512']:
  for r in [json.loads(line) for line in (OUT/f'vision_stage{stage}_{mode}.jsonl').read_text().splitlines()]:
   for image in [r['image'],r['used_image']]:
    if image is not None and image not in checked_images:
     assert hashlib.sha256((ROOT/'CV_data'/image).read_bytes()).hexdigest()==image_manifest[image]['sha256']
     checked_images.add(image)
checks['visual_probe_image_files_verified']=len(checked_images)

assert len(list(csv.DictReader((OUT/'gradient_CE.csv').open())))==45
assert len(list(csv.DictReader((OUT/'gradient_conflicts.csv').open())))==405
checks['gradient_diagnostic_groups']=45
cases=json.loads((OUT/'cases.json').read_text())
for c in cases:
 if c.get('image'):
  assert hashlib.sha256((OUT/c['image']).read_bytes()).hexdigest()==c['image_sha256']
checks['original_image_cases_verified']=6
image_references=0
for p in OUT.glob('*.md'):
 text=p.read_text();assert text.count('$$')%2==0,p
 matches=list(re.finditer(r'!\[[^\]]*\]\(([^)]+)\)',text))
 for i,m in enumerate(matches):
  image=(p.parent/m.group(1)).resolve();assert image.is_file(),image
  assert subprocess.run(['git','check-ignore','--quiet',str(image)],cwd=ROOT).returncode==1,image
  end=matches[i+1].start() if i+1<len(matches) else len(text)
  assert re.search(r'\*\*图 \d+｜',text[m.end():end]),(p,m.group(1))
  image_references+=1
checks['captioned_publishable_image_references']=image_references
manifest=json.loads((OUT/'figures/manifest.json').read_text());assert len(manifest)==5
for m in manifest.values():
 assert hashlib.sha256((OUT/'figures'/m['png']).read_bytes()).hexdigest()==m['png_sha256'];assert (OUT/'figures'/m['pdf']).is_file()
checks['scientific_png_pdf_pairs_verified']=5
(OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
