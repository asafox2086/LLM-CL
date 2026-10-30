"""Read only original completed PaRSP logs/stage metadata. No QA/score/model/CP."""
from pathlib import Path
import json,re,hashlib,math
R=Path('/data2/liyapeng_grp/program/LLMCL');T=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_');F=T/'exp/two_backbone_pct10_native_checkpoint_20261027';B=F/'runs/onevision/pct10/PaRSP'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
P01=R/'exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/PaRSP/training_20261018_040230';P10=B/'recovery_quota_fourth_S18_20261030'
paths={'original':B/'training_20261027_113623','io':B/'recovery_io_20261029_S06','persistent':B/'recovery_persistent_CP_S13_20261029','third':B/'recovery_quota_third_S13_20261029','fourth':P10}
chains=[('io','IO_RECOVERY_SOURCE.json','original',6),('persistent','IO_RECOVERY_SOURCE.json','io',13),('third','QUOTA_RECOVERY_SOURCE.json','persistent',13),('fourth','QUOTA_RECOVERY_SOURCE.json','third',18)]
lineage=[]
for node,name,parent,stage in chains:
 p=paths[node]/name;q=read(p);assert q['original_run']==str(paths[parent]) and q['checkpoint']['stage']==stage
 lineage.append(dict(path=str(p),sha256=sha(p),parent_run=q['original_run'],restored_stage=stage))
configs={};stage_metadata={};chosen={}
for condition,final in [('pct01',P01),('pct10',P10)]:
 cfg=read(final/'config.json');assert cfg['method']=='PaRSP' and cfg['backbone']=='onevision' and cfg['condition']==condition
 configs[condition]=dict(path=str(final/'config.json'),sha256=sha(final/'config.json'),manifest_sha256=cfg['manifest_sha256'])
 selected=[];chosen[condition]={}
 for stage in range(1,20):
  U=P01 if condition=='pct01' else paths['original' if stage<=6 else 'io' if stage<=13 else 'third' if stage<=18 else 'fourth']
  chosen[condition][stage]=U
  assert read(U/'config.json')['execution_source_sha256']==cfg['execution_source_sha256']
  current=final/'stages'/f'stage_{stage:02d}.json';original=U/'stages'/f'stage_{stage:02d}.json';assert sha(current)==sha(original)
  complete=read(current);assert complete['complete'] and complete['weights_saved'] and complete['stage']==stage
  p=U/'auxiliary'/f'parsp_stage_{stage:02d}.json';e=read(p)
  assert e['stage']==stage and not e['preflight_only'] and e['layer_ratio']==.38 and e['neuron_ratio']==.4 and e['nullspace_min_ratio']==.2
  assert len(e['attention_layers'])==len(e['ffn_layers'])==9 and e['trainable_parameters']==134195328
  assert len(e['ffn_neurons'])==17514 and len(e['attention_neurons'])==3222
  assert e['protected_history_covariance']==(stage>1)
  selected.append(dict(stage=stage,source_run=str(U),metadata_path=str(p),metadata_sha256=sha(p),canonical_complete_stage_path=str(current),canonical_complete_stage_sha256=sha(current),attention_layers=e['attention_layers'],ffn_layers=e['ffn_layers'],ffn_neurons_count=len(e['ffn_neurons']),attention_neurons_count=len(e['attention_neurons']),probe_count=e['probe_count'],trainable_parameters=e['trainable_parameters'],layer_ratio=e['layer_ratio'],neuron_ratio=e['neuron_ratio'],nullspace_min_ratio=e['nullspace_min_ratio'],protected_history_covariance=e['protected_history_covariance']))
 stage_metadata[condition]=selected
pattern=re.compile(r'\[Info\]\s+(\d+:(?:qkv|o|gateup|down)):\s+Null Dim\s+(\d+)/(\d+)\s+\(([^)]+)%\)\s+\|\s+\[(Force|Auto)\]\s+([0-9.]+)%\s+([<>]=?)\s+([0-9.]+)%')
log_specs=[('pct01',P01/'supervisor.log',1,19),('pct10',paths['original']/'supervisor.log',1,6),('pct10',paths['io']/'supervisor.log',7,13),('pct10',T/'summary/runtime_recovery_20261029/THIRD_BASELINE_RECOVERY/PaRSP.log',14,18),('pct10',T/'summary/runtime_recovery_20261030/FOURTH_BASELINE_RECOVERY/PaRSP.log',19,19)]
logs=[];observed={'pct01':{},'pct10':{}}
for condition,p,lo,hi in log_specs:
 pending=[];included=excluded=0;discarded_stages={}
 for lineno,line in enumerate(p.open(),1):
  m=pattern.search(line)
  if m:
   group,count,total,printedratio,branch,natural,comparison,floor=m.groups();pending.append(dict(log_path=str(p),line_number=lineno,group=group,null_vector_count=int(count),input_width=int(total),printed_final_percent=printedratio,branch=branch,printed_natural_null_percent=natural,comparison=comparison,printed_floor_percent=floor,original_line=line.strip()))
  if 'parsp_localization_ready' not in line:continue
  index=line.find('{')
  if index<0:continue
  try:e=json.loads(line[index:])
  except json.JSONDecodeError:continue
  if e.get('event')!='parsp_localization_ready':continue
  stage=e['stage']
  if lo<=stage<=hi:
   assert stage not in observed[condition],(condition,stage,p)
   expected=stage_metadata[condition][stage-1];groups={f'{i}:{kind}' for i in expected['attention_layers'] for kind in ['qkv','o']}|{f'{i}:{kind}' for i in expected['ffn_layers'] for kind in ['gateup','down']}
   assert len(pending)==(36 if stage>1 else 0),(condition,stage,len(pending),str(p))
   if stage>1:assert {x['group'] for x in pending}==groups
   for x in pending:
    assert x['null_vector_count']==int(x['input_width']*.2) and float(x['printed_floor_percent'])==20.
   observed[condition][stage]=pending;included+=len(pending)
  else:excluded+=len(pending);discarded_stages[stage]=discarded_stages.get(stage,0)+len(pending)
  pending=[]
 logs.append(dict(condition=condition,path=str(p),sha256=sha(p),canonical_stage_range=[lo,hi],included_projector_records=included,excluded_uncommitted_or_later_branch_records=excluded,excluded_stage_counts=discarded_stages,unassigned_tail_projector_records=len(pending)))
for condition in observed:
 assert set(observed[condition])==set(range(1,20))
 assert sum(len(x) for x in observed[condition].values())==648
 assert all(x['branch']=='Force' for rows in observed[condition].values() for x in rows)
source_specs=[(F/'parsp_adapter.py','d1ebe1831e6d089d434bdde34b59db64eff17dc74d8d77f9ca4f080a51da8366'),(F/'osft_adapter.py','60b05f3b59f3878e65ee9543657b54a5aa5da13c14e139123e1c3591f3007b34'),(F/'host_adam.py','5e733e6b87c12eb9331ff23e8919a142f1861a2beec82ec8b0361d878b9aec8c'),(R/'code/PaRSP/source/parsp/methods/masking.py','bcf0f73c7093436ea986e500805a089d5436b94cc192336ea276d059ec8ea179'),(R/'code/PaRSP/source/parsp/methods/projection.py','e722a4d4192cfeb19f106d86ddeb2688f103dd3266ddfa22c9793d27640e6cfb'),(R/'code/OSFT/source/src/mini_trainer/osft_utils.py','ff67271a66cc6016854f3e1d5736f2931efa5f8feafb5eefbfa70a52105910b3')]
for p,h in source_specs:assert sha(p)==h
summary={condition:dict(canonical_stages=19,protected_stages=18,projector_records=648,forced_records=648,automatic_records=0,stage_metadata=stage_metadata[condition],projector_rows_by_stage={str(k):v for k,v in observed[condition].items()}) for condition in observed}
print(json.dumps(dict(schema='parsp-canonical-protection-process-evidence-v1',conditions=summary,source_logs=logs,config_evidence=configs,checkpoint_recovery_lineage_metadata=lineage,sources=[dict(path=str(p),sha256=h) for p,h in source_specs],definitions=dict(Force='Threshold detected fewer vectors than floor(.2*input_width); author code instead selects that many smallest-eigenvalue vectors.',Auto='Threshold-selected vectors reach floor; never observed in these canonical records.',canonical_pct10='S01..06 original, S07..13 IO branch, S14..18 third branch, S19 fourth branch; exclude abandoned next-stage localizers using source receipts and exact canonical stage-complete hashes.',rounded_ratio='Original printed natural-null percentage has two decimals; 0.00% is not an exact full-rank proof.'),limitations=['Applies only this LLaVA PaRSP 1% and10% completed trajectory, not other algorithms/backbones/seeds.','Forced floor does not prove harmful gradient removal, current-task nonconvergence or impaired acquisition.','Original logs contain no pre-mask/pre-projector dense gradient, actual Adam deltaW, constraint leakage or paired before/after current CE.','Selection counts are registered trainability, not effective permitted dense coefficients after masking/projection.','The forced subspace can contain small nonzero covariance eigenvalues: it is not necessarily a strict mathematical nullspace.'],model_executions=0,checkpoint_payloads_read=0,QA_text_read=0,new_scores=0)))
