from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,subprocess,sys,time,importlib.util
r=Path('/data1/liyapeng/program/LLMCL');read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=r/'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/completed/ebf69e0412424b46f33e.json';proof=read(p)
assert sha(p)=='b4de7085d6c0b0f5cb73a8720928b1ceee63c41eb586a8ae0c40d9f0f3214a72'
collector=r/'analysis/v2_verified_baseline_report.py';assert sha(collector)=='d7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a'
spec=importlib.util.spec_from_file_location('metadata_candidate_only',collector);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.candidate(proof['candidate']['run'],'new3090')==proof['candidate']
run=r/proof['candidate']['run'];latest_path=run/'checkpoints/latest.json';latest=read(latest_path);cp=run/latest['path']
assert latest['stage']==19 and latest['sha256']==proof['checkpoint_sha256'] and cp.stat().st_size==latest['bytes']==4899367968
assert sha(latest_path)==proof['source_sha256'][str(latest_path.relative_to(r))]
reader=r/'analysis/checkpoint_weight_storage_20261024.py';assert sha(reader)=='62bb56da7130270210f752e684d381bdadac0c667bf5b514085f550bad31a3dd'
out=r/'.deployment/qwen_cl_weight_storage_20261028';out.mkdir(exist_ok=False)
layout_path=out/'SAVED_GRAPH_METADATA.json'
with (out/'READER_STDOUT.txt').open('x') as stdout,(out/'READER_STDERR.txt').open('x') as stderr:
 result=subprocess.run([sys.executable,str(reader),'--checkpoint',str(cp),'--method','CL_MoE','--output',str(layout_path)],stdout=stdout,stderr=stderr,timeout=120)
assert result.returncode==0,(out/'READER_STDERR.txt').read_text()[-2500:]
v=read(layout_path);storage=v['storage_layout']
assert v['config_method']=='CL_MoE' and v['config_backbone']=='qwen' and v['config_condition']=='pct01'
assert v['manifest_sha256']=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777' and v['checkpoint_stat_bytes']==latest['bytes']
assert v['weights_not_loaded'] and v['pickle_globals_not_executed'] and v['raw_weight_bytes_not_read']
assert v['reader_sha256']==sha(reader)
derived=[x for x in v['registered_buffers'] if x['name'].rsplit('.',1)[-1] in ('inv_freq','position_ids')]
required=[x for x in v['registered_buffers'] if x not in derived]
assert all(x['name'].rsplit('.',1)[-1] in ('U_high','S_high','V_high','_expert_mask') for x in required)
excluded={x['storage_key'] for x in derived};assert not excluded.intersection(x['storage_key'] for x in v['model_registered_layout'] if x['kind']=='_parameters')
selected=set(v['selected_storage_keys_before_buffer_classification'])-excluded
groups=defaultdict(lambda:dict(storages=0,bytes=0))
for x in v['model_unclassified_storage']:
 label=x['first_path'].rsplit('.state.',1)[-1].split('.')[0];assert label in {'original_inv_freq','_llmcl_visual_cache','total'},label
 groups[label]['storages']+=1;groups[label]['bytes']+=x['storage_bytes']
dtype=Counter()
for key in selected:dtype[storage[key]['dtype']]+=storage[key]['bytes']
base_path=r/'summary/edge_deployment_20261019/component_footprint/v1/COMPLETE.json';base=read(base_path)['models']['qwen']['total_tensor_bytes'];assert base==4417971200
total=sum(storage[k]['bytes'] for k in selected)
unique={}
for x in v['model_registered_layout']:
 if x['kind']=='_parameters' and x['requires_grad']:
  unique[(x['storage_key'],x['storage_offset'],tuple(x['shape']),tuple(x['stride']),x['dtype'])]=x
e=dict(unix=time.time(),backbone='qwen',method='CL_MoE',condition='pct01',proof_file=str(p.relative_to(r)),proof_sha256=sha(p),checkpoint=str(cp),checkpoint_bytes=cp.stat().st_size,checkpoint_sha256_from_original_proof=proof['checkpoint_sha256'],checkpoint_metadata_sha256=sha(latest_path),candidate_signature_unchanged=True,reader_sha256=sha(reader),private_layout_record=str(layout_path),private_layout_sha256=sha(layout_path),metadata_pickle_sha256=v['pickle_sha256'],final_saved_storage_coefficients=sum(storage[k]['elements'] for k in selected),final_required_saved_weight_bytes=total,final_required_saved_weight_MiB=total/1048576,original_base_weight_bytes=base,incremental_weight_bytes=total-base,incremental_weight_MiB=(total-base)/1048576,final_saved_dtype_bytes=dict(dtype),main_optimizer_coordinates=sum(x['logical_elements'] for x in unique.values()),main_optimizer_parameter_tensors=len(unique),unregistered_essential_state_bytes=sum(storage[k]['bytes'] for k in {x['storage_key'] for x in v['essential_unregistered']}),excluded_derived_registered_buffer_bytes=sum(storage[k]['bytes'] for k in excluded),excluded_derived_registered_buffer_names=[x['name'] for x in derived],model_unregistered_exclusion_groups=dict(groups),unclassified_storage_count=0,weights_not_loaded=True,pickle_globals_not_executed=True,raw_weight_bytes_not_read=True,checkpoint_payload_not_rehashed=True,new_models=0,new_generations=0,new_scores=0,representation='Original saved dtype and unmerged graph, required CL anchors/task weights included; storage aliases deduplicated; CP container separate')
assert m.candidate(proof['candidate']['run'],'new3090')==proof['candidate']
with (out/'EVIDENCE.json').open('x') as f:json.dump(e,f,indent=2);f.write('\n')
print(json.dumps(e))
