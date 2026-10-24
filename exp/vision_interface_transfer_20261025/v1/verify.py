"""Unique stdlib verification and original AnswerF1 scoring after all exits."""
from pathlib import Path
import ast,collections,hashlib,json,math,os,re,subprocess,time
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=ROOT/'.deployment/paired_image_support_20261024_view'
DOMAINS=('natural','medical','geography','finance','traffic')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
 return h.hexdigest()
def write(n,v):
 with (RUN/n).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 tick=time.monotonic();assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists()
 ready=read(RUN/'READY.json');assert read(RUN/'ALL_EXIT.json')['complete'];proofs={}
 env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
 raw=subprocess.check_output(['systemctl','--user','show','llmcl-vision-interface-transfer-20261025.service','-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],env=env,text=True)
 state=dict(line.split('=',1) for line in raw.splitlines());assert state['MainPID']=='0' and state['ActiveState']=='inactive' and state['ExecMainStatus']=='0'
 for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
 for n,h in ready['runtime_files'].items():assert sha(ASSET/n)==h,n
 for case in ('fit','parent','candidate'):
  assert not (RUN/(case+'_FAILURE.json')).exists()
  start=read(RUN/(case+'_STARTED.json'));end=read(RUN/(case+'_COMPLETE.json'));exit=read(RUN/(case+'_EXIT.json'))
  assert start['identity']==end['identity'] and exit['pid']==end['identity']['pid'] and exit['returncode']==0 and not (Path('/proc')/str(exit['pid'])).exists()
  assert start['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and start['GPU']==1 and start['identity']['uid']==os.getuid()
  assert start['ready_sha256']==end['ready_sha256']==sha(RUN/'READY.json') and 0<end['seconds']<=1200
  assert end['backwards']==end['optimizer_updates']==0
  assert end['language_attention_backend']=='sdpa'
  for n,h in end.get('files',{}).items():assert sha(RUN/n)==h,n
  proofs[case]=end
 assert sum(x['seconds'] for x in proofs.values())<=3600
 fit=proofs['fit'];assert fit['teacher_vision_QA']==fit['student_vision_QA']==fit['teacher_projector_calls']==325 and fit['decoder_requests']==0
 assert fit['calibration_QA']==293 and fit['diagnostic_QA']==32 and fit['language_unchanged'] and math.isfinite(fit['ridge']) and fit['ridge']>0
 assert fit['layout']['weights']==dict(coefficients=581213056,bytes=1162426112) and fit['layout']['trainable']==0
 assert (RUN/'candidate_complete.pt').stat().st_size==fit['checkpoint']['bytes']<ready['parent']['bytes'] and sha(RUN/'candidate_complete.pt')==fit['checkpoint']['sha256']
 assert sha(RUN/'feature_records.json')==fit['feature_records_sha256'];features=read(RUN/'feature_records.json');assert len(features)==325
 assert [(x['group'],x['id']) for x in features]==[tuple(g) for g in ready['groups']]
 assert [x['mode'] for x in features]==['fit']*293+['diagnostic']*32
 assert all(x['all_crop_positions']==x['tiles']*729 and x['packed_tokens']>0 and len(x['raw_pixels_sha256'])==64 for x in features)
 for x in features[293:]:assert all(math.isfinite(x[k]) and x[k]>=0 for k in ['MSE','target_energy','ratio']) and x['target_energy']>0
 assert proofs['parent']['layout']['weights']==dict(coefficients=893675552,bytes=1787351104) and proofs['parent']['digest']==ready['parent']['digest']
 assert proofs['candidate']['layout']==fit['layout'] and proofs['candidate']['digest']==fit['digest']
 assert fit['language_attention_classes']==proofs['parent']['language_attention_classes']==proofs['candidate']['language_attention_classes']
 assert fit['preserved_nonvision_buffers_digest']==proofs['parent']['preserved_nonvision_buffers_digest']==proofs['candidate']['preserved_nonvision_buffers_digest']
 for case in ('parent','candidate'):assert proofs[case]['NLL_requests']==100 and proofs[case]['new_generations']==105
 split=read(ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json');assert sha(ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json')==ready['split_sha256']
 policy=read(ASSET/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json')['conditions']['pct01'];manifest_path=ASSET/policy['manifest'];assert sha(manifest_path)==policy['manifest_sha256']==ready['manifest_sha256']
 manifest=read(manifest_path);gold={};s1={}
 for t in manifest['tasks']:
  path=ASSET/t['files']['train']['path'];assert sha(path)==t['files']['train']['sha256']
  for line in path.read_bytes().splitlines():
   row=json.loads(line)
   if row['id'] in split['dev_ids'][str(t['task'])]:gold[row['id']]=(t,row)
   if t['task']==1 and row['id'] in split['fit_ids']['1']:s1[row['id']]=row
 assert len(gold)==100 and all(rid in s1 and rid in split['groups'][gid]['ids'] and not split['groups'][gid]['dev'] for gid,rid in ready['groups'])
 tree=ast.parse((ASSET/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
 assert len(nodes)==2;ns=dict(re=re,Counter=collections.Counter);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-score>','exec'),ns)
 scorer_ast=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
 predictions={case:read(RUN/(case+'_predictions.json')) for case in ('parent','candidate')};memory={case:read(RUN/(case+'_memory.json')) for case in predictions}
 for case,rows in predictions.items():
  assert len(rows)==100 and {x['id'] for x in rows}==set(gold)
  for x in rows:
   t,row=gold[x['id']];assert x['domain']==t['domain'] and x['task']==t['task'] and x['references']==row['answers'] and x['generation_cap']==t['max_new_tokens']
   assert 0<=len(x['output_tokens'])<=t['max_new_tokens'] and math.isfinite(x['NLL']) and x['NLL']>=0 and x['target_tokens']>0
 a={x['id']:x for x in predictions['parent']};b={x['id']:x for x in predictions['candidate']}
 for rid in a:
  for k in ['transport_sha256','raw_pixels_sha256','packed_tokens','tiles','target_tokens']:assert a[rid][k]==b[rid][k],(rid,k)
 fixed=[min((rid for rid,(t,row) in gold.items() if t['domain']==d),key=lambda rid:hashlib.sha256(rid.encode()).digest()) for d in DOMAINS]
 for case,rows in memory.items():
  assert len(rows)==5 and [x['id'] for x in rows]==fixed and [x['domain'] for x in rows]==list(DOMAINS)
  for x in rows:
   reference=a[x['id']] if case=='parent' else b[x['id']]
   assert x['new_tokens']==32 and 0<x['baseline_allocated']<=x['peak_allocated']<=x['peak_reserved']
   for k in ['transport_sha256','raw_pixels_sha256','tiles']:assert x[k]==reference[k]
 write('VERIFY_STARTED.json',dict(unix=time.time(),verifier_sha256=sha(__file__),scorer_ast_sha256=scorer_ast))
 quality={}
 for case,rows in predictions.items():
  values=[dict(id=x['id'],domain=x['domain'],F1=ns['score'](x['prediction'],x['references'])['token_f1'],NLL=x['NLL']) for x in rows]
  domain={d:dict(F1=sum(x['F1'] for x in values if x['domain']==d)/20,NLL=sum(x['NLL'] for x in values if x['domain']==d)/20,QA=20) for d in DOMAINS}
  assert all(sum(x['domain']==d for x in values)==20 for d in DOMAINS) and all(0<=x['F1']<=100 for x in values)
  quality[case]=dict(domains=domain,macro=sum(x['F1'] for x in domain.values())/5,per_id=values)
 differences={d:quality['candidate']['domains'][d]['F1']-quality['parent']['domains'][d]['F1'] for d in DOMAINS}
 memory_pass=all(b['peak_allocated']<a['peak_allocated'] for a,b in zip(memory['parent'],memory['candidate']))
 passed=all(x>=0 for x in differences.values()) and memory_pass
 out=dict(complete=True,ready_sha256=sha(RUN/'READY.json'),source_complete_sha256={case:sha(RUN/(case+'_COMPLETE.json')) for case in proofs},quality=quality,domain_F1_candidate_minus_parent=differences,all_domain_floor_passed=all(x>=0 for x in differences.values()),all5_memory_lower=memory_pass,decision='retain_for_independent_CL_review' if passed else 'close_fixed_frozen_student_affine_S1_mapping',whole_weights_parent=893675552,whole_weights_candidate=581213056,whole_weight_saving_fraction=(893675552-581213056)/893675552,checkpoint=fit['checkpoint'],memory=memory,diagnostic32=dict(mean_MSE=sum(x['MSE'] for x in features[293:])/32,mean_target_energy=sum(x['target_energy'] for x in features[293:])/32,mean_per_QA_ratio=sum(x['ratio'] for x in features[293:])/32),model_seconds={case:x['seconds'] for case,x in proofs.items()},teacher_vision_QA=325,student_vision_QA=325,NLL_requests=200,new_generations=210,new_scores=200,backwards=0,optimizer_updates=0,scorer_ast_sha256=scorer_ast,verification_seconds=time.monotonic()-tick,not_novel_framework_CL_independence_or_equivalence=True)
 write('VERIFIED.json',out);print(json.dumps(dict(decision=out['decision'],parent_macro=quality['parent']['macro'],candidate_macro=quality['candidate']['macro'],differences=differences,memory_lower=memory_pass,VERIFIED_sha256=sha(RUN/'VERIFIED.json'))))
if __name__=='__main__':main()
