"""One original-F1 scoring of new interventions; prior no-OCR scores stay sealed."""
from pathlib import Path
import ast,collections,hashlib,json,math,os,re,subprocess,time
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=ROOT/'.deployment/paired_image_support_20261024_view'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(n,v):
 with (RUN/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 tick=time.monotonic();assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists()
 ready=read(RUN/'READY.json');assert read(RUN/'ALL_EXIT.json')['complete']
 for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
 for n,h in ready['prior_files'].items():assert sha(RUN/n)==h,n
 assert sha(RUN/'official_ocr_projection.json')==ready['projection_sha256']
 prior_ready=read(RUN/'prior_READY.json');old=read(RUN/'prior_VERIFIED.json');assert old['complete'] and sha(RUN/'prior_VERIFIED.json')=='976a93301901c112de1987f5314e082d4a18bf0ce67b0962179612befb8c3589'
 assert sha(RUN/'prior_READY.json')==old['ready_sha256']
 for n,h in prior_ready['runtime_files'].items():assert sha(ASSET/n)==h,n
 env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
 raw=subprocess.check_output(['systemctl','--user','show','llmcl-lexical-evidence-probe-20261025.service','-p','ActiveState','-p','MainPID','-p','ExecMainStatus'],env=env,text=True)
 state=dict(x.split('=',1) for x in raw.splitlines());assert state['ActiveState']=='inactive' and state['MainPID']=='0' and state['ExecMainStatus']=='0'
 tree=ast.parse((ASSET/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['tokens','score']];assert len(nodes)==2
 scorer_ast=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest();assert scorer_ast==old['scorer_ast_sha256']
 ns=dict(re=re,Counter=collections.Counter);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-score>','exec'),ns)
 import evidence
 ocr=read(RUN/'official_ocr_projection.json');wanted={r['id'] for r in ocr};assert len(wanted)==20
 split_path=ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json';assert sha(split_path)==prior_ready['split_sha256'];split=read(split_path)
 group_by_id={rid:gid for gid,g in split['groups'].items() for rid in g['ids'] if rid in wanted};donors=evidence.pair_sources(ocr,group_by_id)
 completions={};records={};plans={};anchors={}
 for case in ['parent','candidate']:
  assert not (RUN/(case+'_FAILURE.json')).exists()
  start=read(RUN/(case+'_STARTED.json'));end=read(RUN/(case+'_COMPLETE.json'));exit=read(RUN/(case+'_EXIT.json'))
  assert start['identity']==end['identity'] and exit['pid']==end['identity']['pid'] and exit['returncode']==0 and not (Path('/proc')/str(exit['pid'])).exists()
  assert start['GPU']==1 and start['boot']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and start['identity']['uid']==os.getuid()
  assert start['ready_sha256']==end['ready_sha256']==sha(RUN/'READY.json') and 0<end['seconds']<=900
  assert end['visual_requests']==20 and end['NLL_requests']==end['new_generations']==40 and end['backwards']==end['optimizer_updates']==0
  for n,h in end['files'].items():assert sha(RUN/n)==h,n
  previous=read(RUN/'prior_'+case+'_COMPLETE.json');assert sha(RUN/'prior_'+case+'_COMPLETE.json')==old['source_complete_sha256'][case]
  assert sha(RUN/'prior_'+case+'_predictions.json')==previous['files'][case+'_predictions.json']
  assert end['digest']==previous['digest'] and end['layout']==previous['layout'] and end['preserved_nonvision_buffers_digest']==previous['preserved_nonvision_buffers_digest']
  original={x['id']:x for x in read(RUN/'prior_'+case+'_predictions.json') if x['id'] in wanted};assert len(original)==20
  values=[x for x in old['quality'][case]['per_id'] if x['id'] in wanted];assert len(values)==20 and all(x['domain']=='natural' for x in values)
  anchors[case]=values;completions[case]=end;plans[case]=read(RUN/(case+'_INTERVENTION_PLAN.json'));plan=plans[case]
  assert plan['donors']==donors and plan['groups']==group_by_id and set(plan['bodies'])==wanted
  rows=read(RUN/(case+'_predictions.json'));assert len(rows)==40 and {(x['id'],x['condition']) for x in rows}=={(rid,c) for rid in wanted for c in ['correct','wrong']}
  pairs={(x['id'],x['condition']):x for x in rows};records[case]=pairs
  for rid in wanted:
   a=pairs[rid,'correct'];b=pairs[rid,'wrong'];p=original[rid];body=plan['bodies'][rid];donor=donors[rid];donor_body=plan['bodies'][donor] if donor else []
   for k in ['original_prepared_sha256','raw_pixels_sha256','image_features_sha256','nontext_sha256','original_input_ids','body_start','body_stop','packed_tokens','target_tokens','references','generation_cap','donor']:assert a[k]==b[k],(rid,k)
   assert a['donor']==donor and a['original_prepared_sha256']==p['prepared_sha256'] and a['raw_pixels_sha256']==p['raw_pixels_sha256'] and a['packed_tokens']==p['packed_tokens'] and a['target_tokens']==p['target_tokens']
   assert a['references']==p['references'] and a['generation_cap']==p['generation_cap']
   begin,stop=a['body_start'],a['body_stop'];assert 0<=begin<=stop<=len(a['input_ids']) and stop-begin==len(body)==plan['body_token_counts'][rid]
   assert len(a['input_ids'])==len(b['input_ids']) and a['input_ids'][:begin]==b['input_ids'][:begin] and a['input_ids'][stop:]==b['input_ids'][stop:]
   header=ready['CPU_GATE']['header_ids'];tail=ready['CPU_GATE']['tail_ids'];at=begin-len(header)
   assert at>=0 and a['input_ids'][:begin]==a['original_input_ids'][:at]+header
   assert a['input_ids'][stop:]==tail+a['original_input_ids'][at:]
   assert a['input_ids'][begin:stop]==body
   assert b['input_ids'][begin:stop]==([donor_body[i%len(donor_body)] for i in range(len(body))] if body else [])
   if not body:
    assert a['input_ids']==b['input_ids'] and a['output_tokens']==b['output_tokens'] and abs(a['NLL']-b['NLL'])<1e-6
   for x in [a,b]:assert math.isfinite(x['NLL']) and x['NLL']>=0 and x['target_tokens']>0 and len(x['output_tokens'])<=x['generation_cap']
 assert plans['parent']==plans['candidate'] and sum(x['seconds'] for x in completions.values())<=1800
 for rid in wanted:
  a=records['parent'][rid,'correct'];b=records['candidate'][rid,'correct']
  for k in ['input_ids','original_input_ids','body_start','body_stop','packed_tokens','target_tokens','references','generation_cap','raw_pixels_sha256']:assert a[k]==b[k],(rid,k)
 write('VERIFY_STARTED.json',dict(unix=time.time(),verifier_sha256=sha(__file__),scorer_ast_sha256=scorer_ast))
 quality={}
 for case in ['parent','candidate']:
  arms={'noOCR':anchors[case]}
  for condition in ['correct','wrong']:
   arms[condition]=[dict(id=rid,F1=ns['score'](records[case][rid,condition]['prediction'],records[case][rid,condition]['references'])['token_f1'],NLL=records[case][rid,condition]['NLL']) for rid in sorted(wanted)]
  quality[case]={arm:dict(F1=sum(x['F1'] for x in values)/20,NLL=sum(x['NLL'] for x in values)/20,per_id=values) for arm,values in arms.items()}
 empty_ids=sorted(rid for rid,b in plans['parent']['bodies'].items() if not b);assert len(empty_ids)==2
 empty_control={case:{arm:[x for x in values['per_id'] if x['id'] in empty_ids] for arm,values in arms.items()} for case,arms in quality.items()}
 changes={}
 for case,arms in quality.items():
  keyed={arm:{x['id']:x['F1'] for x in values['per_id']} for arm,values in arms.items()}
  changes[case]={reference:dict(higher=sum(keyed['correct'][rid]>keyed[reference][rid] for rid in wanted),lower=sum(keyed['correct'][rid]<keyed[reference][rid] for rid in wanted),equal=sum(keyed['correct'][rid]==keyed[reference][rid] for rid in wanted)) for reference in ['noOCR','wrong']}
 gap0=quality['parent']['noOCR']['F1']-quality['candidate']['noOCR']['F1'];gap1=quality['parent']['correct']['F1']-quality['candidate']['correct']['F1']
 gain={case:quality[case]['correct']['F1']-quality[case]['noOCR']['F1'] for case in quality};control={case:quality[case]['correct']['F1']-quality[case]['wrong']['F1'] for case in quality}
 passed=gain['candidate']>0 and control['candidate']>0 and gap1<gap0
 out=dict(complete=True,ready_sha256=sha(RUN/'READY.json'),prior_VERIFIED_sha256=sha(RUN/'prior_VERIFIED.json'),source_complete_sha256={c:sha(RUN/(c+'_COMPLETE.json')) for c in completions},quality=quality,correct_minus_noOCR=gain,correct_minus_wrong=control,correct_changes=changes,empty_OCR_control=empty_control,parent_minus_candidate_gap_noOCR=gap0,parent_minus_candidate_gap_correct=gap1,decision='consider_independent_real_lightweight_reading_feasibility' if passed else 'close_fixed_precomputed_OCR_evidence_channel',diagnostic_gate_passed=passed,only_partial_functional_evidence=True,no_deployment_OCR_cost_proof=True,old_candidate_quality_failure_unchanged=True,model_seconds={c:x['seconds'] for c,x in completions.items()},visual_requests=40,NLL_requests=80,new_generations=80,new_scores=80,backwards=0,optimizer_updates=0,scorer_ast_sha256=scorer_ast,verification_seconds=time.monotonic()-tick)
 write('VERIFIED.json',out);print(json.dumps(dict(decision=out['decision'],F1={c:{a:v['F1'] for a,v in q.items()} for c,q in quality.items()},gain=gain,correct_minus_wrong=control,gap0=gap0,gap1=gap1,VERIFIED_sha256=sha(RUN/'VERIFIED.json'))))
if __name__=='__main__':main()
