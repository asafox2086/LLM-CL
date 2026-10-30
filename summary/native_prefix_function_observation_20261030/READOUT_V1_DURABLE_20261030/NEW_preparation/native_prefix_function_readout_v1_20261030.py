"""DRAFT terminal-only stdlib readout; no model, new generation, quality scoring or TEST."""
from pathlib import Path
import argparse,hashlib,json,math,os,subprocess,sys
P=Path('/data2/liyapeng_grp/program/LLMCL');S=P/'summary/native_prefix_function_observation_20261030/path_compat_v1';O=S/'readout_v1'
BSHA='af604698c1fadacdfdd0f7aa2a3836bdf2614acf9394b56b997c32aef7cbe45f';RSHA='11ee3cff9c09534d67f35a09662b1131426e55d570cc92ab5ca3b42012b8673c'
METHODS=('PASs_MoE','CL_MoE');UNIT='llmcl-native-prefix-function-path-compat-v1-20261030.service'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,j):
 with p.open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
def who(pid):
 try:
  p=Path('/proc')/str(pid);x=(p/'stat').read_text().rsplit(') ',1)[1].split();return dict(pid=pid,startticks=x[19],uid=p.stat().st_uid,pgid=int(x[2]))
 except FileNotFoundError:return None
def finite(x):return all(finite(y) for y in x) if isinstance(x,(list,tuple)) else isinstance(x,(int,float)) and math.isfinite(x)
def ratio(x,y):return x/y if y>0 else None
def close(x,y,s=0.):return abs(x-y)<=max(s,8*sys.float_info.epsilon*max(abs(x),abs(y)))
def distribution(values):
 v=sorted(x for x in values if x is not None);n=len(v)
 if not n:return dict(n=0,minimum=None,median=None,p90=None,maximum=None)
 def q(a):
  z=a*(n-1);i=int(z);j=min(i+1,n-1);return v[i]+(v[j]-v[i])*(z-i)
 return dict(n=n,minimum=v[0],median=q(.5),p90=q(.9),maximum=v[-1])
def terminal():
 # Strictly no QA/phase/position streams until all original terminal records close.
 if (S/'CONTROL_FAILURE.json').exists():return dict(status='FAILED_ORIGINAL_CONTROL')
 names=['CONTROL_STARTED.json','CONTROL_COMPLETE.json','SLOT.json']+[m+n for m in METHODS for n in ('_STARTED.json','_EXIT.json')]
 if not all((S/n).exists() for n in names):return dict(status='WAITING')
 start=read(S/'CONTROL_STARTED.json');c=read(S/'CONTROL_COMPLETE.json');slot=read(S/'SLOT.json')
 if not c.get('released') or not slot.get('released') or slot.get('status')!='complete' or who(start['identity']['pid'])==start['identity']:return dict(status='WAITING')
 assert slot['CONTROL_COMPLETE_sha256']==sha(S/'CONTROL_COMPLETE.json')
 u=subprocess.run(['systemctl','--user','show',UNIT,'--property=MainPID,ExecMainPID,ExecMainStatus,Result'],capture_output=True,text=True,check=True);u=dict(x.split('=',1) for x in u.stdout.splitlines() if '=' in x)
 assert u['MainPID']=='0' and u['ExecMainStatus']=='0' and u['Result']=='success' and int(u['ExecMainPID'])==start['identity']['pid']
 assert sha(S/'BINDING.json')==BSHA;b=read(S/'BINDING.json');assert b['admitted'] and start['binding_sha256']==BSHA
 a=read(b['admission_path']);assert a['approved'] and a['binding_sha256']==BSHA and sha(S/'controller.py')==a['controller_sha256']
 for n,h in b['scientific_sources'].items():assert sha(Path(b['observation_source_dir'])/n)==h
 assert sha(b['protocol_path'])==b['protocol_sha256'] and sha(b['old_tf_c_path'])==RSHA
 assert len(c['finished'])==2 and {v['method'] for v in c['finished']}==set(METHODS);finished={v['method']:v for v in c['finished']}
 files=[S/n for n in names]+[S/'BINDING.json',Path(b['admission_path']),Path(b['old_tf_c_path']),Path(b['protocol_path'])];roots={}
 for m in METHODS:
  st=read(S/(m+'_STARTED.json'));ex=read(S/(m+'_EXIT.json'));assert ex['identity']==st['identity'] and st['binding_sha256']==BSHA and ex['returncode']==0 and ex['identity_gone'] and who(ex['identity']['pid'])!=ex['identity']
  o=Path(b['output_root'])/m;assert not (o/'FAILURE.json').exists();co=read(o/'COMPLETE.json');assert co['binding_sha256']==BSHA and co['RESULT_sha256']==sha(o/'RESULT.json')
  assert finished[m]['COMPLETE_sha256']==sha(o/'COMPLETE.json') and finished[m]['RESULT_sha256']==co['RESULT_sha256']
  re=read(o/'RESULT.json');count=re['counts'];assert re['binding_sha256']==BSHA and re['method']==m and len(re['QA'])==80 and count['TRAIN_queries']==count['generations']==80
  assert all(count[k]==0 for k in ('F1','CE','backwards','optimizer_updates','new_checkpoints','TEST_queries')) and re['weights_and_buffers_unchanged'] and re['generation_config_unchanged']
  roots[m]=(o,re);files += [o/'COMPLETE.json',o/'RESULT.json']
 return dict(status='READY',binding=b,roots=roots,files=files,unit_terminal=u)
def read_method(method,root,result,selection,oldc,files):
 E=2 if method=='PASs_MoE' else 8;layers=set(oldc);assert len(layers)==168;data={};forwards=phasecount=0
 for item,expected in zip(result['QA'],selection):
  idx=item['index'];assert idx==len(data) and (item['task'],item['domain'],item['qa_id_sha256'])==(expected['task'],expected['domain'],expected['id_sha256'])
  d=root/('QA_%03d'%idx);cp=d/'COMPLETE.json';assert sha(cp)==item['QA_COMPLETE_sha256'];q=read(cp)
  assert (q['task'],q['domain'],q['qa_id_sha256'])==(item['task'],item['domain'],item['qa_id_sha256'])
  n=q['actual_prediction_positions'];L=q['initial_input_ids_length'];assert 1<=n<=32 and q['prefill_positions']==1 and q['decode_positions']==n-1 and q['decode_eligible']==(n>1)==item['decode_eligible']
  assert len(q['prefix_token_ids'])==len(q['steps'])==n==item['prefix_positions']==q['generation_decoder_forwards'] and q['no_gold_tokens_passed'] and q['own_model_native_trajectory']
  assert q['generation_contract']['generation_mode']=='greedy_search' and q['generation_contract']['effective_config']['max_time'] is None
  for i,v in enumerate(q['steps']):assert v['prediction_index']==i and v['phase']==('prefill' if i==0 else 'decode') and v['actual_L']==(L if i==0 else 1) and v['cache_length_before']==(0 if i==0 else L+i-1) and v['absolute_prediction_position']==v['cache_position'][-1]==L+i-1
  pf=d/'QA_PHASE_MOMENTS.jsonl';sf=d/'PREDICTION_RECORDS.jsonl';assert sha(pf)==q['moments_sha256']==item['moments_sha256'] and sha(sf)==q['position_records_sha256']==item['position_records_sha256']
  seen=set();direct={};phases={}
  with sf.open() as f:
   for line in f:
    x=json.loads(line);i=x['prediction_index'];layer=x['layer_id'];phase='prefill' if i==0 else 'decode';assert layer in layers and 0<=i<n and (layer,i) not in seen and x['phase']==phase;seen.add((layer,i))
    assert x['domain']==q['domain'] and hashlib.sha256(x['qa_id'].encode()).hexdigest()==q['qa_id_sha256'] and x['absolute_prediction_position']==L+i-1 and x['decoder_input_L']==(L if i==0 else 1) and x['cache_length_before']==(0 if i==0 else L+i-1)
    assert len(x['effective_gate'])==len(x['raw_token_gate'])==len(x['actual_softmax_input'])==E and finite(x['effective_gate']) and finite(x['actual_softmax_input'])
    assert x['actual_native_layer_output_closure']['bitwise_equal'] and x['moment_arithmetic_closure']['violating_elements']==0
    if method=='PASs_MoE':assert finite(x['actual_gate_tau']) and x['actual_gate_tau']>0
    direct.setdefault((layer,phase),[]).append(x['old_tf_reference_R_direct'])
  assert len(seen)==168*n==q['position_record_rows']
  with pf.open() as f:
   for line in f:
    x=json.loads(line);layer=x['layer_id'];phase=x['phase'];assert layer in layers and phase in ('prefill','decode') and (phase=='prefill' or n>1) and (layer,phase) not in phases
    assert x['selected_tokens']==(1 if phase=='prefill' else n-1) and x['expert_count']==E and x['prediction_positions']==([L-1] if phase=='prefill' else list(range(L,L+n-1))) and x['position_kinds']==['native_prediction']*x['selected_tokens']
    assert x['domain']==q['domain'] and hashlib.sha256(x['qa_id'].encode()).hexdigest()==q['qa_id_sha256'] and x['old_tf_c']==oldc[layer] and x['closure']['all_elements_within_tolerance']
    assert len(x['g_mean'])==E and finite([x[k] for k in ('g_mean','gg_mean','H','h','a','base_energy','centered_augmented_H','centered_augmented_h','centered_a')])
    assert close(math.fsum(direct[(layer,phase)])/x['selected_tokens'],x['old_tf_reference_R_direct']);phases[(layer,phase)]=x
  assert len(phases)==168*(1+int(n>1))==q['moments_rows'];files += [cp,pf,sf];forwards+=n;phasecount+=len(phases)
  data[q['qa_id_sha256']]=dict(domain=q['domain'],task=q['task'],n=n,phases=phases)
 assert len(data)==80 and result['counts']['native_prediction_forwards']==forwards
 return data,dict(QA=80,prediction_forwards=forwards,phase_rows=phasecount)
def residual(x,c):
 E=len(c);terms=[x['a'],-2*math.fsum(c[i]*x['h'][i] for i in range(E)),math.fsum(c[i]*x['H'][i][j]*c[j] for i in range(E) for j in range(E))]
 aug=c+[math.fsum(c)-1.];H=x['centered_augmented_H'];h=x['centered_augmented_h'];st=[x['centered_a'],-2*math.fsum(aug[i]*h[i] for i in range(E+1)),math.fsum(aug[i]*H[i][j]*aug[j] for i in range(E+1) for j in range(E+1))]
 raw=math.fsum(terms);stable=math.fsum(st);s=64*sys.float_info.epsilon*math.fsum(abs(v) for v in terms);ss=64*sys.float_info.epsilon*math.fsum(abs(v) for v in st)
 return dict(Rraw=raw,Rstable=stable,raw_minus_stable=raw-stable,raw_roundoff_scale=s,stable_roundoff_scale=ss,raw_negative_beyond_scale=raw < -s,stable_negative_beyond_scale=stable < -ss,U=x['a'],base=x['base_energy'])
def pooled_mean(data,layers):
 out={}
 for layer in layers:
  rows=[]
  for v in data.values():
   p=v['phases'][(layer,'prefill')]['g_mean'];d=v['phases'].get((layer,'decode'));n=v['n'];rows.append([(p[i]+((n-1)*d['g_mean'][i] if d else 0.))/n for i in range(len(p))])
  out[layer]=[math.fsum(v[i] for v in rows)/80 for i in range(len(rows[0]))]
 return out
def summarize(data,ids,phase,c):
 if not ids:return dict(QA=0,missing=True,reason='No eligible native decode QAs; never zero-filled')
 rows=[];pl={};diff=0.;nr=ns=0;keys=('Rraw','Rstable','U','base','raw_roundoff_scale','stable_roundoff_scale')
 for qid in sorted(ids):
  v=data[qid];total={k:0. for k in keys}
  for layer in c:
   z=residual(v['phases'][(layer,phase)],c[layer]);diff=max(diff,abs(z['raw_minus_stable']));nr+=int(z['raw_negative_beyond_scale']);ns+=int(z['stable_negative_beyond_scale'])
   for k in keys:total[k]+=z[k]
   pl.setdefault(layer,[]).append(z)
  rows.append(dict(qa_id_sha256=qid,domain=v['domain'],prediction_positions=1 if phase=='prefill' else v['n']-1,**total,R_over_U=ratio(total['Rstable'],total['U']),R_over_base=ratio(total['Rstable'],total['base'])))
 N=len(rows);totals={k:math.fsum(v[k] for v in rows)/N for k in keys};layerout={layer:{k:math.fsum(v[k] for v in vs)/N for k in (*keys,'raw_minus_stable')} for layer,vs in pl.items()}
 return dict(QA=N,missing=False,totals=totals,R_over_U=ratio(totals['Rstable'],totals['U']),R_over_base=ratio(totals['Rstable'],totals['base']),per_QA=rows,per_QA_R_over_U=distribution([v['R_over_U'] for v in rows]),per_QA_R_over_base=distribution([v['R_over_base'] for v in rows]),layers=layerout,max_layer_QA_raw_stable_difference=diff,negative_raw_beyond_scale=nr,negative_stable_beyond_scale=ns)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args();t=terminal()
 if t['status']!='READY':print(json.dumps(dict(status=t['status'],opened_partial_QA_streams=False)));return
 if not a.execute:print(json.dumps(dict(status='READY',new_models=0,new_generations=0,new_scores=0,output_exists=O.exists())));return
 assert not O.exists(),'Unique consumer: no replay or alternate namespace';O.mkdir();write(O/'STARTED.json',dict(binding_sha256=BSHA,source_sha256=sha(Path(__file__)),no_models_scores_or_generations=True))
 b=t['binding'];files=t['files'];ob=read(b['original_binding_path']);selection=read(ob['selection_path'])['calibration'];assert len(selection)==80;files += [Path(b['original_binding_path']),Path(ob['selection_path'])]
 old=read(b['old_tf_c_path'])['methods'];data={};counts={}
 for m in METHODS:data[m],counts[m]=read_method(m,*t['roots'][m],selection,old[m],files)
 assert set(data[METHODS[0]])==set(data[METHODS[1]]);eligible={m:{q for q,v in data[m].items() if v['n']>1} for m in METHODS};common=set.intersection(*eligible.values())
 result=dict(schema='native-prefix-function-readout-v1',binding_sha256=BSHA,methods={},common_decode_QA_ids=sorted(common),new_models=0,new_generations=0,new_scores=0,not_quality_or_resource_verification=True,reference_policy='oldTF_and_native_pooled_global_equal_QA_no_subset_refit',unit_terminal=t['unit_terminal'])
 for m in METHODS:
  v=data[m];native=pooled_mean(v,old[m]);groups={'prefill80':(set(v),'prefill'),'decode_eligible':(eligible[m],'decode'),'decode_common':(common,'decode')};reports={}
  for name,(ids,phase) in groups.items():reports[name]={ref:dict(all=summarize(v,ids,phase,c),domains={d:summarize(v,{q for q in ids if v[q]['domain']==d},phase,c) for d in sorted({x['domain'] for x in v.values()})}) for ref,c in [('oldTF',old[m]),('native_global',native)]}
  means={phase:{layer:([math.fsum(v[q]['phases'][(layer,phase)]['g_mean'][i] for q in ids)/len(ids) for i in range(len(c))] if ids else None) for layer,c in native.items()} for phase,ids in [('prefill',set(v)),('decode',eligible[m])]}
  paired={layer:([math.fsum(v[q]['phases'][(layer,'decode')]['g_mean'][i]-v[q]['phases'][(layer,'prefill')]['g_mean'][i] for q in eligible[m])/len(eligible[m]) for i in range(len(c))] if eligible[m] else None) for layer,c in native.items()}
  result['methods'][m]=dict(counts=counts[m],eligible_decode_QA_ids=sorted(eligible[m]),native_global_c=native,native_global_c_QA=80,phase_gate_mean_descriptive=means,paired_decode_minus_prefill_gate_mean_descriptive=paired,reports=reports)
 result['input_sha256']={str(p):sha(p) for p in files};write(O/'RESULT.json',result);write(O/'COMPLETE.json',dict(RESULT_sha256=sha(O/'RESULT.json'),source_sha256=sha(Path(__file__)),new_models=0,new_scores=0,new_generations=0));print(json.dumps(dict(status='COMPLETE',RESULT_sha256=sha(O/'RESULT.json'),output=str(O))))
if __name__=='__main__':main()
