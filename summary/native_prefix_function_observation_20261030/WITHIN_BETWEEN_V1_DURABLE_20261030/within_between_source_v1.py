"""DRAFT once-only request granularity observation, existing native QA-phase matrices only.
No PREDICTION_RECORDS, answer, model, forward, generation, score or parameter writes.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,sys
P=Path('/data2/liyapeng_grp/program/LLMCL');S=P/'summary/native_prefix_function_observation_20261030/path_compat_v1'
OLD=S/'readout_v1';O=S/'within_between_readout_v1';RESULT_SHA='8e7e509b31501fdea20ebf1de35c0453f1648ee591b7b3bdfe92bb8b72213a93';COMPLETE_SHA='47eb46902fca5d25c279415076d38123a77e2b05db353ddf42ab1b35c837645d'
METHODS=('PASs_MoE','CL_MoE');EPS=sys.float_info.epsilon

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
def finite(v):return all(finite(x) for x in v) if isinstance(v,list) else isinstance(v,(int,float)) and math.isfinite(v)
def mean(values,weights):
 if isinstance(values[0],list):return [mean([x[i] for x in values],weights) for i in range(len(values[0]))]
 return math.fsum(w*x for w,x in zip(weights,values))
def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))
def mv(H,v):return [dot(row,v) for row in H]
def quad(H,v):return dot(v,mv(H,v))
def ratio(a,b):return a/b if b>0 else None
def estimate(terms):
 value=math.fsum(terms);scale=64*EPS*math.fsum(abs(t) for t in terms)
 return dict(value=value,terms=terms,roundoff_scale=scale,negative_beyond_scale=value < -scale)
def residual(a,h,H,c):return estimate([a,-2*dot(c,h),quad(H,c)])
def quadratic_energy(H,d):
 terms=[d[i]*H[i][j]*d[j] for i in range(len(d)) for j in range(len(d))]
 z=estimate(terms);z['absolute_contraction_terms_sum']=math.fsum(abs(v) for v in terms);z['terms']=[z['value']]
 return z
def decompose(x,cq,c):
 d=[a-b for a,b in zip(cq,c)];H=x['H'];h=x['h'];a=x['a']
 R=residual(a,h,H,c);W=residual(a,h,H,cq);B=quadratic_energy(H,d);C=estimate([2*dot(d,h),-2*dot(d,mv(H,cq))]);dotraw=C['value']/2
 # Preserve finite gate-sum errors, not an implicit simplex projection.
 ca=c+[math.fsum(c)-1.];cqa=cq+[math.fsum(cq)-1.];da=d+[math.fsum(d)]
 cH=x['centered_augmented_H'];ch=x['centered_augmented_h'];aa=x['centered_a']
 Rs=residual(aa,ch,cH,ca);Ws=residual(aa,ch,cH,cqa);Bs=quadratic_energy(cH,da);Cs=estimate([2*dot(da,ch),-2*dot(da,mv(cH,cqa))]);dots=Cs['value']/2
 rawclose=R['value']-math.fsum([W['value'],B['value'],C['value']]);stableclose=Rs['value']-math.fsum([Ws['value'],Bs['value'],Cs['value']])
 return dict(full_raw=R,within_raw=W,between_raw=B,cross_twice_raw=C,cross_dot_raw=dotraw,
  full_stable=Rs,within_stable=Ws,between_stable=Bs,cross_twice_stable=Cs,cross_dot_stable=dots,
  raw_closure_error=rawclose,stable_closure_error=stableclose,
  raw_closure_scale=64*EPS*math.fsum(abs(z) for z in [R['value'],W['value'],B['value'],C['value']]),
  stable_closure_scale=64*EPS*math.fsum(abs(z) for z in [Rs['value'],Ws['value'],Bs['value'],Cs['value']]),
  augmented_sum_rounding_difference=da[-1]-(cqa[-1]-ca[-1]),
  raw_minus_stable={k:a['value']-b['value'] for k,a,b in [('full',R,Rs),('within',W,Ws),('between',B,Bs),('cross_twice',C,Cs)]},U=x['a'],base=x['base_energy'])
def scope():
 if not (OLD/'COMPLETE.json').exists():return None
 assert sha(OLD/'COMPLETE.json')==COMPLETE_SHA and sha(OLD/'RESULT.json')==RESULT_SHA
 complete=read(OLD/'COMPLETE.json');j=read(OLD/'RESULT.json');assert complete['RESULT_sha256']==RESULT_SHA and complete['source_sha256']=='94855e95172c36d7975274520a5a04740fba8ab64776e12e22de1fbaee32a0ba'
 assert j['new_models']==j['new_generations']==j['new_scores']==0 and set(j['methods'])==set(METHODS)
 b=read(S/'BINDING.json');assert sha(S/'BINDING.json')==j['binding_sha256']
 # Reuse the full completed verifier; do not open its PRED hashes/streams again.
 for p in [S/'CONTROL_COMPLETE.json',S/'SLOT.json']:
  assert sha(p)==j['input_sha256'][str(p)]
 assert read(S/'CONTROL_COMPLETE.json')['released'] and read(S/'SLOT.json')['released']
 return j,b

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');arg=ap.parse_args();v=scope()
 if v is None:print(json.dumps(dict(status='WAITING',QA_phase_files_opened=False)));return
 if not arg.execute:print(json.dumps(dict(status='READY',QA_phase_files_opened=False,output_exists=O.exists())));return
 assert not O.exists(),'Unique observation; preserve any failed marker, no alternate namespace/retry'
 O.mkdir();write(O/'STARTED.json',dict(source_sha256=sha(Path(__file__)),source_native_READOUT_sha256=RESULT_SHA,no_models_or_PRED_streams=True))
 j,b=v;result=dict(schema='native-request-within-between-v1',source_native_READOUT_sha256=RESULT_SHA,methods={},new_models=0,new_generations=0,new_scores=0,PRED_streams_reopened=False,c_QA_is_full_prefix_descriptive_oracle_not_prompt_deployable=True,no_success_threshold=True);inputs={}
 fields=('H','h','a','base_energy','centered_augmented_H','centered_augmented_h','centered_a')
 for method in METHODS:
  cglobal=j['methods'][method]['native_global_c'];layers=set(cglobal);assert len(layers)==168;E=2 if method=='PASs_MoE' else 8
  expect={q['qa_id_sha256'] for q in j['methods'][method]['reports']['prefill80']['native_global']['all']['per_QA']};assert len(expect)==80
  perQA=[];perlayer={n:[] for n in sorted(layers)};cq_rows={n:[] for n in layers};gate_sum={'within':0.,'between':0.,'total':0.};seen=set()
  for idx in range(80):
   root=Path(b['output_root'])/method/('QA_%03d'%idx);cp=root/'COMPLETE.json';assert sha(cp)==j['input_sha256'][str(cp)];q=read(cp);qid=q['qa_id_sha256'];n=q['actual_prediction_positions'];assert qid in expect and qid not in seen and 1<=n<=32;seen.add(qid)
   p=root/'QA_PHASE_MOMENTS.jsonl';raw=p.read_bytes()
   assert hashlib.sha256(raw).hexdigest()==q['moments_sha256']==j['input_sha256'][str(p)]
   rows=[json.loads(line) for line in raw.splitlines()] # one physical phase-file read, hash and parse same bytes
   inputs[str(cp)]=sha(cp);inputs[str(p)]=q['moments_sha256'];by={}
   for x in rows:
    key=(x['layer_id'],x['phase']);assert key not in by and key[0] in layers and key[1] in ('prefill','decode')
    assert x['selected_tokens']==(1 if x['phase']=='prefill' else n-1) and x['selected_tokens']>0 and x['expert_count']==E
    assert hashlib.sha256(x['qa_id'].encode()).hexdigest()==qid and x['closure']['all_elements_within_tolerance']
    assert all(finite(x[k]) for k in fields) and finite(x['g_mean']) and len(x['g_mean'])==E;by[key]=x
   assert len(by)==168*(1+int(n>1));tot={k:0. for k in ('full','within','between','cross_twice','U','base')};qt={'within':0.,'between':0.,'total':0.}
   for layer in sorted(layers):
    phase=[by[(layer,'prefill')]]+([by[(layer,'decode')]] if n>1 else []);weights=[1/n]+([(n-1)/n] if n>1 else [])
    x={k:mean([p[k] for p in phase],weights) for k in fields};cq=mean([p['g_mean'] for p in phase],weights);c=cglobal[layer];cq_rows[layer].append(cq)
    z=decompose(x,cq,c);z['qa_id_sha256']=qid;perlayer[layer].append(z)
    for k in ('full','within','between','cross_twice'):tot[k]+=z[k+'_stable']['value']
    tot['U']+=z['U'];tot['base']+=z['base']
    # Exact total-variance law on gate coordinates, equal QA weights. Not a function-energy attribution.
    gw=math.fsum(w*(math.fsum(p['gate_covariance_within_QA'][i][i] for i in range(E))+math.fsum((p['g_mean'][i]-cq[i])**2 for i in range(E))) for w,p in zip(weights,phase))
    gb=math.fsum((cq[i]-c[i])**2 for i in range(E))
    gt=math.fsum(w*(math.fsum(p['gate_covariance_within_QA'][i][i] for i in range(E))+math.fsum((p['g_mean'][i]-c[i])**2 for i in range(E))) for w,p in zip(weights,phase))
    for k,val in [('within',gw),('between',gb),('total',gt)]:qt[k]+=val
   for k in gate_sum:gate_sum[k]+=qt[k]/80
   perQA.append(dict(qa_id_sha256=qid,prefix_positions=n,**tot,gate_variance=qt,within_over_U=ratio(tot['within'],tot['U']),between_over_U=ratio(tot['between'],tot['U'])))
  assert seen==expect
  cached_mean_error=max(abs(math.fsum(v[i] for v in cq_rows[layer])/80-cglobal[layer][i]) for layer in layers for i in range(E));assert cached_mean_error<=64*EPS
  totals={k:math.fsum(x[k] for x in perQA)/80 for k in ('full','within','between','cross_twice','U','base')}
  result['methods'][method]=dict(QA=80,per_QA=perQA,per_layer_full_diagnostics=perlayer,cached_native_global_mean_max_error=cached_mean_error,
   totals=totals,closure=totals['full']-math.fsum([totals['within'],totals['between'],totals['cross_twice']]),
   within_to_between=ratio(totals['within'],totals['between']),within_over_U=ratio(totals['within'],totals['U']),between_over_U=ratio(totals['between'],totals['U']),cross_twice_over_U=ratio(totals['cross_twice'],totals['U']),
   gate_variance=gate_sum,gate_variance_closure=gate_sum['total']-gate_sum['within']-gate_sum['between'])
 result['phase_and_QA_input_sha256']=inputs;write(O/'RESULT.json',result);write(O/'COMPLETE.json',dict(RESULT_sha256=sha(O/'RESULT.json'),source_sha256=sha(Path(__file__)),source_native_READOUT_sha256=RESULT_SHA));print(json.dumps(dict(status='COMPLETE',RESULT_sha256=sha(O/'RESULT.json'),output=str(O))))
if __name__=='__main__':main()
