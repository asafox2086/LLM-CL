"""Describe domain organization from published moments; never read raw QA/model."""
from pathlib import Path
import argparse,hashlib,json,math,os,time
import numpy as np
SOURCE_SHA='55e95fd9dde9a76b34aa02bdd12b507e03506feb2137a706ba31067c37da6ef1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ratio(a,b):return float(a/b) if b>0 else None
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
 src=args.root/'summary/router_function_moments_20261030_v2/READOUT.json';d=args.root/'summary/router_function_domain_decomposition_20261030'
 assert sha(src)==SOURCE_SHA
 if not args.execute:print(json.dumps(dict(source_sha256=SOURCE_SHA,executed=False,raw_QA_read=False)));return
 assert not (d/'STARTED.json').exists() and not (d/'RESULT.json').exists()
 with (d/'STARTED.json').open('x') as f:json.dump(dict(unix=time.time(),source_sha256=SOURCE_SHA,consumer_sha256=sha(__file__)),f);f.flush();os.fsync(f.fileno())
 source=json.loads(src.read_bytes());result={};eps=np.finfo(np.float64).eps
 for method,m in source['methods'].items():
  layers={};global_total={k:0. for k in ['W','B','C','R','A','gate_within','gate_between','gate_total']};domain_total={};max_fn_closure=0.;max_gate_closure=0.;unidentifiable=0;negative_beyond_scale=0
  assert len(m['layers'])==168
  for name,L in m['layers'].items():
   all_=L['all'];c=np.array(all_['c_global_mean_gate']);qa=all_['qa_count'];assert qa==80 and len(L['domains'])==5
   domain_rows={};weighted={k:0. for k in global_total};qsum=0
   for domain,x in L['domains'].items():
    n=x['qa_count'];qsum+=n;weight=n/qa;assert n==16
    mu=np.array(x['group_g_mean']);H=np.array(x['mean_H']);h=np.array(x['mean_h']);a=x['mean_routed_update_energy'];delta=mu-c
    terms=[a,-2*float(mu@h),float(mu@H@mu)];W=sum(terms);B=float(delta@H@delta);C=2*float(delta@(h-H@mu));R=x['centered_Rjoint_raw'];A=a
    scale=float(256*eps*(abs(a)+2*np.abs(mu*h).sum()+np.abs(mu[:,None]*H*mu[None,:]).sum()+2*np.abs(c*h).sum()+np.abs(c[:,None]*H*c[None,:]).sum()))
    closure=W+B+C-R;assert abs(closure)<=scale+float(x['Rjoint_fp64_roundoff_scale'])
    gate_total=float(x['gate_deviation_trace_raw']);gate_between=float(delta@delta);gate_within=gate_total-gate_between
    gate_scale=float(256*eps*(np.abs(x['gate_deviation_second_moment']).sum()+abs(gate_between)+abs(gate_total)))
    item=dict(QA=n,W_raw=float(W),B_raw=B,C_raw=C,R_global_reference_raw=R,update_energy=A,
      within_reference_over_global_reference=ratio(W,R),within_reference_over_update=ratio(W,A),
      function_closure_raw=float(closure),function_FP64_scale=scale,
      within_below_FP64_scale=bool(abs(W)<=scale),within_negative_beyond_scale=bool(W < -scale),
      gate_total_raw=gate_total,gate_between_raw=gate_between,gate_within_raw=gate_within,gate_FP64_scale=gate_scale,
      gate_within_negative_beyond_scale=bool(gate_within < -gate_scale))
    assert all(math.isfinite(v) for v in [W,B,C,R,A,gate_total,gate_between,gate_within])
    domain_rows[domain]=item;unidentifiable+=int(item['within_below_FP64_scale']);negative_beyond_scale+=int(item['within_negative_beyond_scale'])
    max_fn_closure=max(max_fn_closure,abs(closure))
    vals=dict(W=W,B=B,C=C,R=R,A=A,gate_within=gate_within,gate_between=gate_between,gate_total=gate_total)
    for key,value in vals.items():weighted[key]+=weight*value
    dest=domain_total.setdefault(domain,{key:0. for key in global_total})
    for key,value in vals.items():dest[key]+=value
   assert qsum==qa
   gate_error=weighted['gate_total']-all_['gate_deviation_trace_raw'];max_gate_closure=max(max_gate_closure,abs(gate_error));assert abs(gate_error)<=max(1e-24,256*eps*max(abs(weighted['gate_total']),abs(all_['gate_deviation_trace_raw'])))
   for key,value in weighted.items():global_total[key]+=value
   layers[name]=dict(domains=domain_rows,QA_weighted_sums=weighted,gate_partition_closure_raw=gate_error)
  def summarize(v):
   return dict(sums=v,domain_mean_reference_over_global_mean_reference=ratio(v['W'],v['R']),
    domain_mean_reference_over_update=ratio(v['W'],v['A']),global_mean_reference_over_update=ratio(v['R'],v['A']),
    gate_between_fraction=ratio(v['gate_between'],v['gate_total']),gate_within_fraction=ratio(v['gate_within'],v['gate_total']),
    function_R_partition_closure_raw=v['W']+v['B']+v['C']-v['R'],
    warning='function W/B/C are not an orthogonal explanation; no function explained fraction or deployment necessity claim')
  result[method]=dict(all_layers=summarize(global_total),domains={k:summarize(v) for k,v in domain_total.items()},layers=layers,
    numerical=dict(max_function_closure=max_fn_closure,max_gate_closure=max_gate_closure,layer_domain_within_below_FP64_scale=unidentifiable,layer_domain_negative_beyond_scale=negative_beyond_scale))
 out=dict(schema='published-router-domain-decomposition-v1',source_sha256=SOURCE_SHA,consumer_sha256=sha(__file__),methods=result,raw_QA_read=0,raw_moment_files_read=0,new_models=0,new_scores=0,TRAIN_only=True)
 with (d/'RESULT.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({method:{'all_layers':v['all_layers'],'domains':v['domains'],'numerical':v['numerical']} for method,v in result.items()}))
if __name__=='__main__':main()
