"""One read-only factorization of the already verified router moments; no model/QA."""
from pathlib import Path
import argparse,hashlib,json,math,os,sys
R=Path('/data1/liyapeng/program/LLMCL')
INPUT=R/'summary/router_function_moments_20261030_v2/READOUT.json'
INPUT_SHA='55e95fd9dde9a76b34aa02bdd12b507e03506feb2137a706ba31067c37da6ef1'
O=R/'summary/conditional_function_factorization_20261030/result_v1'
def write(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
def factors(t):
 keys=('I','T','U','J');valid=all(t[k]>0 and math.isfinite(t[k]) for k in keys)
 if not valid:return dict(valid=False,reason='Nonpositive or nonfinite aggregate denominator; no clamp or epsilon')
 f=dict(joint_coupling=t['R']/t['J'],expert_energy_weighted_gate_deviation=t['I']/t['T'],expert_output_energy_over_dynamic_update=t['T']/t['U'],geometry_trace_contraction=t['J']/t['I'])
 product=math.prod(f.values());direct=t['R']/t['U']
 return dict(valid=True,values=f,product=product,direct_R_over_U=direct,absolute_closure_error=abs(product-direct),variance_normalized_sensitivity=f['expert_output_energy_over_dynamic_update']*f['geometry_trace_contraction'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 if not a.execute:
  print(json.dumps(dict(source_only=True,input=str(INPUT),input_sha256=INPUT_SHA,output=str(O),new_models=0,new_QA=0,new_scores=0)));return
 assert not O.exists(),'Never replay this descriptive consumer'
 raw=INPUT.read_bytes();assert hashlib.sha256(raw).hexdigest()==INPUT_SHA
 j=json.loads(raw);assert j['TRAIN_only'] and j['TEST_QA']==j['new_scores']==j['new_generations']==j['new_model_loads']==0
 assert set(j['methods'])=={'PASs_MoE','CL_MoE'}
 O.mkdir(parents=True);write(O/'STARTED.json',dict(input_sha256=INPUT_SHA,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_models_QA_scores=True))
 result=dict(input_sha256=INPUT_SHA,methods={},new_models=0,new_QA=0,new_scores=0,new_generations=0,descriptive_not_causal=True,all_group_only=True,no_gate_renormalization_or_covariance_projection=True)
 for name,m in j['methods'].items():
  assert len(m['layers'])==168;rows=[]
  for layer,v in sorted(m['layers'].items()):
   q=v['all'];C=q['gate_deviation_second_moment'];H=q['mean_H'];e=len(C)
   assert q['qa_count']==80 and e==(2 if name=='PASs_MoE' else 8) and len(H)==e
   assert all(len(x)==e for x in C+H) and all(math.isfinite(x) for mat in (C,H) for row in mat for x in row)
   tc=math.fsum(C[i][i] for i in range(e));th=math.fsum(H[i][i] for i in range(e))
   terms=[H[i][k]*C[k][i] for i in range(e) for k in range(e)];jc=math.fsum(terms);stored=q['Rindep_raw']
   bound=128*sys.float_info.epsilon*math.fsum(abs(x) for x in terms)
   assert abs(jc-stored)<=bound,'Stored J does not close against C/H trace contraction'
   rows.append(dict(layer=layer,E=e,R=q['Rjoint_raw'],J=stored,U=q['mean_routed_update_energy'],T=th,I=tc*th,gate_deviation_trace=tc,J_from_matrices=jc,J_matrix_closure_bound=bound,centered_R=q['centered_Rjoint_raw'],centered_J=q['centered_Rindep_raw'],R_roundoff_scale=q['Rjoint_fp64_roundoff_scale'],R_negative_beyond_roundoff=q['Rjoint_negative_beyond_roundoff'],mean_gate_sum_error=q['mean_gate_sum_error']))
  totals={k:math.fsum(x[k] for x in rows) for k in ('R','J','U','T','I','centered_R','centered_J','R_roundoff_scale')}
  result['methods'][name]=dict(totals=totals,factorization=factors(totals),layers=rows,nonpositive_gate_trace_layers=[x['layer'] for x in rows if x['gate_deviation_trace']<=0],negative_R_beyond_roundoff_layers=[x['layer'] for x in rows if x['R_negative_beyond_roundoff']],raw_minus_centered_R=totals['R']-totals['centered_R'],raw_minus_centered_J=totals['J']-totals['centered_J'])
 a=result['methods']['PASs_MoE']['factorization'];b=result['methods']['CL_MoE']['factorization']
 if a['valid'] and b['valid'] and all(x>0 for x in a['values'].values()):
  ratios={k:b['values'][k]/a['values'][k] for k in a['values']};result['CL_over_PAS_descriptive_factors']=dict(factors=ratios,product=math.prod(ratios.values()),direct_residual_ratio=b['direct_R_over_U']/a['direct_R_over_U'],variance_normalized_sensitivity_ratio=b['variance_normalized_sensitivity']/a['variance_normalized_sensitivity'])
 write(O/'RESULT.json',result);write(O/'COMPLETE.json',dict(RESULT_sha256=hashlib.sha256((O/'RESULT.json').read_bytes()).hexdigest(),new_models=0,new_scores=0))
 print(json.dumps(dict(complete=str(O/'COMPLETE.json'),methods={k:v['factorization'] for k,v in result['methods'].items()})))
if __name__=='__main__':main()
