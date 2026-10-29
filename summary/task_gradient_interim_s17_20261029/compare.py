from pathlib import Path
import json,hashlib
r=Path('/data1/liyapeng/program/LLMCL');out=r/'summary/task_gradient_interim_s17_20261029'
p=r/'summary/edge_deployment_20261019/sealed_baseline_tradeoffs_20261021/EVIDENCE.json';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='e5afa79b6bfeba63eff51b5855936f1072d00676bf9df1957ba4a70cce8d4626';e=json.loads(b)
p2=r/'summary/conditional_shared_budget_20261026/PAS_ACQUISITION_COMPARISON.json';b2=p2.read_bytes();assert hashlib.sha256(b2).hexdigest()=='9497bb25cfce9a7b37a8cbf4e08e5d9879f70025d38a9707f898c99711d6291a';past=json.loads(b2)
prior_path=r/'summary/task_gradient_interim_s13_20261028/CANDIDATE_SNAPSHOT.json'
prior_bytes=prior_path.read_bytes();assert hashlib.sha256(prior_bytes).hexdigest()=='ece86f6e76de90aea1d48ea05062aa9eb6bf9d5f129b821cfcba6c22a8e0b63a'
prior=json.loads(prior_bytes);extension=json.loads((out/'CANDIDATE_S14_S17.json').read_bytes())
assert [x['task'] for x in prior['rows']]==list(range(1,14))
assert [x['task'] for x in extension['rows']]==list(range(14,18))
c={'rows':prior['rows']+extension['rows']};rows=[]
for x in c['rows']:
 t=x['task'];z=dict(x)
 for m in ['SeqFT_LoRA','PASs_MoE','OSFT']:
  a=e['methods'][m]['task_results'][t-1]
  assert (a['task'],a['domain'],a['test_QA'],a['train_QA'])==(t,x['domain'],x['test_qa'],x['train_qa'])
  z[m+'_initial']=a['initial_F1']
  if m!='OSFT':assert abs(a['initial_F1']-x['initial'])<1e-9
  z[m]=a['acquired_F1'];z['delta_'+m]=x['acquired']-a['acquired_F1']
 a=past['rows'][t-1];assert a['task']==t and a['QA']==x['test_qa']
 for m,k in [('SeqFT_LoRA','Seq'),('PASs_MoE','PAS')]:
  assert abs(z[m]-a[k+'_acquired'])<1e-9
  z[m+'_before']=a[k+'_before'];z[m+'_gain']=a[k+'_change']
 rows.append(z)
metrics=['acquired','before','gain','SeqFT_LoRA','PASs_MoE','OSFT','SeqFT_LoRA_before','PASs_MoE_before','SeqFT_LoRA_gain','PASs_MoE_gain']
groups={}
manifest=next(v['json'] for k,v in prior['sources'].items() if k.endswith('/manifest.json'))
for domain in sorted({x['domain'] for x in c['rows']}):
 a=[x for x in rows if x['domain']==domain];n=sum(x['test_qa'] for x in a)
 expected=[x['task'] for x in manifest['tasks'] if x['domain']==domain]
 groups[domain]={'tasks':[x['task'] for x in a],'all_domain_tasks':expected,'QA':n,'complete_domain_acquisition':[x['task'] for x in a]==expected,**{m:sum(x[m]*x['test_qa'] for x in a)/n for m in metrics}}
result={'status':'INTERIM_17_COMMITTED_STAGES_NOT_FINAL_VERIFIED','rows':rows,'groups':groups,'task_equal_mean':{m:sum(x[m] for x in rows)/17 for m in metrics},'wins':{m:sum(x['delta_'+m]>0 for x in rows) for m in ['SeqFT_LoRA','PASs_MoE','OSFT']},'sources':{str(p):hashlib.sha256(b).hexdigest(),str(p2):hashlib.sha256(b2).hexdigest(),'prior_snapshot':hashlib.sha256(prior_bytes).hexdigest(),'extension_snapshot':hashlib.sha256((out/'CANDIDATE_S14_S17.json').read_bytes()).hexdigest()},'new_scores':0,'prediction_rows_read':0}
result['new_four_task_equal_mean']={m:sum(x[m] for x in rows if x['task']>=14)/4 for m in metrics}
result['boundaries']=['Acquired R[t,t], not final S19 F1','17-task equal mean is not five-domain macro','OSFT initial values differ; no equal-start causal claim','No test-driven changes to frozen method or budget']
with (out/'COMPARISON.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(result,ensure_ascii=False))
