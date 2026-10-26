"""Derive PAS versus SeqFT acquisition from existing sealed aggregates only."""
from pathlib import Path
import base64, hashlib, json, math

def sha(raw): return hashlib.sha256(raw).hexdigest()

def main():
 root=Path.cwd()
 bundle_path=root/'.deployment/private_pas_acquisition_cells_20261026.json'
 seq_path=root/'summary/train_answer_prior_20261025/ALL_TASK_ACQUISITION.json'
 bundle=json.loads(bundle_path.read_bytes()); seq=json.loads(seq_path.read_bytes())
 proof_path=root/Path(bundle['proof_path'])
 proof_raw=proof_path.read_bytes(); proof=json.loads(proof_raw)
 assert sha(proof_raw)==bundle['proof_sha256']=='a17c1000dfff6816644d3e86170fe255d6c3225eebfa911d4c0834fcabb0dc9f'
 assert proof['rescored_cells']==74 and proof['final_QA']==21454
 assert seq['backbone']=='onevision' and seq['method']=='SeqFT_LoRA'
 assert len(seq['tasks'])==19 and len(bundle['files'])==38
 cells={}
 run=proof['candidate']['run']
 for item in bundle['files']:
  t,s,w=item['task'],item['stage'],item['when']
  assert w in ('before','acquired') and s==t-(w=='before')
  name=f'{run}/evaluations/stage_{s:02d}/T{t:02d}_test_predictions.scores.json'
  raw=base64.b64decode(item['raw_base64'],validate=True)
  assert item['relative']==name and sha(raw)==item['sha256']==proof['source_sha256'][name]
  value=json.loads(raw)
  assert set(value)=={'rougeL','token_f1','exact_match','count','exact_match_correct'}
  assert math.isfinite(value['token_f1']) and 0<=value['token_f1']<=100
  assert (t,w) not in cells
  cells[t,w]=dict(F1=value['token_f1'],count=value['count'],source_sha256=sha(raw))
 rows=[]
 for old in seq['tasks']:
  t=old['task']; before=cells[t,'before']; acquired=cells[t,'acquired']
  assert before['count']==acquired['count']==old['TEST_QA']
  test=old['TEST_identity']
  assert proof['source_sha256'][test['source_paths']['pct01']]==test['sha256']
  baseline=old['pct01']
  delta=acquired['F1']-before['F1']
  rows.append(dict(task=t,domain=old['domain'],label=old['native_label'],QA=old['TEST_QA'],
   Seq_before=baseline['before_F1'],Seq_acquired=baseline['acquired_F1'],Seq_change=baseline['acquisition_change_pp'],
   PAS_before=before['F1'],PAS_acquired=acquired['F1'],PAS_change=delta,
   acquired_gap=acquired['F1']-baseline['acquired_F1'],change_gap=delta-baseline['acquisition_change_pp'],
   PAS_cell_sources={'before':before['source_sha256'],'acquired':acquired['source_sha256']}))
 assert [r['task'] for r in rows]==list(range(1,20))
 domains={}
 keys=('Seq_before','Seq_acquired','Seq_change','PAS_before','PAS_acquired','PAS_change','acquired_gap','change_gap')
 for domain in ('natural','medical','geography','finance','traffic'):
  group=[r for r in rows if r['domain']==domain]; n=sum(r['QA'] for r in group)
  domains[domain]={k:sum(r[k]*r['QA'] for r in group)/n for k in keys}
  domains[domain]['QA']=n
 result=dict(status='derived_from_sealed_aggregates',rows=rows,domains=domains,
  domain_macro={k:sum(d[k] for d in domains.values())/5 for k in keys},
  task_equal_mean={k:sum(r[k] for r in rows)/19 for k in keys},
  task_counts={k:{'higher':sum(r[k]>1e-10 for r in rows),'lower':sum(r[k]<-1e-10 for r in rows),'equal':sum(abs(r[k])<=1e-10 for r in rows)} for k in ('acquired_gap','change_gap')},
  source_sha256={'PAS_proof':sha(proof_raw),'Seq_derived':sha(seq_path.read_bytes()),'PAS_bundle':sha(bundle_path.read_bytes()),'analyzer':sha(Path(__file__).read_bytes())},
  activity={'new_model':0,'new_score':0,'QA_or_predictions_read':0,'checkpoint_read':0},
  limits=['Different learned starts, locations, capacity and method configuration; not routing causality.',
          'Descriptive fixed single-seed comparison; no significance test or method tuning.',
          'Acquired absolute level and within-stage change are separate; no forgetting analysis.'])
 out=root/'summary/conditional_shared_budget_20261026/PAS_ACQUISITION_COMPARISON.json'
 with out.open('x',encoding='utf8') as f: json.dump(result,f,indent=2,ensure_ascii=False)
 lines=['# 正式1% PAS与SeqFT：刚习得水平及阶段变化','',
  '原LLaVA-OneVision正式固定配置，全19任务；只派生原封存汇总分数，0模型/评分/QA/CP读取。领域内按测试题数加权，五域等权；另列任务等权，二者不能混用。','',
  '| 域 | Seq刚习得 | PAS刚习得 | 刚习得差 | Seq阶段变化 | PAS阶段变化 |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for name,d in list(domains.items())+[('五域宏',result['domain_macro'])]:
  lines.append(f"| {name} | {d['Seq_acquired']:.6f} | {d['PAS_acquired']:.6f} | {d['acquired_gap']:+.6f} | {d['Seq_change']:+.6f} | {d['PAS_change']:+.6f} |")
 lines+=['','| 任务 | 域/标签 | Seq前→后 | PAS前→后 | 刚习得差 | 阶段增量差 |','| --- | --- | ---: | ---: | ---: | ---: |']
 for d in rows:
  lines.append(f"| T{d['task']:02d} | {d['domain']}/{d['label']} | {d['Seq_before']:.3f}→{d['Seq_acquired']:.3f} | {d['PAS_before']:.3f}→{d['PAS_acquired']:.3f} | {d['acquired_gap']:+.3f} | {d['change_gap']:+.3f} |")
 lines+=['','任务计数：`'+json.dumps(result['task_counts'])+'`。','',
  '不同方法继承的学习前状态不同，不能由阶段增量直接认定哪个模块改善优化；不以TEST派生选择任务、秩或超参。精确数值、原分数SHA及程序SHA见同名JSON。']
 with out.with_suffix('.md').open('x',encoding='utf8') as f: f.write('\n'.join(lines)+'\n')
 print(json.dumps({k:result[k] for k in ('domain_macro','task_equal_mean','task_counts')},ensure_ascii=False))

if __name__=='__main__': main()
