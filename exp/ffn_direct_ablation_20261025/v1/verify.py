"""Score completed, successfully exited full graphs once; stdlib only."""
from pathlib import Path
import ast,collections,hashlib,json,math,os,re,subprocess,time
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN'])
ASSET=ROOT/'.deployment/paired_image_support_20261024_view'
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
 assert not (RUN/'VERIFIED.json').exists() and not (RUN/'VERIFY_STARTED.json').exists()
 env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
 raw=subprocess.check_output(['systemctl','--user','show','llmcl-ffn-direct-ablation-20261025.service','-p','MainPID','-p','ActiveState','-p','ExecMainStatus'],env=env,text=True)
 service=dict(line.split('=',1) for line in raw.splitlines())
 assert service['MainPID']=='0' and service['ActiveState']=='inactive' and service['ExecMainStatus']=='0',service
 ready=read(RUN/'READY.json');complete=read(RUN/'COMPLETE.json');started=read(RUN/'STARTED.json')
 assert complete['complete'] and complete['ready_sha256']==started['ready_sha256']==sha(RUN/'READY.json')
 assert complete['identity']==started['identity'] and not (Path('/proc')/str(started['identity']['pid'])).exists()
 assert complete['seconds']<=5400 and complete['NLL_requests']==1125 and complete['new_generations']==210
 assert complete['backwards']==complete['training_updates']==complete['TEST_QA']==0
 for name,h in complete['files'].items():assert sha(RUN/name)==h,name
 assert len(complete['selected'])==len(set(complete['selected']))==8
 assert complete['metadata']['parent']['layout']['weights']==dict(coefficients=893675552,bytes=1787351104)
 assert complete['metadata']['pruned']['layout']['weights']==dict(coefficients=789080096,bytes=1578160192)
 assert (RUN/complete['pruned']['path']).stat().st_size==complete['pruned']['bytes']
 split=read(ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json')
 assert sha(ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json')==ready['split_sha256']
 policy=read(ASSET/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json')
 manifest=read(ASSET/policy['conditions']['pct01']['manifest']);gold={};taskmap={t['task']:t for t in manifest['tasks']}
 for t in manifest['tasks']:
  wanted=set(split['dev_ids'][str(t['task'])]);rows=[json.loads(x) for x in (ASSET/t['files']['train']['path']).read_bytes().splitlines()]
  for row in rows:
   if row['id'] in wanted:gold[row['id']]=(t,row)
 assert len(gold)==100
 port=ASSET/'exp/two_backbone_native_cl_20261017_v2';pilot=read(ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/READY.json')
 assert sha(port/'runtime_utils.py')==pilot['sources']['exp/two_backbone_native_cl_20261017_v2/runtime_utils.py']
 tree=ast.parse((port/'runtime_utils.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
 ns=dict(re=re,Counter=collections.Counter);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-score>','exec'),ns)
 score=ns['score'];score_ast=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
 write(RUN/'VERIFY_STARTED.json',dict(unix=time.time(),complete_sha256=sha(RUN/'COMPLETE.json'),scorer_ast_sha256=score_ast))
 per_id={};quality={};memory={}
 for name in ('parent','pruned'):
  records=read(RUN/(name+'_predictions.json'));assert len(records)==100 and {x['id'] for x in records}==set(gold)
  seen=set();values=[]
  for row in records:
   assert row['id'] not in seen;seen.add(row['id']);task,original=gold[row['id']]
   assert row['task']==task['task'] and row['domain']==task['domain'] and row['references']==original['answers']
   assert row['generation_cap']==task['max_new_tokens'] and len(row['output_tokens'])<=task['max_new_tokens']
   assert math.isfinite(row['NLL']) and row['NLL']>=0 and row['target_tokens']>0
   f1=score(row['prediction'],row['references'])['token_f1'];assert math.isfinite(f1) and 0<=f1<=100
   values.append(dict(id=row['id'],domain=row['domain'],F1=f1,NLL=row['NLL'],input_sha256=row['input_sha256']))
  domains={d:dict(F1=sum(x['F1'] for x in values if x['domain']==d)/20,NLL=sum(x['NLL'] for x in values if x['domain']==d)/20,QA=20) for d in ('natural','medical','geography','finance','traffic')}
  assert all(sum(x['domain']==d for x in values)==20 for d in domains)
  quality[name]=dict(domains=domains,macro=sum(x['F1'] for x in domains.values())/5,per_id=values)
  per_id[name]={x['id']:x for x in values};memory[name]=read(RUN/(name+'_memory.json'))
 for key in per_id['parent']:assert per_id['parent'][key]['input_sha256']==per_id['pruned'][key]['input_sha256']
 assert [(x['id'],x['input_sha256'],x['prompt_tokens']) for x in memory['parent']]==[(x['id'],x['input_sha256'],x['prompt_tokens']) for x in memory['pruned']]
 assert len(memory['parent'])==len(memory['pruned'])==5 and all(x['new_tokens']==32 for rr in memory.values() for x in rr)
 differences={d:quality['pruned']['domains'][d]['F1']-quality['parent']['domains'][d]['F1'] for d in quality['parent']['domains']}
 passed=all(x>=0 for x in differences.values())
 result=dict(complete=True,source_COMPLETE_sha256=sha(RUN/'COMPLETE.json'),ready_sha256=sha(RUN/'READY.json'),quality=quality,domain_F1_pruned_minus_parent=differences,macro_F1_pruned_minus_parent=quality['pruned']['macro']-quality['parent']['macro'],all_domain_floor_passed=passed,decision='retain_for_independent_adaptation_review' if passed else 'close_fixed8_construction_quality_negative',selected=complete['selected'],whole_weight_saving_bytes=209190912,whole_weight_saving_fraction=104595456/893675552,memory=memory,model_RUN_seconds=complete['seconds'],NLL_requests=1125,new_generations=210,new_scores=200,backwards=0,training_updates=0,TEST_QA=0,scorer_ast_sha256=score_ast,no_equivalence_significance_strongest_or_framework_success_claim=True)
 write(RUN/'VERIFIED.json',result)
 lines=['# 真实8-FFN删除完整结果','','| 领域 | 完整父F1 | 删除F1 | 删除−父pp |','| --- | ---: | ---: | ---: |']
 for d,x in differences.items():lines.append(f'| {d} | {quality["parent"]["domains"][d]["F1"]:.6f} | {quality["pruned"]["domains"][d]["F1"]:.6f} | {x:+.6f} |')
 lines += ['',f'五域宏：父{quality["parent"]["macro"]:.9f}，删除{quality["pruned"]["macro"]:.9f}。',f'固定出口：{result["decision"]}。',f'选择层（0-based）：{complete["selected"]}。','完整权重减少104595456系数／199.5MiB／11.704%；实际GPU峰值见VERIFIED五个同输入32token窗口，不能按权重降幅推峰值。','单seed、已探索100DEV，不称等效、显著、稳定或已胜正式最强。任何通过仅进入独立适应必要性审查；质量负结果不搜索层、幅度或恢复训练。',f'模型RUN{complete["seconds"]:.6f}秒，1125NLL／210生成／0backward／0训练；CPU父导出、运输与全部旧失败另计。','']
 (RUN/'FINDINGS.md').write_text('\n'.join(lines),encoding='utf8')
 print(json.dumps(dict(VERIFIED_sha256=sha(RUN/'VERIFIED.json'),decision=result['decision'],domains=differences,parent_macro=quality['parent']['macro'],pruned_macro=quality['pruned']['macro'])))
if __name__=='__main__':main()
