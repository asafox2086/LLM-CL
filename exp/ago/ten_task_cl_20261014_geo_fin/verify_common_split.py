from pathlib import Path
import json,hashlib,subprocess,sys,os
ROOT=Path('/data1/liyapeng/program/LLMCL');HERE=ROOT/'exp/ten_task_cl_20261014_geo_fin'
def verify():
 policy=json.loads((HERE/'execution_policy.json').read_text(encoding='utf-8'));lock=json.loads((ROOT/policy['split_lock']).read_text(encoding='utf-8'));p=ROOT/policy['manifest'];m=json.loads(p.read_text(encoding='utf-8'))
 assert hashlib.sha256(p.read_bytes()).hexdigest()==policy['manifest_sha256']==lock['manifest_sha256']
 orderp=ROOT/lock['training_order'];assert hashlib.sha256(orderp.read_bytes()).hexdigest()==lock['training_order_sha256'];order=json.loads(orderp.read_text(encoding='utf-8'))
 checks=[]
 for t in m['tasks']:
  roles={}
  for split,info in t['files'].items():
   p=ROOT/info['path'];assert HERE.name in str(p);assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']==lock['tasks'][t['task_id']][split]['sha256'];rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()];assert len(rows)==({'train':100,'test':50}[split]);assert len({x['id'] for x in rows})==len(rows);assert p.stat().st_mode&0o222==0
   roles[split]={x['id'] for x in rows}
  assert not roles['train']&roles['test'];assert len(order[t['task_id']])==100 and set(order[t['task_id']])==roles['train']
  checks.append({'task':t['task_id'],'train':100,'test':50,'qa_id_overlap':0,'hash_passed':True,'training_id_order_passed':True})
 assert not policy['save_checkpoints'] and policy['seed']==42 and not policy['validation_enabled'];assert policy['allowed_gpus']==[1,2,3] and policy['reserved_gpus']==[0]
 status=json.loads((ROOT/'summary'/HERE.name/'run_status.json').read_text(encoding='utf-8'));running=[]
 for method,job in status['jobs'].items():
  if job['status']!='running':continue
  p=ROOT/job['run_dir'];cfg=json.loads((p/'config.json').read_text(encoding='utf-8'));assert cfg['manifest_sha256']==lock['manifest_sha256'] and cfg['experiment_id']==HERE.name;assert cfg['seed']==42 and not cfg['save_checkpoints'];assert job['gpu'] in [1,2,3]
  env=dict(x.split('=',1) for x in Path(f"/proc/{job['pid']}/environ").read_bytes().decode().split('\0') if '=' in x);assert env['CUDA_VISIBLE_DEVICES']==str(job['gpu']);assert not list(p.glob('checkpoints/**/*.pt'));running.append({'method':method,'gpu':job['gpu'],'pid':job['pid'],'manifest_match':True,'checkpoints':0})
 out={'passed':True,'experiment_id':HERE.name,'manifest_sha256':lock['manifest_sha256'],'all_methods_same_split':True,'task_checks':checks,'running_workers_checked':running,'reserved_gpus':[0]}
 p=HERE/'preflight/common_split_audit.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':verify()
