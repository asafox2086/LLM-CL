"""Consume complete fixed-input scalars once, with no model or F1 scorer."""
from pathlib import Path
import collections,hashlib,json,math,os,subprocess,time
RUN=Path(os.environ['LLMCL_RUN']);ROOT=Path(os.environ['LLMCL_ROOT'])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(n,v):
 with (RUN/n).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
begin=time.monotonic();ready=read(RUN/'READY.json');start=read(RUN/'STARTED.json');complete=read(RUN/'COMPLETE.json');exit=read(RUN/'EXIT.json')
assert not (RUN/'FAILURE.json').exists()
assert sha(RUN/'READY.json')==start['ready_sha256']==complete['ready_sha256']
assert complete['identity']==start['identity'] and complete['identity']['pid']==exit['pid'] and exit['returncode']==0
assert not (Path('/proc')/str(exit['pid'])).exists()
assert 0<complete['seconds']<600 and 0<exit['unix']-start['unix']<630
env=dict(os.environ,XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
p=subprocess.run(['systemctl','--user','show','llmcl-ffn-joint-effect-20261025.service','-p','ActiveState','-p','SubState','-p','MainPID','-p','ExecMainStatus'],env=env,capture_output=True,text=True,check=True)
state=dict(line.split('=',1) for line in p.stdout.splitlines());assert state['ActiveState']=='inactive' and state['MainPID']=='0' and state['ExecMainStatus']=='0'
assert complete['requests']==37 and complete['new_generations']==complete['backwards']==complete['training_updates']==0
assert sha(RUN/'worker.py')==ready['worker_sha256'] and sha(RUN/'controller.py')==ready['controller_sha256'] and sha(RUN/'PROTOCOL.md')==ready['protocol_sha256']
source=Path(ready['source_run'])
for n,h in ready['source_files'].items():assert sha(source/n)==h,n
source_complete=read(source/'COMPLETE.json');source_proof=read(source/'VERIFIED.json');selection=read(source/'selection.json')
assert source_proof['complete'] and source_proof['source_COMPLETE_sha256']==sha(source/'COMPLETE.json')
assert complete['digest']==source_complete['metadata']['pruned']['digest'] and complete['layout']==source_complete['metadata']['pruned']['layout']
assert sha(RUN/'joint_NLL.json')==complete['joint_NLL_sha256']
new=read(RUN/'joint_NLL.json');old=selection['matrix'];assert len(new)==len(old)==37
assert selection['selected']==ready['selected']
asset=ROOT/'.deployment/paired_image_support_20261024_view'
for n,h in ready['runtime_files'].items():assert sha(asset/n)==h,n
policy=read(asset/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json')['conditions']['pct01']
assert sha(asset/policy['manifest'])==policy['manifest_sha256']==source_complete['parent']['source_checkpoint']['manifest_sha256']
manifest=read(asset/policy['manifest']);train={}
for task in manifest['tasks']:
 f=task['files']['train'];assert sha(asset/f['path'])==f['sha256']
 for line in (asset/f['path']).read_text().splitlines():
  row=json.loads(line);assert row['id'] not in train;train[row['id']]=row
write('CONSUMER_STARTED.json',dict(unix=time.time(),consumer_sha256=sha(__file__),complete_sha256=sha(RUN/'COMPLETE.json')))
results=[]
for current,prior in zip(new,old):
 for k in ['id','task','domain','input_sha256','target_tokens']:assert current[k]==prior[k],k
 assert hashlib.sha256(train[current['id']]['answer'].encode()).hexdigest()==current['answer_sha256']
 full=prior['full_NLL'];joint=current['joint_NLL'];single=math.fsum(prior['single_deleted_NLL'][i]-full for i in selection['selected']);j=joint-full;interaction=j-single
 assert all(math.isfinite(x) for x in [full,joint,single,j,interaction])
 results.append(dict(id=current['id'],task=current['task'],domain=current['domain'],parent_NLL=full,joint_NLL=joint,J=j,S=single,I=interaction,absI=abs(interaction)))
domains={}
for d in ['natural','medical','geography','finance','traffic']:
 rows=[x for x in results if x['domain']==d];assert rows
 item={k:math.fsum(x[k] for x in rows)/len(rows) for k in ['J','S','I','absI']}
 item.update(QA=len(rows),positive_I=sum(x['I']>0 for x in rows),negative_I=sum(x['I']<0 for x in rows),net_remainder_exceeds_net_single=abs(item['I'])>abs(item['S']))
 domains[d]=item
macro={k:math.fsum(x[k] for x in domains.values())/5 for k in ['J','S','I','absI']}
out=dict(verified=True,source_COMPLETE_sha256=sha(RUN/'COMPLETE.json'),source_VERIFIED_sha256=ready['source_files']['VERIFIED.json'],source_selection_sha256=ready['source_files']['selection.json'],joint_NLL_sha256=sha(RUN/'joint_NLL.json'),requests=37,domains=domains,macro=macro,per_QA=results,net_remainder_exceeds_net_single=abs(macro['I'])>abs(macro['S']),model_seconds=complete['seconds'],new_generations=0,new_scores=0,backwards=0,training_updates=0,unit_state=state,old_graph_still_quality_negative=True,not_hidden_drift_causality=True,consumer_seconds=time.monotonic()-begin)
write('VERIFIED.json',out);print(json.dumps({k:out[k] for k in ['verified','domains','macro','model_seconds','consumer_seconds']}))
