"""Finish only the two frozen graph evaluations under the original deadline."""
from pathlib import Path
import json,os,subprocess,sys,time,hashlib
ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN'])
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
repair=read(RUN/'EVAL_REPAIR.json');started=read(RUN/'STARTED.json');ready=read(RUN/'READY.json')
assert sha(Path(__file__))==repair['controller_source_sha256'] and sha(RUN/'evaluate.py')==repair['evaluation_source_sha256']
exits={};metadata={}
for name in ('parent','pruned'):
 remaining=repair['absolute_deadline_unix']-time.time();assert remaining>0
 with (RUN/(name+'_eval.log')).open('x') as log:
  p=subprocess.Popen([sys.executable,'-u',str(RUN/'evaluate.py'),name],env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  code=p.wait(timeout=remaining)
 complete=read(RUN/(name+'_EVAL_COMPLETE.json')) if (RUN/(name+'_EVAL_COMPLETE.json')).exists() else None
 receipt=dict(returncode=code,pid=p.pid,unix=time.time(),identity=complete['identity'] if complete else None)
 write(RUN/(name+'_EVAL_EXIT.json'),receipt);assert code==0 and complete
 exits[name]=receipt;metadata[name]=dict(layout=complete['layout'],digest=complete['digest'],DEV=100)
assert time.time()<=repair['absolute_deadline_unix']
process_seconds=repair['locator_unit_elapsed_seconds']+sum(read(RUN/(name+'_EVAL_COMPLETE.json'))['seconds'] for name in ('parent','pruned'))
write(RUN/'COMPLETE.json',dict(complete=True,repaired_execution=True,identity=started['identity'],seconds=time.time()-started['unix'],supervised_process_seconds=process_seconds,ready_sha256=sha(RUN/'READY.json'),selected=repair['selected'],parent=ready['parent'],pruned=repair['packages']['pruned'],metadata=metadata,NLL_requests=1125,new_generations=210,backwards=0,training_updates=0,TEST_QA=0,locator_original_exit_status=1,locator_unit_elapsed_seconds=repair['locator_unit_elapsed_seconds'],evaluation_exits=exits,files={p.name:sha(p) for p in RUN.glob('*.json') if p.name not in ['READY.json','PARENT_EXPORT.json']}))
