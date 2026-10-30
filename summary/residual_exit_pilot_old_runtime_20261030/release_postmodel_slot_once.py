"""One explicit post-model SLOT publication repair; never starts/retries a model.
Default is read-only. Preserve original failure, controller, COMPLETE and CLAIM bytes.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,time
D=Path('/data2/liyapeng_grp/program/LLMCL/summary/residual_exit_pilot_old_runtime_20261030');O=D/'CONTROLLER_V1'
BINDING='3d5655b8fad20ab3db9487f5be2d992b925e27c1b6377f87cd3b8c2934b839ef'
FAILURE='f134a48ece8999664f9a940da17661e3b59f7392abfe27362d80c96c64f24541'
COMPLETE='522e714fa2b4e6a25c2b62d4a72d7a01c0d47dbc4d071e919b538396eee77f66'
OLD_SLOT='9b067418f687163c5f3cbd61aa03adbbd9d677343e81205ec00fd7d9b4b6c2ac'
TERMINAL='ef9d2c0c66bad652d360732993320ad774fd997676b84be6fe3aa1fcf8a2fe6a'
RESULTS={'PASs_MoE':'d1e8c183070903bea2625da450c08732a948b21a7e31a4150d828eec87b09bac','CL_MoE':'4d69cd1e2f867c48b434ae724543377ad053048ea73d072d2e974d5f013ae39f'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def gone(w):
 try:
  p=Path('/proc')/str(w['pid']);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return s[19]!=str(w['startticks']) or p.stat().st_uid!=w['uid']
 except FileNotFoundError:return True

def saved(p,raw):
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def encoded(v):return (json.dumps(v,indent=2,allow_nan=False)+chr(10)).encode()
def validate():
 assert os.getuid()==1084 and not (D/'PILOT_READOUT_STARTED.json').exists() and not (D/'PILOT_PAIRED_SUMMARY.json').exists()
 inputs={}
 def check(p,h=None):
  digest=sha(p);assert h is None or digest==h,(str(p),digest,h);inputs[str(p)]=digest;return read(p)
 b=check(D/'BINDING.json',BINDING);assert b['admitted'] is True
 adm=check(b['admission_path']);assert adm['approved'] and adm['binding_sha256']==BINDING
 assert sha(D/'controller.py')==adm['controller_sha256']=='b722ee806e0e10ee3694592855ff484912e046e7e9068a4afd1502004c4ed390'
 inputs[str(D/'controller.py')]=sha(D/'controller.py');assert sha(D/'readout.py')=='70a071dff899d50cc79b70bd8adb21b31d25b6bde41e402561ee1b8b88ac927a';inputs[str(D/'readout.py')]=sha(D/'readout.py')
 for name,h in b['sources'].items():assert sha(Path(b['source_dir'])/name)==h;inputs[str(Path(b['source_dir'])/name)]=h
 for path,h in [(b['protocol_path'],b['protocol_sha256']),(b['asset_receipt'],b['asset_receipt_sha256'])]:assert sha(path)==h;inputs[str(path)]=h
 t=check(D/'ORIGINAL_POSTMODEL_CONTROLLER_TERMINAL_20261030.json',TERMINAL)
 assert t['original_controller_exit_status']==1 and t['original_controller_gone'] is True and t['both_model_EXIT0_and_gone'] is True
 assert t['systemd_fields']['MainPID']=='0' and t['systemd_fields']['ExecMainStatus']=='1' and t['systemd_fields']['ActiveState']=='failed'
 assert gone(t['original_controller'])
 fail=check(O/'FAILURE.json',FAILURE);complete=check(O/'COMPLETE.json',COMPLETE);claim=check(O/'CLAIM.json',OLD_SLOT)
 oldraw=(O/'SLOT.json').read_bytes();assert hashlib.sha256(oldraw).hexdigest()==OLD_SLOT and read(O/'SLOT.json')==claim
 assert claim['binding_sha256']==complete['binding_sha256']==BINDING and claim['released'] is False and claim['status']=='claimed'
 assert claim['methods']==['PASs_MoE','CL_MoE'] and claim['identity']==t['original_controller'] and gone(claim['identity'])
 assert fail['last_worker'] is None and fail['last_worker_gone'] and fail['slot_requires_explicit_terminal_audit']
 assert fail['traceback'].endswith('AssertionError\n') and "line 89, in main" in fail['traceback'] and "assert current==claim" in fail['traceback']
 assert complete['complete'] and complete['released'] and complete['phases']==fail['phases']==t['model_phases']
 assert complete['GPU']==claim['GPU']==2 and complete['gpu_uuid']==claim['gpu_uuid']==b['gpu_uuid']
 assert len(complete['phases'])==2
 for phase in complete['phases']:
  method=phase['method'];assert method in RESULTS and phase['returncode']==0 and phase['worker_gone'] and gone(phase['identity'])
  assert check(O/(method+'_EXIT.json'))==phase
  run=Path(b['output_root'])/method;assert not (run/'FAILURE.json').exists();c=check(run/'COMPLETE.json');assert c['identity']==phase['identity'] and c['binding_sha256']==BINDING and 0<c['seconds']<=7200
  r=check(run/'RESULT.json',RESULTS[method]);assert c['RESULT_sha256']==RESULTS[method] and r['method']==method and r['teacher_unchanged'] and r['not_quality_verification']
  counts=r['counts'];assert counts['checkpoint_loads']==1 and counts['teacher_main_forward']==100 and counts['student_calibration_forward']==counts['student_readout_forward']==80
  assert all(counts[k]==0 for k in ['generations','CE','F1','TEST_QA','backwards','optimizer_updates','new_checkpoints'])
  assert r['fit_sha256']==sha(run/'FIT.json') and r['readout_sha256']==sha(run/'READOUT.jsonl') and r['temporary_resources_sha256']==sha(run/'TEMPORARY_RESOURCES.json')
  for n in ['FIT.json','READOUT.jsonl','TEMPORARY_RESOURCES.json']:inputs[str(run/n)]=sha(run/n)
 # The failure is the JSON tuple->list equality bug, after both successful phases.
 assert json.loads(json.dumps(dict(methods=('PASs_MoE','CL_MoE'))))['methods']==claim['methods']
 return b,inputs,oldraw

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
 b,inputs,oldraw=validate()
 if not a.execute:print(json.dumps(dict(status='READY_FOR_EXPLICIT_RELEASE_AUDIT',model_runs=0,consumer_runs=0,controller_exit_status_preserved=1)));return
 assert not (D/'RELEASE_RECOVERY_STARTED.json').exists() and not (D/'RELEASE_AUDIT.json').exists() and not (O/'SLOT_BEFORE_RELEASE_RECOVERY.json').exists()
 lease=(O/'lease.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 b,inputs,currentraw=validate();assert currentraw==oldraw
 saved(D/'RELEASE_RECOVERY_STARTED.json',encoded(dict(binding_sha256=BINDING,repair_source_sha256=sha(__file__),original_controller_exit_status=1,unix=time.time(),no_model_retry=True)))
 saved(O/'SLOT_BEFORE_RELEASE_RECOVERY.json',oldraw);assert sha(O/'SLOT_BEFORE_RELEASE_RECOVERY.json')==OLD_SLOT
 current=json.loads(oldraw);current.update(status='complete',released=True,terminal_path=str(O/'COMPLETE.json'),terminal_sha256=COMPLETE,release_unix=time.time(),release_recovery=True,release_audit_path=str(D/'RELEASE_AUDIT.json'))
 newraw=encoded(current)
 audit=dict(schema='fixed-exit-postmodel-release-audit-v1',binding_sha256=BINDING,repair_source_sha256=sha(__file__),original_controller_exit_status=1,original_controller_failure_sha256=FAILURE,original_controller_complete_sha256=COMPLETE,original_slot_sha256=OLD_SLOT,archived_original_slot_path=str(O/'SLOT_BEFORE_RELEASE_RECOVERY.json'),published_slot_sha256=hashlib.sha256(newraw).hexdigest(),original_inputs_sha256=inputs,original_failure_preserved_in_place=True,original_controller_complete_preserved_in_place=True,reason='Only post-model SLOT publication failed: METHODS tuple becomes a list in JSON before current==claim. Both original model phases already EXIT0 and gone.',model_retries=0,controller_retries=0,new_models=0,quality_or_scalar_rows_modified=False,publication_only_valid_if_actual_slot_matches_published_sha=True,unix=time.time())
 saved(D/'RELEASE_AUDIT.json',encoded(audit));assert all(sha(p)==h for p,h in inputs.items())
 assert (O/'SLOT.json').read_bytes()==oldraw,'SLOT CAS guard changed'
 tmp=O/'SLOT.release_recovery.tmp';saved(tmp,newraw);assert (O/'SLOT.json').read_bytes()==oldraw;os.replace(tmp,O/'SLOT.json');fd=os.open(O,os.O_DIRECTORY);os.fsync(fd);os.close(fd)
 assert sha(O/'SLOT.json')==audit['published_slot_sha256'] and all(sha(p)==h for p,h in inputs.items())
 print(json.dumps(dict(status='RELEASED_AFTER_EXPLICIT_POSTMODEL_AUDIT',audit_sha256=sha(D/'RELEASE_AUDIT.json'),slot_sha256=sha(O/'SLOT.json'),original_controller_exit_status=1,new_models=0,consumer_runs=0)))
if __name__=='__main__':main()
