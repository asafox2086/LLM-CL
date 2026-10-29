"""Independent authorized S17-recovery proof; unchanged original 74-cell F1 engine.
CLI --root ABS --binding ABS. Never calls the failed original formal verifier
main. Reuses its pinned method_records/acquisition/config checks unchanged.
Original failure stays failure; only this branch receives independent outputs.
"""
import argparse,hashlib,importlib.util,json,math,os,time,traceback
from pathlib import Path
BASE_SHA='5a911bd3936921c17095df29b0afbc46760ea4da9cc0ad57fb1604ff1fb2659d'
RECOVERY_SHA='c3aa7fb71b423ec1bbae151a56632501eda0b912464d532c9446955be51419ea'
CP_SHA='939b7081ad009f387cc142b9baffcbfb6ad1e5e031bc3b1d47463771e7c74ed4'
OLD='exp/task_gradient_formal_20261028/runs/onevision/pct01/TaskGradReuse/training_v1'
BRANCH='exp/task_gradient_formal_20261028/runs/onevision/pct01/TaskGradReuse/training_recovery_S17_20261029'
FRAME='exp/task_gradient_formal_20261028'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def terminal(root,binding,base,files):
 b=files.json(binding);bh=sha(binding);report=root/'summary/runtime_recovery_20261029'
 old=root/OLD;new=root/BRANCH;frame=root/FRAME
 base.check(b['schema']=='task-gradient-formal-v1' and b['method']=='TaskGradReuse','original binding')
 base.check(b['inherited_sources']['analysis/task_gradient_formal_verify_20261028.py']==BASE_SHA,'pinned original validator')
 sources=b['sources'];base.check(len(sources)==29 and {p.name for p in frame.glob('*.py')}==set(sources),'unchanged29 source inventory')
 for name,h in sources.items():files.raw(frame/name,h)
 for relative,h in b['inherited_sources'].items():files.raw(root/relative,h)
 files.raw(frame/'execution_policy.json',b['execution_policy_sha256']);files.raw(b['protocol']['path'],b['protocol']['sha256'])
 files.raw(report/'recover_taskgrad_s17.py',RECOVERY_SHA)
 base.check(not (report/'FAILURE.json').exists(),'recovery failure remains closed')
 prepared=files.json(report/'PREPARED.json');ev=prepared['checkpoint_identity']
 base.check(prepared['prepared'] and prepared['source_sha256']==RECOVERY_SHA and prepared['branch']==BRANCH
  and prepared['original_run']==OLD and prepared['starting_completed_stage']==17
  and prepared['next_stage']==18 and prepared['prefix_cells']==53
  and prepared['checkpoint_full_sha_verified'],'authorized committed S17 branch')
 failure=files.json(root/'run_v1/FORMAL_FAILURE.json',ev['original_failure_sha256'])
 slot=files.json(root/'summary/task_gradient_formal_20261028/SLOT.json',ev['original_slot_sha256'])
 base.check(failure['no_retry'] and failure['binding_sha256']==ev['binding_sha256']==bh
  and slot['status']=='failed_closed' and slot['released'],'original failed state preserved')
 original_start=files.json(root/'run_v1/FORMAL_STARTED.json');protocol=b['protocol']
 for value in (original_start,failure):
  base.check(value['GPU']==1 and value['source_sha256']==sources and value['protocol']==protocol
   and value['binding_sha256']==bh and value['no_retry'] and value['formal_fresh_W0'],'original fresh source/start binding')
 base.check(original_start['identity']==failure['identity'],'original controller identity')
 base.check((original_start['planned_stages'],original_start['TRAIN_QA'],original_start['TEST_QA'],
  original_start['planned_updates'],original_start['calibration_backwards'],original_start['evaluation_cells'])
  ==(19,1313,21454,93,222,74),'original finite work plan')
 base.check(original_start['preflight_weights_not_reused_for_training'],'original training fresh W0')
 original_phases=failure['phases']
 base.check([x['phase'] for x in original_phases]==['preflight','reload','training'],'original native phase chain')
 preflight=root/base.PREFLIGHT
 for item in original_phases[:2]:
  phase=item['phase'];base.gone(item['identity'])
  base.check(item['returncode']==0 and not item['timeout'] and item['identity_absent'],'original preflight/reload real exit0')
  outer=files.json(root/'run_v1'/('PHASE_'+phase+'_STARTED.json'))
  phase_exit=files.json(root/'run_v1'/('PHASE_'+phase+'_EXIT.json'))
  base.check(phase_exit=={k:v for k,v in item.items() if k not in ('proof_sha256','checkpoint')},'original successful phase EXIT receipt')
  base.check(outer['identity']==item['identity'] and outer['phase']==phase and outer['GPU']==1
   and outer['native_run']==item['native_run'] and Path(item['native_run'])==preflight,'original successful phase identity/run')
  base.config_contract(files.json(preflight/'config.json'),sources,True)
  proof_path=preflight/('gpu_preflight.json' if phase=='preflight' else 'checkpoint_reload_proof.json')
  phase_proof=files.json(proof_path,item['proof_sha256'])
  base.check(phase_proof['passed'] and phase_proof['execution_source_sha256']==sources,'original complete preflight/reload proof')
  if phase=='preflight':base.check(phase_proof['task_ids']==list(range(1,20)),'original19 preflight tasks')
 files.raw(root/'analysis/answer_f1.py',base.METRIC_SHA)
 metric_lock=files.json(root/'summary/primary_metric_20261017/metric_lock.json')
 base.check(metric_lock['metric_sha256']==base.METRIC_SHA and not metric_lock['option_mapping']
  and not metric_lock['additional_prefix_or_answer_extraction'],'original locked raw F1 definition')
 for who in [failure['identity']]+[x['identity'] for x in failure['phases']]:base.gone(who)
 if failure.get('last_attempted_worker'):base.gone(failure['last_attempted_worker'])
 for rel,entry in prepared['copied_files'].items():
  # New events/resources append; latest advances to S19. All other prefix
  # evidence remains byte-for-byte copied from the committed original stages.
  if rel in ('logs/events.jsonl','resources.jsonl','checkpoints/latest.json'):continue
  files.raw(new/rel,entry['sha256'])
 meta=files.json(new/'checkpoints/stage_17.json')
 base.check(meta['stage']==17 and meta['sha256']==ev['checkpoint_sha256']==CP_SHA
  and meta['bytes']==ev['checkpoint_bytes']==1928308132,'original S17 full state anchor')
 cp=new/meta['path'];base.check(cp.stat().st_size==meta['bytes'],'S17 branch asset retained')
 cfg=files.json(new/'config.json',ev['original_config_sha256']);base.config_contract(cfg,sources,False)
 old_raw=files.raw(old/'logs/events.jsonl',prepared['copied_files']['logs/events.jsonl']['original_sha256'])
 prefix=[];old_events=[]
 for line in old_raw.splitlines(keepends=True):
  row=json.loads(line);base.check(row['event']=='train_update','original native train events')
  old_events.append(row)
  if row['stage']<=17:prefix.append(line)
 prefix=b''.join(prefix);base.check(hashlib.sha256(prefix).hexdigest()==prepared['copied_files']['logs/events.jsonl']['sha256'],'exact committed event prefix')
 new_raw=files.raw(new/'logs/events.jsonl');base.check(new_raw.startswith(prefix),'unchanged committed events before replay')
 new_events=[json.loads(x) for x in new_raw[len(prefix):].splitlines()]
 base.check(new_events and all(x['event']=='train_update' and x['stage'] in (18,19) for x in new_events),'only remaining stages added')
 resource=prepared['copied_files']['resources.jsonl']
 old_resources=files.raw(old/'resources.jsonl',resource['sha256'])
 base.check(files.raw(new/'resources.jsonl').startswith(old_resources),'resource history retained; multiple PIDs')
 started=files.json(report/'STARTED.json');ex=files.json(report/'EXIT.json');end=files.json(report/'COMPLETE.json')
 base.check(started['source_sha256']==RECOVERY_SHA and started['prepared_sha256']==sha(report/'PREPARED.json')
  and end['prepared_sha256']==sha(report/'PREPARED.json'),'actual continuation source/start')
 base.check(started['worker']==ex['worker']==end['worker'] and started['supervisor']==ex['supervisor']==end['supervisor'],'continuation identities')
 for who in (ex['worker'],ex['supervisor']):base.gone(who)
 base.check(ex['returncode']==end['returncode']==0 and ex['identity_absent'] and end['identity_absent']
  and end['complete'] and end['branch']==BRANCH and end['original_failure_preserved']
  and started['GPU']==ex['GPU']==end['GPU']==1,'real continuation exit0')
 status=files.json(new/'run_status.json');latest=files.json(new/'checkpoints/latest.json')
 base.check(status['status']=='completed' and status['completed_stage']==19 and latest['stage']==19
  and end['final_checkpoint']==latest and end['new_scores']==0,'final native completion')
 return dict(prepared_sha256=sha(report/'PREPARED.json'),complete_sha256=sha(report/'COMPLETE.json'),
  final_checkpoint=latest,original_failure_sha256=ev['original_failure_sha256'],
  retained_observed_main_updates=len(old_events)-len([x for x in old_events if x['stage']>17])+len(new_events),
  original_failed_updates=len(old_events),retained_prefix_updates=len([x for x in old_events if x['stage']<=17]),
  discarded_failed_stage_updates=len([x for x in old_events if x['stage']>17]),
  new_continuation_updates=len(new_events),actual_observed_updates_including_failed_work=len(old_events)+len(new_events),
  resource_scope='Original plus continuation PIDs; no subtraction as a single-process memory/time scope',
  original_failed_work_and_partial_predictions_preserved=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--binding',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert a.root.is_absolute() and a.binding.is_absolute()
 base_path=root/'analysis/task_gradient_formal_verify_20261028.py';assert sha(base_path)==BASE_SHA
 base=load(base_path,'unchanged_task_gradient_method_validator');base.RUN=BRANCH
 report=root/'summary/runtime_recovery_20261029';assert report.is_dir()
 markers=('VERIFY_STARTED.json','VERIFY_FAILURE.json','VERIFIED.json','RESULT.json')
 assert not any((report/n).exists() for n in markers)
 own=sha(__file__);begin=time.monotonic();files=base.Files();invoked=False
 base.write(report/'VERIFY_STARTED.json',dict(unix=time.time(),pid=os.getpid(),source_sha256=own,planned_existing_answer_scores=80316,unique_branch_claim=True,original_failure_not_relabelled=True))
 try:
  chain=terminal(root,a.binding,base,files);manifest=files.json(root/base.MANIFEST,base.MANIFEST_SHA)
  base.check([t['task'] for t in manifest['tasks']]==list(range(1,20)) and manifest['condition']=='pct01' and manifest['seed']==42
   and sum(t['files']['train']['count'] for t in manifest['tasks'])==1313
   and sum(t['files']['test']['count'] for t in manifest['tasks'])==21454,'original data')
  cells=[(s,t) for s in range(20) for t in (range(1,20) if s in (0,19) else (s,s+1))]
  base.check(len(cells)==74 and sum(manifest['tasks'][t-1]['files']['test']['count'] for _,t in cells)==80316,'unchanged evaluation workload')
  base.check(chain['retained_observed_main_updates']==93,'actual retained native train_update count93')
  method=base.method_records(root,files,manifest)  # unchanged19/222/93/48fold checks
  engine_path=root/'analysis/v2_verified_baseline_report.py';files.raw(engine_path,base.ENGINE_SHA)
  engine=load(engine_path,'unchanged_recovery_v2_engine')
  base.check(engine.SOURCE_SHA==base.ENGINE_SHA,'original engine')
  engine.ROOT=root;engine.FRAME=root/FRAME;engine.METHODS=['TaskGradReuse'];engine.BACKBONES=['onevision']
  invoked=True;proof=engine.verify(BRANCH,'old2080')
  base.check(proof['rescored_cells']==74 and proof['final_QA']==21454 and proof['positive_finite_training_stages']==19
   and proof['checkpoint_sha256']==chain['final_checkpoint']['sha256'],'original complete proof')
  acquired=base.acquisition(root,proof,manifest,files);files.unchanged();base.check(sha(__file__)==own,'source unchanged')
  result=dict(status='RECOVERED_FORMAL_QUALITY_VERIFIED_RESOURCES_PENDING',method='TaskGradReuse',condition='pct01',backbone='onevision',
   original_formal_proof=proof,acquired=acquired,method_validation=method,recovery_chain=chain,
   domains=proof['domains'],domain_macro_final=proof['domain_macro_final'],aggregate=proof['aggregate'],final_native19=proof['final_native19'],
   source_sha256=own,base_validator_sha256=BASE_SHA,engine_sha256=base.ENGINE_SHA,binding_sha256=sha(a.binding),source_metadata=files.records,
   original_run_still_failed=True,resources_pending=True,new_models=0,new_generations=0,original_answer_score_checks=80316,
   limits=['Authorized same-source S17 full-method/RNG continuation, not uninterrupted execution.',
    'Failed S18 work and costs remain in provenance; retained trajectory has original93 main updates/222 calibration QA backwards.',
    'Rtt includes earlier transfer and current learning; not pure direction causality or one-time joint state.',
    'Resources require this actual recovered final graph; old local measurements cannot substitute.'])
  base.write(report/'RESULT.json',result)
  base.write(report/'VERIFIED.json',dict(status='RECOVERED_VERIFIED_ONCE_RESOURCES_PENDING',unix=time.time(),verifier_sha256=own,
   original_validator_sha256=BASE_SHA,engine_sha256=base.ENGINE_SHA,binding_sha256=sha(a.binding),result_sha256=sha(report/'RESULT.json'),
   recovery_complete_sha256=chain['complete_sha256'],original_formal_cells=74,existing_answer_score_checks=80316,
   extra_acquisition_scores=0,main_optimizer_calls_retained=93,calibration_backwards_retained=222,original_run_still_failed=True,
   retained_actual_train_update_count=chain['retained_observed_main_updates'],actual_observed_updates_including_failed_work=chain['actual_observed_updates_including_failed_work'],resources_pending=True,new_models=0,new_generations=0,seconds=time.monotonic()-begin))
  print(json.dumps(dict(verified=str(report/'VERIFIED.json'),new_scores_in_this_verifier=80316)))
 except BaseException:
  base.write(report/'VERIFY_FAILURE.json',dict(unix=time.time(),source_sha256=own,traceback=traceback.format_exc(),engine_invoked=invoked,
   scoring_checks_completed_unknown_if_engine_failed=invoked,original_failure_preserved=True,no_automatic_retry=True))
  raise

if __name__=='__main__':main()
