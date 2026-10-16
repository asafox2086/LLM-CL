"""New paired deterministic native-task study; never mix with old default runs.

Waits for existing mechanism queue to finish, then admits only idle new GPU1.
Full first-task weights must be exactly equal across independent/sequence fits.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import fcntl
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'exp/onevision_diagnostics_20261016'
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(ROOT/'analysis'))
from run import freeze, hashed, write
from onevision_research_worker import ResearchExperiment, atomic_torch, digest
from onevision_knowledge_intervention import Intervention, reset_seed, case_complete, valid_checkpoint
from onevision_matched_native_controls import compare_native_history

REPORT = ROOT/'summary/onevision_diagnostics_20261016/deterministic_native'
STATUS = ROOT/'summary/onevision_mechanisms_20261016/deterministic_native_status.json'


def wait_for_admission():
    while True:
        path = ROOT/'summary/onevision_mechanisms_20261016/queue_state.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        if state.get('phase') != 'studies_finished_or_need_attention':
            write(STATUS,{'phase':'waiting_existing_mechanism_queue','pid':os.getpid()})
            time.sleep(30); continue
        old_pid=int(subprocess.check_output(['systemctl','--user','show','llmcl-onevision-research.service','-p','MainPID','--value'],text=True).strip() or '0')
        if old_pid:
            write(STATUS,{'phase':'waiting_original_research_service','research_pid':old_pid,'pid':os.getpid()})
            time.sleep(15);continue
        used,util = map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
        owners = subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if used < 500 and util < 10 and not owners: return
        write(STATUS,{'phase':'waiting_idle_gpu1','owner_pids':owners.splitlines(),'pid':os.getpid()})
        time.sleep(15)


def quarantine_partial_events(run,label):
    path = run/'events.jsonl'
    if not path.exists(): return
    keep=[]; drop=[]
    for line in path.read_text().splitlines():
        try: target = json.loads(line).get('label') == label
        except json.JSONDecodeError: target = True
        (drop if target else keep).append(line)
    if drop:
        folder=run/'interrupted';folder.mkdir(exist_ok=True)
        (folder/f'{label}_{time.time_ns()}.jsonl').write_text('\n'.join(drop)+'\n')
        path.write_text('\n'.join(keep)+ ('\n' if keep else ''))


def checked_events(run,label,qa_count):
    rows=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()
          if json.loads(line).get('label')==label]
    assert len(rows)==math.ceil(qa_count/16) and all(e['learning_rate_used']>0 for e in rows)
    assert rows[-1]['qa_seen']==qa_count
    return hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True);reset_seed()
    manifest,data,domains,tests,old_generation,probes,original=freeze()
    source_files=[Path(__file__),HERE/'run.py',ROOT/'analysis/onevision_research_worker.py',
                  ROOT/'analysis/onevision_knowledge_intervention.py',ROOT/'analysis/onevision_matched_native_controls.py']
    hashes={str(p.relative_to(ROOT)):digest(p) for p in source_files}
    # Predeclare <=10 generation QA per native type from the same diagnostic500.
    generation={}
    for domain,rows in tests.items():
        generation[domain]=[r for t in domains[domain]
                            for r in sorted([q for q in rows if q['_task']==t],key=lambda q:hashed(q,'deterministic-native-generation'))[:10]]
    spec={'experiment':'onevision_deterministic_native_20261016','source_sha256':hashes,
          'manifest_sha256':original['source_manifest_sha256'],'model_revision':original['model_revision'],
          'seed':42,'epochs':1,'rank':8,'alpha':32,'batch':16,
          'backend':'default SDPA with torch.use_deterministic_algorithms(True)',
          'CUBLAS_WORKSPACE_CONFIG':':4096:8','generation_ids':{d:[r['id'] for r in x] for d,x in generation.items()},
          'test_ids':original['diagnostic_test_ids'],'scope':'500 diagnostic QA, not full21454 test',
          'selection':'fixed backend from TRAIN reproducibility/resource audit; no test-based hyperparameter selection',
          'counterfactual':'same own-task QA/order/optimizer/RNG/backend; sequential includes historical exposure'}
    pointer=REPORT/'attempt.json'
    if pointer.exists():
        info=json.loads(pointer.read_text());assert info['spec']==spec;run=ROOT/info['run']
    else:
        run=ROOT/'exp/onevision_deterministic_native_20261016'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
        write(pointer,{'run':str(run.relative_to(ROOT)),'spec':spec});write(run/'protocol.json',spec)
    completed=REPORT/'complete.json'
    if completed.exists():
        proof=json.loads(completed.read_text());assert proof['source_sha256']==hashes
        assert digest(run/'native_history_comparison.json')==proof['comparison_sha256']
        for relative,expected in proof['artifact_sha256'].items(): assert digest(run/relative)==expected
        return
    if (REPORT/'needs_attention.json').exists():return
    wait_for_admission()
    assert shutil.disk_usage(ROOT).free > 8*1024**3, 'Need8GiB free for real gradients and durable states'
    x=Intervention(manifest,run);initial=x.original_initial
    initial_file=run/'initial_adapter.pt'
    if not initial_file.exists():atomic_torch(initial_file,{'adapter':initial,'source_sha256':hashes})
    assert valid_checkpoint(initial_file)
    saved_initial=torch.load(initial_file,map_location='cpu',weights_only=True)['adapter']
    assert all(torch.equal(initial[k][b],saved_initial[k][b]) for k in initial for b in ['A','B'])
    probe=json.loads((ROOT/'summary/onevision_mechanisms_20261016/reproducibility_complete.json').read_text())
    assert probe['results']['deterministic_default_sdpa']['adapter_exact_equal']
    write(run/'runtime.json',{'torch':str(torch.__version__),'cuda':torch.version.cuda,
          'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
          'gpu':torch.cuda.get_device_name(0),'short_probe_reference':probe['run'],
          'short_probe_is_not_full_task_gate':True})

    def fit(label,start,rows):
        x.configure(start,finetune_shared=True);path=run/'adapters'/(label+'.pt')
        if valid_checkpoint(path):
            saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['source_sha256']==hashes
            assert checked_events(run,label,len(rows))==saved['events_sha256'], 'Checkpoint lacks matching complete durable training log'
            x.restore(saved['adapter']);return saved['adapter']
        quarantine_partial_events(run,label);x.cache.clear();reset_seed()
        adapter=x.train(label,rows)
        event_hash=checked_events(run,label,len(rows))
        with (run/'events.jsonl').open('rb') as stream:os.fsync(stream.fileno())
        atomic_torch(path,{'adapter':adapter,'source_sha256':hashes,'rank':8,'alpha':32,
                          'events_sha256':event_hash,
                          'train_ids_in_order':[r['id'] for r in sorted(rows,key=lambda r:hashed(r,'training'))]})
        return adapter

    def evaluate(label,selected,generated):
        if not case_complete(run,label,selected):x.evaluate(label,selected,generated)

    def gradients_complete(label,selected):
        path=run/'gradients'/(label+'.json')
        if not path.exists():return False
        info=json.loads(path.read_text())
        for domain in selected:
            raw=run/'raw_probe_gradients'/label/(domain+'.pt')
            if not raw.exists() or digest(raw)!=info['raw_sample_files'][domain]['sha256']:return False
        return True

    x.configure(initial,finetune_shared=True);evaluate('base_initial',tests,generation)
    previous=initial
    for task in manifest['tasks']:
        tid,domain=task['task'],task['domain'];native_generation={domain:[r for r in generation[domain] if r['_task']==tid]}
        assert native_generation[domain]
        selected={domain:tests[domain]}
        write(STATUS,{'phase':'deterministic_paired_native','native_task':tid,'run':str(run.relative_to(ROOT)),'pid':os.getpid()})
        x.configure(previous,finetune_shared=True);evaluate(f'pre_T{tid:02d}',selected,native_generation)
        current=fit(f'sequential_T{tid:02d}',previous,data[tid,'train'])
        evaluate(f'sequential_T{tid:02d}',tests if tid==19 else selected,generation if tid==19 else native_generation)
        if not gradients_complete(f'sequential_T{tid:02d}',probes):
            ResearchExperiment.gradients(x,f'sequential_T{tid:02d}',probes)
        isolated=fit(f'matched_native_T{tid:02d}',initial,data[tid,'train'])
        if tid==1:
            error=max(float((current[k][b]-isolated[k][b]).abs().max()) for k in current for b in ['A','B'])
            exact=all(torch.equal(current[k][b],isolated[k][b]) for k in current for b in ['A','B'])
            write(run/'full_T01_reproduction_gate.json',{'exact_equal':exact,'max_abs_difference':error,
                  'own_training_QA':len(data[1,'train']),'backend':spec['backend'],'old_failed_gate_not_overwritten':True})
            assert exact, 'Deterministic full-T01 pair still differs; stop before later-task causal comparisons'
        evaluate(f'matched_native_T{tid:02d}',selected,native_generation)
        if not gradients_complete(f'matched_native_T{tid:02d}',{domain:probes[domain]}):
            ResearchExperiment.gradients(x,f'matched_native_T{tid:02d}',{domain:probes[domain]})
        previous=current
        write(run/'progress.json',{'completed_paired_tasks':tid,'source_sha256':hashes})
    compare_native_history(run,run,manifest,tests)
    # Helper writes its historical report location; also retain versioned result.
    legacy=ROOT/'summary/onevision_mechanisms_20261016/MATCHED_NATIVE.md'
    (REPORT/'MATCHED_NATIVE.md').write_text('确定性配对实验；不包含旧默认计算结果。\n\n'+legacy.read_text())
    x.configure(previous,finetune_shared=True);x.monitor.snapshot('training_after',model=x.model);x.monitor.finish(x.model)
    artifacts=[run/'native_history_comparison.json',run/'full_T01_reproduction_gate.json',run/'memory_summary.json',
               run/'initial_adapter.pt',run/'events.jsonl',run/'resources.jsonl',run/'protocol.json',run/'progress.json']
    artifacts+=list((run/'adapters').glob('*.pt'))+list((run/'raw_probe_gradients').rglob('*.pt'))
    artifacts+=list((run/'evaluations').rglob('*.jsonl'))
    write(completed,{'complete':True,'run':str(run.relative_to(ROOT)),'source_sha256':hashes,
          'comparison_sha256':digest(run/'native_history_comparison.json'),
          'artifact_sha256':{str(p.relative_to(run)):digest(p) for p in artifacts},'formal_full_test':False})
    write(STATUS,{'phase':'deterministic_paired_native_complete','run':str(run.relative_to(ROOT))})


if __name__=='__main__':
    try:main()
    except BlockingIOError:raise SystemExit(0)
    except Exception:
        # Hard gate/OOM/source failures need investigation; do not retry blindly.
        REPORT.mkdir(parents=True,exist_ok=True)
        write(REPORT/'needs_attention.json',{'traceback':traceback.format_exc(),'pid':os.getpid(),
              'action':'Preserve run; explicit audited repair/new attempt required. No gate relaxed.'})
        write(STATUS,{'phase':'deterministic_native_needs_attention','report':str(REPORT.relative_to(ROOT))})
