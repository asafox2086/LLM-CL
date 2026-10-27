"""One native preflight, reload and fresh pct01 formal run on an allocated GPU.

No GPU selection, baseline signals, scoring or automatic model retry. Formal
work ends at the original finite dataset/stage boundary, not an old FIT timer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8',newline='\n') as f:
        json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        if s[0]=='Z':return None
        return dict(pid=pid,startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
    except (OSError,ValueError,IndexError):return None
def stop_own(child,expected):
    if identity(child.pid)==expected:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if identity(child.pid)==expected:os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=10)
    assert identity(expected['pid'])!=expected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--binding',type=Path,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--python',type=Path,required=True)
    a=p.parse_args();root=a.root;here=Path(__file__).resolve().parent;out=root/'run_v1'
    assert os.getuid()==1084 and a.gpu==1 and root.is_absolute() and root.is_dir()
    assert here==root/'exp/task_gradient_formal_20261028' and not out.exists()
    assert a.python.is_absolute() and a.python.is_file() and a.binding.is_absolute()
    b=read(a.binding);binding_sha=sha(a.binding)
    assert b['schema']=='task-gradient-formal-v1' and b['method']=='TaskGradReuse'
    sources={p.name:sha(p) for p in here.glob('*.py')};assert sources==b['sources']
    for relative,expected in b['inherited_sources'].items():
        rel=Path(relative);assert not rel.is_absolute() and '..' not in rel.parts
        assert sha(root/rel)==expected
    assert sha(here/'BUILD_RECEIPT.json')==b['build_receipt_sha256']
    protocol=b['protocol'];assert sha(protocol['path'])==protocol['sha256']
    mp=root/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(mp)==b['manifest_sha256']=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    m=read(mp);assert sum(t['files']['train']['count'] for t in m['tasks'])==1313
    assert sum(t['files']['test']['count'] for t in m['tasks'])==21454
    assert sum((t['files']['train']['count']+15)//16 for t in m['tasks'])==93
    assert sum(min(16,t['files']['train']['count']) for t in m['tasks'])==222
    assert b['training_plan']==[dict(stage=t['task'],TRAIN_QA=t['files']['train']['count'],
        updates=(t['files']['train']['count']+15)//16,calibration_QA=min(16,t['files']['train']['count']),
        training_order_sha256=t['training_order_sha256']) for t in m['tasks']]
    policy=read(here/'execution_policy.json');assert sha(here/'execution_policy.json')==b['execution_policy_sha256']
    assert policy['conditions']['pct01']['manifest_sha256']==b['manifest_sha256']
    assert policy['hosts']['old2080']['backbone']=='onevision'
    assert a.gpu in policy['hosts']['old2080']['conditions']['pct01']
    owner=identity(os.getpid());assert owner and owner['pid']==owner['pgid']
    # The one original allocator has already claimed this exact empty card.
    owners=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    assert not owners.strip()
    out.mkdir(exist_ok=False)
    start=time.monotonic();phases=[];active=None;last=None;child=None
    shared=dict(identity=owner,GPU=1,source_sha256=sources,protocol=protocol,
        binding_sha256=binding_sha,no_retry=True,formal_fresh_W0=True)
    write(out/'FORMAL_STARTED.json',dict(shared,unix=time.time(),planned_stages=19,
        TRAIN_QA=1313,TEST_QA=21454,planned_updates=93,calibration_backwards=222,
        evaluation_cells=74,preflight_weights_not_reused_for_training=True,
        training_wallclock_policy='finite original formal work; not the old FIT deadline'))
    def interrupted(signum,frame):raise InterruptedError('Formal supervisor interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    env=dict(os.environ,LLMCL_ROOT=str(root),LLMCL_HOST='old2080',LLMCL_BACKBONE='onevision',
        CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    base=here/'runs/onevision/pct01/TaskGradReuse'
    preflight=base/'preflight_v1';training=base/'training_v1'
    try:
        for phase,destination,limit in [('preflight',preflight,1800),('reload',preflight,600),('training',training,None)]:
            assert {p.name:sha(p) for p in here.glob('*.py')}==sources
            assert sha(a.binding)==binding_sha and sha(protocol['path'])==protocol['sha256']
            if phase!='reload':destination.mkdir(parents=True,exist_ok=False)
            if phase=='reload':command=[str(a.python),'-u',str(here/'checkpoint_probe.py'),'--run-dir',str(destination)]
            else:
                command=[str(a.python),'-u',str(here/'run.py'),'--method','TaskGradReuse','--condition','pct01','--run-dir',str(destination)]
                if phase=='preflight':command.append('--preflight-only')
            launched=time.time();clock=time.monotonic();timeout=False
            with (destination/'supervisor.log').open('a') as log:
                child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active=last=identity(child.pid);assert active and active['uid']==1084 and active['pid']==active['pgid']
                write(out/('PHASE_'+phase+'_STARTED.json'),dict(unix=launched,phase=phase,identity=active,
                    GPU=1,native_run=str(destination),command=command,wall_limit_seconds=limit))
                try:code=child.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    timeout=True;stop_own(child,active);code=child.returncode
            item=dict(phase=phase,identity=active,returncode=code,identity_absent=identity(child.pid)!=active,
                timeout=timeout,seconds=time.monotonic()-clock,native_run=str(destination))
            write(out/('PHASE_'+phase+'_EXIT.json'),item);phases.append(item)
            assert code==0 and not timeout and item['identity_absent'],phase+' failed; no automatic retry'
            config=read(destination/'config.json')
            assert config['method']=='TaskGradReuse' and config['condition']=='pct01'
            assert config['backbone']=='onevision' and config['execution_source_sha256']==sources
            assert config['manifest_sha256']==b['manifest_sha256'] and config['checkpoint_all_stages']
            assert config['test_fraction']==1 and config['train_fraction']==.01
            if phase in ('preflight','reload'):
                proof_path=destination/('gpu_preflight.json' if phase=='preflight' else 'checkpoint_reload_proof.json')
                proof=read(proof_path);assert proof['passed'] and proof['execution_source_sha256']==sources
                if phase=='preflight':assert proof['task_ids']==list(range(1,20)) and config['preflight_only']
                item['proof_sha256']=sha(proof_path)
            else:
                status=read(destination/'run_status.json');cp=read(destination/'checkpoints/latest.json')
                assert not config['preflight_only'] and status['status']=='completed' and status['completed_stage']==19
                assert cp['stage']==19 and cp['method']=='TaskGradReuse'
                assert (destination/cp['path']).stat().st_size==cp['bytes']
                item['checkpoint']=cp
            active=child=None
        write(out/'FORMAL_COMPLETE.json',dict(shared,unix=time.time(),phases=phases,
            active_worker=None,last_attempted_worker=last,training_run=str(training),
            preflight_run=str(preflight),seconds=time.monotonic()-start,new_verification_scores=0,
            original_native_production_scores_in_training=True,postcompletion_resource_measurement_pending=True))
    except BaseException:
        error=traceback.format_exc()
        if child is not None and active is not None:
            try:stop_own(child,active)
            except BaseException:error+='\nOwn cleanup: '+traceback.format_exc()
        write(out/'FORMAL_FAILURE.json',dict(shared,unix=time.time(),phases=phases,
            active_worker=identity(active['pid']) if active else None,last_attempted_worker=last,
            seconds=time.monotonic()-start,traceback=error,complete_quality_missing=True))
        raise


if __name__=='__main__':main()
