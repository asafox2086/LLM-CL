"""Deferred host allocation only; never alter a running training program."""
import datetime
import fcntl
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = 'exp/five_domain_native_cl_20261016_sparse'
REPORT = ROOT / 'summary/five_domain_native_cl_20261016_sparse'
SERVICE = 'llmcl-native-five.service'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i',
       str(ROOT / '.deployment/old_pct01_id_ed25519'), '-p', '8666',
       'liyapeng_grp@10.254.29.8',
       '/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python -']


def read(p):
    return json.loads(p.read_text())


def write(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def remote(code):
    p = subprocess.run(SSH, input=code, text=True, capture_output=True, timeout=240)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:])
    return json.loads(p.stdout)


def service(action):
    subprocess.run(['systemctl', '--user', action, SERVICE], check=True)


def live_ticks(pid):
    p = Path('/proc') / str(pid) / 'stat'
    return p.read_text().split()[21] if p.exists() else None


def old_prepare(expected_sources, archive_name, policy, jobs):
    code = r'''
import json,hashlib,subprocess,sys
from pathlib import Path
R=Path('/data2/liyapeng_grp/program/LLMCL');REL='exp/five_domain_native_cl_20261016_sparse'
D=R/'summary/five_domain_native_cl_20261016_sparse';S='llmcl-native-five.service'
assert subprocess.check_output(['systemctl','--user','show',S,'-p','KillMode','--value'],text=True).strip()=='process'
sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/REL).glob('*.py')}
assert sources==EXPECTED
sys.path.insert(0,str(R/REL));from run import load_protocol
_,manifest,digest,data=load_protocol('full')
images={i for rows in data.values() for row in rows for i in row['images']}
assert all((R/i).is_file() for i in images)
path=D/'old2080_run_status.json';state=json.loads(path.read_text())
def ticks(j):
 p=Path('/proc')/str(j.get('pid'))/'stat'
 return p.read_text().split()[21] if p.exists() else None
before={k:ticks(j) for k,j in state['jobs'].items() if j['status'] in ['training','preflight','reload']}
assert all(before.values())
archive=D/ARCHIVE;archive.mkdir(exist_ok=True)
(archive/'policy_before.json').write_bytes((R/REL/'execution_policy.json').read_bytes())
(archive/'state_before.json').write_bytes(path.read_bytes())
subprocess.run(['systemctl','--user','stop',S],check=True)
try:
 state=json.loads(path.read_text())
 for k,j in JOBS.items():
  assert k not in state['jobs'] or state['jobs'][k]['status']=='waiting'
  state['jobs'][k]=j
 (R/REL/'execution_policy.json').write_text(json.dumps(POLICY,ensure_ascii=False,indent=2)+chr(10))
 path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+chr(10))
finally:subprocess.run(['systemctl','--user','start',S],check=True)
assert before=={k:ticks(state['jobs'][k]) for k in before}
print(json.dumps({'manifest_sha256':digest,'all_full_QA':sum(map(len,data.values())),
 'all_image_paths_present':len(images),'sources_identical':True,'preserved_worker_ticks':before}))
'''
    code = code.replace('EXPECTED', repr(expected_sources)).replace('ARCHIVE', repr(archive_name))
    code = code.replace('JOBS', repr(jobs)).replace('POLICY', repr(policy))
    return remote(code)


def main():
    lock = (REPORT / 'host_handoff.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = REPORT / 'HOST_HANDOFF_20261017.json'
    plan = read(REPORT / 'HOST_HANDOFF_PLAN_20261017.json')
    if status.exists() and read(status).get('phase') == 'completed':
        return
    while True:
        state = read(REPORT / 'new3090_run_status.json')
        job = state['jobs']['full:SeqFT_LoRA']
        assert job['run_dir'] == plan['current_full_run']
        actual = read(ROOT / job['run_dir'] / 'run_status.json')
        if job['status'].startswith('failed'):
            raise RuntimeError('current full job failed; no automatic host handoff')
        if job['status'] != 'completed' or actual.get('status') != 'completed' or live_ticks(job['pid']):
            write(status, {'phase': 'waiting_current_full_completion', 'pid': job['pid'],
                           'actual_worker_ticks': live_ticks(job['pid']),
                           'actual_run_status': actual.get('status'), 'completed_stage': actual.get('completed_stage')})
            time.sleep(30)
            continue
        assert actual['completed_stage'] == 19
        latest = read(ROOT / job['run_dir'] / 'checkpoints/latest.json')
        h = hashlib.sha256()
        with (ROOT / job['run_dir'] / latest['path']).open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(chunk)
        assert h.hexdigest() == latest['sha256']
        break
    policy_path = ROOT / REL / 'execution_policy.json'
    policy = read(policy_path)
    assert policy['hosts']['new3090']['reserved_gpus'] == [0, 1]
    assert subprocess.check_output(['systemctl', '--user', 'show', SERVICE,
                                    '-p', 'KillMode', '--value'], text=True).strip() == 'process'
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / REL).glob('*.py')}
    assert sources == plan['execution_source_sha256']
    archive = REPORT / 'host_handoff_20261017'
    archive.mkdir(exist_ok=True)
    (archive / 'policy_before.json').write_bytes(policy_path.read_bytes())
    (archive / 'state_before.json').write_bytes((REPORT / 'new3090_run_status.json').read_bytes())
    preserve = {str(p): live_ticks(p) for p in plan['preserve_new_worker_pids'] if live_ticks(p)}
    service('stop')
    try:
        state = read(REPORT / 'new3090_run_status.json')
        jobs = {}
        for key in plan['deferred_full_keys']:
            j = state['jobs'][key]
            assert j['status'] == 'waiting' and j['host'] == 'old2080'
            jobs[key] = {'condition': 'full', 'method': j['method'], 'host': 'old2080', 'status': 'waiting',
                         'handoff': 'fresh original preflight/reload/from-base; no prediction reuse'}
        policy['hosts']['new3090']['conditions'] = {'full': [], 'pct01': [2, 3]}
        policy['hosts']['old2080']['conditions'] = {'full': [2], 'pct01': [0, 3]}
        policy['conditions']['full']['allowed_gpus'] = [2]
        policy['conditions']['pct01']['allowed_gpus'] = [0, 2, 3]
        policy['ownership']['old2080'] = list(dict.fromkeys(policy['ownership']['old2080'] + list(jobs)))
        proof = old_prepare(sources, archive.name, policy, jobs)
        write(policy_path, policy)
        for key, value in jobs.items():
            state['jobs'][key] = value
        write(REPORT / 'new3090_run_status.json', state)
    finally:
        service('start')
    assert preserve == {p: live_ticks(p) for p in preserve}
    write(status, {'phase': 'completed', 'completed_at': datetime.datetime.now().astimezone().isoformat(),
                   'full_final_checkpoint_sha256': h.hexdigest(), 'old_admission': proof,
                   'preserved_new_worker_ticks': preserve, 'new_pct01_GPUs': [2, 3],
                   'old_full_GPU': 2, 'old_GPU1': 'new OneVision 1% baseline admission',
                   'new_reserved_GPUs': [0, 1],
                   'old_OOM_full_methods': 'original pipeline requeues on24GB; wait while24GB assigned pct01'})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(REPORT / 'HOST_HANDOFF_20261017.json', {'phase': 'needs_attention',
                                                    'error': repr(exc)})
        raise
