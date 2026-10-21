"""Bounded, isolated PC engineering admission. Never auto-retry a model request."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / '.deployment/pc_cpu_admission_20261022/v1'
REPORT = ROOT / 'summary/edge_deployment_20261019/pc_cpu_admission_20261022/v1'
PY = Path.home() / 'AppData/Local/Programs/Python/Python311/python.exe'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-o', 'HostKeyAlias=[100.87.141.22]:8666',
       '-i', str(Path.home() / '.ssh/10_254_25_46_liyapeng_id_rsa'), '-p', '8666', 'liyapeng@100.87.141.22']
SCP = ['scp', '-B', '-o', 'ConnectTimeout=8', '-o', 'HostKeyAlias=[100.87.141.22]:8666',
       '-i', str(Path.home() / '.ssh/10_254_25_46_liyapeng_id_rsa'), '-P', '8666']
VERSIONS = {'torch': '2.6.0+cpu', 'transformers': '4.46.3', 'safetensors': '0.8.0',
            'accelerate': '1.2.1', 'tokenizers': '0.20.3', 'numpy': '1.26.4', 'Pillow': '12.3.0', 'psutil': '7.2.2'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, obj):
    p = REPORT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(p)


def command(args, log, timeout, env=None):
    with (PRIVATE / log).open('xb') as out:
        p = subprocess.run([str(x) for x in args], stdout=out, stderr=subprocess.STDOUT,
                           timeout=timeout, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
    assert p.returncode == 0, f'{log}: exit {p.returncode}; preserve log, no automatic retry'


def environment():
    envpath = PRIVATE / 'venv'
    command([PY, '-m', 'venv', envpath], 'venv.log', 120)
    py = envpath / 'Scripts/python.exe'
    command([py, '-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '0', '--timeout', '30',
             'torch==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cpu'], 'pip_torch.log', 1200)
    pins = [f'{n}=={v}' for n, v in VERSIONS.items() if n != 'torch']
    command([py, '-m', 'pip', 'install', '--disable-pip-version-check', '--retries', '0', '--timeout', '30',
             *pins], 'pip_runtime.log', 1200)
    out = subprocess.check_output([str(py), '-m', 'pip', 'freeze', '--all'], timeout=30)
    (PRIVATE / 'requirements.lock.txt').write_bytes(out)
    return py


def transfer():
    remote = r'''
import base64,hashlib,json
from pathlib import Path
r=Path('/data1/liyapeng/program/LLMCL')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
report=r/'summary/edge_deployment_20261019/structured_width_20261020/v1'
prepare=json.loads((report/'prepare_COMPLETE.json').read_text())
ready=json.loads((report/'READY.json').read_text())
assert sha(report/'prepare_COMPLETE.json')==expected_prepare_sha
assert sha(report/'READY.json')==prepare['ready_sha256']
package=prepare['packages']['dense'];p=r/package['package']
for n,x in package['files'].items():
 assert (p/n).stat().st_size==x['bytes'] and sha(p/n)==x['sha256'],n
selected=next(x for x in ready['selected'] if x['role']=='calibration' and x['domain']=='medical')
assert selected['id']=='SLAKE:5162'
policy=json.loads((r/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json').read_text())
manifest_rel=policy['conditions']['pct01']['manifest']
assert sha(r/manifest_rel)==ready['input_source_sha256'][manifest_rel]
manifest=json.loads((r/manifest_rel).read_text())
t=next(t for t in manifest['tasks'] if t['task']==selected['task'])
f=t['files']['train'];assert sha(r/f['path'])==f['sha256']==ready['input_source_sha256'][f['path']]
row=next(json.loads(line) for line in (r/f['path']).read_text().splitlines() if json.loads(line)['id']==selected['id'])
assert row['id']==selected['id']
files={}
runner=r/'exp/edge_teacher_transfer_20261019/onevision_osft_final/runner_source'
for n in ['runtime_utils.py','onevision_bridge.py','frozen_vision.py','retrieval.py']:
 data=(runner/n).read_bytes();assert sha(runner/n)==ready['teacher_runner_sources'][n]
 files['runner_source/'+n]=base64.b64encode(data).decode()
for rel in row['images']:
 assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
 assert sha(r/rel)==ready['input_source_sha256'][rel]
 files[rel]=base64.b64encode((r/rel).read_bytes()).decode()
print(json.dumps({'package':package,'row':row,'task':{'task':t['task'],'instruction':t['instruction']},
 'reference_NLL':prepare['reference_NLL'],'original_prepare_sha256':expected_prepare_sha,'data':files}))
'''
    original = ROOT / 'summary/edge_deployment_20261019/structured_width_20261020/v1/prepare_COMPLETE.json'
    code = 'expected_prepare_sha=' + repr(sha(original)) + '\n' + remote
    result = subprocess.run(SSH + ['/data1/liyapeng/program/LLMCL/.assets-env/bin/python', '-'],
                            input=code.encode(), capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr.decode()
    data = json.loads(result.stdout)
    inputs = {}
    for rel, value in data.pop('data').items():
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        p = PRIVATE / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(base64.b64decode(value))
        inputs[rel] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    data['inputs'] = inputs
    (PRIVATE / 'bundle.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    incoming = PRIVATE / 'incoming'
    incoming.mkdir()
    command(SCP + ['-r', 'liyapeng@100.87.141.22:/data1/liyapeng/program/LLMCL/' + data['package']['package'], incoming],
            'transfer.log', 1800)
    p = incoming / 'dense'
    assert sorted(str(f.relative_to(p)).replace('\\', '/') for f in p.rglob('*') if f.is_file()) == sorted(data['package']['files'])
    for n, info in data['package']['files'].items():
        assert (p/n).stat().st_size == info['bytes'] and sha(p/n) == info['sha256'], n
    p.rename(PRIVATE / 'package')
    write('TRANSFER.json', {'package_bytes': data['package']['actual_package_bytes'],
                             'files': data['package']['files'], 'inputs': inputs,
                             'bundle_sha256': sha(PRIVATE / 'bundle.json'), 'all_SHA_exact': True})


def main():
    PRIVATE.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source = ROOT / 'analysis/pc_cpu_admission_20261022.py'
    write('ENGINEERING_LAUNCH.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'bootstrap_sha256': sha(__file__), 'source_sha256': sha(source),
        'protocol_sha256': sha(REPORT/'PROTOCOL.md'), 'versions': VERSIONS,
        'CPU_only': True, 'no_global_environment_changes': True, 'no_automatic_retry': True})
    assert shutil.disk_usage(ROOT).free > 20 * 1024**3
    with ThreadPoolExecutor(max_workers=2) as executor:
        env = executor.submit(environment)
        package = executor.submit(transfer)
        py = env.result()
        package.result()
    assert shutil.disk_usage(ROOT).free > 15 * 1024**3
    write('READY.json', {'source_sha256': sha(source), 'protocol_sha256': sha(REPORT/'PROTOCOL.md'),
         'bundle_sha256': sha(PRIVATE/'bundle.json'), 'versions': VERSIONS,
         'environment_lock_sha256': sha(PRIVATE/'requirements.lock.txt'),
         'runtime_disk_bytes': sum(p.stat().st_size for p in (PRIVATE/'venv').rglob('*') if p.is_file()),
         'setup_seconds': time.monotonic() - start})
    environment_vars = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                            OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                            PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    command([py, '-u', source], 'model.log', 1800, environment_vars)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('BOOTSTRAP_FAILURE.json', {'traceback': traceback.format_exc(), 'source_sha256': sha(__file__)})
        raise
