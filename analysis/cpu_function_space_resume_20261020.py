"""One-shot CPU v2 continuation; never restarts an attempted model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'summary/edge_deployment_20261019/cpu_function_space_20261020/v2'
V1=D.parent/'v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(n,v):
    p=D/n;tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    s=(p/'stat').read_text();a=s[s.rfind(')')+2:].split()
    return {'pid':pid,'startticks':a[19],'cmdline':(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()}


def spawn(source,log):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    with (D/log).open('xb') as f:
        p=subprocess.Popen(['nohup','nice','-n','10',str(ROOT/'.vision-env/bin/python'),'-u',str(source)],
                           cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    result=identity(p.pid);assert result is not None;return result


def main():
    admission=json.loads((D/'ADMISSION.json').read_text())
    assert admission['controller_sha256']==sha(__file__)
    for rel,h in admission['frozen_files'].items():assert sha(ROOT/rel)==h
    assert not (D/'CONTROLLER_STARTED.json').exists(),'One controller only; inspect existing job before recovery'
    write('CONTROLLER_STARTED.json',identity(os.getpid()))
    gate=D/'repair_gate'
    while True:
        if (gate/'FAILURE.json').exists():raise RuntimeError('Repair gate failed; no model launch')
        if (gate/'COMPLETE.json').exists():break
        live=identity(admission['gate_identity']['pid'])
        assert live and live['startticks']==admission['gate_identity']['startticks'],'Gate handle missing/reused without proof'
        write('CONTROLLER_STATE.json',{'phase':'waiting_context_gate','gate_identity':live,'CPU_only':True})
        time.sleep(30)
    c=json.loads((gate/'COMPLETE.json').read_text());e=json.loads((gate/'EVIDENCE.json').read_text())
    assert c['status']=='CPU_CONTEXT_REPAIR_ALL168_EXACT' and c['operators']==168
    assert c['evidence_sha256']==sha(gate/'EVIDENCE.json') and c['source_sha256']==sha(ROOT/'analysis/cpu_function_context_gate_20261020.py')
    assert len(e['rows'])==168 and all(x['original_vs_matching_context_exact'] for x in e['rows'])
    for n in ['READY.json','LAUNCH.json','STARTED.json','FAILURE.json','COMPLETE.json']:
        assert not (D/n).exists(),'Existing v2 artifact; no blind retry: '+n
    ready=json.loads((V1/'READY.json').read_text())
    ready.update(source_sha256=sha(ROOT/'analysis/cpu_function_space_20261020_v2.py'),
        protocol_sha256=sha(D/'PROTOCOL.md'),inherited_protocol_sha256=sha(V1/'PROTOCOL.md'),
        repair_protocol_sha256=sha(D/'REPAIR_PROTOCOL.md'),repair_gate_evidence_sha256=c['evidence_sha256'],
        selection_and_controls_AST_exact_v1=True,CPU_controls=admission['CPU_controls'])
    write('READY.json',ready)
    launch=spawn(ROOT/'analysis/cpu_function_space_20261020_v2.py','run.log');write('LAUNCH.json',launch)
    while True:
        if (D/'FAILURE.json').exists():raise RuntimeError('CPU v2 model failed; preserve evidence, no retry')
        if (D/'COMPLETE.json').exists():
            p=subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/'analysis/cpu_function_space_report_20261020_v2.py')],
                             cwd=ROOT,capture_output=True,text=True)
            (D/'report.log').write_text(p.stdout+p.stderr)
            assert p.returncode==0,'Independent report failed; no model retry'
            write('CONTROLLER_STATE.json',{'phase':'complete_verified','proof_sha256':sha(D/'COMPLETE.json'),
                'report_sha256':sha(D/'VERIFIED_RESULTS.json'),'CPU_only':True});return
        live=identity(launch['pid'])
        assert live and live['startticks']==launch['startticks'],'CPU model handle missing/reused without terminal proof'
        write('CONTROLLER_STATE.json',{'phase':'model_running','worker_identity':live,'CPU_only':True})
        time.sleep(30)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write('NEEDS_ATTENTION.json',{'error':traceback.format_exc(),'no_blind_retry':True})
        raise
