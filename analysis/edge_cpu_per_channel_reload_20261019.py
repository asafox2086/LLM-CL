"""Fresh process reconstruction of the actual CPU deployment state packages."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'summary/edge_deployment_20261019/cpu_precision/per_channel/v1'
REPORT=STUDY/'fresh_reload'
VIEWS=['all_linear_int8_per_channel']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    temp=Path(str(p)+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(p)


def load_view(view):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch,psutil
    from transformers import AutoConfig,AutoProcessor,LlavaOnevisionForConditionalGeneration
    from edge_cpu_visual_probe_20261019 import input_function
    import edge_cpu_per_channel_20261019 as worker
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    torch.backends.quantized.engine='fbgemm'
    proof_path=STUDY/(view+'_COMPLETE.json');proof=json.loads(proof_path.read_text())
    assert proof['complete'] and proof['source_sha256']==sha(ROOT/'analysis/edge_cpu_per_channel_20261019.py')
    assert not proof['cuda_initialized'] and proof['TEST_QA']==0
    assert not (REPORT/(view+'_STARTED.json')).exists(),'Never silently retry an attempted reload'
    write(REPORT/(view+'_STARTED.json'),dict(pid=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21],source_sha256=sha(__file__)))
    package=(ROOT/proof['package']).resolve();assert package.is_relative_to((ROOT/'exp/edge_cpu_precision_per_channel_20261019/v1').resolve())
    for name,meta in proof['package_files'].items():assert sha(package/name)==meta['sha256']
    spec=json.loads((package/'runtime_spec.json').read_text());assert spec['view']==view and spec['quantized_engine']=='fbgemm'
    process=psutil.Process();before=process.memory_full_info()._asdict();peak=[process.memory_info().rss];stop=threading.Event()
    def observe():
        while not stop.wait(.01):peak[0]=max(peak[0],process.memory_info().rss)
    monitor=threading.Thread(target=observe,daemon=True);monitor.start();start=time.perf_counter()
    # Construct from the exported config only, not the source checkpoint or HF
    # initial pretrained weights; packed INT8 parameters come from this package.
    config=AutoConfig.from_pretrained(package,local_files_only=True)
    model=LlavaOnevisionForConditionalGeneration._from_config(config,torch_dtype=torch.float32).eval()
    names=spec['quantized_modules'];assert len(set(names))==len(names)
    if names:model=torch.ao.quantization.quantize_dynamic(model,set(names),dtype=torch.qint8,inplace=True).eval()
    actual=[n for n,m in model.named_modules() if isinstance(m,torch.ao.nn.quantized.dynamic.Linear)]
    assert set(actual)==set(names)
    saved=torch.load(package/'model_state.pt',map_location='cpu',mmap=True,weights_only=False)
    result=model.load_state_dict(saved,strict=True);assert not result.missing_keys and not result.unexpected_keys
    def equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    state=model.state_dict();assert set(state)==set(saved) and all(equal(state[k],saved[k]) for k in state)
    del state,saved;gc.collect()
    processor=AutoProcessor.from_pretrained(package,local_files_only=True)
    load_seconds=time.perf_counter()-start;stop.set();monitor.join()
    after=process.memory_full_info()._asdict();peak[0]=max(peak[0],process.memory_info().rss)
    prepare=input_function()
    manifest_path=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path)==proof['manifest_sha256'];manifest=json.loads(manifest_path.read_text())
    gates=[]
    for task_id in [1,2,12,16,17]:
        task=manifest['tasks'][task_id-1];meta=task['files']['train'];p=ROOT/meta['path'];assert sha(p)==meta['sha256']
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        row=min(rows,key=lambda r:hashlib.sha256(('edge-profile42:'+r['id']).encode()).digest())
        got=worker.measure(model,processor,prepare,task,row,True)
        reference=next(r for r in proof['requests'] if r['id']==row['id'] and r['speed_only'] and r['repeat']==0)
        exact=got['output_tokens']==reference['output_tokens']
        gates.append(dict(id=row['id'],generation_tokens_exact=exact,new_request=got,
            expected_tokens=reference['output_tokens']))
        write(REPORT/(view+'_partial.json'),dict(complete=False,gates=gates))
        assert exact,'Fresh-runtime output differs; preserve evidence, do not reuse original quality'
    assert not torch.cuda.is_initialized()
    write(REPORT/(view+'_COMPLETE.json'),dict(complete=True,view=view,fresh_process_state_and_tokens_exact=True,
        original_model_process_pid=json.loads((STUDY/(view+'_STARTED.json')).read_text())['pid'],
        reload_process_pid=os.getpid(),source_sha256=sha(__file__),original_complete_sha256=sha(proof_path),
        package_file_hashes_rechecked=True,strict_all_state_entries_exact=True,gates=gates,
        mmap_load=True,load_seconds=load_seconds,load_peak_RSS_bytes=peak[0],before_load=before,after_load=after,
        model_loader_code_bytes=Path(__file__).stat().st_size,CPU_only=True,TEST_QA=0,
        TRAIN_requests=5,not_full_quality=True,not_main_table_admitted=True))


def watch():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (REPORT/'OBSERVER_STARTED.json').exists()
    write(REPORT/'OBSERVER_STARTED.json',dict(pid=os.getpid(),source_sha256=sha(__file__),CPU_only=True))
    # Keep this supplemental work serial with all precision inference. It does
    # not monopolize CPU cores while models are still being measured.
    while not (STUDY/'all_linear_int8_per_channel_COMPLETE.json').exists():
        assert not any((STUDY/n).exists() for n in ['worker_FAILURE.json','CONTROLLER_FAILURE.json','RESUME_FAILURE.json'])
        write(REPORT/'observer_state.json',dict(phase='waiting_per_channel_precision_view',no_model_loaded=True))
        time.sleep(20)
    for view in VIEWS:
        with (REPORT/(view+'_nohup.log')).open('ab') as log:
            child=subprocess.Popen(['/usr/bin/nohup',sys.executable,'-u',str(Path(__file__).resolve()),'--view',view],
                cwd=ROOT,env=os.environ,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            write(REPORT/'observer_state.json',dict(phase='running',view=view,worker_pid=child.pid))
            code=child.wait()
        assert code==0 and (REPORT/(view+'_COMPLETE.json')).exists(),'Keep failure; no blind retry'
    write(REPORT/'ALL_COMPLETE.json',dict(complete=True,views=VIEWS,all_fresh_reload_gates_passed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--view',choices=VIEWS);p.add_argument('--watch',action='store_true');a=p.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True)
    try:
        if a.watch:watch()
        else:load_view(a.view)
    except Exception:
        write(REPORT/((a.view or 'observer')+'_FAILURE.json'),dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
