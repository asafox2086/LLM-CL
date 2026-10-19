"""Per-output-channel CPU INT8 control after the all-tensor negative TRAIN result.

All five requests are frozen TRAIN IDs. This is standard quantization, not a
new CL algorithm, a phone benchmark, or permission to reuse original TEST F1.
"""
import argparse
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
REPORT=ROOT/'summary/edge_deployment_20261019/cpu_precision/per_channel/v1'
WORK=ROOT/'exp/edge_cpu_precision_per_channel_20261019/v1'
VIEWS=['all_linear_int8_per_channel']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def memory(process):return process.memory_full_info()._asdict()


def measure(model,processor,prepare,task,row,speed):
    import torch,psutil
    from transformers.generation.streamers import BaseStreamer
    process=psutil.Process();peak=[process.memory_info().rss];stop=threading.Event()
    def observe():
        while not stop.wait(.01):peak[0]=max(peak[0],process.memory_info().rss)
    monitor=threading.Thread(target=observe,daemon=True);monitor.start()
    before=memory(process);start=time.perf_counter()
    with torch.inference_mode():
        batch,shapes=prepare(row,task,processor,'cpu')
        batch['pixel_values']=batch['pixel_values'].float()
        class Clock(BaseStreamer):
            def __init__(self):self.first=True;self.tokens=[];self.times=[]
            def put(self,value):
                if self.first:self.first=False;return
                self.tokens.extend(value.reshape(-1).tolist());self.times.append(time.perf_counter())
            def end(self):pass
        clock=Clock()
        out=model.generate(**batch,max_new_tokens=8 if speed else 64,min_new_tokens=8 if speed else 0,
            do_sample=False,use_cache=True,streamer=clock,
            pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
        elapsed=time.perf_counter()-start
    stop.set();monitor.join();peak[0]=max(peak[0],process.memory_info().rss)
    assert clock.times and (not speed or len(clock.tokens)==8)
    prediction=processor.tokenizer.decode(clock.tokens,skip_special_tokens=True)
    from runtime_utils import score
    return dict(id=row['id'],domain=task['domain'],speed_only=speed,shape=shapes,
        input_tokens=int(batch['input_ids'].shape[1]),TTFT_seconds=clock.times[0]-start,
        decode_tokens_per_second=(len(clock.times)-1)/(clock.times[-1]-clock.times[0]) if len(clock.times)>1 else None,
        total_seconds=elapsed,output_tokens=clock.tokens,
        TRAIN_AnswerF1=None if speed else score(prediction,row.get('references') or [row['answer']])['token_f1'],
        prediction=None if speed else prediction,before_memory=before,after_memory=memory(process),
        generation_peak_RSS_bytes=peak[0],system_loadavg=Path('/proc/loadavg').read_text())


def run(view):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch,psutil,gc
    from transformers import LlavaOnevisionForConditionalGeneration,AutoProcessor
    from edge_cpu_visual_probe_20261019 import input_function
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(42)
    assert 'fbgemm' in torch.backends.quantized.supported_engines
    torch.backends.quantized.engine='fbgemm'
    assert not torch.cuda.is_initialized()
    ready=json.loads((REPORT/'READY.json').read_text())
    assert ready['sources']['analysis/edge_cpu_per_channel_20261019.py']==sha(__file__)
    assert sha(REPORT/'PROTOCOL.md')==ready['protocol_sha256']
    assert not (REPORT/(view+'_STARTED.json')).exists(),'An attempted model is not blindly retried'
    write(REPORT/(view+'_STARTED.json'),dict(pid=os.getpid(),startticks=Path('/proc/self/stat').read_text().split()[21]))
    export_path=ROOT/'summary/edge_deployment_20261019/static_export/v1/OSFT_EXPORTED.json'
    source=json.loads(export_path.read_text());package=ROOT/source['package']
    assert json.loads((export_path.parent/'OSFT_RELOAD.json').read_text())['parameter_digest_exact']
    for name,info in source['package_files'].items():assert sha(package/name)==info['sha256']
    process=psutil.Process();before_load=memory(process);start=time.perf_counter()
    model=LlavaOnevisionForConditionalGeneration.from_pretrained(package,torch_dtype=torch.float32,local_files_only=True).eval()
    processor=AutoProcessor.from_pretrained(package,local_files_only=True)
    named=[(n,m) for n,m in model.named_modules() if type(m) is torch.nn.Linear]
    if view=='decoder_int8':names=[n for n,m in named if n.startswith('language_model.model.')]
    elif view=='all_linear_int8_per_channel':names=[n for n,m in named]
    else:names=[]
    if names:
        model=torch.ao.quantization.quantize_dynamic(model,{n:torch.ao.quantization.per_channel_dynamic_qconfig for n in names},dtype=torch.qint8,inplace=True).eval()
    del named  # Drop audit-only references to replaced FP32 modules.
    converted=[n for n,m in model.named_modules() if isinstance(m,torch.ao.nn.quantized.dynamic.Linear)]
    assert set(converted)==set(names) and len(converted)==327
    schemes={}
    for n,m in model.named_modules():
        if n in names:
            w=m.weight()
            assert w.qscheme()==torch.per_channel_affine and w.q_per_channel_axis()==0
            assert len(w.q_per_channel_scales())==w.shape[0] and torch.isfinite(w.q_per_channel_scales()).all()
            schemes[n]=dict(qscheme=str(w.qscheme()),axis=0,scales=len(w.q_per_channel_scales()))
    del w,m
    gc.collect();after_load=memory(process);load_seconds=time.perf_counter()-start
    # Same PyTorch state_dict container, including actual packed INT8 buffers.
    # Not the old pickle training graph and not an estimated parameter count.
    destination=WORK/view/'deployment';assert not destination.exists();destination.mkdir(parents=True)
    model.config.save_pretrained(destination);processor.save_pretrained(destination)
    torch.save(model.state_dict(),destination/'model_state.pt')
    spec=dict(view=view,quantized_modules=converted,weight_qscheme='per_output_channel',torch=torch.__version__,quantized_engine='fbgemm',
              construct='LlavaOnevision FP32 from config; apply named dynamic qint8; strict-load state',
              original_teacher_checkpoint_sha256=source['checkpoint_sha256'])
    write(destination/'runtime_spec.json',spec)
    # Check saved tensors against live state, including packed weights, exactly.
    saved=torch.load(destination/'model_state.pt',map_location='cpu',weights_only=False)
    def equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    current=model.state_dict();assert current.keys()==saved.keys() and all(equal(current[k],saved[k]) for k in saved)
    del saved,current;gc.collect()
    files={str(p.relative_to(destination)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(destination.rglob('*')) if p.is_file()}
    package_bytes=sum(x['bytes'] for x in files.values())
    prepare=input_function()
    manifest=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    data=json.loads(manifest.read_text());selected=[]
    for task_id in [1,2,12,16,17]:
        task=data['tasks'][task_id-1];info=task['files']['train'];p=ROOT/info['path'];assert sha(p)==info['sha256']
        rows=[json.loads(s) for s in p.read_text().splitlines()]
        row=min(rows,key=lambda r:hashlib.sha256(('edge-profile42:'+r['id']).encode()).digest())
        selected.append((task,row))
    records=[]
    for task,row in selected:
        warm=measure(model,processor,prepare,task,row,True)
        for repeat in range(3):
            r=measure(model,processor,prepare,task,row,True);r['repeat']=repeat;records.append(r)
        quality=measure(model,processor,prepare,task,row,False);records.append(quality)
        write(REPORT/(view+'_partial.json'),dict(complete=False,requests=records))
        print(json.dumps(dict(view=view,id=row['id'],TRAIN_AnswerF1=quality['TRAIN_AnswerF1'],TTFT=records[-2]['TTFT_seconds'])),flush=True)
    result=dict(complete=True,view=view,requests=records,TRAIN_unique_requests=5,TEST_QA=0,
        quantized_modules=converted,quantized_count=len(converted),weight_quantization_schemes=schemes,source_sha256=sha(__file__),
        ready_sha256=sha(REPORT/'READY.json'),manifest_sha256=sha(manifest),teacher_checkpoint_sha256=source['checkpoint_sha256'],
        base_export_sha256=sha(export_path),package=str(destination.relative_to(ROOT)),package_files=files,
        actual_package_bytes=package_bytes,saved_state_exact=True,fresh_process_reload_NOT_verified=True,
        runtime=torch.__version__,engine=torch.backends.quantized.engine,CPU=subprocess.check_output(['lscpu'],text=True),
        threads=2,load_seconds=load_seconds,before_load_memory=before_load,after_load_memory=after_load,
        cuda_initialized=torch.cuda.is_initialized(),CPU_server_proxy=True,not_mobile_measurement=True,
        not_a_new_algorithm=True,not_full_TEST_quality=True,no_main_table_publication=True)
    assert not result['cuda_initialized'];write(REPORT/(view+'_COMPLETE.json'),result)


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    ready=dict(sources={p:sha(ROOT/p) for p in ['analysis/edge_cpu_per_channel_20261019.py','analysis/edge_cpu_visual_probe_20261019.py']},
        protocol_sha256=sha(REPORT/'PROTOCOL.md'),views=VIEWS,threads=2,engine='fbgemm',
        speed_tokens=8,quality_max_new_tokens=64,TRAIN_requests=5,TEST_QA=0,GPU_started=False)
    if (REPORT/'READY.json').exists():assert json.loads((REPORT/'READY.json').read_text())==ready
    else:write(REPORT/'READY.json',ready)


def main():
    p=argparse.ArgumentParser();p.add_argument('--view',choices=VIEWS);p.add_argument('--all',action='store_true');p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare();return
    if a.all:
        assert not (REPORT/'CONTROLLER_STARTED.json').exists()
        write(REPORT/'CONTROLLER_STARTED.json',dict(pid=os.getpid()))
        results={}
        for view in VIEWS:
            code=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--view',view],cwd=ROOT).returncode
            results[view]=code
            if code:write(REPORT/'CONTROLLER_FAILURE.json',dict(view=view,returncode=code,results=results,no_blind_retry=True));return
        write(REPORT/'ALL_COMPLETE.json',dict(views=VIEWS,complete=True,all_views_preserved=True,not_full_quality=True))
    else:run(a.view)


if __name__=='__main__':
    try:main()
    except Exception:
        write(REPORT/((os.environ.get('EDGE_PRECISION_VIEW') or 'worker')+'_FAILURE.json'),dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
