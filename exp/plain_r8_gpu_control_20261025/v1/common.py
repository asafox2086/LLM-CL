"""Fixed original data and GPU transport for the plain r8 Q/V baseline."""
from pathlib import Path
import gc,hashlib,importlib.util,json,math,os,random,signal,sys,time

ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=Path(os.environ['LLMCL_ASSET'])
LIMITS={'train_qv':3600,'eval_qv':600}
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(n,v):
    with (RUN/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v

class Context:
    def __init__(self,phase):
        self.phase=phase;self.ready=read(RUN/'READY.json');self.rh=sha(RUN/'READY.json')
        for n,h in self.ready['sources'].items():assert sha(RUN/n)==h,n
        for n,h in self.ready['environment'].items():assert sha(n)==h,n
        for n,h in self.ready['prior_files'].items():assert sha(RUN/n)==h,n
        for n,h in self.ready['runtime_files'].items():
            if not n.endswith('.safetensors'):assert sha(ASSET/n)==h,n
        self.gpu=int(os.environ['LLMCL_GPU']);ids=module('ids_only',ASSET/'exp/paired_image_support_20261024/v1/common.py')
        card=ids.inventory()[self.gpu];assert not card['owners'] and card['memory_MiB']<500 and card['utilization']<10
        self.who=ids.identity(os.getpid());assert self.who['pid']==self.who['pgid'];self.tick=time.monotonic()
        write(phase+'_STARTED.json',dict(unix=time.time(),identity=self.who,GPU=self.gpu,ready_sha256=self.rh,limit=LIMITS[phase],boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
        def expired(*args):raise TimeoutError('Fixed plain r8 phase expired: '+phase)
        signal.signal(signal.SIGALRM,expired);signal.alarm(LIMITS[phase])
        os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
        import torch,numpy as np
        torch.set_num_threads(8);torch.cuda.set_device(self.gpu);self.device=f'cuda:{self.gpu}'
        sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'));sys.path.insert(0,str(ASSET/'exp/osft_increment_screen_20261019/v1'))
        import study,onevision_bridge as bridge,runtime_utils as rt,frozen_vision as fv
        self.study,self.bridge,self.rt,self.fv=study,bridge,rt,fv
        self.storage=module('spatial_storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py')
        assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==self.ready['storage_helper_sha256']
        from transformers import AutoProcessor
        self.base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV';self.processor=AutoProcessor.from_pretrained(self.base,local_files_only=True)
        self.stop=dict(generation_stop_ids=self.processor.tokenizer.eos_token_id,pad_id=self.processor.tokenizer.pad_token_id,context_limit=None)
        path,self.manifest,original=study.original_train();self.split,self.fit,self.dev=study.split_train(self.manifest,original)
        assert sha(path)==self.ready['manifest_sha256'] and sha(study.REPORT/'SPLIT.json')==self.ready['split_sha256']
        assert self.split==read(study.REPORT/'SPLIT.json')
        self.rows=[(t,r) for t in self.manifest['tasks'] for r in self.dev[t['task']]];assert len(self.rows)==100
        self.reset();fv.LIMIT=0
        # Common transport policy: keep already-computed FP16 visual features
        # on GPU, without the legacy CPU staging round trip or visual cache.
        # No vision/decoder arithmetic or precision is changed.
        def gpu_prepare(batch,row,model,original=None):
            pixels=batch.get('pixel_values')
            if pixels is None or pixels.ndim==2:return False
            assert not any(p.requires_grad for p in model.visual.parameters())
            extractor=original or model._llmcl_visual_original;dtype=model.visual.get_dtype()
            with torch.no_grad(),torch.autocast('cuda',dtype=dtype):features=extractor(pixels.to(dtype),batch['image_grid_thw']).detach()
            assert features.is_cuda and features.ndim==2 and features.shape[-1]==896 and torch.isfinite(features).all()
            model._llmcl_visual_cache_misses+=1;batch['pixel_values']=features
            assert model._llmcl_visual_cache_bytes==0
            return False
        fv.prepare=gpu_prepare

    def reset(self):
        import numpy as np,torch
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)

    def bind(self,model):
        self.bridge.bind(model);self.bridge.LAST_NATIVE=None;self.bridge.NATIVE_GATE_MODE=False
        if not hasattr(model,'_llmcl_visual_original'):self.fv.install(model)
        assert self.fv.LIMIT==0

    def end(self,proof,files):
        value=dict(identity=self.who,ready_sha256=self.rh,files={p.relative_to(RUN).as_posix():sha(p) for p in files},**proof)
        value['seconds']=time.monotonic()-self.tick;assert value['seconds']<=LIMITS[self.phase]
        write(self.phase+'_COMPLETE.json',value);signal.alarm(0)

class Batcher:
    def __init__(self,ctx,model,phase):self.ctx,self.model,self.phase=ctx,model,phase;self.records=[];self.checked=set();self.last_pixels={}
    def __call__(self,*args):
        # Original training input_batch(row,task,processor,device) signature.
        row,task=args[:2];ctx=self.ctx
        for n in row['images']:
            if n not in self.checked:assert sha(ASSET/n)==ctx.split['image_sha256'][n];self.checked.add(n)
        ctx.bridge.bind(self.model);batch,details=ctx.rt.input_batch(row,task,ctx.processor,self.model.device)
        context=row.get('context','');indexes=[]
        if len(context)>6000:
            from retrieval import retrieve
            chunks=[context[i:i+1000] for i in range(0,len(context),800)];indexes=sorted(retrieve(row['question'],chunks,5));context='\n'.join(chunks[i] for i in indexes)
        assert indexes==details['retrieved_chunk_indexes']
        contract=dict(images=[dict(path=n,sha256=ctx.split['image_sha256'][n]) for n in row['images']],instruction=task['instruction'],question=row['question'],original_context=row.get('context',''),resolved_context=context,retrieved_chunk_indexes=indexes,image_shapes=details['image_shapes'],common_image_area_cap=401408)
        details.update(raw_input_sha256=hashlib.sha256(json.dumps(contract,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            transport_sha256=ctx.storage.digest({n:v for n,v in batch.items() if n!='pixel_values'}),image_SHA256=contract['images'])
        record=dict(details,id=row['id'],task=task['task'],domain=task['domain'],phase=self.phase)
        self.records.append(record);return batch,details
    def readout(self,model,task,row):assert model is self.model;return self(row,task)
