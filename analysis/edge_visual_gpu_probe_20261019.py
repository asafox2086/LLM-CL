"""TRAIN-only GPU cost/quality feasibility on the sealed strongest teacher.

No new training, TEST selection, or formal-quality claim. All declared views
are attempted. Standard static export is a baseline optimization, not novelty.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/edge_deployment_20261019/gpu_visual_probe/v1'
TEACHER = ROOT/'exp/edge_teacher_transfer_20261019/onevision_osft_final'
VIEWS = ['static_dense', 'exact_view_reuse', 'uniform50', 'author_divprune50']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def exact_indices(pixels):
    """Compare raw CPU pixels, including dtype/shape; never merge near images."""
    assert pixels.device.type == 'cpu' and pixels.ndim == 4
    unique, inverse, seen = [], [], {}
    for i, image in enumerate(pixels):
        key = hashlib.sha256(image.contiguous().numpy().tobytes()).digest()
        if key not in seen:
            seen[key] = len(unique); unique.append(i)
        else:
            assert __import__('torch').equal(image, pixels[unique[seen[key]]])
        inverse.append(seen[key])
    return unique, inverse


def checks():
    import torch
    x = torch.arange(3*2*2).float().reshape(3,1,2,2)
    a,b = exact_indices(x); assert a == b == [0,1,2]
    y = x[[1,1,2,1,2]]; a,b = exact_indices(y)
    assert a == [0,2] and b == [0,0,1,0,1] and torch.equal(y[a][b], y)
    z = y.clone(); z[1,0,0,0] += .0001
    a,b = exact_indices(z); assert len(a) == 3 and torch.equal(z[a][b], z)
    assert not torch.cuda.is_initialized()
    write(REPORT/'CPU_CHECK.json', dict(checks=3, tensor_scatter_exact=True,
        approximate_pixel_merging=False, GPU_started=False))


def selections():
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = json.loads(manifest_path.read_text()); selected = []
    for task_id in [1,2,12,16,17]:
        task = manifest['tasks'][task_id-1]; info = task['files']['train']
        path = ROOT/info['path']; assert sha(path) == info['sha256']
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        row = min(rows, key=lambda r: hashlib.sha256(('edge-profile42:'+r['id']).encode()).digest())
        selected.append((task,row))
    return manifest_path, selected


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    checks()
    manifest, selected = selections()
    ready = dict(sources={rel: sha(ROOT/rel) for rel in [
        'analysis/edge_visual_gpu_probe_20261019.py', 'analysis/edge_cpu_visual_probe_20261019.py',
        'analysis/onevision_research_device_entry.py', 'code/DivPrune/source_manifest.json']},
        protocol_sha256=sha(REPORT/'PROTOCOL.md'), manifest_sha256=sha(manifest),
        teacher_proof_sha256=sha(ROOT/'summary/edge_deployment_20261019/teacher_transfer/READY.json'),
        views=VIEWS, TRAIN_ids=[r['id'] for _,r in selected], TEST_QA=0,
        repetitions=3, forced_speed_tokens=8, TRAIN_quality_max_new_tokens=64,
        physical_GPU=1, scientific_sources_FROZEN=True, GPU_model_NOT_started=True)
    if (REPORT/'READY.json').exists(): assert json.loads((REPORT/'READY.json').read_text()) == ready
    else: write(REPORT/'READY.json', ready)


def run():
    # Guard before importing torch/CUDA, independently of the CPU scheduler.
    import onevision_research_device_entry as device
    inventory = device.inventory()
    assert device.select(inventory, os.getuid()) == 1
    assert not inventory[1]['owners'] and inventory[1]['memory_MiB'] < 500 and inventory[1]['utilization'] < 10
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '6'
    ready = json.loads((REPORT/'READY.json').read_text())
    for rel,h in ready['sources'].items(): assert sha(ROOT/rel) == h
    assert sha(REPORT/'PROTOCOL.md') == ready['protocol_sha256']
    assert not (REPORT/'STARTED.json').exists(), 'Do not blindly rerun an attempted experiment'
    write(REPORT/'STARTED.json', dict(pid=os.getpid(), startticks=Path('/proc/self/stat').read_text().split()[21], inventory=inventory))
    import torch
    import psutil
    from transformers.generation.streamers import BaseStreamer
    from edge_cpu_visual_probe_20261019 import author_selector
    torch.set_num_threads(6); torch.set_num_interop_threads(1)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    proof = json.loads((ROOT/'summary/edge_deployment_20261019/teacher_transfer/READY.json').read_text())
    checkpoint = TEACHER/'checkpoints/stage_19.pt'
    assert sha(checkpoint) == proof['checkpoint_sha256']
    config = json.loads((TEACHER/'config.json').read_text())
    for rel,h in config['execution_source_sha256'].items(): assert sha(TEACHER/'runner_source'/rel) == h
    sys.path.insert(0,str(TEACHER/'runner_source'))
    from runtime_utils import score
    import onevision_bridge
    # Keep the original common image processor without frozen-vision cache.
    # This helper only changes the teacher source location, not its AST.
    import ast, math
    from PIL import Image
    import retrieval
    source = TEACHER/'runner_source/runtime_utils.py'
    fn = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'input_batch')
    cut = next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Import) and any(a.name=='onevision_bridge' for a in n.names))
    fn.body = fn.body[:cut]+[ast.Return(ast.Tuple(elts=[ast.Name(id='batch',ctx=ast.Load()),ast.Name(id='image_shapes',ctx=ast.Load())],ctx=ast.Load()))]
    env = dict(Image=Image,math=math,ROOT=ROOT,retrieve=retrieval.retrieve)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(source),'exec'),env)
    prepare_input = env['input_batch']
    payload = torch.load(checkpoint,map_location='cpu',pickle_module=pickle,weights_only=False)
    assert payload['stage'] == 19
    method = payload['method']; bridge = method.model
    core = bridge.core.cuda().eval(); bridge.cuda().eval()
    processor = payload['processor']; onevision_bridge.bind(bridge)
    proc = psutil.Process(); selector = author_selector()
    manifest, rows = selections()
    assert sha(manifest) == ready['manifest_sha256']
    original_feature = core.get_image_features
    records = []; original = []

    def call(row, task, view, tokens, speed):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        before = dict(cpu=proc.memory_full_info()._asdict(), allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        start = time.perf_counter()
        batch, shapes = prepare_input(row,task,processor,'cpu')
        pixels = batch['pixel_values']; assert pixels.shape[0] == 1
        indices, inverse = exact_indices(pixels[0])
        batch = {k:v.cuda() for k,v in batch.items()}
        encode_start = time.perf_counter()
        features = original_feature(batch['pixel_values'].half(), batch['image_sizes'],
            core.config.vision_feature_layer, core.config.vision_feature_select_strategy) if view != 'exact_view_reuse' else None
        if view == 'exact_view_reuse':
            x = batch['pixel_values'][0,indices].half()
            output = core.vision_tower(x,output_hidden_states=True)
            f = output.hidden_states[core.config.vision_feature_layer]
            if core.config.vision_feature_select_strategy == 'default': f = f[:,1:]
            elif core.config.vision_feature_select_strategy != 'full': raise AssertionError('unknown vision strategy')
            f = core.multi_modal_projector(f)
            features = (f[inverse],)
        features,_ = core.pack_image_features(features,batch['image_sizes'],core.image_newline,core.config.vision_aspect_ratio)
        torch.cuda.synchronize(); encode_seconds = time.perf_counter()-encode_start
        original_visual = len(features)
        special = (features==core.image_newline.to(features.dtype).unsqueeze(0)).all(dim=1)
        structural = special.nonzero().flatten(); content = (~special).nonzero().flatten()
        sel_start = time.perf_counter()
        if view in ['uniform50','author_divprune50']:
            keep = round(len(content)*.5)
            if view == 'uniform50': keep_indices = torch.linspace(0,len(content)-1,keep,device='cuda').round().long()
            else:
                keep_indices,_ = selector.DivPrune(features[content].float(),len(content),threshold_ratio=.5)
                assert keep_indices.unique().numel() == len(keep_indices), 'Author duplicate-index corner; preserve failure'
            chosen = torch.cat([content[keep_indices],structural]).sort().values
            features = features[chosen]
        torch.cuda.synchronize(); selection_seconds = time.perf_counter()-sel_start
        ids = batch['input_ids']; token = core.config.image_token_index
        pos = (ids[0]==token).nonzero().flatten(); assert int(pos[-1]-pos[0]+1)==len(pos)
        begin,end = int(pos[0]),int(pos[-1])+1
        oldtext = ids[ids!=token].clone()
        ids = torch.cat([ids[:,:begin], torch.full((1,len(features)),token,device='cuda',dtype=ids.dtype), ids[:,end:]],dim=1)
        assert torch.equal(oldtext,ids[ids!=token])
        embeds = core.get_input_embeddings()(ids)
        embeds = embeds.masked_scatter((ids==token).unsqueeze(-1).expand_as(embeds),features.to(embeds.dtype))
        class Clock(BaseStreamer):
            def __init__(self): self.first=True; self.tokens=[]; self.times=[]
            def put(self,value):
                if self.first: self.first=False; return
                torch.cuda.synchronize(); self.times.append(time.perf_counter()); self.tokens.extend(value.reshape(-1).tolist())
            def end(self): pass
        clock = Clock()
        out = core.language_model.generate(input_ids=ids,inputs_embeds=embeds,attention_mask=torch.ones_like(ids),
            do_sample=False,use_cache=True,max_new_tokens=tokens,min_new_tokens=tokens if speed else 0,
            streamer=clock,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        answer = processor.tokenizer.decode(clock.tokens,skip_special_tokens=True)
        result = dict(id=row['id'],domain=task['domain'],view=view,visual_before=original_visual,visual_after=len(features),
            input_views=len(inverse),unique_views=len(indices),vision_seconds=encode_seconds,selection_seconds=selection_seconds,
            TTFT_seconds=clock.times[0]-start,seconds=elapsed,output_tokens=clock.tokens,
            decode_tokens_per_second=(len(clock.times)-1)/(clock.times[-1]-clock.times[0]) if len(clock.times)>1 else None,
            memory_before=before,cpu_after=proc.memory_full_info()._asdict(),
            GPU_allocated_after=torch.cuda.memory_allocated(),GPU_reserved_after=torch.cuda.memory_reserved(),
            GPU_peak_allocated=torch.cuda.max_memory_allocated(),GPU_peak_reserved=torch.cuda.max_memory_reserved(),
            TRAIN_AnswerF1=None if speed else score(answer,row.get('references') or [row['answer']])['token_f1'],
            answer=answer if not speed else None,speed_only=speed,feature_sha256=hashlib.sha256(features.cpu().contiguous().numpy().tobytes()).hexdigest())
        return result

    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.float16):
        # Original dynamic execution is retained as the proper numerical reference.
        for task,row in rows:
            original.append(call(row,task,'original_dynamic',8,True))
            write(REPORT/'original_dynamic_partial.json',original)
        replacements=[]
        for name,layer in list(core.named_modules()):
            if type(layer).__name__ != 'OSFTLinear': continue
            w=layer.kernel['reconstruct_weight_matrix'](layer.svd(),torch.float32,layer.output_dtype).detach()
            dense=torch.nn.Linear(w.shape[1],w.shape[0],bias=layer.bias is not None,device='cuda',dtype=w.dtype)
            dense.weight=torch.nn.Parameter(w,requires_grad=False)
            if layer.bias is not None: dense.bias=torch.nn.Parameter(layer.bias.detach().clone(),requires_grad=False)
            x=torch.randn(2,w.shape[1],device='cuda',dtype=w.dtype)*.1
            a,b=layer(x),dense(x)
            replacements.append(dict(module=name,exact=torch.equal(a,b),max_abs=float((a.float()-b.float()).abs().max())))
            assert torch.equal(a,b),'GPU static operator differs from original arithmetic'
            parent,child=name.rsplit('.',1);setattr(core.get_submodule(parent),child,dense)
        assert len(replacements)==168
        method.layers.clear(); del layer,a,b,w,x
        import gc;gc.collect();torch.cuda.empty_cache()
        write(REPORT/'GPU_STATIC_OPERATORS.json',dict(replacements=replacements,CPU_export_not_used=True))
        for view in VIEWS:
            for task,row in rows:
                # One explicit warmup, then three timed requests; no hidden feature cache.
                warm = call(row,task,view,8,True)
                if view=='static_dense':
                    old=next(r for r in original if r['id']==row['id'])
                    assert warm['output_tokens']==old['output_tokens'] and warm['feature_sha256']==old['feature_sha256']
                for repeat in range(3):
                    r=call(row,task,view,8,True);r['repeat']=repeat;records.append(r)
                    write(REPORT/'requests_partial.json',dict(complete=False,requests=records))
                quality=call(row,task,view,64,False);records.append(quality)
                write(REPORT/'requests_partial.json',dict(complete=False,requests=records))
                print(json.dumps(dict(view=view,id=row['id'],TRAIN_AnswerF1=quality['TRAIN_AnswerF1'],visual=quality['visual_after'])),flush=True)
    write(REPORT/'COMPLETE.json',dict(complete=True,requests=records,original_dynamic=original,
        TRAIN_requests=5,TEST_QA=0,views=VIEWS,teacher_sha256=proof['checkpoint_sha256'],
        readiness_sha256=sha(REPORT/'READY.json'),source_sha256=sha(__file__),
        GPU_name=torch.cuda.get_device_name(0),torch=torch.__version__,physical_GPU=1,
        server_consumer_GPU_proxy=True,not_phone_measurement=True,not_full_AnswerF1=True,
        no_main_table_publication=True,no_model_updates=True,all_declared_views_retained=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--gpu',choices=['1']);a=p.parse_args()
    if a.prepare:prepare()
    else:
        assert a.gpu=='1'
        try:run()
        except Exception:
            write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
