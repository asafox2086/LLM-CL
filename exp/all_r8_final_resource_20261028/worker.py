"""Read-only original all-r8 S19 (unfolded, original dtype): five fixed GPU windows, no F1.

CLI: --root ABS --output NEW_ABS --binding ABS --gpu PHYSICAL_INDEX
Binding descriptors {path,sha256}: original_verified, original_ready,
original_complete, checkpoint_metadata; plus original_run, config_sha256,
worker_sha256, verifier_sha256, input_manifest/split, prior_windows,
helpers:{storage,readouts,clock}, runtime_files, environment, versions, GPU_uuids.
The isolated runtime ROOT must preserve all37 original READY source bytes,
including the historical policy, not alter the currently active native runner.
No fold, export, dtype conversion, training, score or alternate candidate.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

LIMIT = 600
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
INPUT_MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
INPUT_SPLIT_SHA = 'd8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
METHOD = 'LoRA_all'
LOGICAL_RUN = 'exp/osft_increment_confirmation_20261022/v1/private'
PORT = 'exp/two_backbone_native_cl_20261017_v2'
STUDY = 'exp/osft_increment_screen_20261019/v1/study.py'
VERIFIED_SHA = 'abbd2da43753c6bbeb137d7bef535983efbcafdaf08686f61550886b2f18598d'
READY_SHA = 'a8c4b99aef2e7b5e803d027e45d31aebd5ad60df49fea7c2b459d719249da628'
COMPLETE_SHA = 'ea9bd2283e2cc7d8948267289ff2e4cc7a20685cdc7b511fc176b628870d6a5c'
CONFIG_SHA = 'd346e0174fa6c0f714b94e161f617375268e13c5f1d228c0d03c43af933bfbca'
META_SHA = 'f9c7f55f568da722f22a6303036748730046685f06156b9da88edf70fc61cf48'
CP_SHA = '416ed177803b9bf986dacd73a6d1bf18c6c0c59a1eb65f42785d81a628d31519'
CP_BYTES = 1941599549
PRIOR_WINDOWS_SHA = '45bb9f4ca6d233b139f3c6a9b0255d29eee07e160b085fcef487bad8a3d8d7cd'
COMPARISON_GPU_UUID = 'GPU-3c72e4ce-0896-0b40-d610-be585d2bb51e'



def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def resolve(root, path):
    p = Path(path)
    assert p.is_absolute() or '..' not in p.parts
    return p if p.is_absolute() else root/p  # frozen data may be read-only symlinks


def checked(root, descriptor):
    p = resolve(root, descriptor['path']); assert sha(p) == descriptor['sha256'], str(p)
    return p


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def write(output, name, value):
    with (output/name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def identity(pid):
    p = Path('/proc')/str(pid); f = (p/'stat').read_text().rsplit(') ', 1)[1].split()
    assert f[0] != 'Z'
    return dict(pid=int(pid), startticks=f[19], uid=p.stat().st_uid, pgid=int(f[2]))


def legacy_contract(ready,complete,verified,config,meta):
    """Pure metadata identity contract; no CP/model dependency."""
    assert complete['complete'] and verified['complete'] and complete['case']==METHOD
    assert complete['TRAIN_QA']==verified['TRAIN_QA']==1313 and complete['updates']==verified['updates']==93
    assert complete['coverage']=='final_only_19_tasks_not_74_cells' and verified['final_only_not_original_74_cell_baseline']
    assert complete['quality']==verified['quality'] and complete['checkpoint']==verified['checkpoint']==meta
    assert complete['ready_sha256']==verified['ready_sha256']==READY_SHA
    assert verified['completion_sha256']==COMPLETE_SHA
    assert config==complete['config'] and config['method']==METHOD and config['condition']=='pct01'
    assert config['seed']==42 and config['rank']==8 and config['alpha']==32 and config['dropout']==.1
    assert config['learning_rate']==1e-4 and config['TRAIN_QA']==1313 and config['stages']==19 and config['updates']==93
    assert config['target_modules']==168 and config['trainable_parameters']==4399104
    assert config['fresh_base_not_FIT_checkpoint'] and config['TEST_QA']==21454
    assert config['manifest_sha256']==ready['manifest_sha256']==INPUT_MANIFEST_SHA
    assert config['execution_source_sha256']==complete['source_sha256']==ready['sources']['exp/osft_increment_confirmation_20261022/v1/run.py']
    assert meta['stage']==19 and meta['path']=='checkpoints/stage_19.pt' and meta['method']==METHOD
    assert meta['bytes']==CP_BYTES and meta['sha256']==CP_SHA and meta['full_architecture_and_runtime']
    assert not meta['optimizer_saved']
    assert complete['trainable_digest_before_TEST']==complete['trainable_digest_after_TEST']
    assert ready['sources'][STUDY]=='d189bc45b8086e862083320aed1808989638fc2ece214aa48a0a4287b46c7e30'
    assert ready['sources'][PORT+'/execution_policy.json']=='f6d33cc939ff833ec7799d2d1ec4c238bb97d47d35488034d9174b36f07c2502'


def validate_metadata(root,binding_path):
    """Small metadata/source/data checks, no Torch/GPU or CP payload load/hash."""
    root=Path(root).resolve();b=read(binding_path)
    assert sha(Path(__file__))==b['worker_sha256']
    proof_path=checked(root,b['original_verified']);assert sha(proof_path)==VERIFIED_SHA
    ready_path=checked(root,b['original_ready']);assert sha(ready_path)==READY_SHA
    complete_path=checked(root,b['original_complete']);assert sha(complete_path)==COMPLETE_SHA
    proof,ready,complete=read(proof_path),read(ready_path),read(complete_path)
    original=resolve(root,b['original_run'])
    assert original==root/LOGICAL_RUN
    config_path=original/'config.json';config=read(config_path)
    assert sha(config_path)==b['config_sha256']==CONFIG_SHA
    mp=checked(root,b['checkpoint_metadata']);assert sha(mp)==META_SHA
    meta=read(mp);assert meta==read(original/'checkpoints/latest.json')
    legacy_contract(ready,complete,proof,config,meta)
    cp=original/meta['path'];assert cp.is_file() and not cp.is_symlink()
    assert cp.stat().st_size==CP_BYTES and cp.stat().st_uid==os.getuid()
    # The active policy changed historically. Only this isolated snapshot must
    # contain the original f6d33cc bytes; no fallback to the current policy.
    assert len(ready['sources'])==37
    for relative,h in ready['sources'].items():
        p=Path(relative);assert not p.is_absolute() and '..' not in p.parts
        assert sha(root/p)==h,relative
    assert sha(root/STUDY)==ready['sources'][STUDY]
    port=root/PORT
    for field in ('runtime_files','environment'):
        assert b[field]
        for path,h in b[field].items():assert sha(resolve(root,path))==h,path
    for distribution,h in b['versions'].items():assert importlib.metadata.version(distribution)==h
    assert {'torch','transformers','accelerate','cloudpickle'}<=set(b['versions'])
    helpers={k:checked(root,b['helpers'][k]) for k in ('storage','readouts','clock')}
    manifest_path=checked(root,b['input_manifest']);manifest=read(manifest_path)
    split_path=checked(root,b['input_split']);split=read(split_path)
    assert b['input_manifest']['sha256']==INPUT_MANIFEST_SHA and b['input_split']['sha256']==INPUT_SPLIT_SHA
    assert len(manifest['tasks'])==19 and manifest['seed']==split['seed']==42
    rows=[]
    for task in manifest['tasks']:
        info=task['files']['train'];path=resolve(root,info['path']);assert sha(path)==info['sha256']
        items=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        assert len(items)==info['count']
        wanted=split['dev_ids'][str(task['task'])];mapping={r['id']:r for r in items}
        assert len(mapping)==len(items) and set(wanted)<=set(mapping)
        rows.extend((task,mapping[key]) for key in wanted)
    assert len(rows)==len({r['id'] for _,r in rows})==100
    assert all(sum(t['domain']==d for t,_ in rows)==20 for d in DOMAINS)
    selected=[min(((t,r) for t,r in rows if t['domain']==d),key=lambda tr:hashlib.sha256(tr[1]['id'].encode()).digest()) for d in DOMAINS]
    assert b['prior_windows']['sha256']==PRIOR_WINDOWS_SHA
    assert b['GPU_uuids']['1']==COMPARISON_GPU_UUID
    prior=read(checked(root,b['prior_windows']))
    assert [(x['domain'],x['id']) for x in prior]==[(t['domain'],r['id']) for t,r in selected]
    assert all(x['new_tokens']==32 and 'prepared_sha256' in x for x in prior)
    anchor=dict(coefficients=898074656,weight_bytes=1804947520,modules=168,adapter_coordinates=4399104,
                checkpoint=CP_SHA,bytes=CP_BYTES,config=CONFIG_SHA)
    return SimpleNamespace(root=root,binding=b,binding_sha256=sha(binding_path),backbone='onevision',
        anchor=anchor,original=original,config=config,proof=proof,meta=meta,port=port,helpers=helpers,
        rows=selected,split=split,prior=prior,ready=ready,study_path=root/STUDY,
        formal_proof_sha256=sha(proof_path),complete_sha256=sha(complete_path),ready_sha256=sha(ready_path),
        sources=dict(worker=sha(__file__),helpers={k:sha(p) for k,p in helpers.items()},
                     original_READY=sha(ready_path),study=sha(root/STUDY)))


def reachable_tensors(value):
    """Owned containers/attributes, bound-method self and closure cells, no globals.

    This inspects metadata only and does not keep returned Tensor references.
    Module/class globals are not walked; the pinned source declares no banks
    there. Actual load/measurement CUDA allocation includes all live references.
    """
    import torch
    import types
    pending=[('root',value)];seen=set();records={}
    while pending:
        path,obj=pending.pop()
        if isinstance(obj,torch.Tensor):
            storage=obj.untyped_storage();key=(str(obj.device),storage.data_ptr(),storage.nbytes())
            r=records.setdefault(key,dict(device=str(obj.device),bytes=storage.nbytes(),dtype=str(obj.dtype),
                first_path=path,views=0))
            r['views']+=1
            continue
        if obj is None or isinstance(obj,(str,bytes,int,float,bool,Path,torch.dtype,torch.device)):continue
        if id(obj) in seen:continue
        seen.add(id(obj))
        if isinstance(obj,dict):pending.extend((path+'.'+str(k),v) for k,v in obj.items())
        elif isinstance(obj,(list,tuple,set,frozenset)):pending.extend((path+'.'+str(i),v) for i,v in enumerate(obj))
        elif isinstance(obj,types.MethodType):pending.append((path+'.__self__',obj.__self__))
        elif isinstance(obj,types.FunctionType):
            for i,cell in enumerate(obj.__closure__ or ()):
                try:pending.append((path+'.closure.'+str(i),cell.cell_contents))
                except ValueError:pass
        elif isinstance(obj,(types.ModuleType,type)):continue
        elif hasattr(obj,'__dict__'):pending.append((path+'.__dict__',vars(obj)))
    return records


def storage_report(records):
    devices={}
    for r in records.values():
        d=devices.setdefault(r['device'],dict(bytes=0,storages=0))
        d['bytes']+=r['bytes'];d['storages']+=1
    return dict(devices=devices,storages=list(records.values()))


def payload_references(payload,model):
    all_records=reachable_tensors(payload);model_records=reachable_tensors(model)
    outside={k:r for k,r in all_records.items() if k not in model_records}
    outside_cuda=[r for r in outside.values() if r['device'].startswith('cuda')]
    assert outside_cuda==[],outside_cuda
    # The native payload has no learned tensor state outside its method/model;
    # CPU RNG byte tensors remain resident and are counted rather than dropped.
    assert all(r['first_path'].startswith('root.rng.') for r in outside.values()),outside
    core=getattr(model,'core',model)
    registered={}
    for n,v in list(core.named_parameters(remove_duplicate=False))+list(core.named_buffers(remove_duplicate=False)):
        st=v.untyped_storage();registered[(str(v.device),st.data_ptr(),st.nbytes())]=n
    extra_model={k:r for k,r in model_records.items() if k not in registered}
    # These are the two explicitly classified original native cache families.
    # A second W, A/B, G, or an unknown tensor bank cannot hide in __dict__.
    for r in extra_model.values():
        assert 'original_inv_freq' in r['first_path'] or '_llmcl_visual_cache' in r['first_path'],r
    return dict(all_payload=storage_report(all_records),model_owned=storage_report(model_records),
        outside_model=storage_report(outside),outside_model_CUDA_storages=0,
        classified_model_unregistered=storage_report(extra_model),unknown_tensor_bank_storages=0,
        actual_payload_retained=True,scope='Owned attributes/containers and closures, not Python module/class globals')


def structure(model,method,seq_type,anchor):
    import torch
    assert method.name=='SeqFT_LoRA' and method.original_revision=='existing-qv-r8-alpha32-dropout0.1-with-location-control'
    assert type(method).__name__=='TargetLoRA' and type(method).__module__=='confirmation_inherited_study'
    assert method.embedding is None and getattr(method,'_train_rows',{})=={}
    owned={k:v for k,v in vars(method).items() if k not in ('model','layers','vision_original')}
    assert reachable_tensors(owned)=={}, 'Unexpected method-owned tensor bank'
    core=getattr(model,'core',model);decoder=core.language_model.model;assert len(decoder.layers)==24
    discovered={id(m):n for n,m in model.named_modules() if isinstance(m,seq_type)}
    assert len(discovered)==len(method.layers)==168
    result={};coordinates=0
    for i,block in enumerate(decoder.layers):
        for scope,names in (('self_attn',('q_proj','k_proj','v_proj','o_proj')),('mlp',('gate_proj','up_proj','down_proj'))):
            for part in names:
                name=f'model.layers.{i}.{scope}.{part}';layer=getattr(getattr(block,scope),part)
                assert isinstance(layer,seq_type) and layer in method.layers and id(layer) in discovered
                assert layer.A.weight.shape==(8,layer.weight.shape[1])
                assert layer.B.weight.shape==(layer.weight.shape[0],8)
                assert layer.scale==4 and layer.dropout.p==.1 and not layer.training
                assert layer.weight.dtype==torch.float16 and layer.A.weight.dtype==layer.B.weight.dtype==torch.float32
                assert all(t.is_cuda and t.device.index==0 for t in (layer.weight,layer.A.weight,layer.B.weight))
                coordinates+=layer.A.weight.numel()+layer.B.weight.numel()
                result[name]=dict(A_shape=list(layer.A.weight.shape),B_shape=list(layer.B.weight.shape),
                    base_dtype=str(layer.weight.dtype),adapter_dtype=str(layer.A.weight.dtype),scale=4,dropout=.1)
    assert coordinates==4399104 and {id(model.get_submodule(n)) for n in result}==set(discovered)
    assert set(discovered)=={id(m) for m in method.layers}
    assert all(not p.requires_grad and p.grad is None for p in core.parameters())
    return dict(resident_adapter_banks=1,projection_modules=168,adapter_coordinates=coordinates,layers=result,
        method_layer_list_aliases_registered_modules=True,no_additional_method_tensor_bank=True,
        definition='One shared unmerged r8 adapter over seven projection types, not168 independent task banks')


def generation(model):
    core = getattr(model, 'core', model)
    values = {'outer': core.generation_config}
    if hasattr(core, 'language_model'): values['language'] = core.language_model.generation_config
    return {k: v.to_dict() for k, v in values.items()}


def dtype_storage(model):
    result = {}; seen = set()
    for name, value in model.named_parameters(remove_duplicate=False):
        s = value.untyped_storage(); key = (str(value.device), s.data_ptr(), s.nbytes())
        if key in seen: continue
        seen.add(key); item = result.setdefault(str(value.dtype), dict(bytes=0, coefficients=0))
        item['bytes'] += s.nbytes(); item['coefficients'] += s.nbytes()//value.element_size()
    return result


def raw_binding(row, task, details, split):
    context = row.get('context', ''); indexes = details['retrieved_chunk_indexes']
    if len(context) > 6000:
        chunks = [context[i:i+1000] for i in range(0, len(context), 800)]
        context = '\n'.join(chunks[i] for i in indexes)
    else: assert indexes == []
    contract = dict(images=[dict(path=n, sha256=split['image_sha256'][n]) for n in row['images']],
        instruction=task['instruction'], question=row['question'], original_context=row.get('context', ''),
        resolved_context=context, retrieved_chunk_indexes=indexes, image_shapes=details['image_shapes'], common_image_area_cap=401408)
    return dict(raw_input_sha256=hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), image_SHA256=contract['images'])


def main(args):
    root = args.root.resolve(); output = args.output
    assert root.is_absolute() and output.is_absolute() and args.binding.is_absolute() and args.gpu==1
    output.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic(); who = identity(os.getpid()); assert who['pid'] == who['pgid']
    write(output, 'STARTED.json', dict(unix=time.time(), identity=who, physical_GPU=args.gpu, virtual_GPU=0,
        binding_sha256=sha(args.binding), limit=LIMIT, boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    def expired(*unused): raise TimeoutError('Fixed all-r8 resource600 deadline expired')
    signal.signal(signal.SIGALRM, expired); signal.alarm(LIMIT)
    c = validate_metadata(root, args.binding); b = c.binding
    # A caller's allocator supplies the slot. This worker only refuses occupied
    # hardware; it never allocates another GPU or signals an existing process.
    q = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert q[0].strip() == b['GPU_uuids'][str(args.gpu)] and int(q[1]) < 500 and int(q[2]) < 10
    assert not subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', LLMCL_ROOT=str(root))
    import torch
    torch.set_num_threads(8); torch.cuda.set_device(0); assert torch.cuda.device_count() == 1
    sys.path.insert(0, str(c.port))
    import checkpoints, methods, runtime_utils as rt, onevision_bridge as bridge, frozen_vision as fv
    for name, value in [('checkpoints.py', checkpoints), ('methods.py', methods), ('runtime_utils.py', rt), ('onevision_bridge.py', bridge), ('frozen_vision.py', fv)]:
        assert sha(Path(value.__file__))==c.ready['sources'][PORT+'/'+name]
    # Original load hashes the full CP once and restores original CPU/CUDA0
    # placement/RNG. It does not invoke checkpoint_probe or delete anything.
    # Original common.py loaded this exact study under this name without
    # invoking prepare or cpu_gate. Register it so both by-value cloudpickle
    # classes and a possible GLOBAL reference have their original source path.
    legacy_study=module('confirmation_inherited_study',c.study_path)
    assert legacy_study.TargetLoRA.__module__=='confirmation_inherited_study'
    assert sha(legacy_study.__file__)==c.ready['sources'][STUDY]
    sys.modules['confirmation_inherited_study']=legacy_study
    load_tick = time.monotonic()
    payload = checkpoints.load(c.original, expected=c.config)
    checkpoint_load_seconds = time.monotonic()-load_tick
    assert payload['format'] == 'llmcl-method-graph-v1' and payload['stage'] == 19
    assert payload['probe'] is None and payload['config']==c.config
    # The old phase label is known to be inherited: S19 was saved before TEST.
    original_payload_phase_label=payload['checkpoint_phase']
    assert set(payload)=={'format','method','processor','stage','config','rng','probe',
                          'optimizer_lifecycle','checkpoint_phase'}
    method = payload['method']; model = method.model.eval(); processor = payload['processor']
    assert method.name=='SeqFT_LoRA'
    checkpoint_load_allocated=torch.cuda.memory_allocated(0)
    checkpoint_load_reserved=torch.cuda.memory_reserved(0)
    checkpoint_load_peak_allocated=torch.cuda.max_memory_allocated(0)
    loaded_references=payload_references(payload,model)
    assert model.device == torch.device('cuda:0') and model.dtype == torch.float16
    saved_requires_grad_coordinates=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert saved_requires_grad_coordinates==4399104
    model.requires_grad_(False)  # Same inference freezing as original Seq resource worker, no value/dtype change.
    core = getattr(model, 'core', model)
    assert {id(p) for p in model.parameters()} == {id(p) for p in core.parameters()}
    assert all(p.is_cuda and p.device.index == 0 and p.grad is None for p in core.parameters())
    storage = module('final_seq_storage', c.helpers['storage'])
    readouts = module('final_seq_readouts', c.helpers['readouts'])
    clock_module = module('final_seq_token_clock', c.helpers['clock'])
    layout = storage.layout(core)
    assert layout['weights'] == dict(coefficients=c.anchor['coefficients'], bytes=c.anchor['weight_bytes'])
    assert layout['trainable'] == 0
    before_structure = structure(model, method, methods.SeqLinear, c.anchor)
    before = storage.digest(core); buffers_before = storage.digest(list(core.named_buffers(remove_duplicate=False)))
    generation_before = generation(model); dtypes = dtype_storage(core)
    old_cache = model._llmcl_visual_cache_bytes
    model._llmcl_visual_cache.clear(); model._llmcl_visual_cache_bytes = 0
    model._llmcl_visual_cache_hits = model._llmcl_visual_cache_misses = 0; fv.LIMIT = 0
    # Match the existing frozen GPU transport: same visual arithmetic, no
    # CPU visual-feature staging/cache. No hashing is done inside preparation.
    def gpu_prepare(batch, row, active, original=None):
        pixels = batch.get('pixel_values')
        if pixels is None or (pixels.ndim == 2 and pixels.shape[-1] == active.config.hidden_size): return False
        extractor = original or active._llmcl_visual_original; dtype = active.visual.get_dtype()
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            features = extractor(pixels.to(dtype), batch['image_grid_thw']).detach()
        assert features.is_cuda and features.ndim == 2 and features.shape[-1] == active.config.hidden_size
        active._llmcl_visual_cache_misses += 1; batch['pixel_values'] = features
        return False
    fv.prepare = gpu_prepare
    bridge.bind(model); bridge.LAST_NATIVE = None; bridge.NATIVE_GATE_MODE = False
    retained_references=payload_references(payload,model)
    # Check all five files before timing. rt.ROOT may be the original asset
    # root behind a read-only symlink, so bind the actual copy it will open.
    for _, row in c.rows:
        for name in row['images']: assert sha(Path(rt.ROOT)/name) == c.split['image_sha256'][name], name
    stop = dict(generation_stop_ids=processor.tokenizer.eos_token_id, pad_id=processor.tokenizer.pad_token_id, context_limit=None)
    records = []
    for (task, row), prior in zip(c.rows, c.prior):
        method.set_instruction(task['task']); bridge.bind(model); bridge.LAST_NATIVE = None
        assert model._llmcl_visual_cache_bytes == 0
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
        baseline = torch.cuda.memory_allocated(0)
        clock = clock_module.FixedWindowTokenClock(synchronize=lambda: torch.cuda.synchronize(0), expected_tokens=32)
        original_generate = model.generate; saved_instance_generate = model.__dict__.get('generate')
        def measured_generate(*positional, **keywords):
            assert 'streamer' not in keywords; keywords['streamer'] = clock
            return original_generate(*positional, **keywords)
        model.generate = measured_generate
        try:
            # Only real preparation/visual/decoder work plus the original
            # finite-logit guard and synchronized streamer are inside timing.
            clock.start(); batch, details = rt.input_batch(row, task, processor, model.device)
            batch = {n: v for n, v in batch.items() if not n.startswith('_')}
            tokens = readouts._generate(model, batch, 32, stop, True)
            timing = clock.result(); torch.cuda.synchronize(0)
            peak = torch.cuda.max_memory_allocated(0); reserved = torch.cuda.max_memory_reserved(0)
        finally:
            # Restore the original instance/class dispatch, never another graph.
            if saved_instance_generate is None: del model.generate
            else: model.generate = saved_instance_generate
        assert len(tokens) == 32 and timing['generated_tokens'] == 32
        # Audit-only GPU->CPU tensor digests and contract SHA are deliberately
        # after the timing endpoints and the captured allocation peak.
        binding = raw_binding(row, task, details, c.split)
        assert binding['raw_input_sha256'] == prior['raw_input_sha256']
        assert binding['image_SHA256'] == prior['image_SHA256']
        for name in ('image_shapes', 'retrieved_chunk_indexes', 'common_image_area_cap'):
            assert details[name] == prior[name], name
        binding.update(details, prepared_sha256=storage.digest(batch),
            transport_sha256=storage.digest({n: v for n, v in batch.items() if n != 'pixel_values'}),
            prompt_tokens=batch['input_ids'].shape[1], input_shape=list(batch['input_ids'].shape),
            input_tensor_shapes={n: list(v.shape) for n, v in batch.items()})
        for key in ('prepared_sha256','transport_sha256','input_shape','input_tensor_shapes'):
            assert binding[key]==prior[key],key
        records.append(dict(binding, id=row['id'], task=task['task'], domain=task['domain'], new_tokens=32,
            baseline_allocated=baseline, peak_allocated=peak, peak_reserved=reserved, timing=timing,
            audit_hashes_in_timing=False, finite_logit_guard_in_timing=True, synchronized_token_callbacks=True))
        del batch, tokens, clock, original_generate, measured_generate, saved_instance_generate
    assert structure(model, method, methods.SeqLinear, c.anchor) == before_structure
    assert storage.digest(core) == before and storage.digest(list(core.named_buffers(remove_duplicate=False))) == buffers_before
    assert generation(model) == generation_before and model._llmcl_visual_cache_bytes == 0
    write(output, 'windows.json', records)
    seconds = time.monotonic()-tick; assert seconds <= LIMIT
    write(output, 'COMPLETE.json', dict(complete=True, identity=who, seconds=seconds,
        binding_sha256=c.binding_sha256, source_sha256=c.sources, backbone=c.backbone,
        original_run=str(c.original), sealed_logical_run=LOGICAL_RUN,formal_proof_sha256=c.formal_proof_sha256,
        original_COMPLETE_sha256=c.complete_sha256,original_READY_sha256=c.ready_sha256,
        config_sha256=c.anchor['config'], checkpoint_sha256=c.anchor['checkpoint'], checkpoint_bytes=c.anchor['bytes'],
        checkpoint_load_seconds=checkpoint_load_seconds,
        checkpoint_load_seconds_scope='original checkpoints.load including full checkpoint SHA, torch.load and RNG restoration; subset of total seconds',
        original_full_unfolded_graph=True,resident_adapter_banks=1,adapter_structure=before_structure,
        saved_requires_grad_coordinates=saved_requires_grad_coordinates,inference_parameter_freezing_only=True,
        original_checkpoint_saved_before_TEST=True,original_payload_phase_label=original_payload_phase_label,
        coverage='final_only_19_tasks_not_74_cells',
        loaded_payload_references=loaded_references,retained_payload_references=retained_references,
        checkpoint_load_allocated=checkpoint_load_allocated,checkpoint_load_reserved=checkpoint_load_reserved,
        checkpoint_load_peak_allocated=checkpoint_load_peak_allocated,
        no_fold_cast_or_export=True,dynamic_TargetLoRA_source_sha256=c.sources['study'],
        layout=layout, dtype_storage=dtypes, state_digest=before, all_buffers_digest=buffers_before,
        state_and_registered_buffers_unchanged=True, generation_public_unchanged=True,
        original_CPU_visual_cache_removed_bytes=old_cache, original_other_CPU_payload_retained=True,
        physical_GPU=args.gpu, virtual_GPU=0, GPU_uuid=q[0].strip(), runtime_image_root=str(rt.ROOT),
        window_order='fresh_reload_before_any_other_generation', window_selection='original per-domain hash-min DEV raw IDs',
        windows_sha256=sha(output/'windows.json'), new_generations=5, forced_new_tokens=160,
        new_scores=0, NLL_requests=0, backwards=0, optimizer_updates=0,
        timing='instrumented wall time; finite-logit checks and per-token synchronization included; not production throughput',
        audit_hashes_in_timing=False, audit_digest_after_captured_peak=True,
        checkpoint_deleted_or_rewritten=False, no_reconstruction_or_export=True,old_closed_quality_result_unchanged=True,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate', 'cloudpickle')}))
    signal.alarm(0)
    print(json.dumps(dict(complete=True, output=str(output), backbone=c.backbone, new_generations=5, new_scores=0, seconds=seconds)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'output', 'binding'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(1,), required=True)
    arguments = parser.parse_args()
    try: main(arguments)
    except BaseException:
        marker = arguments.output/'STARTED.json'
        if marker.is_file() and read(marker)['identity']['pid'] == os.getpid() and not (arguments.output/'FAILURE.json').exists():
            write(arguments.output, 'FAILURE.json', dict(unix=time.time(), traceback=traceback.format_exc(), new_quality_claim=False))
        raise
