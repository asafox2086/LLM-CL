"""Read-only independently folded TaskGradReuse S16: four fixed GPU windows, no F1.

CLI: --root ABS --output NEW_ABS --binding ABS --gpu PHYSICAL_INDEX
Binding descriptors are {path,sha256}. Fields: worker_sha256, quality_result,
quality_verified, quality_verifier, quality_binding, quality_complete, quality_fold,
quality_supervisor_exit, checkpoint_metadata, checkpoint_sha256, config_sha256,
input_manifest, input_split, prior_windows (original PAS five-window record;
only its first four are selected), helpers:{storage,readouts,clock}, runtime_files,
environment, versions, GPU_uuids. Final CP and quality hashes are filled only
after successful final completion. Root is the original formal science root.
No fold/export or dtype conversion is performed. Loading retains the actual
payload and audits reachable tensor storage, including hidden references.
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
DOMAINS = ('natural', 'medical', 'geography', 'finance')
INPUT_MANIFEST_SHA = '6ffae80b7e290b4bcd7b3ef8df7613c176a344221d28e874125f3faf87bd6798'
INPUT_SPLIT_SHA = 'd8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
METHOD = 'TaskGradReuse'
FORMAL_FRAME = 'exp/task_gradient_formal_20261028'
FINAL_RUN = 'exp/task_gradient_four_domain_final_20261029/runs/onevision/pct01/final_s16_v1'
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


def validate_metadata(root, binding_path):
    """Final four-domain success/source metadata, then four fixed original inputs."""
    root=Path(root).resolve();b=read(binding_path)
    assert b['schema']=='task-gradient-four-domain-resource-v1'
    assert sha(Path(__file__))==b['worker_sha256']
    quality_binding_path=checked(root,b['quality_binding']);qb=read(quality_binding_path)
    assert sha(quality_binding_path)=='4c7a351c3e9873384652ad4b01d3cfbb0b15acb6b3b2613d80e4b1037863dd89'
    assert qb['schema']=='task-gradient-four-domain-final-v1' and qb['GPU']==1
    assert qb['worker_sha256']=='7221a74e7553e7bf3a7ef26ad07e86037e56e4e177de63268d3637e4894daceb'
    result_path=checked(root,b['quality_result']);result=read(result_path)
    verified_path=checked(root,b['quality_verified']);verified=read(verified_path)
    verifier_path=checked(root,b['quality_verifier'])
    assert verified['verifier_sha256']==result['verifier_sha256']==sha(verifier_path)
    assert result['status']==verified['status']=='FOUR_DOMAIN_NATIVE_SCORE_METADATA_VERIFIED_DEPLOYMENT_PENDING'
    assert verified['result_sha256']==sha(result_path) and verified['existing_answer_records_checked']==19331
    assert result['binding_sha256']==verified['binding_sha256']==sha(quality_binding_path)
    assert verified['new_score_calls']==result['new_scoring_calls']==0
    assert result['new_models']==result['new_generations']==0
    assert result['terminal_stage']==16 and result['independent_dense_reload_pending']
    original=root/FINAL_RUN;report=root/'summary/task_gradient_four_domain_final_20261029'
    complete_path=checked(root,b['quality_complete']);terminal=read(complete_path)
    assert complete_path==original/'COMPLETE.json'
    assert sha(complete_path)==result['worker_complete_sha256']
    assert not (original/'FAILURE.json').exists() and not (report/'SUPERVISOR_FAILURE.json').exists()
    supervisor=read(report/'SUPERVISOR_COMPLETE.json');worker_exit=read(report/'WORKER_EXIT.json')
    supervisor_exit_path=checked(root,b['quality_supervisor_exit']);supervisor_exit=read(supervisor_exit_path)
    assert supervisor['returncode']==worker_exit['returncode']==supervisor_exit['returncode']==0
    assert supervisor['complete_sha256']==sha(complete_path)
    assert supervisor['worker']==worker_exit['identity'] and supervisor['supervisor']==supervisor_exit['identity']
    assert worker_exit['identity_absent'] and supervisor_exit['identity_absent']
    assert supervisor_exit['binding_sha256']==sha(quality_binding_path)
    for who in (supervisor['worker'],supervisor['supervisor']):
        proc=Path('/proc')/str(who['pid'])/'stat'
        if proc.exists():
            f=proc.read_text().rsplit(') ',1)[1].split()
            assert f[0]=='Z' or f[19]!=str(who['startticks'])
    assert terminal['binding_sha256']==sha(quality_binding_path) and terminal['worker_sha256']==qb['worker_sha256']
    assert terminal['source_stage']==terminal['terminal_stage']==16 and terminal['new_training_updates']==0
    assert terminal['new_generations']==19331 and terminal['self_contained_final_graph']
    port=Path(qb['source_dir']);assert port==root/FORMAL_FRAME
    assert {p.name for p in port.glob('*.py')}==set(qb['original_sources']) and len(qb['original_sources'])==29
    for name,h in qb['original_sources'].items():assert sha(port/name)==h
    config_path=original/'config.json';config=read(config_path)
    assert sha(config_path)==b['config_sha256']
    assert config['method']==METHOD and config['backbone']=='onevision' and config['condition']=='pct01'
    assert config['manifest_sha256']==INPUT_MANIFEST_SHA and config['terminal_stage']==16
    assert config['execution_source_sha256']==qb['original_sources']
    assert config['postprocess_binding_sha256']==sha(quality_binding_path)
    assert config['model_dtype']==config['autocast_dtype']=='torch.float16'
    mp=checked(root,b['checkpoint_metadata']);meta=read(mp)
    assert mp==original/'checkpoints/stage_16.json'
    assert meta==read(original/'checkpoints/latest.json')==terminal['checkpoint']==result['final_checkpoint']
    assert meta['stage']==16 and meta['path']=='checkpoints/stage_16.pt' and meta['method']==METHOD
    assert meta['sha256']==b['checkpoint_sha256'] and meta['manifest_sha256']==INPUT_MANIFEST_SHA
    assert (original/meta['path']).stat().st_size==meta['bytes']
    fold_path=checked(root,b['quality_fold']);fold=read(fold_path)
    assert fold_path==original/'FOLD.json' and fold['source_stage']==fold['terminal_stage']==16
    assert len(fold['modules'])==48 and fold['original_fold_function'] and fold['training_context_cleared']
    assert fold['dense_state']==terminal['dense_state']==terminal['state_after_evaluation']
    for path in [complete_path,config_path,mp,fold_path]:
        assert verified['source_ledger'][str(path)]['sha256']==sha(path)
    for field in ('runtime_files','environment'):
        assert b[field]
        for path,h in b[field].items():assert sha(resolve(root,path))==h,path
    for distribution,h in b['versions'].items():assert importlib.metadata.version(distribution)==h
    assert {'torch','transformers','accelerate','cloudpickle'}<=set(b['versions'])
    helpers={k:checked(root,b['helpers'][k]) for k in ('storage','readouts','clock')}
    manifest_path=checked(root,b['input_manifest']);manifest=read(manifest_path)
    split_path=checked(root,b['input_split']);split=read(split_path)
    assert b['input_manifest']['sha256']==INPUT_MANIFEST_SHA and b['input_split']['sha256']==INPUT_SPLIT_SHA
    assert [t['task'] for t in manifest['tasks']]==list(range(1,17)) and manifest['seed']==split['seed']==42
    rows=[]
    for task in manifest['tasks']:
        info=task['files']['train'];path=resolve(root,info['path']);assert sha(path)==info['sha256']
        items=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        assert len(items)==info['count']
        wanted=split['dev_ids'][str(task['task'])];mapping={r['id']:r for r in items}
        assert len(mapping)==len(items) and set(wanted)<=set(mapping)
        rows.extend((task,mapping[key]) for key in wanted)
    assert len(rows)==len({r['id'] for _,r in rows})==80
    assert all(sum(t['domain']==d for t,_ in rows)==20 for d in DOMAINS)
    selected=[min(((t,r) for t,r in rows if t['domain']==d),key=lambda tr:hashlib.sha256(tr[1]['id'].encode()).digest()) for d in DOMAINS]
    assert b['prior_windows']['sha256']==PRIOR_WINDOWS_SHA and b['GPU_uuids']['1']==COMPARISON_GPU_UUID
    all_prior=read(checked(root,b['prior_windows']))
    assert len(all_prior)==5 and [x['domain'] for x in all_prior]==list(DOMAINS)+['traffic']
    prior=all_prior[:4]
    assert [(x['domain'],x['id']) for x in prior]==[(t['domain'],r['id']) for t,r in selected]
    assert all(x['new_tokens']==32 and 'prepared_sha256' in x for x in prior)
    anchor=dict(coefficients=893675552,weight_bytes=1787351104,modules=48,adapter_coordinates=0,
                checkpoint=meta['sha256'],bytes=meta['bytes'],config=sha(config_path))
    return SimpleNamespace(root=root,binding=b,binding_sha256=sha(binding_path),backbone='onevision',
        anchor=anchor,original=original,config=config,meta=meta,port=port,helpers=helpers,
        rows=selected,split=split,prior=prior,fold=fold,quality_result_sha256=sha(result_path),
        quality_verified_sha256=sha(verified_path),sources=dict(worker=sha(__file__),
        helpers={k:sha(p) for k,p in helpers.items()},quality_binding=sha(quality_binding_path)),
        prior_four_window_max_peak_allocated=max(x['peak_allocated'] for x in prior))


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
    assert method.name==METHOD and method.original_revision=='task-current-default-CUDA-SVD-right-A-r8-B0-native-v1'
    assert method.deployed and method.completed_stage==method.initialized_stage==16 and method.active_stage is None
    assert method.layers==[] and method._train_rows=={} and method._processor is method._manifest is None
    assert method._preflight is False and method.embedding is None
    assert len(method._history)==16 and [r['stage'] for r in method._history]==list(range(1,17))
    owned={k:v for k,v in vars(method).items() if k not in ('model','vision_original')}
    assert reachable_tensors(owned)=={}, 'Method retains a Tensor outside model/visual alias'
    core=getattr(model,'core',model);decoder=core.language_model.model
    assert len(decoder.layers)==24
    assert not any(isinstance(m,seq_type) for m in model.modules())
    assert not any(n.endswith(('.A.weight','.B.weight')) or '.lora_' in n for n,_ in core.named_parameters())
    result={}
    for i,block in enumerate(decoder.layers):
        for part in ('q','v'):
            name=f'model.layers.{i}.self_attn.{part}_proj';layer=getattr(block.self_attn,part+'_proj')
            assert type(layer) is torch.nn.Linear and not hasattr(layer,'A') and not hasattr(layer,'B')
            assert list(layer.weight.shape)==([896,896] if part=='q' else [128,896])
            assert layer.weight.is_cuda and layer.weight.device.index==0 and layer.weight.dtype==torch.float16
            result[name]=dict(weight_shape=list(layer.weight.shape),dtype=str(layer.weight.dtype),dense=True)
    assert len(result)==48 and all(not p.requires_grad and p.grad is None for p in core.parameters())
    return dict(resident_adapter_banks=0,projection_modules=48,adapter_coordinates=0,
        no_AB_bank_G_or_TRAIN_context=True,method_owned_tensor_storages=0,layers=result,
        definition='All 48 task-gradient q/v sites are final dense replacements; cumulative update folded independently at S16')


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
    def expired(*unused): raise TimeoutError('Fixed TaskGrad resource600 deadline expired')
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
    import checkpoints, methods, task_gradient_adapter, runtime_utils as rt, onevision_bridge as bridge, frozen_vision as fv
    for name, value in [('checkpoints.py', checkpoints), ('methods.py', methods), ('runtime_utils.py', rt), ('onevision_bridge.py', bridge), ('frozen_vision.py', fv), ('task_gradient_adapter.py',task_gradient_adapter)]:
        assert sha(Path(value.__file__)) == c.config['execution_source_sha256'][name]
    # Original load hashes the full CP once and restores original CPU/CUDA0
    # placement/RNG. It does not invoke checkpoint_probe or delete anything.
    load_tick = time.monotonic()
    payload = checkpoints.load(c.original, expected=c.config)
    checkpoint_load_seconds = time.monotonic()-load_tick
    assert payload['format'] == 'llmcl-method-graph-v1' and payload['stage'] == 16
    assert payload['probe'] is None and payload['checkpoint_phase']=='after_stage_evaluation'
    assert set(payload)=={'format','method','processor','stage','config','rng','probe',
                          'optimizer_lifecycle','checkpoint_phase'}
    method = payload['method']; model = method.model.eval(); processor = payload['processor']
    assert method.name==METHOD and method.deployed
    checkpoint_load_allocated=torch.cuda.memory_allocated(0)
    checkpoint_load_reserved=torch.cuda.memory_reserved(0)
    checkpoint_load_peak_allocated=torch.cuda.max_memory_allocated(0)
    loaded_references=payload_references(payload,model)
    assert model.device == torch.device('cuda:0') and model.dtype == torch.float16
    assert not any(p.requires_grad for p in model.parameters())
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
    assert method._state_digest()==c.fold['dense_state']
    assert generation(model)==dict(outer=c.fold['generation_config']['core'],language=c.fold['generation_config']['language_model'])
    before = storage.digest(core); buffers_before = storage.digest(list(core.named_buffers(remove_duplicate=False)))
    generation_before = generation(model); dtypes = dtype_storage(core)
    for record in c.fold['modules']:
        actual=task_gradient_adapter._digest([('weight',model.get_submodule(record['module']).weight)])
        assert actual==record['base_after_digest'],record['module']
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
    # Check all four selected inputs before timing. rt.ROOT may be the original asset
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
        original_run=str(c.original), sealed_logical_run=FINAL_RUN, quality_result_sha256=c.quality_result_sha256,quality_verified_sha256=c.quality_verified_sha256,
        config_sha256=c.anchor['config'], checkpoint_sha256=c.anchor['checkpoint'], checkpoint_bytes=c.anchor['bytes'],
        checkpoint_load_seconds=checkpoint_load_seconds,
        checkpoint_load_seconds_scope='original checkpoints.load including full checkpoint SHA, torch.load and RNG restoration; subset of total seconds',
        original_full_folded_graph=True,resident_adapter_banks=0,adapter_structure=before_structure,
        loaded_payload_references=loaded_references,retained_payload_references=retained_references,
        checkpoint_load_allocated=checkpoint_load_allocated,checkpoint_load_reserved=checkpoint_load_reserved,
        checkpoint_load_peak_allocated=checkpoint_load_peak_allocated,
        final_qv_values_match_saved_fold_digests=True,no_extra_fold_or_dtype_conversion=True,
        layout=layout, dtype_storage=dtypes, state_digest=before, all_buffers_digest=buffers_before,
        state_and_registered_buffers_unchanged=True, generation_public_unchanged=True,
        original_CPU_visual_cache_removed_bytes=old_cache, original_other_CPU_payload_retained=True,
        physical_GPU=args.gpu, virtual_GPU=0, GPU_uuid=q[0].strip(), runtime_image_root=str(rt.ROOT),
        window_order='fresh_reload_before_any_other_generation', window_selection='original per-domain hash-min DEV raw IDs',
        windows_sha256=sha(output/'windows.json'), measured_domains=list(DOMAINS),
        historical_PAS_comparison_windows=4, historical_PAS_four_window_max_peak_allocated=c.prior_four_window_max_peak_allocated,
        comparison_scope='Same four fixed inputs/GPU/dtype protocol; PAS historical five-domain-trained model, not matched four-domain endpoint quality', new_generations=4, forced_new_tokens=128,
        new_scores=0, NLL_requests=0, backwards=0, optimizer_updates=0,
        timing='instrumented wall time; finite-logit checks and per-token synchronization included; not production throughput',
        audit_hashes_in_timing=False, audit_digest_after_captured_peak=True,
        checkpoint_deleted_or_rewritten=False, no_reconstruction_or_export=True,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate', 'cloudpickle')}))
    signal.alarm(0)
    print(json.dumps(dict(complete=True, output=str(output), backbone=c.backbone, new_generations=4, new_scores=0, seconds=seconds)), flush=True)


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
