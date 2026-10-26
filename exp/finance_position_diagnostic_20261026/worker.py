"""One fixed formal-checkpoint TRAIN position diagnostic; no training/scoring.

Only main performs I/O/imports model code. Binding FileRefs are absolute
{path,sha256}. LLMCL_ROOT is the selected state's original repository/overlay.
The caller owns the one-free-GPU policy, four-state order and 1200s deadline.
There is no resume, checkpoint rewrite, export, CPU model or quality consumer.
"""
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
import traceback

MANIFESTS = {
    'pct01': ('data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json',
              '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777', 89),
    'pct10': ('data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json',
              '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7', 887),
}
PROOFS = {'pct01':'c0c010f18db89b088cd3ac13b92e79a76c560c13ee9f9960f0e5e2ddc8e88c69',
          'pct10':'73fdbbd139a6910225088e78cc44033824456e09dec93411ed93582ac4d67c99'}
DECISION_SHA = 'f2088a3fbed06d1dd2e6a328ece573fb1b74746c98aaaf1dfc60d4b764ad9cda'
PACK_SHA = '698f6e898d296ea7f6fedb4089cefbc4989125c19fa2acb20125470f88468eb0'
ORDER = (3, 0, 1, 2)  # New A/B/C/D receive original D/A/B/C.


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def relative(root, name):
    p = Path(name)
    assert not p.is_absolute() and '..' not in p.parts, 'Expected ROOT-relative source'
    return root / p  # Frozen asset symlinks are legitimate; exact bytes are bound.


def bound(ref, ledger):
    p = Path(ref['path'])
    assert p.is_absolute() and p.is_file()
    raw = p.read_bytes()
    assert digest(raw) == ref['sha256'], str(p)
    ledger[str(p)] = {'sha256':ref['sha256'], 'bytes':len(raw)}
    return raw


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def ending(text):
    if text.endswith('\r\n'):
        return text[:-2], '\r\n'
    if text.endswith('\n'):
        return text[:-1], '\n'
    return text, ''


def cyclic_context(context):
    """Move unchanged payloads; keep each position's original frame/separator.

    The final CRLF/LF before the next header is a position separator. A final
    CRLF/LF, if present, stays at the final position. Other whitespace belongs
    to payloads. Inversion uses these fixed frames and exact payload lengths,
    avoiding ambiguity when a moved payload itself ends with a newline.
    """
    matches = list(re.finditer(r'(?m)^([A-D]):[ \t]*', context))
    assert len(matches) == 4 and [m.group(1) for m in matches] == list('ABCD')
    prefix = context[:matches[0].start()]
    assert not prefix.strip()
    headers = [m.group(0) for m in matches]
    payloads, separators = [], []
    for index, match in enumerate(matches):
        region = context[match.end():matches[index+1].start() if index < 3 else len(context)]
        payload, separator = ending(region)
        assert payload.strip(), 'Empty qualified option payload'
        if index < 3:
            assert separator, 'Adjacent option headers without original line separator'
        payloads.append(payload); separators.append(separator)
    def render(parts):
        return prefix + ''.join(h + p + s for h, p, s in zip(headers, parts, separators))
    assert render(payloads) == context
    moved = [payloads[index] for index in ORDER]
    result = render(moved)
    # Parse the actual constructed string against the declared position frame.
    cursor, recovered = len(prefix), []
    assert result.startswith(prefix)
    for header, payload, separator in zip(headers, moved, separators):
        assert result[cursor:cursor+len(header)] == header
        cursor += len(header)
        recovered.append(result[cursor:cursor+len(payload)])
        cursor += len(payload)
        assert result[cursor:cursor+len(separator)] == separator
        cursor += len(separator)
    assert cursor == len(result) and recovered == moved
    restored = [None] * 4
    for position, original in enumerate(ORDER):
        restored[original] = recovered[position]
    assert render(restored) == context, 'Fixed inverse failed; do not drop/rescue this QA'
    assert len(context) <= 6000 and len(result) <= 6000, 'No retrieval-confounded input admitted'
    return result, {'prefix_sha256':digest(prefix.encode()),
        'position_header_sha256':[digest(x.encode()) for x in headers],
        'position_separators':separators,
        'original_payload_sha256':[digest(x.encode()) for x in payloads],
        'cyclic_payload_sha256':[digest(x.encode()) for x in moved],
        'original_context_sha256':digest(context.encode()),
        'cyclic_context_sha256':digest(result.encode()), 'strict_inverse_exact':True,
        'header_payload_separator_definition':'Original position headers; last CRLF/LF is position separator; all other payload bytes unchanged'}


def inputs(binding, ledger):
    decision = json.loads(bound(binding['decision'], ledger))
    assert binding['decision']['sha256'] == DECISION_SHA
    assert decision['counts'] == {'accepted':45, 'excluded_fixed':7, 'uncertain':1}
    ids = decision['accepted_id_hashes']
    assert len(ids) == len(set(ids)) == 45
    assert digest(json.dumps(ids).encode()) == decision['accepted_ids_sha256']
    receipt = json.loads(bound(binding['inputs_receipt'], ledger))
    assert binding['inputs_receipt']['sha256'] == decision['evidence_sha256']['OPTION_ELIGIBILITY_INPUTS.json']
    assert receipt['private_pack_sha256'] == binding['masked_pack']['sha256'] == PACK_SHA
    pack = json.loads(bound(binding['masked_pack'], ledger))
    assert pack['original_gold_and_predictions_absent'] and len(pack['records']) == 53
    assert pack['intended_old_label_to_new_label'] == {'A':'B','B':'C','C':'D','D':'A'}
    masked = {x['id_sha256']:x for x in pack['records']}
    pools, tasks, manifests = {}, {}, {}
    for condition, (name, expected, count) in MANIFESTS.items():
        root = Path(binding['data_roots'][condition])
        assert root.is_absolute() and root.is_dir()
        path = relative(root, name)
        manifest = json.loads(bound({'path':str(path),'sha256':expected}, ledger))
        assert manifest['condition'] == condition and manifest['seed'] == 42
        assert manifest['formal_ready'] and not manifest['assets_pending']
        assert [t['task'] for t in manifest['tasks']] == list(range(1,20))
        task = manifest['tasks'][15]
        assert task['task'] == 16 and task['domain'] == 'finance' and task['max_new_tokens'] == 512
        item = task['files']['train']; assert item['count'] == count
        assert Path(item['path']).name == 'T16_train.jsonl'
        raw = bound({'path':str(relative(root,item['path'])),'sha256':item['sha256']}, ledger)
        if condition == 'pct01':
            assert item['sha256'] == receipt['train_file_sha256'] and item['path'] == receipt['train_file']
        lines = raw.splitlines(); assert len(lines) == count and all(lines)
        rows = {}
        for line in lines:
            row = json.loads(line); key = digest(row['id'].encode())
            assert key not in rows
            rows[key] = (row, digest(line))
        assert all(key in rows for key in ids)
        pools[condition], tasks[condition], manifests[condition] = rows, task, manifest
    assert tasks['pct01']['instruction'] == tasks['pct10']['instruction']
    rows, image_states = [], {}
    for key in ids:
        row, line_sha = pools['pct01'][key]
        assert canonical(row) == canonical(pools['pct10'][key][0]), 'Common TRAIN object differs'
        masked_row = masked[key]
        assert masked_row['source_row_sha256'] == line_sha and masked_row['original_screen_candidate']
        assert masked_row['question'] == row['question'] and masked_row['context'] == row['context']
        assert digest(json.dumps(row['images']).encode()) == masked_row['image_identifiers_sha256']
        assert row['dataset'] == 'FinMME' and row['metadata']['question_type'] == 'single_choice'
        changed, transform = cyclic_context(row['context'])
        assert [x['label'] for x in masked_row['options']] == list('ABCD')
        for name in row['images']:
            expected = binding['image_sha256'][name]
            for condition in MANIFESTS:
                path = relative(Path(binding['data_roots'][condition]), name)
                if str(path) not in image_states:
                    assert sha(path) == expected, str(path)
                    stat = path.stat()
                    image_states[str(path)] = {'sha256':expected,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        rows.append((row, changed, transform, line_sha))
    return ids, rows, tasks, manifests, image_states


def tensor_digest(items, torch):
    """Read exact tensor bytes; CPU transfer is serialization, not a model pass."""
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        meta = canonical([name, str(value.dtype), list(value.shape)])
        h.update(len(meta).to_bytes(8,'little')); h.update(meta)
        raw = value.reshape(-1).view(torch.uint8).numpy().reshape(-1)
        for start in range(0,raw.size,8*1024**2):
            h.update(raw[start:start+8*1024**2].tobytes())
        del value, raw
    return h.hexdigest()


def measure(items):
    seen, by_dtype, tensors = set(), {}, []
    coefficients = size = trainable = 0
    for name, value in items:
        storage = value.untyped_storage()
        key = (str(value.device),storage.data_ptr(),storage.nbytes())
        tensors.append({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),
                        'device':str(value.device),'numel':value.numel(),'requires_grad':value.requires_grad})
        if key in seen:
            continue
        seen.add(key); assert storage.nbytes() % value.element_size() == 0
        n, b = storage.nbytes() // value.element_size(), storage.nbytes()
        coefficients += n; size += b
        trainable += n if value.requires_grad else 0
        d = by_dtype.setdefault(str(value.dtype),{'coefficients':0,'bytes':0})
        d['coefficients'] += n; d['bytes'] += b
    return {'coefficients':coefficients,'bytes':size,'MiB':size/1024**2,
            'trainable_coefficients':trainable,'unique_storages':len(seen),'by_dtype':by_dtype,'tensors':tensors}


def model_state(model, torch):
    core = getattr(model,'core',model)
    parameters = list(core.named_parameters(remove_duplicate=False))
    buffers = list(core.named_buffers(remove_duplicate=False))
    assert parameters and all(p.is_cuda for _,p in parameters)
    assert all(p.grad is None for _,p in parameters)
    return {'weights':measure(parameters),'registered_buffers':measure(buffers),
            'parameters_sha256':tensor_digest(parameters,torch),
            'all_registered_buffers_sha256':tensor_digest(buffers,torch),
            'parameter_flags':[(n,p.requires_grad) for n,p in parameters]}


def structure(method, torch):
    from methods import SeqLinear
    model = method.model
    layers = {n:m for n,m in model.named_modules() if isinstance(m,SeqLinear)}
    expected = {f'model.layers.{i}.self_attn.{kind}_proj' for i in range(24) for kind in ('q','v')}
    assert set(layers) == expected and len(method.layers) == 48
    assert {id(x) for x in layers.values()} == {id(x) for x in method.layers}
    count = 0
    for name, layer in layers.items():
        out = 896 if name.endswith('q_proj') else 128
        assert tuple(layer.A.weight.shape) == (8,896) and tuple(layer.B.weight.shape) == (out,8)
        assert layer.scale == 4 and layer.dropout.p == .1 and not layer.training
        assert layer.weight.dtype == torch.float16
        assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        count += layer.A.weight.numel() + layer.B.weight.numel()
    assert count == 540672 and model.config.hidden_size == 896
    return {'qv_modules':48,'adapter_tensors':96,'adapter_coefficients':count,
            'rank':8,'scale':4,'dropout':.1,'unfolded_original_graph':True}


def run(args, root, output, binding, start):
    ledger = {}
    protocol = bound(binding['protocol'],ledger)
    assert set(binding['formal_proofs']) == set(binding['data_roots']) == set(MANIFESTS)
    proofs = {}
    for condition in MANIFESTS:
        ref = binding['formal_proofs'][condition]
        assert ref['sha256'] == PROOFS[condition]
        proofs[condition] = json.loads(bound(ref,ledger))
    ids, rows, tasks, manifests, image_states = inputs(binding,ledger)
    inventory = json.loads(bound(binding['source_inventory'],ledger))
    state = next(x for x in inventory['states'] if x['condition'] == args.condition)
    assert Path(state['root']) == root and binding['data_roots'][args.condition] == str(root)
    run_relative = Path(state['run']); original = relative(root,state['run'])
    assert run_relative.parts[0] == 'exp' and run_relative.name.startswith('training_')
    assert run_relative.parts[-4:-1] == ('onevision',args.condition,'SeqFT_LoRA')
    port = root / run_relative.parts[0] / run_relative.parts[1]
    config = json.loads(bound({'path':str(original/'config.json'),'sha256':state['config_sha256']},ledger))
    assert config['condition'] == args.condition and config['method'] == 'SeqFT_LoRA'
    assert config['backbone'] == 'onevision' and not config['preflight_only']
    assert config['manifest_sha256'] == MANIFESTS[args.condition][1] and config['seed'] == 42
    assert config['epochs'] == 1 and config['model_dtype'] == config['autocast_dtype'] == 'torch.float16'
    proof_ref = binding['formal_proofs'][args.condition]
    proof = proofs[args.condition]
    assert proof['candidate']['run'] == state['run'] and proof['config_sha256'] == state['config_sha256']
    assert proof['candidate']['backbone'] == 'onevision' and proof['candidate']['method'] == 'SeqFT_LoRA'
    assert proof['rescored_cells'] == 74 and proof['final_QA'] == 21454 and proof['positive_finite_training_stages'] == 19
    assert proof['primary_metric'] == 'existing raw Answer F1'
    observed = state['stages'][str(args.stage)]
    marker_name = f'stages/stage_{args.stage:02d}.json'
    meta_name = f'checkpoints/stage_{args.stage:02d}.json'
    assert proof['source_sha256'][str(run_relative/marker_name)] == observed[marker_name]['sha256']
    marker = json.loads(bound({'path':str(original/marker_name),'sha256':observed[marker_name]['sha256']},ledger))
    meta = json.loads(bound({'path':str(original/meta_name),'sha256':observed[meta_name]['sha256']},ledger))
    assert marker == observed[marker_name]['value'] and marker['complete'] and marker['weights_saved']
    assert marker['stage'] == args.stage and meta == observed[meta_name]['value']
    assert meta['stage'] == args.stage and meta['method'] == 'SeqFT_LoRA'
    assert meta['manifest_sha256'] == config['manifest_sha256'] and meta['full_architecture_and_runtime']
    assert not meta['optimizer_saved'] and meta['path'] == f'checkpoints/stage_{args.stage:02d}.pt'
    for name, expected in config['execution_source_sha256'].items():
        assert Path(name).name == name
        assert sha(port/name) == sha(original/'runner_source'/name) == expected, name
    source_validated_seconds = time.monotonic()-start
    # Same loader body as formal checkpoints.load, selecting stage metadata
    # instead of latest.json. Never rewrite the formal pointer or graph.
    sys.path.insert(0,str(port))
    import torch
    torch.set_num_threads(8); torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1 and os.environ.get('CUDA_VISIBLE_DEVICES')
    rt = importlib.import_module('runtime_utils')
    bridge = importlib.import_module('onevision_bridge')
    fv = importlib.import_module('frozen_vision')
    monitor_class = importlib.import_module('resource_monitor').ResourceMonitor
    assert rt.ROOT == fv.ROOT == root
    monitor = monitor_class(output/'resources')
    monitor.stage = args.stage
    generated_count = tokens = 0
    try:
        monitor.snapshot('before_checkpoint',args.stage)
        checkpoint = (original/meta['path']).resolve()
        assert checkpoint.is_relative_to((original/'checkpoints').resolve())
        assert checkpoint.stat().st_size == meta['bytes'] == observed['checkpoint_stat']['bytes']
        tick = time.monotonic(); assert sha(checkpoint) == meta['sha256']
        cp_sha_seconds = time.monotonic()-tick
        tick = time.monotonic()
        payload = torch.load(checkpoint,pickle_module=pickle,weights_only=False)
        assert payload['format'] == 'llmcl-method-graph-v1' and payload['stage'] == args.stage
        assert payload['checkpoint_phase'] == 'after_stage_evaluation' and payload['probe'] is None
        for field in ('method','backbone','condition','manifest_sha256','execution_source_sha256'):
            assert payload['config'][field] == config[field], field
        method, processor = payload['method'], payload['processor']
        if hasattr(method,'io'):
            method.io.run = original
        if hasattr(method,'state_root'):
            method.state_root = original/'method_state'
        rt.restore_rng(payload['rng'])
        model = method.model.eval()
        assert model.device == torch.device('cuda:0') and model.dtype == torch.float16
        assert method.name == 'SeqFT_LoRA'
        assert method.original_revision == 'standard-sequential-LoRA-qv-r8-alpha32-dropout0.1'
        bridge.bind(model); bridge.NATIVE_GATE_MODE = False
        method.set_instruction(16)
        cp_load_seconds = time.monotonic()-tick
        original_structure = structure(method,torch)
        tick = time.monotonic(); before = model_state(model,torch)
        before_seconds = time.monotonic()-tick
        generation_before = copy.deepcopy(model.core.language_model.generation_config.to_dict())
        monitor.snapshot('model_loaded_before_readouts',args.stage,model)
        predictions = output/'predictions.jsonl'
        generation_seconds = 0.
        with predictions.open('x',encoding='utf-8',newline='\n') as stream:
            for index, (source, changed, transform, line_sha) in enumerate(rows):
                pair_visual = None
                for view in ('original','cyclic'):
                    row = dict(source)
                    if view == 'cyclic':
                        row['context'] = changed
                    task = tasks[args.condition]
                    method.set_instruction(16)
                    batch, details = rt.input_batch(row,task,processor,model.device)
                    if hasattr(method,'vision_original'):
                        fv.prepare(batch,row,model,method.vision_original)
                        if hasattr(method,'prepare_row'):
                            method.prepare_row(row,batch)
                    assert not details['retrieved_chunk_indexes']
                    prepared = {name:{'sha256':tensor_digest([(name,value)],torch),
                                      'shape':list(value.shape),'dtype':str(value.dtype),'device':str(value.device)}
                                for name,value in batch.items()}
                    if row['images']:
                        assert 'pixel_values' in batch and 'image_grid_thw' in batch
                    else:
                        assert 'pixel_values' not in batch
                    visual = {name:prepared.get(name) for name in ('pixel_values','image_grid_thw')}
                    if pair_visual is None:
                        pair_visual = visual
                    else:
                        assert visual == pair_visual, 'Two views changed actual image features/layout'
                    monitor.snapshot('generation_before',args.stage)
                    tick = time.monotonic()
                    with torch.inference_mode(), torch.autocast('cuda',dtype=model.dtype):
                        generated = model.generate(**batch,do_sample=False,num_beams=1,
                            max_new_tokens=task['max_new_tokens'],use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    seconds = time.monotonic()-tick
                    response = generated[0,batch['input_ids'].shape[1]:]
                    raw = processor.tokenizer.decode(response,skip_special_tokens=True)
                    n = int(response.numel()); assert n <= 512
                    generated_count += 1; tokens += n; generation_seconds += seconds
                    record = {'condition':args.condition,'stage':args.stage,'index':index+1,
                        'id_sha256':ids[index],'source_id':source['id'],'view':view,
                        'source_train_line_sha256':line_sha,'common_QA_canonical_sha256':digest(canonical(source)),
                        'source_bytes_include_gold':True,'gold_used_for_generation_or_selection':False,
                        'question_sha256':digest(row['question'].encode()),
                        'context_sha256':digest(row['context'].encode()),
                        'instruction_sha256':digest(task['instruction'].encode()),
                        'image_identifiers_sha256':digest(json.dumps(row['images']).encode()),
                        'image_sha256':{name:binding['image_sha256'][name] for name in row['images']},
                        'raw_input_sha256':digest(canonical({'question':row['question'],'context':row['context'],
                            'images':row['images'],'instruction':task['instruction']})),
                        'transform':transform,'prepared_tensors':prepared,
                        'prepared_prompt_sha256':prepared['input_ids']['sha256'],
                        'prepared_prompt_token_ids':batch['input_ids'][0].tolist(),
                        'visual_features_sha256':prepared.get('pixel_values',{}).get('sha256'),
                        'raw_prediction':raw,'output_token_ids':response.tolist(),'output_tokens':n,
                        'generation_cap':512,'greedy':True,'num_beams':1,'use_cache':True,
                        'pad_token_id':processor.tokenizer.pad_token_id,
                        'eos_token_id':processor.tokenizer.eos_token_id,
                        'generation_seconds':seconds,'input_details':details,'scores_computed':0}
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                    monitor.snapshot('generation_after',args.stage)
                    del generated, response, batch
        assert generated_count == 90
        tick = time.monotonic(); after = model_state(model,torch)
        after_seconds = time.monotonic()-tick
        assert after == before and structure(method,torch) == original_structure
        generation_after = model.core.language_model.generation_config.to_dict()
        assert generation_after == generation_before
        for path, original_stat in image_states.items():
            stat = Path(path).stat()
            assert (stat.st_size,stat.st_mtime_ns) == (original_stat['bytes'],original_stat['mtime_ns'])
        monitor.snapshot('after_readouts_integrity',args.stage,model)
        write(output/'MODEL_STATE.json',{'before':before,'after':after,'unchanged':True,
            'original_structure':original_structure,'checkpoint_bytes':meta['bytes'],
            'original_dtype_and_unfolded_graph':True,'generation_before':generation_before,
            'generation_after':generation_after,'runtime_visual_cache_may_change':True})
        write(output/'SOURCE_BINDING.json',{'binding':binding,'source_files':ledger,
            'image_files':image_states,'source_inventory_observation_only_for_intermediate_payload':True,
            'checkpoint_payload_SHA_checked_once_this_phase':meta['sha256']})
    finally:
        monitor.done.set(); monitor.thread.join(timeout=6)
        monitor.snapshot('phase_finished',args.stage)
    resource_rows = [json.loads(line) for line in monitor.path.read_text().splitlines()]
    keys = ('VmRSS_bytes','Pss_bytes','cpu_uss_bytes','gpu_allocated_bytes','gpu_reserved_bytes',
            'gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_process_nvml_bytes')
    write(output/'RESOURCE_SUMMARY.json',{'unit':'bytes','rows':len(resource_rows),
        'observed_peak_bytes':{key:max((r[key] for r in resource_rows if r.get(key) is not None),default=None)
                               for key in keys},
        'scope':'Entire phase checkpoint/load/integrity/readouts sampling; not a controlled inference/latency benchmark',
        'no_cross_process_memory_subtraction':True})
    seconds = time.monotonic()-start
    assert seconds <= 1200, 'Fixed phase budget exhausted; no extension/resume'
    assert sha(Path(__file__)) == binding['worker_sha256'] and sha(args.binding) == args.binding_sha256
    write(output/'COMPLETE.json',{'status':'COMPLETE_POSITION_READOUT_ONLY','condition':args.condition,
        'stage':args.stage,'worker_sha256':binding['worker_sha256'],'binding_sha256':args.binding_sha256,
        'protocol_sha256':digest(protocol),'checkpoint_sha256':meta['sha256'],'checkpoint_bytes':meta['bytes'],
        'config_sha256':state['config_sha256'],'manifest_sha256':config['manifest_sha256'],
        'formal_proof_sha256':proof_ref['sha256'],'decision_sha256':DECISION_SHA,
        'ordered_45_ids':ids,'new_generations':90,'output_tokens':tokens,
        'new_scores':0,'optimizer_updates':0,'backwards':0,'TEST_QA':0,'CPU_model_passes':0,
        'full_parameters_and_registered_buffers_unchanged':True,
        'files':{name:sha(output/name) for name in ('predictions.jsonl','MODEL_STATE.json','SOURCE_BINDING.json','RESOURCE_SUMMARY.json')},
        'seconds':seconds,'cost_seconds':{'source_and_input_binding':source_validated_seconds,
            'checkpoint_first_SHA':cp_sha_seconds,'checkpoint_load_and_restore':cp_load_seconds,
            'before_integrity':before_seconds,'after_integrity':after_seconds,
            'generation_sum':generation_seconds,'timings_overlap_full_phase':True},
        'GPU':{'visible':os.environ['CUDA_VISIBLE_DEVICES'],'virtual':0,'name':torch.cuda.get_device_name(0),
               'uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid',None))},
        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate')},
        'TRAIN_diagnostic_not_heldout_or_causal':True,'quality_decision_deferred_until_all_four_states':True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=list(MANIFESTS),required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stage',type=int,choices=[15,16])
    group.add_argument('--stage15',dest='stage',action='store_const',const=15)
    group.add_argument('--stage16',dest='stage',action='store_const',const=16)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert os.environ.get('LLMCL_ROOT') and Path(os.environ['LLMCL_ROOT']).is_absolute()
    root = Path(os.environ['LLMCL_ROOT']).resolve()
    assert args.binding.is_absolute() and args.output.is_absolute()
    output = args.output.resolve()
    assert root not in output.parents and output != root and not output.exists() and output.parent.is_dir()
    binding_raw = args.binding.read_bytes()
    binding = json.loads(binding_raw)
    args.binding_sha256 = digest(binding_raw)
    assert binding['schema'] == 'finance-position-diagnostic-v1'
    assert binding['worker_sha256'] == sha(Path(__file__))
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    write(output/'STARTED.json',{'unix':time.time(),'pid':os.getpid(),'uid':os.getuid(),'pgid':os.getpgid(0),
        'condition':args.condition,'stage':args.stage,'limit_seconds':1200,
        'worker_sha256':binding['worker_sha256'],'binding_sha256':args.binding_sha256,
        'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
    try:
        run(args,root,output,binding,start)
    except BaseException:
        write(output/'FAILURE.json',{'unix':time.time(),'condition':args.condition,'stage':args.stage,
            'traceback':traceback.format_exc(),'no_automatic_resume_or_retry':True,'no_quality_claim':True})
        raise


if __name__ == '__main__':
    main()
