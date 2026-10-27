"""Full-data and nested 1% continual QA, with mandatory resource instrumentation."""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, get_cosine_schedule_with_warmup

from methods import ROOT, make_method
from runtime_utils import input_batch, answer_loss, evaluate, make_optimizer, activation_storage, write, restore_rng
from frozen_vision import install, prepare, stats as visual_cache_stats
from resource_monitor import ResourceMonitor
from checkpoints import save as save_checkpoint, load as load_checkpoint
from sparse_results import aggregate as aggregate_sparse, required_tasks
from training_validity import warmup_updates, parameter_digest, validate_stage
import onevision_bridge

HERE = Path(__file__).resolve().parent
METHODS = ['SeqFT_LoRA', 'SMoLoRA', 'PASs_MoE', 'PaRSP',
           'OSFT', 'SpaRTA', 'FV_guided_Training', 'ALER', 'ER_LoRA', 'CL_MoE']


def load_protocol(condition):
    policy = json.loads((HERE / 'execution_policy.json').read_text())
    info = policy['conditions'][condition]
    path = ROOT / info['manifest']
    manifest = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == info['manifest_sha256']
    lock=json.loads((path.parent.parent/'split_lock.json').read_text())['conditions'][condition]
    assert digest==lock['manifest_sha256']
    assert hashlib.sha256((path.parent/'frozen_ids.json').read_bytes()).hexdigest()==lock['frozen_ids_sha256']
    assert manifest['formal_ready'] and not manifest['assets_pending']
    assert manifest['condition'] == condition and manifest['seed'] == 42
    assert manifest['experiment_id'] == 'five_domain_native_cl_20261016_sparse' and len(manifest['tasks']) == policy['task_count']
    assert policy.get('training_enabled') is True
    assert policy.get('save_final_parameters') is True
    data = {}
    for task in manifest['tasks']:
        for split, f in task['files'].items():
            p = ROOT / f['path']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256']
            assert lock['files'][f['path']]==f['sha256']
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            assert len(rows) == f['count'] and len({r['id'] for r in rows}) == len(rows)
            data[task['task'], split] = rows
        assert not {x['id'] for x in data[task['task'], 'train']} & {x['id'] for x in data[task['task'], 'test']}
    return policy, manifest, digest, data


def aggregate(run, manifest):
    matrices = {score: [] for score in ['exact_match', 'token_f1', 'rougeL']}
    for stage in range(len(manifest['tasks'])+1):
        for score in matrices:
            row = []
            for t in manifest['tasks']:
                p = run / 'evaluations' / f'stage_{stage:02d}' / f"T{t['task']:02d}_test_predictions.scores.json"
                if not p.exists():
                    return
                result = json.loads(p.read_text())
                assert result['count'] == t['files']['test']['count']
                row.append(result[score])
            matrices[score].append(row)
    from result_tables import metrics
    import csv
    for score, values in matrices.items():
        with (run / f'{score}_{len(values)}x{len(manifest["tasks"])}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['stage']+[task['task_id'] for task in manifest['tasks']])
            writer.writerows([[stage]+row for stage,row in enumerate(values)])
    write(run / 'score_matrices.json', matrices)
    write(run / 'metrics_by_score.json', {s: metrics(a) for s, a in matrices.items()})


def pause_at_boundary(method, processor, run, stage, config, monitor, checkpoint=None):
    """Cooperative pause after a complete stage; never interrupt an update/evaluation."""
    request_path = run / 'PAUSE_REQUEST.json'
    if not request_path.exists():
        return False
    raw = request_path.read_bytes()
    request = json.loads(raw)
    assert isinstance(request, dict)
    assert isinstance(request.get('request_id'), str) and request['request_id'].strip()
    assert isinstance(request.get('reason'), str) and request['reason'].strip()
    assert not config['preflight_only']
    if stage == len(config['task_counts']):
        return False  # S19 already completes the run; preserve the pending request.
    if checkpoint is None:
        assert stage == 0
        checkpoint = save_checkpoint(method, processor, run, stage, config)
    assert checkpoint['stage'] == stage and checkpoint['full_architecture_and_runtime']
    assert checkpoint == json.loads((run / 'checkpoints/latest.json').read_text())
    history = run / 'pause_history'
    history.mkdir(exist_ok=True)
    request_sha = hashlib.sha256(raw).hexdigest()
    archived = history / ('request_' + request_sha + '.json')
    assert not archived.exists(), 'A pause request may only be consumed once'
    # Stop instrumentation before publication: paused means the process can exit normally.
    monitor.done.set()
    monitor.thread.join(timeout=6)
    assert not monitor.thread.is_alive()
    monitor.snapshot('stage_boundary_paused', stage, method.model)
    assert request_path.read_bytes() == raw, 'Pause request changed while being consumed'
    request_path.rename(archived)
    receipt_path = history / ('pause_S%02d_%d.json' % (stage, time.time_ns()))
    receipt = dict(status='paused_checkpoint', completed_stage=stage, unix=time.time(),
        request_id=request['request_id'], request_sha256=request_sha,
        request_path=str(archived.relative_to(run)), checkpoint=checkpoint,
        checkpoint_metadata_sha256=hashlib.sha256((run / 'checkpoints/latest.json').read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256((run / 'config.json').read_bytes()).hexdigest(),
        manifest_sha256=config['manifest_sha256'], execution_source_sha256=config['execution_source_sha256'],
        optimizer_lifecycle='fresh optimizer/scheduler/scaler at each stage',
        next_stage=stage + 1, full_stage_complete=True)
    write(receipt_path, receipt)
    write(run / 'run_status.json', dict(status='paused_checkpoint', completed_stage=stage,
        checkpoint=checkpoint, pause_receipt=str(receipt_path.relative_to(run)),
        pause_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest()))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=METHODS, required=True)
    ap.add_argument('--condition', choices=['pct10'], required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    policy, manifest, digest, data = load_protocol(args.condition)
    physical = int(os.environ['CUDA_VISIBLE_DEVICES'])
    host = os.environ['LLMCL_HOST']
    host_policy=policy['hosts'][host]
    assert physical in host_policy['conditions'].get(args.condition,[]) and physical not in host_policy['reserved_gpus']
    if args.method=='SMoLoRA':
        assert torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0]>=8, 'SMoLoRA requires native BF16; do not force unstable FP16 on2080Ti'
    assert policy['seed'] == 42
    run = args.run_dir.resolve()
    assert run.is_relative_to(HERE / 'runs')
    assert args.resume or not (run / 'config.json').exists(), 'Use --resume only with a matching durable stage checkpoint'
    paused = None
    if args.resume:
        previous = json.loads((run / 'run_status.json').read_text())
        if previous.get('status') == 'paused_checkpoint':
            receipt_path = (run / previous['pause_receipt']).resolve()
            assert receipt_path.is_relative_to((run / 'pause_history').resolve())
            assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == previous['pause_receipt_sha256']
            paused = json.loads(receipt_path.read_text())
            resume_path = run / 'RESUME_REQUEST.json'
            resume_raw = resume_path.read_bytes()
            assert json.loads(resume_raw)['pause_receipt_sha256'] == previous['pause_receipt_sha256']
            assert paused['checkpoint'] == json.loads((run / 'checkpoints/latest.json').read_text())
            assert hashlib.sha256((run / 'config.json').read_bytes()).hexdigest() == paused['config_sha256']
            assert hashlib.sha256((run / 'checkpoints/latest.json').read_bytes()).hexdigest() == paused['checkpoint_metadata_sha256']
    (run / 'logs').mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(run)
    monitor.snapshot('process_before_model')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    backbone = host_policy['backbone']
    model_path = ROOT / 'model' / ('LLaVA-OneVision-Qwen2-0.5B-OV' if backbone=='onevision' else 'Qwen2-VL-2B-Instruct')
    factory = onevision_bridge.Factory if backbone=='onevision' else Qwen2VLForConditionalGeneration
    resume_obj=load_checkpoint(run) if args.resume else None
    processor = resume_obj['processor'] if resume_obj else AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = resume_obj['method'].model if resume_obj else factory.from_pretrained(model_path, torch_dtype=torch.bfloat16 if args.method=='SMoLoRA' else torch.float16,
                 attn_implementation='sdpa', local_files_only=True).cuda()
    onevision_bridge.bind(model)
    onevision_bridge.NATIVE_GATE_MODE = args.preflight_only and backbone=='onevision'
    name = 'SeqLoRA' if args.method == 'SeqFT_LoRA' else args.method
    method = resume_obj['method'] if resume_obj else make_method(model, name, [t['instruction'] for t in manifest['tasks']])
    if args.method == 'SeqFT_LoRA':
        method.name = args.method
        method.original_revision = 'standard-sequential-LoRA-qv-r8-alpha32-dropout0.1'
    if hasattr(method, 'state_root'):
        method.state_root = run / 'method_state'
        method.save_checkpoints = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    if hasattr(method, 'bind_context') and not resume_obj:
        method.bind_context(processor, manifest, data, run, args.preflight_only)
    if not resume_obj: method.vision_original = install(model)
    task_counts = {t['task_id']: {s: f['count'] for s, f in t['files'].items()} for t in manifest['tasks']}
    config = {'experiment_id': HERE.name, 'condition': args.condition, 'method': args.method,
              'seed': 42, 'epochs': 1, 'manifest_sha256': digest, 'task_counts': task_counts,
              'save_checkpoints': True, 'checkpoint_all_stages': True,
              'checkpoint_retention': 'all_completed_stages',
              'save_final_parameters': True, 'batch_size': 16, 'micro_batch_size': 1,
              'host': host, 'evaluation': policy['evaluation'],
              'model_dtype': str(model.dtype), 'autocast_dtype': str(model.dtype),
              'gpu_name': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
              'base_model': str(model_path), 'backbone': backbone, 'framework': 'shared-v2-uniform-input-and-cache', 'image_area_cap': 401408, 'visual_cache_limit_bytes': 134217728, 'initial_loss_scale': 1, 'primary_score': 'token_f1',
              'test_fraction': 1.0, 'train_fraction': policy['conditions'][args.condition]['train_fraction'],
              'preflight_only': args.preflight_only,
              'execution_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    if resume_obj:
        for key in ['method','condition','manifest_sha256','execution_source_sha256']:
            assert config[key]==resume_obj['config'][key], 'Resume mismatch: '+key
        assert config['checkpoint_all_stages']
        assert not resume_obj['config']['preflight_only'] and resume_obj['probe'] is None
        assert 0 <= resume_obj['stage'] <= len(manifest['tasks'])
        if paused:
            assert paused['completed_stage'] == resume_obj['stage']
            assert paused['manifest_sha256'] == digest
            assert paused['execution_source_sha256'] == config['execution_source_sha256']
            archive = run / 'pause_history' / ('resume_' + hashlib.sha256(resume_raw).hexdigest() + '.json')
            assert not archive.exists()
            assert resume_path.read_bytes() == resume_raw
            resume_path.rename(archive)
        # The first interrupted stage has no durable weights. Discard its predictions before rerunning it.
        import shutil
        for p in (run/'evaluations').glob('stage_*'):
            if int(p.name.split('_')[-1])>resume_obj['stage']:
                target=run/'interrupted_evaluations'/str(time.time_ns())/p.name
                target.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(target))
    else:
        write(run / 'config.json', config)
        snapshot=run/'runner_source';snapshot.mkdir(exist_ok=True)
        for name,expected in config['execution_source_sha256'].items():
            blob=(HERE/name).read_bytes();assert hashlib.sha256(blob).hexdigest()==expected
            (snapshot/name).write_bytes(blob)
        (snapshot/'execution_policy.json').write_bytes((HERE/'execution_policy.json').read_bytes())
    gc.collect()
    torch.cuda.empty_cache()
    monitor.snapshot('model_loaded_before_initial_evaluation', model=model)
    write(run / 'run_status.json', {'status': 'resuming' if resume_obj else 'initial_evaluation',
          'completed_stage': resume_obj['stage'] if resume_obj else 0})
    if args.preflight_only:
        monitor.snapshot('training_before', model=model)
        # Test the same gradient and resource paths on a real new-task image and a text QA.
        method.begin_stage(1)
        optimizer = make_optimizer(method)
        scaler = torch.amp.GradScaler('cuda', init_scale=1, enabled=args.method!='SMoLoRA')
        preflight_cases=[]
        native_forward_cases=[]
        for tid in range(1,len(manifest['tasks'])+1):
            if args.method=='ER_LoRA':method.begin_stage(tid)
            method.set_instruction(tid)
            row = data[tid, 'train'][0]
            if backbone=='onevision':
                native_forward_cases.append(onevision_bridge.forward_gate(row, manifest['tasks'][tid-1], processor))
            batch, _ = input_batch(row, manifest['tasks'][tid-1], processor, model.device)
            prepare(batch, row, model, method.vision_original)
            if hasattr(method, 'prepare_row'):
                method.prepare_row(row, batch)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(method, 'prepare_update'):
                method.prepare_update(processor)
            if hasattr(method, 'recording'):
                method.recording(True)
            with activation_storage(method), torch.autocast('cuda', dtype=model.dtype):
                loss, _ = method.training_loss(batch, row, processor) if hasattr(method, 'training_loss') else answer_loss(model, batch, row['answer'], processor)
                if hasattr(method, 'extra_loss'):
                    loss = loss + method.extra_loss()
            if hasattr(method, 'recording'):
                method.recording(False)
            scaler.scale(loss).backward()
            if hasattr(method, 'post_backward'):
                method.post_backward(scaler.get_scale(), .5)
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            assert torch.isfinite(loss) and torch.isfinite(grad)
            scaler.step(optimizer)
            scaler.update()
            case={'event':'preflight','task':tid,'dataset':row['dataset'],'source_id':row['id'],
                  'image_count':len(row['images']),'loss':float(loss.detach()),'gradient_l2_preclip':float(grad)}
            preflight_cases.append(case)
            print(json.dumps(case),flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler, batch, loss
        if args.method=='CL_MoE':
            method.end_stage(1)
            method.begin_stage(2)
            method.end_stage(2)
        gc.collect()
        torch.cuda.empty_cache()
        monitor.snapshot('training_after', model=model)
        monitor.finish(model)
        # This diagnostic used one row per task, not a complete training stage.
        # Do not call end_stage: author statistics require the entire actual stage pool.
        model.eval(); task=manifest['tasks'][-1];row=data[task['task'],'train'][0]
        method.set_instruction(task['task']);batch,_=input_batch(row,task,processor,model.device)
        prepare(batch,row,model,method.vision_original)
        if hasattr(method,'prepare_row'):method.prepare_row(row,batch)
        with torch.no_grad(),torch.autocast('cuda',dtype=model.dtype): probe_loss,_=answer_loss(model,batch,row['answer'],processor)
        save_checkpoint(method,processor,run,1,config,{'task':task,'row':row,'loss':float(probe_loss)})
        write(run / 'gpu_preflight.json', {'passed':True,'weights_discarded':False,'not_formal_results':True,
              'task_ids':[c['task'] for c in preflight_cases],'cases':preflight_cases,'native_forward_cases':native_forward_cases,
              'manifest_sha256':digest,'execution_source_sha256':config['execution_source_sha256']})
        return
    if not resume_obj: evaluate(method, processor, manifest['tasks'], data, run, 0)
    gc.collect()
    torch.cuda.empty_cache()
    if not resume_obj: monitor.snapshot('training_before', model=model)
    if resume_obj:
        # Loading, rebinding and instrumentation must not shift the saved next-stage RNG.
        restore_rng(resume_obj['rng'])
    elif pause_at_boundary(method, processor, run, 0, config, monitor):
        return
    for stage in range(resume_obj['stage']+1 if resume_obj else 1, len(manifest['tasks'])+1):
        monitor.stage = stage
        monitor.snapshot('stage_before_auxiliary', stage, model)
        write(run / 'run_status.json', {'status': 'training', 'completed_stage': stage-1, 'active_stage': stage})
        method.begin_stage(stage)
        monitor.snapshot('stage_after_auxiliary', stage, model)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = make_optimizer(method)
        rows = data[stage, 'train']
        # Same deterministic order for all methods; independent of initialization RNG.
        order = sorted(rows, key=lambda x: hashlib.sha256(f"42:{stage}:{x['id']}".encode()).digest())
        assert hashlib.sha256(''.join(x['id']+'\n' for x in order).encode()).hexdigest()==manifest['tasks'][stage-1]['training_order_sha256']
        updates = math.ceil(len(order)/16)
        warmup = warmup_updates(updates)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, updates)
        parameter_before = parameter_digest(parameters)
        positive_updates = 0
        scaler = torch.amp.GradScaler('cuda', init_scale=1, enabled=args.method!='SMoLoRA')
        method.set_instruction(stage)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        with (run / 'logs/events.jsonl').open('a', encoding='utf-8') as events:
            for step, start in enumerate(range(0, len(order), 16), 1):
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                minibatch = order[start:start+16]
                if hasattr(method, 'prepare_update'):
                    method.prepare_update(processor)
                loss_sum = 0.
                target_tokens = prompt_tokens = 0
                for index, row in enumerate(minibatch):
                    batch, details = input_batch(row, manifest['tasks'][stage-1], processor, model.device)
                    prepare(batch, row, model, method.vision_original)
                    if hasattr(method, 'prepare_row'):
                        method.prepare_row(row, batch)
                    if hasattr(method, 'recording'):
                        method.recording(True)
                    with activation_storage(method), torch.autocast('cuda', dtype=model.dtype):
                        loss, length = method.training_loss(batch, row, processor) if hasattr(method, 'training_loss') else answer_loss(model, batch, row['answer'], processor)
                        if hasattr(method, 'extra_loss') and (getattr(method, 'extra_loss_per_micro', False) or index == len(minibatch)-1):
                            loss = loss + method.extra_loss()
                    if hasattr(method, 'recording'):
                        method.recording(False)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite loss')
                    scaler.scale(loss/len(minibatch)).backward()
                    if hasattr(method, 'post_backward'):
                        method.post_backward(scaler.get_scale(), start/len(order))
                    loss_sum += float(loss.detach())/len(minibatch)
                    target_tokens += length
                    prompt_tokens += details['prompt_tokens']
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if not torch.isfinite(grad):
                    raise FloatingPointError('nonfinite gradient')
                postclip=torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm(2)
                    for p in parameters if p.grad is not None]),2)
                scale = scaler.get_scale()
                lr_used = [group["lr"] for group in optimizer.param_groups]
                scaler.step(optimizer)
                scaler.update()
                positive_updates += int(any(value > 0 for value in lr_used))
                if scaler.get_scale() < scale:
                    raise FloatingPointError('optimizer step skipped')
                scheduler.step()
                if hasattr(method, 'after_update'):
                    method.after_update(step, updates)
                record = {'event': 'train_update', 'stage': stage, 'step': step,
                          'qa_seen': start+len(minibatch), 'total_qa': len(order), 'loss': loss_sum,
                          'gradient_l2_preclip': float(grad), 'gradient_l2_postclip': float(postclip),
                          'gradient_clip_threshold': 1., 'learning_rate_used': lr_used,
                          'learning_rate_after_scheduler': scheduler.get_last_lr(), 'warmup_updates': warmup,
                          'seconds': time.monotonic()-tick, 'prompt_tokens': prompt_tokens,
                          'target_tokens': target_tokens,
                          **visual_cache_stats(model), 'gpu_allocated_bytes': torch.cuda.memory_allocated(),
                          'gpu_reserved_bytes': torch.cuda.memory_reserved(),
                          'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                events.write(json.dumps(record)+'\n')
                events.flush()
                print(json.dumps(record), flush=True)
        monitor.snapshot('stage_training_end_with_optimizer', stage, model)
        validity = validate_stage(updates, positive_updates, parameter_before, parameter_digest(parameters))
        write(run / 'validity' / f'stage_{stage:02d}.json', validity)
        optimizer.zero_grad(set_to_none=True)
        if hasattr(method, 'end_stage'):
            method.end_stage(stage)
        del optimizer, scheduler, scaler, batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        monitor.snapshot('stage_training_end_persistent', stage, model)
        if stage == len(manifest['tasks']):
            monitor.snapshot('training_after', stage, model)
        write(run / 'run_status.json', {'status': 'evaluation', 'completed_stage': stage-1, 'active_stage': stage})
        wanted=required_tasks(stage,len(manifest['tasks']))
        scores = evaluate(method, processor, [t for t in manifest['tasks'] if t['task'] in wanted], data, run, stage)
        checkpoint=None
        if config['checkpoint_all_stages'] or stage==len(manifest['tasks']):
            checkpoint=save_checkpoint(method,processor,run,stage,config)
        write(run / 'stages' / f'stage_{stage:02d}.json', {'complete': True, 'weights_saved': checkpoint is not None,
              'stage': stage, 'task_scores': scores, 'train_qa': len(order), 'test_fraction': 1.0})
        monitor.snapshot('stage_after_evaluation', stage, model)
        write(run / 'run_status.json', {'status': 'stage_complete', 'completed_stage': stage})
        if pause_at_boundary(method, processor, run, stage, config, monitor, checkpoint):
            return
    aggregate_sparse(run, manifest, write)
    monitor.finish(model)
    write(run / 'run_status.json', {'status': 'saving_final_parameters', 'completed_stage': len(manifest['tasks'])})
    artifact=json.loads((run/'checkpoints/latest.json').read_text())
    write(run / 'run_status.json', {'status': 'completed', 'completed_stage': len(manifest['tasks']),
          'final_checkpoint': artifact['path'], 'final_parameter_file_bytes': artifact['bytes']})


if __name__ == '__main__':
    main()
