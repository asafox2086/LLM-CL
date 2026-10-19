"""One independent, image-disjoint FIT/DEV screen. No official TEST access."""
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import traceback
from types import SimpleNamespace
from study import ROOT, PORT, REPORT, WORK, CASES, sha, write, original_train, split_train, training_ast, make_method


class ResourceBusy(RuntimeError): pass


def run(case):
    sys.path.insert(0, str(ROOT/'analysis'))
    import onevision_research_device_entry as device
    cards = device.inventory(); ours = {g for g, v in cards.items() if any(p['uid'] == os.getuid() for p in v['owners'])}
    if cards[1]['owners'] or cards[1]['memory_MiB'] >= 500 or cards[1]['utilization'] >= 10 or (ours & {0,1}) or len(ours | {1}) > 3:
        raise ResourceBusy('GPU inventory changed before STARTED/CUDA; healthy wait')
    ready = json.loads((REPORT/'READY.json').read_text())
    assert all(sha(ROOT/p) == h for p, h in ready['sources'].items())
    assert all(sha(ROOT/p) == h for p, h in ready['base_files'].items())
    assert sha(REPORT/'PROTOCOL.md') == ready['protocol_sha256']
    assert sha(REPORT/'SPLIT.json') == ready['split_sha256']
    assert sha(REPORT/'CPU_GATE.json') == ready['CPU_GATE_sha256']
    marker = REPORT/(case+'_STARTED.json'); assert not marker.exists(), 'Preserve failed attempt; no blind retry'
    write(marker, dict(pid=os.getpid(), startticks=Path('/proc/self/stat').read_text().split()[21], inventory=cards))
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['OMP_NUM_THREADS'] = '8'; os.environ['MKL_NUM_THREADS'] = '8'
    import torch, numpy as np
    from transformers import AutoProcessor, get_cosine_schedule_with_warmup
    sys.path.insert(0, str(PORT))
    import importlib.util
    spec = importlib.util.spec_from_file_location('increment_formal_run', PORT/'run.py')
    formal = importlib.util.module_from_spec(spec); spec.loader.exec_module(formal); sys.modules['run'] = formal
    path, manifest, original = original_train(); assert sha(path) == ready['manifest_sha256']
    split, fit, dev = split_train(manifest, original)
    assert split == json.loads((REPORT/'SPLIT.json').read_text())
    data = {}
    for task in manifest['tasks']:
        t = task['task']; data[t, 'train'] = fit[t]; data[t, 'dev'] = dev[t]
        order = sorted(fit[t], key=lambda r: hashlib.sha256(f"42:{t}:{r['id']}".encode()).digest())
        task['training_order_sha256'] = hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        task['files'] = {'train': {'count':len(fit[t])}, 'dev': {'count':len(dev[t])}}
    run = WORK/case; run.mkdir(parents=True); (run/'logs').mkdir()
    monitor = formal.ResourceMonitor(run); monitor.snapshot('process_before_model')
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); torch.set_num_threads(8)
    base = ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    model = formal.onevision_bridge.Factory.from_pretrained(base, torch_dtype=torch.float16,
        attn_implementation='sdpa', local_files_only=True).cuda()
    formal.onevision_bridge.bind(model)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    method = make_method(model, case, [t['instruction'] for t in manifest['tasks']])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.enable_input_require_grads()
    method.vision_original = formal.install(model)
    config = dict(method=case, manifest_sha256=ready['manifest_sha256'], split_sha256=ready['split_sha256'],
        FIT_QA=ready['FIT_QA'], DEV_QA=ready['DEV_QA'], TEST_QA=0,
        dtype='FP16 base and outer autocast; FP32 LoRA or OSFT factors as in formal port',
        learning_rate=CASES[case]['learning_rate'], trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        shared_qv_initialization=True, seed=42, rank_for_LoRA=8, model_training_not_a_full_main_run=True)
    write(run/'config.json', config)
    formal.onevision_bridge.NATIVE_GATE_MODE = True
    gates = [formal.onevision_bridge.forward_gate(fit[t['task']][0], t, processor) for t in manifest['tasks']]
    formal.onevision_bridge.NATIVE_GATE_MODE = False
    write(run/'NATIVE_GATE.json', dict(complete=True, gates=gates, FIT_QA=19, TEST_QA=0))
    # All pilot conditions start TRAIN with the same external RNG state.
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    monitor.snapshot('training_before', model=model)
    namespace = dict(vars(formal), model=model, method=method, processor=processor, manifest=manifest, data=data,
        run=run, monitor=monitor, args=SimpleNamespace(method=method.name), resume_obj=None)
    prior = None
    def save_current(stage):
        nonlocal prior
        meta = formal.save_checkpoint(method, processor, run, stage, config)
        if prior is not None:
            old = (run/prior['path']).resolve(); assert old.is_relative_to((run/'checkpoints').resolve())
            old.unlink()
        prior = meta
    namespace['save_current'] = save_current
    prefix, h = training_ast(); assert h == ready['train_AST_sha256']
    tail = ast.parse('save_current(stage)').body
    node = ast.For(target=ast.Name(id='stage',ctx=ast.Store()),
        iter=ast.Call(func=ast.Name(id='range',ctx=ast.Load()),args=[ast.Constant(1),ast.Constant(20)],keywords=[]),
        body=prefix+tail, orelse=[])
    raw_clip = torch.nn.utils.clip_grad_norm_; counts = {}
    def record_raw(parameters, *args, **kwargs):
        stage = monitor.stage; counts[stage] = counts.get(stage,0)+1
        grads = {n:dict(norm=float(p.grad.detach().float().norm()), finite=bool(torch.isfinite(p.grad).all()))
                 for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert grads and all(v['finite'] for v in grads.values())
        write(run/'gradients'/f'S{stage:02d}_{counts[stage]:03d}.json', dict(raw=grads))
        if counts[stage] == 1:
            torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},
                       run/'gradients'/f'S{stage:02d}_first_raw.pt')
        return raw_clip(parameters,*args,**kwargs)
    torch.nn.utils.clip_grad_norm_ = record_raw
    try: exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])), '<original-formal-TRAIN-prefix>', 'exec'), namespace)
    finally: torch.nn.utils.clip_grad_norm_ = raw_clip
    events = [json.loads(s) for s in (run/'logs/events.jsonl').read_text().splitlines()]
    assert len(events) == ready['updates'] and all(any(lr>0 for lr in e['learning_rate_used']) for e in events)
    assert len(counts) == 19
    checkpoint = json.loads((run/'checkpoints/latest.json').read_text())
    write(run/'TRAIN_COMPLETE.json', dict(complete=True, checkpoint=checkpoint, updates=len(events), TEST_QA=0))
    tasks = [t for t in manifest['tasks'] if dev[t['task']]]
    formal.evaluate(method, processor, tasks, data, run, 19, split='dev')
    domains = {}; files = {}
    for t in tasks:
        p = run/'evaluations/stage_19'/f"T{t['task']:02d}_dev_predictions.jsonl"
        records = [json.loads(s) for s in p.read_text().splitlines()]
        assert [v['id'] for v in records] == [v['id'] for v in dev[t['task']]]
        for v, gold in zip(records, dev[t['task']]):
            assert v['references'] == gold['answers']
            f1 = formal.evaluate.__globals__['score'](v['prediction'],gold['answers'])['token_f1']
            assert v['scores']['token_f1'] == f1
            domains.setdefault(t['domain'],[]).append(f1)
        files[str(p.relative_to(ROOT))] = sha(p)
    memory = monitor.finish(model)
    result = {k:dict(F1=sum(v)/len(v),QA=len(v)) for k,v in domains.items()}; assert len(result) == 5
    write(REPORT/(case+'_COMPLETE.json'), dict(complete=True, case=case, FIT_QA=ready['FIT_QA'], DEV_QA=ready['DEV_QA'],
        TEST_QA=0, updates=len(events), domains=result, domain_macro=sum(v['F1'] for v in result.values())/5,
        checkpoint=checkpoint, checkpoint_root=str(run.relative_to(ROOT)), prediction_sha256=files,
        source_sha256=sha(__file__), ready_sha256=sha(REPORT/'READY.json'), memory_summary=memory,
        deployment_NOT_measured=True, not_full_TEST_or_formal_main_result=True))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--case', choices=CASES, required=True); a = p.parse_args()
    try: run(a.case)
    except ResourceBusy as e: print(str(e), flush=True); sys.exit(75)
    except Exception:
        write(REPORT/(a.case+'_FAILURE.json'), dict(traceback=traceback.format_exc(), no_blind_retry=True)); raise
