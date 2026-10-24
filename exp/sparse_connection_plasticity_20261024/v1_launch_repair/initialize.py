"""One shared S1-only initialization. Imported only after actual STARTED."""
import gc
import hashlib
import json
import math
import time
import torch
from core import Packed2of4Linear
from pruning import load_pruner, prune_fixed
from common import ASSETS, BASE_VALUES, FFN_SLOTS, PORT, sha, tensor_digest, write


def prepared_digest(batch, row, processor):
    target = processor.tokenizer(row['answer'], add_special_tokens=False,
        truncation=True, max_length=512)['input_ids'] + [processor.tokenizer.eos_token_id]
    # Exact prepared tensors are checked on the 16 diagnostic QA, not inferred
    # from matching raw IDs or shapes. This GPU->CPU work is charged.
    return dict(prepared_tensors_sha256=tensor_digest(list(batch.items())),
        target_sha256=hashlib.sha256(json.dumps(target).encode()).hexdigest(), target_tokens=len(target))


def diagnose(model, processor, formal, rows, task, checked_input):
    records = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            batch, details = checked_input(row, task, processor, model.device)
            proof = prepared_digest(batch, row, processor)
            with torch.autocast('cuda', dtype=torch.float16):
                loss, tokens = formal.answer_loss(model, batch, row['answer'], processor)
            assert tokens == proof['target_tokens'] and bool(torch.isfinite(loss))
            records.append(dict(id_sha256=hashlib.sha256(row['id'].encode()).hexdigest(),
                mean_answer_token_NLL=float(loss), **proof))
    return dict(QA=len(records), QA_equal_mean_NLL=sum(r['mean_answer_token_NLL'] for r in records) / len(records),
        records=records, aggregation='per-QA token mean, then QA-equal mean; natural log; EOS included')


def initialize(model, processor, formal, calibration, diagnostic, task, checked_input, monitor, output):
    assert len(calibration) == len(diagnostic) == 16
    assert sum(p.numel() for p in model.parameters()) == BASE_VALUES
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    def compatibility():
        formal.onevision_bridge.NATIVE_GATE_MODE = True
        try:
            return formal.onevision_bridge.forward_gate(diagnostic[0], task, processor)
        finally:
            formal.onevision_bridge.NATIVE_GATE_MODE = False
            formal.onevision_bridge.LAST_NATIVE = None
    dense_gate = compatibility()
    dense = diagnose(model, processor, formal, diagnostic, task, checked_input)
    source = json.loads((ASSETS / 'SOURCE_RECEIPT.json').read_text())
    pruner_class = load_pruner(ASSETS / 'sparsegpt.py', source['files']['sparsegpt.py']['sha256'], observe_mask=True)
    assert not pruner_class._llmcl_CPU_sync_shim
    records = []; requests = 0
    for block in range(24):
        started = time.monotonic()
        names = [f'model.layers.{block}.mlp.{name}' for name in ('gate_proj', 'up_proj', 'down_proj')]
        solvers = {name: pruner_class(model.get_submodule(name)) for name in names}
        handles = []
        for name in names:
            def capture(module, arguments, result, name=name):
                solvers[name].add_batch(arguments[0].detach(), result.detach())
            handles.append(model.get_submodule(name).register_forward_hook(capture))
        try:
            with torch.no_grad():
                for row in calibration:
                    batch, _ = checked_input(row, task, processor, model.device)
                    with torch.autocast('cuda', dtype=torch.float16):
                        loss, _ = formal.answer_loss(model, batch, row['answer'], processor)
                    assert bool(torch.isfinite(loss))
                    requests += 1
        finally:
            for handle in handles:
                handle.remove()
        for name in names:
            solver = solvers[name]
            assert solver.nsamples == 16
            keep = prune_fixed(solver)
            base = model.get_submodule(name)
            shape = list(base.weight.shape)
            assert sorted(shape) == [896, 4864]
            support_sha = tensor_digest([('keep', keep)])
            packed = Packed2of4Linear(base, keep)
            assert torch.equal(packed.materialize(), base.weight.detach())
            parent, child = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), child, packed)
            records.append(dict(name=name, shape=shape, calibration_QA=16,
                exact_author_support_sha256=support_sha, layer_layout=packed.metadata(),
                physical_pack_materialization_exact=True))
            del base, keep, packed, solver
        solvers.clear()
        del handles, solvers, batch, loss
        gc.collect(); torch.cuda.empty_cache()
        monitor.snapshot('initialization_block_pruned', block + 1, model)
        write(output / f'block_{block + 1:02d}.json', dict(block=block + 1, seconds=time.monotonic() - started,
            cumulative_calibration_requests=requests, layers=records[-3:]))
    assert requests == 384 and len(records) == 72
    assert sum(p.numel() for p in model.parameters()) == BASE_VALUES - FFN_SLOTS
    assert not any(name.endswith(tuple('.mlp.' + t + '.weight' for t in ('gate_proj', 'up_proj', 'down_proj')))
                   for name in model.state_dict())
    compact_gate = compatibility()
    compact = diagnose(model, processor, formal, diagnostic, task, checked_input)
    for before, after in zip(dense['records'], compact['records']):
        for key in ('id_sha256', 'prepared_tensors_sha256', 'target_sha256', 'target_tokens'):
            assert before[key] == after[key], key
    gap = compact['QA_equal_mean_NLL'] - dense['QA_equal_mean_NLL']
    result = dict(complete=True, dense_diagnostic=dense, compact_diagnostic=compact,
        NLL_gap=gap, resource_gate_limit_nats=math.log(2),
        resource_gate_passed=math.isfinite(gap) and gap <= math.log(2),
        calibration_requests=384, diagnostic_requests=32, compatibility_NLL_paths=4,
        compatibility_short_generations=4, total_extra_teacher_forced_requests=420,
        dense_native_gate=dense_gate, compact_native_gate=compact_gate,
        no_future_task_calibration=True, no_optimizer_or_training_yet=True,
        no_claim_of_AnswerF1_preservation=True, source_receipt_sha256=sha(ASSETS / 'SOURCE_RECEIPT.json'),
        physical_layers=records)
    write(output / 'INITIAL_FUNCTION.json', result)
    return result
