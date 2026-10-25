"""Frozen readout operators reused without scoring or architecture assumptions."""
import gc,hashlib
DOMAINS=('natural','medical','geography','finance','traffic')
def _device(device):
    import torch
    value = torch.device(device)
    assert value.type == 'cuda', 'CPU is restricted to I/O/statistics'
    return torch.device('cuda', torch.cuda.current_device() if value.index is None else value.index)

def _rows(rows):
    assert len(rows) == len({r['id'] for _, r in rows}) == 100
    assert all((sum((t['domain'] == d for t, _ in rows)) == 20 for d in DOMAINS))

def _batch(model, task, row, storage, batch_provider):
    import torch
    batch, details = batch_provider(model, task, row)
    assert isinstance(details, dict) and isinstance(details['raw_input_sha256'], str)
    payload = {n: v for n, v in batch.items() if not n.startswith('_')}
    assert all((isinstance(v, torch.Tensor) and v.is_cuda for v in payload.values()))
    assert all((not v.is_floating_point() or v.dtype == torch.float16 for v in payload.values()))
    assert payload['input_ids'].shape[0] == 1
    prepared = storage.digest(payload)
    if 'prepared_sha256' in details:
        assert details['prepared_sha256'] == prepared
    binding = dict(details)
    binding.update(prepared_sha256=prepared, prompt_tokens=payload['input_ids'].shape[1], input_shape=list(payload['input_ids'].shape), input_tensor_shapes={n: list(v.shape) for n, v in payload.items()})
    return (payload, binding)

def _generate(model, payload, cap, generation_policy, fixed=False):
    import torch

    class FiniteScores:

        def __call__(self, input_ids, scores):
            assert not torch.isnan(scores).any() and (not torch.isposinf(scores).any())
            assert torch.isfinite(scores).any(dim=-1).all()
            return scores
    length = payload['input_ids'].shape[1]
    limit = generation_policy.get('context_limit')
    assert limit is None or length + cap <= limit
    options = dict(do_sample=False, num_beams=1, use_cache=True, max_new_tokens=cap, eos_token_id=generation_policy['generation_stop_ids'], pad_token_id=generation_policy['pad_id'], logits_processor=[FiniteScores()])
    if fixed:
        options['min_new_tokens'] = cap
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        output = model.generate(**payload, **options)
    tokens = output[0, length:].tolist()
    assert len(tokens) == cap if fixed else len(tokens) <= cap
    return tokens

def evaluate_dev(model, processor, rows, storage, *, batch_provider, generation_policy, kind='native'):
    """Exactly 100 original DEV generations; no NLL or scoring."""
    assert kind == 'native'
    _rows(rows)
    model.eval()
    records = []
    for task, row in rows:
        batch, binding = _batch(model, task, row, storage, batch_provider)
        tokens = _generate(model, batch, task['max_new_tokens'], generation_policy)
        records.append(dict(binding, id=row['id'], task=task['task'], domain=task['domain'], prediction=processor.tokenizer.decode(tokens, skip_special_tokens=True), references=row['answers'], generation_cap=task['max_new_tokens'], output_tokens=tokens))
        del batch, tokens
    return (records, dict(new_generations=100, new_scores=0, NLL_requests=0, backwards=0, optimizer_updates=0, native_template=True))

def resource_windows(model, processor, rows, storage, *, batch_provider, generation_policy, kind='native', prior_windows=None):
    """Five hash-min raw IDs, 32 tokens, fresh-load order fixed by caller.

    Call before any 100-DEV readout in the parent/final-native processes.
    Allocation windows include preparation and actual image/decoder execution.
    """
    import torch
    _rows(rows)
    device = _device(next(model.parameters()).device)
    model.eval()
    prior = None if prior_windows is None else {x['id']: x for x in prior_windows}
    if prior is not None:
        assert len(prior) == 5
    values = []
    for domain in DOMAINS:
        task, row = min(((t, r) for t, r in rows if t['domain'] == domain), key=lambda tr: hashlib.sha256(tr[1]['id'].encode()).digest())
        if kind == 'parent':
            import onevision_bridge as bridge
            bridge.LAST_NATIVE = None
            assert model._llmcl_visual_cache_bytes == 0
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            batch, binding = _batch(model, task, row, storage, batch_provider)
            if prior is not None:
                old = prior[row['id']]
                assert old['domain'] == domain and old['new_tokens'] == 32
                assert binding['raw_input_sha256'] == old['raw_input_sha256']
            tokens = _generate(model, batch, 32, generation_policy, True)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
        values.append(dict(binding, id=row['id'], domain=domain, new_tokens=32, baseline_allocated=baseline, peak_allocated=peak, peak_reserved=reserved))
        del batch, tokens
    return (values, dict(new_generations=5, new_scores=0, NLL_requests=0, fixed_new_tokens=32, actual_device=str(device), window_order='fresh_reload_before_any_DEV', hash_min_raw_ids=True, cross_backbone_prepared_equality_required=False))
