"""New-source epoch engine for full-stream two-module experiments.

Control path mirrors the immutable original AdamW/cosine/batch16 trainer.
Diagnostic callbacks preserve Python/NumPy/Torch RNG; no model/QA is created here.
"""
import copy
import math
import random

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from onevision_two_module_core import accept_displacement


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def isolated(function, *args):
    state = rng_state()
    try:
        return function(*args)
    finally:
        restore_rng(state)


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return copy.deepcopy(value)


def epoch(named_parameters, ordered_rows, loss_function, snapshot, restore,
          evaluate=None, current_ids=(), memory_ids=None, references=None,
          use_B=False, on_update=None):
    """One original epoch, with actual finite-step protection when enabled.

    snapshot/restore operate on A/B dictionaries; evaluation receives TRAIN only.
    on_update receives preclip gradient plus actual start/proposed/accepted A/B.
    Scheduler advances per attempted minibatch; rejected steps undo optimizer/RNG.
    """
    assert ordered_rows
    parameters = [p for _, p in named_parameters]
    assert parameters and all(p.requires_grad for p in parameters)
    updates = math.ceil(len(ordered_rows)/16)
    warmup = min(int(.03*updates), max(0, updates-1))
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, updates)
    records = [];effective = 0
    for offset in range(0, len(ordered_rows), 16):
        optimizer.zero_grad(set_to_none=True)
        start = snapshot()
        before_optimizer = copy.deepcopy(optimizer.state_dict()) if use_B else None
        before_rng = rng_state()
        batch = ordered_rows[offset:offset+16];values = [];tokens = 0
        for row in batch:
            loss, details = loss_function(row)
            assert torch.isfinite(loss)
            (loss/len(batch)).backward()
            values.append(float(loss.detach()));tokens += details['target_tokens']
        raw = torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None
                         else torch.zeros(p.numel()) for p in parameters])
        assert torch.isfinite(raw).all()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(norm)
        used_lr = optimizer.param_groups[0]['lr']
        optimizer.step()
        proposed = snapshot()
        if use_B:
            assert evaluate is not None and memory_ids and references
            accepted, decision = isolated(accept_displacement, start, proposed, evaluate,
                                          current_ids, memory_ids, references)
            restore(accepted)
            if decision['accepted_fraction'] == 0:
                optimizer.load_state_dict(before_optimizer)
                restore_rng(before_rng)
        else:
            accepted, decision = proposed, {'accepted_fraction': 1., 'disabled': True}
        scheduler.step()
        changed = any(not torch.equal(accepted[n][f], start[n][f]) for n in start for f in ('A', 'B'))
        positive = used_lr > 0 and decision['accepted_fraction'] > 0 and changed
        effective += int(positive)
        event = {'update': len(records)+1, 'train_ids': [r['id'] for r in batch],
                 'loss': sum(values)/len(values), 'target_tokens': tokens,
                 'actual_learning_rate_used': used_lr, 'warmup_updates': warmup,
                 'gradient_l2_preclip': float(norm), 'decision': decision,
                 'parameters_changed': changed, 'positive_effective_update': positive}
        records.append(event)
        if on_update is not None:
            isolated(on_update, event, raw, {'start': start, 'proposed': proposed, 'accepted': accepted})
    optimizer.zero_grad(set_to_none=True)
    result = {'events': records, 'attempted_updates': len(records), 'effective_updates': effective,
              'original_TRAIN_QA': len(ordered_rows), 'warmup_updates': warmup,
              'optimizer_state': cpu_state(optimizer.state_dict()), 'RNG_after': cpu_state(rng_state())}
    del optimizer, scheduler
    return result
