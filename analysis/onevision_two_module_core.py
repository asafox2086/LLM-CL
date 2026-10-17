"""Experimental two-module LoRA primitives; GPU behavior remains unvalidated.

Module A separates reuse of an old A basis from strength of its old BA function.
Module B accepts actual finite parameter displacements using TRAIN losses only.
Neither primitive selects on TEST, changes answers, nor asserts test gains.
"""
import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class TrainLoss:
    value: float
    ids: tuple[str, ...]
    split: str = 'train'

    def validate(self, allowed_ids):
        assert self.split == 'train' and self.ids and len(set(self.ids)) == len(self.ids)
        assert set(self.ids) <= set(allowed_ids)
        assert math.isfinite(self.value) and self.value >= 0


def clone(adapter):
    return {name: {factor: value.detach().clone() for factor, value in layer.items()}
            for name, layer in adapter.items()}


def reuse_function(adapter, strength):
    assert strength in (0., .5, 1.)
    result = clone(adapter)
    for layer in result.values():
        assert set(layer) == {'A', 'B'}
        layer['B'].mul_(strength)
    return result


def interpolate(start, proposed, fraction):
    assert 0 < fraction <= 1 and start.keys() == proposed.keys()
    result = {}
    for name in start:
        assert start[name].keys() == proposed[name].keys()
        result[name] = {}
        for factor, old in start[name].items():
            new = proposed[name][factor]
            assert old.shape == new.shape and old.dtype == new.dtype and old.device == new.device
            assert torch.isfinite(old).all() and torch.isfinite(new).all()
            result[name][factor] = (new.detach().clone() if fraction == 1 else
                                    old.detach() + fraction * (new.detach() - old.detach()))
    return result


def check_evidence(evidence, current_ids, memory_ids):
    current, old = evidence
    current.validate(current_ids)
    assert set(current.ids) == set(current_ids)
    assert old.keys() == memory_ids.keys()
    for task, loss in old.items():
        loss.validate(memory_ids[task])
        assert set(loss.ids) == set(memory_ids[task])


def old_guard(old, references, tolerance):
    assert tolerance >= 0 and old.keys() == references.keys()
    assert all(math.isfinite(v) and v >= 0 for v in references.values())
    return all(loss.value <= references[task] + tolerance for task, loss in old.items())


def choose_reuse(start, evaluate: Callable, current_ids, memory_ids,
                 references: Mapping[str, float], guard_enabled=False, tolerance=.02):
    """Evaluate all fixed strengths; ties preserve more of the old function.

    evaluate is a TRAIN-only loss callback over immutable predeclared cohorts.
    This chooses initialization, not a TEST-time router or optimal module count.
    """
    assert not set(current_ids) & {i for ids in memory_ids.values() for i in ids}
    records = []
    eligible = []
    for strength in (1., .5, 0.):
        candidate = reuse_function(start, strength)
        current, old = evaluate(candidate)
        check_evidence((current, old), current_ids, memory_ids)
        permitted = not guard_enabled or old_guard(old, references, tolerance)
        records.append({'strength': strength, 'current_TRAIN_NLL': current.value,
                        'old_TRAIN_NLL': {t: v.value for t, v in old.items()}, 'eligible': permitted})
        if permitted:
            eligible.append((current.value, -strength, candidate))
    assert eligible, 'No initialization satisfies TRAIN guard; save failure, do not silently relax it'
    current, negative_strength, candidate = min(eligible, key=lambda row: row[:2])
    return candidate, {'chosen_strength': -negative_strength, 'all_candidates': records,
                       'extra_TRAIN_forward_evaluations': 3, 'uses_TEST': False}


def accept_displacement(start, proposed, evaluate: Callable, current_ids, memory_ids,
                        references: Mapping[str, float], tolerance=.02):
    """Check the actual bilinear LoRA function at fixed finite step fractions.

    Caller owns optimizer/RNG rollback. A rejected step MUST restore both; this
    helper never mutates start/proposed or optimizer state. Zero accepted updates
    for an entire task is an invalid training result, not a retention success.
    """
    assert not set(current_ids) & {i for ids in memory_ids.values() for i in ids}
    initial, old = evaluate(start)
    check_evidence((initial, old), current_ids, memory_ids)
    records = []
    for fraction in (1., .5, .25, .125):
        candidate = interpolate(start, proposed, fraction)
        current, old = evaluate(candidate)
        check_evidence((current, old), current_ids, memory_ids)
        permitted = current.value <= initial.value and old_guard(old, references, tolerance)
        records.append({'fraction': fraction, 'current_TRAIN_NLL': current.value,
                        'old_TRAIN_NLL': {t: v.value for t, v in old.items()}, 'accepted': permitted})
        if permitted:
            return candidate, {'accepted_fraction': fraction, 'trials': records,
                               'current_before': initial.value, 'uses_TEST': False}
    return clone(start), {'accepted_fraction': 0., 'trials': records,
                          'current_before': initial.value, 'uses_TEST': False,
                          'caller_must_restore_optimizer_and_RNG': True}
