"""Pinned FOREVER kernels for an independent, not-yet-admitted multimodal port.

No model loading, CUDA selection, dataset reading or training happens on import.
The upstream selector, drift callback and regularized loss are compiled without
editing their AST. The loss bridge supplies our existing answer-only image loss
in place of the text-only Hugging Face Trainer superclass.
"""
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
REVISION = 'bbeb549a21e59d9490f20a2b02099ee766a67f48'


def source(relative):
    folder = ROOT / 'code/FOREVER'
    manifest = json.loads((folder / 'source_manifest.json').read_text())
    assert manifest['revision'] == REVISION
    expected = next(x['sha256'] for x in
                    json.loads((folder / 'files_sha256.json').read_text())
                    if x['path'] == relative)
    path = folder / 'source' / relative
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, relative
    return path, ast.parse(raw.decode('utf-8')), expected


class DatasetBridge:
    """Only the Dataset operations used by the unchanged author selector."""
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, column):
        assert column == 'output'
        return [row['answer'] for row in self.rows]

    def select(self, indices):
        return DatasetBridge([self.rows[i] for i in indices])


def select_history(train_tasks, stage):
    """Original 2%-per-answer-class selector, original TRAIN order/strings.

    stage is one-based. Future task rows are never supplied to the selector.
    The upstream random.seed(42) side effect is retained and documented.
    """
    assert 1 <= stage <= len(train_tasks)
    if stage == 1:
        return [], None
    history = [DatasetBridge(rows) for rows in train_tasks[:stage-1]]
    ids = [row['id'] for task in history for row in task.rows]
    assert len(ids) == len(set(ids)) and ids
    path, tree, digest = source('utils/load_data.py')
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'load_memory_buffer')
    names = [f'T{i+1:02d}' for i in range(len(history))]
    lookup = {os.path.join('/FOREVER_TRAIN_BRIDGE', 'train', name+'.json'): task
              for name, task in zip(names, history)}

    def load_dataset(kind, data_files, cache_dir):
        assert kind == 'json' and cache_dir is None
        return {'train': lookup[data_files]}

    namespace = {
        'Counter': Counter, 'random': random,
        'os': SimpleNamespace(path=SimpleNamespace(join=os.path.join,
                                                   exists=lambda p: p in lookup)),
        'get_dataset_order': lambda _: names, 'load_dataset': load_dataset,
        'concatenate_datasets': lambda datasets: DatasetBridge(
            row for dataset in datasets for row in dataset.rows),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    chosen = namespace['load_memory_buffer'](1, stage-1, '/FOREVER_TRAIN_BRIDGE',
                                             2, None, 'multimodal', 42).rows
    selected_ids = [row['id'] for row in chosen]
    assert len(selected_ids) == len(set(selected_ids))
    assert set(selected_ids) <= set(ids)
    return chosen, {
        'selector_source_sha256': digest, 'ratio_argument_percent': 2,
        'history_QA': len(ids), 'actual_memory_QA': len(chosen),
        'selected_ids_sha256': hashlib.sha256(
            json.dumps(selected_ids).encode()).hexdigest(),
        'selected_ids_encoding': 'json.dumps default ensure_ascii=True, UTF-8',
        'selected_ids': selected_ids,
    }


class _AnswerLossBridge:
    """Inject a precomputed common multimodal loss into author ReplayTrainer."""
    def __init__(self, *args, **kwargs):
        assert not args and not kwargs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss = inputs['common_answer_loss']
        assert loss.ndim == 0
        return (loss, None) if return_outputs else loss


def kernels():
    path, tree, digest = source('src/finetune_cl_our_qwen.py')
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name in {'DriftRecorderCallback', 'ReplayTrainer'}]
    assert len(selected) == 2
    namespace = {'torch': torch, 'os': os, 'MyTrainer': _AnswerLossBridge,
                 'transformers': SimpleNamespace(TrainerCallback=object)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace, digest


def intensity_beta(drift):
    """Exact original final-replay expression with original fixed bounds."""
    mu = drift.mu if drift.mu is not None else drift.mu0
    mu0 = drift.mu0 if drift.mu0 is not None else 1.0
    if mu is None:
        mu = mu0
    ratio = mu / (mu0 + 1e-12)
    return min(3.0, max(0.5, 1.0 + (ratio - 1.0)))


class ForeverStage:
    """Per-stage state; caller must execute and instrument the actual replay.

    Current 1% stages have <=22 updates. Original 24-step calibration therefore
    consumes all main updates. Fail closed on larger budgets: this class does
    not silently omit a reachable dynamic replay branch. A complete general
    scheduler would be required for such a protocol.
    """
    def __init__(self, model, train_tasks, stage, log_dir):
        self.model = model
        self.stage = stage
        self.main_updates = math.ceil(len(train_tasks[stage-1]) / 16)
        assert 1 <= self.main_updates <= 24, 'General dynamic scheduler required'
        self.memory, self.selection = select_history(train_tasks, stage)
        self.theta_star = {name: p.detach().clone() for name, p in
                           model.named_parameters() if p.requires_grad}
        assert self.theta_star
        self.importance = dict.fromkeys(self.theta_star, 1.0)
        self.k, self.source_sha256 = kernels()
        log_dir = Path(log_dir)
        # Stage reuse/resume must be handled at the outer checkpoint boundary.
        assert not (log_dir / 'drift_log.csv').exists(), 'Preserve prior drift'
        self.drift = self.k['DriftRecorderCallback'](str(log_dir))
        self.drift.on_train_begin(None, None, None, model=model)
        self.calibrated = False
        self.replay_loss = None

    def main_update_complete(self):
        assert not self.calibrated and self.drift.step < self.main_updates
        self.drift.on_step_end(None, None, None, model=self.model)
        return {'stage': self.stage, 'current_step': self.drift.step,
                'tau': self.drift.tau}

    def end_current(self):
        assert not self.calibrated and self.drift.step == self.main_updates
        assert self.drift.tau > 0 and math.isfinite(self.drift.tau)
        self.model_day = max(self.drift.tau, 1e-8)
        self.thresholds = [d*self.model_day for d in [1, 2, 4, 7, 15, 30, 60, 90, 120]]
        self.drift.mu0 = self.drift.tau / float(self.drift.step)
        self.drift.mu = self.drift.mu0
        self.drift.ema_alpha = 0.05
        self.beta = intensity_beta(self.drift)
        self.calibrated = True
        if self.memory:
            self.replay_loss = self.k['ReplayTrainer'](
                beta_t=self.beta, theta_star=self.theta_star,
                importance=self.importance)
        return {
            'stage': self.stage, 'main_updates': self.main_updates,
            'dynamic_replay_triggers': 0, 'calibration_steps_per_day': 24,
            'tau': self.drift.tau, 'model_day': self.model_day,
            'mu': self.drift.mu, 'mu0': self.drift.mu0, 'beta': self.beta,
            'final_replay_required': bool(self.memory),
            'memory_QA': len(self.memory), 'author_memory_epochs': 2,
            'author_replay_warmup_steps': 5,
            'reference_parameter_bytes': sum(x.numel()*x.element_size()
                                             for x in self.theta_star.values()),
        }

    def loss(self, common_answer_loss):
        assert self.calibrated and self.replay_loss is not None
        total = self.replay_loss.compute_loss(
            self.model, {'common_answer_loss': common_answer_loss})
        return total, {'base_loss': self.replay_loss.last_base_loss,
                       'unscaled_regularizer': self.replay_loss.last_loss_reg,
                       'regularizer_coefficient': 0.001*self.beta}

    def replay_complete(self):
        assert self.calibrated and self.memory
        # Author replay does not advance model time or its intensity EMA.
        step, tau = self.drift.step, self.drift.tau
        self.drift.update_reference_point(self.model)
        assert self.drift.step == step and self.drift.tau == tau

