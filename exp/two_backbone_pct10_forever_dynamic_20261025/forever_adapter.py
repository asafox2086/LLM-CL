"""Independent pct10 FOREVER: author replay mathematics, common current loop."""
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from methods import Method, ROOT

_core_path = Path(__file__).with_name('core.py')
_spec = importlib.util.spec_from_file_location('pct10_forever_dynamic_core', _core_path)
_core = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _core
_spec.loader.exec_module(_core)
ForeverStage, REVISION = _core.ForeverStage, _core.REVISION
sys.path.insert(0, str(ROOT / 'analysis'))
from forever_multimodal_replay_v2_20261019 import execute_replay, tensor_digest


class ForeverMethod(Method):
    def __init__(self, model, name, instructions, embedding=None):
        super().__init__(model, 'SeqLoRA', instructions, embedding)
        self.name, self.original_revision = name, REVISION
        self.core = None
        self.last_replay = None
        self.embedding = None
        self.stage_summaries = []

    def bind_context(self, processor, manifest, data, run, preflight):
        self.processor, self.manifest = processor, manifest
        self.train_tasks = [data[t['task'], 'train'] for t in manifest['tasks']]
        total = sum(t['files']['train']['count'] for t in manifest['tasks'])
        assert len(self.train_tasks) == len(manifest['tasks']) == 19
        assert sum(map(len, self.train_tasks)) == total
        self.task_by_id = {row['id']: i + 1 for i, rows in enumerate(self.train_tasks) for row in rows}
        assert len(self.task_by_id) == total
        self.io, self.preflight = SimpleNamespace(run=Path(run)), preflight

    def begin_stage(self, stage):
        assert self.core is None
        self.core = ForeverStage(self.model, self.train_tasks, stage,
            self.io.run / 'method_state' / f'stage_{stage:02d}' / 'drift')
        self.set_instruction(stage)

    def _execute_replay(self, kind):
        from runtime_utils import write
        from resource_monitor import ResourceMonitor
        stage = self.core.stage
        request = self.core.begin_replay(kind)
        folder = self.io.run / 'method_state' / f'stage_{stage:02d}'
        label = f"dynamic_{request['trigger_index']:02d}" if kind == 'dynamic' else 'final'
        directory = folder / label
        write(directory / 'REPLAY_REQUEST.json', request)
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        monitor = ResourceMonitor(directory / 'resources')
        monitor.stage = stage
        monitor.snapshot('training_before', stage, self.model)
        try:
            result = execute_replay(self.core, self.common_loss, directory / 'actual_author_replay')
            assert result is not None and result['model_time_unchanged_by_replay']
            assert result['requested_epochs'] == 2 and result['beta'] == request['beta']
            self.last_replay = result
            write(directory / 'REPLAY_CLOSED.json', dict(request=request, actual=result,
                scheduler_after=self.core.summary(), source_complete=True))
            return result
        finally:
            monitor.snapshot('training_after', stage, self.model)
            monitor.finish(self.model)
            self.set_instruction(stage)
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

    def after_update(self, step, updates):
        from runtime_utils import write
        assert self.core.main_updates == updates
        event = self.core.main_update_complete()
        assert event['current_step'] == step
        folder = self.io.run / 'method_state' / f'stage_{self.core.stage:02d}'
        write(folder / f'main_update_{step:03d}.json', event)
        # Match the author's while-loop boundary: no new threshold replay once
        # current N is exhausted; final replay still occurs in end_stage.
        while self.core.dynamic_due():
            self._execute_replay('dynamic')

    def common_loss(self, model, row):
        from runtime_utils import input_batch, answer_loss
        from frozen_vision import prepare
        task = self.manifest['tasks'][self.task_by_id[row['id']] - 1]
        self.set_instruction(task['task'])
        batch, _ = input_batch(row, task, self.processor, model.device)
        prepare(batch, row, model, self.vision_original)
        with torch.autocast('cuda', dtype=model.dtype):
            loss, _ = answer_loss(model, batch, row['answer'], self.processor)
        return loss

    def end_stage(self, stage):
        from runtime_utils import write
        assert self.core.stage == stage
        folder = self.io.run / 'method_state' / f'stage_{stage:02d}'
        write(folder / 'calibration.json', self.core.end_current())
        if self.core.selection is not None:
            write(folder / 'memory_selection.json', self.core.selection)
        if self.core.memory:
            self._execute_replay('final')
        else:
            self.core.no_history_complete()
            write(folder / 'REPLAY_SKIPPED.json', dict(reason='no historical TRAIN', stage=stage))
        summary = self.core.summary()
        assert summary['final_complete'] and summary['completed_current_updates'] == summary['main_updates']
        write(folder / 'SCHEDULER_COMPLETE.json', summary)
        self.stage_summaries.append(summary)
        self.core = None
        self.set_instruction(stage)
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    def preflight_replay(self):
        """Execute the real entire S2 current budget, then actual final replay.

        Disposable checkpoint/reload preflight, not a scientific trajectory or
        evidence that the >24 dynamic branch has actually executed on GPU.
        """
        from runtime_utils import make_optimizer, write
        from training_validity import warmup_updates
        from transformers import get_cosine_schedule_with_warmup
        assert self.preflight and self.core is None
        self.begin_stage(2)
        order = sorted(self.train_tasks[1], key=lambda row:
            hashlib.sha256(f"42:2:{row['id']}".encode()).digest())
        updates = math.ceil(len(order) / 16)
        assert updates == self.core.main_updates
        before = tensor_digest(self.changed_state())
        optimizer = make_optimizer(self)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_updates(updates), updates)
        scaler = torch.amp.GradScaler('cuda', init_scale=1)
        self.model.train()
        gradients, learning_rates, seen = [], [], []
        for step, start in enumerate(range(0, len(order), 16), 1):
            optimizer.zero_grad(set_to_none=True)
            minibatch = order[start:start + 16]
            for row in minibatch:
                loss = self.common_loss(self.model, row)
                assert torch.isfinite(loss)
                scaler.scale(loss / len(minibatch)).backward()
                seen.append(row['id'])
            scaler.unscale_(optimizer)
            gradient = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.)
            assert torch.isfinite(gradient)
            learning_rates.append([group['lr'] for group in optimizer.param_groups])
            assert any(lr > 0 for lr in learning_rates[-1])
            prior_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            assert scaler.get_scale() >= prior_scale
            scheduler.step()
            gradients.append(float(gradient))
            self.after_update(step, updates)
        after = tensor_digest(self.changed_state())
        assert before != after and seen == [row['id'] for row in order]
        del optimizer, scheduler, scaler, loss
        self.end_stage(2)
        assert self.last_replay['positive_LR_updates'] > 0
        write(self.io.run / 'FOREVER_REPLAY_GPU_GATE.json', dict(
            not_formal_results=True, extra_current_TRAIN_id=seen[0],
            extra_current_TRAIN_ids=seen, extra_current_updates=updates,
            extra_current_QA=len(seen), extra_current_gradient_l2=gradients[0],
            all_current_gradient_l2=gradients, learning_rate_used=learning_rates,
            parameter_before_sha256=before, parameter_after_sha256=after,
            actual_replay=self.last_replay, actual_scheduler=self.stage_summaries[-1],
            dynamic_branch_GPU_validated=self.stage_summaries[-1]['dynamic_replay_triggers'] > 0,
            longer_than_24_branch_GPU_validation_still_required=updates <= 24))

    def method_state(self, stage):
        return dict(method=self.name, upstream_revision=REVISION, completed_stage=stage,
            last_actual_author_replay=self.last_replay,
            current_main_budget='common one epoch, task-level optimizer/cosine/order unchanged',
            auxiliary_replay='author two epochs, HF linear, five warmup steps',
            memory_ratio_argument_percent=2, actual_memory_may_exceed_ratio=True,
            dynamic_schedule_scope=_core.SCHEDULER_REVISION,
            scheduler_completed_stages=self.stage_summaries,
            multimodal_adaptation_not_original_paper_reproduction=True)
