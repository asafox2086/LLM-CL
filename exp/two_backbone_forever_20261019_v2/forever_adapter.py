"""Copied into an independent runner; never installed into active V2 sources."""
import gc
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from methods import Method, ROOT

sys.path.insert(0, str(ROOT/'analysis'))
from forever_multimodal_core_20261019 import ForeverStage, REVISION
from forever_multimodal_replay_v2_20261019 import execute_replay, tensor_digest


class ForeverMethod(Method):
    def __init__(self, model, name, instructions, embedding=None):
        super().__init__(model, 'SeqLoRA', instructions, embedding)
        self.name = name
        self.original_revision = REVISION
        self.core = None
        self.last_replay = None
        self.embedding = None

    def bind_context(self, processor, manifest, data, run, preflight):
        self.processor = processor
        self.manifest = manifest
        self.train_tasks = [data[t['task'], 'train'] for t in manifest['tasks']]
        assert len(self.train_tasks) == 19 and sum(map(len, self.train_tasks)) == 1313
        self.task_by_id = {row['id']: i+1 for i, rows in enumerate(self.train_tasks)
                           for row in rows}
        assert len(self.task_by_id) == 1313
        self.io = SimpleNamespace(run=Path(run))
        self.preflight = preflight

    def begin_stage(self, stage):
        assert self.core is None
        self.core = ForeverStage(self.model, self.train_tasks, stage,
                                 self.io.run/'method_state'/f'stage_{stage:02d}'/'drift')
        self.set_instruction(stage)

    def after_update(self, step, updates):
        from runtime_utils import write
        assert self.core.main_updates == updates
        result = self.core.main_update_complete()
        assert result['current_step'] == step
        write(self.io.run/'method_state'/f'stage_{self.core.stage:02d}'/
              f'main_update_{step:03d}.json', result)

    def common_loss(self, model, row):
        from runtime_utils import input_batch, answer_loss
        from frozen_vision import prepare
        task = self.manifest['tasks'][self.task_by_id[row['id']]-1]
        self.set_instruction(task['task'])
        batch, _ = input_batch(row, task, self.processor, model.device)
        prepare(batch, row, model, self.vision_original)
        with torch.autocast('cuda', dtype=model.dtype):
            loss, _ = answer_loss(model, batch, row['answer'], self.processor)
        return loss

    def end_stage(self, stage):
        from runtime_utils import write
        from resource_monitor import ResourceMonitor
        assert self.core.stage == stage
        folder = self.io.run/'method_state'/f'stage_{stage:02d}'
        calibration = self.core.end_current()
        write(folder/'calibration.json', calibration)
        if self.core.selection is not None:
            write(folder/'memory_selection.json', self.core.selection)
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        monitor = ResourceMonitor(folder/'replay_resources')
        monitor.stage = stage
        monitor.snapshot('training_before', stage, self.model)
        try:
            self.last_replay = execute_replay(self.core, self.common_loss,
                                             folder/'actual_author_replay')
        finally:
            monitor.snapshot('training_after', stage, self.model)
            monitor.finish(self.model)
        self.core = None
        self.model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

    def preflight_replay(self):
        """One new-task TRAIN update then actual 303-row historical replay.

        Earlier common preflight rows/updates are disposable diagnostics, not
        a lawful stage-one trajectory. This is explicitly not a formal run.
        """
        from runtime_utils import make_optimizer, write
        assert self.preflight
        self.core = None
        self.begin_stage(2)
        before = tensor_digest(self.changed_state())
        optimizer = make_optimizer(self)
        scaler = torch.amp.GradScaler('cuda', init_scale=1)
        self.model.train()
        row = self.train_tasks[1][0]
        optimizer.zero_grad(set_to_none=True)
        loss = self.common_loss(self.model, row)
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], 1.)
        assert torch.isfinite(gradient) and float(gradient) > 0
        prior_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= prior_scale
        after = tensor_digest(self.changed_state())
        assert before != after
        self.after_update(1, 1)
        del optimizer, scaler, loss
        self.end_stage(2)
        assert self.last_replay['positive_LR_updates'] > 0
        write(self.io.run/'FOREVER_REPLAY_GPU_GATE.json', {
            'not_formal_results': True, 'extra_current_TRAIN_id': row['id'],
            'extra_current_updates': 1, 'extra_current_gradient_l2': float(gradient),
            'parameter_before_sha256': before, 'parameter_after_sha256': after,
            'actual_replay': self.last_replay,
        })

    def method_state(self, stage):
        return {'method': self.name, 'upstream_revision': REVISION,
                'completed_stage': stage, 'last_actual_author_replay': self.last_replay,
                'current_main_budget': 'common 1 epoch, cosine floor warmup',
                'auxiliary_replay': 'author 2 epochs, HF linear, 5 warmup steps',
                'memory_ratio_argument_percent': 2,
                'actual_memory_may_exceed_ratio': True,
                'dynamic_schedule_scope': '1% <=24 main updates only',
                'multimodal_adaptation_not_original_paper_reproduction': True}
