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
        """One disposable, derived T13 (>24 updates) replay/reload check.

        Neither the inherited nineteen one-row diagnostics nor this view is a
        formal learned prefix. The unchanged caller supplies its T19 probe and
        complete checkpoint/reload tail after this function returns.
        """
        import copy
        from runtime_utils import make_optimizer, write
        from training_validity import warmup_updates
        from transformers import get_cosine_schedule_with_warmup
        assert self.preflight and self.core is not None
        # The unchanged common preflight begins stage1, then executes nineteen
        # one-row diagnostics without after_update. That live core is not a
        # learned S1 prefix and must never seed the scheduler calibration.
        abandoned = self.core
        assert abandoned.stage == 1 and abandoned.drift.step == 0 and abandoned.drift.tau == 0
        assert not abandoned.calibrated and abandoned.active_replay is None
        assert not abandoned.completed_replays and not abandoned.memory and not abandoned.final_complete
        assert not self.stage_summaries
        write(self.io.run / 'FOREVER_COMMON_PREFLIGHT_DISCARD.json', dict(
            not_formal_results=True, not_a_real_S1_prefix=True,
            discarded_stage=1, drift_observed_current_updates=0,
            calibrated=False, active_replay=False, completed_replays=0,
            inherited_common_diagnostic_cases_by_source_contract=[
                dict(task=i + 1, source_id=rows[0]['id']) for i, rows in enumerate(self.train_tasks)],
            inherited_common_diagnostic_case_count_by_source_contract=19,
            outer_gpu_preflight_actual_cases_binding_required=True,
            model_weights_after_common_diagnostics_retained=True,
            discarded_scheduler_state_only=True))
        self.core = None
        del abandoned
        gc.collect()
        original_tasks = self.train_tasks
        original_mapping = dict(self.task_by_id)
        stage = 13
        view = [[] for _ in original_tasks]
        for index in range(stage):
            ordered = sorted(original_tasks[index], key=lambda row:
                hashlib.sha256(f"42:{index + 1}:{row['id']}".encode()).digest())
            needed = 385 if index + 1 == stage else 1
            assert len(ordered) >= needed
            view[index] = ordered[:needed]
        assert len(view) == 19 and all(len(rows) == 1 for rows in view[:12])
        assert len(view[12]) == 385 and not any(view[13:])
        view_ids = [[row['id'] for row in rows] for rows in view]
        write(self.io.run / 'FOREVER_PREFLIGHT_VIEW.json', dict(
            not_formal_results=True, not_a_real_learned_history=True,
            original_manifest_unchanged=True, stage=stage,
            hash_order='SHA256(42:task:ID), raw digest ascending',
            task_ids=view_ids, view_ids_sha256=hashlib.sha256(
                json.dumps(view_ids).encode()).hexdigest(),
            view_ids_encoding='json.dumps defaults, UTF-8',
            common_prefix_TRAIN_diagnostics=19,
            later_original_run_probe_task=19,
            later_original_run_checkpoint_reload_not_yet_executed=True))
        # Only the core receives the derived view. Loss routing and the formal
        # task lists remain the original full, manifest-bound objects.
        self.core = ForeverStage(self.model, view, stage,
            self.io.run / 'method_state' / 'stage_13' / 'drift')
        self.set_instruction(stage)
        order = view[stage - 1]
        updates = math.ceil(len(order) / 16)
        assert updates == self.core.main_updates == 25
        assert self.core.selection['history_QA'] == 12 and self.core.memory
        assert all(self.task_by_id[row['id']] < stage for row in self.core.memory)
        before = tensor_digest(self.changed_state())
        optimizer = make_optimizer(self)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_updates(updates), updates)
        scaler = torch.amp.GradScaler('cuda', init_scale=1)
        self.model.train()
        gradients, learning_rates, seen = [], [], []
        sequence, boundary_checks, dynamic_reports = [], [], []

        def outer_state():
            tensors, scalars, state_ids = {}, {}, []
            for index, parameter in enumerate(
                    p for group in optimizer.param_groups for p in group['params']):
                for key, value in optimizer.state.get(parameter, {}).items():
                    name = f'{index}:{key}'
                    if torch.is_tensor(value):
                        # Adam's step is a 0-D tensor. The inherited raw-byte
                        # digest views as uint8, which needs a non-scalar view;
                        # retain its original shape in the digest key.
                        tensors[f'{name}:shape={tuple(value.shape)}'] = value.reshape(-1)
                        state_ids.append((name, id(value), value.data_ptr()))
                    else:
                        scalars[name] = copy.deepcopy(value)
            return dict(
                objects=[id(optimizer), id(scheduler), id(scaler)],
                parameters=[(name, id(p), p.data_ptr(), p.requires_grad)
                            for name, p in self.model.named_parameters()],
                optimizer_parameter_groups=[
                    dict(parameters=[id(p) for p in group['params']],
                         options=copy.deepcopy({k: v for k, v in group.items() if k != 'params'}))
                    for group in optimizer.param_groups],
                optimizer_state_tensor_ids=state_ids,
                optimizer_state_sha256=tensor_digest(tensors), optimizer_state_scalars=scalars,
                scheduler=copy.deepcopy(scheduler.state_dict()),
                scaler=copy.deepcopy(scaler.state_dict()),
                gradient_checkpointing=[(name, module.gradient_checkpointing)
                    for name, module in self.model.named_modules()
                    if isinstance(getattr(module, 'gradient_checkpointing', None), bool)])

        original_execute = self._execute_replay

        def observed_dynamic(kind):
            assert kind == 'dynamic'
            snapshot = outer_state()
            current_step = self.core.drift.step
            sequence.append(dict(event='dynamic_replay_start', current_step=current_step))
            result = original_execute(kind)
            assert outer_state() == snapshot, 'Replay altered outer optimizer/scaler/parameter contract'
            boundary_checks.append(dict(current_step=current_step,
                external_optimizer_scheduler_scaler_objects_and_state_exact=True,
                parameter_objects_storage_requires_grad_exact=True,
                checkpoint_flags_exact=True,
                optimizer_state_sha256=snapshot['optimizer_state_sha256'],
                RNG_preserved=False, weights_preserved=False))
            dynamic_reports.append(result)
            sequence.append(dict(event='dynamic_replay_complete', current_step=current_step))
            return result

        assert '_execute_replay' not in self.__dict__
        self._execute_replay = observed_dynamic
        try:
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
                sequence.append(dict(event='current_update_complete', current_step=step))
                self.after_update(step, updates)
        finally:
            del self._execute_replay
        after = tensor_digest(self.changed_state())
        assert before != after and seen == [row['id'] for row in order]
        assert dynamic_reports and all(x['positive_LR_updates'] > 0 for x in dynamic_reports)
        assert self.core.drift.step == 25
        assert sequence[-1] == dict(event='current_update_complete', current_step=25)
        assert any(x['event'] == 'dynamic_replay_complete' and x['current_step'] == 24 for x in sequence)
        del optimizer, scheduler, scaler, loss
        sequence.append(dict(event='final_replay_start', current_step=25))
        self.end_stage(stage)
        sequence.append(dict(event='final_replay_complete', current_step=25))
        assert self.core is None and self.train_tasks is original_tasks
        assert self.task_by_id == original_mapping
        assert self.last_replay['positive_LR_updates'] > 0
        write(self.io.run / 'FOREVER_REPLAY_GPU_GATE.json', dict(
            not_formal_results=True, extra_current_TRAIN_id=seen[0],
            extra_current_TRAIN_ids=seen, extra_current_updates=updates,
            extra_current_QA=len(seen), extra_current_gradient_l2=gradients[0],
            all_current_gradient_l2=gradients, learning_rate_used=learning_rates,
            parameter_before_sha256=before, parameter_after_sha256=after,
            diagnostic_stage=stage, original_task_mapping_unchanged=True,
            common_prefix_diagnostic_QA_by_source_contract=19,
            common_prefix_not_a_learned_S1=True,
            actual_history_QA=12, actual_memory_QA=self.stage_summaries[-1]['memory_QA'],
            actual_dynamic_replays=dynamic_reports, dynamic_outer_state_checks=boundary_checks,
            ordered_execution=sequence,
            replay_TRAIN_QA=sum(x['actual_QA_exposures'] for x in dynamic_reports) +
                self.last_replay['actual_QA_exposures'],
            replay_unwrap_extra_TRAIN_NLL_forwards=2 * (len(dynamic_reports) + 1),
            later_original_run_TRAIN_probe_task=19,
            later_original_run_probe_CP_reload_cost_requires_outer_proof=True,
            new_TEST_QA=0, new_generations=0, new_F1_scores=0,
            actual_replay=self.last_replay, actual_scheduler=self.stage_summaries[-1],
            dynamic_branch_GPU_validated=self.stage_summaries[-1]['dynamic_replay_triggers'] > 0,
            longer_than_24_branch_GPU_validation_still_required=False))

    def method_state(self, stage):
        return dict(method=self.name, upstream_revision=REVISION, completed_stage=stage,
            last_actual_author_replay=self.last_replay,
            current_main_budget='common one epoch, task-level optimizer/cosine/order unchanged',
            auxiliary_replay='author two epochs, HF linear, five warmup steps',
            memory_ratio_argument_percent=2, actual_memory_may_exceed_ratio=True,
            dynamic_schedule_scope=_core.SCHEDULER_REVISION,
            scheduler_completed_stages=self.stage_summaries,
            multimodal_adaptation_not_original_paper_reproduction=True)
