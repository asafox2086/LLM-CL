"""FOREVER parameter-time scheduling under the common current-task loop.

Pinned author selector, drift callback and replay loss are inherited unchanged.
This module implements their previously unreachable (>24 current updates)
state machine. Importing it does not construct or execute a model.
"""
from pathlib import Path
import hashlib
import importlib.util
import math

ROOT = Path(__file__).resolve().parents[2]
LEGACY_CORE_SHA256 = '2d9d5be87d81df80eab6ddf0095ce399c9a73d9c8c29909a5022d2364fdeeed8'
_path = ROOT / 'analysis/forever_multimodal_core_20261019.py'
assert hashlib.sha256(_path.read_bytes()).hexdigest() == LEGACY_CORE_SHA256
_spec = importlib.util.spec_from_file_location('pct10_forever_pinned_kernels', _path)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
REVISION = _legacy.REVISION
SCHEDULER_REVISION = 'pct10-author-triggers-common-current-loop-v1'
TRIGGER_DAYS = (1, 2, 4, 7, 15, 30, 60, 90, 120)


class ForeverStage:
    def __init__(self, model, train_tasks, stage, log_dir):
        assert 1 <= stage <= len(train_tasks)
        self.model, self.stage = model, stage
        self.main_updates = math.ceil(len(train_tasks[stage - 1]) / 16)
        assert self.main_updates >= 1
        self.calibration_steps = min(24, self.main_updates)
        self.memory, self.selection = _legacy.select_history(train_tasks, stage)
        self.theta_star = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        assert self.theta_star
        self.importance = dict.fromkeys(self.theta_star, 1.0)
        self.k, self.source_sha256 = _legacy.kernels()
        directory = Path(log_dir)
        assert not (directory / 'drift_log.csv').exists(), 'Preserve the existing stage'
        self.drift = self.k['DriftRecorderCallback'](str(directory))
        self.drift.on_train_begin(None, None, None, model=model)
        self.calibrated = False
        self.model_day = None
        self.thresholds = []
        self.next_trigger_idx = 0
        self.replay_loss = None
        self.active_replay = None
        self.completed_replays = []
        self.final_complete = False
        self.beta = None
        self.theta_star_digest = None

    def _calibrate(self):
        assert not self.calibrated and self.drift.step == self.calibration_steps
        assert math.isfinite(self.drift.tau) and self.drift.tau >= 0
        self.model_day = max(self.drift.tau, 1e-8)
        self.thresholds = [day * self.model_day for day in TRIGGER_DAYS]
        self.drift.mu0 = self.drift.tau / float(self.drift.step)
        self.drift.mu = self.drift.mu0
        self.drift.ema_alpha = .05
        self.calibration_tau = self.drift.tau
        self.calibrated = True

    def main_update_complete(self):
        assert self.active_replay is None and not self.final_complete
        assert self.drift.step < self.main_updates
        before = self.drift.step
        self.drift.on_step_end(None, None, None, model=self.model)
        assert self.drift.step == before + 1
        if not self.calibrated and self.drift.step == self.calibration_steps:
            self._calibrate()
        return dict(stage=self.stage, current_step=self.drift.step, main_updates=self.main_updates,
                    tau=self.drift.tau, calibrated=self.calibrated,
                    calibration_steps=self.calibration_steps, model_day=self.model_day,
                    mu=self.drift.mu, mu0=self.drift.mu0,
                    next_trigger_idx=self.next_trigger_idx, dynamic_replay_due=self.dynamic_due())

    def dynamic_due(self):
        return bool(self.calibrated and self.memory and self.drift.step < self.main_updates
                    and self.next_trigger_idx < len(self.thresholds)
                    and self.drift.tau >= self.thresholds[self.next_trigger_idx])

    def begin_replay(self, kind):
        assert kind in ('dynamic', 'final')
        assert self.calibrated and self.memory and self.active_replay is None and not self.final_complete
        if kind == 'dynamic':
            assert self.dynamic_due()
        else:
            assert self.drift.step == self.main_updates
        self.beta = _legacy.intensity_beta(self.drift)
        assert .5 <= self.beta <= 3 and math.isfinite(self.beta)
        self.replay_loss = self.k['ReplayTrainer'](
            beta_t=self.beta, theta_star=self.theta_star, importance=self.importance)
        self.active_replay = dict(kind=kind, sequence=len(self.completed_replays) + 1,
            trigger_index=self.next_trigger_idx if kind == 'dynamic' else None,
            trigger_threshold=self.thresholds[self.next_trigger_idx] if kind == 'dynamic' else None,
            current_step=self.drift.step, tau=self.drift.tau, mu=self.drift.mu,
            mu0=self.drift.mu0, beta=self.beta, memory_QA=len(self.memory),
            author_memory_epochs=2, author_replay_warmup_steps=5,
            regularizer_coefficient=.001 * self.beta,
            theta_star_scope='fixed task-entry trainable parameters')
        return dict(self.active_replay)

    def loss(self, common_answer_loss):
        assert self.active_replay is not None and self.replay_loss is not None
        total = self.replay_loss.compute_loss(self.model, {'common_answer_loss': common_answer_loss})
        return total, dict(base_loss=self.replay_loss.last_base_loss,
                           unscaled_regularizer=self.replay_loss.last_loss_reg,
                           regularizer_coefficient=.001 * self.beta)

    def replay_complete(self):
        assert self.active_replay is not None
        before = (self.drift.step, self.drift.tau, self.drift.mu, self.drift.mu0)
        event = dict(self.active_replay)
        assert before == (event['current_step'], event['tau'], event['mu'], event['mu0'])
        self.drift.update_reference_point(self.model)
        assert before == (self.drift.step, self.drift.tau, self.drift.mu, self.drift.mu0)
        if event['kind'] == 'dynamic':
            assert event['trigger_index'] == self.next_trigger_idx
            self.next_trigger_idx += 1
        else:
            assert self.drift.step == self.main_updates
            self.final_complete = True
        self.completed_replays.append(dict(event, model_time_unchanged_by_replay=True,
                                           drift_reference_updated=True))
        self.active_replay = None
        self.replay_loss = None

    def end_current(self):
        assert self.active_replay is None and self.drift.step == self.main_updates
        assert self.calibrated and not self.final_complete
        return self.summary()

    def no_history_complete(self):
        assert self.drift.step == self.main_updates and self.calibrated and not self.memory
        assert not self.completed_replays and self.active_replay is None
        self.final_complete = True

    def summary(self):
        return dict(stage=self.stage, scheduler_revision=SCHEDULER_REVISION,
            source_sha256=self.source_sha256, main_updates=self.main_updates,
            completed_current_updates=self.drift.step, calibration_steps_per_day=24,
            actual_calibration_steps=self.calibration_steps, calibrated=self.calibrated,
            calibration_tau=getattr(self, 'calibration_tau', None),
            tau=self.drift.tau, model_day=self.model_day, mu=self.drift.mu, mu0=self.drift.mu0,
            thresholds=self.thresholds, next_trigger_idx=self.next_trigger_idx,
            dynamic_replay_triggers=sum(x['kind'] == 'dynamic' for x in self.completed_replays),
            completed_replays=self.completed_replays, final_replay_required=bool(self.memory),
            final_complete=self.final_complete, memory_QA=len(self.memory),
            author_memory_epochs=2, author_replay_warmup_steps=5,
            reference_parameter_bytes=sum(x.numel() * x.element_size() for x in self.theta_star.values()),
            current_optimizer_scheduler_policy='one common task-level optimizer/cosine; no phase restart',
            author_HF_phase_restarts_reproduced=False)
