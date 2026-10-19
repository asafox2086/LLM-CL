"""Actual HF replay engine for the independent FOREVER multimodal port draft.

The caller supplies the unchanged common multimodal answer-loss function.
Author replay epochs/warmup/linear schedule remain explicit. CPU controls are
not model experiments; the module has no launch/import-time GPU side effects.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments


def tensor_digest(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), 'Never overwrite replay evidence'
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class ReplayDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {'forever_row': self.rows[index]}


def collate(rows):
    assert len(rows) == 1, 'Shared micro batch size is one'
    return {'forever_row': rows[0]['forever_row']}


class ReplayTrace(TrainerCallback):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.records = []
        self.raw = None
        self.micro_losses = []
        self.qa_seen = 0
        self.before = None
        self.lr = None

    def training_step_complete(self, trainer, model):
        if trainer.accelerator.sync_gradients:
            scaler = trainer.accelerator.scaler
            scale = scaler.get_scale() if scaler is not None else 1.0
            # Read before HF clipping/optimizer. Charge this CPU transfer/storage.
            self.raw = {name: p.grad.detach().float().cpu().clone()/scale
                        for name, p in model.named_parameters()
                        if p.requires_grad and p.grad is not None}

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model, optimizer = kwargs['model'], kwargs['optimizer']
        assert self.raw and all(torch.isfinite(x).all() for x in self.raw.values())
        self.before = {name: p.detach().cpu().clone()
                       for name, p in model.named_parameters() if p.requires_grad}
        self.lr = [float(group['lr']) for group in optimizer.param_groups]

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs['model']
        after = {name: p.detach().cpu().clone()
                 for name, p in model.named_parameters() if p.requires_grad}
        assert self.before and self.lr is not None
        assert all(torch.isfinite(x).all() for x in after.values())
        step = len(self.records)+1
        raw_file = self.directory/f'update_{step:04d}_raw_gradient.pt'
        assert not raw_file.exists()
        torch.save(self.raw, raw_file)
        norm = math.sqrt(sum(float((x.double()**2).sum()) for x in self.raw.values()))
        movement = math.sqrt(sum(float(((after[n].double()-p.double())**2).sum())
                                 for n, p in self.before.items()))
        record = {
            'event': 'author_final_replay_update', 'step': step,
            'learning_rate_used': self.lr,
            'author_zero_warmup_step': not any(x > 0 for x in self.lr),
            'raw_gradient_l2': norm, 'parameter_delta_l2': movement,
            'parameter_before_sha256': tensor_digest(self.before),
            'parameter_after_sha256': tensor_digest(after),
            'raw_gradient_file': raw_file.name,
            'raw_gradient_sha256': hashlib.sha256(raw_file.read_bytes()).hexdigest(),
            'cumulative_QA': self.qa_seen,
            'micro_losses': self.micro_losses,
        }
        assert math.isfinite(norm) and math.isfinite(movement)
        if any(x > 0 for x in self.lr):
            assert movement > 0, 'Positive LR did not update replay parameters'
        else:
            assert movement == 0, 'Author zero-warmup step moved parameters'
        self.records.append(record)
        with (self.directory/'updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
        self.before = self.raw = self.lr = None
        self.micro_losses = []


class MultimodalReplayTrainer(Trainer):
    def __init__(self, *args, forever_stage, common_loss, trace, **kwargs):
        self.forever_stage, self.common_loss, self.trace = forever_stage, common_loss, trace
        super().__init__(*args, **kwargs)
        # Common answer_loss returns a scalar micro-batch mean. Do not let HF
        # reinterpret its signature as a token-summed custom loss.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        row = inputs['forever_row']
        base = self.common_loss(model, row)
        loss, terms = self.forever_stage.loss(base)
        assert torch.isfinite(loss)
        self.trace.qa_seen += 1
        self.trace.micro_losses.append({'TRAIN_id': row['id'], **terms,
                                       'total_loss': float(loss.detach())})
        return (loss, {}) if return_outputs else loss

    def training_step(self, model, inputs, *args, **kwargs):
        value = super().training_step(model, inputs, *args, **kwargs)
        self.trace.training_step_complete(self, model)
        return value


def execute_replay(stage, common_loss, directory, learning_rate=1e-4,
                   cpu_control=False):
    """Execute all actual author-HF replay, or skip only an empty stage-one pool.

    CPU control is explicitly recorded and may never be published as a formal
    backbone result. The enclosing runner must measure process/allocator memory
    before/after/peak and checkpoint/reload the resulting method graph.
    """
    assert stage.calibrated
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    if not stage.memory:
        save_json(directory/'REPLAY_SKIPPED.json', {'reason': 'no historical TRAIN',
                                                  'stage': stage.stage})
        return None
    device = next(stage.model.parameters()).device.type
    assert (cpu_control and device == 'cpu') or (not cpu_control and device == 'cuda')
    trace = ReplayTrace(directory/'trace')
    settings = dict(
        output_dir=str(directory/'hf_output'),
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        warmup_steps=5, num_train_epochs=2, learning_rate=learning_rate,
        lr_scheduler_type='linear', optim='adamw_torch', weight_decay=0.,
        adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8, max_grad_norm=1.,
        fp16=not cpu_control, bf16=False, use_cpu=cpu_control,
        logging_steps=1, save_strategy='no', report_to=[],
        remove_unused_columns=False, dataloader_num_workers=0,
        seed=42, data_seed=42, disable_tqdm=True,
    )
    save_json(directory/'REQUESTED_ARGUMENTS.json', {
        **settings, 'CPU_control_not_formal': cpu_control,
        'author_reference_sha256': stage.source_sha256,
        'stage': stage.stage, 'beta': stage.beta, 'memory_QA': len(stage.memory),
    })
    start = time.monotonic()
    old_time = (stage.drift.step, stage.drift.tau)
    trainer = MultimodalReplayTrainer(
        model=stage.model, args=TrainingArguments(**settings),
        train_dataset=ReplayDataset(stage.memory), data_collator=collate,
        callbacks=[trace], forever_stage=stage, common_loss=common_loss, trace=trace)
    # Shared V2 FP16 framework starts at loss scale 1, rather than HF's default
    # 65536. Apply it before the first backward; this is an explicit framework
    # adaptation, not an author hyperparameter or performance intervention.
    if not cpu_control:
        assert trainer.accelerator.scaler is not None
        initial = trainer.accelerator.scaler.state_dict()
        changed = {**initial, 'scale': 1.0}
        trainer.accelerator.scaler.load_state_dict(changed)
        assert trainer.accelerator.scaler.get_scale() == 1.0
        save_json(directory/'SHARED_FP16_INITIAL_SCALE.json', {
            'HF_default_state': initial, 'shared_framework_state': changed,
            'actual_before_first_backward': trainer.accelerator.scaler.get_scale(),
        })
    result = trainer.train()
    positive = sum(any(x > 0 for x in record['learning_rate_used'])
                   for record in trace.records)
    assert trace.records and positive > 0, 'Replay has no meaningful updates'
    assert trainer.state.global_step == len(trace.records)
    assert old_time == (stage.drift.step, stage.drift.tau)
    stage.replay_complete()
    # v2 repair: HF adds a non-picklable AMP forward wrapper. Remove only
    # that wrapper before the common full-method graph checkpoint. The
    # common framework already uses explicit autocast for train/evaluation.
    model = stage.model
    signature = [(name, id(p), p.data_ptr(), p._version)
                 for name, p in model.named_parameters()]
    weights = tensor_digest({name: p for name, p in model.named_parameters()
                             if p.requires_grad})
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == 'cuda' else None
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            probe_before = common_loss(model, stage.memory[0]).detach().float().cpu()
        assert trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False) is model
        with torch.no_grad():
            probe_after = common_loss(model, stage.memory[0]).detach().float().cpu()
        assert torch.equal(probe_before, probe_after), 'AMP cleanup changed common TRAIN NLL'
        assert signature == [(name, id(p), p.data_ptr(), p._version)
                             for name, p in model.named_parameters()]
        assert weights == tensor_digest({name: p for name, p in model.named_parameters()
                                          if p.requires_grad})
        save_json(directory/'AMP_UNWRAP_EXACT.json', {
            'keep_fp32_wrapper': False, 'model_identity_storage_versions_exact': True,
            'trainable_parameter_sha256': weights,
            'TRAIN_probe_id': stage.memory[0]['id'],
            'NLL_before': float(probe_before), 'NLL_after': float(probe_after),
            'common_explicit_autocast_NLL_exact': True,
            'extra_TRAIN_NLL_forwards': 2, 'extra_optimizer_updates': 0,
            'resource_scope_includes_post_train_unwrap_and_two_NLL': True,
        })
    finally:
        model.train(was_training)
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    report = {
        'status': 'CPU_CONTROL_COMPLETE' if cpu_control else 'ACTUAL_REPLAY_COMPLETE',
        'stage': stage.stage, 'actual_optimizer_updates': len(trace.records),
        'positive_LR_updates': positive,
        'author_zero_warmup_updates': len(trace.records)-positive,
        'actual_QA_exposures': trace.qa_seen, 'requested_epochs': 2,
        'HF_effective_num_train_epochs': trainer.state.num_train_epochs,
        'HF_final_epoch': trainer.state.epoch, 'HF_train_metrics': result.metrics,
        'seconds': time.monotonic()-start, 'beta': stage.beta,
        'model_time_unchanged_by_replay': True,
        'CPU_control_not_formal': cpu_control,
    }
    save_json(directory/'REPLAY_COMPLETE.json', report)
    del trainer
    gc.collect()
    return report
