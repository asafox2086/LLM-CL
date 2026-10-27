"""Current-task FIT gradient estimate, no optimizer update or future data."""
import hashlib
import time


def estimate(ctx, model, method, stage, operators):
    import torch
    task = ctx.manifest['tasks'][stage - 1]
    rows = ctx.fit[stage][:min(16, len(ctx.fit[stage]))]
    assert rows
    parameters = list(model.parameters())
    modes = [(m, m.training) for m in model.modules()]
    masks = [(p, p.requires_grad) for p in parameters]
    rng = operators.rng_snapshot()
    targets = {name: model.get_submodule(name).weight for name in method.rank_plan}
    before = ctx.storage.digest(model.core)
    buffers = ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
    batcher = ctx.batcher(model, 'DIRECTION_CALIBRATION')
    started = time.monotonic()
    losses = []
    gradients = None
    try:
        model.requires_grad_(False).eval()
        model.zero_grad(set_to_none=True)
        # FP32 gradient accumulation; autocast keeps the original FP16 forward.
        for p in targets.values():
            assert p.is_cuda and p.dtype == torch.float16
            p.data = p.data.float()
            p.requires_grad_(True)
        for row in rows:
            batch, _ = batcher(row, task)
            with torch.autocast('cuda', dtype=torch.float16):
                loss, count = ctx.rt.answer_loss(model, batch, row['answer'], ctx.processor)
            assert torch.isfinite(loss) and count > 0
            losses.append(float(loss.detach()))
            (loss / len(rows)).backward()
            del loss, batch
        gradients = {}
        for name, p in targets.items():
            assert p.grad is not None and p.grad.dtype == torch.float32
            assert torch.isfinite(p.grad).all()
            gradients[name] = p.grad.detach().clone()
    finally:
        model.zero_grad(set_to_none=True)
        for p in targets.values():
            p.data = p.data.half()
        for p, flag in masks:
            p.requires_grad_(flag)
        for m, flag in modes:
            m.training = flag
        operators.rng_restore(rng)
    assert ctx.storage.digest(model.core) == before
    assert ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == buffers
    assert all(p.grad is None for p in parameters)
    return gradients, dict(stage=stage, QA=len(rows), backwards=len(rows), optimizer_updates=0,
        ordered_IDs_sha256=hashlib.sha256(''.join(r['id']+'\n' for r in rows).encode()).hexdigest(),
        input_records=batcher.records, answer_losses=losses, QA_equal_weight=True,
        current_task_only=True, model_state_unchanged=True, RNG_restored=True,
        gradient_bytes=sum(g.numel()*g.element_size() for g in gradients.values()),
        gradient_dtype='torch.float32', forward_autocast='torch.float16',
        seconds=time.monotonic()-started)
