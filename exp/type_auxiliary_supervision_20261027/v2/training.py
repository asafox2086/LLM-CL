"""Common-prefix / three-suffix training engine. No execution on import.

Context and sealed readout/export operators are supplied by the worker. This
module never schedules GPUs or scores answers. All optimization uses the
original frozen stage AST, including its accumulation and LR schedule.
"""
import ast
import hashlib
from types import SimpleNamespace

AST_SHA = '2b31ff9e06234e967ea42d3cb6d594c82050054c7c2c058d77f61bf8100dda7f'


def run_stages(ctx, method, first, last, operators, helpers, checkpoint_rng=None):
    import torch
    model = method.model
    assert (first, last) in ((1, 15), (16, 19))
    assert ctx.arm in ('prefix', 'plain', 'true', 'shuffled')
    assert (ctx.arm == 'prefix') == (first == 1)
    batcher = ctx.old_common.Batcher(ctx, model, 'TRAIN')
    loss_with_prompt = helpers.bind_answer_loss(ctx.port / 'runtime_utils.py', torch)
    true, shuffled = helpers.fit_labels(ctx.fit[16])
    labels = shuffled if ctx.arm == 'shuffled' else true
    formal = operators.module('type_aux_formal_training', ctx.port / 'run.py')
    directory = ctx.run / 'training'
    directory.mkdir()
    (directory / 'logs').mkdir()
    monitor = formal.ResourceMonitor(directory)
    data, plan, counts, optimizer_holder = {}, [], {}, []
    before, acquired, readout_events, supervision = [], [], [], []
    frozen = lambda: ctx.storage.digest({n:p for n,p in model.core.named_parameters()
                                         if not n.endswith(('.A.weight','.B.weight'))})
    frozen_before = frozen()
    buffers_before = ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
    for task in ctx.manifest['tasks']:
        stage = task['task']
        data[stage, 'train'], data[stage, 'dev'] = ctx.fit[stage], ctx.dev[stage]
        order = sorted(ctx.fit[stage], key=lambda r: hashlib.sha256(
            f"42:{stage}:{r['id']}".encode()).digest())
        task['training_order_sha256'] = hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        if first <= stage <= last:
            plan.append(dict(stage=stage, QA=len(order), updates=(len(order)+15)//16,
                             order_sha256=task['training_order_sha256']))
    expected_steps = 69 if first == 1 else 18
    assert sum(p['updates'] for p in plan) == expected_steps

    def training_loss(batch, row, processor):
        assert monitor.stage == 16 and row['id'] in labels
        answer, length, hidden = loss_with_prompt(model, batch, row['answer'], processor)
        aux = helpers.auxiliary_loss(torch, model._type_aux_head, hidden, labels[row['id']])
        supervision.append(dict(id_sha256=hashlib.sha256(row['id'].encode()).hexdigest(),
            label=labels[row['id']], answer_CE=float(answer.detach()), auxiliary_CE=float(aux.detach())))
        return answer + aux, length

    def before_stage(stage):
        operators.stage_readout(ctx, model, ctx.manifest['tasks'][stage-1],
                               'DEV_BEFORE', before, readout_events)
        if stage == 16:
            state = ctx.rt.rng()
            head = helpers.make_head(torch, 896, ctx.device)
            # Also verify creation did not perturb streams; restore exact state
            # before the stage regardless of future initialization internals.
            actual = ctx.rt.rng()
            assert torch.equal(state['cpu'], actual['cpu'])
            assert len(state['cuda']) == len(actual['cuda'])
            assert all(torch.equal(a,b) for a,b in zip(state['cuda'], actual['cuda']))
            ctx.rt.restore_rng(state)
            head.requires_grad_(ctx.arm != 'plain')
            model.add_module('_type_aux_head', head)
            operators.write(ctx.run / 'HEAD_INITIAL.json', dict(
                digest=ctx.storage.digest(head), parameters=sum(p.numel() for p in head.parameters()),
                trainable=ctx.arm != 'plain', RNG_restored=True))
            if ctx.arm != 'plain':
                method.training_loss = training_loss

    def optimizer(m):
        assert not optimizer_holder
        parameters = [p for p in model.parameters() if p.requires_grad]
        expected = 543363 if monitor.stage == 16 and ctx.arm != 'plain' else 540672
        assert sum(p.numel() for p in parameters) == expected
        opt = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9,.999), eps=1e-8,
                               weight_decay=0, foreach=False, fused=False, capturable=True)
        optimizer_holder.append(opt)
        return opt

    raw_clip = torch.nn.utils.clip_grad_norm_
    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        stage = monitor.stage
        counts[stage] = counts.get(stage, 0) + 1
        grads = [p.grad for p in parameters if p.grad is not None]
        expected = 543363 if stage == 16 and ctx.arm != 'plain' else 540672
        assert sum(g.numel() for g in grads) == expected
        assert all(g.is_cuda and torch.isfinite(g).all() for g in grads)
        return raw_clip(parameters, *args, **kwargs)

    def after_stage(stage):
        assert frozen() == frozen_before
        assert ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == buffers_before
        opt = optimizer_holder.pop()
        states = [v for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
        assert all(v.is_cuda and torch.isfinite(v).all() for v in states)
        operators.write(ctx.run / f'STAGE_{stage:02}_COMPLETE.json', dict(
            stage=stage, updates=counts[stage], optimizer_state_bytes=sum(v.numel()*v.element_size() for v in states),
            model_digest=ctx.storage.digest(model.core)))
        if stage == 16:
            if hasattr(method, 'training_loss'):
                del method.training_loss
            # Preserve trained head for diagnosis but it is not a deployment dependency.
            torch.save({k:v.detach().cpu() for k,v in model._type_aux_head.state_dict().items()},
                       ctx.run / 'training_head.pt')
            del model._type_aux_head
        del states, opt
        operators.stage_readout(ctx, model, ctx.manifest['tasks'][stage-1],
                               'DEV_ACQUIRED', acquired, readout_events)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    monitor.snapshot('training_before', model=model)
    namespace = dict(vars(formal), model=model, method=method, processor=ctx.processor,
        manifest=ctx.manifest, data=data, run=directory, monitor=monitor,
        args=SimpleNamespace(method='SeqFT_LoRA'), resume_obj=None,
        input_batch=batcher, answer_loss=ctx.rt.answer_loss, prepare=lambda *unused: None,
        visual_cache_stats=ctx.fv.stats, make_optimizer=optimizer,
        before_stage=before_stage, after_stage=after_stage)
    prefix, digest = ctx.study.training_ast()
    assert digest == AST_SHA
    loop = ast.For(target=ast.Name(id='stage', ctx=ast.Store()),
        iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()),
                      args=[ast.Constant(first), ast.Constant(last+1)], keywords=[]),
        body=ast.parse('before_stage(stage)').body+prefix+ast.parse('after_stage(stage)').body, orelse=[])
    if checkpoint_rng is not None:
        ctx.rt.restore_rng(checkpoint_rng)
    else:
        ctx.reset()
    torch.nn.utils.clip_grad_norm_ = clip
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])),
                     '<type-aux-original-stage-training>', 'exec'), namespace)
    finally:
        torch.nn.utils.clip_grad_norm_ = raw_clip
    assert sum(counts.values()) == expected_steps and not optimizer_holder
    assert len(batcher.records) == sum(p['QA'] for p in plan)
    assert len(supervision) == (69 if ctx.arm in ('true', 'shuffled') else 0)
    assert not hasattr(method, 'training_loss') and not hasattr(model, '_type_aux_head')
    assert frozen() == frozen_before
    model.core.language_model.disable_input_require_grads()
    model.core.language_model.gradient_checkpointing_disable()
    namespace.clear()
    monitor.snapshot('training_after', model=model)
    memory = monitor.finish(model)
    assert not monitor.thread.is_alive()
    for name,value in [('before_predictions.json',before),('acquired_predictions.json',acquired),
                       ('READOUT_EVENTS.json',readout_events),('FIT_INPUTS.json',batcher.records),
                       ('SUPERVISION.json',supervision),('TRAIN_MEMORY.json',memory)]:
        operators.write(ctx.run/name,value)
    return dict(training_plan=plan, optimizer_updates=expected_steps,
                FIT_backwards=len(batcher.records), before_QA=len(before), acquired_QA=len(acquired))
