"""GPU-only response transfer on the fixed eight-FFN-deleted graph.

The caller constructs models, binds native input transport, resets paired RNG,
enables gradient checkpointing/input gradients, and owns all serialization.
Memory is list[(task_dict, row_dict)] in the prescribed bottom-k order within
each domain. No model is constructed or moved to CPU by this module.
"""
import hashlib
import json
import math
import time

DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
DELETED = (2, 5, 13, 16, 17, 18, 20, 21)
ADAPTER_COORDINATES = 3293184
ADAPTER_PARAMETERS = 288


def configure_gpu_math():
    import torch
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return dict(cuda_available=True, matmul_allow_tf32=False,
                cudnn_allow_tf32=False, float32_matmul_precision='highest',
                adapter_parameters='torch.float32',
                adapter_forward='unchanged native SeqLinear under outer CUDA FP16 autocast',
                losses='GPU FP32', fold='GPU FP32 product/add, autocast disabled, then FP16')


def _device(model):
    import torch
    device = next(model.parameters()).device
    assert device.type == 'cuda' and device.index is not None
    core = getattr(model, 'core', model)
    assert all(p.device == device for p in core.parameters())
    assert all(v.device == device for _, v in core.named_buffers())
    return torch.device(device)


def _sync(*devices):
    import torch
    for device in dict.fromkeys(str(d) for d in devices if d is not None):
        torch.cuda.synchronize(torch.device(device))


def remaining_modules():
    return {f'model.layers.{i}.{suffix}' for i in range(24)
            for suffix in ('self_attn.q_proj', 'self_attn.k_proj',
                           'self_attn.v_proj', 'self_attn.o_proj',
                           'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
            if not (i in DELETED and suffix.startswith('mlp.'))}


def install_remaining_lora(student, target_lora_class):
    """Use the original TargetLoRA and its caller-seeded initialization policy."""
    import torch
    _device(student)
    assert len(student.model.layers) == 24
    assert all(not list(student.model.layers[i].mlp.parameters()) for i in DELETED)
    method = target_lora_class(student, ['q_proj', 'v_proj', 'k_proj', 'o_proj',
                                        'gate_proj', 'up_proj', 'down_proj'])
    wanted = remaining_modules()
    modules = {n: m for n, m in student.named_modules() if n in wanted}
    assert set(modules) == wanted and len(modules) == len(method.layers) == 144
    assert {id(m) for m in modules.values()} == {id(m) for m in method.layers}
    method.rank_plan = {n: 8 for n in modules}
    records = []
    for name, layer in modules.items():
        assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        assert layer.A.weight.is_cuda and layer.B.weight.device == layer.A.weight.device
        assert layer.A.weight.shape == (8, layer.weight.shape[1])
        assert layer.B.weight.shape == (layer.weight.shape[0], 8)
        assert layer.scale == 4 and layer.dropout.p == .1
        assert bool(torch.isfinite(layer.A.weight).all())
        assert bool(torch.isfinite(layer.B.weight).all()) and torch.count_nonzero(layer.B.weight).item() == 0
        records.append(dict(module=name, rank=8, alpha=32, scale=4, dropout=.1,
            A_shape=list(layer.A.weight.shape), B_shape=list(layer.B.weight.shape),
            parameter_dtype='torch.float32', device=str(layer.weight.device), B_zero=True))
    trainable = {n: p for n, p in student.named_parameters() if p.requires_grad}
    names = {n + ending for n in wanted for ending in ('.A.weight', '.B.weight')}
    assert set(trainable) == names and len(trainable) == ADAPTER_PARAMETERS
    assert sum(p.numel() for p in trainable.values()) == ADAPTER_COORDINATES
    return method, dict(modules=144, parameters=ADAPTER_PARAMETERS,
        coordinates=ADAPTER_COORDINATES, parameter_bytes=ADAPTER_COORDINATES * 4,
        rank_plan=method.rank_plan, initialization=records,
        trainable_names=sorted(names), deleted=list(DELETED),
        initialization_policy='original TargetLoRA q/v stream and extra-module named fork_rng seeds; caller resets RNG',
        original_SeqLinear_forward_unchanged=True)


def answer_hidden(model, batch, answer, processor):
    """Native answer_loss prefix, through the decoder's final-normalized output.

Tokenize at most 512 answer tokens, append EOS, and select P-1:-1 so the L
retained hidden states predict all L targets, including EOS. This deliberately
matches native answer_loss's single unpadded QA mask and position construction.
"""
    import torch
    device = _device(model)
    assert batch['input_ids'].device == device and batch['input_ids'].shape[0] == 1
    assert model.dtype == torch.float16
    target = processor.tokenizer(answer, add_special_tokens=False,
        truncation=True, max_length=512)['input_ids']
    if not target:
        raise ValueError('Empty tokenized answer')
    target = list(target)
    target.append(processor.tokenizer.eos_token_id)
    assert all(isinstance(token, int) and token >= 0 for token in target)
    prompt_length = batch['input_ids'].shape[1]
    assert prompt_length > 0
    target_tensor = torch.tensor([target], device=device, dtype=torch.long)
    with torch.cuda.device(device), torch.autocast('cuda', dtype=torch.float16):
        ids = torch.cat([batch['input_ids'], target_tensor], dim=1)
        mask = torch.ones_like(ids)
        embeddings = model.get_input_embeddings()(ids)
        grid = batch.get('image_grid_thw')
        if 'pixel_values' in batch:
            visual = model.visual(batch['pixel_values'].to(model.visual.get_dtype()), grid_thw=grid)
            image_mask = (ids == model.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
            if image_mask.sum().item() != visual.numel():
                raise ValueError('image feature/token count mismatch')
            embeddings = embeddings.masked_scatter(image_mask, visual.to(embeddings.dtype))
        positions, _ = model.get_rope_index(ids, grid, attention_mask=mask)
        output = model.model(inputs_embeds=embeddings, attention_mask=mask,
            position_ids=positions, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state[:, prompt_length - 1:-1, :]
    assert hidden.shape[:2] == target_tensor.shape and hidden.device == device
    contract = dict(prompt_tokens=prompt_length, target_tokens=len(target),
        target_ids=target, target_sha256=hashlib.sha256(json.dumps(target, separators=(',', ':')).encode()).hexdigest(),
        target_truncation_before_EOS=512, EOS_included=True,
        hidden_slice='prompt_length-1:-1', decoder_last_hidden_state_after_final_norm=True)
    return hidden, target_tensor, contract


def round_robin_memory(memory):
    """Preserve the supplied hash order within each domain, never resample."""
    buckets = {d: [] for d in DOMAINS}
    ids = set()
    for task, row in memory:
        assert row['id'] not in ids
        ids.add(row['id'])
        buckets[task['domain']].append((task, row))
    assert len(ids) == 160 and all(len(v) == 32 for v in buckets.values())
    return [buckets[d][i] for i in range(32) for d in DOMAINS]


def cache_teacher(teacher, memory, processor, batch_provider, on_item=None):
    """160 no-grad decoder prefixes; retain only detached GPU answer hidden.

batch_provider(model, task, row) must bind native transport and returns
(batch, details). on_item(record, entry) may serialize metadata/tensors as I/O.
The returned entries never retain a model, full sequence, or autograd graph.
The caller may retain only teacher.lm_head and release the rest of the teacher.
"""
    import torch
    device = _device(teacher)
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    teacher.eval()
    ordered = round_robin_memory(memory)
    cache, records = {}, []
    _sync(device)
    started = time.monotonic()
    for index, (task, row) in enumerate(ordered, 1):
        _sync(device)
        tick = time.monotonic()
        with torch.no_grad():
            batch, details = batch_provider(teacher, task, row)
            hidden, _, contract = answer_hidden(teacher, batch, row['answer'], processor)
            hidden = hidden.detach().contiguous().clone()
        assert hidden.device == device and not hidden.requires_grad and hidden.grad_fn is None
        assert not hidden.is_inference() and bool(torch.isfinite(hidden).all())
        entry = dict(hidden=hidden, target_ids=contract['target_ids'],
            target_sha256=contract['target_sha256'], prompt_tokens=contract['prompt_tokens'],
            target_tokens=contract['target_tokens'], details=dict(details))
        cache[row['id']] = entry
        _sync(device)
        record = dict(index=index, id=row['id'], task=task['task'], domain=task['domain'],
            prompt_tokens=contract['prompt_tokens'], target_tokens=contract['target_tokens'],
            target_sha256=contract['target_sha256'], hidden_shape=list(hidden.shape),
            hidden_dtype=str(hidden.dtype), hidden_device=str(hidden.device),
            hidden_bytes=hidden.numel() * hidden.element_size(), details=dict(details),
            post_final_norm_shifted_prediction_positions=True, EOS_included=True,
            teacher_decoder_requests=1, teacher_head_requests=0,
            synchronized_wall_seconds=time.monotonic() - tick)
        records.append(record)
        if on_item is not None:
            on_item(record, entry)
        del batch, hidden
    assert len(cache) == 160
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    _sync(device)
    receipt = dict(teacher_decoder_requests=160, teacher_head_requests=0,
        new_generations=0, backwards=0, optimizer_updates=0, device=str(device),
        hidden_bytes=sum(r['hidden_bytes'] for r in records), records=records,
        synchronized_wall_seconds=time.monotonic() - started)
    return cache, receipt


def _teacher_probabilities(entry, teacher_head, student_device):
    import torch
    import torch.nn.functional as F
    teacher_device = _device(teacher_head)
    assert entry['hidden'].device == teacher_device
    assert all(not p.requires_grad and p.grad is None for p in teacher_head.parameters())
    assert teacher_head.weight.dtype == torch.float16
    teacher_head.eval()
    # no_grad, rather than inference_mode: the returned target is saved by the
    # student's autograd KL operation, so it must be an ordinary tensor.
    with torch.no_grad(), torch.cuda.device(teacher_device):
        with torch.autocast('cuda', dtype=torch.float16):
            logits = teacher_head(entry['hidden'])
        with torch.autocast('cuda', enabled=False):
            probabilities = F.softmax(logits.float(), dim=-1)
        assert bool(torch.isfinite(probabilities).all())
        probabilities = probabilities.to(device=student_device)
    assert not probabilities.requires_grad and not probabilities.is_inference()
    return probabilities


def ce_kl_loss(logits, targets, teacher_probabilities=None):
    """Mean answer-token CE; KL sums vocab then means answer tokens at T=1.

One call is one QA. The caller averages QA losses, not pooled token counts.
The KD objective is (CE + KL)/2, with fixed 1:1 mixture and temperature one.
"""
    import torch
    import torch.nn.functional as F
    assert logits.is_cuda and targets.device == logits.device
    assert logits.shape[:-1] == targets.shape and targets.shape[0] == 1
    with torch.autocast('cuda', enabled=False):
        values = logits.float()
        ce = F.cross_entropy(values.flatten(0, 1), targets.flatten())
        if teacher_probabilities is None:
            loss, kl = ce, None
        else:
            assert teacher_probabilities.shape == values.shape
            assert teacher_probabilities.device == values.device
            assert teacher_probabilities.dtype == torch.float32 and not teacher_probabilities.requires_grad
            kl = F.kl_div(F.log_softmax(values, dim=-1), teacher_probabilities,
                          reduction='none').sum(dim=-1).mean()
            loss = (ce + kl) * .5
    assert bool(torch.isfinite(ce)) and bool(torch.isfinite(loss))
    assert kl is None or bool(torch.isfinite(kl))
    return loss, dict(CE=ce.detach(), KL=None if kl is None else kl.detach())


def train_arm(student, method, memory, processor, batch_provider, arm,
              teacher_cache=None, teacher_head=None, on_qa=None, on_update=None):
    """Exactly 320 micro-QA backwards / 20 GPU AdamW updates, no DEV.

The caller has enabled input gradients/checkpointing and reset paired RNG.
on_qa(record) receives JSON-compatible metadata. After each actual update,
on_update(record, gradients) sees detached clipped FP32 GPU gradients before
zero_grad; callbacks may write proofs as I/O. All optimizer tensors stay CUDA.
"""
    import torch
    from transformers import get_cosine_schedule_with_warmup
    assert arm in ('CE', 'KD')
    device = _device(student)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == 'highest'
    named = {n: p for n, p in student.named_parameters() if p.requires_grad}
    expected = {n + s for n in remaining_modules() for s in ('.A.weight', '.B.weight')}
    assert set(named) == expected and len(named) == ADAPTER_PARAMETERS
    assert sum(p.numel() for p in named.values()) == ADAPTER_COORDINATES
    assert all(p.device == device and p.dtype == torch.float32 for p in named.values())
    assert set(method.rank_plan) == remaining_modules() and all(r == 8 for r in method.rank_plan.values())
    parameters = list(named.values())
    order = round_robin_memory(memory) * 2
    assert len(order) == 320
    teacher_device = None
    if arm == 'KD':
        assert teacher_cache is not None and teacher_head is not None
        assert set(teacher_cache) == {row['id'] for _, row in memory}
        teacher_device = _device(teacher_head)
        assert teacher_device != device
        assert all(not p.requires_grad and p.grad is None for p in teacher_head.parameters())
    else:
        assert teacher_cache is None and teacher_head is None
    student.train()
    qa_records, update_records = [], []
    with torch.cuda.device(device):
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=.01,
            betas=(.9, .999), eps=1e-8, foreach=False, fused=False, capturable=True)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)
        scaler = torch.amp.GradScaler('cuda', init_scale=1, enabled=True)
        _sync(device, teacher_device)
        started = time.monotonic()
        for update, offset in enumerate(range(0, 320, 16), 1):
            _sync(device, teacher_device)
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            minibatch = order[offset:offset + 16]
            loss_sum, ce_sum, kl_sum = 0., 0., 0.
            for micro, (task, row) in enumerate(minibatch, 1):
                batch, details = batch_provider(student, task, row)
                hidden, targets, contract = answer_hidden(student, batch, row['answer'], processor)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = student.lm_head(hidden)
                teacher_probabilities = None
                if arm == 'KD':
                    entry = teacher_cache[row['id']]
                    assert entry['target_ids'] == contract['target_ids']
                    assert entry['prompt_tokens'] == contract['prompt_tokens']
                    assert entry['target_tokens'] == contract['target_tokens']
                    for key in ('input_sha256', 'transport_sha256', 'packed_tokens'):
                        assert entry['details'][key] == details[key], (row['id'], key)
                    teacher_probabilities = _teacher_probabilities(entry, teacher_head, device)
                loss, parts = ce_kl_loss(logits, targets, teacher_probabilities)
                scaler.scale(loss / len(minibatch)).backward()
                loss_value = loss.detach().item()
                ce_value = parts['CE'].item()
                kl_value = None if parts['KL'] is None else parts['KL'].item()
                loss_sum += loss_value / 16
                ce_sum += ce_value / 16
                if kl_value is not None:
                    kl_sum += kl_value / 16
                record = dict(index=offset + micro, pass_index=offset // 160 + 1,
                    update=update, micro=micro, id=row['id'], task=task['task'], domain=task['domain'],
                    details=dict(details), prompt_tokens=contract['prompt_tokens'],
                    target_tokens=contract['target_tokens'], target_sha256=contract['target_sha256'],
                    loss=loss_value, CE=ce_value, KL=kl_value, backwards=1,
                    student_decoder_requests=1, student_head_requests=1,
                    teacher_decoder_requests=0, teacher_head_requests=int(arm == 'KD'))
                qa_records.append(record)
                if on_qa is not None:
                    on_qa(record)
                del batch, hidden, targets, logits, teacher_probabilities, loss, parts
            scaler.unscale_(optimizer)
            gradients = {n: p.grad.detach() for n, p in named.items() if p.grad is not None}
            assert set(gradients) == expected
            assert all(g.is_cuda and g.dtype == torch.float32 and bool(torch.isfinite(g).all())
                       for g in gradients.values())
            preclip = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            postclip = torch.linalg.vector_norm(torch.stack([g.float().norm(2) for g in gradients.values()]), 2)
            assert bool(torch.isfinite(preclip)) and bool(torch.isfinite(postclip)) and postclip.item() <= 1.0001
            lr_used = [group['lr'] for group in optimizer.param_groups]
            expected_lr = 1e-4 * .5 * (1 + math.cos(math.pi * (update - 1) / 20))
            assert all(abs(lr - expected_lr) <= 1e-15 and lr > 0 for lr in lr_used)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                raise FloatingPointError('Optimizer step skipped by GradScaler')
            scheduler.step()
            state_tensors = [value for state in optimizer.state.values()
                             for value in state.values() if isinstance(value, torch.Tensor)]
            assert state_tensors and all(value.device == device for value in state_tensors)
            assert all(bool(torch.isfinite(value).all()) for value in state_tensors)
            _sync(device, teacher_device)
            event = dict(step=update, qa_seen=offset + 16, total_qa=320,
                loss=loss_sum, CE=ce_sum, KL=kl_sum if arm == 'KD' else None,
                gradient_l2_preclip=preclip.item(), gradient_l2_postclip=postclip.item(),
                gradient_parameters=ADAPTER_PARAMETERS, gradient_bytes=ADAPTER_COORDINATES * 4,
                gradient_clip_threshold=1., learning_rate_used=lr_used,
                learning_rate_after_scheduler=scheduler.get_last_lr(), warmup_updates=0,
                optimizer_state_devices=sorted({str(value.device) for value in state_tensors}),
                synchronized_wall_seconds=time.monotonic() - tick)
            update_records.append(event)
            if on_update is not None:
                on_update(event, gradients)
        optimizer.zero_grad(set_to_none=True)
        _sync(device, teacher_device)
        seconds = time.monotonic() - started
        optimizer_state_bytes = sum(value.numel() * value.element_size() for value in state_tensors)
    assert len(qa_records) == 320 and len(update_records) == 20
    student.eval()
    return dict(arm=arm, FIT_QA_presentations=320, unique_memory_QA=160, passes=2,
        backwards=320, optimizer_updates=20, student_decoder_requests=320, student_head_requests=320,
        teacher_decoder_requests=0, teacher_head_requests=320 if arm == 'KD' else 0,
        teacher_prefix_cache_cost_separate=True, new_generations=0, DEV_used_for_optimization=0,
        TEST_QA=0, student_device=str(device), teacher_head_device=None if teacher_device is None else str(teacher_device),
        trainable_parameters=ADAPTER_PARAMETERS, adapter_coordinates=ADAPTER_COORDINATES,
        objective='CE' if arm == 'CE' else '(CE+KL(teacher||student))/2; T=1; lambda ratio=1',
        token_reduction='sum_vocab_for_KL_then_mean_answer_tokens_including_EOS', QA_reduction='equal_QA_then_batch16',
        optimizer=dict(name='torch.optim.AdamW', device=str(device), lr=1e-4, weight_decay=.01,
            betas=[.9, .999], eps=1e-8, capturable=True, foreach=False, fused=False,
            state_bytes=optimizer_state_bytes, all_state_tensors_cuda=True),
        schedule='cosine k=0..19 / 20; no warmup', clip=1.,
        order_ids=[row['id'] for _, row in order], qa_records=qa_records,
        update_records=update_records, synchronized_wall_seconds=seconds)


def _fold_one_gpu(layer):
    import torch
    from torch import nn
    device, dtype = layer.weight.device, layer.weight.dtype
    assert device.type == 'cuda' and dtype == torch.float16
    assert layer.A.weight.device == layer.B.weight.device == device
    assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == 'highest'
    with torch.cuda.device(device), torch.no_grad(), torch.autocast('cuda', enabled=False):
        merged = (layer.weight.detach().float() + layer.scale *
                  (layer.B.weight.detach().float() @ layer.A.weight.detach().float())).to(dtype)
        assert bool(torch.isfinite(merged).all())
        linear = nn.Linear(layer.weight.shape[1], layer.weight.shape[0],
            bias=layer.bias is not None, device=device, dtype=dtype)
        linear.weight.copy_(merged)
        if layer.bias is not None:
            linear.bias.copy_(layer.bias.detach())
    return linear


def fold_gpu(student, method):
    """Physically replace all 144 adapters; no dense residual or teacher remains."""
    import torch
    device = _device(student)
    assert set(method.rank_plan) == remaining_modules() and len(method.layers) == 144
    records = []
    for name, rank in method.rank_plan.items():
        layer = student.get_submodule(name)
        assert rank == 8 and layer.scale == 4
        linear = _fold_one_gpu(layer)
        parent, child = name.rsplit('.', 1)
        setattr(student.get_submodule(parent), child, linear)
        records.append(dict(module=name, rank=8, scale=4,
            shape=list(linear.weight.shape), dtype=str(linear.weight.dtype),
            merge_device=str(device), merge_dtype='torch.float32', output_device=str(device),
            matmul_allow_tf32=False, autocast_enabled=False))
    method.layers.clear()
    student.requires_grad_(False)
    student.eval()
    assert not any(n.endswith(('.A.weight', '.B.weight')) for n in student.core.state_dict())
    assert all(p.grad is None and not p.requires_grad for p in student.parameters())
    return records


def gpu_operator_check(device='cuda:1'):
    """One callable GPU-only seam check; never constructs a real model or QA.

This function is not executed on import. The caller decides whether/when to run
the single new check and records its cost independently from real experiments.
"""
    import torch
    import torch.nn.functional as F
    from types import SimpleNamespace
    configure_gpu_math()
    device = torch.device(device)
    assert device.type == 'cuda' and device.index is not None
    _sync(device)
    tick = time.monotonic()
    with torch.cuda.device(device):
        logits = torch.tensor([[[.2, -.3, .4], [.7, -.2, .1]]], device=device, requires_grad=True)
        target = torch.tensor([[0, 2]], device=device)
        teacher_logits = torch.tensor([[[.3, .1, -.4], [-.2, .6, .2]]], device=device)
        teacher_probs = F.softmax(teacher_logits, -1)
        loss, parts = ce_kl_loss(logits, target, teacher_probs)
        manual_kl = (teacher_probs * (teacher_probs.log() - F.log_softmax(logits, -1))).sum(-1).mean()
        assert torch.allclose(parts['KL'], manual_kl.detach(), atol=1e-6, rtol=1e-6)
        gradient, = torch.autograd.grad(loss, logits)
        expected = (F.softmax(logits.detach(), -1) - .5 *
                    (F.one_hot(target, 3).float() + teacher_probs)) / target.numel()
        assert torch.allclose(gradient, expected, atol=1e-6, rtol=1e-6)
        _, same = ce_kl_loss(logits.detach(), target, F.softmax(logits.detach(), -1))
        assert same['KL'].abs().item() < 1e-6
        # Repeating token positions must not change a QA's weight or loss.
        long_logits = logits.detach().repeat_interleave(3, dim=1)
        long_target = target.repeat_interleave(3, dim=1)
        long_teacher = teacher_probs.repeat_interleave(3, dim=1)
        long_loss, _ = ce_kl_loss(long_logits, long_target, long_teacher)
        assert torch.allclose(loss.detach(), long_loss, atol=1e-6, rtol=1e-6)
        # Distinct QA losses and lengths distinguish equal-QA weighting from
        # pooling all answer positions into one token-weighted objective.
        other_logits = logits.detach().flip(-1).repeat_interleave(3, dim=1)
        other_loss, _ = ce_kl_loss(other_logits, long_target, long_teacher)
        qa_mean = (loss.detach() + other_loss) / 2
        pooled_mean = (loss.detach() * target.numel() +
                       other_loss * long_target.numel()) / (target.numel() + long_target.numel())
        assert not torch.allclose(qa_mean, pooled_mean, atol=1e-5, rtol=1e-5)
        parameter = torch.nn.Parameter(torch.tensor([.1, -.2], device=device))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=.01,
            betas=(.9, .999), eps=1e-8, capturable=True, foreach=False, fused=False)
        old = parameter.detach().clone()
        parameter.square().mean().backward()
        optimizer.step()
        assert bool(torch.isfinite(parameter).all()) and not torch.equal(parameter.detach(), old)
        assert all(v.device == device for v in optimizer.state[parameter].values() if isinstance(v, torch.Tensor))
        base = torch.nn.Linear(4, 3, bias=True, device=device, dtype=torch.float16)
        a = torch.nn.Linear(4, 2, bias=False, device=device, dtype=torch.float32)
        b = torch.nn.Linear(2, 3, bias=False, device=device, dtype=torch.float32)
        layer = SimpleNamespace(weight=base.weight, bias=base.bias, A=a, B=b, scale=4.)
        merged = _fold_one_gpu(layer)
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            expected_weight = (base.weight.float() + 4 * (b.weight @ a.weight)).half()
            assert torch.equal(merged.weight, expected_weight) and torch.equal(merged.bias, base.bias)
            output = merged(torch.ones((1, 4), device=device, dtype=torch.float16))
            assert bool(torch.isfinite(output).all())
    _sync(device)
    return dict(passed=True, device=str(device), real_model_QA=0, CPU_model_probes=0,
        KL_vocabulary_sum_and_answer_mean=True, normalized_objective_gradient=True,
        teacher_equal_student_KL_zero=True, variable_length_QA_equal_weight=True,
        GPU_AdamW_state_and_update=True, GPU_FP32_fold_and_FP16_storage=True,
        synchronized_wall_seconds=time.monotonic() - tick)
