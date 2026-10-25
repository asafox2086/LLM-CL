"""Answer-readout mask for the inherited I-LoRA whole-hidden consistency.

Original packing is compiled directly from frozen runtime_utils.answer_loss;
only hidden-state return support and a no-unused-head teacher exit are added.
All decoder/head/MSE operations execute on GPU. No teacher is copied densely.
"""
import ast
import copy
import hashlib
import inspect
import math

_FORWARD = None
_SOURCE_AST_SHA256 = None


def _extended_forward():
    global _FORWARD, _SOURCE_AST_SHA256
    if _FORWARD is not None:
        return _FORWARD, _SOURCE_AST_SHA256
    import runtime_utils as rt
    original = ast.parse(inspect.getsource(rt.answer_loss)).body[0]
    assert isinstance(original, ast.FunctionDef) and original.name == 'answer_loss'
    assert [a.arg for a in original.args.args] == ['model', 'batch', 'answer', 'processor']
    _SOURCE_AST_SHA256 = hashlib.sha256(ast.dump(original, include_attributes=False).encode()).hexdigest()
    node = copy.deepcopy(original)
    node.name = '_answer_forward_with_hidden'
    node.args.args.append(ast.arg(arg='hidden_only'))
    node.args.defaults.append(ast.Constant(False))
    outputs = [n for n in node.body if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'output']
    assert len(outputs) == 1 and isinstance(outputs[0].value, ast.Call)
    output_call = outputs[0].value
    assert not any(k.arg == 'output_hidden_states' for k in output_call.keywords)
    output_call.keywords.append(ast.keyword(arg='output_hidden_states', value=ast.Constant(True)))
    # Same embeddings/visual packing/positions and same decoder; the teacher
    # needs hidden states only, so it does not execute a discarded answer head.
    output_index = node.body.index(outputs[0])
    early = ast.parse('if hidden_only: return None,len(target),output,None,target_tensor,prompt_length,ids.shape[1]').body[0]
    node.body.insert(output_index + 1, early)
    returns = [n for n in node.body if isinstance(n, ast.Return)]
    assert len(returns) == 1 and isinstance(returns[0].value, ast.Tuple) and len(returns[0].value.elts) == 2
    original_values = returns[0].value.elts
    extra = ast.parse('(output,logits,target_tensor,prompt_length,ids.shape[1])', mode='eval').body.elts
    returns[0].value = ast.Tuple(elts=original_values + extra, ctx=ast.Load())
    scope = dict(vars(rt))
    tree = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(tree, '<inherited_answer_loss_hidden_extension>', 'exec'), scope)
    _FORWARD = scope[node.name]
    return _FORWARD, _SOURCE_AST_SHA256


def answer_loss(model, batch, answer, processor, *, method=None, replay=False,
                arm='original', applied_mask=None):
    """Return (working CE tensor, answer count, diagnostics).

    Replay diagnostics contain one non-JSON key, consistency_loss_tensor,
    already multiplied by the chosen scalar mask. Caller pops it and forms
    (current_CE + replay_CE)/2 + consistency. All remaining fields are
    detached Python scalars/strings. No acceptance-count renormalization.
    """
    import torch
    import torch.nn.functional as F
    if not replay:
        import runtime_utils as rt
        loss, count = rt.answer_loss(model, batch, answer, processor)
        return loss, count, dict(replay=False, working_NLL=float(loss.detach()), answer_tokens=count)
    import adaptation
    assert method is not None and method.model is model
    assert arm in ('original', 'gate', 'shuffle')
    assert batch['input_ids'].is_cuda and batch['input_ids'].ndim == 2 and batch['input_ids'].shape[0] == 1
    assert not any(p.requires_grad for p in model.lm_head.parameters())
    forward, source_hash = _extended_forward()
    was_training = model.training
    ce, count, working, logits, target_ids, prompt_length, packed_length = forward(model, batch, answer, processor)
    assert ce.is_cuda and logits.is_cuda and target_ids.is_cuda
    assert logits.shape[:2] == target_ids.shape == (1, count)
    assert count >= 2 and count <= 513 and packed_length == prompt_length + count
    assert target_ids[0, -1].item() == processor.tokenizer.eos_token_id
    # no_grad preserves the source's training/dropout policy. Route selection
    # restores in finally even if a teacher forward raises an exception.
    with adaptation.slow_mode(method):
        _, slow_count, slow, slow_logits, slow_ids, slow_prompt, slow_length = forward(model, batch, answer, processor, hidden_only=True)
    assert model.training == was_training and slow_logits is None
    assert slow_count == count and slow_prompt == prompt_length and slow_length == packed_length
    assert torch.equal(target_ids, slow_ids)
    working_hidden, slow_hidden = working.hidden_states, slow.hidden_states
    assert isinstance(working_hidden, tuple) and isinstance(slow_hidden, tuple)
    assert len(working_hidden) == len(slow_hidden) == 25
    # Qwen2's last tuple entry already includes the original final norm.
    assert working_hidden[-1] is working.last_hidden_state
    assert slow_hidden[-1] is slow.last_hidden_state
    layer_losses = []
    target_last = None
    for index, (plastic, stable) in enumerate(zip(working_hidden, slow_hidden)):
        # An input-embedding hook may mark a no-grad input leaf as requiring
        # grad. The teacher target boundary is detached regardless of flags.
        stable = stable.detach()
        assert plastic.is_cuda and stable.is_cuda and plastic.device == stable.device
        assert plastic.shape == stable.shape == (1, packed_length, 896)
        assert not stable.requires_grad
        target = torch.maximum(plastic.detach(), stable.detach()).detach()
        assert torch.isfinite(plastic).all() and torch.isfinite(target).all()
        # Streaming layer means equal concatenated full-tuple MSE because all
        # 25 entries have the same packed-sequence/channel dimensions.
        layer_losses.append(F.mse_loss(plastic.float(), target.float(), reduction='mean'))
        if index == 24:
            target_last = target
    raw_mse = torch.stack(layer_losses).mean()
    assert raw_mse.is_cuda and raw_mse.requires_grad and torch.isfinite(raw_mse)
    with torch.no_grad():
        # No extra norm: use the actual post-finalnorm max target directly.
        target_logits = model.lm_head(target_last[:, prompt_length - 1:-1, :]).float()
        assert target_logits.shape == logits.shape
        target_nll = F.cross_entropy(target_logits.flatten(0, 1), target_ids.flatten())
        working_nll = ce.detach()
        assert torch.isfinite(target_nll) and torch.isfinite(working_nll)
        gate = (target_nll < working_nll).detach()
    natural_gate = int(gate.item())
    if arm == 'original':
        assert applied_mask is None
        mask = 1.0
    elif arm == 'gate':
        assert applied_mask is None
        mask = float(natural_gate)
    else:
        assert isinstance(applied_mask, (int, float)) and not isinstance(applied_mask, bool)
        assert math.isfinite(applied_mask) and float(applied_mask) in (0.0, 1.0)
        mask = float(applied_mask)
    consistency = raw_mse * mask
    stats = dict(replay=True, arm=arm, working_NLL=float(working_nll), target_NLL=float(target_nll),
                 natural_gate=natural_gate, applied_mask=mask, raw_MSE=float(raw_mse.detach()),
                 masked_MSE=float(consistency.detach()), answer_tokens=count, prompt_tokens=prompt_length,
                 packed_tokens=packed_length, hidden_states_count=25, hidden_width=896,
                 all_packed_positions_effective=True, target_detached=True,
                 teacher_same_training_mode=model.training == was_training,
                 slow_decoder_forwards=1, working_decoder_forwards=1, extra_target_head_calls=1,
                 slow_answer_head_calls=0, source_answer_loss_AST_sha256=source_hash,
                 consistency_loss_tensor=consistency)
    return ce, count, stats
