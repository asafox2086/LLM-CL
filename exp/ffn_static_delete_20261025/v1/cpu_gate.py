"""Real tiny Qwen2 CPU checks; no pretrained assets, CUDA, QA, or training run."""
import io
import json
import time

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from core import GatedFFN, ZeroFFN, delete, digest, layout, state_cpu


def main():
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    half_original = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU(),
                                        torch.nn.Linear(8, 4)).half().eval()
    half_input = torch.randn(2, 3, 4, dtype=torch.float16)
    with torch.no_grad():
        half_expected = half_original(half_input).clone()
    half_original_digest = digest(half_original)
    half_gate = GatedFFN(half_original)
    half_output = half_gate(half_input)
    assert half_output.dtype == torch.float16 and torch.equal(half_output, half_expected)
    half_output.float().square().sum().backward()
    assert half_gate.gate.dtype == torch.float32
    assert half_gate.gate.grad is not None and half_gate.gate.grad.dtype == torch.float32
    assert torch.isfinite(half_gate.gate.grad) and half_gate.gate.grad.abs() > 0
    assert all(p.grad is None for p in half_original.parameters())
    assert digest(half_original) == half_original_digest
    del half_original, half_input, half_expected, half_gate, half_output
    config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=4, num_attention_heads=2,
                         num_key_value_heads=2, max_position_embeddings=64,
                         attention_dropout=0., tie_word_embeddings=True,
                         use_cache=False)
    config._attn_implementation = 'eager'
    model = Qwen2ForCausalLM(config).eval()
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    def logits(m):
        return m(input_ids=ids, use_cache=False).logits
    original_digest = digest(model)
    original_layout = layout(model)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    # The storage counter must remove the head's duplicate state_dict entry.
    assert original_layout['weights']['coefficients'] == sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        expected = logits(model).clone()
    model.requires_grad_(False)
    original_ffn_digest = digest(model.model.layers[1].mlp)
    model.model.layers[1].mlp = GatedFFN(model.model.layers[1].mlp)
    gate = model.model.layers[1].mlp.gate
    assert gate.dtype == torch.float32
    gated = logits(model)
    assert torch.equal(gated, expected), 'gate=1 changed the forward'
    (gated.square().sum()).backward()
    assert gate.grad is not None and torch.isfinite(gate.grad) and gate.grad.abs() > 0
    assert all(p.grad is None for n, p in model.named_parameters() if not n.endswith('.gate'))
    assert digest(model.model.layers[1].mlp.original) == original_ffn_digest
    model.model.layers[1].mlp = model.model.layers[1].mlp.original
    assert digest(model) == original_digest

    before = layout(model)
    removed = delete(model, [1, 3])
    assert removed['coefficients'] == 2 * 3 * 16 * 32
    assert removed['bytes'] == removed['coefficients'] * 4
    assert layout(model)['weights']['coefficients'] == before['weights']['coefficients'] - removed['coefficients']
    assert all(isinstance(model.model.layers[i].mlp, ZeroFFN) for i in [1, 3])
    assert not any(f'model.layers.{i}.mlp.' in n for i in [1, 3] for n in model.state_dict())
    captured, handles = {}, []
    for i in [1, 3]:
        def capture_input(module, args, i=i):
            captured[i] = args[0].detach().clone()
        def check_residual(module, args, result, i=i):
            value = result[0] if isinstance(result, tuple) else result
            assert torch.equal(value, captured[i]), 'deleted MLP changed attention residual'
        handles.append(model.model.layers[i].post_attention_layernorm.register_forward_pre_hook(capture_input))
        handles.append(model.model.layers[i].register_forward_hook(check_residual))
    with torch.no_grad():
        deleted_logits = logits(model).clone()
    for handle in handles:
        handle.remove()
    snapshot = state_cpu(model)
    assert snapshot['lm_head.weight'] is snapshot['model.embed_tokens.weight']
    stream = io.BytesIO()
    torch.save(snapshot, stream)
    stream.seek(0)
    restored_state = torch.load(stream, map_location='cpu', weights_only=True)
    restored = Qwen2ForCausalLM(config).eval()
    delete(restored, [1, 3])
    restored.load_state_dict(restored_state, strict=True)
    assert digest(restored) == digest(model)
    assert layout(restored)['weights'] == layout(model)['weights']
    assert restored.lm_head.weight is restored.model.embed_tokens.weight
    with torch.no_grad():
        assert torch.equal(logits(restored), deleted_logits)
    assert not torch.cuda.is_initialized()
    print(json.dumps({'passed': True, 'removed': removed,
                      'original_layout': original_layout, 'deleted_layout': layout(model),
                      'complete_state_sha256': digest(model),
                      'strict_reload_equal': True, 'gate_one_equal': True,
                      'only_gate_gradient': True, 'deleted_residual_equal': True,
                      'FP16_gate_one_equal_FP32_gradient': True,
                      'tied_alias_preserved': True, 'cuda_initialized': False,
                      'seconds': time.monotonic() - started}, indent=2))


if __name__ == '__main__':
    main()
