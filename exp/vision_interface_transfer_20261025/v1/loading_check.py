"""Check parameter-only FP16 reload while preserving derived FP32 language buffers."""
import io,json,time
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
start=time.monotonic();torch.set_num_threads(1);torch.manual_seed(42)
cfg=Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,max_position_embeddings=64,tie_word_embeddings=True)
cfg._attn_implementation='sdpa'
model=Qwen2ForCausalLM(cfg).eval().requires_grad_(False);buffers={n:b.clone() for n,b in model.named_buffers()}
assert any('inv_freq' in n and b.dtype==torch.float32 for n,b in buffers.items())
with torch.no_grad():
 for p in model.parameters():p.data=p.data.to(torch.float16)
assert model.lm_head.weight is model.model.embed_tokens.weight
assert all(b.dtype==buffers[n].dtype and torch.equal(b,buffers[n]) for n,b in model.named_buffers())
stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0);state=torch.load(stream,weights_only=True)
restored=Qwen2ForCausalLM(cfg).eval().requires_grad_(False)
with torch.no_grad():
 for p in restored.parameters():p.data=p.data.to(torch.float16)
restored.load_state_dict(state,strict=True)
assert restored.lm_head.weight is restored.model.embed_tokens.weight
assert all(b.dtype==buffers[n].dtype and torch.equal(b,buffers[n]) for n,b in restored.named_buffers())
with torch.inference_mode():assert torch.equal(model(torch.tensor([[1,2,3]])).logits,restored(torch.tensor([[1,2,3]])).logits)
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,FP32_nonpersistent_RoPE_preserved=True,FP16_parameter_ties_preserved=True,strict_reload_logits_exact=True,CUDA_initialized=False,real_QA=0,seconds=time.monotonic()-start)))
