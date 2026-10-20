"""Portable FP16 OneVision runtime for genuinely narrower FFNs.

The only extra operation is the FLAP-inspired constant output compensation.
No quantization, sparse kernel, expert, router, or adapter is required.
"""
from torch import nn
from transformers import LlavaOnevisionForConditionalGeneration


class WidthOneVision(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, 'llmcl_compensated_text_down_bias', False):
            for layer in self.language_model.model.layers:
                old = layer.mlp.down_proj
                layer.mlp.down_proj = nn.Linear(
                    old.in_features, old.out_features, bias=True,
                    device=old.weight.device, dtype=old.weight.dtype)

