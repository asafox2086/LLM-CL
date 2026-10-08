"""Sequential LoRA baseline: one adapter updated throughout the task stream."""

import math

import torch
from torch import nn


class SequentialLinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.adapters = nn.ModuleList()

    def initialize(self):
        adapter = nn.ModuleDict({
            'down': nn.Linear(self.base.in_features, self.rank, bias=False, device=self.base.weight.device, dtype=torch.float32),
            'up': nn.Linear(self.rank, self.base.out_features, bias=False, device=self.base.weight.device, dtype=torch.float32),
        })
        nn.init.kaiming_uniform_(adapter['down'].weight, a=math.sqrt(5))
        nn.init.zeros_(adapter['up'].weight)
        self.adapters.append(adapter)

    def forward(self, inputs):
        output = self.base(inputs)
        if not self.adapters:
            return output
        adapter = self.adapters[0]
        hidden = adapter['down'](self.dropout(inputs).to(adapter['down'].weight.dtype))
        return output + adapter['up'](hidden).to(output.dtype) * self.scale


class SeqLoRA:
    def __init__(self, model, config):
        self.model = model
        self.task_count = 0
        model.requires_grad_(False)
        self.layers = {}
        for name, module in list(model.named_modules()):
            if name.rsplit('.', 1)[-1] in config['targets']:
                parent, child = name.rsplit('.', 1)
                wrapper = SequentialLinear(module, config['rank'], config['alpha'], config['dropout'])
                setattr(model.get_submodule(parent), child, wrapper)
                self.layers[name] = wrapper
        if not self.layers:
            raise ValueError('No LoRA target layers found')

    def begin_task(self):
        if not self.task_count:
            for layer in self.layers.values():
                layer.initialize()
        self.task_count += 1

    def parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def penalty(self, orthogonal_weight, l2_weight):
        return 0.0

    def save(self, path):
        torch.save({'task_count': self.task_count,
                    'layers': {name: {key: value.detach().cpu() for key, value in layer.adapters.state_dict().items()}
                               for name, layer in self.layers.items()}}, path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if set(state['layers']) != set(self.layers) or state['task_count'] < 1:
            raise ValueError('Invalid SeqLoRA checkpoint layout')
        if not self.task_count:
            self.begin_task()
        self.task_count = state['task_count']
        for name, layer in self.layers.items():
            layer.adapters.load_state_dict(state['layers'][name], strict=True)
