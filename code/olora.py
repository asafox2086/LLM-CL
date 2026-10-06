"""O-LoRA: frozen historical adapters with orthogonal current-task updates.

Algorithm reference: cmnfriend/O-LoRA, src/uie_trainer_lora.py and
src/peft/tuners/lora.py. Historical contributions remain separate to avoid
repeatedly requantizing the frozen base. All adapters contribute at inference.
"""

import math

import torch
from torch import nn
from torch.nn import functional as functional


class OrthogonalLinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.adapters = nn.ModuleList()

    def add_task(self):
        self.requires_grad_(False)
        device = self.base.weight.device
        adapter = nn.ModuleDict({
            'down': nn.Linear(self.base.in_features, self.rank, bias=False, device=device, dtype=torch.float32),
            'up': nn.Linear(self.rank, self.base.out_features, bias=False, device=device, dtype=torch.float32),
        })
        nn.init.kaiming_uniform_(adapter['down'].weight, a=math.sqrt(5))
        nn.init.zeros_(adapter['up'].weight)
        self.adapters.append(adapter)

    def forward(self, inputs):
        output = self.base(inputs)
        if not self.adapters:
            return output
        dropped = self.dropout(inputs)
        for adapter in self.adapters:
            delta = adapter['up'](adapter['down'](dropped.to(adapter['down'].weight.dtype)))
            output = output + delta.to(output.dtype) * self.scale
        return output

    def regularization(self):
        current = self.adapters[-1]
        with torch.autocast(device_type=current['down'].weight.device.type, enabled=False):
            current_down = current['down'].weight.float()
            orthogonal = current_down.new_zeros(())
            for historical in self.adapters[:-1]:
                orthogonal = orthogonal + functional.linear(historical['down'].weight.float(), current_down).abs().sum()
            l2 = current_down.norm() + current['up'].weight.float().norm()
        return orthogonal, l2


class OLoRA:
    def __init__(self, model, rank=8, alpha=32, dropout=0.1, targets=('q_proj', 'v_proj')):
        self.model = model
        model.requires_grad_(False)
        self.layers = {}
        for name, module in list(model.named_modules()):
            if name.rsplit('.', 1)[-1] not in targets:
                continue
            parent_name, child_name = name.rsplit('.', 1)
            wrapper = OrthogonalLinear(module, rank, alpha, dropout)
            setattr(model.get_submodule(parent_name), child_name, wrapper)
            self.layers[name] = wrapper
        if not self.layers:
            raise ValueError('No target linear layers found')

    @property
    def task_count(self):
        return len(next(iter(self.layers.values())).adapters)

    def begin_task(self):
        for layer in self.layers.values():
            layer.add_task()

    def parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def penalty(self, orthogonal_weight, l2_weight):
        penalties = [layer.regularization() for layer in self.layers.values()]
        return sum(orthogonal * orthogonal_weight + l2 * l2_weight for orthogonal, l2 in penalties)

    def save(self, path):
        torch.save({
            'task_count': self.task_count,
            'layers': {name: {key: value.detach().cpu() for key, value in layer.adapters.state_dict().items()}
                       for name, layer in self.layers.items()},
        }, path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if self.task_count > state['task_count']:
            raise ValueError('Cannot load a checkpoint with fewer tasks into this model')
        while self.task_count < state['task_count']:
            self.begin_task()
        if set(state['layers']) != set(self.layers):
            raise ValueError('Adapter target layers differ from checkpoint')
        for name, layer in self.layers.items():
            layer.adapters.load_state_dict(state['layers'][name], strict=True)
