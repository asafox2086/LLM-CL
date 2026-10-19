"""Compare the new replay engine to an independent upstream HF-class control."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from transformers import Trainer, TrainingArguments

from forever_multimodal_core_20261019 import ROOT, source
from forever_multimodal_replay_20261019 import ReplayDataset, collate, tensor_digest


class CommonLossHFParent(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        row = inputs['forever_row']
        i = int(row['id'].rsplit(':', 1)[1])
        x = torch.tensor([[1., (i % 7)/7., (i % 3)/3.]])
        loss = (model(x)-.25).pow(2).mean()
        return (loss, {}) if return_outputs else loss


def main():
    report = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v1'
    destination = report/'ORIGINAL_HF_REFERENCE_CPU.json'
    assert not destination.exists()
    prior = ROOT/'exp/forever_multimodal_cpu_control_20261019/v1/replay'
    requested = json.loads((prior/'REQUESTED_ARGUMENTS.json').read_text())
    records = [json.loads(x) for x in (prior/'trace/updates.jsonl').read_text().splitlines()]
    original_path, tree, digest = source('src/finetune_cl_our_qwen.py')
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                and x.name == 'ReplayTrainer')
    namespace = {'MyTrainer': CommonLossHFParent, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(original_path), 'exec'), namespace)
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 1, bias=False)
    theta = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        model.weight.add_(.01)
    assert tensor_digest(dict(model.named_parameters())) == records[0]['parameter_before_sha256']
    allowed = TrainingArguments.__dataclass_fields__
    settings = {key: value for key, value in requested.items() if key in allowed}
    rows = [{'id': f'SYNTHETIC_CPU_ONLY:{i}', 'answer': str(i)} for i in range(64)]
    with tempfile.TemporaryDirectory(prefix='llmcl-forever-reference-') as temp:
        settings['output_dir'] = temp
        trainer = namespace['ReplayTrainer'](
            beta_t=requested['beta'], theta_star=theta,
            importance=dict.fromkeys(theta, 1.0), model=model,
            args=TrainingArguments(**settings), train_dataset=ReplayDataset(rows),
            data_collator=collate)
        result = trainer.train()
        actual = tensor_digest(dict(model.named_parameters()))
        assert actual == records[-1]['parameter_after_sha256'], 'New integration changed HF optimizer result'
        assert trainer.state.global_step == len(records) == 8
        log = [x for x in trainer.state.log_history if 'grad_norm' in x]
        assert len(log) == 8
        for old, ref in zip(records, log):
            # Callback uses FP64 norm accumulation; HF uses FP32. This verifies
            # the original framework's scale, not a newly normalized objective.
            assert abs(old['raw_gradient_l2']-ref['grad_norm']) < 2e-6
        assert not torch.cuda.is_initialized()
        evidence = {
            'status': 'ORIGINAL_HF_REFERENCE_EXACT_PARAMETER_MATCH',
            'original_ReplayTrainer_AST_unchanged': True,
            'source_sha256': digest, 'actual_reference_optimizer_updates': 8,
            'new_engine_final_parameter_sha256': records[-1]['parameter_after_sha256'],
            'original_HF_final_parameter_sha256': actual,
            'raw_gradient_norm_matches_original_HF_all_updates': True,
            'transformers_version': __import__('transformers').__version__,
            'requested_gradient_accumulation_steps': 16,
            'actual_accelerator_gradient_accumulation_steps': trainer.accelerator.gradient_accumulation_steps,
            'normalization_policy': 'Retain pinned author ReplayTrainer on installed HF; no silent scale repair. Shared main training remains separate.',
            'CPU_toy_only': True, 'GPU_forward': 0, 'backbone_loaded': False,
            'TEST_read': 0, 'formal_GPU_admitted': False,
        }
    destination.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'reference_parameter_match': True, 'updates': 8,
                      'GPU_admitted': False}))


if __name__ == '__main__':
    main()
