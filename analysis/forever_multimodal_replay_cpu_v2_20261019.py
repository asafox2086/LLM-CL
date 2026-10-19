"""Real CPU HF optimizer integration control, never a backbone result."""
import hashlib
import argparse
import json
from pathlib import Path
import tempfile

import torch

from forever_multimodal_core_20261019 import ROOT, ForeverStage
from forever_multimodal_replay_v2_20261019 import execute_replay, tensor_digest


def main():
    version = 'v2_amp_unwrap'
    suffix = Path('v2')
    report = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019'/suffix
    report.mkdir(parents=True, exist_ok=True)
    target = report/'REPLAY_CPU_CONTROL.json'
    assert not target.exists()
    assert not torch.cuda.is_initialized()
    # Synthetic scalar targets exercise actual HF accumulation/Adam/scheduling.
    # These are not images or actual experimental answers.
    history = [{'id': f'SYNTHETIC_CPU_ONLY:{i}', 'answer': str(i)} for i in range(64)]
    current = [{'id': 'SYNTHETIC_CPU_ONLY:current', 'answer': 'x'}]
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 1, bias=False)
    with tempfile.TemporaryDirectory(prefix='llmcl-forever-hf-') as temp:
        stage = ForeverStage(model, [history, current], 2, Path(temp)/'drift')
        with torch.no_grad():
            model.weight.add_(.01)
        stage.main_update_complete()
        stage.end_current()
        before = tensor_digest(dict(model.named_parameters()))

        def common_loss(model, row):
            index = int(row['id'].rsplit(':', 1)[1])
            x = torch.tensor([[1., (index % 7)/7., (index % 3)/3.]])
            return (model(x)-.25).pow(2).mean()

        result = execute_replay(stage, common_loss, Path(temp)/'replay', cpu_control=True)
        assert result['status'] == 'CPU_CONTROL_COMPLETE'
        assert result['actual_QA_exposures'] == 128
        assert result['actual_optimizer_updates'] == 8
        assert result['positive_LR_updates'] == 7
        assert result['author_zero_warmup_updates'] == 1
        assert tensor_digest(dict(model.named_parameters())) != before
        records = [json.loads(line) for line in
                   (Path(temp)/'replay/trace/updates.jsonl').read_text().splitlines()]
        assert len(records) == 8
        assert all(x['raw_gradient_l2'] > 0 for x in records)
        assert all(x['parameter_delta_l2'] > 0 for x in records[1:])
        assert records[0]['parameter_delta_l2'] == 0
        assert any(term['unscaled_regularizer'] > 0 for record in records
                   for term in record['micro_losses'])
        assert result['model_time_unchanged_by_replay']
        # Keep synthetic trace evidence, including actual raw gradients, loss,
        # and used LR, in exp/. Do not commit binary gradient tensors to Git.
        destination = ROOT/'exp/forever_multimodal_cpu_control_20261019'/suffix
        assert not destination.exists()
        import shutil
        shutil.copytree(Path(temp), destination)
    assert not torch.cuda.is_initialized()
    source_names = ['analysis/forever_multimodal_core_20261019.py',
                    'analysis/forever_multimodal_replay_v2_20261019.py',
                    'analysis/forever_multimodal_replay_cpu_v2_20261019.py']
    evidence = {
        'status': 'ACTUAL_HF_CPU_CONTROL_PASSED_NOT_GPU_ADMITTED',
        'synthetic_not_experimental_QA': True, 'GPU_forward': 0,
        'TEST_rows_or_predictions_read': 0, 'backbone_loaded': False,
        'formal_performance_result': False, 'control': result,
        'evidence_version': version,
        'used_LR': [x['learning_rate_used'] for x in records],
        'raw_gradient_l2': [x['raw_gradient_l2'] for x in records],
        'parameter_delta_l2': [x['parameter_delta_l2'] for x in records],
        'exp_trace': str(destination.relative_to(ROOT)),
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in source_names},
    }
    target.write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({'HF_CPU_control': 'PASS', 'updates': 8,
                      'positive_LR_updates': 7, 'GPU_admitted': False}))


if __name__ == '__main__':
    main()
