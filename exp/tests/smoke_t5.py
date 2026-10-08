"""Seven-task CPU smoke with a tiny T5 and the real T5 tokenizer; no formal scores."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import T5Config, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='llmcl_t5_smoke_') as temporary:
    root = Path(temporary)
    model_path = root/'model'
    torch.manual_seed(42)
    model = T5ForConditionalGeneration(T5Config(vocab_size=32128, d_model=32, d_kv=8,
        d_ff=64, num_layers=1, num_decoder_layers=1, num_heads=4,
        decoder_start_token_id=0, eos_token_id=1, pad_token_id=0))
    model.save_pretrained(model_path)
    for filename in ['tokenizer.json', 'spiece.model']:
        shutil.copy2(ROOT/'model/t5-large'/filename, model_path/filename)
    for method in ['olora', 'migu_lora', 'sapt_lora', 'seq_lora']:
        config = json.loads((ROOT/f'exp/configs/{method}_t5large.json').read_text())
        config.update(smoke=True, precision='fp32_cpu', prompt_tokens=32, target_tokens=8,
                      batch_size=2, eval_batch_size=2, warmup_ratio=0, rank=2)
        config_path = root/f'{method}.json'
        config_path.write_text(json.dumps(config))
        output = root/method
        command = [sys.executable, '-u', str(ROOT/'exp/run_olora.py'), '--config', str(config_path),
                   '--model', str(model_path), '--output', str(output)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONNOUSERSITE': '1'}
        subprocess.run(command, env=env, check=True)
        matrix = json.loads((output/'score_matrix.json').read_text())
        assert len(matrix['rows']) == 8 and all(len(row) == 7 for row in matrix['rows'])
        assert len(list((output/'predictions').rglob('*.jsonl'))) == 56
        assert len(list((output/'checkpoints').rglob('completed.pt'))) == 8
        if method == 'sapt_lora':
            assert len(list((output/'reflection').rglob('stage_state.pt'))) == 6
        subprocess.run(command + ['--resume'], env=env, check=True)
        assert json.loads((output/'score_matrix.json').read_text()) == matrix
        print(f'PASS: {method} seven-stage training/evaluation/reflection and completed-stage resume', flush=True)
