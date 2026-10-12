"""CPU regression: an interrupted update must match uninterrupted training."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_olora as engine


class RecoveryTest(unittest.TestCase):
    def test_interrupted_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=2))
            tokenizer = type('Tokenizer', (), {'pad_token_id': 0, 'eos_token_id': 2,
                'batch_decode': lambda self, rows, **kwargs: ['answer' for row in rows]})()
            rows = [{'task_id': 'task', 'instance_id': str(i), 'prompt': 'input',
                     'prompt_ids': [1, 3], 'references': ['answer']} for i in range(6)]
            config = {'eval_batch_size': 2, 'target_tokens': 2}
            output = Path(directory) / 'predictions.jsonl'
            calls = []
            def generate(**kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise KeyboardInterrupt('simulated interruption')
                inputs = kwargs['input_ids']
                return torch.cat([inputs, torch.full((inputs.shape[0], 1), 2)], dim=1)
            with patch.object(model, 'generate', generate):
                with self.assertRaises(KeyboardInterrupt):
                    engine.evaluate(model, tokenizer, rows, config, output, {'stage': 0})
                # A torn final record must not discard the preceding complete batch.
                with output.with_suffix('.jsonl.tmp').open('a') as handle:
                    handle.write('{')
                scores = engine.evaluate(model, tokenizer, rows, config, output, {'stage': 0}, resume=True)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(scores['reused_predictions'], 2)
            self.assertEqual(scores['exact_match'], 100)
            self.assertEqual([r['instance_id'] for r in records], [str(i) for i in range(6)])
            self.assertTrue(all('generated_token_ids' in r and 'prompt' in r for r in records))

    def test_interrupted_training(self):
        torch.set_num_threads(1)
        for name in ('olora', 'migu_lora', 'sapt_lora', 'seq_lora'):
            with self.subTest(method=name), tempfile.TemporaryDirectory() as directory:
                config = json.loads((engine.ROOT / 'exp/configs' / f'{name}_first.json').read_text())
                config.update(epochs=2, batch_size=1, micro_batch_size=1, warmup_ratio=0,
                              checkpoint_selection='last_epoch', rank=2)
                def fresh():
                    random.seed(42)
                    np.random.seed(42)
                    torch.manual_seed(42)
                    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
                        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=2))
                    return engine.create_method(model, config)
                rows = [{'instance_id': str(i), 'prompt_ids': [1, 3],
                         'input_ids': [1, 3, 4 + i, 2], 'labels': [-100, -100, 4 + i, 2]}
                        for i in range(3)]
                tokenizer = type('Tokenizer', (), {'pad_token_id': 0})()
                full = fresh()
                engine.train_task(full, tokenizer, {'train': rows}, config, Path(directory)/'full', 'task')
                expected = [p.detach().clone() for p in full.parameters()]
                expected_rng = torch.get_rng_state().clone()
                interrupted = fresh()
                original_save = engine.save_training
                def stop_after_save(*args, **kwargs):
                    original_save(*args, **kwargs)
                    if kwargs['update'] == 2:
                        raise KeyboardInterrupt('simulated interruption')
                with patch.object(engine, 'save_training', stop_after_save):
                    with self.assertRaises(KeyboardInterrupt):
                        engine.train_task(interrupted, tokenizer, {'train': rows}, config,
                                          Path(directory)/'resumed', 'task')
                resumed = fresh()
                engine.train_task(resumed, tokenizer, {'train': rows}, config,
                                  Path(directory)/'resumed', 'task', resume=True)
                for left, right in zip(expected, resumed.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
                records = [json.loads(line) for line in (Path(directory)/'resumed/training.jsonl').read_text().splitlines()]
                self.assertEqual([r['update'] for r in records], list(range(1, 7)))


if __name__ == '__main__':
    unittest.main()
