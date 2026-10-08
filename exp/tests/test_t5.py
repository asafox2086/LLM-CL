"""T5 target alignment, attention masks, generation and exact checkpoint recovery."""
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_olora as engine
from common import tokenize_examples


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = None

    def encode(self, text, **kwargs):
        return [3 + ord(char) % 25 for char in text]

    def batch_decode(self, rows, **kwargs):
        return [' '.join(str(token) for token in row if token > 1) for row in rows]


def small_model():
    return T5ForConditionalGeneration(T5Config(vocab_size=32, d_model=16, d_kv=8,
        d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2,
        decoder_start_token_id=0, eos_token_id=1, pad_token_id=0))


class T5Test(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tokenizer = Tokenizer()
        self.config = {'model_architecture': 'seq2seq', 'prompt_tokens': 12, 'target_tokens': 5}
        self.raw = [{'task_id': 'task', 'instance_id': str(i), 'prompt': 'input' * (i + 1),
                     'references': ['reply'[:i + 1]]} for i in range(3)]
        self.rows, _ = tokenize_examples(self.tokenizer, self.raw, self.config)

    def test_target_and_masks(self):
        batch = engine.batch_tensors(self.rows, self.tokenizer, 'cpu', True, seq2seq=True)
        self.assertEqual(batch['input_ids'].shape[1], 12)
        self.assertEqual(batch['labels'].shape[1], 4)
        self.assertEqual(batch['labels'][0].tolist()[2:], [-100, -100])
        self.assertEqual(batch['decoder_attention_mask'][0].tolist(), [1, 1, 0, 0])
        self.assertEqual(self.rows[0]['input_ids'], self.rows[0]['prompt_ids'])
        self.assertEqual(self.rows[-1]['input_ids'][-1], 1)
        cfg = json.loads((engine.ROOT/'exp/configs/migu_lora_t5large.json').read_text())
        method = engine.create_method(small_model(), cfg)
        method.begin_task()
        method.begin_task()
        method.prepare_batch(self.rows, self.tokenizer, batch, True)
        for name, layer in method.layers.items():
            encoder = name.startswith('encoder.') or ('.EncDecAttention.' in name and name.endswith('.v'))
            mask = batch['attention_mask'] if encoder else batch['decoder_attention_mask']
            self.assertTrue(torch.equal(layer.token_mask, mask.bool()), name)
        model = method.model
        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                       labels=batch['labels'], decoder_attention_mask=batch['decoder_attention_mask']).logits
        self.assertEqual(logits.shape[:2], batch['labels'].shape)

    def test_generation_does_not_slice_off_prompt_length(self):
        model = small_model()
        def generate(**kwargs):
            return torch.tensor([[0, 7, 8, 1]] * kwargs['input_ids'].shape[0])
        with tempfile.TemporaryDirectory() as directory, patch.object(model, 'generate', generate):
            path = Path(directory)/'predictions.jsonl'
            config = {**self.config, 'eval_batch_size': 2}
            engine.evaluate(model, self.tokenizer, self.rows, config, path, {'stage': 0})
            predictions = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(all(row['prediction'] == '7 8' for row in predictions))
            self.assertTrue(all(row['generated_token_ids'] == [7, 8, 1] for row in predictions))

    def test_all_methods_resume_second_task(self):
        for name in ['olora', 'migu_lora', 'sapt_lora', 'seq_lora']:
            with self.subTest(method=name), tempfile.TemporaryDirectory() as directory:
                config = json.loads((engine.ROOT/f'exp/configs/{name}_t5large.json').read_text())
                config.update(self.config, epochs=2, precision='fp32_cpu', batch_size=2,
                              micro_batch_size=1, warmup_ratio=0, checkpoint_selection='last_epoch', rank=2)
                def fresh():
                    random.seed(42)
                    np.random.seed(42)
                    torch.manual_seed(42)
                    method = engine.create_method(small_model(), config)
                    method.begin_task()
                    if name == 'sapt_lora':
                        method.memory = [{'task_id': 'previous', 'prompt_ids': [[3, 4, 1]], 'attention': [1.0]}]
                    return method
                full = fresh()
                engine.train_task(full, self.tokenizer, {'train': self.rows}, config, Path(directory)/'full', 'task')
                expected = [p.detach().clone() for p in full.parameters()]
                expected_rng = torch.get_rng_state().clone()
                interrupted = fresh()
                save = engine.save_training
                def interrupt(*args, **kwargs):
                    save(*args, **kwargs)
                    if kwargs['update'] == 1:
                        raise KeyboardInterrupt('simulated interruption')
                with patch.object(engine, 'save_training', interrupt), self.assertRaises(KeyboardInterrupt):
                    engine.train_task(interrupted, self.tokenizer, {'train': self.rows}, config, Path(directory)/'resume', 'task')
                resumed = fresh()
                engine.train_task(resumed, self.tokenizer, {'train': self.rows}, config, Path(directory)/'resume', 'task', resume=True)
                for left, right in zip(expected, resumed.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
