"""End-to-end continuation, parent isolation and interrupted optimizer recovery."""
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import medical_continuation as medical
import run_olora as engine
from common import ROOT, fingerprint, prepare_data, tokenize_examples, write_json
from recovery import atomic_save, capture_rng, method_bytes


def equal_state(a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: equal_state(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal_state(x,y)
    else: assert a==b,(a,b)


class MedicalContinuationTest(unittest.TestCase):
    def test_all_methods_continue_seven_to_nine_and_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='llmcl_medical_test_') as directory:
            root=Path(directory)
            model_path=root/'model'
            torch.manual_seed(42)
            model=T5ForConditionalGeneration(T5Config(vocab_size=32128,d_model=16,d_kv=8,d_ff=32,
                  num_layers=1,num_decoder_layers=1,num_heads=2,decoder_start_token_id=0,eos_token_id=1,pad_token_id=0))
            model.save_pretrained(model_path)
            for name in ['tokenizer.json','spiece.model']:
                shutil.copy2(ROOT/'model/t5-large'/name,model_path/name)
            tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
            for name in ['seq_lora','migu_lora','olora','sapt_lora']:
                with self.subTest(method=name):
                    config=medical.read_json(ROOT/f'exp/configs/{name}_t5large_medical2.json')
                    parent_config=medical.read_json(ROOT/config['parent_run']/'config.json')
                    config.update(smoke=True,precision='fp32_cpu',prompt_tokens=32,target_tokens=8,
                                  batch_size=1,micro_batch_size=1,eval_batch_size=2,rank=2,warmup_ratio=0,
                                  parent_run=str(root/name/'parent'),model_path=str(model_path))
                    parent=Path(config['parent_run']); parent.mkdir(parents=True)
                    parent_config.update(smoke=True,precision='fp32_cpu')
                    write_json(parent/'config.json',parent_config)
                    worker_config={**parent_config,'job':'continual'}
                    write_json(parent/'continual/config.json',worker_config)
                    data,splits=prepare_data(parent_config)
                    write_json(parent/'split_manifest.json',splits)
                    model=T5ForConditionalGeneration.from_pretrained(model_path)
                    method=engine.create_method(model,config)
                    for _ in range(7): method.begin_task()
                    # Nonzero trained-like adapters ensure we test loading learned state, not just zero-init.
                    with torch.no_grad():
                        for layer in method.layers.values():
                            for adapter in layer.adapters: adapter['up'].weight.normal_(std=.01)
                    if name=='sapt_lora':
                        method.memory=[{'task_id':config['old_tasks'][i],'prompt_ids':[[3,4,1]],
                                        'attention':[1/(i+1)]*(i+1)} for i in range(6)]
                    row=[]
                    for task in config['old_tasks']:
                        examples,_=tokenize_examples(tokenizer,data[task]['test'][:2],config)
                        score=engine.evaluate(model,tokenizer,examples,config,parent/'continual/predictions/stage_07'/f'{task}.jsonl',{'stage':7},method)
                        row.append(score['rougeL'])
                    matrix=[row]*8
                    write_json(parent/'continual/score_matrix.json',{'tasks':config['old_tasks'],'rows':matrix})
                    checkpoint=parent/'continual/checkpoints/stage_07/completed.pt'
                    atomic_save({'config_sha256':fingerprint(worker_config),'method':method_bytes(method),
                                 'rng':capture_rng(),'matrix':matrix,'resources':[]},checkpoint)
                    config['parent_checkpoint_sha256']=medical.sha(checkpoint)
                    original=checkpoint.read_bytes()
                    full=root/name/'full';full.mkdir();write_json(full/'config.json',config)
                    medical.worker(config,full,False)
                    self.assertEqual(medical.read_json(full/'status.json')['total_learned_tasks'],9)
                    self.assertEqual(len(medical.read_json(full/'score_matrix.json')['rows']),3)
                    expected=torch.load(full/'checkpoints/stage_09/completed.pt',weights_only=False)
                    self.assertEqual(checkpoint.read_bytes(),original)
                    # Stage 7 must preserve parent adapters and SAPT memory without modification.
                    start=torch.load(full/'checkpoints/stage_07/completed.pt',weights_only=False)
                    equal_state(torch.load(io.BytesIO(start['method']),weights_only=True),torch.load(io.BytesIO(torch.load(checkpoint,weights_only=False)['method']),weights_only=True))
                    interrupted=root/name/'interrupted';interrupted.mkdir();write_json(interrupted/'config.json',config)
                    save=engine.save_training
                    def interrupt(*args,**kwargs):
                        save(*args,**kwargs)
                        if kwargs['task_id']==config['medical_tasks'][0] and kwargs['update']==1:
                            raise KeyboardInterrupt('simulated medical training interruption')
                    with patch.object(engine,'save_training',interrupt),self.assertRaises(KeyboardInterrupt):
                        medical.worker(config,interrupted,False)
                    medical.worker(config,interrupted,True)
                    actual=torch.load(interrupted/'checkpoints/stage_09/completed.pt',weights_only=False)
                    equal_state(torch.load(io.BytesIO(expected['method']),weights_only=True),torch.load(io.BytesIO(actual['method']),weights_only=True))
                    self.assertEqual(expected['matrix'],actual['matrix'])
                    medical.worker(config,interrupted,True)
                    self.assertEqual(checkpoint.read_bytes(),original)
                    if name=='sapt_lora':
                        self.assertEqual(len(torch.load(io.BytesIO(actual['method']),weights_only=True)['memory']),8)
                    print('PASS:',name,'7 -> 9, unchanged parent, exact interrupted recovery, completed resume',flush=True)

if __name__=='__main__': unittest.main()
