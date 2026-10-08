"""SAPT generative reflection, image-conditioned for visual tasks.
Stores pooled frozen pseudo-prompt embeddings rather than raw images in router memory.
"""
import json
import sys
import time
import torch
from common import write_json
from cv_data import processor, tokenize
from cv_runtime import prompt_pools, generate
from cv_train import train_task
from recovery import save_method

def finish_task(method, tokenizer, examples, destination, config):
    destination.mkdir(parents=True, exist_ok=True)
    complete = destination / 'stage_state.pt'
    if complete.exists():
        method.load(complete)
        return json.loads((destination/'resources.json').read_text())
    method.model.eval()
    started = time.monotonic()
    with torch.inference_mode():
        target = torch.zeros(method.task_count, device=method.model.device)
        for start in range(0, len(examples), 8):
            pool = prompt_pools(method, examples[start:start+8], tokenizer)
            target += method.router(pool).softmax(-1).sum(0)
        target = (target/len(examples)).cpu().tolist()
    proc = processor()
    def input_text(row):
        return (row['question']+'\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(row['choices']))) if row.get('choices') else row.get('question') or row['prompt'].split('\n\nInput: ',1)[-1].rsplit('\n\nResponse:\n',1)[0]
    generator_instruction = 'Generate a plausible question about this image.' if examples[0].get('image') else 'Generate a plausible medical multiple-choice question with four answer options A, B, C, D. Do not include its answer.'
    reconstructed = [tokenize({**row, 'prompt': generator_instruction,
        'references': [input_text(row)], 'training_answer': input_text(row)}, proc, row.get('image_grid_thw')) for row in examples]
    # Import the same module instance as the main SAPT method.
    generator_class = method.begin_task.__func__.__globals__['GeneratorMethod']
    generator = generator_class(method)
    auxiliary = {**config, 'epochs': config['sapt_generator_epochs'], 'checkpoint_selection': 'last_epoch'}
    with torch.random.fork_rng(devices=[method.model.device.index]):
        torch.manual_seed(config['seed']+10000+method.task_count)
        resources = train_task(generator, tokenizer, {'train': reconstructed}, auxiliary,
            destination/'generator', f'generator_task_{method.task_count}', resume=(destination/'generator/recovery.pt').exists())
        method.model.eval()
        count = config['sapt_pseudo_samples']
        generated = []
        pools = []
        for start in range(0, count, config['eval_batch_size']):
            selected = reconstructed[start:min(start+config['eval_batch_size'],count)]
            tokens, texts = generate(method.model, generator, tokenizer, selected, config, sample=True)
            pseudo = []
            for j, (row, ids, text) in enumerate(zip(selected, tokens, texts)):
                # Empty outputs remain auditable, with a deterministic training-only fallback.
                origin = examples[start+j]
                prompt = (origin['prompt'].split('\n\nInput: ',1)[0]+'\n\nInput: '+text+'\n\nResponse:\n'
                          if '\n\nInput: ' in origin['prompt'] else ('Answer the medical question briefly using the image.\n'+text if origin.get('image') else 'Answer this medical multiple-choice question. Respond only with the letter of the correct answer.\n\n'+text))
                if not text.strip():
                    prompt = origin['prompt']
                pseudo.append(tokenize({**origin,'prompt':prompt}, proc, origin.get('image_grid_thw')))
                generated.append({'text': text, 'tokens': ids, 'source_instance_id':origin['instance_id'],
                    'image':origin.get('image'), 'empty_fallback':not bool(text.strip()), 'prompt':prompt})
            pools.append(prompt_pools(method, pseudo, tokenizer).cpu())
            print(json.dumps({'event':'sapt_reflection','task':examples[0]['task_id'],'done':len(generated),'total':count}),flush=True)
    generator.release()
    method.memory.append({'task_id':examples[0]['task_id'], 'pools':torch.cat(pools), 'attention':target})
    write_json(destination/'pseudo_inputs.json',generated)
    resources.update(pseudo_count=count, attention_target=target, seconds_with_generation=time.monotonic()-started,
                     memory_bytes=sum(x.numel()*x.element_size() for x in pools),
                     adaptation='Image-conditioned pseudo questions; frozen text+image pools; train inputs only')
    write_json(destination/'resources.json',resources)
    save_method(method, complete)
    return resources
