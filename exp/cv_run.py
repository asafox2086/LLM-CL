"""Detached, resumable language->medical language->vision->medical vision CL."""
import argparse
import csv
import fcntl
import io
import json
import os
import random
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration
from common import ROOT, digest_bytes, fingerprint, write_json, continual_metrics
from recovery import atomic_save, capture_rng, restore_rng, method_bytes
from cv_data import CACHE, MODEL, processor
from cv_runtime import install_cached_vision, install_sapt, evaluate
from cv_train import train_task
from cv_reflection import finish_task
from run_olora import create_method

def implementation():
    paths = ['exp/cv_data.py','exp/cv_runtime.py','exp/cv_train.py','exp/cv_reflection.py','exp/cv_run.py',
             'exp/common.py','exp/recovery.py','exp/run_olora.py'] + ['code/'+m+'.py' for m in ['seq_lora','migu_lora','olora','sapt_lora']]
    return {p:digest_bytes((ROOT/p).read_bytes()) for p in paths}

def report(output, results, tasks):
    path = output/'scores.csv'
    with path.with_suffix('.tmp').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['stage','learned_task','evaluated_task','rougeL','exact_match','token_f1','count'])
        writer.writeheader()
        for stage, scores in enumerate(results):
            for task, metrics in scores.items():
                writer.writerow({'stage':stage,'learned_task':tasks[stage-1] if stage else 'base', 'evaluated_task':task,
                                 **{k:metrics[k] for k in ['rougeL','exact_match','token_f1','count']}})
    path.with_suffix('.tmp').replace(path)
    write_json(output/'scores.json',results)
    lines = ['| Stage | Task | ROUGE-L | EM (%) | F1 (%) |','|---|---|---:|---:|---:|']
    for stage, scores in enumerate(results):
        for task, s in scores.items():
            lines.append(f"| {stage} | {task} | {s['rougeL']:.3f} | {s['exact_match']:.3f} | {s['token_f1']:.3f} |")
    (output/'scores.md').write_text('\n'.join(lines)+'\n')

def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    config = json.loads(Path(args.config).read_text())
    config['implementation'] = implementation()
    manifest = json.loads((CACHE/'manifest.json').read_text())
    assert digest_bytes((CACHE/'data.pt').read_bytes()) == manifest['data_sha256']
    for name, sha in manifest['feature_sha256'].items():
        assert digest_bytes((CACHE/'features'/name).read_bytes()) == sha
    config['data_manifest_sha256'] = fingerprint(manifest)
    if (output/'config.json').exists():
        assert json.loads((output/'config.json').read_text()) == config, 'Resume config/code/data changed'
        assert args.resume, 'Existing run requires --resume'
    write_json(output/'config.json', config)
    write_json(output/'data_manifest.json', manifest)
    for path in config['implementation']:
        dest = output/'source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/path,dest)
    def status(state, **details):
        write_json(output/'status.json', {'status':state,'pid':os.getpid(),'time':time.time(),**details})
    status('running',phase='loading')
    try:
        torch.set_num_threads(4)
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed']); torch.cuda.manual_seed_all(config['seed'])
        data = torch.load(CACHE/'data.pt',weights_only=True)
        assert list(data) == config['tasks']
        proc = processor(); tokenizer = proc.tokenizer
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,
            torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
        install_cached_vision(model)
        method = create_method(model,config); install_sapt(method)
        assert len(method.layers) == 56 and all(name.startswith('model.layers.') for name in method.layers)
        write_json(output/'environment.json', {'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),'command':sys.argv,'base_parameters':sum(p.numel() for p in model.parameters()),
            'adapter_modules':list(method.layers)})
        results = []
        for stage in range(10):
            dest = output/'checkpoints'/f'stage_{stage:02d}'
            boundary = dest/'completed.pt'
            if boundary.exists():
                state = torch.load(boundary, map_location='cpu',weights_only=False)
                assert state['config_sha256'] == fingerprint(config)
                method.load(io.BytesIO(state['method'])); restore_rng(state['rng'])
                results = state['results']
                continue
            if stage:
                task = config['tasks'][stage-1]
                status('running',phase='training',stage=stage,task=task)
                train_task(method,tokenizer,data[task],config,dest/'train',task,resume=(dest/'train/recovery.pt').exists())
                if config['method']=='sapt_lora' and stage < 9:
                    status('running',phase='reflection',stage=stage,task=task)
                    finish_task(method,tokenizer,data[task]['train'],dest/'reflection',config)
            scores = {}
            for task in config['tasks']:
                status('running',phase='evaluation',stage=stage,task=task)
                scores[task] = evaluate(model,tokenizer,data[task]['test'],config,
                    output/'predictions'/f'stage_{stage:02d}'/f'{task}.jsonl',
                    {'stage':stage,'method':config['method'],'split':'test'},method,resume=True)
                report(output,results+[scores],config['tasks'])
            results.append(scores)
            atomic_save({'config_sha256':fingerprint(config),'method':method_bytes(method),
                'rng':capture_rng(),'results':results,'stage':stage},boundary)
        matrix = [[scores[t]['rougeL'] for t in config['tasks']] for scores in results]
        write_json(output/'continual_metrics.json',{'primary':'ROUGE-L (0-100)',**continual_metrics(matrix),'matrix':matrix})
        report(output,results,config['tasks'])
        status('completed',stage=9)
    except BaseException as exc:
        status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise

if __name__=='__main__':
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True);parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
