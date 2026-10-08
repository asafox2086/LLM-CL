"""Branch a completed seven-task run into a resumable two-task medical extension.

Stage 0 is the exact parent checkpoint (seven learned tasks), stages 1/2 learn
medical tasks 8/9. The old seven test sets are unchanged, never used for training.
"""
import argparse
import fcntl
import io
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from common import ROOT, digest_bytes, fingerprint, normalize_input, prepare_data, tokenize_examples, validate_model_config, write_json
from recovery import atomic_save, capture_rng, method_bytes, restore_rng
from run_olora import create_method, evaluate, train_task


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    import hashlib
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def load_data(config):
    parent = ROOT / config['parent_run']
    parent_config = read_json(parent / 'config.json')
    datasets, splits = prepare_data(parent_config)
    if splits != read_json(parent / 'split_manifest.json'):
        raise ValueError('Parent SuperNI data splits differ')
    manifest_path = ROOT / config['medical_manifest']
    if sha(manifest_path) != config['medical_manifest_sha256']:
        raise ValueError('Medical manifest changed')
    manifest = read_json(manifest_path)
    if [entry['task_id'] for entry in manifest['tasks']] != config['medical_tasks']:
        raise ValueError('Medical task order differs')
    for entry in manifest['tasks']:
        task = entry['task_id']
        datasets[task], splits[task] = {}, {}
        for split, meta in entry['splits'].items():
            path = ROOT / meta['path']
            if sha(path) != meta['sha256']:
                raise ValueError(f'Data checksum mismatch: {task}/{split}')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != meta['count'] or not rows:
                raise ValueError('Invalid medical split size')
            if len({r['instance_id'] for r in rows}) != len(rows):
                raise ValueError('Duplicate medical IDs')
            if any(r['task_id'] != task or not r['references'] or not all(r['references']) for r in rows):
                raise ValueError('Invalid medical example')
            datasets[task][split] = rows
            splits[task][split] = [{'instance_id': r['instance_id'], 'content_sha256': fingerprint(r)} for r in rows]
    # Include old and new tasks in the leakage audit, stripping task instructions.
    keys = {s: set() for s in ['train', 'dev', 'test']}
    for task_data in datasets.values():
        for split, rows in task_data.items():
            keys[split].update(normalize_input(r['prompt'].split('\n\nInput: ',1)[1].rsplit('\n\nResponse:',1)[0]).casefold() for r in rows)
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        if keys[a] & keys[b]:
            raise ValueError(f'Input leakage between {a} and {b}')
    return datasets, splits


def extension_metrics(rows, old_count):
    if not rows:
        return {}
    baseline, current = rows[0], rows[-1]
    old_mean = sum(current[:old_count]) / old_count
    med_mean = sum(current[old_count:]) / (len(current)-old_count)
    return {'old_task_AP': old_mean, 'old_task_AP_before_medical': sum(baseline[:old_count])/old_count,
            'old_task_AP_change': sum(current[i]-baseline[i] for i in range(old_count))/old_count,
            'medical_AP': med_mean, 'medical_AP_before_medical': sum(baseline[old_count:])/(len(current)-old_count),
            'medical_AP_gain': sum(current[i]-baseline[i] for i in range(old_count,len(current)))/(len(current)-old_count),
            'all_task_AP': sum(current)/len(current), 'medical_stages_completed': len(rows)-1,
            'metric': 'ROUGE-L (0-100); changes are score points, not clinical accuracy',
            'first_medical_task_change_after_second': current[old_count]-rows[1][old_count] if len(rows)==3 else None}


def write_tables(output, config, matrix):
    old_count = len(config['old_tasks'])
    metrics = extension_metrics(matrix, old_count)
    write_json(output / 'metrics.json', metrics)
    lines = ['# 医学续训阶段结果', '', f'父实验：`{config["parent_run"]}`', '',
             'ROUGE-L，0–100；阶段 7 是医学训练前，8/9 为两个新增任务之后。', '',
             '| 任务 | 医学前（7） | 医学任务一后（8） | 医学任务二后（9） | 相对医学前变化 |',
             '|---|---:|---:|---:|---:|']
    import csv
    with (output/'comparison.csv').open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['task','group','after7','after8','after9','change_from_after7'])
        for i,task in enumerate(config['old_tasks']+config['medical_tasks']):
            scores=[row[i] for row in matrix]
            padded=scores+[None]*(3-len(scores))
            delta=scores[-1]-scores[0]
            writer.writerow([task,'old' if i<old_count else 'medical',*padded,delta])
            lines.append('| '+task+' | '+' | '.join('—' if v is None else f'{v:.3f}' for v in padded)+f' | {delta:+.3f} |')
    lines += ['', f'旧任务平均分变化：{metrics["old_task_AP_change"]:+.3f}；医学任务平均分变化：{metrics["medical_AP_gain"]:+.3f}。',
              '', '全部逐题输入、参考答案、生成文字及三项分数见 predictions/；参数、checkpoint、恢复状态分别见 config.json、checkpoints/、status.json。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')


def worker(config, output, resume):
    parent = ROOT / config['parent_run']
    config_sha = fingerprint(config)
    started = time.monotonic()
    write_json(output/'status.json', {'status':'running','phase':'load_parent','pid':os.getpid()})
    try:
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.set_num_threads(4)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed']); torch.cuda.reset_peak_memory_stats()
        datasets, split_manifest = load_data(config)
        write_json(output/'split_manifest.json',split_manifest)
        model_path = ROOT / config['model_path']
        validate_model_config(read_json(model_path/'config.json'),config)
        tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        stats = {}
        for task,splits in datasets.items():
            stats[task] = {}
            for split,rows in splits.items():
                if config.get('smoke'):
                    rows = rows[:2]
                datasets[task][split],stats[task][split] = tokenize_examples(tokenizer,rows,config)
        write_json(output/'tokenization.json',stats)
        device = 'cpu' if config.get('smoke') else 'cuda:0'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True,
                    torch_dtype=torch.float32,attn_implementation='eager').to(device)
        method = create_method(model,config)
        parent_checkpoint = parent/'continual/checkpoints/stage_07/completed.pt'
        if sha(parent_checkpoint)!=config['parent_checkpoint_sha256']:
            raise ValueError('Parent checkpoint changed')
        state = torch.load(parent_checkpoint,map_location='cpu',weights_only=False)
        parent_worker_config = read_json(parent/'continual/config.json')
        if state['config_sha256'] != fingerprint(parent_worker_config):
            raise ValueError('Parent checkpoint configuration mismatch')
        method.load(io.BytesIO(state['method']))
        if method.task_count != len(config['old_tasks']):
            raise ValueError('Parent checkpoint has wrong number of learned tasks')
        restore_rng(state['rng'])
        parent_matrix = read_json(parent/'continual/score_matrix.json')
        if state['matrix'] != parent_matrix['rows'] or parent_matrix['tasks']!=config['old_tasks']:
            raise ValueError('Parent checkpoint score matrix mismatch')
        (output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
        write_json(output/'provenance.json',{'parent_checkpoint':str(parent_checkpoint),'parent_checkpoint_sha256':config['parent_checkpoint_sha256'],
            'initial_learned_tasks':method.task_count,'initial_memory_tasks':len(getattr(method,'memory',[])),
            'gpu':torch.cuda.get_device_name(0) if device!='cpu' else 'CPU','torch':torch.__version__,
            'optimizer_policy':'New optimizer per new task, identical to the original task-boundary protocol.',
            'baseline_policy':'Old-task predictions reused verbatim from completed parent stage 7; medical predictions freshly generated.',
            'command':sys.argv})
        tasks = config['old_tasks']+config['medical_tasks']
        matrix, resources = [], []
        for stage in range(len(config['medical_tasks'])+1):
            absolute_stage=len(config['old_tasks'])+stage
            boundary=output/'checkpoints'/f'stage_{absolute_stage:02d}'/'completed.pt'
            if resume and boundary.exists():
                saved=torch.load(boundary,map_location='cpu',weights_only=False)
                if saved['config_sha256']!=config_sha:
                    raise ValueError('Continuation checkpoint config differs')
                method.load(io.BytesIO(saved['method'])); restore_rng(saved['rng'])
                matrix,resources=saved['matrix'],saved['resources']
                print(json.dumps({'event':'stage_resume','completed_stage':absolute_stage}),flush=True)
                continue
            if stage:
                # The original final task had no next task, hence no SAPT finish_task.
                # Materialize its memory now, and likewise each new task before moving on.
                if hasattr(method,'finish_task'):
                    previous_task=tasks[absolute_stage-2]
                    reflection=output/'reflection'/f'stage_{absolute_stage-1:02d}'
                    write_json(output/'status.json',{'status':'running','phase':'auxiliary_generator','stage':absolute_stage-1,'task':previous_task,'pid':os.getpid()})
                    if resume and (reflection/'stage_state.pt').exists():
                        method.load(reflection/'stage_state.pt')
                        aux=read_json(reflection/'resources.json')
                    else:
                        aux=method.finish_task(tokenizer,datasets[previous_task]['train'],reflection,partial(train_task,resume=resume))
                    resources.append({'auxiliary_for_task':previous_task,'auxiliary':aux})
                task=config['medical_tasks'][stage-1]
                write_json(output/'status.json',{'status':'running','phase':'train','stage':absolute_stage,'task':task,'pid':os.getpid()})
                resources.append(train_task(method,tokenizer,datasets[task],config,output/'checkpoints'/f'stage_{absolute_stage:02d}',task,resume))
                if method.task_count!=absolute_stage:
                    raise ValueError('Wrong task count after continuation')
            row,per_task=[],{}
            for task in tasks:
                destination=output/'predictions'/f'stage_{absolute_stage:02d}'/f'{task}.jsonl'
                write_json(output/'status.json',{'status':'running','phase':'test','stage':absolute_stage,'task':task,'pid':os.getpid()})
                if stage==0 and task in config['old_tasks']:
                    source=parent/'continual/predictions/stage_07'/f'{task}.jsonl'
                    cached=[json.loads(line) for line in source.read_text().splitlines()]
                    examples=datasets[task]['test']
                    if len(cached)!=len(examples) or any(r['instance_id']!=e['instance_id'] or r['references']!=e['references'] or r['prompt']!=e['prompt'] for r,e in zip(cached,examples)):
                        raise ValueError('Parent prediction identity mismatch')
                    scores={metric:sum(r['scores'][metric] for r in cached)/len(cached) for metric in ['rougeL','exact_match','token_f1']}
                    if abs(scores['rougeL']-state['matrix'][-1][config['old_tasks'].index(task)])>1e-8:
                        raise ValueError('Parent score differs from predictions')
                    scores.update(count=len(cached),imported_from=str(source),source_sha256=sha(source))
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,destination)
                else:
                    metadata={'protocol':config['protocol'],'method':config['method'],'seed':config['seed'],'split':'test',
                              'stage':absolute_stage,'extension_stage':stage,'job':'medical_continuation'}
                    scores=evaluate(model,tokenizer,datasets[task]['test'],config,destination,metadata,method,resume)
                row.append(scores['rougeL']); per_task[task]=scores
                write_json(output/'scores'/f'stage_{absolute_stage:02d}'/f'{task}.json',scores)
            matrix.append(row)
            write_json(output/'score_matrix.json',{'tasks':tasks,'absolute_stages':list(range(len(config['old_tasks']),absolute_stage+1)),'rows':matrix})
            write_json(output/'stages'/f'stage_{absolute_stage:02d}.json',{'stage':absolute_stage,'learned_tasks':method.task_count,
                       'config_sha256':config_sha,'per_task':per_task,'resources':resources})
            write_json(output/'resources.json',{'stages':resources,'wall_seconds_this_process':time.monotonic()-started})
            atomic_save({'config_sha256':config_sha,'method':method_bytes(method),'rng':capture_rng(),'matrix':matrix,'resources':resources},boundary)
            write_tables(output,config,matrix)
        write_tables(output,config,matrix)
        write_json(output/'status.json',{'status':'completed','medical_stages':len(config['medical_tasks']),'total_learned_tasks':method.task_count})
    except BaseException as error:
        write_json(output/'status.json',{'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed','error':str(error),'traceback':traceback.format_exc()})
        raise


def launch(args):
    config=read_json(Path(args.resume)/'config.json' if args.resume else args.config)
    output=Path(args.resume).resolve() if args.resume else ROOT/'exp/result'/config['protocol']/config['run_label']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not args.resume:
        output.mkdir(parents=True,exist_ok=False)
    lock=(output/'.run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    process=None
    try:
        parent=ROOT/config['parent_run']
        if read_json(parent/'status.json')['status']!='completed':
            raise ValueError('Parent run is not complete')
        parent_config=read_json(parent/'config.json')
        # Keep model/method, optimizer hyperparameters, and token budgets identical to parent.
        for key in ['method','model_id','model_revision','rank','alpha','dropout','targets','precision','seed',
                    'prompt_tokens','target_tokens','batch_size','micro_batch_size','eval_batch_size','learning_rate','epochs',
                    'warmup_ratio','weight_decay','max_grad_norm','orthogonal_weight','l2_weight',
                    'migu_mask_ratio','sapt_router_dim','sapt_kl_weight','sapt_replay_batch_size','sapt_pseudo_samples','sapt_generator_epochs']:
            if config.get(key)!=parent_config.get(key):
                raise ValueError('Parent setting changed: '+key)
        method_file='code/'+config['method']+'.py'
        if sha(ROOT/method_file)!=read_json(parent/'code_identity.json')[method_file]:
            raise ValueError('Method implementation differs from parent')
        identity=read_json(parent/'model_identity.json')
        for name,digest in identity['files_sha256'].items():
            if sha(ROOT/config['model_path']/name)!=digest:
                raise ValueError('Base model differs from parent: '+name)
        code_files=['exp/medical_continuation.py','exp/run_olora.py','exp/common.py','exp/recovery.py',method_file]
        code_identity={name:sha(ROOT/name) for name in code_files}
        if args.resume and code_identity!=read_json(output/'code_identity.json'):
            raise ValueError('Resume implementation differs')
        if not args.resume:
            for name in code_files:
                destination=output/'source_snapshot'/name
                destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,destination)
            shutil.copy2(ROOT/'rules/003_medical_manifest.json',output/'medical_manifest.json')
            write_json(output/'code_identity.json',code_identity)
            write_json(output/'config.json',config)
            write_json(output/'parent_identity.json',{'parent':config['parent_run'],'checkpoint_sha256':config['parent_checkpoint_sha256'],'model':identity})
        load_data(config)
        log=output/'logs/continuation.log'; log.parent.mkdir(exist_ok=True)
        with log.open('a' if args.resume else 'w') as handle:
            command=[sys.executable,'-u',str(ROOT/'exp/medical_continuation.py'),'--worker','--config',str(output/'config.json'),'--output',str(output)]
            if args.resume: command+=['--resume-worker']
            env={**os.environ,'CUDA_VISIBLE_DEVICES':args.gpu,'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env,cwd=ROOT)
            write_json(output/'supervisor.json',{'pid':os.getpid(),'worker_pid':process.pid,'gpu':args.gpu,'command':command,'resume':bool(args.resume)})
            print(json.dumps({'run':str(output),'gpu':args.gpu,'worker_pid':process.pid}),flush=True)
            code=process.wait()
            if code: raise RuntimeError(f'Worker exit={code}; see {log}')
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        write_json(output/'supervisor_error.json',{'error':str(error),'traceback':traceback.format_exc()})
        raise
    finally:
        lock.close()

if __name__=='__main__':
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}; resume this run to continue')
    signal.signal(signal.SIGTERM,interrupt)
    parser=argparse.ArgumentParser()
    parser.add_argument('--config'); parser.add_argument('--gpu',default='0'); parser.add_argument('--resume')
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--output'); parser.add_argument('--resume-worker',action='store_true')
    args=parser.parse_args()
    if args.worker: worker(read_json(args.config),Path(args.output),args.resume_worker)
    else: launch(args)
