import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, continual_metrics, fingerprint, prepare_data, write_json


def resolve_model(model_path, config):
    if model_path:
        path = Path(model_path).expanduser().resolve()
        source = {'kind': 'user_supplied_local_weights', 'requested_model_id': config['model_id']}
    else:
        from huggingface_hub import HfApi, snapshot_download
        info = HfApi().model_info(config['model_id'], revision=config.get('model_revision'), timeout=20)
        names = [sibling.rfilename for sibling in info.siblings]
        weight_pattern = '*.safetensors' if any(name.endswith('.safetensors') for name in names) else 'pytorch_model*.bin'
        path = Path(snapshot_download(config['model_id'], revision=info.sha,
                    allow_patterns=['*.json', 'tokenizer.model', weight_pattern], local_dir=ROOT / 'models/Llama-2-7b-hf'))
        source = {'kind': 'huggingface_snapshot', 'model_id': config['model_id'], 'revision': info.sha}
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Model config missing: {path}')
    files = sorted(file for file in path.iterdir() if file.is_file() and file.suffix in {'.json', '.model', '.safetensors', '.bin'}
                   and file.name != 'llmcl_identity.json')
    if not any(file.suffix in {'.safetensors', '.bin'} for file in files):
        raise FileNotFoundError('Model weight shards are missing')
    hashes = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        hashes[file.name] = digest.hexdigest()
    identity = {'source': source, 'files_sha256': hashes, 'weights_fingerprint': fingerprint(hashes)}
    write_json(path / 'llmcl_identity.json', identity)
    return path, identity


def launch(args):
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(args.output).resolve() if args.output else ROOT / 'exp/result' / config['protocol'] / config['run_label'] / stamp
    output.mkdir(parents=True, exist_ok=False)
    processes = []
    handles = []
    try:
        write_json(output / 'status.json', {'status': 'preparing', 'phase': 'data', 'pid': os.getpid()})
        datasets, splits = prepare_data(config)
        write_json(output / 'split_manifest.json', splits)
        write_json(output / 'config.json', config)
        print(f'Run directory: {output}', flush=True)
        if args.prepare_only:
            write_json(output / 'status.json', {'status': 'data_prepared', 'tasks': len(datasets)})
            return
        write_json(output / 'status.json', {'status': 'preparing', 'phase': 'model', 'pid': os.getpid()})
        try:
            model_path, identity = resolve_model(args.model, config)
        except Exception as error:
            write_json(output / 'status.json', {'status': 'blocked_model', 'error': str(error),
                       'action': 'Supply --model /path/to/Llama-2-7b-hf or configure authorized Hugging Face access; rerun with a fresh output directory.'})
            raise
        write_json(output / 'model_identity.json', identity)
        code_files = [ROOT / 'code/olora.py', ROOT / 'exp/common.py', ROOT / 'exp/run_olora.py',
                      ROOT / 'exp/launch_olora.py', ROOT / 'summary/collect.py']
        write_json(output / 'code_identity.json', {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files})
        gpus = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
        if not gpus or len(set(gpus)) != len(gpus):
            raise ValueError('Supply a nonempty list of distinct GPU IDs')
        jobs = [('continual', None)] + [('single', task) for task in config['tasks']]
        pending = list(jobs)
        active = {}
        failures = []
        write_json(output / 'status.json', {'status': 'running', 'pid': os.getpid(), 'jobs': len(jobs)})
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending:
                    continue
                job, task = pending.pop(0)
                relative = 'continual' if job == 'continual' else f'single/{task}'
                log_path = output / 'logs' / (relative.replace('/', '_') + '.log')
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open('w')
                handles.append(handle)
                command = [sys.executable, '-u', str(ROOT / 'exp/run_olora.py'), '--config', str(output / 'config.json'),
                           '--model', str(model_path), '--output', str(output / relative), '--job', job]
                if task:
                    command.extend(['--single-task', task])
                environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'TOKENIZERS_PARALLELISM': 'false',
                               'PYTHONUNBUFFERED': '1'}
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
                processes.append(process)
                active[gpu] = (process, relative)
                print(f'Started {relative} on GPU {gpu}, pid={process.pid}', flush=True)
            for gpu, (process, relative) in list(active.items()):
                code = process.poll()
                if code is not None:
                    print(f'Finished {relative}: exit={code}', flush=True)
                    if code:
                        failures.append({'job': relative, 'exit_code': code})
                    del active[gpu]
            write_json(output / 'status.json', {'status': 'running', 'pid': os.getpid(), 'pending': len(pending),
                       'active': {gpu: {'job': relative, 'pid': process.pid} for gpu, (process, relative) in active.items()}, 'failures': failures})
            if failures:
                raise RuntimeError(f'Worker failed; stopping remaining jobs: {failures}')
            if active:
                time.sleep(5)
        if failures:
            write_json(output / 'status.json', {'status': 'failed', 'failures': failures})
            raise RuntimeError(f'Worker failures: {failures}')
        matrix = json.loads((output / 'continual/score_matrix.json').read_text())
        if matrix['tasks'] != config['tasks']:
            raise ValueError('Matrix columns differ from task order')
        singles = [json.loads((output / 'single' / task / 'single_score.json').read_text())['rougeL'] for task in config['tasks']]
        scores = continual_metrics(matrix['rows'], singles)
        write_json(output / 'metrics.json', {'protocol': config['protocol'], 'method': config['method'], 'order': config['order'],
                   'seed': config['seed'], 'run_label': config['run_label'], 'smoke': config.get('smoke', False),
                   'config_sha256': fingerprint(config), 'model_fingerprint': identity['weights_fingerprint'],
                   'single_scores': dict(zip(config['tasks'], singles)), **scores})
        write_json(output / 'status.json', {'status': 'completed', 'smoke': config.get('smoke', False),
                                          'scope': 'one_order_one_seed_with_single_task_controls'})
        subprocess.run([sys.executable, str(ROOT / 'summary/collect.py')], cwd=ROOT, check=True)
    except BaseException as error:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        current = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
        if current.get('status') != 'blocked_model':
            write_json(output / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                              'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'exp/configs/olora_first.json'))
    parser.add_argument('--model', default=os.environ.get('LLAMA_MODEL_PATH'))
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--output')
    parser.add_argument('--prepare-only', action='store_true')
    launch(parser.parse_args())
