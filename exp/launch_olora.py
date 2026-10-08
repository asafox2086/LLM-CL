import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT, continual_metrics, fingerprint, prepare_data, write_json, validate_model_config


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
                    allow_patterns=['*.json', 'tokenizer.model', 'spiece.model', '*.txt', weight_pattern],
                    local_dir=ROOT / 'models' / config['model_id'].split('/')[-1]))
        source = {'kind': 'huggingface_snapshot', 'model_id': config['model_id'], 'revision': info.sha}
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(f'Model config missing: {path}')
    validate_model_config(json.loads((path / 'config.json').read_text()), config)
    files = sorted(file for file in path.iterdir() if file.is_file() and file.suffix in {'.json', '.model', '.safetensors', '.bin', '.txt'}
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
    config_path = Path(args.resume).resolve() / 'config.json' if args.resume else Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config['method'] not in {'olora', 'migu_lora', 'sapt_lora', 'seq_lora'}:
        raise ValueError(f'Unsupported method: {config["method"]}')
    config['evaluation_protocol'] = 'cl_standard_fwt_v1'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(args.resume).resolve() if args.resume else (Path(args.output).resolve() if args.output else ROOT / 'exp/result' / config['protocol'] / config['run_label'] / stamp)
    if args.resume:
        if args.output or args.prepare_only:
            raise ValueError('--resume cannot be combined with --output or --prepare-only')
        previous_status = json.loads((output / 'status.json').read_text())
        worker_status = output / 'continual/status.json'
        worker = json.loads(worker_status.read_text()) if worker_status.exists() else {}
        candidates = [previous_status.get('pid'), worker.get('pid')]
        for process_id in candidates:
            process_path = Path(f'/proc/{process_id}/cmdline')
            if process_id and process_path.exists():
                command_line = process_path.read_bytes()
                if b'run_olora.py' in command_line or b'launch_olora.py' in command_line:
                    raise RuntimeError(f'Refusing concurrent resume while process {process_id} is alive')
        write_json(output / 'resume_events' / f'{stamp}.json', {
            'previous_status': previous_status, 'previous_worker_status': worker,
            'previous_code_identity': json.loads((output / 'code_identity.json').read_text()) if (output / 'code_identity.json').exists() else None,
            'command': sys.argv, 'resume_policy': 'reuse validated predictions and complete optimizer-step checkpoints'})
    else:
        output.mkdir(parents=True, exist_ok=False)
    run_lock = (output / '.run.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        run_lock.close()
        raise RuntimeError(f'Another launcher owns this run: {output}')
    processes = []
    handles = []
    try:
        write_json(output / 'status.json', {'status': 'preparing', 'phase': 'data', 'pid': os.getpid()})
        datasets, splits = prepare_data(config)
        if args.resume and json.loads((output / 'split_manifest.json').read_text()) != splits:
            raise ValueError('Resume data split or content fingerprints differ')
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
                       'action': 'Supply --model /path/to/configured-model or configure authorized Hugging Face access; rerun with a fresh output directory.'})
            raise
        if args.resume and (output / 'model_identity.json').exists():
            previous_identity = json.loads((output / 'model_identity.json').read_text())
            if previous_identity['weights_fingerprint'] != identity['weights_fingerprint']:
                raise ValueError('Resume base-model weights differ')
        write_json(output / 'model_identity.json', identity)
        code_files = [ROOT / 'code' / f'{config["method"]}.py', ROOT / 'exp/common.py', ROOT / 'exp/run_olora.py',
                      ROOT / 'exp/launch_olora.py', ROOT / 'exp/recovery.py', ROOT / 'summary/collect.py']
        code_identity = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files}
        if args.resume and (output / 'code_identity.json').exists():
            previous_code = json.loads((output / 'code_identity.json').read_text())
            changed = [name for name, digest in code_identity.items()
                       if previous_code.get(name) != digest and name != 'summary/collect.py']
            # Legacy baseline evaluations can adopt logging/recovery before any training.
            trained = any((output / 'continual/checkpoints').rglob('*.pt'))
            if changed and trained:
                raise ValueError(f'Resume implementation differs after training: {changed}')
            if changed:
                print(f'Baseline-only recovery implementation migration: {changed}', flush=True)
        write_json(output / 'code_identity.json', code_identity)
        gpus = [gpu.strip() for gpu in args.gpus.split(',') if gpu.strip()]
        if not gpus or len(set(gpus)) != len(gpus):
            raise ValueError('Supply a nonempty list of distinct GPU IDs')
        log_path = output / 'logs/continual.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open('a' if args.resume else 'w')
        handles.append(handle)
        command = [sys.executable, '-u', str(ROOT / 'exp/run_olora.py'), '--config', str(output / 'config.json'),
                   '--model', str(model_path), '--output', str(output / 'continual')]
        if args.resume:
            command.append('--resume')
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpus[0], 'TOKENIZERS_PARALLELISM': 'false',
                       'PYTHONUNBUFFERED': '1'}
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        processes.append(process)
        print(f'Started continual on GPU {gpus[0]}, pid={process.pid}', flush=True)
        write_json(output / 'status.json', {'status': 'running', 'pid': os.getpid(), 'jobs': 1,
                   'active': {gpus[0]: {'job': 'continual', 'pid': process.pid}}})
        code = process.wait()
        if code:
            raise RuntimeError(f'Continual worker failed: exit={code}; see {log_path}')
        matrix = json.loads((output / 'continual/score_matrix.json').read_text())
        if matrix['tasks'] != config['tasks']:
            raise ValueError('Matrix columns differ from task order')
        scores = continual_metrics(matrix['rows'])
        write_json(output / 'metrics.json', {'protocol': config['protocol'], 'method': config['method'], 'order': config['order'],
                   'evaluation_protocol': config['evaluation_protocol'],
                   'seed': config['seed'], 'run_label': config['run_label'], 'smoke': config.get('smoke', False),
                   'config_sha256': fingerprint(config), 'model_fingerprint': identity['weights_fingerprint'],
                   **scores})
        write_json(output / 'status.json', {'status': 'completed', 'smoke': config.get('smoke', False),
                                          'scope': 'one_order_one_seed_continual_only'})
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
        run_lock.close()


if __name__ == '__main__':
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'exp/configs/olora_t5large.json'))
    parser.add_argument('--model', default=os.environ.get('LLMCL_MODEL_PATH'))
    parser.add_argument('--gpus', default='0', help='Uses the first listed GPU for the single continual-learning stream')
    parser.add_argument('--output')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', metavar='RUN_DIRECTORY')
    launch(parser.parse_args())
