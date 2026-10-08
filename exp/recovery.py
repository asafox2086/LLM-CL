import io
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from common import ROOT, digest_bytes, fingerprint


def atomic_save(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.', suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def method_bytes(method):
    buffer = io.BytesIO()
    method.save(buffer)
    return buffer.getvalue()


def save_method(method, destination):
    state = torch.load(io.BytesIO(method_bytes(method)), map_location='cpu', weights_only=True)
    atomic_save(state, destination)


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def method_identity(method, config):
    return {'kind': type(method).__name__, 'config_sha256': fingerprint(config),
            'method_sha256': digest_bytes((ROOT / 'code' / f'{config["method"]}.py').read_bytes())}


def save_training(destination, method, config, optimizer, scheduler, scaler, **cursor):
    atomic_save({'format': 'llmcl_training_v1', 'identity': method_identity(method, config),
                 'method': method_bytes(method), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                 'rng': capture_rng(), **cursor}, destination)


def load_training(destination, method, config, optimizer, scheduler, scaler):
    state = torch.load(destination, map_location='cpu', weights_only=False)
    if state['format'] != 'llmcl_training_v1' or state['identity'] != method_identity(method, config):
        raise ValueError('Recovery checkpoint configuration or method implementation differs')
    method.load(io.BytesIO(state['method']))
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    restore_rng(state['rng'])
    return state
