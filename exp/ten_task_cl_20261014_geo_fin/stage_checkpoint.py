"""Stage snapshot contract; each method must supply its full model/state exporter."""
import json
import os
import random
import shutil
import tempfile
from pathlib import Path


def save_stage(run_dir, stage, export_model, training_state, metadata, evaluation):
    """Publish a complete stage atomically. Never overwrite an earlier stage.

    export_model(path) must save ALL changed weights/configuration for the method.
    Required state values may be None only if the component genuinely does not apply.
    Training adapters must prove load/resume equivalence before formal experiments.
    """
    import numpy as np
    import torch

    if not 1 <= stage <= 10:
        raise ValueError('stage must be 1..10')
    required = {'optimizer', 'scheduler', 'scaler', 'method_state', 'global_step'}
    if required - training_state.keys():
        raise ValueError(f'missing training state: {sorted(required - training_state.keys())}')
    if len(evaluation.get('task_scores', [])) != 10:
        raise ValueError('a stage must record scores for all ten tasks')
    root = Path(run_dir) / 'checkpoints'
    root.mkdir(parents=True, exist_ok=True)
    target = root / f'stage_{stage:02d}'
    if target.exists():
        raise FileExistsError(target)
    temporary = Path(tempfile.mkdtemp(prefix=f'.stage_{stage:02d}_', dir=root))
    try:
        model_dir = temporary / 'model'
        model_dir.mkdir()
        export_model(model_dir)
        if not any(model_dir.iterdir()):
            raise ValueError('model exporter produced an empty snapshot')
        state = dict(training_state)
        state['rng'] = {'python': random.getstate(), 'numpy': np.random.get_state(),
                        'torch_cpu': torch.get_rng_state(),
                        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        torch.save(state, temporary / 'training_state.pt')
        for name, value in [('metadata.json', dict(metadata, stage=stage)),
                            ('evaluation.json', evaluation),
                            ('COMPLETE.json', {'stage': stage, 'complete': True})]:
            (temporary / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Flush snapshot files before publishing. Concurrent writers are not supported.
        for path in temporary.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
        os.rename(temporary, target)
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return target
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def load_stage(path, import_model, optimizer=None, scheduler=None, scaler=None,
               restore_method=None):
    """Load a trusted project snapshot, restoring resume state including RNG.

    The caller recreates the matching architecture and components first. Loading
    an optimizer is appropriate for resuming; a new stage may instead reset it.
    """
    import numpy as np
    import torch

    path = Path(path)
    complete = json.loads((path / 'COMPLETE.json').read_text(encoding='utf-8'))
    if complete.get('complete') is not True:
        raise ValueError('incomplete stage snapshot')
    state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if complete['stage'] != metadata['stage']:
        raise ValueError('stage metadata mismatch')
    import_model(path / 'model')
    for name, component in [('optimizer', optimizer), ('scheduler', scheduler), ('scaler', scaler)]:
        if component is not None:
            if state[name] is None:
                raise ValueError(f'snapshot lacks {name} state')
            component.load_state_dict(state[name])
    if restore_method is not None:
        restore_method(state['method_state'])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch_cpu'])
    if state['rng']['torch_cuda']:
        if not torch.cuda.is_available():
            raise ValueError('CUDA RNG state requires a compatible CUDA runtime')
        torch.cuda.set_rng_state_all(state['rng']['torch_cuda'])
    return metadata, state
