"""Load pinned SparseGPT mathematics with an optional support-mask observer.

The upstream source and license remain private assets, outside this repository.
CPU synchronization is an explicit no-op shim for the tiny CPU admission only;
real GPU execution uses the real torch namespace and original synchronization.
No quantizer is installed or enabled.
"""
import ast
import copy
import hashlib
import math
import time
from pathlib import Path
from types import SimpleNamespace


class CPUOnlyTorch:
    def __init__(self, module):
        self.module = module
        self.cuda = SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None)

    def __getattr__(self, key):
        return getattr(self.module, key)


def load_pruner(source_path, expected_sha256, *, observe_mask, cpu_admission=False):
    import torch
    import transformers
    from torch import nn
    path = Path(source_path)
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected_sha256
    tree = ast.parse(raw.decode('utf8'))
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseGPT')
    node = copy.deepcopy(original)
    injected = []
    if observe_mask:
        fn = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'fasterprune')
        conversion = [i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
            and ast.unparse(n) == 'W = W.float()']
        assert len(conversion) == 1
        init = ast.parse('self._observed_prune_mask = torch.zeros_like(W, dtype=torch.bool)').body[0]
        fn.body.insert(conversion[0] + 1, init)
        injected.append(init)
        outer = [n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'i1']
        assert len(outer) == 1
        write = [i for i, n in enumerate(outer[0].body) if isinstance(n, ast.Assign)
            and ast.unparse(n) == 'W[:, i1:i2] = Q1']
        assert len(write) == 1
        record = ast.parse('self._observed_prune_mask[:, i1:i2] = mask1').body[0]
        outer[0].body.insert(write[0], record)
        injected.append(record)
        # Only observer statements may differ from the original mathematics.
        stripped = copy.deepcopy(node)
        fn2 = next(n for n in stripped.body if isinstance(n, ast.FunctionDef) and n.name == 'fasterprune')
        fn2.body = [n for n in fn2.body if '_observed_prune_mask' not in ast.unparse(n)
                    or isinstance(n, ast.For)]
        outer2 = next(n for n in fn2.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'i1')
        outer2.body = [n for n in outer2.body if '_observed_prune_mask' not in ast.unparse(n)]
        assert ast.dump(stripped, include_attributes=False) == ast.dump(original, include_attributes=False)
    env = dict(torch=CPUOnlyTorch(torch) if cpu_admission else torch,
               nn=nn, transformers=transformers, math=math, time=time, DEBUG=False)
    code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec')
    exec(code, env)
    klass = env['SparseGPT']
    klass._llmcl_source_sha256 = expected_sha256
    klass._llmcl_mask_observer = observe_mask
    klass._llmcl_CPU_sync_shim = cpu_admission
    return klass


def prune_fixed(solver):
    assert not hasattr(solver, 'quantizer')
    assert solver._llmcl_mask_observer
    assert solver.layer.weight.shape[1] % 4 == 0
    assert solver.nsamples == 16, 'Fixed current-S1 calibration only'
    solver.fasterprune(sparsity=.5, prunen=2, prunem=4, blocksize=128, percdamp=.01)
    keep = ~solver._observed_prune_mask
    assert bool((keep.reshape(keep.shape[0], -1, 4).sum(-1) == 2).all())
    assert bool((solver.layer.weight.detach().masked_select(~keep) == 0).all())
    return keep
