"""Same-seed repeated gradient/update audit after the stage1 reproduction failure.

No tolerance is loosened and no failed isolated result is admitted. Only the
first32 frozen TRAIN QA are used; all diagnostic updates are discarded.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import contextlib
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from onevision_knowledge_intervention import Intervention, reset_seed
from onevision_research_worker import atomic_torch, digest
from run import freeze, hashed, write


def flat_grad(named):
    return torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None
                      else torch.zeros(p.numel()) for _, p in named])


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    torch.set_num_threads(6)
    reset_seed()
    manifest, data, domains, tests, generation, probes, protocol = freeze()
    report = ROOT / 'summary/onevision_mechanisms_20261016'
    sources = [Path(__file__), ROOT / 'analysis/onevision_knowledge_intervention.py',
               ROOT / 'analysis/onevision_research_worker.py', ROOT / 'exp/onevision_diagnostics_20261016/run.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    pointer = report / 'reproducibility_attempt.json'
    if pointer.exists():
        info = json.loads(pointer.read_text()); assert info['source_sha256'] == hashes
        run = ROOT / info['run']
    else:
        run = ROOT / 'exp/onevision_mechanisms_20261016/reproducibility' / time.strftime('%Y%m%d_%H%M%S')
        run.mkdir(parents=True)
        write(pointer, {'run': str(run.relative_to(ROOT)), 'source_sha256': hashes})
    rows = sorted(data[1, 'train'], key=lambda r: hashed(r, 'training'))[:32]
    modes = ['default_sdpa', 'deterministic_default_sdpa', 'deterministic_math_sdpa']
    write(run / 'protocol.json', {'source_sha256': hashes, 'seed': 42, 'modes': modes,
          'train_ids': [r['id'] for r in rows], 'repetitions': 2, 'updates_per_repeat': 2,
          'target_test_gold_accessed': False, 'diagnostic_updates_discarded': True,
          'scheduler': 'same first two updates of original347QA stage22-update cosine schedule',
          'scope': 'Reproducibility audit only; numerical backend changes are NOT a new benchmark method.'})
    x = Intervention(manifest, run); initial = x.original_initial
    write(run / 'runtime.json', {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
          'gpu': torch.cuda.get_device_name(0), 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'],
          'allow_tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
          'allow_tf32_cudnn': torch.backends.cudnn.allow_tf32,
          'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
          'caveat': 'Workspace config applies to all three modes; default mode is not an exact replay of the old process environment.'})
    results = {}
    for mode in modes:
        proof = run / (mode + '.json')
        if proof.exists():
            results[mode] = json.loads(proof.read_text()); continue
        records = []; snapshots = []; first_gradients = []
        for repeat in range(2):
            optimizer = None
            try:
                write(report / 'status.json', {'phase': 'reproducibility_probe', 'mode': mode,
                      'repeat': repeat, 'run': str(run.relative_to(ROOT)), 'pid': os.getpid()})
                x.configure(initial, finetune_shared=True)
                x.cache.clear()
                x.model.zero_grad(set_to_none=True)
                reset_seed()
                torch.use_deterministic_algorithms(mode != 'default_sdpa')
                x.model.train()
                named = [(k,p) for k,p in x.model.named_parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW([p for _,p in named], lr=1e-4, weight_decay=0)
                context = sdpa_kernel(SDPBackend.MATH) if mode == 'deterministic_math_sdpa' else contextlib.nullcontext()
                x.monitor.snapshot(f'{mode}_{repeat}_before', model=x.model)
                steps = []
                with context:
                    for step in range(2):
                        optimizer.zero_grad(set_to_none=True); losses = []
                        for row in rows[step*16:(step+1)*16]:
                            loss, _ = x.loss(row); assert torch.isfinite(loss)
                            (loss/16).backward(); losses.append(float(loss.detach()))
                        gradient = flat_grad(named); assert torch.isfinite(gradient).all()
                        if step == 0:
                            first_gradients.append(gradient)
                            atomic_torch(run / f'{mode}_repeat{repeat}_first_gradient.pt',
                                {'gradient': gradient, 'parameter_layout': [(k,list(p.shape)) for k,p in named],
                                 'train_ids': [r['id'] for r in rows[:16]], 'preclip': True})
                        norm = torch.nn.utils.clip_grad_norm_([p for _,p in named], 1.)
                        assert torch.isfinite(norm)
                        lr = 1e-4 * .5 * (1 + math.cos(math.pi*step/22))
                        for group in optimizer.param_groups: group['lr'] = lr
                        optimizer.step()
                        steps.append({'step': step, 'mean_loss': sum(losses)/16, 'preclip_norm': float(norm), 'lr': lr})
                snapshot = x.snapshot(); snapshots.append(snapshot)
                atomic_torch(run / f'{mode}_repeat{repeat}_adapter.pt', {'adapter': snapshot, 'source_sha256': hashes})
                records.append({'repeat': repeat, 'finite_forward_backward_optimizer': True, 'steps': steps})
            except (RuntimeError, AssertionError) as error:
                records.append({'repeat': repeat, 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()})
                break
            finally:
                if optimizer is not None: optimizer.zero_grad(set_to_none=True)
                del optimizer
                x.model.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                x.monitor.snapshot(f'{mode}_{repeat}_after', model=x.model)
        result = {'mode': mode, 'repeats': records, 'successful_repeats': len(snapshots)}
        if len(snapshots) == 2:
            result['first_gradient_max_abs_difference'] = float((first_gradients[0]-first_gradients[1]).abs().max())
            result['first_gradient_relative_l2_difference'] = float((first_gradients[0]-first_gradients[1]).double().norm()/first_gradients[0].double().norm().clamp_min(1e-30))
            result['adapter_max_abs_difference'] = max(float((snapshots[0][k][b]-snapshots[1][k][b]).abs().max()) for k in initial for b in ['A','B'])
            result['adapter_exact_equal'] = all(torch.equal(snapshots[0][k][b],snapshots[1][k][b]) for k in initial for b in ['A','B'])
        write(proof, result); results[mode] = result
    torch.use_deterministic_algorithms(False)
    x.monitor.snapshot('training_after', model=x.model); x.monitor.finish(x.model)
    write(report / 'reproducibility_complete.json', {'complete': True, 'run': str(run.relative_to(ROOT)),
          'source_sha256': hashes, 'results': results, 'scope': 'Audit executed; errors/inequality are results, not a passed reproduction gate.',
          'matched_native_still_requires_resolution': True, 'formal_full_test': False})


if __name__ == '__main__':
    main()
