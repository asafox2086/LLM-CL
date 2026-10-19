"""CPU-only, read-only analysis of trusted Qwen S1/S2/S3 checkpoint graphs.

No training, inference, TEST predictions or answer access. Compare 4BA, not
factor-coordinate distances, and keep full-data/V2 input-budget confounds explicit.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'summary/two_backbone_pct01_20261017/partial_full_vs_pct01_20261019/parameter_geometry/v2'
WORK = ROOT/'exp/qwen_full_pct01_parameter_geometry_20261019/v2'
RUNS = {
    'full': 'exp/five_domain_native_cl_20261016_sparse/runs/full/SeqFT_LoRA/training_20261016_205816',
    'pct01': 'exp/two_backbone_native_cl_20261017_v2/runs/qwen/pct01/SeqFT_LoRA/training_20261017_161209',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def extract(condition, stage):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import pickle
    import resource
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    p = ROOT/RUNS[condition]
    sys.path.insert(0, str(p/'runner_source'))
    meta = json.loads((p/f'checkpoints/stage_{stage:02}.json').read_text())
    checkpoint = p/meta['path']
    actual_sha = sha(checkpoint)
    assert actual_sha == meta['sha256']
    assert checkpoint.stat().st_size == meta['bytes']
    config = json.loads((p/'config.json').read_text())
    method_source = p/'runner_source/methods.py'
    assert sha(method_source) == config['execution_source_sha256']['methods.py']
    start = time.monotonic()
    payload = torch.load(checkpoint, map_location='cpu', pickle_module=pickle, weights_only=False)
    assert payload['stage'] == stage and payload['config'] == config
    model = payload['method'].model
    trainable = {n: x for n, x in model.named_parameters() if x.requires_grad}
    assert len(trainable) == 112 and sum(x.numel() for x in trainable.values()) == 1089536
    frozen = hashlib.sha256()
    frozen_numel = 0
    for n, x in model.named_parameters():
        assert x.device.type == 'cpu'
        if not x.requires_grad:
            frozen.update(json.dumps([n, list(x.shape), str(x.dtype)]).encode())
            frozen.update(x.detach().contiguous().numpy().tobytes())
            frozen_numel += x.numel()
    state = {}
    for n, module in model.named_modules():
        if hasattr(module, 'A') and hasattr(module, 'B') and hasattr(module, 'scale'):
            assert module.scale == 4.0
            a = module.A.weight.detach().clone()
            b = module.B.weight.detach().clone()
            assert a.dtype == b.dtype == torch.float32 and a.shape[0] == 8
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            state[n] = {'A': a, 'B': b, 'scale': float(module.scale),
                        'base_norm2': float(module.weight.detach().double().square().sum())}
    assert len(state) == 56
    dest = WORK/f'{condition}_S{stage:02}.pt'
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dest)
    proof = {'condition': condition, 'stage': stage,
             'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': actual_sha,
             'checkpoint_bytes': meta['bytes'], 'method_source_sha256': sha(method_source),
             'frozen_parameter_sha256': frozen.hexdigest(), 'frozen_parameter_numel': frozen_numel,
             'trainable_numel': 1089536, 'trainable_tensors': 112, 'lora_modules': 56,
             'portable_state': str(dest.relative_to(ROOT)), 'portable_sha256': sha(dest),
             'seconds_load_and_extract': time.monotonic()-start,
             'cpu_process_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             'cuda_initialized': torch.cuda.is_initialized()}
    assert not proof['cuda_initialized']
    write(dest.with_suffix('.json'), proof)
    print(json.dumps(proof), flush=True)


def spectrum(matrix):
    """For reporting only: matrix rank <=16, use exact singular values."""
    import torch
    sv = torch.linalg.svdvals(matrix)
    energy = sv.square()
    total = float(energy.sum())
    if total == 0:
        return {'stable_rank': 0., 'rank90': 0}
    return {'stable_rank': total/float(energy[0]),
            'rank90': int(torch.searchsorted(energy.cumsum(0), .9*energy.sum()))+1}


def delta(state, previous, name):
    s = state[name]
    now = (s['B'].double() @ s['A'].double())*s['scale']
    if previous is None:
        return now
    p = previous[name]
    return now-(p['B'].double() @ p['A'].double())*p['scale']


def lowrank_spectrum(state, previous, name):
    import torch
    s = state[name]
    l, r = s['B'].double()*s['scale'], s['A'].double()
    if previous is not None:
        p = previous[name]
        l = torch.cat([l, -p['B'].double()*p['scale']], dim=1)
        r = torch.cat([r, p['A'].double()], dim=0)
    ql, rl = torch.linalg.qr(l, mode='reduced')
    qr, rr = torch.linalg.qr(r.T, mode='reduced')
    return spectrum(rl@rr.T)


def training_summary(condition, stage):
    p = ROOT/RUNS[condition]
    config = json.loads((p/'config.json').read_text())
    events = [json.loads(s) for s in (p/'logs/events.jsonl').read_text().splitlines()]
    e = [r for r in events if r.get('event') == 'train_update' and r['stage'] == stage]
    assert len(e) == math.ceil(config['task_counts'][f'T{stage:02}']['train']/16)
    if condition == 'full':
        used_lr = [0.]+[r['learning_rate'][0] for r in e[:-1]]
    else:
        used_lr = [r['learning_rate_used'][0] for r in e]
    assert all(math.isfinite(r['loss']) and math.isfinite(r['gradient_l2_preclip']) for r in e)
    return {'train_qa': config['task_counts'][f'T{stage:02}']['train'], 'updates': len(e),
            'positive_used_lr_updates': sum(v > 0 for v in used_lr),
            'sum_used_lr': sum(used_lr),
            'mean_loss_first_up_to5': sum(r['loss'] for r in e[:5])/len(e[:5]),
            'mean_loss_last_up_to5': sum(r['loss'] for r in e[-5:])/len(e[-5:]),
            'mean_raw_gradient_l2': sum(r['gradient_l2_preclip'] for r in e)/len(e),
            'gradient_clip_fraction': sum(r['gradient_l2_preclip'] > 1. for r in e)/len(e),
            'total_target_tokens': sum(r['target_tokens'] for r in e),
            'events_sha256': sha(p/'logs/events.jsonl')}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    for condition in RUNS:
        for stage in range(1, 4):
            subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()),
                            '--extract', condition, str(stage)], check=True, cwd=ROOT)
    import torch
    torch.set_num_threads(2)
    proofs = [json.loads((WORK/f'{c}_S{s:02}.json').read_text()) for c in RUNS for s in range(1, 4)]
    assert len({p['frozen_parameter_sha256'] for p in proofs}) == 1
    assert len({p['method_source_sha256'] for p in proofs}) == 1
    states = {(c, s): torch.load(WORK/f'{c}_S{s:02}.pt', map_location='cpu', weights_only=True)
              for c in RUNS for s in range(1, 4)}
    names = sorted(states['full', 1])
    assert all(sorted(v) == names for v in states.values())
    comparison = []
    for stage in range(1, 4):
        for mode in ['cumulative', 'stage_increment']:
            rows = []
            for name in names:
                fprev = states['full', stage-1] if mode == 'stage_increment' and stage > 1 else None
                pprev = states['pct01', stage-1] if mode == 'stage_increment' and stage > 1 else None
                f = delta(states['full', stage], fprev, name)
                p = delta(states['pct01', stage], pprev, name)
                fn, pn, dot = float(f.square().sum()), float(p.square().sum()), float((f*p).sum())
                layer = int(name.split('.')[2])
                rows.append({'module': name, 'layer': layer, 'projection': name.split('.')[-1],
                             'full_norm2': fn, 'pct01_norm2': pn, 'dot': dot,
                             'cosine': dot/math.sqrt(fn*pn) if fn*pn else None,
                             'norm_ratio': math.sqrt(fn/pn) if pn else None,
                             'base_norm2': states['full', stage][name]['base_norm2'],
                             'full_spectrum': lowrank_spectrum(states['full', stage], fprev, name),
                             'pct01_spectrum': lowrank_spectrum(states['pct01', stage], pprev, name)})
            def aggregate(rs):
                fn, pn, dot = (sum(r[k] for r in rs) for k in ['full_norm2', 'pct01_norm2', 'dot'])
                cosine = dot/math.sqrt(fn*pn) if fn*pn else None
                return {'full_frobenius': math.sqrt(fn), 'pct01_frobenius': math.sqrt(pn),
                        'norm_ratio': math.sqrt(fn/pn) if pn else None, 'cosine': cosine,
                        'full_energy_not_in_pct01_single_direction': 1-cosine*cosine if cosine is not None else None,
                        'full_relative_to_frozen_qv': math.sqrt(fn/sum(r['base_norm2'] for r in rs)),
                        'pct01_relative_to_frozen_qv': math.sqrt(pn/sum(r['base_norm2'] for r in rs))}
            item = {'stage': stage, 'mode': mode, 'global': aggregate(rows), 'modules': rows,
                    'q': aggregate([r for r in rows if r['projection'] == 'q_proj']),
                    'v': aggregate([r for r in rows if r['projection'] == 'v_proj'])}
            item['top_full_increment_layers'] = sorted(
                [{'layer': l, 'full_norm2': sum(r['full_norm2'] for r in rows if r['layer'] == l),
                  'pct01_norm2': sum(r['pct01_norm2'] for r in rows if r['layer'] == l)} for l in sorted({r['layer'] for r in rows})],
                key=lambda r: r['full_norm2'], reverse=True)[:5]
            comparison.append(item)
    evidence = {'scope': 'CPU checkpoint parameter geometry, zero model forward/backward/optimizer/TEST access',
                'source_sha256': sha(Path(__file__)), 'checkpoint_proofs': proofs,
                'all_six_frozen_parameters_identical': True, 'trainable_coordinates_each': 1089536,
                'lora_modules_each': 56, 'rank_each': 8, 'scale_each': 4.,
                'comparisons': comparison,
                'training': {c: {str(s): training_summary(c, s) for s in range(1, 4)} for c in RUNS},
                'limitations': ['Different training input/cache/scheduler/scaler across full legacy and valid V2 pct01.',
                    'Medical stages start from different previously learned functions.',
                    'Dataset size and optimizer opportunity co-vary under one epoch.',
                    '4BA is the inference LoRA linear operator; not measured activation change, knowledge or causal F1 contribution.',
                    'The orthogonal residual uses one entire pct01 update as a vector, not its parameter-space capacity.',
                    'No intervention or new performance experiment was performed.']}
    assert not torch.cuda.is_initialized()
    evidence['cuda_initialized'] = False
    write(OUT/'EVIDENCE.json', evidence)
    for r in comparison:
        print('RESULT', r['stage'], r['mode'], json.dumps(r['global']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', nargs=2)
    args = parser.parse_args()
    if args.extract:
        extract(args.extract[0], int(args.extract[1]))
    else:
        main()
