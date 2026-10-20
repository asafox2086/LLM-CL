"""Read-only final formal weight-space study; no QA, model forward or GPU."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REL = 'summary/edge_deployment_20261019/cpu_update_space_20261020/v1'
REPORT = ROOT / REL
PRIVATE = ROOT / 'exp/cpu_update_space_20261020/v1'
SEQ_SHA = '911d8937d74bb2525bcabd630e9fc28528e3aa1cc8413b293a49c90e3ad6b5a5'
OSFT_SHA = 'f93481daa528bdca4cd6bdb68ca6ec522698dfaf4703cb277cf5548b19aba541'
RANKS = [2, 4, 8, 16, 32, 64, 128, 256]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def td(x):
    return hashlib.sha256(x.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def gram_basis(x):
    u, s, _ = torch.linalg.svd(x, full_matrices=False)
    tol = max(x.shape)*torch.finfo(x.dtype).eps*s[0]
    return u[:, s > tol]


def tangent(delta, a, b):
    ub, va = gram_basis(b), gram_basis(a.T)
    p = ub@(ub.T@delta) + (delta@va)@va.T - ub@((ub.T@delta)@va)@va.T
    return p, ub.shape[1]*delta.shape[1]+va.shape[1]*delta.shape[0]-ub.shape[1]*va.shape[1]


def controls():
    d = torch.diag(torch.tensor([4., 3., 2., 1.], dtype=torch.float64))
    s = torch.linalg.svdvals(d)
    assert torch.equal(s, torch.tensor([4., 3., 2., 1.], dtype=torch.float64))
    assert float(s[:2].square().sum()/s.square().sum()) == 25/30
    g = torch.Generator().manual_seed(42)
    a = torch.randn(2, 7, dtype=torch.float64, generator=g)
    b = torch.randn(5, 2, dtype=torch.float64, generator=g)
    x = torch.randn(5, 7, dtype=torch.float64, generator=g)
    p, dim = tangent(x, a, b)
    assert dim == 2*(5+7-2)
    assert abs(float((p*(x-p)).sum())) < 1e-10
    assert torch.allclose(tangent(p, a, b)[0], p, atol=1e-12, rtol=1e-12)
    qb, rb = torch.linalg.qr(b, mode='reduced')
    qa, ra = torch.linalg.qr(a.T, mode='reduced')
    assert torch.allclose(torch.linalg.svdvals(b@a)[:2], torch.linalg.svdvals(rb@ra.T), atol=1e-12)
    return {'known_spectrum': True, 'orthogonal_tangent': True, 'independent_QR': True}


def extract():
    run = ROOT/'exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/SeqFT_LoRA/training_20261017_161154'
    cp = run/'checkpoints/stage_19.pt'
    assert sha(cp) == SEQ_SHA
    assert not (PRIVATE/'seqft_factors.pt').exists()
    sys.path.insert(0, str(run/'runner_source'))
    payload = torch.load(cp, map_location='cpu', pickle_module=pickle, weights_only=False)
    assert payload['stage'] == 19 and payload['config']['backbone'] == 'onevision'
    layers = {}
    for name, m in payload['method'].model.core.named_modules():
        if hasattr(m, 'A') and hasattr(m, 'B') and hasattr(m, 'scale'):
            assert name.startswith('language_model.model.layers.')
            layers[name] = {'A': m.A.weight.detach().cpu().clone(), 'B': m.B.weight.detach().cpu().clone(),
                            'scale': float(m.scale), 'base_sha256': td(m.weight)}
    assert len(layers) == 48
    count = sum(v['A'].numel()+v['B'].numel() for v in layers.values())
    assert count == 540672
    PRIVATE.mkdir(parents=True, exist_ok=True)
    path = PRIVATE/'seqft_factors.pt'
    torch.save({'layers': layers}, path)
    proof = {'source_sha256': sha(__file__), 'checkpoint_sha256': SEQ_SHA,
             'checkpoint_path': str(cp.relative_to(ROOT)), 'factors_sha256': sha(path),
             'trainable_coordinates': count, 'config': payload['config'],
             'CPU_only': True, 'QA_access': False, 'cuda_initialized': torch.cuda.is_initialized()}
    assert not proof['cuda_initialized']
    write(REPORT/'SEQ_EXTRACT.json', proof)
    print('SEQ_EXTRACT_COMPLETE', count, flush=True)


def spectrum(delta, base, name):
    delta, base = delta.double(), base.double()
    s = torch.linalg.svdvals(delta)
    e, se = float(delta.square().sum()), float(s.square().sum())
    assert e > 0 and math.isclose(e, se, rel_tol=1e-9, abs_tol=1e-12)
    c = s.square().cumsum(0)/se
    return {'name': name, 'shape': list(delta.shape), 'energy': e,
            'base_energy': float(base.square().sum()), 'relative_norm': math.sqrt(e/float(base.square().sum())),
            'singular_values': s.tolist(), 'rank_energy': {str(k): float(s[:k].square().sum()) for k in RANKS},
            'rank90': int(torch.searchsorted(c, .90))+1, 'rank95': int(torch.searchsorted(c, .95))+1,
            'rank99': int(torch.searchsorted(c, .99))+1,
            'participation_rank': se*se/float(s.pow(4).sum()), 'delta_tensor_sha256': td(delta)}


def analyze():
    from safetensors import safe_open
    assert not (REPORT/'COMPLETE.json').exists() and not (REPORT/'STARTED.json').exists()
    ready = json.loads((REPORT/'READY.json').read_text())
    assert ready['source_sha256'] == sha(__file__)
    assert ready['protocol_sha256'] == sha(REPORT/'PROTOCOL.md')
    started = time.monotonic()
    write(REPORT/'STARTED.json', {'pid': os.getpid(), 'source_sha256': sha(__file__), 'CPU_only': True})
    base_path = ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors'
    teacher = ROOT/'exp/edge_teacher_transfer_20261019/onevision_osft_final'
    cp = teacher/'checkpoints/stage_19.pt'
    assert sha(cp) == OSFT_SHA
    ep = json.loads((REPORT/'SEQ_EXTRACT.json').read_text())
    fp = PRIVATE/'seqft_factors.pt'
    assert ep['source_sha256'] == sha(__file__) and ep['checkpoint_sha256'] == SEQ_SHA and sha(fp) == ep['factors_sha256']
    seq = torch.load(fp, map_location='cpu', weights_only=True)['layers']
    sys.path.insert(0, str(teacher/'runner_source'))
    payload = torch.load(cp, map_location='cpu', pickle_module=pickle, weights_only=False)
    assert payload['stage'] == 19 and payload['config']['method'] == 'OSFT'
    a, b = payload['config'], ep['config']
    for key in ['manifest_sha256', 'seed', 'epochs', 'batch_size', 'micro_batch_size', 'model_dtype', 'autocast_dtype', 'image_area_cap']:
        assert a[key] == b[key], key
    model = payload['method'].model.core
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 113766144
    source_stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in [base_path, cp, fp]}
    sources = {'base_sha256': sha(base_path), 'OSFT_checkpoint_sha256': OSFT_SHA,
               'SeqFT_checkpoint_sha256': SEQ_SHA, 'SeqFT_factors_sha256': ep['factors_sha256'],
               'OSFT_config_sha256': sha(teacher/'config.json')}
    osft_rows, seq_rows, checks = [], [], []
    local_dim, local_coords = 0, 0
    with safe_open(base_path, framework='pt', device='cpu') as reader, torch.inference_mode():
        for name, m in model.named_modules():
            if type(m).__name__ != 'OSFTLinear': continue
            original = reader.get_tensor(name+'.weight')
            actual = m.kernel['reconstruct_weight_matrix'](m.svd(), torch.float32, m.output_dtype).detach()
            assert actual.dtype == original.dtype == torch.float16
            delta = actual.double()-original.double()
            row = spectrum(delta, original, name)
            k = m.U_high.shape[1]
            row['protected_local_dimension'] = (actual.shape[0]-k)*(actual.shape[1]-k)
            row['trainable_coordinates'] = sum(p.numel() for p in m.parameters() if p.requires_grad)
            local_dim += row['protected_local_dimension']; local_coords += row['trainable_coordinates']
            if name in seq:
                z = seq[name]
                assert td(original) == z['base_sha256']
                aa, bb = z['A'].double(), z['B'].double()
                sd = z['scale']*(bb@aa)
                sr = spectrum(sd, original, name)
                _, rb = torch.linalg.qr(bb, mode='reduced'); _, ra = torch.linalg.qr(aa.T, mode='reduced')
                small = torch.linalg.svdvals(z['scale']*(rb@ra.T))
                assert torch.allclose(torch.tensor(sr['singular_values'][:8], dtype=torch.float64), small, rtol=1e-8, atol=1e-10)
                sr['QR_spectrum_verified'] = True
                p, dim = tangent(delta, aa, bb)
                pe = float(p.square().sum())
                assert pe <= row['energy']*(1+1e-9)
                row['SeqFT_tangent_retained_energy'] = pe
                row['SeqFT_tangent_dimension'] = dim
                row['isotropic_tangent_fraction'] = dim/delta.numel()
                row['SeqFT_cosine'] = float((delta*sd).sum())/math.sqrt(row['energy']*sr['energy'])
                seq_rows.append(sr)
            if name in ['language_model.model.layers.0.self_attn.q_proj', 'language_model.model.layers.23.mlp.down_proj']:
                u, s, vh = torch.linalg.svd(delta, full_matrices=False)
                approx = (u[:, :8]*s[:8])@vh[:8]
                measured = float((delta-approx).square().sum())
                predicted = row['energy']-row['rank_energy']['8']
                assert math.isclose(measured, predicted, rel_tol=1e-9, abs_tol=1e-10)
                checks.append({'name': name, 'rank8_residual_direct': measured, 'rank8_residual_spectrum': predicted})
            osft_rows.append(row)
            write(REPORT/'PROGRESS.json', {'CPU_only': True, 'matrices_done': len(osft_rows), 'expected': 168, 'seconds': time.monotonic()-started})
            print('MATRIX', len(osft_rows), name, 'rank95', row['rank95'], flush=True)
    assert len(osft_rows) == 168 and len(seq_rows) == 48 and len(checks) == 2
    assert local_coords == count
    def aggregate(rows):
        e = sum(x['energy'] for x in rows)
        return {'matrices': len(rows), 'energy': e,
                'relative_norm': math.sqrt(e/sum(x['base_energy'] for x in rows)),
                'rank_retention': {str(k): sum(x['rank_energy'][str(k)] for x in rows)/e for k in RANKS}}
    groups = {}
    for suffix in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
        groups[suffix] = aggregate([x for x in osft_rows if x['name'].endswith(suffix)])
    total = aggregate(osft_rows)
    for g in groups.values(): g['total_energy_fraction'] = g['energy']/total['energy']
    qv = [x for x in osft_rows if x['name'] in seq]
    qe = sum(x['energy'] for x in qv)
    projection = {'retained_energy_fraction': sum(x['SeqFT_tangent_retained_energy'] for x in qv)/qe,
                  'isotropic_fraction': sum(x['energy']*x['isotropic_tangent_fraction'] for x in qv)/qe,
                  'local_dimension': sum(x['SeqFT_tangent_dimension'] for x in qv),
                  'note': 'Final LoRA tangent is not optimizer history, optimal direction or causal F1 evidence.'}
    result = {'CPU_only': True, 'QA_access': False, 'model_forward': False, 'new_optimizer': False,
              'source_sha256': sha(__file__), 'sources': sources, 'controls': controls(),
              'OSFT_trainable_coordinates': count, 'OSFT_final_protected_local_dimension': local_dim,
              'SeqFT_trainable_coordinates': ep['trainable_coordinates'], 'OSFT': total,
              'SeqFT_algebraic_BA': aggregate(seq_rows), 'OSFT_by_projection': groups,
              'OSFT_qv_to_SeqFT_final_tangent': projection, 'independent_residual_checks': checks,
              'OSFT_layers': osft_rows, 'SeqFT_layers': seq_rows,
              'seconds': time.monotonic()-started, 'CPU_max_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'threads': torch.get_num_threads(), 'cuda_initialized': torch.cuda.is_initialized(),
              'limits': ['Descriptive final weights only; no F1, knowledge, causality or deployment saving proved.',
                         'OSFT actual FP16 reconstruction minus base includes training and decomposition/rounding history.',
                         'SeqFT scale BA is algebraic, not a bit-exact dense merge of its original forward.',
                         'Learning rates, parameterizations, projections and locations differ.',
                         'Small relative norm or energy does not imply a module is unnecessary.',
                         'Best rank approximation is an oracle geometric bound, not a trained small adapter or selected rank.',
                         'Final local tangent dimension does not equal 19-stage trajectory reachability.']}
    assert not result['cuda_initialized']
    for p, stat in source_stats.items():
        assert stat == [Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
    write(REPORT/'EVIDENCE.json', result)
    write(REPORT/'COMPLETE.json', {'status': 'CPU_FINAL_WEIGHT_SPACE_COMPLETE', 'source_sha256': sha(__file__),
          'evidence_sha256': sha(REPORT/'EVIDENCE.json'), 'matrices': [168, 48], 'CPU_only': True, 'QA_access': False})
    print('CPU_FINAL_WEIGHT_SPACE_COMPLETE', json.dumps(total), flush=True)


if __name__ == '__main__':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['controls', 'extract', 'analyze'], required=True)
    args = parser.parse_args()
    if args.mode == 'controls': print(json.dumps(controls()))
    else:
        try:
            extract() if args.mode == 'extract' else analyze()
        except Exception:
            write(REPORT/(args.mode+'_FAILURE.json'), {'traceback': traceback.format_exc(), 'source_sha256': sha(__file__), 'no_blind_retry': True})
            raise
