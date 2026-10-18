"""CPU-only TRAIN parameter audit of sealed initialization-source controls.

Never changes experiments, reads TEST answers, or selects a method/threshold.
Use after a new complete proof appears; no polling daemon or GPU model load.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def basis(a, b):
    """Row-space comparison invariant to invertible factor-coordinate changes."""
    a, b = a.double(), b.double()
    assert a.shape == b.shape and a.ndim == 2
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    rank = a.shape[0]
    assert torch.linalg.matrix_rank(a) == torch.linalg.matrix_rank(b) == rank
    qa = torch.linalg.qr(a.T, mode='reduced').Q
    qb = torch.linalg.qr(b.T, mode='reduced').Q
    principal_cosines = torch.linalg.svdvals(qa.T @ qb).clamp(0, 1)
    return dict(relative_A_displacement=float((b-a).norm()/a.norm()),
                normalized_projector_distance=float(1-principal_cosines.square().mean()),
                principal_cosines=principal_cosines.tolist())


def product_norm(a, b):
    a, b = a.double(), b.double()
    assert b.shape[1] == a.shape[0]
    # ||BA||_F without materializing a dense dout x din update.
    squared = ((b.T @ b) * (a @ a.T).T).sum()
    assert float(squared) >= -1e-10
    return math.sqrt(max(0., float(squared)))


def parameter_change(start, end, scale):
    layers = []
    assert start.keys() == end.keys()
    for name in start:
        x, y = start[name], end[name]
        geom = basis(x['A'], y['A'])
        # B1 A1 - B0 A0 = (B1-B0) A0 + B1 (A1-A0).
        # Avoid subtracting two large almost equal product norms.
        delta_a = torch.cat([x['A'].double(), y['A'].double()-x['A'].double()], dim=0)
        delta_b = torch.cat([y['B'].double()-x['B'].double(), y['B'].double()], dim=1)
        layers.append(dict(layer=name, **geom,
            A_displacement_l2=float((y['A'].double()-x['A'].double()).norm()),
            B_displacement_l2=float((y['B'].double()-x['B'].double()).norm()),
            effective_deltaW_displacement_l2=scale*product_norm(delta_a, delta_b)))
    return dict(layers=layers,
        mean_normalized_projector_distance=sum(v['normalized_projector_distance'] for v in layers)/len(layers),
        effective_deltaW_displacement_l2=math.sqrt(sum(v['effective_deltaW_displacement_l2']**2 for v in layers)))


def gradient_values(obj):
    vector = obj['preclip_gradient'].double();offset = 0
    sums = {'A': 0., 'B': 0.};nonzero = {'A': 0, 'B': 0};counts = {'A': 0, 'B': 0}
    assert torch.isfinite(vector).all()
    for block in obj['layout']:
        name, size = block['name'], block['count']
        part = vector[offset:offset+size];offset += size
        assert part.numel() == size == math.prod(block['shape'])
        factor = 'A' if name.endswith('.A.weight') else 'B' if name.endswith('.B.weight') else None
        assert factor is not None, name
        sums[factor] += float(part.square().sum());counts[factor] += 1
        nonzero[factor] += int(bool(torch.count_nonzero(part)))
    assert offset == vector.numel() and counts == {'A': 48, 'B': 48}
    return dict(TRAIN_ids=obj['train_ids'],
        preclip_A_gradient_l2=math.sqrt(sums['A']),preclip_B_gradient_l2=math.sqrt(sums['B']),
        nonzero_gradient_layers=nonzero,
        zero_vector_does_not_distinguish_missing_gradient_from_exact_zero=True)


def gradient_profile(path):
    return gradient_values(load(path))


def accepted(name, spec):
    pointer = REPORT/(name+'_complete.json')
    if not pointer.exists():
        return None
    proof = json.loads(pointer.read_text());assert proof['complete'] and proof['spec'] == spec
    run = ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_initialization_source_20261018/v1').resolve())
    for rel, expected in proof['artifact_sha256'].items():
        p = run/rel;assert p.resolve().is_relative_to(run.resolve()) and sha(p) == expected
    return run


def check():
    # A coordinate change changes factors but must not imply a changed row space.
    a = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    mixed = torch.tensor([[2., 1.], [0., .5]], dtype=torch.float64) @ a
    assert basis(a, mixed)['normalized_projector_distance'] < 1e-12
    changed = torch.tensor([[1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    assert abs(basis(a, changed)['normalized_projector_distance']-.5) < 1e-12
    b = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float64)
    assert abs(product_norm(a, b)-float((b@a).norm())) < 1e-12
    z = torch.zeros_like(b)
    assert product_norm(a, z) == 0
    x = {'x': {'A': a, 'B': b}};y = {'x': {'A': changed, 'B': b+1}}
    actual = parameter_change(x, y, 4)['effective_deltaW_displacement_l2']
    assert abs(actual-4*float(((b+1)@changed-b@a).norm())) < 1e-12
    layout = [{'name': f'layer{i}.{factor}.weight', 'shape': [2, 2], 'count': 4}
              for i in range(48) for factor in ['A', 'B']]
    raw = torch.cat([torch.zeros(4) if item['name'].endswith('.A.weight') else torch.ones(4) for item in layout])
    profile = gradient_values(dict(layout=layout, preclip_gradient=raw, train_ids=['synthetic_TRAIN']))
    assert profile['nonzero_gradient_layers'] == {'A': 0, 'B': 48}
    assert profile['preclip_A_gradient_l2'] == 0
    assert abs(profile['preclip_B_gradient_l2']-math.sqrt(192)) < 1e-12
    print('PASS: factor gauge, changed-subspace, dense product, zero product, deltaW, actual-naming gradient layout controls')


def main():
    torch.set_num_threads(1)
    spec = json.loads((REPORT/'CPU_PLAN_CHECK.json').read_text())['spec']
    for rel, expected in spec['source_sha256'].items():
        assert sha(ROOT/rel) == expected
    proofs = {p.name: sha(p) for p in REPORT.glob('*_complete.json')}
    output = REPORT/'TRAIN_PARAMETER_GEOMETRY.json'
    if output.exists():
        previous = json.loads(output.read_text())
        if previous['proof_sha256'] == proofs and previous['analyzer_sha256'] == sha(Path(__file__)):
            return
    source_run = accepted('sources', spec)
    sources, targets = {}, {}
    scale = spec['alpha']/spec['rank']
    if source_run:
        initial = load(source_run/'original_initial.pt')['adapter']
        assert len(initial) == 48
        for name in ['within', 'cross']:
            sources[name] = parameter_change(initial, load(source_run/(name+'.pt'))['adapter'], scale)
    for name in spec['cases']:
        run = accepted(name, spec)
        if run is None:
            continue
        start = load(run/'initial_adapter.pt')['adapter']
        end = load(run/'adapters/target_T04.pt')['adapter']
        targets[name] = dict(parameter_change=parameter_change(start, end, scale),
            raw_preclip_gradients=[gradient_profile(run/f'raw_update_gradients/target_T04/step{step}.pt') for step in [1, 2]],
            run=str(run.relative_to(ROOT)))
    result = dict(CPU_only=True,TRAIN_parameter_only=True,TEST_predictions_or_answers_parsed=False,
        complete_artifact_hashing_may_read_bound_TEST_file_bytes=True,
        analyzer_sha256=sha(Path(__file__)),proof_sha256=proofs,sources=sources,targets=targets,
        missing_conditions=[name for name in spec['cases'] if name not in targets],
        limits=['No semantic redundancy or F1 causality inferred from parameter geometry.',
                'Tiny source changes can be weak interventions; report amplitudes, not only nonzero flags.',
                'Factor coordinates are not identifiable; row-space and effective deltaW reported separately.',
                'No route, capacity, optimizer, threshold, or experiment is selected by this audit.'])
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(accepted_source=bool(source_run),accepted_targets=list(targets),pending=result['missing_conditions'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--check', action='store_true')
    if parser.parse_args().check:
        check()
    else:
        main()
