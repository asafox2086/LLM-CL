"""CPU preflight against sealed source factors; no backbone or QA inference."""
import hashlib
import json
from pathlib import Path
import torch
from onevision_function_neutral_init_core import checks, neutral_increment


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1'
REPORT = ROOT/'summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    synthetic = checks()
    proof_file = SOURCE/'sources_complete.json'
    proof = json.loads(proof_file.read_text())
    assert proof['complete']
    run = ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_initialization_source_20261018/v1').resolve())
    for name, expected in proof['artifact_sha256'].items():
        p = run/name
        assert p.resolve().is_relative_to(run.resolve()) and sha(p) == expected
    initial = torch.load(run/'original_initial.pt', map_location='cpu', weights_only=True)['adapter']
    assert len(initial) == 48
    profiles = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        for name in ['within', 'cross']:
            source = torch.load(run/(name+'.pt'), map_location='cpu', weights_only=True)['adapter']
            assert source.keys() == initial.keys()
            values = []
            for key, pair in source.items():
                a = pair['A'].clone().requires_grad_()
                b = pair['B'].clone().requires_grad_()
                reference_a, reference_b = a.detach().clone(), b.detach().clone()
                assert a.dtype == b.dtype == torch.float32
                assert torch.count_nonzero(b) > 0
                x = torch.randn(5, a.shape[1], requires_grad=True)
                # One shared input already after dropout: this test reads no QA.
                increment = neutral_increment(x, a, b, reference_a, reference_b)
                assert torch.count_nonzero(increment) == 0
                upstream = torch.randn_like(increment)
                ga, gb, gx = torch.autograd.grad((increment*upstream).sum(), (a, b, x))
                assert torch.isfinite(ga).all() and torch.isfinite(gb).all()
                assert ga.norm() > 0 and gb.norm() > 0
                assert float(gx.abs().max()) < 1e-6
                values.append(dict(layer=key, A_shape=list(a.shape), B_shape=list(b.shape),
                    A_gradient_l2=float(ga.norm()), B_gradient_l2=float(gb.norm()),
                    input_gradient_abs_max=float(gx.abs().max()), initial_increment_exact_zero=True))
            reference_parameters = sum(p.numel() for pair in source.values() for p in pair.values())
            assert reference_parameters == 540672
            profiles[name] = dict(checkpoint_sha256=sha(run/(name+'.pt')), layers=values,
                frozen_reference_parameters=reference_parameters,
                frozen_reference_FP32_bytes=reference_parameters*4)
    result = dict(CPU_only=True, synthetic=synthetic, actual_sealed_sources=profiles,
        sources_complete_sha256=sha(proof_file),
        source_sha256={f'analysis/{p.name}':sha(p) for p in [Path(__file__),
            ROOT/'analysis/onevision_function_neutral_init_core.py']},
        QA_predictions_or_answers_parsed=False, model_loaded=False, GPU_started=False,
        limits=['Random CPU inputs and upstream derivatives are not actual model loss gradients.',
                'Actual BF16 initial32 equality and two-update cold/full507 equality remain mandatory.',
                'This CPU preflight alone is not GPU queue admission or actual model performance.'])
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'CPU_PREFLIGHT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(CPU_only=True, synthetic_checks=8,
        actual_source_layers={k:len(v['layers']) for k,v in profiles.items()},
        exact_zero_initial_increment=True, GPU_started=False)))


if __name__ == '__main__':
    main()
