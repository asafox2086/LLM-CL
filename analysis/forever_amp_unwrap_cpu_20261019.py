"""Reproduce the actual AMP serialization guard and test the precise repair."""
import hashlib
import json
from pathlib import Path

from accelerate import Accelerator
import cloudpickle
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v2/AMP_UNWRAP_CPU.json'
    assert not destination.exists()
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    accelerator = Accelerator(cpu=True, mixed_precision='bf16')
    model = accelerator.prepare_model(model)
    assert not torch.cuda.is_initialized()
    try:
        cloudpickle.dumps(model)
    except Exception as error:
        assert 'Cannot pickle a prepared model with automatic mixed precision' in str(error)
        failure = repr(error)
    else:
        raise AssertionError('CPU positive control did not reproduce AMP guard')
    x = torch.tensor([[.25,.5,.75]])
    before = {name: value.detach().clone() for name,value in model.state_dict().items()}
    signature = [(name,id(p),p.data_ptr(),p._version) for name,p in model.named_parameters()]
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16):
        original = model(x).float()
    unwrapped = accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    assert unwrapped is model
    with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16):
        repaired = model(x).float()
    assert torch.equal(original,repaired)
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())
    assert signature == [(name,id(p),p.data_ptr(),p._version) for name,p in model.named_parameters()]
    assert torch.equal(rng,torch.random.get_rng_state())
    blob = cloudpickle.dumps(model)
    restored = cloudpickle.loads(blob)
    with torch.no_grad(), torch.autocast('cpu',dtype=torch.bfloat16):
        reloaded = restored(x).float()
    assert torch.equal(repaired,reloaded)
    assert not torch.cuda.is_initialized()
    proof = {'status':'EXACT_AMP_GUARD_REPRODUCED_AND_REPAIRED_CPU_ONLY',
             'original_failure':failure,'keep_fp32_wrapper':False,
             'parameter_values_identity_storage_versions_unchanged':True,
             'explicit_autocast_output_exact':True,'RNG_unchanged':True,
             'cloudpickle_reload_output_exact':True,'GPU':False,
             'not_model_performance':True,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
