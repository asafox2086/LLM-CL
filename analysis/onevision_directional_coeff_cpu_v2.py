"""Real frozen-shape/basis/import checks before new GPU admission."""
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from safetensors import safe_open
import onevision_directional_coeff_core as core
import onevision_directional_coeff_trial_v2 as worker


def main():
    assert not torch.cuda.is_initialized()
    checks=core.checks();rows=[]
    with safe_open(core.ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors',framework='pt',device='cpu') as f:
        names=sorted(n for n in f.keys() if n.startswith('language_model.model.layers.') and n.endswith(('.q_proj.weight','.v_proj.weight')))
        assert len(names)==48
        for name in names:
            weight=f.get_tensor(name).to(torch.bfloat16)
            layer=core.DirectionalLinear(weight,None,name,core.VIEWS[0])
            rows.append(dict(name=name,shape=list(weight.shape),trainable_parameters=layer.coeff.numel(),
                             frozen_basis_parameters=layer.left.numel()+layer.right.numel(),payload=layer.memory_payload()))
            assert not torch.count_nonzero(layer.coeff)
    assert sum(r['trainable_parameters'] for r in rows)==12288
    assert sum(r['frozen_basis_parameters'] for r in rows)==1081344
    spec,values=worker.plan();_,data,_,tests,full,original,_=values
    assert sum(len(data[t,'train']) for t in range(1,20))==1313
    assert sum(map(len,full.values()))==21454 and sum(map(len,tests.values()))==500
    assert sum((len(data[t,'train'])+15)//16 for t in range(1,20))==93
    assert sum(map(len,spec['generation_ids'].values()))==155
    # Actual sealed-reference schema/ID compatibility, not another baseline
    # performance analysis or reading live partial predictions.
    for label in ['base_initial','sequential_T19']:
        assert worker.case_complete(original,label,tests)
        for domain in tests:
            reference=worker.semantic_rows(original,label,domain)
            assert [r['id'] for r in reference]==spec['diagnostic_ids'][domain]
            assert all(r['native_task'] is not None and r['nll'] is not None for r in reference)
    for gpu in ['0','1']:
        script=("import os,sys;os.environ['CUDA_VISIBLE_DEVICES']="+repr(gpu)+";sys.path.insert(0,"+repr(str(core.ROOT/'analysis'))+");import onevision_directional_coeff_trial_v2 as w;w.plan();assert os.environ['CUDA_VISIBLE_DEVICES']=="+repr(gpu)+";assert not w.torch.cuda.is_initialized()")
        subprocess.run([sys.executable,'-c',script],check=True)
    worker.write(worker.REPORT/'CPU_PLAN.json',dict(CPU_only=True,spec=spec))
    worker.write(worker.REPORT/'CPU_CHECK.json',dict(CPU_only=True,checks=8,synthetic=checks,actual48=rows,
        source_sha256={str(p.relative_to(core.ROOT)):worker.digest(p) for p in [Path(__file__),core.ROOT/'analysis/onevision_directional_coeff_core.py',core.ROOT/'analysis/onevision_directional_coeff_trial_v2.py',core.ROOT/'analysis/onevision_directional_coeff_entry_v2.py']},
        device_imports_preserve_0_and_1=True,all_frozen_ids_preserved=True,GPU_started=False,
        trainable_parameters=12288,original_LoRA_parameters=540672,parameter_reduction_factor=44,
        note='Fixed bases/Gram monitoring/optimizer/activation costs still require actual CPU-GPU measurements.'))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(CPU_only=True,checks=8,actual_weights=48,trainable_parameters=12288,frozen_basis_parameters=1081344,original_LoRA_parameters=540672,GPU_started=False)))


if __name__=='__main__':main()
