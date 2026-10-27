"""A separate process must reload each method's actual 2B preflight checkpoint."""
import argparse
import math
from pathlib import Path
import torch
from checkpoints import load
from runtime_utils import input_batch,answer_loss,write,activation_storage
from frozen_vision import prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    obj=load(args.run_dir);method=obj['method'];processor=obj['processor'];probe=obj['probe']
    import onevision_bridge
    onevision_bridge.bind(method.model)
    row=probe['row'];task=probe['task'];method.model.eval();method.set_instruction(task['task'])
    batch,_=input_batch(row,task,processor,method.model.device)
    prepare(batch,row,method.model,method.vision_original)
    if hasattr(method,'prepare_row'):method.prepare_row(row,batch)
    with torch.no_grad(),torch.autocast('cuda',dtype=method.model.dtype):
        loss,_=answer_loss(method.model,batch,row['answer'],processor)
    value=float(loss);error=abs(value-probe['loss'])
    assert math.isfinite(value) and error<max(.005,abs(probe['loss'])*.002), (value,probe['loss'])
    method.model.train();method.model.zero_grad(set_to_none=True)
    with activation_storage(method),torch.autocast('cuda',dtype=method.model.dtype):
        loss,_=answer_loss(method.model,batch,row['answer'],processor)
    loss.backward()
    norms=[p.grad.detach().float().norm() for p in method.model.parameters() if p.grad is not None]
    assert norms and all(torch.isfinite(x) for x in norms)
    write(args.run_dir/'checkpoint_reload_proof.json',{'passed':True,'fresh_process':True,
          'expected_loss':probe['loss'],'reloaded_loss':value,'absolute_error':error,'finite_backward':True,
          'method':obj['config']['method'],'manifest_sha256':obj['config']['manifest_sha256'],
          'execution_source_sha256':obj['config']['execution_source_sha256']})
    # Preflight weights must never be used as formal starting weights.
    for p in (args.run_dir/'checkpoints').glob('*.pt'):p.unlink()


if __name__=='__main__':main()
