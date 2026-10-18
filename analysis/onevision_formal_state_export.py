"""Export trusted formal SeqFT stage adapters on CPU; no training or TEST reading.

Full graphs are our own cloudpickle checkpoints. Keep this out of the frozen
baseline source directory so existing source manifests remain unchanged.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)


def main(root,bundle_path,output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    frame=root/'exp/two_backbone_native_cl_20261017_v2'
    sys.path.insert(0,str(frame))
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    bundle=json.loads(bundle_path.read_text())
    item=next(i for i in bundle['baseline_proofs'] if i['proof']['candidate']['method']=='SeqFT_LoRA')
    proof=item['proof'];run=root/proof['candidate']['run']
    assert proof['final_QA']==21454 and proof['rescored_cells']==74 and proof['positive_finite_training_stages']==19
    assert proof['candidate']['backbone']=='onevision'
    config=json.loads((run/'config.json').read_text())
    assert sha(run/'config.json')==proof['config_sha256']
    assert config['condition']=='pct01' and config['model_dtype']=='torch.float16'
    assert config['batch_size']==16 and config['epochs']==1 and config['seed']==42
    assert config['image_area_cap']==401408 and config['visual_cache_limit_bytes']==134217728
    assert all(sha(frame/name)==value for name,value in config['execution_source_sha256'].items())
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>12*1024**3, 'Do not crowd live baseline CPU memory'
    output.mkdir(parents=True,exist_ok=False)
    records=[];started=time.monotonic()
    try:
        for stage in range(1,20):
            p=run/f'checkpoints/stage_{stage:02}.pt';meta=json.loads(p.with_suffix('.json').read_text())
            assert meta['stage']==stage and meta['method']=='SeqFT_LoRA'
            assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']
            obj=torch.load(p,map_location='cpu',pickle_module=pickle,weights_only=False)
            assert obj['stage']==stage and obj['config']['execution_source_sha256']==config['execution_source_sha256']
            assert obj['config']['manifest_sha256']==config['manifest_sha256']
            method=obj['method'];state=method.changed_state()
            assert len(method.layers)==48 and len(state)==96
            assert sum(v.numel() for v in state.values())==540672
            assert all(v.device.type=='cpu' and v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
            assert all(k.endswith(('.A.weight','.B.weight')) for k in state)
            own=method.model.state_dict();assert all(torch.equal(own[k],v) for k,v in state.items())
            tensors={k:v.detach().clone() for k,v in state.items()}
            dest=output/f'stage_{stage:02}.pt';temp=dest.with_suffix('.pt.partial')
            torch.save({'stage':stage,'state':tensors,'source_checkpoint_sha256':meta['sha256'],
                        'manifest_sha256':config['manifest_sha256']},temp)
            temp.replace(dest)
            reload=torch.load(dest,map_location='cpu',weights_only=True)
            assert all(torch.equal(v,reload['state'][k]) for k,v in tensors.items())
            assert not torch.cuda.is_initialized()
            records.append(dict(stage=stage,source_checkpoint_sha256=meta['sha256'],
                source_checkpoint_bytes=meta['bytes'],export_file=dest.name,export_sha256=sha(dest),
                tensor_parameters=540672,tensor_bytes=sum(v.numel()*v.element_size() for v in tensors.values()),
                exact_original_and_fresh_CPU_reload=True))
            write(output/'progress.json',dict(CPU_only=True,complete=False,stages=records))
            print(json.dumps(dict(stage=stage,CPU_only=True,parameters=540672)),flush=True)
            del reload,tensors,state,own,method,obj;gc.collect()
        assert all(sha(frame/n)==v for n,v in config['execution_source_sha256'].items())
        write(output/'COMPLETE.json',dict(complete=True,CPU_only=True,GPU_operations=0,optimizer_updates=0,
            TEST_predictions_read=0,selection_or_routing_performed=False,stage_count=19,stages=records,
            original_training_updates_charged=93,original_training_config=config,
            original_complete_proof_sha256=item['sha256'],input_bundle_sha256=sha(bundle_path),
            source_sha256=sha(Path(__file__)),seconds=time.monotonic()-started,
            no_performance_or_memory_advantage_claim=True,
            next_required='common FP16/input frozen TRAIN-only cache construction and original final-prediction reload gates'))
    except Exception:
        import traceback
        write(output/'FAILED.json',dict(traceback=traceback.format_exc(),no_blind_retry=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.bundle,a.output)
