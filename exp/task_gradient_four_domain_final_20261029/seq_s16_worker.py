"""Independent native SeqFT-LoRA S16 four-domain evaluation; no fold or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time

SCHEMA='seq-four-domain-final-v1'
MANIFEST_SHA='6ffae80b7e290b4bcd7b3ef8df7613c176a344221d28e874125f3faf87bd6798'
CP_SHA='773b304e66c194902c3d76e3d5923d1a58aa274bc744cc30c6a08a67487a4160'
CP_BYTES=1926007972
META_SHA='a737448c3e296b67e184a4e7c376745beb00ff6733250f5bbb8cd89e8c4ad199'
CONFIG_SHA='20db5ebdd6400ba5750a3862071ea3a3d6829259ee5f6b6f8a62e51d141124f2'
STAGE_SHA='36180a6a91528f857b97300a0cdfa9502c1b2be6447357fdc7f642cd30492444'
RUN='/data2/liyapeng_grp/program/LLMCL/exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/SeqFT_LoRA/training_20261017_161154'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_metadata(binding_path,output):
    b=read(binding_path);assert b['schema']==SCHEMA and b['worker_sha256']==sha(__file__)
    root=Path(b['source_root']);port=Path(b['source_dir']);run=Path(b['source_run'])
    assert all(p.is_absolute() for p in [root,port,run,Path(output)]) and str(run)==RUN
    # Frozen methods.py derives ROOT from parents[2], not LLMCL_ROOT.
    assert (port/'methods.py').resolve().parents[2]==root.resolve(),'Put exact old sources under SCIROOT/exp/NAME'
    assert sha(run/'config.json')==CONFIG_SHA and sha(run/'checkpoints/stage_16.json')==META_SHA
    assert sha(run/'stages/stage_16.json')==STAGE_SHA
    config=read(run/'config.json');meta=read(run/'checkpoints/stage_16.json');stage=read(run/'stages/stage_16.json')
    assert stage['stage']==16 and stage['complete'] and stage['weights_saved'] and stage['train_qa']==89
    assert config['method']=='SeqFT_LoRA' and config['condition']=='pct01' and config['backbone']=='onevision' and not config['preflight_only']
    sources=config['execution_source_sha256'];assert len(sources)==25
    assert set(sources)=={p.name for p in port.glob('*.py')}
    for name,h in sources.items():
        assert Path(name).name==name and sha(port/name)==sha(run/'runner_source'/name)==h,name
    assert meta['path']=='checkpoints/stage_16.pt' and meta['stage']==16 and meta['method']=='SeqFT_LoRA'
    assert meta['sha256']==CP_SHA and meta['bytes']==CP_BYTES and meta['full_architecture_and_runtime']
    cp=run/meta['path'];assert cp.is_file() and cp.stat().st_size==CP_BYTES
    manifest_path=Path(b['manifest']);assert manifest_path.is_absolute() and sha(manifest_path)==MANIFEST_SHA
    manifest=read(manifest_path);tasks=manifest['tasks']
    assert manifest['source_manifest_sha256']==config['manifest_sha256']
    assert [t['task'] for t in tasks]==list(range(1,17)) and manifest['seed']==42
    assert {t['domain'] for t in tasks}=={'natural','medical','geography','finance'}
    assert sum(t['files']['test']['count'] for t in tasks)==19331
    for task in tasks:
        f=task['files']['test'];assert sha(root/f['path'])==f['sha256']
    assert int(os.environ['CUDA_VISIBLE_DEVICES'])==b['GPU']==1
    assert not Path(output).exists(),'New output only; no prediction reuse or automatic retry'
    return b,root,port,run,cp,config,meta,manifest


def digest(named):
    import torch
    h=hashlib.sha256()
    for name,t in named:
        h.update(str((name,tuple(t.shape),str(t.dtype))).encode())
        raw=t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().reshape(-1)
        for i in range(0,raw.size,8*1024**2):h.update(raw[i:i+8*1024**2].tobytes())
        del raw
    return h.hexdigest()


def state(model):
    core=model.core
    return dict(parameters=digest(core.named_parameters(remove_duplicate=False)),buffers=digest(core.named_buffers(remove_duplicate=False)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--timeout-seconds',type=int,required=True)
    ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    b,root,port,run,cp,oldconfig,meta,manifest=validate_metadata(a.binding,a.output)
    assert a.timeout_seconds>0
    if not a.execute:print(json.dumps(dict(metadata_validated=True,model_loaded=False)));return
    out=a.output;out.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    def write(name,v):
        fpath=out/name;fpath.parent.mkdir(parents=True,exist_ok=True)
        with fpath.open('x',encoding='utf-8') as f:
            json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    write('STARTED.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),GPU=1,source_stage=16,
        worker_sha256=sha(__file__),binding_sha256=sha(a.binding),timeout_seconds=a.timeout_seconds,
        source_checkpoint_sha256=CP_SHA,new_training_updates=0,new_generations=19331))
    def expired(*_):raise TimeoutError('Independent Seq S16 final-evaluation deadline expired')
    signal.signal(signal.SIGALRM,expired);signal.alarm(a.timeout_seconds)
    try:
        sys.path.insert(0,str(port))
        import torch
        import methods
        import checkpoints
        import runtime_utils as rt
        import onevision_bridge as bridge
        torch.set_num_threads(8)
        assert methods.ROOT.resolve()==rt.ROOT.resolve()==root.resolve()
        for module in [methods,checkpoints,rt,bridge]:
            assert sha(Path(module.__file__))==oldconfig['execution_source_sha256'][Path(module.__file__).name]
        assert sha(cp)==CP_SHA
        obj=torch.load(cp,pickle_module=pickle,weights_only=False)
        assert obj['format']=='llmcl-method-graph-v1' and obj['stage']==16 and obj['probe'] is None
        assert obj['config']==oldconfig and obj['checkpoint_phase']=='after_stage_evaluation'
        method=obj['method'];model=method.model;processor=obj['processor']
        assert type(method) is methods.Method and method.name=='SeqFT_LoRA' and method.embedding is None
        assert str(model.device)=='cuda:0' and model.dtype==torch.float16
        layers=[];coordinates=0
        for i in range(24):
            for part in ['q','v']:
                layer=model.get_submodule(f'model.layers.{i}.self_attn.{part}_proj')
                assert type(layer) is methods.SeqLinear and layer.scale==4 and layer.dropout.p==.1
                assert list(layer.weight.shape)==([896,896] if part=='q' else [128,896]) and layer.weight.dtype==torch.float16
                assert list(layer.A.weight.shape)==[8,896] and list(layer.B.weight.shape)==([896,8] if part=='q' else [128,8])
                assert layer.A.weight.dtype==layer.B.weight.dtype==torch.float32
                coordinates+=layer.A.weight.numel()+layer.B.weight.numel();layers.append(layer)
        assert coordinates==540672 and len(method.layers)==48 and {id(x) for x in layers}=={id(x) for x in method.layers}
        assert sum(p.numel() for p in model.core.parameters())==894216224
        assert sum(p.numel()*p.element_size() for p in model.core.parameters())==1789513792
        assert all(p.grad is None for p in model.core.parameters())
        before=state(model);model.requires_grad_(False).eval()
        bridge.bind(model);bridge.NATIVE_GATE_MODE=False;rt.restore_rng(obj['rng'])
        def generation():return dict(core=model.core.generation_config.to_dict(),language_model=model.core.language_model.generation_config.to_dict())
        gen=generation();write('GRAPH.json',dict(source_stage=16,terminal_stage=16,source_checkpoint_sha256=CP_SHA,
            state=before,generation_config=gen,projection_modules=48,adapter_coordinates=coordinates,
            weight_coefficients=894216224,weight_bytes=1789513792,folded=False,dtype_conversion=False,
            inference_requires_grad_disabled=True,source_model_and_method_preserved=True))
        del obj
        data={}
        for task in manifest['tasks']:
            f=task['files']['test'];rows=[json.loads(line) for line in (root/f['path']).read_text(encoding='utf-8').splitlines()]
            assert len(rows)==f['count'] and len({x['id'] for x in rows})==len(rows)
            data[task['task'],'test']=rows
        config=dict(oldconfig);config.update(experiment_id='seq_four_domain_final_20261029',manifest_sha256=MANIFEST_SHA,
            terminal_stage=16,scope='four_domain_final_from_equivalent_Seq_S16_TRAIN_prefix',
            task_counts={t['task_id']:{'test':t['files']['test']['count']} for t in manifest['tasks']},
            source_checkpoint_sha256=CP_SHA,postprocess_worker_sha256=sha(__file__),
            postprocess_binding_sha256=sha(a.binding),new_training_updates=0,reported_primary_metric='token_f1')
        rt.write(out/'config.json',config)
        rt.evaluate(method,processor,manifest['tasks'],data,out,16,split='test')
        after=state(model);assert after==before and generation()==gen
        saved=checkpoints.save(method,processor,out,16,config)
        files={str(p.relative_to(out)):sha(p) for p in (out/'evaluations/stage_16').iterdir() if p.is_file()}
        assert len(files)==48
        write('COMPLETE.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),GPU=1,
            binding_sha256=sha(a.binding),worker_sha256=sha(__file__),manifest_sha256=MANIFEST_SHA,
            source_checkpoint_sha256=CP_SHA,source_stage=16,terminal_stage=16,new_generations=19331,new_training_updates=0,
            evaluation_files=files,checkpoint=saved,state=before,state_after_evaluation=after,
            graph_unfolded=True,adapter_coordinates=540672,projection_modules=48,
            state_and_public_generation_unchanged=True,independent_reload_and_resource_pending=True,
            elapsed_seconds=time.monotonic()-tick))
    except BaseException as e:
        write('FAILURE.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),error_type=type(e).__name__,
            error=str(e),elapsed_seconds=time.monotonic()-tick,no_retry=True));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
