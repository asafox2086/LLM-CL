"""Independent S16 dense export and native four-domain final evaluation. No training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time

SCHEMA = 'task-gradient-four-domain-final-v1'
MANIFEST_SHA = '6ffae80b7e290b4bcd7b3ef8df7613c176a344221d28e874125f3faf87bd6798'
CP_SHA = '7a34ff44b996dbdff200d1125b13f4ef6b44328025e4df83f43ec7f30dc56acb'
CP_BYTES = 1928263780
META_SHA = 'bcebbe7e3a67961868a738f55313898189ba856081b5aa3b25e4409b6f33a0d7'
CONFIG_SHA = '3b585ecc0d2bd6b7331f33b968da674f502376e84dc0fa025c0202b287c28e26'
ADAPTER_SHA = 'a1ea922be7a774538eae2a9090f87ae48c2610c20587ec15517028b16574c6ef'
RUNTIME_SHA = '4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_metadata(binding_path, output):
    b=read(binding_path);assert b['schema']==SCHEMA
    assert b['worker_sha256']==sha(__file__)
    root=Path(b['source_root']);port=Path(b['source_dir']);run=Path(b['source_run'])
    assert all(p.is_absolute() for p in [root,port,run,Path(output)])
    assert len(b['original_sources'])==29
    assert set(b['original_sources'])=={p.name for p in port.glob('*.py')}
    for name,digest in b['original_sources'].items():assert sha(port/name)==digest,name
    assert b['original_sources']['task_gradient_adapter.py']==ADAPTER_SHA
    assert b['original_sources']['runtime_utils.py']==RUNTIME_SHA
    assert sha(run/'config.json')==CONFIG_SHA
    assert sha(run/'checkpoints/stage_16.json')==META_SHA
    meta=read(run/'checkpoints/stage_16.json');config=read(run/'config.json')
    assert meta['stage']==16 and meta['method']=='TaskGradReuse' and meta['full_architecture_and_runtime']
    assert meta['path']=='checkpoints/stage_16.pt' and meta['sha256']==CP_SHA and meta['bytes']==CP_BYTES
    cp=run/meta['path'];assert cp.is_file() and not cp.is_symlink() and cp.stat().st_size==CP_BYTES
    assert config['method']=='TaskGradReuse' and config['condition']=='pct01'
    for name,digest in b['original_sources'].items():
        assert config['execution_source_sha256'][name]==digest,name
    manifest_path=Path(b['manifest']);assert manifest_path.is_absolute() and sha(manifest_path)==MANIFEST_SHA
    manifest=read(manifest_path);tasks=manifest['tasks']
    assert [t['task'] for t in tasks]==list(range(1,17))
    assert {t['domain'] for t in tasks}=={'natural','medical','geography','finance'}
    assert sum(t['files']['test']['count'] for t in tasks)==19331
    assert manifest['source_manifest_sha256']==config['manifest_sha256']
    for task in tasks:
        f=task['files']['test'];assert sha(root/f['path'])==f['sha256']
    assert int(os.environ['CUDA_VISIBLE_DEVICES'])==b['GPU']==1
    assert not Path(output).exists(),'Independent output must be absent; never resume evaluation'
    return b,root,port,run,cp,meta,config,manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--timeout-seconds',type=int,required=True);args=ap.parse_args()
    b,root,port,source_run,cp,meta,oldconfig,manifest=validate_metadata(args.binding,args.output)
    assert args.timeout_seconds>0
    if not args.execute:
        print(json.dumps({'schema':SCHEMA,'metadata_validated':True,'model_loaded':False}));return
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x',encoding='utf-8') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    started=time.monotonic()
    write('STARTED.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),GPU=1,
        binding_sha256=sha(args.binding),worker_sha256=sha(__file__),timeout_seconds=args.timeout_seconds,
        source_CP_sha256=CP_SHA,stage=16,new_training=False,new_generations=19331))
    def timeout(*_):raise TimeoutError('Independent final-evaluation budget expired; no retry')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(args.timeout_seconds)
    try:
        # Exact original module names are needed by the trusted full-method pickle.
        sys.path.insert(0,str(port))
        import torch
        import checkpoints
        import onevision_bridge
        import runtime_utils as rt
        torch.set_num_threads(8)
        assert sha(cp)==CP_SHA,'Trusted S16 full payload mismatch'
        obj=torch.load(cp,pickle_module=pickle,weights_only=False)
        assert obj['format']=='llmcl-method-graph-v1' and obj['stage']==16 and obj['probe'] is None
        assert obj['config']==oldconfig
        method=obj['method'];processor=obj['processor'];model=method.model
        assert method.name=='TaskGradReuse' and not method.deployed and not method._preflight
        assert method.completed_stage==method.initialized_stage==16 and method.active_stage is None
        assert len(method.layers)==48 and len(method._history)==16
        assert str(model.device)=='cuda:0' and model.dtype==torch.float16
        method.state_root=out/'method_state';method.save_checkpoints=False
        onevision_bridge.bind(model);onevision_bridge.NATIVE_GATE_MODE=False
        rt.restore_rng(obj['rng'])
        before=method._state_digest()
        def generation_values():
            return {name:getattr(model.core if name=='core' else model.core.language_model,'generation_config').to_dict()
                    for name in ['core','language_model']}
        public_generation=generation_values()
        fold_start=time.monotonic();fold=method._fold(final=True)
        method._train_rows.clear();method._processor=None;method._manifest=None
        assert method.deployed and method.completed_stage==method.initialized_stage==16 and method.active_stage is None
        assert len(fold)==48 and not method.layers and not any(p.requires_grad for p in model.parameters())
        assert not any(n.endswith(('.A.weight','.B.weight')) for n,_ in model.named_parameters())
        assert generation_values()==public_generation
        dense=method._state_digest()
        write('FOLD.json',dict(source_stage=16,terminal_stage=16,source_state=before,dense_state=dense,
            modules=fold,seconds=time.monotonic()-fold_start,method_state=method.method_state(16),
            original_fold_function=True,training_context_cleared=True,generation_config=public_generation))
        del obj
        data={}
        for task in manifest['tasks']:
            f=task['files']['test'];rows=[json.loads(line) for line in (root/f['path']).read_text(encoding='utf-8').splitlines()]
            assert len(rows)==f['count'] and len({x['id'] for x in rows})==len(rows)
            data[task['task'],'test']=rows
        config=dict(oldconfig)
        config.update(experiment_id='task_gradient_four_domain_final_20261029',
            manifest_sha256=MANIFEST_SHA,task_counts={t['task_id']:{'test':t['files']['test']['count']} for t in manifest['tasks']},
            terminal_stage=16,scope='four_domain_final_from_equivalent_S16_TRAIN_prefix',
            source_checkpoint_sha256=CP_SHA,postprocess_worker_sha256=sha(__file__),
            postprocess_binding_sha256=sha(args.binding),new_training_updates=0)
        rt.write(out/'config.json',config)
        # Native evaluator, native input/prepare/generation/caps/scoring, no S19 slicing.
        rt.evaluate(method,processor,manifest['tasks'],data,out,16,split='test')
        after=method._state_digest()
        assert after==dense,'Final evaluation changed deployed parameters or ordinary buffers'
        assert generation_values()==public_generation
        saved=checkpoints.save(method,processor,out,16,config)
        files={str(p.relative_to(out)):sha(p) for p in (out/'evaluations/stage_16').iterdir() if p.is_file()}
        assert len(files)==48
        write('COMPLETE.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),GPU=1,
            binding_sha256=sha(args.binding),worker_sha256=sha(__file__),source_stage=16,terminal_stage=16,
            manifest_sha256=MANIFEST_SHA,source_checkpoint_sha256=CP_SHA,
            new_generations=19331,new_training_updates=0,evaluation_files=files,
            checkpoint=saved,dense_state=dense,state_after_evaluation=after,
            self_contained_final_graph=True,teacher_or_adapter_bank=False,
            independent_reload_and_fixed_window_resource='pending separate process',
            elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        write('FAILURE.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),
            error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started,no_retry=True))
        raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
