# WIP full-VLM baseline; explicit CLI only. No GPU admission or allocator.
import argparse,ast,gc,hashlib,json,math,os,random,time
from pathlib import Path


# The original computational runtime has NO pause helper. Bind the separate,
# already-reviewed checkpoint IO source explicitly; do not change core.NATIVE.
PAUSE_SOURCE_RELATIVE = 'exp/two_backbone_native_checkpoint_20261027/run.py'
PAUSE_SOURCE_SHA256 = 'ea347e655c8f3a20d07c86568b351ab1615868bcf9b88d209756c45d1cef1280'
PAUSE_AST_SHA256 = '3d34b44af1b0d9cf10e0e731cd971ad8e9c8829366a1b422935e55c67093d8b3'

def load_pause_helper(write_callback, save_callback):
    """stdlib-only real source extraction/definition; never calls runner.main."""
    import builtins,dis
    path=Path(os.environ['LLMCL_ROOT']).resolve()/PAUSE_SOURCE_RELATIVE
    raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==PAUSE_SOURCE_SHA256
    tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pause_at_boundary']
    assert len(nodes)==1
    node=nodes[0];digest=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
    assert digest==PAUSE_AST_SHA256
    namespace={'write':write_callback,'save_checkpoint':save_callback,'json':json,'hashlib':hashlib,'time':time}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path)+':pause-only','exec'),namespace)
    function=namespace['pause_at_boundary']
    required={i.argval for i in dis.get_instructions(function) if i.opname=='LOAD_GLOBAL'}
    assert required-set(vars(builtins))=={'write','save_checkpoint','json','hashlib','time'}
    assert all(name in namespace or hasattr(builtins,name) for name in required)
    assert function.__code__.co_argcount==7 and function.__defaults__==(None,)
    return function,dict(path=str(path),sha256=PAUSE_SOURCE_SHA256,function_AST_sha256=digest,
                         required_globals=sorted(required),actual_definition_executed=True,body_called=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',choices=['pct01','pct10'],required=True)
    ap.add_argument('--backbone',choices=['onevision','qwen'],required=True)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--manifest-sha256',required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--preflight-only',action='store_true')
    a=ap.parse_args()
    # Validate the actual helper before importing Torch or checking CUDA.
    _,pause_dependency=load_pause_helper(None,None)
    import numpy as np
    import torch
    from transformers import AutoProcessor,Qwen2VLForConditionalGeneration,get_cosine_schedule_with_warmup
    import core
    rt,bridge,checkpoints=core.install_runtime()
    from resource_monitor import ResourceMonitor
    from sparse_results import required_tasks,aggregate
    from training_validity import parameter_digest,warmup_updates,validate_stage
    assert torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0]>=8
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) in (1,2,3), 'NEW GPU0 is reserved'
    assert os.environ['LLMCL_HOST']=={'qwen':'new3090','onevision':'new3090_onevision'}[a.backbone]
    assert core.ROOT==Path('/data1/liyapeng/program/LLMCL')
    pause_at_boundary,pause_runtime=load_pause_helper(rt.write,checkpoints.save)
    assert pause_runtime==pause_dependency
    mp=a.manifest.resolve();assert mp.is_relative_to(core.ROOT)
    raw=mp.read_bytes();assert hashlib.sha256(raw).hexdigest()==a.manifest_sha256
    manifest=json.loads(raw)
    assert manifest['formal_ready'] and not manifest['assets_pending']
    assert manifest['condition']==a.condition and manifest['seed']==42
    assert len(manifest['tasks'])==19 and [t['task'] for t in manifest['tasks']]==list(range(1,20))
    assert manifest['experiment_id']=='five_domain_native_cl_20261016_sparse'
    lock=json.loads((mp.parent.parent/'split_lock.json').read_text())['conditions'][a.condition]
    assert lock['manifest_sha256']==a.manifest_sha256
    assert hashlib.sha256((mp.parent/'frozen_ids.json').read_bytes()).hexdigest()==lock['frozen_ids_sha256']
    data={}
    for task in manifest['tasks']:
        for split,info in task['files'].items():
            blob=(core.ROOT/info['path']).read_bytes()
            assert hashlib.sha256(blob).hexdigest()==info['sha256']==lock['files'][info['path']]
            rows=[json.loads(line) for line in blob.decode().splitlines()]
            assert len(rows)==info['count']==len({r['id'] for r in rows})
            data[task['task'],split]=rows
        assert not {r['id'] for r in data[task['task'],'train']} & {r['id'] for r in data[task['task'],'test']}
    assert sum(len(data[t,'train']) for t in range(1,20))=={'pct01':1313,'pct10':13105}[a.condition]
    assert sum(len(data[t,'test']) for t in range(1,20))==21454
    run=a.run_dir.resolve();assert run.is_relative_to(Path(__file__).resolve().parent/'runs')
    if not a.resume:run.mkdir(parents=True,exist_ok=False)
    else:assert run.is_dir() and not a.preflight_only
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    sources={str(p.relative_to(core.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in list(core.NATIVE.glob('*.py'))+list(Path(__file__).resolve().parent.glob('*.py'))}
    sources[PAUSE_SOURCE_RELATIVE]=pause_dependency['sha256']
    resume_raw=None
    if a.resume:
        prior=json.loads((run/'run_status.json').read_text());assert prior['status']=='paused_checkpoint'
        receipt=(run/prior['pause_receipt']).resolve();assert receipt.is_relative_to((run/'pause_history').resolve())
        assert hashlib.sha256(receipt.read_bytes()).hexdigest()==prior['pause_receipt_sha256']
        paused=json.loads(receipt.read_text())
        assert paused['checkpoint']==json.loads((run/'checkpoints/latest.json').read_text())
        assert hashlib.sha256((run/'checkpoints/latest.json').read_bytes()).hexdigest()==paused['checkpoint_metadata_sha256']
        assert hashlib.sha256((run/'config.json').read_bytes()).hexdigest()==paused['config_sha256']
        request=run/'RESUME_REQUEST.json';resume_raw=request.read_bytes()
        assert json.loads(resume_raw)['pause_receipt_sha256']==prior['pause_receipt_sha256']
    old=checkpoints.load(run) if a.resume else None
    if old:
        config=json.loads((run/'config.json').read_text());assert old['config']==config
        for k,v in {'method':'FullFT_Seq','condition':a.condition,'backbone':a.backbone,
                    'manifest_sha256':a.manifest_sha256,'execution_source_sha256':sources}.items():assert config[k]==v
        assert not config['preflight_only'] and old['probe'] is None
        stage=old['stage'];assert 0<=stage<19
        assert paused['completed_stage']==stage and paused['manifest_sha256']==a.manifest_sha256
        assert paused['execution_source_sha256']==sources
        if stage:
            marker=json.loads((run/'stages'/f'stage_{stage:02d}.json').read_text())
            assert marker['complete'] and marker['weights_saved'] and marker['checkpoint']==json.loads((run/'checkpoints/latest.json').read_text())
        archive=run/'pause_history'/('resume_'+hashlib.sha256(resume_raw).hexdigest()+'.json')
        assert not archive.exists() and request.read_bytes()==resume_raw
        request.rename(archive)
        method=old['method'];processor=old['processor'];model=method.model
        assert method.finalized==(stage==19)
        for folder in (run/'evaluations').glob('stage_*'):
            if int(folder.name.split('_')[-1])>stage:
                dest=run/'interrupted_evaluations'/str(time.time_ns())/folder.name
                dest.parent.mkdir(parents=True,exist_ok=True);folder.rename(dest)
    else:
        base=core.ROOT/'model'/('LLaVA-OneVision-Qwen2-0.5B-OV' if a.backbone=='onevision' else 'Qwen2-VL-2B-Instruct')
        processor=AutoProcessor.from_pretrained(base,local_files_only=True)
        factory=bridge.Factory if a.backbone=='onevision' else Qwen2VLForConditionalGeneration
        model=factory.from_pretrained(base,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cuda()
        method=core.FullFTMethod(model,[t['instruction'] for t in manifest['tasks']]);stage=0
        config={'method':'FullFT_Seq','condition':a.condition,'backbone':a.backbone,'seed':42,'epochs':1,
            'batch_size':16,'micro_batch_size':1,'manifest_sha256':a.manifest_sha256,
            'training_parameter_dtype':'torch.float32','autocast_dtype':'torch.bfloat16',
            'final_deployment_dtype':'torch.bfloat16','model_dtype':'stage-dependent; see checkpoint_parameter_dtype_by_stage',
            'checkpoint_parameter_dtype_by_stage':{str(i):('torch.bfloat16' if i==19 else 'torch.float32') for i in range(1,20)},
            'optimizer':'native HostAdamW; FP32 CPU moments, GPU arithmetic',
            'pause_helper_dependency':pause_dependency,
            'evaluation_precision':'S0-S18 FP32 parameters/BF16AMP; S19 BF16 parameters/BF16AMP',
            'activation_storage':'torch.autograd.graph.save_on_cpu(pin_memory=True)',
            'base_model':str(base),'test_fraction':1.,'train_fraction':{'pct01':.01,'pct10':.10}[a.condition],
            'save_final_parameters':True,'checkpoint_all_stages':True,'preflight_only':a.preflight_only,
            'execution_source_sha256':sources,'task_counts':{t['task_id']:{s:f['count'] for s,f in t['files'].items()} for t in manifest['tasks']}}
        rt.write(run/'config.json',config);snapshot=run/'runner_source';snapshot.mkdir()
        for rel,sha in sources.items():
            blob=(core.ROOT/rel).read_bytes();assert hashlib.sha256(blob).hexdigest()==sha
            target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
    bridge.bind(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    assert all(p.requires_grad for p in core.unique_parameters(model))
    monitor=ResourceMonitor(run);monitor.snapshot('training_before',stage,model)
    if old:rt.restore_rng(old['rng'])
    if a.preflight_only:
        method.begin_stage(1);optimizer=core.make_optimizer(method);cases=[]
        assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in core.unique_parameters(model)}
        for task in manifest['tasks']:
            row=data[task['task'],'train'][0]
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                case=core.accumulate_update(method,optimizer,[row],task,processor,diagnostic=True)
            cases.append({'task':task['task'],'source_id':row['id'],'image_count':len(row.get('images',[])),**case})
        optimizer.zero_grad(set_to_none=True);del optimizer
        seen={g for case in cases for g,v in case['module_probes'].items() if v['nonzero_gradient'] and v['parameter_changed']}
        assert {'language','vision','projector'}<=seen
        reference=core.make_reload_reference(method,processor,row,task)
        cp=checkpoints.save(method,processor,run,1,config,probe=reference)
        rt.write(run/'gpu_preflight.json',{'passed':True,'not_formal':True,'cases':cases,'checkpoint':cp,'required_nonzero_updated_groups':sorted(seen)})
        monitor.snapshot('training_after',1,model);monitor.finish(model)
        return
    if not old:rt.evaluate(method,processor,manifest['tasks'],data,run,0)
    if old:rt.restore_rng(old['rng'])
    elif pause_at_boundary(method,processor,run,0,config,monitor):return
    for stage in range(stage+1,20):
        monitor.stage=stage;method.begin_stage(stage)
        rt.write(run/'run_status.json',{'status':'training','completed_stage':stage-1,'active_stage':stage})
        params=core.unique_parameters(model);before=parameter_digest(params);optimizer=core.make_optimizer(method)
        assert {id(p) for group in optimizer.param_groups for p in group['params']}=={id(p) for p in params}
        rows=sorted(data[stage,'train'],key=lambda row:hashlib.sha256(f"42:{stage}:{row['id']}".encode()).digest())
        assert hashlib.sha256(''.join(r['id']+'\n' for r in rows).encode()).hexdigest()==manifest['tasks'][stage-1]['training_order_sha256']
        updates=math.ceil(len(rows)/16);warmup=warmup_updates(updates)
        scheduler=get_cosine_schedule_with_warmup(optimizer,warmup,updates);positive=0
        (run/'logs').mkdir(exist_ok=True)
        with (run/'logs/events.jsonl').open('a',encoding='utf-8') as events:
            for step,start in enumerate(range(0,len(rows),16),1):
                tick=time.monotonic();lr=[g['lr'] for g in optimizer.param_groups]
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    info=core.accumulate_update(method,optimizer,rows[start:start+16],manifest['tasks'][stage-1],processor)
                positive+=int(any(x>0 for x in lr));scheduler.step()
                event={'event':'train_update','stage':stage,'step':step,'qa_seen':min(start+16,len(rows)),
                    'total_qa':len(rows),'learning_rate_used':lr,'warmup_updates':warmup,'seconds':time.monotonic()-tick,**info}
                events.write(json.dumps(event,allow_nan=False)+'\n');events.flush();print(json.dumps(event),flush=True)
        rt.write(run/'validity'/f'stage_{stage:02d}.json',validate_stage(updates,positive,before,parameter_digest(params)))
        optimizer.zero_grad(set_to_none=True);method.end_stage(stage);del optimizer,scheduler,params
        gc.collect();torch.cuda.empty_cache()
        if stage==19:rt.write(run/'final_dtype_transition.json',core.finalize_for_actual_final_test(method))
        wanted=required_tasks(stage,19)
        rt.write(run/'run_status.json',{'status':'evaluation','completed_stage':stage-1,'active_stage':stage})
        scores=rt.evaluate(method,processor,[t for t in manifest['tasks'] if t['task'] in wanted],data,run,stage)
        cp=core.save_stage(method,processor,run,stage,config)
        rt.write(run/'stages'/f'stage_{stage:02d}.json',{'stage':stage,'complete':True,'weights_saved':True,'checkpoint':cp,
            'train_qa':len(rows),'task_scores':scores,'model_parameter_dtype':str(next(model.parameters()).dtype),'evaluation_matches_saved_graph':True})
        rt.write(run/'run_status.json',{'status':'stage_complete','completed_stage':stage})
        if pause_at_boundary(method,processor,run,stage,config,monitor,cp):return
    aggregate(run,manifest,rt.write);monitor.snapshot('training_after',19,model);monitor.finish(model)
    rt.write(run/'run_status.json',{'status':'completed','completed_stage':19,'final_checkpoint':cp['path'],'final_parameter_file_bytes':cp['bytes']})


if __name__=='__main__':main()
