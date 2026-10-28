"""Fresh-process fullFT preflight CP reload, same used TRAIN QA; no TEST/F1.
Never changes/deletes a checkpoint or performs an optimizer update.
"""
import argparse,hashlib,json,math,os
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--config-sha256',required=True)
    ap.add_argument('--checkpoint-sha256',required=True)
    ap.add_argument('--checkpoint-metadata-sha256',required=True)
    a=ap.parse_args();run=a.run_dir.resolve()
    assert run.is_relative_to(Path(__file__).resolve().parent/'runs')
    config_raw=(run/'config.json').read_bytes();assert hashlib.sha256(config_raw).hexdigest()==a.config_sha256
    meta_raw=(run/'checkpoints/latest.json').read_bytes();assert hashlib.sha256(meta_raw).hexdigest()==a.checkpoint_metadata_sha256
    config=json.loads(config_raw);meta=json.loads(meta_raw)
    assert config['method']=='FullFT_Seq' and config['preflight_only']
    assert config['training_parameter_dtype']=='torch.float32' and config['autocast_dtype']=='torch.bfloat16'
    assert config['final_deployment_dtype']=='torch.bfloat16'
    assert config['condition'] in ('pct01','pct10') and config['backbone'] in ('qwen','onevision')
    assert meta['stage']==1 and meta['method']=='FullFT_Seq' and meta['sha256']==a.checkpoint_sha256
    assert meta['manifest_sha256']==config['manifest_sha256'] and meta['full_architecture_and_runtime']
    cp=(run/meta['path']).resolve();assert cp.is_relative_to((run/'checkpoints').resolve())
    assert cp.stat().st_size==meta['bytes']
    root=Path(os.environ['LLMCL_ROOT']).resolve();assert root==Path('/data1/liyapeng/program/LLMCL')
    assert os.environ['LLMCL_HOST']=={'qwen':'new3090','onevision':'new3090_onevision'}[config['backbone']]
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('1','2','3')
    for rel,sha in config['execution_source_sha256'].items():
        file=(root/rel).resolve();assert file.is_relative_to(root)
        assert hashlib.sha256(file.read_bytes()).hexdigest()==sha
    for name in ('core.py','run.py','reload_probe.py'):
        rel=str((Path(__file__).resolve().parent/name).relative_to(root))
        assert rel in config['execution_source_sha256']
    preflight=json.loads((run/'gpu_preflight.json').read_text())
    assert preflight['passed'] and preflight['not_formal'] and preflight['checkpoint']==meta
    assert len(preflight['cases'])==19
    # Exclusive marker refuses duplicate model work, even after a failed attempt.
    started=run/'GPU_RELOAD_STARTED.json'
    with started.open('x',encoding='utf-8') as f:
        json.dump({'pid':os.getpid(),'config_sha256':a.config_sha256,'checkpoint_sha256':a.checkpoint_sha256,
                   'reload_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},f)
    try:
        import torch
        import core
        rt,bridge,checkpoints=core.install_runtime()
        from training_validity import parameter_digest
        assert torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0]>=8
        obj=checkpoints.load(run,expected=config)  # original full-file SHA + full graph loader
        method=obj['method'];model=method.model;processor=obj['processor'];probe=obj['probe']
        assert obj['stage']==1 and obj['config']==config and obj['format']=='llmcl-method-graph-v1'
        assert probe['schema']=='fullft-checkpoint-probe-v1' and probe['preflight_only'] and probe['split']=='train'
        assert probe['producer_identity']['pid']!=os.getpid(), 'Must use an independent process'
        assert probe['AMP_dtype']=='torch.bfloat16' and probe['parameter_dtype']=='torch.float32'
        assert preflight['cases'][-1]['source_id']==probe['row']['id'] and preflight['cases'][-1]['task']==probe['task']['task']
        row_hash=hashlib.sha256(json.dumps(probe['row'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        assert row_hash==probe['row_sha256'] and probe['row']['images']
        bridge.bind(model)
        inventory=core.full_parameter_inventory(method);assert inventory==probe['parameter_inventory']
        before=parameter_digest(core.unique_parameters(model));assert before==probe['parameter_sha256']
        assert all(p.grad is None for p in core.unique_parameters(model))
        model.eval();method.set_instruction(probe['task']['task'])
        with torch.no_grad(),torch.autocast('cuda',dtype=core.DTYPE):
            batch,details=rt.input_batch(probe['row'],probe['task'],processor,model.device)
            loss,n=rt.answer_loss(model,batch,probe['row']['answer'],processor)
        assert core.batch_digest(batch)==probe['input_sha256'] and details==probe['input_details']
        value=float(loss);error=abs(value-probe['loss']);tolerance=max(.005,abs(probe['loss'])*.002)
        assert math.isfinite(value) and error<tolerance and n==probe['answer_tokens']
        del batch,loss
        # Repack under TRAIN mode: eval-packed OneVision features have no graph.
        # This is the same saved TRAIN QA, one backward, zero optimizer updates.
        model.train();model.zero_grad(set_to_none=True)
        with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=core.DTYPE):
            batch,_=rt.input_batch(probe['row'],probe['task'],processor,model.device)
            loss,_=rt.answer_loss(model,batch,probe['row']['answer'],processor)
        assert torch.isfinite(loss);loss.backward()
        groups={};all_finite=True
        for name,p in model.named_parameters():
            if p.grad is None:continue
            finite=bool(torch.isfinite(p.grad).all());all_finite &= finite
            group=('projector' if ('projector' in name or 'visual.merger' in name) else 'newline' if 'image_newline' in name else 'vision' if name.startswith('visual.') else 'head' if name.startswith('lm_head.') else 'language')
            item=groups.setdefault(group,{'gradient_tensors':0,'nonzero_gradient_tensors':0})
            item['gradient_tensors']+=1;item['nonzero_gradient_tensors']+=int(bool(torch.count_nonzero(p.grad)))
        assert all_finite and all(groups.get(g,{}).get('nonzero_gradient_tensors',0)>0 for g in ('language','vision','projector'))
        model.zero_grad(set_to_none=True)
        assert parameter_digest(core.unique_parameters(model))==before
        result={'passed':True,'preflight_only':True,'not_formal_results':True,'fresh_process':True,
                'same_used_TRAIN_QA':True,'TEST_QA':0,'F1_scores':0,'backwards':1,'optimizer_updates':0,
                'expected_loss':probe['loss'],'reloaded_loss':value,'absolute_error':error,'tolerance':tolerance,
                'tolerance_source':'unchanged native checkpoint_probe max(.005,abs(loss)*.002)',
                'input_digest_exact':True,'parameter_digest_exact':True,'parameter_inventory':inventory,
                'finite_backward':all_finite,'module_gradients':groups,'config_sha256':a.config_sha256,
                'checkpoint_sha256':a.checkpoint_sha256,'checkpoint_metadata_sha256':a.checkpoint_metadata_sha256,
                'execution_source_sha256':config['execution_source_sha256'],'checkpoint_retained':True}
        with (run/'checkpoint_reload_proof.json').open('x',encoding='utf-8') as f:
            json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    except BaseException as exc:
        with (run/'GPU_RELOAD_FAILURE.json').open('x',encoding='utf-8') as f:
            json.dump({'failed':True,'exception':type(exc).__name__,'message':str(exc),'checkpoint_retained':True},f)
        raise


if __name__=='__main__':main()
