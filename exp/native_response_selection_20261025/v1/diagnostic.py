"""Read-only CUDA response diagnostic; no training or answer generation."""
from pathlib import Path
import hashlib,importlib.util,json,os,signal,sys,time,traceback

ROOT=Path(os.environ['LLMCL_ROOT']);RUN=Path(os.environ['LLMCL_RUN']);ASSET=Path(os.environ['LLMCL_ASSET'])
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(n,v):
    with (RUN/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def module(n,p):
    spec=importlib.util.spec_from_file_location(n,p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v

def main():
    ready=read(RUN/'READY.json');rh=sha(RUN/'READY.json')
    for n,h in ready['sources'].items():assert sha(RUN/n)==h,n
    for n,h in ready['environment'].items():assert sha(n)==h,n
    for n,h in ready['runtime_files'].items():
        if not n.endswith('.safetensors'):assert sha(ASSET/n)==h,n
    assert sha(ROOT/'exp/ffn_static_delete_20261025/v1/core.py')==ready['storage_helper_sha256']
    ids=module('native_ids',ASSET/'exp/paired_image_support_20261024/v1/common.py');gpu=int(os.environ['LLMCL_GPU']);card=ids.inventory()[gpu]
    assert not card['owners'] and card['memory_MiB']<500 and card['utilization']<10
    who=ids.identity(os.getpid());assert who['pid']==who['pgid'];tick=time.monotonic()
    write('STARTED.json',dict(unix=time.time(),identity=who,GPU=gpu,ready_sha256=rh,limit=600,boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    def timeout(*args):raise TimeoutError('Fixed native response diagnostic 600 seconds expired')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(8);torch.cuda.set_device(gpu);device=f'cuda:{gpu}'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ASSET/'exp/two_backbone_native_cl_20261017_v2'))
    import onevision_bridge as bridge,runtime_utils as rt,frozen_vision as fv
    storage=module('native_storage',ROOT/'exp/ffn_static_delete_20261025/v1/core.py')
    from transformers import AutoProcessor
    base=ASSET/'model/LLaVA-OneVision-Qwen2-0.5B-OV';processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    core,info=bridge.LlavaOnevisionForConditionalGeneration.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True,device_map={'':device},low_cpu_mem_usage=True,output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']),info
    model=bridge.OneVisionBridge(core).requires_grad_(False).eval();bridge.bind(model);fv.LIMIT=0;fv.install(model)
    def gpu_prepare(batch,row,active,original=None):
        pixels=batch.get('pixel_values')
        if pixels is None or pixels.ndim==2:return False
        with torch.no_grad(),torch.autocast('cuda',dtype=active.visual.get_dtype()):features=(original or active._llmcl_visual_original)(pixels.to(active.visual.get_dtype()),batch['image_grid_thw']).detach()
        assert features.is_cuda and features.ndim==2 and features.shape[-1]==896 and torch.isfinite(features).all()
        active._llmcl_visual_cache_misses+=1;batch['pixel_values']=features;assert active._llmcl_visual_cache_bytes==0;return False
    fv.prepare=gpu_prepare
    policy=read(ASSET/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json')['conditions']['pct01']
    assert sha(ASSET/policy['manifest'])==ready['manifest_sha256'];manifest=read(ASSET/policy['manifest']);task=manifest['tasks'][0];assert task['task']==1
    split_path=ASSET/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json';assert sha(split_path)==ready['split_sha256'];split=read(split_path)
    train=task['files']['train'];assert sha(ASSET/train['path'])==train['sha256']
    fit_ids=set(split['fit_ids']['1']);rows={r['id']:r for r in map(json.loads,(ASSET/train['path']).read_text().splitlines()) if r['id'] in fit_ids};assert len(rows)==327
    groups=[]
    for gid,g in split['groups'].items():
        selected=sorted(set(g['ids']).intersection(rows))
        if selected:assert not g['dev'];groups.append([gid,selected[0]])
    groups.sort(key=lambda g:hashlib.sha256(('vision-interface-20261025:'+g[0]).encode()).digest());groups=groups[:32];assert groups==ready['groups']
    before=storage.digest(core);before_buffers=storage.digest(list(core.named_buffers(remove_duplicate=False)));layout=storage.layout(core)
    assert layout['weights']==dict(coefficients=893675552,bytes=1787351104)
    context={};pending={};records=[];hooks=[]
    def gate_hook(index):
        def hook(layer,args,out):
            assert index not in pending and out.ndim==3 and out.shape[0]==1 and out.is_cuda
            selected=out[0].index_select(0,context['positions']).float();a=selected[:,0::2].square().mean(1);b=selected[:,1::2].square().mean(1)
            d=(a-b)/(a+b+1e-8);assert torch.isfinite(d).all()
            pending[index]=dict(d=d,gate_dtype=str(out.dtype))
        return hook
    def down_hook(index):
        def hook(layer,args,out):
            item=pending.pop(index);phi=args[0][0].index_select(0,context['positions']);h=out[0].index_select(0,context['positions']);assert phi.is_cuda and h.is_cuda
            signs=torch.ones(phi.shape[-1],device=phi.device,dtype=phi.dtype);signs[1::2]=-1
            with torch.autocast('cuda',dtype=torch.float16):diff=torch.nn.functional.linear(phi*signs,layer.weight,layer.bias)
            # Qwen2's down has no bias; retaining this explicit assertion avoids
            # accidentally adding a bias into the signed contrast.
            assert layer.bias is None
            d=item.pop('d');h2=h.float().square().mean();delta2=diff.float().square().mean();weighted2=(d[:,None]*diff.float()).square().mean()
            values=torch.stack([d.mean(),d.square().mean(),h2,delta2,weighted2,d.min(),d.max()]);assert torch.isfinite(values).all()
            context['layers'][index]=dict(layer=index,**item,phi_dtype=str(phi.dtype),h_dtype=str(h.dtype),contrast_dtype=str(diff.dtype),positions=len(d),mean_d=float(values[0]),mean_d2=float(values[1]),mean_h2=float(values[2]),mean_contrast2=float(values[3]),mean_weighted_contrast2=float(values[4]),min_d=float(values[5]),max_d=float(values[6]))
        return hook
    assert len(model.model.layers)==24
    for i,layer in enumerate(model.model.layers):
        assert layer.mlp.gate_proj.out_features==layer.mlp.up_proj.out_features==4864 and layer.mlp.down_proj.bias is None
        hooks.extend([layer.mlp.gate_proj.register_forward_hook(gate_hook(i)),layer.mlp.down_proj.register_forward_hook(down_hook(i))])
    try:
        for gid,rid in groups:
            row=rows[rid]
            for name in row['images']:assert sha(ASSET/name)==split['image_sha256'][name]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                batch,details=rt.input_batch(row,task,processor,device)
                target=processor.tokenizer(row['answer'],add_special_tokens=False,truncation=True,max_length=512)['input_ids'];assert target;target=target+[processor.tokenizer.eos_token_id]
                start=batch['input_ids'].shape[1]-1;n=min(64,len(target));positions=start+torch.arange(n,device=device)*(len(target)-1)//max(1,n-1)
                context.update(positions=positions,layers={});assert not pending
                loss,count=rt.answer_loss(model,batch,row['answer'],processor);assert count==len(target) and torch.isfinite(loss)
            assert not pending and set(context['layers'])==set(range(24))
            records.append(dict(id=rid,group_id=gid,input_digest=storage.digest(batch),image_sha256={p:split['image_sha256'][p] for p in row['images']},target_tokens=count,prediction_positions=positions.cpu().tolist(),layers=[context['layers'][i] for i in range(24)]))
            del batch,loss
    finally:
        for hook in hooks:hook.remove()
    assert storage.digest(core)==before and storage.digest(list(core.named_buffers(remove_duplicate=False)))==before_buffers
    assert all(not p.requires_grad and p.grad is None and p.is_cuda for p in core.parameters())
    summary=[]
    for i in range(24):
        values=[r['layers'][i] for r in records];mean={k:sum(v[k] for v in values)/32 for k in ['mean_d','mean_d2','mean_h2','mean_contrast2','mean_weighted_contrast2']}
        variance=max(0.,mean['mean_d2']-mean['mean_d']**2);summary.append(dict(layer=i,**mean,RMS=mean['mean_d2']**.5,variance=variance,min_d=min(v['min_d'] for v in values),max_d=max(v['max_d'] for v in values)))
    passed=all(v['RMS']>0 and v['mean_contrast2']>0 and v['mean_weighted_contrast2']>0 for v in summary) and any(v['min_d']<v['max_d'] for v in summary)
    seconds=time.monotonic()-tick;assert seconds<=600
    write('COMPLETE.json',dict(complete=True,ready_sha256=rh,identity=who,GPU=gpu,seconds=seconds,records=records,layers=summary,nondegenerate_only=passed,
        decision='eligible_for_separate_paired_design_review' if passed else 'close_fixed_response_partition',
        layout=layout,initial_state_digest=before,all_buffers_digest=before_buffers,state_and_buffers_unchanged=True,
        language_NLL_requests=32,vision_requests=32,extra_sampled_Down_requests=768,new_generations=0,new_scores=0,backwards=0,optimizer_updates=0,TEST_QA=0,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),quality_or_expert_deletability_proven=False))
    signal.alarm(0)

if __name__=='__main__':
    try:main()
    except BaseException:write('FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),no_quality_claim=True));raise
