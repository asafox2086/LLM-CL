"""Independent complete-model reload, original five windows, then full frozen TEST."""
import argparse
import ast
import gc
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace
from export import validate,sha,read,write,module,identity,idle_gpu,FOLD_DIGEST,MANIFEST_SHA

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--export-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    b,root,run,port,config,cal,probe=validate(a.binding);assert b['evaluation_timeout_seconds']==86400
    assert a.output.is_absolute() and not a.output.exists() and a.export_dir.is_absolute()
    exported=read(a.export_dir/'EXPORT.json');ec=read(a.export_dir/'COMPLETE.json');ex=read(a.export_dir/'EXIT.json')
    assert not (a.export_dir/'FAILURE.json').exists() and ec['complete'] and ex['returncode']==0
    assert ec['binding_sha256']==sha(a.binding) and ec['EXPORT_sha256']==sha(a.export_dir/'EXPORT.json')
    old=ec['identity'];proc=Path('/proc')/str(old['pid'])/'stat';assert not proc.exists() or proc.read_text().rsplit(') ',1)[1].split()[19]!=str(old['startticks'])
    cp=a.export_dir/'model.pt';assert cp.stat().st_size==exported['checkpoint']['bytes'] and sha(cp)==exported['checkpoint']['sha256']
    assert exported['fold_digest']==FOLD_DIGEST and exported['checkpoint']['external_base_weights_required'] is False
    actual_files={str(p.relative_to(a.export_dir/'processor')):sha(p) for p in (a.export_dir/'processor').rglob('*') if p.is_file()};assert actual_files==exported['processor_files']
    manifest=read(b['manifest_path']);assert len(manifest['tasks'])==19
    data={};test_count=0;devrows=[];split=read(b['split_path'])
    for task in manifest['tasks']:
        tid=task['task'];f=task['files']['test'];path=root/f['path'];assert sha(path)==f['sha256']
        rows=[json.loads(line) for line in path.read_text().splitlines()];assert len(rows)==f['count'];data[tid,'test']=rows;test_count+=len(rows)
        tf=task['files']['train'];path=root/tf['path'];assert sha(path)==tf['sha256']
        byid={r['id']:r for r in map(json.loads,path.read_text().splitlines())};devrows.extend((task,byid[i]) for i in split['dev_ids'][str(tid)])
    assert test_count==21454 and len(devrows)==100
    if not a.execute:print(json.dumps(dict(metadata_validated=True,model_loaded=False,TEST_count=test_count)));return
    idle_gpu(b);a.output.mkdir(parents=True,exist_ok=False);out=a.output;who=identity();assert who['pid']==who['pgid'];tick=time.monotonic()
    write(out/'STARTED.json',dict(identity=who,binding_sha256=sha(a.binding),unix=time.time(),limit_seconds=86400,checkpoint_sha256=exported['checkpoint']['sha256']))
    def expired(*_):raise TimeoutError('Fixed PAS static final evaluation86400 deadline expired')
    signal.signal(signal.SIGALRM,expired);signal.alarm(86400)
    try:
        import torch
        from transformers import AutoProcessor
        torch.set_num_threads(8);torch.cuda.set_device(0);assert torch.cuda.device_count()==1
        os.environ.update(LLMCL_ROOT=str(root),LLMCL_RUN=str(out),LLMCL_ASSET=str(root))
        sys.path.insert(0,str(port));import runtime_utils as rt,onevision_bridge as bridge,frozen_vision as fv
        assert rt.ROOT.resolve()==root.resolve()
        deployment=module('pas_final_deployment',Path(__file__).parent/'deployment.py');storage=module('pas_final_storage',b['storage_helper_path'])
        model,reloaded=deployment.load_complete(cp,'cuda:0',storage)
        assert reloaded['digest']==FOLD_DIGEST and reloaded['strict_complete_reload'] and reloaded['self_contained_meta_construction']
        processor=AutoProcessor.from_pretrained(a.export_dir/'processor',local_files_only=True)
        before=storage.digest(model.core);buffers_before=storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
        bridge.bind(model);bridge.LAST_NATIVE=None;bridge.NATIVE_GATE_MODE=False;visual_original=fv.install(model)
        assert storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers_before
        generation_before={key:dict(public=value.to_dict(),semantics=deployment.generation_semantics(value)) for key,value in [('outer',model.core.generation_config),('language',model.core.language_model.generation_config)]}
        assert generation_before==exported['generation_before']
        # Copy exactly the already reviewed original PAS resource-only GPU preparation.
        old_prepare=fv.prepare;old_limit=fv.LIMIT;fv.LIMIT=0
        resource_source=Path(b['pas_resource_worker_path']).read_text();tree=ast.parse(resource_source)
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='gpu_prepare')
        namespace={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),b['pas_resource_worker_path'],'exec'),namespace)
        fv.prepare=namespace['gpu_prepare']
        batchers=module('pas_final_batcher',b['batcher_path']);readouts=module('pas_final_readouts',b['readouts_path'])
        ctx=SimpleNamespace(storage=storage,split=split,bridge=bridge,rt=rt,processor=processor)
        batcher=batchers.Batcher(ctx,model,'FINAL_RESOURCE')
        prior=read(b['pas_memory_path']);assert len(prior)==5
        def nvml():
            import subprocess
            result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
            assert result.returncode==0
            found=[line.split(',') for line in result.stdout.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
            assert len(found)==1;return dict(unix=time.time(),process_used_MiB=float(found[0][1].strip()))
        observations=[];original_generate=readouts._generate
        def observed_generate(*args,**kwargs):
            before_nvml=nvml();tokens=original_generate(*args,**kwargs);observations.append(dict(before=before_nvml,after=nvml()));return tokens
        readouts._generate=observed_generate
        def provider(model_,task,row):
            for path in row['images']:assert sha(root/path)==split['image_sha256'][path]
            return batcher.readout(model_,task,row)
        windows,counts=readouts.resource_windows(model,processor,devrows,storage,batch_provider=provider,generation_policy=dict(generation_stop_ids=processor.tokenizer.eos_token_id,pad_id=processor.tokenizer.pad_token_id,context_limit=None),prior_windows=prior)
        assert counts['new_generations']==5 and len(observations)==5
        for current,old,observed in zip(windows,prior,observations):
            for key in ['id','domain','raw_input_sha256','prepared_sha256','transport_sha256','input_shape','input_tensor_shapes','new_tokens']:assert current[key]==old[key],key
            current['NVML_observation']=observed
        write(out/'windows.json',windows)
        fv.prepare=old_prepare;fv.LIMIT=old_limit;readouts._generate=original_generate
        assert model._llmcl_visual_cache_bytes==0
        assert storage.digest(model.core)==before and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffers_before
        write(out/'RESOURCE_COMPLETE.json',dict(windows_sha256=sha(out/'windows.json'),layout=storage.layout(model.core),resource_counts=counts,raw_and_prepared_match_original_PAS=True,physical_GPU=1,GPU_uuid=b['GPU_uuid'],NVML_is_observation_not_peak=True,TTFT_measured=False,speed_measured=False,state_unchanged=True))
        # Original complete native evaluation: no NLL, no changed caps or scoring.
        method=SimpleNamespace(model=model,set_instruction=lambda task:None,vision_original=visual_original)
        rt.evaluate(method,processor,manifest['tasks'],data,out,19,split='test')
        task_values=[];domain_sums={};outputs={}
        for task in manifest['tasks']:
            path=out/'evaluations/stage_19'/('T%02d_test_predictions.jsonl'%task['task']);total=0;f1=0.
            expected=[r['id'] for r in data[task['task'],'test']];actual=[]
            with path.open() as stream:
                for line in stream:
                    row=json.loads(line);actual.append(row['id']);value=row['scores']['token_f1'];assert math.isfinite(value) and 0<=value<=100;f1+=value;total+=1
            assert actual==expected and total==len(expected)
            task_values.append(dict(task=task['task'],domain=task['domain'],count=total,AnswerF1=f1/total))
            agg=domain_sums.setdefault(task['domain'],dict(count=0,sum_F1=0.));agg['count']+=total;agg['sum_F1']+=f1
            outputs[str(path.relative_to(out))]=sha(path)
        domains={name:dict(count=v['count'],AnswerF1=v['sum_F1']/v['count']) for name,v in domain_sums.items()};assert len(domains)==5
        actual=deployment.finish(model,reloaded,storage,response_installed=False)
        generation_after={key:dict(public=value.to_dict(),semantics=deployment.generation_semantics(value)) for key,value in [('outer',model.core.generation_config),('language',model.core.language_model.generation_config)]}
        assert generation_after==generation_before
        write(out/'RESULT.json',dict(tasks=task_values,domains=domains,domain_macro_final=sum(v['AnswerF1'] for v in domains.values())/5,AP19=sum(v['AnswerF1'] for v in task_values)/19,metric='existing raw Answer F1',score_field='token_f1',TEST_QA=21454,quality_generations=21454,resource_generations=5,new_scores=21454,NLL_requests=0,optimizer_updates=0,backwards=0,FWT=None,BWT=None,checkpoint_sha256=exported['checkpoint']['sha256'],checkpoint_bytes=exported['checkpoint']['bytes'],fold_digest=FOLD_DIGEST,outputs=outputs,reloaded=reloaded,invariant=actual,generation_before=generation_before,generation_after=generation_after))
        write(out/'COMPLETE.json',dict(complete=True,identity=who,binding_sha256=sha(a.binding),EXPORT_COMPLETE_sha256=sha(a.export_dir/'COMPLETE.json'),RESULT_sha256=sha(out/'RESULT.json'),RESOURCE_COMPLETE_sha256=sha(out/'RESOURCE_COMPLETE.json'),seconds=time.monotonic()-tick))
        signal.alarm(0)
    except BaseException:write(out/'FAILURE.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
