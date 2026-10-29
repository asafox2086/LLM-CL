"""One sealed PAS S19 deployment-function observation; no training or TEST."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import weakref

CP_SHA='583ab5d02e187fbd6a54bd4a05a019e33101deb8af2d9533795253140721441a'
CP_BYTES=3051292733
CONFIG_SHA='79edcca96dd315260816932438b7c0ec57835f5761d2ec0b88d8ecdb5c837fb6'
MANIFEST_SHA='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
SCHEMA='pas-static-function-probe-v1'
ARMS=('dynamic','constant_unfolded','constant_folded')
LIMIT=2400

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_bytes())

def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def durable(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def id_hash(rows):return hashlib.sha256(''.join(str(t['task'])+':'+row['id']+'\n' for t,row in rows).encode()).hexdigest()

def scope(b):
    root=Path(b['source_root']);run=Path(b['original_run']);port=Path(b['source_dir'])
    assert all(p.is_absolute() for p in [root,run,port,Path(b['manifest']),Path(b['split']),Path(b['storage_helper']),Path(b['protocol_path']),Path(b['selection_path'])])
    assert b['schema']==SCHEMA and sha(__file__)==b['worker_sha256']
    assert sha(b['protocol_path'])==b['protocol_sha256'] and sha(b['selection_path'])==b['selection_sha256']
    selection=read(b['selection_path']);assert selection['schema']=='pas-static-selection-v1'
    assert selection['manifest_sha256']==MANIFEST_SHA and selection['split_sha256']==b['split_sha256']
    assert selection['calibration_QA']==80 and selection['readout_QA']==100 and selection['TEST_QA']==0
    assert selection['original_PAS_saw_all_original_TRAIN_including_readout'] and selection['readout_is_calibration_out_not_training_heldout']
    assert sha(run/'config.json')==b['config_sha256']==CONFIG_SHA
    config=read(run/'config.json');assert config['method']=='PASs_MoE' and config['condition']=='pct01' and config['backbone']=='onevision'
    meta=read(run/'checkpoints/latest.json');assert meta['stage']==19 and meta['sha256']==b['checkpoint_sha256']==CP_SHA and meta['bytes']==CP_BYTES
    assert meta['path']=='checkpoints/stage_19.pt' and (run/meta['path']).stat().st_size==CP_BYTES
    for n,h in config['execution_source_sha256'].items():assert sha(port/n)==sha(run/'runner_source'/n)==h,n
    for p,h in b['runtime_files'].items():assert sha(root/p)==h,p
    assert sha(b['manifest'])==MANIFEST_SHA and sha(b['split'])==b['split_sha256']
    assert sha(b['storage_helper'])==b['storage_helper_sha256']
    manifest=read(b['manifest']);split=read(b['split']);assert len(manifest['tasks'])==19 and manifest['seed']==42
    assert split['FIT_QA']==1213 and split['DEV_QA']==100 and split['TEST_rows_opened']==0
    fit=[];dev=[];allids=set()
    for task in manifest['tasks']:
        f=task['files']['train'];p=root/f['path'];assert sha(p)==f['sha256']
        rows=[json.loads(s) for s in p.read_text().splitlines()];assert len(rows)==f['count']
        byid={row['id']:row for row in rows};assert len(byid)==len(rows) and not allids.intersection(byid);allids.update(byid)
        a=split['fit_ids'][str(task['task'])];d=split['dev_ids'][str(task['task'])]
        assert set(a).isdisjoint(d) and set(a)|set(d)==set(byid)
        fit.extend((task,byid[i]) for i in a);dev.extend((task,byid[i]) for i in d)
    assert len(allids)==1313 and len(dev)==100
    domains=list(dict.fromkeys(t['domain'] for t in manifest['tasks']));assert len(domains)==5
    calibration=[]
    for domain in domains:
        pool=[pair for pair in fit if pair[0]['domain']==domain]
        pool.sort(key=lambda pair:hashlib.sha256(('pas-static-cal-v1:'+pair[1]['id']).encode()).digest())
        assert len(pool)>=16;calibration.extend(pool[:16])
    assert len(calibration)==80 and {r['id'] for _,r in calibration}.isdisjoint({r['id'] for _,r in dev})
    assert isinstance(split['groups'],dict) and all(isinstance(v['ids'],list) and isinstance(v['dev'],bool) for v in split['groups'].values())
    def summary(rows):return [dict(task=t['task'],domain=t['domain'],id_sha256=hashlib.sha256(row['id'].encode()).hexdigest()) for t,row in rows]
    assert summary(calibration)==selection['calibration'] and summary(dev)==selection['readout']
    calgroups={gid for gid,v in split['groups'].items() if any(r['id'] in v['ids'] for _,r in calibration)}
    devgroups={gid for gid,v in split['groups'].items() if v['dev']};assert calgroups.isdisjoint(devgroups)
    assert id_hash(calibration)==b['calibration_ids_sha256'] and id_hash(dev)==b['readout_ids_sha256']
    for _,row in calibration+dev:
        for image in row['images']:assert sha(root/image)==split['image_sha256'][image]
    return root,run,port,config,calibration,dev,domains

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--binding',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    b=read(args.binding);root,run,port,config,cal,dev,domains=scope(b)
    assert not args.output.exists() and args.output.is_absolute()
    if not args.execute:print(json.dumps(dict(validated=True,model_loaded=False,calibration=80,readout=100)));return
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    args.output.mkdir(parents=True,exist_ok=False);out=args.output;started=time.monotonic()
    stat=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    identity=dict(pid=os.getpid(),startticks=stat[19],uid=os.getuid(),pgid=os.getpgrp());assert identity['pid']==identity['pgid']
    durable(out/'STARTED.json',dict(schema=SCHEMA,identity=identity,unix=time.time(),binding_sha256=sha(args.binding),worker_sha256=sha(__file__),limit_seconds=LIMIT,TEST_QA=0,backwards=0,optimizer_updates=0))
    def deadline(*_):raise TimeoutError('Fixed PAS static-function 2400s deadline expired')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(LIMIT)
    try:
        import torch
        import torch.nn.functional as F
        import types
        torch.set_num_threads(8);torch.cuda.set_device(0);assert torch.cuda.device_count()==1
        sys.path.insert(0,str(port))
        import checkpoints,runtime_utils as rt,onevision_bridge as bridge,frozen_vision as fv
        assert rt.ROOT.resolve()==root.resolve()
        for mod in [checkpoints,rt,bridge,fv]:assert sha(mod.__file__)==config['execution_source_sha256'][Path(mod.__file__).name]
        payload=checkpoints.load(run,expected=config);assert payload['format']=='llmcl-method-graph-v1' and payload['stage']==19 and payload['probe'] is None
        method=payload['method'];model=method.model.eval();processor=payload['processor'];del payload
        model.requires_grad_(False);bridge.bind(model);bridge.NATIVE_GATE_MODE=False
        storage=load_module('pas_static_storage',b['storage_helper'])
        layers={n:m for n,m in model.core.named_modules() if hasattr(m,'_current_E')}
        assert len(layers)==168 and {id(x) for x in layers.values()}=={id(x) for x in method.layers}
        configs={}
        for n,m in layers.items():
            a=m.active_adapter
            assert m._current_E()==m.expert_num==2 and m.r[a]==128 and m.gate_router=='a' and m.gate_tie=='A_T'
            assert not m.training and not m.merged and not m.disable_adapters and not m.fan_in_fan_out and not m._record_enabled
            assert m._expert_mask.detach().cpu().tolist()==[1.,1.], 'Original sealed S19 has two active experts at all 168 sites'
            assert m.weight.dtype==torch.float16 and all(p.dtype==torch.float32 for p in m.lora_A.parameters())
            configs[n]=dict(mask=m._expert_mask.detach().cpu().tolist(),tau=m.gate_tau,scaling=m.scaling[a])
        initial_digest=storage.digest(model.core);initial_buffers=storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
        initial_layout=storage.layout(model.core);assert initial_layout['weights']['coefficients']==1034446880
        generation_before=model.core.language_model.generation_config.to_dict()
        # Preserve native vision preparation. Bound the existing CPU visual cache at zero.
        model._llmcl_visual_cache.clear();model._llmcl_visual_cache_bytes=0;fv.LIMIT=0
        calls=dict(calibration_forward=0,readout_CE=0,generations=0,scored=0)
        active_positions={'start':None,'end':None};sums={n:[0.,0.,0] for n in layers}
        def calibrate(name,layer,x):
            assert len(x)==1 and x[0].ndim==3 and x[0].shape[0]==1
            z=x[0];start,end=active_positions['start'],active_positions['end'];assert start>=0 and z.shape[1]==end+1
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                z=z[:,start:end,:].reshape(-1,z.shape[-1]).float();assert z.shape[0]>0
                a=layer.active_adapter;mask=layer._expert_mask[:2]>0.5;count=int(mask.sum())
                assert count==2 and layer._expert_mask[:2].detach().cpu().tolist()==[1.,1.]
                A=[layer.As_weight(e) for e in range(2)];projected=[F.linear(z,w) for w in A]
                scores=torch.cat([v.square().mean(-1,keepdim=True) for v in projected],-1)
                scores=scores/max(layer.gate_tau,1e-6);scores=scores-scores.max(-1,keepdim=True).values
                g=torch.softmax(scores,-1)[:,0]
                B=[layer.lora_B[a].loraB[e].weight for e in range(2)]
                u=[F.linear(projected[e],B[e])*layer.scaling[a] for e in range(2)]
                energy=(u[0]-u[1]).double().square().sum(-1)
                numerator=float((g.double()*energy).mean());denominator=float(energy.mean())
                assert math.isfinite(numerator) and math.isfinite(denominator) and denominator>=0
                sums[name][0]+=numerator;sums[name][1]+=denominator;sums[name][2]+=1
        hooks=[m.register_forward_pre_hook(lambda layer,x,name=n:calibrate(name,layer,x)) for n,m in layers.items()]
        for task,row in cal:
            method.set_instruction(task['task']);batch,_=rt.input_batch(row,task,processor,model.device)
            P=batch['input_ids'].shape[1];target=processor.tokenizer(row['answer'],add_special_tokens=False,truncation=True,max_length=512)['input_ids'];assert target
            active_positions.update(start=P-1,end=P+len(target))
            with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):rt.answer_loss(model,batch,row['answer'],processor)
            calls['calibration_forward']+=1
        for handle in hooks:handle.remove()
        del hooks
        constants={}
        for n,m in layers.items():
            numerator,denominator,count=sums[n];assert count==80
            mask=m._expert_mask[:2].detach().cpu().tolist();valid=[i for i,v in enumerate(mask) if v>0.5]
            assert len(valid)==2 and mask==[1.,1.]
            c0=numerator/denominator if denominator>0 else .5
            assert 0<=c0<=1
            constants[n]=dict(c=[c0,1-c0],numerator=numerator,denominator=denominator,QA=count,mask=mask)
        assert storage.digest(model.core)==initial_digest and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==initial_buffers
        durable(out/'CALIBRATION.json',dict(schema=SCHEMA,rule='per-QA answer-prediction-position mean then equal-QA sum; energy-weighted teacher-path constant',constants=constants,calibration_ids_sha256=id_hash(cal),QA=80,readout_loss_not_yet_read=True,weights_and_buffers_unchanged=True))
        original_forwards={n:m.forward for n,m in layers.items()}
        def constant_forward(layer,x,**kwargs):
            previous=x.dtype;a=layer.active_adapter
            with torch.autocast(x.device.type,enabled=False):
                result=F.linear(x,layer.weight,layer.bias)
                flat=x.reshape(-1,x.shape[-1]).to(layer.lora_A[a].loraA[0].weight.dtype)
                ys=layer.lora_B[a](layer.lora_A[a](layer.lora_dropout[a](flat)))
                for e in range(2):
                    if layer._expert_mask[e]<.5:continue
                    result=result+ys[e].view(*x.shape[:-1],layer.out_features)*layer.scaling[a]*layer._pas_static_c[e]
            return result.to(previous)
        arms={}
        for arm in ARMS:
            if arm=='constant_unfolded':
                for n,m in layers.items():m._pas_static_c=constants[n]['c'];m.forward=types.MethodType(constant_forward,m)
            if arm=='constant_folded':
                old_layer_refs=[weakref.ref(value) for value in layers.values()]
                for n,m in layers.items():
                    a=m.active_adapter
                    with torch.no_grad(),torch.autocast('cuda',enabled=False):
                        w=m.weight.detach().float().clone()
                        for e,c in enumerate(constants[n]['c']):
                            if m._expert_mask[e]>.5:w.add_(m.lora_B[a].loraB[e].weight.float()@m.As_weight(e).float(),alpha=m.scaling[a]*c)
                        linear=torch.nn.Linear(m.in_features,m.out_features,bias=m.bias is not None,device=m.weight.device,dtype=m.weight.dtype)
                        linear.weight.copy_(w.to(m.weight.dtype))
                        if m.bias is not None:linear.bias.copy_(m.bias)
                        linear.requires_grad_(False);linear.eval()
                    parent,child=n.rsplit('.',1);setattr(model.core.get_submodule(parent),child,linear)
                method.layers=[];method.learner=None
                del m,layers,original_forwards;gc.collect();torch.cuda.empty_cache()
                assert all(ref() is None for ref in old_layer_refs), 'Old PAS layer still strongly referenced after fold'
                assert not any(hasattr(m,'_current_E') for m in model.core.modules())
                assert storage.layout(model.core)['weights']['coefficients']==893675552
            records=[];path=out/(arm+'.jsonl')
            arm_digest=storage.digest(model.core);buffer_digest=storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
            with path.open('x',encoding='utf-8') as stream:
                for task,row in dev:
                    method.set_instruction(task['task']);batch,details=rt.input_batch(row,task,processor,model.device)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=model.dtype):
                        loss,tokens=rt.answer_loss(model,batch,row['answer'],processor)
                        ce=float(loss);assert math.isfinite(ce);calls['readout_CE']+=1
                        output=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=task['max_new_tokens'],use_cache=True,pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
                    response=output[0,batch['input_ids'].shape[1]:];prediction=processor.tokenizer.decode(response,skip_special_tokens=True)
                    calls['generations']+=1;scores=rt.score(prediction,row['answers']);calls['scored']+=1
                    record=dict(id=row['id'],task=task['task'],domain=task['domain'],AnswerCE=ce,target_tokens=tokens,prediction=prediction,references=row['answers'],scores=scores,output_tokens=len(response),generation_cap=task['max_new_tokens'])
                    records.append(record);stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                os.fsync(stream.fileno())
            assert storage.digest(model.core)==arm_digest and storage.digest(list(model.core.named_buffers(remove_duplicate=False)))==buffer_digest
            if arm!='constant_folded':assert arm_digest==initial_digest and buffer_digest==initial_buffers
            aggregates={}
            for domain in domains:
                items=[v for v in records if v['domain']==domain];assert len(items)==20
                aggregates[domain]=dict(count=len(items),AnswerCE=sum(v['AnswerCE'] for v in items)/len(items),AnswerF1=sum(v['scores']['token_f1'] for v in items)/len(items))
            arms[arm]=dict(records_sha256=sha(path),domains=aggregates,macro_F1=sum(v['AnswerF1'] for v in aggregates.values())/5,layout=storage.layout(model.core),state_digest=arm_digest,old_PAS_modules_released=(all(ref() is None for ref in old_layer_refs) if arm=='constant_folded' else None),training_learner_removed=(method.learner is None if arm=='constant_folded' else False),weights_and_buffers_unchanged_during_readout=True)
            durable(out/(arm+'_COMPLETE.json'),arms[arm])
        assert calls==dict(calibration_forward=80,readout_CE=300,generations=300,scored=300)
        after_generation=model.core.language_model.generation_config.to_dict()
        assert after_generation==generation_before
        result=dict(schema=SCHEMA,arms=arms,counts=calls,calibration_sha256=sha(out/'CALIBRATION.json'),checkpoint_sha256=CP_SHA,calibration_ids_sha256=id_hash(cal),readout_ids_sha256=id_hash(dev),TEST_QA=0,backwards=0,optimizer_updates=0,new_checkpoints=0,calibration_outside_readout=True,readout_seen_in_original_training=True,protocol_sha256=b['protocol_sha256'],selection_sha256=b['selection_sha256'],not_new_acquisition_evidence=True,not_independent_deployment_reload=True)
        durable(out/'RESULT.json',result)
        durable(out/'COMPLETE.json',dict(schema=SCHEMA,identity=identity,seconds=time.monotonic()-started,binding_sha256=sha(args.binding),RESULT_sha256=sha(out/'RESULT.json'),worker_sha256=sha(__file__),complete=True))
        signal.alarm(0)
    except BaseException:
        durable(out/'FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
