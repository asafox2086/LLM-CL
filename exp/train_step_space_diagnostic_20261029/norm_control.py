"""Reconstruct the sealed TRAIN pair, then one dense-norm-matched Adam-parameter-ray readout."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import signal
import sys
import time

SCHEMA='train-step-space-norm-control-v1'
EXPECTED_RESULT_SHA='971499e4c9b7843629efbdd0d1978bd502623cb017efbc43b53fbc0a51360294'
ORIGINAL_WORKER_SHA='e86290a7d93936057acfebdfb21ecab81379c5f03b2f18b7bd2e723f841f3931'
CP_SHA='84fe80998252e6e5000afd676e7710d7c262f759c85faa666e967e52431b0376'
CP_BYTES=1931082056
CONFIG_SHA='20db5ebdd6400ba5750a3862071ea3a3d6829259ee5f6b6f8a62e51d141124f2'
MANIFEST_SHA='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
STUDY_SHA='d189bc45b8086e862083320aed1808989638fc2ece214aa48a0a4287b46c7e30'
RUN='/data2/liyapeng_grp/program/LLMCL/exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/SeqFT_LoRA/training_20261017_161154'
TARGETS={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate(binding_path,output):
    b=read(binding_path);assert b['schema']==SCHEMA and b['worker_sha256']==sha(__file__)
    root=Path(b['source_root']);port=Path(b['source_dir']);run=Path(b['source_run'])
    assert all(p.is_absolute() for p in [root,port,run,Path(output)]) and str(run)==RUN
    assert (port/'methods.py').resolve().parents[2]==root.resolve()
    assert sha(run/'config.json')==CONFIG_SHA
    config=read(run/'config.json');sources=config['execution_source_sha256']
    assert config['method']=='SeqFT_LoRA' and config['backbone']=='onevision' and config['condition']=='pct01'
    assert len(sources)==25 and set(sources)=={p.name for p in port.glob('*.py')}
    for n,h in sources.items():assert sha(port/n)==sha(run/'runner_source'/n)==h,n
    meta=read(run/'checkpoints/stage_01.json')
    assert meta['stage']==1 and meta['path']=='checkpoints/stage_01.pt' and meta['sha256']==CP_SHA and meta['bytes']==CP_BYTES
    cp=run/meta['path'];assert cp.is_file() and cp.stat().st_size==CP_BYTES
    mp=Path(b['manifest']);assert mp.is_absolute() and sha(mp)==MANIFEST_SHA
    manifest=read(mp);assert manifest['condition']=='pct01' and manifest['seed']==42
    task=manifest['tasks'][1];assert task['task']==2 and task['files']['train']['count']==11
    f=task['files']['train'];assert sha(root/f['path'])==f['sha256']
    sp=Path(b['all_r8_source']);assert sp.is_absolute() and sha(sp)==STUDY_SHA
    assert int(os.environ['CUDA_VISIBLE_DEVICES'])==b['GPU']==1
    expected_path=Path(b['expected_result']['path']);assert expected_path.is_absolute()
    assert b['expected_result']['sha256']==EXPECTED_RESULT_SHA and sha(expected_path)==EXPECTED_RESULT_SHA
    expected=read(expected_path);assert expected['schema']=='train-step-space-diagnostic-v1' and expected['source_checkpoint_sha256']==CP_SHA
    assert expected['TRAIN_QA']==11 and expected['stage']==2 and set(expected['arms'])=={'qv','all'}
    assert not Path(output).exists()
    return b,root,port,run,cp,config,meta,task,sp,expected


def tensor_digest(items):
    import torch
    h=hashlib.sha256()
    for n,t in items:
        h.update(str((n,tuple(t.shape),str(t.dtype))).encode())
        raw=t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().reshape(-1)
        for i in range(0,raw.size,8*1024**2):h.update(raw[i:i+8*1024**2].tobytes())
    return h.hexdigest()


def rng_digest(state):
    h=hashlib.sha256(pickle.dumps((state['python'],state['numpy']),protocol=4))
    for t in [state['cpu']]+state['cuda']:h.update(t.cpu().numpy().tobytes())
    return h.hexdigest()


def smallest_positive_root(a,b,c,target):
    """Scalar metadata solve, partitioned at all derivative roots; no loss input."""
    assert all(math.isfinite(x) for x in [a,b,c,target]) and a>=0 and c>=0 and target>0
    stationary=[]
    if c>0:
        discriminant=9*b*b-32*a*c
        if discriminant>=0:
            d=math.sqrt(discriminant)
            stationary=[(-3*b-d)/(8*c),(-3*b+d)/(8*c)]
    elif b!=0:stationary=[-2*a/(3*b)]
    points=sorted(set([0.,1.]+[x for x in stationary if 0<x<1]))
    def f(s):return ((c*s+b)*s+a)*s*s-target*target
    candidates=[]
    for lo,hi in zip(points[:-1],points[1:]):
        flo,fhi=f(lo),f(hi)
        if lo>0 and flo==0:candidates.append(lo)
        if hi>0 and fhi==0:candidates.append(hi)
        if flo*fhi<0:
            left,right=lo,hi
            for _ in range(100):
                mid=(left+right)/2;fm=f(mid)
                if fm==0:left=right=mid;break
                if flo*fm<=0:right=mid
                else:left=mid;flo=fm
            candidates.append((left+right)/2)
    assert candidates,'No positive norm root on original Adam parameter segment'
    return min(x for x in candidates if 0<x<=1),dict(a2=a,a3=b,a4=c,target_norm=target,
        derivative_stationary_points=stationary,monotone_partition=points,candidate_roots=sorted(set(candidates)),
        selection='smallest positive real root in (0,1], independent of any loss')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--timeout-seconds',type=int,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    b,root,port,run,cp,config,meta,task,study_path,expected=validate(a.binding,a.output);assert a.timeout_seconds>0
    if not a.execute:print(json.dumps(dict(metadata_validated=True,model_loaded=False)));return
    out=a.output;out.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    def write(n,v):
        with (out/n).open('x',encoding='utf-8') as f:
            json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    write('STARTED.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),GPU=1,binding_sha256=sha(a.binding),
        worker_sha256=sha(__file__),source_checkpoint_sha256=CP_SHA,stage=2,TRAIN_QA=11,
        common_backwards=11,planned_optimizer_steps=2,TEST_QA=0,new_generations=0,new_scores=0,timeout_seconds=a.timeout_seconds))
    def expired(*_):raise TimeoutError('Fixed TRAIN mechanism diagnostic deadline expired')
    signal.signal(signal.SIGALRM,expired);signal.alarm(a.timeout_seconds)
    try:
        sys.path.insert(0,str(port))
        import torch
        import methods,checkpoints,runtime_utils as rt,onevision_bridge as bridge,frozen_vision as fv
        from transformers import get_cosine_schedule_with_warmup
        torch.set_num_threads(8)
        assert methods.ROOT.resolve()==rt.ROOT.resolve()==root.resolve()
        for m in [methods,checkpoints,rt,bridge,fv]:assert sha(Path(m.__file__))==config['execution_source_sha256'][Path(m.__file__).name]
        assert sha(cp)==CP_SHA
        obj=torch.load(cp,pickle_module=pickle,weights_only=False)
        assert obj['format']=='llmcl-method-graph-v1' and obj['stage']==1 and obj['probe'] is None and obj['config']==config
        method=obj['method'];processor=obj['processor'];model=method.model
        assert type(method) is methods.Method and method.name=='SeqFT_LoRA' and method.embedding is None
        assert len(method.layers)==48 and str(model.device)=='cuda:0' and model.dtype==torch.float16
        assert not getattr(method,'host_optimizer',False) and getattr(method,'learning_rate',1e-4)==1e-4
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
        bridge.bind(model);bridge.NATIVE_GATE_MODE=False;rt.restore_rng(obj['rng']);del obj
        f=task['files']['train'];rows=[json.loads(x) for x in (root/f['path']).read_text(encoding='utf-8').splitlines()]
        assert len(rows)==11 and len({x['id'] for x in rows})==11
        rows=sorted(rows,key=lambda x:hashlib.sha256(('42:2:'+x['id']).encode()).digest())
        assert hashlib.sha256(''.join(x['id']+'\n' for x in rows).encode()).hexdigest()==task['training_order_sha256']
        # Native input_batch and frozen-vision preparation once; all arms use these same GPU batches.
        model.train();batches=[];input_info=[]
        for row in rows:
            batch,details=rt.input_batch(row,task,processor,model.device);fv.prepare(batch,row,model,method.vision_original)
            batches.append(batch);input_info.append(dict(id_sha256=hashlib.sha256(row['id'].encode()).hexdigest(),
                prompt_tokens=details['prompt_tokens'],batch_digest=tensor_digest(sorted(batch.items()))))
        def loss_value(batch,row):
            with rt.activation_storage(method),torch.autocast('cuda',dtype=model.dtype):return rt.answer_loss(model,batch,row['answer'],processor)
        # A first-QA eval-only numerical sanity supplements B0's structural no-op proof.
        saved_rng=rt.rng();model.eval()
        with torch.no_grad():sanity_before=float(loss_value(batches[0],rows[0])[0])
        original_qv={n:p for n,p in model.named_parameters() if n.endswith(('.A.weight','.B.weight'))}
        assert len(original_qv)==96 and sum(p.numel() for p in original_qv.values())==540672
        qv_digest=tensor_digest(sorted(original_qv.items()));qv_ids={n:id(p) for n,p in original_qv.items()}
        spec=importlib.util.spec_from_file_location('mechanism_existing_all_r8',study_path);study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
        assert study.QV==['q_proj','v_proj'] and set(study.QV+study.EXTRA)==TARGETS
        # Existing q/v SeqLinear sites are not nn.Linear candidates: only the 120 extra sites are installed.
        extra=study.TargetLoRA(model,study.QV+study.EXTRA);assert len(extra.layers)==120
        method.layers.extend(extra.layers);del extra
        params={n:p for n,p in model.named_parameters() if n.endswith(('.A.weight','.B.weight'))}
        assert len(params)==336 and sum(p.numel() for p in params.values())==4399104
        assert {n:id(params[n]) for n in original_qv}==qv_ids and tensor_digest(sorted((n,params[n]) for n in original_qv))==qv_digest
        modules={n:m for n,m in model.named_modules() if isinstance(m,methods.SeqLinear)}
        assert len(modules)==168 and all(m.scale==4 and m.dropout.p==.1 for m in modules.values())
        assert all(bool((m.B.weight==0).all()) for n,m in modules.items() if n.rsplit('.',1)[-1] not in study.QV)
        for p in params.values():p.requires_grad_(True);p.grad=None
        with torch.no_grad():sanity_after=float(loss_value(batches[0],rows[0])[0])
        assert sanity_before==sanity_after
        rt.restore_rng(saved_rng);model.train();method.begin_stage(2)
        initial={n:p.detach().clone() for n,p in params.items()}
        assert tensor_digest(sorted(initial.items()))==expected['common_initial_AB_digest'],'Common initial AB reproduction failed'
        frozen_digest=tensor_digest((n,p) for n,p in model.core.named_parameters(remove_duplicate=False) if not n.endswith(('.A.weight','.B.weight')))
        optimizer=rt.make_optimizer(method);assert type(optimizer) is torch.optim.AdamW
        scaler=torch.amp.GradScaler('cuda',init_scale=1,enabled=True);optimizer.zero_grad(set_to_none=True)
        common_rng=rt.rng();before=[];target_tokens=0
        assert rng_digest(common_rng)==expected['common_dropout_RNG_start_sha256'],'Common RNG start reproduction failed'
        for batch,row in zip(batches,rows):
            loss,length=loss_value(batch,row);assert torch.isfinite(loss)
            scaler.scale(loss/11).backward();before.append(float(loss.detach()));target_tokens+=length
        scaler.unscale_(optimizer)
        gradients={n:p.grad.detach().clone() for n,p in params.items()}
        assert all(torch.isfinite(g).all() for g in gradients.values())
        common_end_rng=rt.rng();del optimizer,scaler,loss
        assert rng_digest(common_end_rng)==expected['common_dropout_RNG_end_sha256'],'Common RNG end reproduction failed'
        arms={}
        for arm in ['qv','all']:
            for n,p in params.items():
                with torch.no_grad():p.copy_(initial[n])
                p.grad=None;p.requires_grad_(arm=='all' or n in original_qv)
            assert tensor_digest(sorted(params.items()))==tensor_digest(sorted(initial.items()))
            active={n:p for n,p in params.items() if p.requires_grad}
            optimizer=rt.make_optimizer(method);assert type(optimizer) is torch.optim.AdamW
            scheduler=get_cosine_schedule_with_warmup(optimizer,0,1)
            scaler=torch.amp.GradScaler('cuda',init_scale=1,enabled=True)
            scaler.scale(torch.zeros((),device=model.device)) # initialize native scale before assigning the common scaled gradients
            for n,p in active.items():p.grad=gradients[n].clone()*scaler.get_scale()
            scaler.unscale_(optimizer)
            grad=torch.nn.utils.clip_grad_norm_(list(active.values()),1.);assert torch.isfinite(grad)
            postclip=torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm(2) for p in active.values()]),2)
            lr_used=[g['lr'] for g in optimizer.param_groups];scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=scale
            scheduler.step()
            assert all(float(v['step'])==1 for v in optimizer.state.values())
            assert all(v.is_cuda for st in optimizer.state.values() for k,v in st.items() if k!='step')
            groups={};per_projection=[]
            previous_tf32=torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    for name,layer in modules.items():
                        keys=[name+'.A.weight',name+'.B.weight'];kind=name.rsplit('.',1)[-1]
                        dot=sum((gradients[k].double()*(params[k].detach()-initial[k]).double()).sum() for k in keys)
                        psq=sum((params[k].detach()-initial[k]).double().square().sum() for k in keys)
                        gsq=sum(gradients[k].double().square().sum() for k in keys)
                        dense=4*(layer.B.weight.float()@layer.A.weight.float()-initial[keys[1]].float()@initial[keys[0]].float())
                        dsq=dense.double().square().sum()
                        rec=dict(module=name,active=any(k in active for k in keys),raw_gradient_dot_parameter_update=float(dot),
                            parameter_update_squared_norm=float(psq),common_parameter_gradient_squared_norm=float(gsq),effective_dense_update_squared_norm=float(dsq))
                        per_projection.append(rec);g=groups.setdefault(kind,dict(raw_gradient_dot_parameter_update=0.,parameter_update_squared_norm=0.,common_parameter_gradient_squared_norm=0.,effective_dense_update_squared_norm=0.))
                        for k in g:g[k]+=rec[k]
                        del dense,dot,psq,gsq,dsq
            finally:torch.backends.cuda.matmul.allow_tf32=previous_tf32
            for g in groups.values():
                for k in ['parameter_update','common_parameter_gradient','effective_dense_update']:g[k+'_norm']=math.sqrt(g[k+'_squared_norm'])
            updated_digest=tensor_digest(sorted(params.items()))
            assert updated_digest==expected['arms'][arm]['adapter_after_digest'],arm+' Adam endpoint reproduction failed'
            rt.restore_rng(common_rng);model.train();after=[]
            with torch.no_grad():
                for batch,row in zip(batches,rows):
                    value,_=loss_value(batch,row);assert torch.isfinite(value);after.append(float(value))
            assert rng_digest(rt.rng())==rng_digest(common_end_rng),'Dropout RNG replay differs'
            assert tensor_digest(sorted(params.items()))==updated_digest
            assert tensor_digest((n,p) for n,p in model.core.named_parameters(remove_duplicate=False) if not n.endswith(('.A.weight','.B.weight')))==frozen_digest
            arms[arm]=dict(active_parameters=len(active),active_coordinates=sum(p.numel() for p in active.values()),
                before_AnswerCE=sum(before)/11,after_AnswerCE=sum(after)/11,CE_change=(sum(after)-sum(before))/11,
                per_QA_before=before,per_QA_after=after,raw_gradient_dot_parameter_update=sum(v['raw_gradient_dot_parameter_update'] for v in per_projection),
                parameter_update_norm=math.sqrt(sum(v['parameter_update_squared_norm'] for v in per_projection)),
                effective_dense_update_norm=math.sqrt(sum(v['effective_dense_update_squared_norm'] for v in per_projection)),
                projection_groups=groups,per_projection=per_projection,
                gradient_l2_preclip=float(grad),gradient_l2_postclip=float(postclip),clip_threshold=1.,
                nominal_clip_coefficient=min(1.,1./(float(grad)+1e-6)),learning_rate_used=lr_used,
                learning_rate_after_scheduler=scheduler.get_last_lr(),scaler_before=scale,scaler_after=scaler.get_scale(),
                optimizer_class='torch.optim.AdamW',optimizer_steps=1,optimizer_moments_fresh=True,
                actual_parameters_changed=updated_digest!=tensor_digest(sorted(initial.items())),same_dropout_RNG_replay=True,
                base_parameters_unchanged=True,adapter_after_digest=updated_digest)
            del optimizer,scheduler,scaler
        # Both original endpoints are reproduced before selecting or reading the new arm.
        deltas={n:p.detach().clone()-initial[n] for n,p in params.items()}
        a2=a3=a4=0.;previous_tf32=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                for name,layer in modules.items():
                    ka,kb=name+'.A.weight',name+'.B.weight'
                    u=4*(deltas[kb].float()@initial[ka].float()+initial[kb].float()@deltas[ka].float())
                    v=4*(deltas[kb].float()@deltas[ka].float())
                    a2+=float(u.double().square().sum());a3+=2*float((u.double()*v.double()).sum());a4+=float(v.double().square().sum())
                    del u,v
        finally:torch.backends.cuda.matmul.allow_tf32=previous_tf32
        target=arms['qv']['effective_dense_update_norm']
        alpha,selection=smallest_positive_root(a2,a3,a4,target)
        with torch.no_grad():
            for n,p in params.items():p.copy_(initial[n]+alpha*deltas[n])
        total_dense_sq=total_parameter_sq=raw_dot=0.;matched_projection=[]
        previous_tf32=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=False
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                for name,layer in modules.items():
                    ka,kb=name+'.A.weight',name+'.B.weight'
                    dense=4*(layer.B.weight.float()@layer.A.weight.float()-initial[kb].float()@initial[ka].float())
                    dsq=float(dense.double().square().sum());total_dense_sq+=dsq
                    psq=sum(float((params[k].detach()-initial[k]).double().square().sum()) for k in [ka,kb]);total_parameter_sq+=psq
                    dot=sum(float((gradients[k].double()*(params[k].detach()-initial[k]).double()).sum()) for k in [ka,kb]);raw_dot+=dot
                    matched_projection.append(dict(module=name,effective_dense_update_squared_norm=dsq,
                        parameter_update_squared_norm=psq,raw_gradient_dot_parameter_update=dot))
                    del dense
        finally:torch.backends.cuda.matmul.allow_tf32=previous_tf32
        actual=math.sqrt(total_dense_sq);error=abs(actual-target);tolerance=1e-3*target+1e-7
        matched_digest=tensor_digest(sorted(params.items()))
        selection.update(alpha=alpha,actual_FP32_dense_norm=actual,absolute_residual=error,
            relative_residual=error/target,tolerance_relative=1e-3,tolerance_absolute=1e-7,tolerance_combined=tolerance,
            norm_tolerance_passed=error<=tolerance,matched_adapter_digest=matched_digest,
            expected_result_sha256=EXPECTED_RESULT_SHA,old_initial_AB_and_RNG_and_both_endpoints_reproduced=True,
            expected_qv_dense_norm=expected['arms']['qv']['effective_dense_update_norm'],
            polynomial_norm_at_one=math.sqrt(max(0.,a2+a3+a4)),reconstructed_all_native_dense_norm=arms['all']['effective_dense_update_norm'],
            selected_before_matched_loss=True,parameter_ray_preserved_up_to_FP32_rounding=True,dense_direction_preservation_claimed=False)
        write('NORM_SELECTION.json',selection)
        assert error<=tolerance,'Actual FP32 dense norm did not match predetermined tolerance'
        rt.restore_rng(common_rng);model.train();matched_losses=[]
        with torch.no_grad():
            for batch,row in zip(batches,rows):
                value,_=loss_value(batch,row);assert torch.isfinite(value);matched_losses.append(float(value))
        assert rng_digest(rt.rng())==rng_digest(common_end_rng)
        assert tensor_digest(sorted(params.items()))==matched_digest
        assert tensor_digest((n,p) for n,p in model.core.named_parameters(remove_duplicate=False) if not n.endswith(('.A.weight','.B.weight')))==frozen_digest
        arms['all_normmatched']=dict(before_AnswerCE=sum(before)/11,after_AnswerCE=sum(matched_losses)/11,
            CE_change=(sum(matched_losses)-sum(before))/11,per_QA_before=before,per_QA_after=matched_losses,
            effective_dense_update_norm=actual,parameter_update_norm=math.sqrt(total_parameter_sq),
            raw_gradient_dot_parameter_update=raw_dot,per_projection=matched_projection,alpha=alpha,
            active_coordinates=4399104,optimizer_steps=0,source='Interpolation of reconstructed all Adam endpoint, not an extra optimizer step',
            same_dropout_RNG_replay=True,base_parameters_unchanged=True,adapter_after_digest=matched_digest)
        write('RESULT.json',dict(schema=SCHEMA,status='ONE_PREDECLARED_DENSE_NORM_MATCHED_PARAMETER_RAY_OBSERVED',
            expected_result_sha256=EXPECTED_RESULT_SHA,derived_original_worker_sha256=ORIGINAL_WORKER_SHA,
            norm_selection=selection,norm_selection_sha256=sha(out/'NORM_SELECTION.json'),stage=2,TRAIN_QA=11,
            source_checkpoint_sha256=CP_SHA,source_config_sha256=CONFIG_SHA,source_snapshot=config['execution_source_sha256'],
            all_r8_registration_source_sha256=STUDY_SHA,input_records=input_info,target_tokens=target_tokens,
            original_qv_AB_digest=qv_digest,common_initial_AB_digest=tensor_digest(sorted(initial.items())),
            B0_extra_modules=120,shared_function_sanity=dict(QA=1,mode='eval',before_AnswerCE=sanity_before,after_AnswerCE=sanity_after),
            common_gradient_scope='all-r8 graph; per-QA answer-token mean then QA-equal mean; 11 backwards, no common optimizer step',
            same_function_at_start='Original q/v and base preserved; all extra branches B0. Both masks use exactly the same graph and shared gradients.',
            common_dropout_RNG_start_sha256=rng_digest(common_rng),common_dropout_RNG_end_sha256=rng_digest(common_end_rng),
            arms=arms,new_optimizer_steps=2,new_backward_calls=11,new_answer_loss_forwards=46,
            TRAIN_QA_used=11,TEST_QA=0,new_generations=0,new_F1_scores=0,new_checkpoint_saved=False,
            effective_dense_norm_definition='FP32 scale*(B_after A_after-B_before A_before), TF32 disabled for measurement only; not merged into W',
            limits=['One predetermined current task and one shared gradient, not a convergence curve or generalization result.',
                    'Common dummy extra branches consume identical RNG/computation in both masks; not a replay of the original q/v-only trajectory.',
                    'The added arm matches one global undropped dense update norm, not per-layer allocation or functional distance.',
                    'The Adam A/B parameter ray is scaled; dense direction may rotate due to the quadratic factor term. Not pure direction causality.',
                    'Replay uses the same dropout realization, not an independent fresh-mask loss or held-out QA.',
                    'Base is frozen and initial q/v already learned at S1; do not apply B0/A-zero-gradient arguments to that q/v state.']))
        write('COMPLETE.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),GPU=1,binding_sha256=sha(a.binding),
            worker_sha256=sha(__file__),result_sha256=sha(out/'RESULT.json'),elapsed_seconds=time.monotonic()-tick,
            TEST_QA=0,new_generations=0,new_F1_scores=0,new_optimizer_steps=2,new_backward_calls=11))
    except BaseException as e:
        write('FAILURE.json',dict(schema=SCHEMA,pid=os.getpid(),unix=time.time(),error_type=type(e).__name__,error=str(e),
            elapsed_seconds=time.monotonic()-tick,no_automatic_retry=True));raise
    finally:signal.alarm(0)


if __name__=='__main__':main()
