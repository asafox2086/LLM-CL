"""Single fresh GPU1 diagnostic. No continual-language training or generation."""
import gc, importlib, importlib.util, json, math, os, random, signal, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace
from core import *

class ResourceBusy(RuntimeError):pass

def main():
    integrity_begin=time.monotonic()
    ready=json.loads((REPORT/'READY.json').read_text())
    for key in ('sources','dependencies','evidence','base_files'):
        for name,h in ready[key].items():assert sha(ROOT/name)==h,name
    task,calibration,diagnostic,ledger=data();assert ledger==ready['data']
    sys.path.insert(0,str(ROOT/'analysis'));import onevision_research_device_entry as device
    cards=device.inventory();ours={g for g,c in cards.items() if any(p['uid']==os.getuid() for p in c['owners'])}
    if cards[1]['owners'] or cards[1]['memory_MiB']>=500 or cards[1]['utilization']>=10 or ours&{0,1} or len(ours|{1})>3:
        raise ResourceBusy('Before STARTED/CUDA: changed resource, no model attempt')
    assert not WORK.exists() and not any((REPORT/n).exists() for n in ('STARTED.json','FAILURE.json','COMPLETE.json'))
    begin=time.monotonic()
    write(REPORT/'STARTED.json',dict(pid=os.getpid(),startticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),UTC=time.time(),monotonic_start=begin,
        preSTARTED_integrity_seconds=begin-integrity_begin,ready_sha256=sha(REPORT/'READY.json'),budget_seconds=BUDGET))
    def expired(*_):raise TimeoutError('Frozen 1800-second STARTED budget; stop without retry')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.001,BUDGET-(time.monotonic()-begin)))
    os.environ['CUDA_VISIBLE_DEVICES']='1';os.environ['OMP_NUM_THREADS']='8';os.environ['MKL_NUM_THREADS']='8'
    import torch,numpy as np
    from transformers import AutoProcessor,get_cosine_schedule_with_warmup
    assert str(Path(sys.executable).resolve())==ready['runtime_interpreter']
    assert {n:str(Path(importlib.import_module(n).__file__).resolve()) for n in ready['runtime_source_paths']}==ready['runtime_source_paths']
    sys.path.insert(0,str(PORT))
    spec=importlib.util.spec_from_file_location('visual_function_formal',PORT/'run.py')
    formal=importlib.util.module_from_spec(spec);spec.loader.exec_module(formal)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(8)
    WORK.mkdir();monitor=formal.ResourceMonitor(WORK);monitor.snapshot('before_model')
    base=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor=AutoProcessor.from_pretrained(base,local_files_only=True)
    model=formal.onevision_bridge.Factory.from_pretrained(base,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    formal.onevision_bridge.bind(model)
    student=make_student(model).cuda().eval()
    assert sum(p.numel() for p in student.parameters())==ready['student_training_coordinates']
    write(WORK/'STUDENT_LAYOUT.json',{n:dict(shape=list(p.shape),coordinates=p.numel(),dtype=str(p.dtype)) for n,p in student.named_parameters()})
    assert count(model)['coefficients']==BASE
    frozen_before=digest(model.parameters());initial_student=digest(student.parameters())
    formal.install(model);formal.onevision_bridge.NATIVE_GATE_MODE=True
    def batch_for(row):
        batch,details=formal.input_batch(row,task,processor,model.device)
        raw=formal.onevision_bridge.LAST_NATIVE
        assert raw is not None and raw['pixel_values'] is not None
        return batch,raw['pixel_values'],raw['image_sizes'],details
    @torch.no_grad()
    def observe(label,reference=None):
        records=[]
        for row in diagnostic:
            batch,pixels,sizes,details=batch_for(row)
            with torch.autocast('cuda',dtype=torch.float16):
                s=extract_student(model,student,pixels,sizes)
                error=float(normalized_error(s,batch['pixel_values']))
                if reference is None:parent,_=formal.answer_loss(model,batch,row['answer'],processor)
                else:parent=reference[len(records)]['parent_NLL']
                candidate={**batch,'pixel_values':s}
                small,_=formal.answer_loss(model,candidate,row['answer'],processor)
            rec=dict(id=row['id'],parent_NLL=float(parent),student_NLL=float(small),feature_error=error,
                input_tensors=tensor_digest({k:v for k,v in batch.items() if k!='pixel_values'}),
                raw_pixels=tensor_digest({'pixels':pixels,'sizes':sizes}),details=details)
            assert all(math.isfinite(rec[k]) for k in ('parent_NLL','student_NLL','feature_error'))
            records.append(rec)
        write(WORK/(label+'.json'),records);return records
    before=observe('before');monitor.snapshot('training_before',model=model)
    state_scope=dict(parent_in_model_field=count(model),student_before_KD=count(student),
        model_parameter_bytes_scope='ResourceMonitor model field counts frozen full parent only; student is independently registered outside it during KD',
        actual_process_RAM_and_CUDA_include_student=True,not_sum_of_parameter_bytes_as_peak_RAM=True)
    # FP32 is the fixed training master; both diagnostic boundaries use FP16.
    student.float().train();params=list(student.parameters())
    for p in params:p.requires_grad_(True)
    state_scope['student_during_KD']=count(student)
    assert all(p.dtype==torch.float32 for p in params)
    from host_adam import HostAdamW
    optimizer=HostAdamW(params,lr=1e-4,weight_decay=0.,eps=1e-8,betas=(.9,.999))
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=formal.warmup_updates(17),num_training_steps=17)
    scaler=torch.amp.GradScaler('cuda',init_scale=1.);events=[]
    for offset in range(0,len(calibration),16):
        rows=calibration[offset:offset+16];optimizer.zero_grad(set_to_none=True);losses=[];input_details=[]
        for row in rows:
            batch,pixels,sizes,details=batch_for(row);input_details.append(details)
            with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast('cuda',dtype=torch.float16):
                packed=extract_student(model,student,pixels,sizes)
                loss=normalized_error(packed,batch['pixel_values'])
            assert torch.isfinite(loss);scaler.scale(loss/len(rows)).backward();losses.append(float(loss))
        scaler.unscale_(optimizer)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
        lr=[g['lr'] for g in optimizer.param_groups];assert all(x>0 and math.isfinite(x) for x in lr)
        oldscale=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=oldscale
        witness=dict(parameter_IDs_match={id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in params},
            coordinates=sum(p.numel() for p in params),gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in params),
            moment_bytes=sum(v.numel()*v.element_size() for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')),
            moment_devices=sorted({str(v.device) for s in optimizer.state.values() for k,v in s.items() if k in ('exp_avg','exp_avg_sq')}))
        assert witness['parameter_IDs_match'] and witness['moment_devices']==['cpu']
        event=dict(step=len(events)+1,QA_ids=[r['id'] for r in rows],mean_normalized_MSE=math.fsum(losses)/len(losses),
            raw_gradient_norm=norm,learning_rate_used=lr,optimizer=witness,input_details=input_details)
        assert math.isfinite(norm);write(WORK/'updates'/f'{len(events)+1:02d}.json',event);events.append(event);scheduler.step()
    assert len(events)==17 and sum(len(e['QA_ids']) for e in events)==260
    del optimizer,scheduler,scaler,params,packed,loss,batch,pixels,sizes
    student.half().eval()
    for p in student.parameters():p.requires_grad_(False)
    state_scope['student_after_KD']=count(student);write(WORK/'TRAIN_STATE_SCOPE.json',state_scope)
    assert digest(student.parameters())!=initial_student
    monitor.snapshot('training_after',model=model)
    after=observe('after',before)
    assert digest(model.parameters())==frozen_before
    assert all(a['input_tensors']==b['input_tensors'] and a['raw_pixels']==b['raw_pixels'] and a['parent_NLL']==b['parent_NLL'] for a,b in zip(before,after))
    cleanup_compact(model,student);del student
    formal.onevision_bridge.NATIVE_GATE_MODE=False;formal.onevision_bridge.LAST_NATIVE=None
    layout=count(model);assert layout['coefficients']<=.8*BASE and layout['trainable']==0
    model.eval();gc.collect();torch.cuda.empty_cache()
    config=dict(method='VisualTailFunctionProbe',manifest_sha256=ledger['manifest_sha256'],source_sha256=sha(__file__),
        ready_sha256=sha(REPORT/'READY.json'),diagnostic_only=True,language_updates=0,Answer_F1_measured=False)
    method=SimpleNamespace(model=model,name='VisualTailFunctionProbe')
    cp=formal.save_checkpoint(method,processor,WORK,0,config)
    final_digest=digest(model.parameters())
    recovered=formal.load_checkpoint(WORK)['method'].model
    assert count(recovered)==layout and len(teacher_layers(recovered))==14 and digest(recovered.parameters())==final_digest
    memory=monitor.finish(model)
    result=outcome(before,after,layout)
    all_details=[r['details'] for r in before+after]+[d for e in events for d in e['input_details']]
    teacher_cost=dict(visual_requests=len(all_details),actual_extractor_calls=sum(not d['visual_cache_hit'] for d in all_details),
        cache_hits=sum(bool(d['visual_cache_hit']) for d in all_details),
        scope='Feature extractor invocations, not independent images/patches/FLOPs; per-request shapes retained')
    assert time.monotonic()-begin<=BUDGET
    files={p.relative_to(ROOT).as_posix():sha(p) for p in [WORK/'TRAIN_STATE_SCOPE.json',WORK/'STUDENT_LAYOUT.json',WORK/'before.json',WORK/'after.json',WORK/'resources.jsonl',WORK/'memory_summary.json',*sorted((WORK/'updates').glob('*.json'))]}
    write(REPORT/'COMPLETE.json',dict(complete=True,result=result,checkpoint=cp,full_graph_reload=True,
        original_parent_digest_before=frozen_before,original_parent_digest_after=frozen_before,compact_digest=final_digest,
        model_RUN_seconds=time.monotonic()-begin,extra_KD_QA=260,diagnostic_NLL_forwards=195,
        diagnostic_note='Parent NLL evaluated once; after records bind same raw/input SHA and unchanged parent digest; 3x65 NLL',
        extra_student_feature_diagnostic_forwards=130,language_updates=0,DEV_QA=0,TEST_QA=0,
        training_updates=17,teacher_feature_cost=teacher_cost,file_sha256=files,memory=memory,ready_sha256=sha(REPORT/'READY.json')))
    signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    try:main()
    except ResourceBusy as e:print(str(e),flush=True);sys.exit(75)
    except Exception:
        write(REPORT/'FAILURE.json',dict(traceback=traceback.format_exc(),no_retry_or_extension=True));raise
