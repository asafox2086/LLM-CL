"""Sequential open-answer Qwen2-VL experiment; real images and full stage recovery."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import random
import re
import time
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration,get_cosine_schedule_with_warmup
from methods import ROOT,make_method as Method
from prepare_assets import retrieve
from stage_checkpoint import save_stage,load_stage


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def tokens(text):
    text=text.lower()
    return re.findall(r'[a-z0-9]+|[\u3400-\u9fff]',text)


def score(prediction,references):
    a=tokens(prediction)
    values=[]
    for reference in references:
        b=tokens(reference)
        exact=float(a==b)
        if not a or not b:
            values.append({'exact_match':100*exact,'token_f1':100*exact,'rougeL':100*exact})
            continue
        common=sum((Counter(a)&Counter(b)).values())
        f1=2*common/(len(a)+len(b))
        masks={}
        for index,word in enumerate(b):
            masks[word]=masks.get(word,0)|(1<<index)
        state=0
        for word in a:
            union=masks.get(word,0)|state
            state=union&~(union-((state<<1)|1))
        values.append({'exact_match':100*exact,'token_f1':100*f1,'rougeL':200*state.bit_count()/(len(a)+len(b))})
    return {name:max(value[name] for value in values) for name in values[0]}


def input_batch(row,task,processor,device):
    context=row.get('context','')
    retrieval_indexes=[]
    if len(context)>6000:
        chunks=[context[start:start+1000] for start in range(0,len(context),800)]
        retrieval_indexes=sorted(retrieve(row['question'],chunks,5))
        context='\n'.join(chunks[index] for index in retrieval_indexes)
    content=[]
    images=[]
    for name in row.get('images',[]):
        with Image.open(ROOT/name) as image:
            images.append(image.convert('RGB'))
        content.append({'type':'image'})
    text=task['instruction']
    if context:
        text+='\nContext:\n'+context
    text+='\nQuestion: '+row['question']
    content.append({'type':'text','text':text})
    prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
    # Both methods preserve the author's effective frozen-visual setting.
    total_visual=512 if len(images)==1 else 1024
    pixel_budget=max(16,total_visual//max(1,len(images)))*28*28
    # Transformers 4.46 reads these limits from the image processor;
    # passing them to the combined processor forwards them to the tokenizer.
    processor.image_processor.min_pixels=4*28*28
    processor.image_processor.max_pixels=pixel_budget
    processor.image_processor.size={'min_pixels':4*28*28,'max_pixels':pixel_budget}
    batch=processor(text=[prompt],images=images or None,return_tensors='pt')
    if batch['input_ids'].shape[1]>2304:
        raise ValueError('prompt over budget; do not truncate image placeholder sequences silently')
    batch={key:value.to(device) for key,value in batch.items()}
    return batch,{'prompt_tokens':int(batch['input_ids'].shape[1]),'retrieved_chunk_indexes':retrieval_indexes}


def answer_loss(model,batch,answer,processor):
    target=processor.tokenizer(answer,add_special_tokens=False,truncation=True,max_length=512)['input_ids']
    if not target:
        raise ValueError('empty answer tokens')
    target.append(processor.tokenizer.eos_token_id)
    prompt_length=batch['input_ids'].shape[1]
    target_tensor=torch.tensor([target],device=batch['input_ids'].device)
    ids=torch.cat([batch['input_ids'],target_tensor],dim=1)
    mask=torch.ones_like(ids)
    embeddings=model.get_input_embeddings()(ids)
    grid=batch.get('image_grid_thw')
    if 'pixel_values' in batch:
        visual=model.visual(batch['pixel_values'].to(model.visual.get_dtype()),grid_thw=grid)
        image_mask=(ids==model.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
        if image_mask.sum().item()!=visual.numel():
            raise ValueError('image feature/token count mismatch')
        embeddings=embeddings.masked_scatter(image_mask,visual.to(embeddings.dtype))
    positions,_=model.get_rope_index(ids,grid,attention_mask=mask)
    output=model.model(inputs_embeds=embeddings,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True)
    hidden=output.last_hidden_state[:,prompt_length-1:-1,:]
    logits=model.lm_head(hidden).float()
    return torch.nn.functional.cross_entropy(logits.flatten(0,1),target_tensor.flatten()),len(target)


def evaluate(method,processor,tasks,data,run,stage,split='test',only_task=None):
    method.model.eval()
    scores=[]
    limit={1:128,2:64,3:512,4:64,5:256,6:256,7:128,8:128,9:64,10:64}
    for task in tasks:
        task_id=task['task']
        if only_task is not None and task_id!=only_task:
            continue
        method.set_instruction(task_id)
        rows=data[(task_id,split)]
        output=run/'evaluations'/f'stage_{stage:02d}'/f'T{task_id:02d}_{split}_predictions.jsonl'
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix('.jsonl.partial')
        existing=[]
        prior=output if output.exists() else temporary
        if prior.exists():
            for line in prior.read_text().splitlines():
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            if [row['id'] for row in existing]!=[row['id'] for row in rows[:len(existing)]]:
                raise ValueError('prediction resume manifest mismatch')
        if not output.exists():
            temporary.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in existing))
            with temporary.open('a') as stream:
                for row in rows[len(existing):]:
                    batch,details=input_batch(row,task,processor,method.model.device)
                    if hasattr(method,'vision_original'):
                        from frozen_vision import prepare
                        details['visual_cache_hit']=prepare(batch,row,method.model,method.vision_original)
                    started=time.monotonic()
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                        generated=method.model.generate(**batch,do_sample=False,num_beams=1,
                            max_new_tokens=limit[task_id],use_cache=True,
                            pad_token_id=processor.tokenizer.pad_token_id,eos_token_id=processor.tokenizer.eos_token_id)
                    response=generated[0,batch['input_ids'].shape[1]:]
                    prediction=processor.tokenizer.decode(response,skip_special_tokens=True)
                    record={'id':row['id'],'dataset':row.get('dataset',row['id'].split(':',1)[0]),
                            'prediction':prediction,'references':row['answers'],
                            'scores':score(prediction,row['answers']),'generation_seconds':time.monotonic()-started,
                            'output_tokens':len(response),'generation_cap':limit[task_id],**details}
                    existing.append(record)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
                    stream.flush()
                    if len(existing)%20==0:
                        print(json.dumps({'event':'evaluation','stage':stage,'task':task_id,'split':split,'done':len(existing),'total':len(rows)}),flush=True)
                os.fsync(stream.fileno())
            temporary.replace(output)
        if len(existing)!=len(rows):
            raise ValueError('incomplete predictions')
        result={name:sum(row['scores'][name] for row in existing)/len(existing)
                for name in ['rougeL','token_f1','exact_match']}
        result['count']=len(existing)
        write(output.with_suffix('.scores.json'),result)
        scores.append(result['rougeL'])
    return scores


def rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def make_optimizer(method):
    groups={}
    for name,parameter in method.model.named_parameters():
        if not parameter.requires_grad:
            continue
        decay=method.name=='SMoLoRA' and parameter.ndim>=2 and not name.endswith('.bias')
        key=(1e-4,.01 if decay else 0.)
        groups.setdefault(key,[]).append(parameter)
    from host_adam import HostAdamW
    optimizer=HostAdamW if method.name=='PASs_MoE' else torch.optim.AdamW
    return optimizer([{'params':parameters,'lr':lr,'weight_decay':decay}
                              for (lr,decay),parameters in groups.items()],eps=1e-8,betas=(.9,.999))


def activation_storage(method):
    # Keep the original expert computation and parameter precision. Transfer saved
    # autograd tensors to host RAM until backward instead of retaining them on GPU.
    return torch.autograd.graph.save_on_cpu(pin_memory=True) if method.name=='SMoLoRA' else nullcontext()


def save_last(path,method,optimizer,scheduler,scaler,stage,offset,step,manifest_sha):
    temporary=path.with_suffix('.pt.tmp')
    torch.save({'weights':method.changed_state(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'scaler':scaler.state_dict(),'method_state':method.method_state(stage),'rng':rng(),
                'stage':stage,'next_offset':offset,'global_step':step,'manifest_sha':manifest_sha},temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--method',choices=['SeqLoRA','SMoLoRA','PASs_MoE'],required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--save-checkpoints',action='store_true',help='Opt in; results-only is the default')
    args=parser.parse_args()
    run=args.run_dir.resolve()
    run.mkdir(parents=True,exist_ok=True)
    (run/'logs').mkdir(exist_ok=True)
    policy=json.loads((Path(__file__).parent/'execution_policy.json').read_text())
    if policy.get('training_enabled',True) is False and not args.preflight_only:
        raise RuntimeError('Training disabled pending user-requested domain/image dataset revision')
    manifest_path=ROOT/policy['manifest']
    manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest=json.loads(manifest_path.read_text())
    if not manifest.get('formal_ready') or len(manifest['tasks'])!=10:
        raise ValueError('ten-task data not ready')
    data={}
    for task in manifest['tasks']:
        for split,info in task['files'].items():
            path=ROOT/info['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
                raise ValueError('source view checksum mismatch')
            data[(task['task'],split)]=[json.loads(line) for line in path.open()]
            if len(data[(task['task'],split)])!=manifest['counts_per_task'][split]:
                raise ValueError('unequal task count')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda:0')
    model_path=ROOT/'model/Qwen2-VL-2B-Instruct'
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
    model=Qwen2VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.float16,
        attn_implementation='sdpa',local_files_only=True).to(device)
    method=Method(model,args.method,[task['instruction'] for task in manifest['tasks']])
    if hasattr(method,'state_root'):
        method.state_root=run/'method_state'
        method.save_checkpoints=args.save_checkpoints
    method.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    method.model.enable_input_require_grads()
    if args.preflight_only:
        # from_pretrained defaults to eval mode; checkpointing is active only
        # in train mode. Preflight must exercise the actual training memory path.
        model.train()
        method.begin_stage(1)
        optimizer=make_optimizer(method)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        losses=[]
        print(json.dumps({'event':'gpu_preflight_start','model_training':model.training,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'allocated_mib':torch.cuda.memory_allocated()/1024**2}),flush=True)
        for task_id in [1,2,3,5,6,8]:
            method.set_instruction(task_id)
            optimizer.zero_grad(set_to_none=True)
            row=max(data[(task_id,'train')],key=lambda value:len(value.get('context','')[:6000])
                    +len(value['answer'][:4000])+len(value.get('images',[]))*1000)
            batch,_=input_batch(row,manifest['tasks'][task_id-1],processor,device)
            if hasattr(method,'recording'):
                method.recording(True)
            with activation_storage(method),torch.autocast('cuda',dtype=torch.float16):
                loss,_=answer_loss(model,batch,row['answer'],processor)
            if hasattr(method,'recording'):
                method.recording(False)
            if not torch.isfinite(loss):
                raise FloatingPointError('GPU preflight loss')
            scaler.scale(loss).backward()
            if hasattr(method,'post_backward'):
                method.post_backward(scaler.get_scale(),.5)
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('GPU preflight gradient')
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            print(json.dumps({'event':'gpu_preflight_case','task':task_id,'loss':losses[-1],
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}),flush=True)
        write(run/'gpu_preflight.json',{'method':args.method,'tasks_checked':[1,2,3,5,6,8],'losses':losses,
              'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'passed':True,
              'formal_training_started':False,'preflight_weights_discarded':True})
        print('GPU_PREFLIGHT_PASSED',args.method,flush=True)
        return
    from frozen_vision import install
    method.vision_original=install(model)
    config={'method':args.method,'seed':args.seed,'base_model':str(model_path),'manifest_sha256':manifest_sha,
            'batch_size':16,'micro_batch_size':1,'epochs':1,'learning_rate':1e-4,'target_token_cap':512,
            'counts_per_task':manifest['counts_per_task'],
            'primary_score':'ROUGE-L F1, 0..100; generated-answer comparison, not original benchmark leaderboard metric',
            'smolora_upstream_revision':method.original_revision,'training_started':True,
            'saved_autograd_tensors_on_cpu':args.method in ['SMoLoRA','PASs_MoE'],
            'optimizer_moments_on_cpu':args.method=='PASs_MoE',
            'save_checkpoints':args.save_checkpoints,
            'execution_source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                for name in ['run.py','methods.py','frozen_vision.py','stage_checkpoint.py']+
                    (['pass_adapter.py','host_adam.py'] if args.method=='PASs_MoE' else [])}}
    write(run/'config.json',config)
    write(run/'environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
          'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'precision':'fp16 base, fp32 adapters'})
    matrix=[]
    if manifest.get('initial_evaluation_cache') and args.method in ['SeqLoRA','SMoLoRA','PASs_MoE']:
        # Only the untrained base can be reused, never a trained-stage result.
        zero=[value for key,value in model.state_dict().items() if '.B.weight' in key or '.lora_B.' in key]
        if not zero or any(torch.count_nonzero(value).item() for value in zero):
            raise ValueError('Initial frozen-base cache requires exactly zero LoRA B weights')
        cache=ROOT/manifest['initial_evaluation_cache']
        proof=json.loads((cache.parent/'initial_cache_proof.json').read_text())
        if proof['model_asset_manifest_sha256']!=hashlib.sha256((model_path/'asset_manifest.json').read_bytes()).hexdigest():
            raise ValueError('Initial evaluation base model changed')
        destination=run/'evaluations/stage_00'
        destination.mkdir(parents=True,exist_ok=True)
        for item in proof['tasks']:
            name=f'T{item["task"]:02d}_test_predictions.jsonl'
            if hashlib.sha256((cache/name).read_bytes()).hexdigest()!=item['cache_sha256']:
                raise ValueError('Frozen initial evaluation cache changed')
            if not (destination/name).exists():
                shutil.copy2(cache/name,destination/name)
        write(run/'initial_evaluation_reuse.json',proof)
    initial=evaluate(method,processor,manifest['tasks'],data,run,0)
    matrix.append(initial)
    write(run/'score_matrix.json',matrix)
    from collect_report import collect
    collect()
    for stage in range(1,11):
        checkpoint=run/'checkpoints'/f'stage_{stage:02d}'
        if checkpoint.exists() and args.save_checkpoints:
            metadata,_=load_stage(checkpoint,method.restore)
            if metadata['manifest_sha256']!=manifest_sha:
                raise ValueError('stage manifest changed')
            evaluation=json.loads((checkpoint/'evaluation.json').read_text())
            matrix.append(evaluation['task_scores'])
            continue
        method.begin_stage(stage)
        parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer=make_optimizer(method)
        updates=math.ceil(len(data[(stage,'train')])/16)
        scheduler=get_cosine_schedule_with_warmup(optimizer,math.ceil(.03*updates),updates)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024)
        rows=data[(stage,'train')]
        order=list(range(len(rows)))
        random.Random(args.seed+stage).shuffle(order)
        last=run/'checkpoints'/f'stage_{stage:02d}_last.pt'
        offset,step=0,0
        if last.exists():
            state=torch.load(last,map_location='cpu',weights_only=False)
            if state['manifest_sha']!=manifest_sha or state['stage']!=stage:
                raise ValueError('last checkpoint protocol mismatch')
            model.load_state_dict(state['weights'],strict=False)
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            scaler.load_state_dict(state['scaler'])
            restore_rng(state['rng'])
            if hasattr(method,'restore_method_state'):
                method.restore_method_state(state['method_state'])
            offset,step=state['next_offset'],state['global_step']
            if not args.save_checkpoints:
                # One-time migration of the active run, then no new snapshots.
                del state
                last.unlink()
                print('RESUMED_EXISTING_CHECKPOINT_THEN_REMOVED',stage,step,flush=True)
        model.train()
        method.set_instruction(stage)
        with (run/'logs'/'events.jsonl').open('a') as events:
            for start in range(offset,len(order),16):
                indexes=order[start:start+16]
                optimizer.zero_grad(set_to_none=True)
                loss_sum,target_tokens,prompt_tokens=0.,0,0
                tick=time.monotonic()
                for index in indexes:
                    row=rows[index]
                    batch,details=input_batch(row,manifest['tasks'][stage-1],processor,device)
                    from frozen_vision import prepare
                    cache_hit=prepare(batch,row,model,method.vision_original)
                    if hasattr(method,'recording'):
                        method.recording(True)
                    with activation_storage(method),torch.autocast('cuda',dtype=torch.float16):
                        loss,length=answer_loss(model,batch,row['answer'],processor)
                    if hasattr(method,'recording'):
                        method.recording(False)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite loss')
                    scaler.scale(loss/len(indexes)).backward()
                    if hasattr(method,'post_backward'):
                        method.post_backward(scaler.get_scale(),start/len(order))
                    loss_sum+=float(loss.detach())/len(indexes)
                    target_tokens+=length
                    prompt_tokens+=details['prompt_tokens']
                scaler.unscale_(optimizer)
                gradient=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(gradient):
                    raise FloatingPointError('nonfinite gradient')
                old_scale=scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale()<old_scale:
                    raise FloatingPointError('optimizer step skipped; refusing silent sample loss')
                scheduler.step()
                step+=1
                if hasattr(method,'after_update'):
                    method.after_update(step,updates)
                record={'event':'train_update','stage':stage,'step':step,'qa_seen':start+len(indexes),
                        'loss':loss_sum,'seconds':time.monotonic()-tick,'prompt_tokens':prompt_tokens,
                        'target_tokens':target_tokens,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
                events.write(json.dumps(record)+'\n')
                events.flush()
                print(json.dumps(record),flush=True)
                if args.save_checkpoints and (step%25==0 or start+len(indexes)==len(order)):
                    last.parent.mkdir(exist_ok=True)
                    save_last(last,method,optimizer,scheduler,scaler,stage,start+len(indexes),step,manifest_sha)
        if hasattr(method,'end_stage'):
            method.end_stage(stage)
        test=evaluate(method,processor,manifest['tasks'],data,run,stage)
        evaluation={'task_scores':test,'validation_enabled':False,'primary_score':'rougeL'}
        metadata={'method':args.method,'seed':args.seed,'manifest_sha256':manifest_sha,
                  'base_model':str(model_path),'current_task_training_qa':len(rows),'global_step':step,
                  'trainable_parameters':sum(parameter.numel() for parameter in parameters)}
        if args.save_checkpoints:
            save_stage(run,stage,method.export,{'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                  'scaler':scaler.state_dict(),'method_state':method.method_state(stage),'global_step':step},metadata,evaluation)
        write(run/'stages'/f'stage_{stage:02d}.json',{'metadata':metadata,'evaluation':evaluation,
              'complete':True,'weights_saved':args.save_checkpoints})
        matrix.append(test)
        write(run/'score_matrix.json',matrix)
        write(run/'run_status.json',{'status':'stage_complete','completed_stage':stage,'method':args.method,'seed':args.seed})
        collect()
        print('STAGE_RESULTS_COMPLETE',stage,str(run/'stages'/f'stage_{stage:02d}.json'),flush=True)
        # Complete stage snapshot already contains recovery state; redundant last can be removed.
        last.unlink(missing_ok=True)
        del optimizer,scheduler,scaler
        torch.cuda.empty_cache()
    array=np.asarray(matrix)
    assert array.shape==(11,10) and np.isfinite(array).all()
    metrics={'AP':float(array[10].mean()),
       'ForgetRate':float(np.mean([array[j:10,j-1].max()-array[10,j-1] for j in range(1,10)])),
       'BWT':float(np.mean([array[10,j-1]-array[j,j-1] for j in range(1,10)])),
       'FWT':float(np.mean([array[j-1,j-1]-array[0,j-1] for j in range(2,11)]))}
    write(run/'metrics.json',metrics)
    write(run/'run_status.json',{'status':'completed','completed_stage':10,'method':args.method,'seed':args.seed,'metrics':metrics})
    collect()


if __name__=='__main__':
    main()
