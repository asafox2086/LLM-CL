"""Matched Qwen2-VL diagnosis CL; raw predictions at every task boundary."""
import argparse, gc, hashlib, io, json, math, os, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration

ROOT=Path(__file__).resolve().parents[3]
EXP=Path(__file__).resolve().parent
DATA=ROOT/'analysis/omnimed_20261009'
sys.path.insert(0,str(ROOT/'exp'))
import cv_data as D, cv_runtime as C
from run_olora import create_method
from cv_reflection_med import finish_task
from recovery import atomic_save, capture_rng, restore_rng
torch.set_num_threads(4)

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)
def append(path,value):
    with path.open('a') as handle:handle.write(json.dumps(value,ensure_ascii=False)+'\n')
class CachedQwen(Qwen2VLForConditionalGeneration):
    @property
    def device(self):return self.get_input_embeddings().weight.device
def main(args):
    out=EXP/'runs'/f'{args.method}_seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    selection=json.loads((DATA/'selection.json').read_text())
    probes=selection['probes'];assert [p['source'] for p in probes]==['ISIC2019','Retinal OCT-C8','Fitzpatrick 17k']
    protocol=json.loads((EXP/'protocol.json').read_text())
    special=['ewc_lora','kd_lora','medqwen','moe_lora','seq_capacity']
    cfg=json.loads((ROOT/'exp/configs'/f'{"seq_lora" if args.method in special else args.method}_qwen2vl_med.json').read_text())
    cfg.update(seed=args.seed,epochs=3,batch_size=8,micro_batch_size=2,eval_batch_size=4,target_tokens=64,
               learning_rate=1e-4,warmup_ratio=0.,method='seq_lora' if args.method in special else args.method,
               tasks=[p['source'] for p in probes],protocol='diagnosis_cl_20261010',
               primary_metric='candidate_accuracy',selection='current_task_dev_candidate_accuracy',
               data_manifest=str(DATA/'selection.json'),train_count=372,dev_count=127,test_count=131)
    if args.method=='seq_capacity':cfg.update(rank=13,alpha=52)
    cfg.update(protocol_sha256=hashlib.sha256((EXP/'protocol.json').read_bytes()).hexdigest())
    write(out/'config.json',dict(cfg,reported_method=args.method))
    def status(state,**kw):write(out/'status.json',dict(status=state,pid=os.getpid(),method=args.method,seed=args.seed,time=time.time(),**kw))
    status('running',phase='loading')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    proc=D.processor();tok=proc.tokenizer
    model=CachedQwen.from_pretrained(D.MODEL,local_files_only=True,torch_dtype=torch.float16,
                                   device_map={'':0},attn_implementation='sdpa')
    model.visual.to('cpu');torch.cuda.empty_cache();C.install_cached_vision(model)
    C.feature_path=lambda row:Path(row['_feature'])
    if args.method in ['medqwen','moe_lora']:
        from spectral_lora import SpectralLoRA
        method=SpectralLoRA(model,cfg,spectral=args.method=='medqwen')
    else:method=create_method(model,cfg)
    C.install_sapt(method)
    letterids=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCD']
    def encode(r,p,condition='image'):
        feature=DATA/'features128'/(r['image_sha256']+'.pt')
        grid=torch.load(feature,weights_only=True)['grid'] if condition=='image' else None
        prompt='Answer the following multiple-choice question'+(' using the image' if condition=='image' else '')+'. Reply with only the correct option letter.\nWhich diagnostic category best describes this medical image?\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(p['classes']))
        return D.tokenize(dict(r,image=r['image'] if condition=='image' else None,prompt=prompt,
            references=[chr(65+r['probe_label'])],training_answer=chr(65+r['probe_label']),answer_index=r['probe_label'],
            choices=p['classes'],instance_id=r['question_id'],task_id=p['source'],_feature=str(feature)),proc,grid)
    ds=[{split:[encode(r,p) for r in p['splits'][split]] for split in ['train','dev','test']} for p in probes]
    for task,p in zip(ds,probes):task['test_no_image']=[encode(r,p,'no_image') for r in p['splits']['test']]
    parameters=[];consolidation=[];matrix=[]
    def prepare(rs,training):
        batch=C.batch_tensors(rs,tok,model.device,training)
        if hasattr(method,'prepare_batch'):method.prepare_batch(rs,tok,batch,training)
        return batch
    def option_logits(batch):
        pos,_=model.get_rope_index(batch['input_ids'],batch.get('image_grid_thw'),attention_mask=batch['attention_mask'])
        hidden=model.model(inputs_embeds=C.fused_embeddings(model,batch),position_ids=pos,
                           attention_mask=batch['attention_mask'],use_cache=False,return_dict=True).last_hidden_state[:,-1]
        return model.lm_head(hidden).float()
    def answer_logits(batch):
        pos,_=model.get_rope_index(batch['input_ids'],batch.get('image_grid_thw'),attention_mask=batch['attention_mask'])
        hidden=model.model(inputs_embeds=C.fused_embeddings(model,batch),position_ids=pos,
                           attention_mask=batch['attention_mask'],use_cache=False,return_dict=True).last_hidden_state
        z=[];ys=[]
        for h,y in zip(hidden,batch['labels']):
            valid=y[1:]!=-100;z.append(model.lm_head(h[:-1][valid]).float());ys.append(y[1:][valid])
        return z,ys
    def score(rows,stage,task,condition,save=True,epoch=None):
        model.eval();records=[]
        path=out/'predictions'/f'stage_{stage:02d}'/f'{task}_{condition}.jsonl'
        path.parent.mkdir(parents=True,exist_ok=True)
        if save and path.exists():
            records=[json.loads(s) for s in path.read_text().splitlines()]
            assert len(records)<=len(rows)
            for r,rec in zip(rows,records):assert r['question_id']==rec['question_id']
        for start in range(len(records),len(rows),4):
            rs=rows[start:start+4];batch=prepare(rs,False)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):z=option_logits(batch)
            candidate=z[:,letterids].argmax(-1).tolist();raw=z.argmax(-1).tolist()
            for r,a,t in zip(rs,candidate,raw):
                first=tok.decode([t]).strip().strip('.,:;!?').casefold()
                rec=dict(method=args.method,seed=args.seed,stage=stage,epoch=epoch,task=task,condition=condition,
                    question_id=r['question_id'],image_sha256=r['image_sha256'],answer_index=r['answer_index'],
                    candidate_prediction=a,candidate_correct=int(a==r['answer_index']),raw_first_token=first,
                    strict_correct=int(first==chr(65+r['answer_index']).casefold()))
                records.append(rec)
                if save:append(path,rec)
        acc=100*np.mean([r['candidate_correct'] for r in records])
        strict=100*np.mean([r['strict_correct'] for r in records])
        balanced=100*np.mean([np.mean([r['candidate_correct'] for r in records if r['answer_index']==label]) for label in range(4)])
        result=dict(stage=stage,epoch=epoch,task=task,condition=condition,n=len(rows),accuracy=acc,balanced_accuracy=balanced,strict_first_token_accuracy=strict)
        print('SCORE',json.dumps(result),flush=True)
        return result
    for stage in range(4):
        boundary=out/'checkpoints'/f'stage_{stage:02d}.pt'
        if boundary.exists():
            saved=torch.load(boundary,map_location='cpu',weights_only=False)
            assert saved['protocol_sha256']==cfg['protocol_sha256']
            if stage:method.load(io.BytesIO(saved['method']))
            matrix=saved['matrix'];consolidation=saved['consolidation'];restore_rng(saved['rng'])
            continue
        if stage:
            task=ds[stage-1];name=probes[stage-1]['source']
            status('running',phase='training',stage=stage,task=name)
            teacher={}
            if args.method=='kd_lora' and stage>1:
                model.eval()
                for start in range(0,len(task['train']),4):
                    rs=task['train'][start:start+4];batch=prepare(rs,True)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):zs,_=answer_logits(batch)
                    for r,z in zip(rs,zs):teacher[r['question_id']]=z.to('cpu',dtype=torch.float16)
                write(out/f'teacher_stage{stage}.json',dict(samples=len(teacher),vocabulary=model.config.vocab_size,
                    temperature=2.,lambda_kd=.5,positions='all supervised answer tokens including EOS'))
            method.begin_task();parameters=method.parameters()
            optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0.)
            scaler=torch.amp.GradScaler('cuda',init_scale=1024)
            total=3*math.ceil(len(task['train'])/8);step=0;best=-1;best_epoch=None
            train_recovery=out/'checkpoints'/f'train_stage_{stage}.pt'
            first_epoch=1;first_offset=0
            if train_recovery.exists():
                saved=torch.load(train_recovery,map_location='cpu',weights_only=False)
                method.load(io.BytesIO(saved['method']));optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
                first_epoch=saved['epoch'];first_offset=saved['offset'];step=saved['step'];best=saved['best'];best_epoch=saved['best_epoch'];restore_rng(saved['rng'])
            for epoch in range(first_epoch,4):
                model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
                order=list(range(len(task['train'])));random.Random(args.seed+epoch+100*stage).shuffle(order)
                for start in range(first_offset if epoch==first_epoch else 0,len(order),8):
                    rs=[task['train'][i] for i in order[start:start+8]];optimizer.zero_grad(set_to_none=True)
                    if hasattr(method,'before_update'):method.before_update(step)
                    for group in optimizer.param_groups:group['lr']=1e-4*(1-step/total)
                    ce_value=0.;pen_value=0.;tick=time.monotonic()
                    for offset in range(0,len(rs),cfg['micro_batch_size']):
                        micro=rs[offset:offset+cfg['micro_batch_size']];batch=prepare(micro,True)
                        with torch.autocast('cuda',dtype=torch.float16):
                            if teacher:
                                zs,ys=answer_logits(batch)
                                ce=torch.stack([torch.nn.functional.cross_entropy(z,y) for z,y in zip(zs,ys)]).mean()
                                kd=torch.stack([torch.nn.functional.kl_div((z/2.).log_softmax(-1),
                                    (teacher[r['question_id']].to(model.device).float()/2.).softmax(-1),reduction='batchmean')
                                    for r,z in zip(micro,zs)]).mean()*.5
                            else:ce=C.answer_loss(model,batch);kd=0.
                            penalty=method.penalty(cfg['orthogonal_weight'],cfg['l2_weight'])+kd
                        if args.method=='ewc_lora' and consolidation:
                            ewc=sum((f.to(p.device)*(p-s.to(p.device)).square()).sum()
                                    for item in consolidation for p,f,s in zip(parameters,item['fisher'],item['snapshot']))*2500.
                            penalty=penalty+ewc
                        loss=(ce+penalty)*len(micro)/len(rs)
                        if hasattr(method,'after_forward'):method.after_forward()
                        assert torch.isfinite(loss)
                        scaler.scale(loss).backward()
                        ce_value+=float(ce.detach())*len(micro)/len(rs)
                        pen_value+=float(penalty.detach() if torch.is_tensor(penalty) else penalty)*len(micro)/len(rs)
                    scaler.unscale_(optimizer)
                    if hasattr(method,'before_step'):method.before_step()
                    gn=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(gn)
                    scale=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=scale
                    step+=1
                    rec=dict(method=args.method,seed=args.seed,stage=stage,epoch=epoch,step=step,total=total,
                        ce=ce_value,regularizer=pen_value,gradient_norm=float(gn),seconds=time.monotonic()-tick,
                        questions=[r['question_id'] for r in rs])
                    append(out/'updates.jsonl',rec)
                    state=io.BytesIO();method.save(state)
                    atomic_save(dict(method=state.getvalue(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                        epoch=epoch,offset=start+len(rs),step=step,best=best,best_epoch=best_epoch,rng=capture_rng()),train_recovery)
                    if step%5==0:status('running',phase='training',**{k:rec[k] for k in ['stage','epoch','step','total']});print('TRAIN',json.dumps(rec),flush=True)
                model.gradient_checkpointing_disable();model.disable_input_require_grads();model.config.use_cache=True
                result=score(task['dev'],stage,name,'dev',False,epoch);append(out/'dev_scores.jsonl',result)
                if result['accuracy']>best:
                    best=result['accuracy'];best_epoch=epoch;method.save(out/'checkpoints'/f'best_stage_{stage}.pt')
                state=io.BytesIO();method.save(state)
                atomic_save(dict(method=state.getvalue(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                    epoch=epoch+1,offset=0,step=step,best=best,best_epoch=best_epoch,rng=capture_rng()),train_recovery)
            method.load(out/'checkpoints'/f'best_stage_{stage}.pt')
            write(out/f'selection_stage{stage}.json',dict(epoch=best_epoch,dev_accuracy=best,updates=total,
                trainable_parameters=sum(p.numel() for p in parameters),adapter_parameters=sum(p.numel() for l in method.layers.values() for p in l.adapters.parameters())))
            del optimizer,scaler,teacher;gc.collect();torch.cuda.empty_cache()
            if args.method=='ewc_lora' and stage<3:
                model.eval();parameters=method.parameters();fisher=[torch.zeros_like(p,device='cpu') for p in parameters]
                chosen=[]
                for label in range(4):
                    rs=[r for r in task['train'] if r['answer_index']==label]
                    random.Random(args.seed+label+stage).shuffle(rs);chosen+=rs[:8]
                for r in chosen:
                    model.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):loss=C.answer_loss(model,prepare([r],True))
                    loss.backward()
                    for f,p in zip(fisher,parameters):f.add_(p.grad.detach().cpu().float().square()/len(chosen))
                consolidation.append(dict(fisher=fisher,snapshot=[p.detach().cpu().clone() for p in parameters]))
                write(out/f'fisher_stage{stage}.json',dict(n=len(chosen),questions=[r['question_id'] for r in chosen],
                    lambda_ewc=5000.,sum_fisher=sum(float(f.sum()) for f in fisher)))
                model.zero_grad(set_to_none=True)
            if args.method=='sapt_lora' and stage<3:
                status('running',phase='reflection',stage=stage)
                finish_task(method,tok,task['train'],out/'reflection'/f'stage_{stage}',cfg)
        model.eval();stage_results=[]
        for i,(task,p) in enumerate(zip(ds,probes)):
            for condition in ['image','no_image']:
                status('running',phase='evaluation',stage=stage,task=p['source'],condition=condition)
                stage_results.append(score(task['test' if condition=='image' else 'test_no_image'],stage,p['source'],condition))
        matrix.append(stage_results);write(out/'scores.json',matrix)
        state=io.BytesIO();method.save(state)
        atomic_save(dict(protocol_sha256=cfg['protocol_sha256'],method=state.getvalue(),matrix=matrix,
            consolidation=consolidation,rng=capture_rng()),boundary)
    status('completed',stages=3)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True);parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    try:main(args)
    except BaseException:
        out=EXP/'runs'/f'{args.method}_seed{args.seed}';write(out/'status.json',dict(status='failed',traceback=traceback.format_exc()));raise
