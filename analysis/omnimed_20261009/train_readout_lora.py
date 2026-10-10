"""Same labeled images as the frozen-feature readout, fresh medical-only LoRA."""
import sys,json,random,time,gc,math,hashlib,traceback
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
import cv_data as D,cv_runtime as C
from run_olora import create_method
torch.set_num_threads(4)
def write(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2))
def append(name,x):
    with (OUT/name).open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
class CachedQwen(Qwen2VLForConditionalGeneration):
    @property
    def device(self):return self.get_input_embeddings().weight.device
def main():
    assert json.loads((OUT/'status.json').read_text())['status']=='completed'
    sel=json.loads((OUT/'selection.json').read_text());proc=D.processor();tok=proc.tokenizer
    cfg={**json.loads((ROOT/'exp/configs/seq_lora_qwen2vl_med.json').read_text()),'rank':8,'alpha':32,'target_tokens':4}
    model=CachedQwen.from_pretrained(D.MODEL,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
    model.visual.to('cpu');torch.cuda.empty_cache();C.install_cached_vision(model);C.feature_path=lambda r:Path(r['_feature'])
    def encode(r,classes,condition='image'):
        p=OUT/'features128'/(r['image_sha256']+'.pt');grid=torch.load(p,weights_only=True)['grid'] if condition=='image' else None
        prompt='Answer the following multiple-choice question'+(' using the image' if condition=='image' else '')+'. Reply with only the correct option letter.\nWhich diagnostic category best describes this medical image?\n'+'\n'.join(f'{chr(65+i)}. {c}' for i,c in enumerate(classes))
        return D.tokenize(dict(r,image=r['image'] if condition=='image' else None,prompt=prompt,references=[chr(65+r['probe_label'])],training_answer=chr(65+r['probe_label']),answer_index=r['probe_label'],choices=classes,_feature=str(p)),proc,grid)
    ds={s:[encode(r,p['classes']) for p in sel['probes'] for r in p['splits'][s]] for s in ['train','dev','test']}
    ds['test_no_image']=[encode(r,p['classes'],'no_image') for p in sel['probes'] for r in p['splits']['test']]
    letterids=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCD']
    write('adaptation_protocol.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seeds=[42,43,44],epochs=3,train_n=len(ds['train']),dev_n=len(ds['dev']),test_n=len(ds['test']),rank=8,alpha=32,targets=cfg['targets'],lr=1e-4,weight_decay=0.,clip=1.,batch=8,microbatch=4,loss='CE option letter plus EOS',selection='best pooled dev candidate accuracy among epochs1/2/3, ties earliest',training_labels='Exactly the same images and diagnostic category labels used by linear readouts; joint LoRA over three sources',test_scope='Only custom held-out image split; no claim about the full benchmark or patient separation',timing='Follow-up initiated after inspecting frozen readout results; exploratory confirmation, no test hyperparameter selection',execution_repair='Microbatch-1 execution pilot stopped during first epoch; restarted all seeds from base with microbatch4 for throughput, effective batch unchanged; pilot logs retained separately'))
    def evaluate(method,rows,seed,epoch,split,save=False):
        model.eval();recs=[]
        for start in range(0,len(rows),4):
            rs=rows[start:start+4];b=C.batch_tensors(rs,tok,model.device,False)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                pos,_=model.get_rope_index(b['input_ids'],b.get('image_grid_thw'),attention_mask=b['attention_mask'])
                h=model.model(inputs_embeds=C.fused_embeddings(model,b),position_ids=pos,attention_mask=b['attention_mask'],use_cache=False,return_dict=True).last_hidden_state[:,-1]
                logits=model.lm_head(h).float();pred=logits[:,letterids].argmax(-1).tolist();raw=logits.argmax(-1).tolist()
            for r,a,t in zip(rs,pred,raw):
                first=tok.decode([t]).strip().strip('.,:;!?').casefold()
                rec=dict(seed=seed,epoch=epoch,split=split,source=r['dataset'],question_id=r['question_id'],image_sha256=r['image_sha256'],answer_index=r['answer_index'],candidate_prediction=a,candidate_correct=int(a==r['answer_index']),raw_first_token=first,strict_first_token_correct=int(first==chr(65+r['answer_index']).casefold()))
                recs.append(rec)
                if save:append('adaptation_predictions.jsonl',rec)
        result=dict(seed=seed,epoch=epoch,split=split,n=len(recs),candidate_accuracy=float(np.mean([r['candidate_correct'] for r in recs])*100),strict_accuracy=float(np.mean([r['strict_first_token_correct'] for r in recs])*100))
        append('adaptation_scores.jsonl',result);print('SCORE',json.dumps(result),flush=True)
        return result['candidate_accuracy']
    ck=OUT/'adaptation_checkpoints';ck.mkdir(exist_ok=True)
    for seed in [42,43,44]:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        method=create_method(model,cfg);method.begin_task();opt=torch.optim.AdamW(method.parameters(),lr=1e-4,weight_decay=0.)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024);steps=3*math.ceil(len(ds['train'])/8);step=0;best=-1;selected=None
        for epoch in [1,2,3]:
            model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
            order=list(range(len(ds['train'])));random.Random(seed+epoch).shuffle(order)
            for start in range(0,len(order),8):
                rs=[ds['train'][i] for i in order[start:start+8]];opt.zero_grad(set_to_none=True);losses=[]
                lr=1e-4*(1-step/steps)
                for group in opt.param_groups:group['lr']=lr
                for micro in range(0,len(rs),4):
                    chunk=rs[micro:micro+4]
                    with torch.autocast('cuda',dtype=torch.float16):loss=C.answer_loss(model,C.batch_tensors(chunk,tok,model.device,True))
                    assert torch.isfinite(loss);scaler.scale(loss*len(chunk)/len(rs)).backward();losses.extend([float(loss.detach())]*len(chunk))
                scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(method.parameters(),1.);assert torch.isfinite(gn)
                oldscale=scaler.get_scale();scaler.step(opt);scaler.update();assert scaler.get_scale()>=oldscale
                step+=1;rec=dict(seed=seed,epoch=epoch,step=step,total_steps=steps,lr=lr,loss=float(np.mean(losses)),gradient_norm=float(gn))
                append('adaptation_updates.jsonl',rec)
                if step%5==0:write('adaptation_status.json',dict(status='running',**rec));print('TRAIN',json.dumps(rec),flush=True)
            model.gradient_checkpointing_disable();model.disable_input_require_grads();model.config.use_cache=True
            score=evaluate(method,ds['dev'],seed,epoch,'dev')
            if score>best:best=score;selected=epoch;method.save(ck/f'seed{seed}.pt')
        method.load(ck/f'seed{seed}.pt')
        evaluate(method,ds['test'],seed,selected,'test',True);evaluate(method,ds['test_no_image'],seed,selected,'test_no_image',True)
        for n,l in method.layers.items():parent,child=n.rsplit('.',1);setattr(model.get_submodule(parent),child,l.base)
        del method,opt;gc.collect();torch.cuda.empty_cache()
    write('adaptation_status.json',dict(status='completed',time=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException:write('adaptation_status.json',dict(status='failed',traceback=traceback.format_exc()));raise
