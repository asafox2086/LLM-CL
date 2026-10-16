"""Independent frozen-ID OneVision diagnostic controls; no proposed algorithm."""
import gc,hashlib,json,math,os,random,sys,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor,LlavaOnevisionForConditionalGeneration
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp/five_domain_native_cl_20261016_sparse'))
from runtime_utils import score,write
from resource_monitor import ResourceMonitor
MODEL=ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV'
REPORT=ROOT/'summary'/HERE.name

def hashed(row,key):return hashlib.sha256(('42:'+key+':'+row['id']).encode()).digest()

def freeze():
    source=ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    manifest=json.loads(source.read_text());data={};domains={}
    for task in manifest['tasks']:
        domains.setdefault(task['domain'],[]).append(task['task'])
        for split,info in task['files'].items():
            path=ROOT/info['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
            data[task['task'],split]=[dict(json.loads(line),_task=task['task']) for line in path.read_text().splitlines()]
    tests={};generation={};trainprobes={}
    for domain,tasks in domains.items():
        # Allocate at least one per native type, then proportional largest remainder.
        sizes={t:len(data[t,'test']) for t in tasks};n=100;total=sum(sizes.values())
        quotas={t:min(sizes[t],1+int((n-len(tasks))*sizes[t]/total)) for t in tasks}
        while sum(quotas.values())<n:
            t=max(tasks,key=lambda t:((n-len(tasks))*sizes[t]/total+1-quotas[t],-t) if quotas[t]<sizes[t] else (-1e9,-t));quotas[t]+=1
        tests[domain]=[row for t in tasks for row in sorted(data[t,'test'],key=lambda r:hashed(r,'diagnostic-test'))[:quotas[t]]]
        generation[domain]=sorted(tests[domain],key=lambda r:hashed(r,'diagnostic-generation'))[:10]
        training=[row for t in tasks for row in data[t,'train']]
        trainprobes[domain]=sorted(training,key=lambda r:hashed(r,'diagnostic-train-probe'))[:16]
        assert not {r['id'] for r in training}&{r['id'] for r in tests[domain]}
    protocol={'seed':42,'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'train_ids':{d:[r['id'] for t in tasks for r in data[t,'train']] for d,tasks in domains.items()},
        'diagnostic_test_ids':{d:[r['id'] for r in rows] for d,rows in tests.items()},
        'generation_ids':{d:[r['id'] for r in rows] for d,rows in generation.items()},
        'train_probe_ids':{d:[r['id'] for r in rows] for d,rows in trainprobes.items()},
        'test_status':'500 subset QA, 50 generation QA; NOT full-test formal baseline scores',
        'model_revision':'74dd0bf867a4cda7950c17663794267c60cf4b40','rank':8,'alpha':32,'batch':16,'epochs':1,
        'warmup_rule':'floor(0.03*updates), capped below updates; single-update task gets zero warmup',
        'geometry':'deltaW=B*A exact Gram cosine and low-rank singular spectrum; not A/B element cosine',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    path=HERE/'frozen_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol changed; create a new experiment ID'
    else:write(path,protocol);path.chmod(0o444)
    return manifest,data,domains,tests,generation,trainprobes,protocol

class LoRA(nn.Module):
    def __init__(self,base):
        super().__init__();self.weight,self.bias=base.weight,base.bias;self.enabled=True
        self.A=nn.Linear(base.in_features,8,bias=False,device=base.weight.device,dtype=torch.float32)
        self.B=nn.Linear(8,base.out_features,bias=False,device=base.weight.device,dtype=torch.float32)
        nn.init.kaiming_uniform_(self.A.weight,a=5**.5);nn.init.zeros_(self.B.weight);self.drop=nn.Dropout(.1)
    def forward(self,x):
        y=F.linear(x,self.weight,self.bias)
        if not self.enabled:return y
        with torch.autocast('cuda',enabled=False):delta=4*self.B(self.A(self.drop(x.float())))
        return y+delta.to(y.dtype)

class Experiment:
    def __init__(self,manifest,run):
        self.manifest,self.run=manifest,run;self.cache=OrderedDict()
        self.monitor=ResourceMonitor(run);self.monitor.snapshot('process_before_model')
        self.processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        self.model=LlavaOnevisionForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
        for p in self.model.parameters():p.requires_grad=False
        self.layers={}
        for key,base in list(self.model.named_modules()):
            if key.startswith('language_model.model.layers.') and key.rsplit('.',1)[-1] in ['q_proj','v_proj'] and isinstance(base,nn.Linear):
                parent,name=key.rsplit('.',1);layer=LoRA(base);setattr(self.model.get_submodule(parent),name,layer);self.layers[key]=layer
        assert len(self.layers)==48
        self.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.language_model.enable_input_require_grads()
        self.initial=self.snapshot();self.monitor.snapshot('training_before',model=self.model)

    def snapshot(self):
        return {key:{'A':layer.A.weight.detach().cpu().clone(),'B':layer.B.weight.detach().cpu().clone()} for key,layer in self.layers.items()}

    def restore(self,state):
        with torch.no_grad():
            for key,l in self.layers.items():l.A.weight.copy_(state[key]['A']);l.B.weight.copy_(state[key]['B'])

    def inputs(self,row,view='image',wrong=None,retain_pixels=False):
        task=self.manifest['tasks'][row['_task']-1];images=[];content=[]
        paths=(wrong or row)['images'] if view!='text' else []
        # CODA's native question already contains the LLaVA <image> marker.
        # Render that marker once; do not add a second template image token.
        native_markers=row['question'].count('<image>')
        if paths and native_markers:assert native_markers==len(paths)
        for path in paths:
            with Image.open(ROOT/path) as image:images.append(image.convert('RGB'))
            if not native_markers:content.append({'type':'image'})
        text=task['instruction']
        if row.get('context'):text+='\nContext: '+row['context']
        question=row['question'].replace('<image>','') if view=='text' else row['question']
        text+='\nQuestion: '+question;content.append({'type':'text','text':text})
        prompt=self.processor.apply_chat_template([{'role':'user','content':content}],add_generation_prompt=True,tokenize=False)
        assert prompt.count('<image>')==len(paths),'Image placeholder rendering must match actual assets'
        batch=self.processor(text=prompt,images=images or None,return_tensors='pt')
        batch={k:v.to('cuda',dtype=torch.bfloat16) if v.is_floating_point() else v.cuda() for k,v in batch.items()}
        # Frozen visual/projector cache; native AnyRes preprocessing stays unchanged.
        cachekey=tuple(paths)
        if paths:
            if cachekey not in self.cache:
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    f=self.model.get_image_features(batch['pixel_values'],batch['image_sizes'],self.model.config.vision_feature_layer,self.model.config.vision_feature_select_strategy)
                    f,_=self.model.pack_image_features(f,batch['image_sizes'],self.model.image_newline,self.model.config.vision_aspect_ratio)
                self.cache[cachekey]=f.detach().cpu()
                while len(self.cache)>32:self.cache.popitem(last=False)
            batch['_features']=self.cache[cachekey].to('cuda');self.cache.move_to_end(cachekey)
            if not retain_pixels:del batch['pixel_values']
        return batch

    def embeddings(self,ids,batch):
        emb=self.model.get_input_embeddings()(ids)
        if '_features' in batch:
            mask=(ids==self.model.config.image_token_index).unsqueeze(-1).expand_as(emb)
            assert int(mask.sum())==batch['_features'].numel()
            emb=emb.masked_scatter(mask,batch['_features'].to(emb.dtype))
        return emb

    def loss(self,row,view='image',wrong=None):
        batch=self.inputs(row,view,wrong)
        target=self.processor.tokenizer(row['answer'],add_special_tokens=False)['input_ids']
        truncated=len(target)>512;target=target[:512]+[self.processor.tokenizer.eos_token_id]
        labels=torch.tensor([target],device='cuda');ids=torch.cat([batch['input_ids'],labels],dim=1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=self.model.language_model(inputs_embeds=self.embeddings(ids,batch),attention_mask=torch.ones_like(ids),use_cache=False,num_logits_to_keep=len(target)+1)
            loss=F.cross_entropy(out.logits[:,:-1,:].float().reshape(-1,out.logits.shape[-1]),labels.flatten())
        return loss,{'prompt_tokens':int(batch['input_ids'].shape[1]),'target_tokens':len(target),'target_truncated':truncated,
                     'visual_tokens':int((batch['input_ids']==self.model.config.image_token_index).sum())}

    @torch.no_grad()
    def generate(self,row,view='image',wrong=None):
        batch=self.inputs(row,view,wrong);ids=batch['input_ids'];tick=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model.language_model.generate(input_ids=ids,inputs_embeds=self.embeddings(ids,batch),attention_mask=batch['attention_mask'],
                max_new_tokens=self.manifest['tasks'][row['_task']-1]['max_new_tokens'],do_sample=False,use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,eos_token_id=self.processor.tokenizer.eos_token_id)
        prediction=self.processor.tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True)
        return {'prediction':prediction,'scores':score(prediction,row['answers']),'generation_seconds':time.monotonic()-tick}

    def evaluate(self,label,tests,generation,views=('image',)):
        self.model.eval();folder=self.run/'evaluations'/label;folder.mkdir(parents=True,exist_ok=True);summary={}
        for domain,rows in tests.items():
            for view in views:
                path=folder/f'{domain}_{view}.jsonl';records=[]
                if path.exists():records=[json.loads(x) for x in path.read_text().splitlines()]
                assert [x['id'] for x in records]==[x['id'] for x in rows[:len(records)]]
                gids={x['id'] for x in generation[domain]}
                with path.open('a') as out:
                    for index,row in enumerate(rows[len(records):],len(records)):
                        wrong=next((rows[(index+j)%len(rows)] for j in range(1,len(rows)) if rows[(index+j)%len(rows)]['images']!=row['images']),None)
                        if view=='wrong' and wrong is None:raise ValueError('No distinct image for mismatch')
                        with torch.no_grad():loss,details=self.loss(row,view,wrong if view=='wrong' else None)
                        record={'id':row['id'],'native_task':row['_task'],'view':view,'nll':float(loss),'references':row['answers'],**details}
                        if view=='wrong':record['wrong_image_source_id']=wrong['id']
                        if row['id'] in gids:record.update(self.generate(row,view,wrong if view=='wrong' else None))
                        assert math.isfinite(record['nll']);records.append(record);out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
                        print(json.dumps({'event':'diagnostic_eval','label':label,'domain':domain,'view':view,'done':len(records),'total':len(rows)}),flush=True)
                generated=[r for r in records if 'scores' in r]
                summary[domain,view]={'count':len(records),'mean_nll':sum(x['nll'] for x in records)/len(records),
                    'generated_count':len(generated),**{k:(sum(x['scores'][k] for x in generated)/len(generated) if generated else None) for k in ['rougeL','token_f1','exact_match']}}
        write(folder/'summary.json',{d+':'+v:result for (d,v),result in summary.items()})

    def train(self,label,rows):
        self.model.train();self.monitor.snapshot(label+'_before',model=self.model)
        parameters=[p for p in self.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
        order=sorted(rows,key=lambda r:hashed(r,'training'))
        self.model.zero_grad(set_to_none=True);updates=math.ceil(len(order)/16)
        from transformers import get_cosine_schedule_with_warmup
        warmup=min(int(.03*updates),max(0,updates-1))
        scheduler=get_cosine_schedule_with_warmup(optimizer,warmup,updates)
        effective_updates=0
        for start in range(0,len(order),16):
            tick=time.monotonic();optimizer.zero_grad(set_to_none=True);minibatch=order[start:start+16];losses=[]
            for row in minibatch:
                loss,details=self.loss(row);assert torch.isfinite(loss);(loss/len(minibatch)).backward();losses.append(float(loss.detach()))
            grad=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(grad)
            used_lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
            effective_updates+=int(used_lr>0)
            record={'event':'train','label':label,'qa_seen':start+len(minibatch),'total_qa':len(order),'loss':sum(losses)/len(losses),
                'gradient_l2_preclip':float(grad),'learning_rate_used':used_lr,'warmup_updates':warmup,
                'seconds':time.monotonic()-tick,'gpu_allocated_bytes':torch.cuda.memory_allocated(),**details}
            with (self.run/'events.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        optimizer.zero_grad(set_to_none=True);del optimizer,scheduler,loss;gc.collect();torch.cuda.empty_cache()
        assert effective_updates>0,'No effective training updates; cannot interpret as learning failure'
        self.monitor.snapshot(label+'_after',model=self.model)
        state=self.snapshot();path=self.run/'adapters'/f'{label}.pt';path.parent.mkdir(exist_ok=True)
        torch.save({'adapter':state,'rank':8,'alpha':32,'base_revision':'74dd0bf867a4cda7950c17663794267c60cf4b40'},path)
        return state

    def gradients(self,label,probes):
        self.model.eval();parameters=[p for p in self.model.parameters() if p.requires_grad];vectors={};losses={}
        for domain,rows in probes.items():
            self.model.zero_grad(set_to_none=True);values=[]
            for row in rows[:2]:
                loss,_=self.loss(row);values.append(float(loss.detach()));(loss/min(2,len(rows))).backward()
            vectors[domain]=torch.cat([(p.grad.detach().float().cpu().flatten() if p.grad is not None else torch.zeros(p.numel())) for p in parameters])
            assert torch.isfinite(vectors[domain]).all();losses[domain]=sum(values)/len(values)
        self.model.zero_grad(set_to_none=True)
        matrix={d:{e:float(F.cosine_similarity(a[None],b[None])) for e,b in vectors.items()} for d,a in vectors.items()}
        write(self.run/'gradients'/f'{label}.json',{'train_only':True,'probe_qa_per_domain':2,'train_probe_nll':losses,
              'same_model_parameterization_gradient_cosine':matrix,'negative_cosine_is_local_conflict_not_causal_proof':True})

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1','Diagnostic GPU1 only'
    REPORT.mkdir(exist_ok=True,parents=True)
    integrity=json.loads((MODEL/'download_integrity.json').read_text());assert integrity['complete']
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.set_num_threads(6)
    manifest,data,domains,tests,generation,probes,protocol=freeze()
    run=HERE/'runs'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
    write(REPORT/'status.json',{'phase':'initializing','run':str(run.relative_to(ROOT))})
    write(run/'protocol.json',protocol);x=Experiment(manifest,run)
    write(run/'model_counts.json',{'total_parameters':sum(p.numel() for p in x.model.parameters()),'trainable_parameters':sum(p.numel() for p in x.model.parameters() if p.requires_grad),'dtype':str(x.model.dtype),'vision_preprocessing':'official native AnyRes unchanged'})
    # All five domains: native image forward, backward, actual optimizer step.
    cases=[];optimizer=torch.optim.AdamW([p for p in x.model.parameters() if p.requires_grad],lr=1e-4)
    for domain,rows in probes.items():
        row=rows[0];x.model.eval();optimizer.zero_grad(set_to_none=True);loss,details=x.loss(row)
        # Compare cache+answer-only-logit path to the standard official model.
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            batch=x.inputs(row,retain_pixels=True);batch.pop('_features',None)
            target=x.processor.tokenizer(row['answer'],add_special_tokens=False)['input_ids'][:512]+[x.processor.tokenizer.eos_token_id]
            labels=torch.tensor([target],device='cuda');batch['input_ids']=torch.cat([batch['input_ids'],labels],dim=1);batch['attention_mask']=torch.ones_like(batch['input_ids'])
            native=x.model(**batch,use_cache=False,num_logits_to_keep=len(target)+1).logits[:,:-1,:].float()
            expected=F.cross_entropy(native.reshape(-1,native.shape[-1]),labels.flatten());error=abs(float(expected)-float(loss))
            assert error<.015,(domain,error)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_([p for p in x.model.parameters() if p.requires_grad],1.)
        assert torch.isfinite(loss) and torch.isfinite(grad);optimizer.step()
        with torch.no_grad():after,_=x.loss(row)
        cases.append({'domain':domain,'loss':float(loss.detach()),'after_one_update_nll':float(after),'gradient_l2_preclip':float(grad),'native_forward_loss_error':error,**details})
        print(json.dumps({'event':'preflight',**cases[-1]}),flush=True)
    optimizer.zero_grad(set_to_none=True);del optimizer,loss,native,batch;gc.collect();torch.cuda.empty_cache()
    x.restore(x.initial);write(run/'preflight.json',{'passed':True,'cases':cases,'diagnostic_updates_discarded':True})
    write(REPORT/'status.json',{'phase':'base_visual_dependency','run':str(run.relative_to(ROOT))})
    x.evaluate('base',tests,generation,views=('image','text','wrong'))
    x.gradients('base',probes)
    states={}
    for domain,tasks in domains.items():
        x.restore(x.initial);write(REPORT/'status.json',{'phase':'isolated_training','domain':domain,'run':str(run.relative_to(ROOT))})
        rows=[row for t in tasks for row in data[t,'train']]
        states[domain]=x.train('isolated_'+domain,rows)
        x.gradients('isolated_'+domain,probes)
        x.evaluate('isolated_'+domain,tests,generation)
    # Stage-matched isolated controls remove the domain pooling/scheduler confound.
    for task in manifest['tasks']:
        stage=task['task'];x.restore(x.initial)
        write(REPORT/'status.json',{'phase':'native_isolated','native_stage':stage,'run':str(run.relative_to(ROOT))})
        x.train(f'native_isolated_T{stage:02d}',data[stage,'train'])
        subset={d:[r for r in rows if r['_task']==stage] for d,rows in tests.items()}
        subset={d:rows for d,rows in subset.items() if rows}
        matched_generation={d:[r for r in generation[d] if r['_task']==stage] for d in subset}
        x.evaluate(f'native_isolated_T{stage:02d}',subset,matched_generation)
    x.restore(x.initial)
    for task in manifest['tasks']:
        stage=task['task'];write(REPORT/'status.json',{'phase':'sequential','native_stage':stage,'run':str(run.relative_to(ROOT))})
        x.train(f'sequential_T{stage:02d}',data[stage,'train']);x.gradients(f'sequential_T{stage:02d}',probes);x.evaluate(f'sequential_T{stage:02d}',tests,generation)
    x.monitor.snapshot('training_after',model=x.model);x.monitor.finish(x.model)
    from geometry import analyze
    analyze(states,x.run)
    write(REPORT/'status.json',{'phase':'diagnostic_controls_complete','run':str(run.relative_to(ROOT)),'formal_full_test':False,'adaptive_method_implemented':False})

if __name__=='__main__':main()
