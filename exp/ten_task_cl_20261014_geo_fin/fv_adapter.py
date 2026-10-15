"""FV-guided Training: author FV intervention, causal heads, two KL terms and head PR."""
import ast,json,math,time,random,copy
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from native_auxiliary import ROOT,NativeIO,author_nodes

def author_components():
    env={'torch':torch,'np':np,'F':F};hashes={}
    def qwen_paths(text):
        return text.replace("or 'mistral' in model_config['name_or_path'].lower()", "or 'mistral' in model_config['name_or_path'].lower() or 'qwen' in model_config['name_or_path'].lower()")
    for path,names in [('src/fvector/utils/intervention_utils.py',{'get_module','replace_activation_w_avg'}),('src/fvector/utils/extract_utils.py',{'compute_function_vector'}),('src/tuning/trainer/base.py',{'add_function_vector'})]:
        hashes[path]=author_nodes('FV_guided_Training',path,names,env,qwen_paths)
    base=ROOT/'code/FV_guided_Training';p=base/'source/src/tuning/trainer/base.py';tree=ast.parse(p.read_text());kd=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='KD_loss')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[kd],type_ignores=[])),str(p),'exec'),env)
    return env,json.loads((base/'source_manifest.json').read_text())['revision'],hashes

class FVLinear(nn.Module):
    """The author's canonical q/v LoRA component; FV losses define the CL method."""
    def __init__(self,base):
        super().__init__();self.weight,self.bias=base.weight,base.bias;self.enabled=True
        self.A=nn.Linear(base.in_features,8,bias=False,device=base.weight.device,dtype=torch.float32)
        self.B=nn.Linear(8,base.out_features,bias=False,device=base.weight.device,dtype=torch.float32)
        self.dropout=nn.Dropout(.1);nn.init.kaiming_uniform_(self.A.weight,a=math.sqrt(5));nn.init.zeros_(self.B.weight)
    def forward(self,x):
        y=F.linear(x,self.weight,self.bias)
        return y+(2*self.B(self.A(self.dropout(x.float())))).to(y.dtype) if self.enabled else y
    @torch.no_grad()
    def merge_reset(self):
        self.weight.add_((2*self.B.weight@self.A.weight).to(self.weight.dtype));nn.init.kaiming_uniform_(self.A.weight,a=math.sqrt(5));nn.init.zeros_(self.B.weight)

class FVMethod:
    def __init__(self,model,name,instructions,embedding=None):
        self.model,self.name=model,name;self.learning_rate=1e-4;self.offload_activations=True;self.stage=0;self.layers=[];self.io=None;self.mean=None;self.function_vector=None;self.heads=[];self.edit_layer=9
        self.kernel,self.original_revision,self.source_hashes=author_components()
        self.model_config={'name_or_path':'Qwen2-VL-2B-Instruct','n_layers':model.config.num_hidden_layers,'n_heads':model.config.num_attention_heads,'resid_dim':model.config.hidden_size}
        for p in model.parameters():p.requires_grad=False
        targets=[(name,module) for name,module in model.named_modules() if name.startswith('model.layers.') and name.rsplit('.',1)[-1] in ['q_proj','v_proj'] and isinstance(module,nn.Linear)]
        for key,base in targets:
            parent,child=key.rsplit('.',1);layer=FVLinear(base);setattr(model.get_submodule(parent),child,layer);self.layers.append(layer)
    def bind_context(self,processor,manifest,data,run,preflight=False):self.io=NativeIO(self,processor,manifest,data,run);self.preflight=preflight
    def set_instruction(self,task):pass
    @contextmanager
    def disabled(self):
        old=[x.enabled for x in self.layers]
        try:
            for layer in self.layers:layer.enabled=False
            yield
        finally:
            for layer,value in zip(self.layers,old):layer.enabled=value
    def capture_heads(self,position,detach=True,selected=None):
        values={};handles=[];width=self.model.config.hidden_size//self.model.config.num_attention_heads
        wanted=range(self.model.config.num_hidden_layers) if selected is None else sorted({int(x[0]) for x in selected})
        for i in wanted:
            def capture(module,args,i=i):
                value=args[0][:,position,:].reshape(args[0].shape[0],self.model.config.num_attention_heads,width)
                values[i]=value.detach().cpu() if detach else value
            handles.append(self.model.model.layers[i].self_attn.o_proj.register_forward_pre_hook(capture))
        return values,handles
    def projection_capture(self,position,heads):return self.capture_heads(position,False,heads)
    @torch.no_grad()
    def mean_activations(self,rows,stage,trials):
        mean=torch.zeros(self.model.config.num_hidden_layers,self.model.config.num_attention_heads,1,self.model.config.hidden_size//self.model.config.num_attention_heads)
        rng=random.Random(42+stage);evidence=[]
        for t in range(trials):
            query=rows[t%len(rows)];demos=rng.sample([x for x in rows if x['id']!=query['id']],10)
            batch=self.io.icl(query,demos,stage);values,handles=self.capture_heads(-1)
            try:
                with torch.autocast('cuda',dtype=torch.float16):self.io.forward(batch)
            finally:
                for h in handles:h.remove()
            mean+=torch.stack([values[i][0] for i in range(mean.shape[0])]).unsqueeze(2).float()/trials
            evidence.append({'query_id':query['id'],'demo_ids':[x['id'] for x in demos]})
            if (t+1)%10==0 or t+1==trials:print(json.dumps({'event':'fv_mean_activations','stage':stage,'done':t+1,'total':trials}),flush=True)
        return mean,evidence
    @staticmethod
    def cache_for_batch(legacy,count):
        return DynamicCache.from_legacy_cache(tuple(tuple(x.repeat(count,1,1,1) for x in pair) for pair in legacy))
    def nll(self,hidden,target):
        # Same per-example mean answer NLL; chunking avoids full-sequence vocab tensors.
        losses=[]
        for start in range(0,target.shape[1],64):
            n=min(64,target.shape[1]-start);logits=self.model.lm_head(hidden[:,start:start+n]).float()
            labels=target[:,start:start+n].expand(hidden.shape[0],-1)
            losses.append(F.cross_entropy(logits.flatten(0,1),labels.flatten(),reduction='none').view(hidden.shape[0],n).sum(1))
        return torch.stack(losses).sum(0)/target.shape[1]
    @torch.no_grad()
    def causal_heads(self,rows,stage,trials):
        effects=torch.zeros(trials,self.model.config.num_hidden_layers,self.model.config.num_attention_heads,2);rng=random.Random(2042+stage);evidence=[]
        for t in range(trials):
            query=rows[t];demos=rng.sample([x for x in rows if x['id']!=query['id']],10)
            b=self.io.icl(query,demos,stage,corrupt=True);target=self.io.target(query['answer'])[:,:-1];cut=b['input_ids'].shape[1]-1;ids=torch.cat([b['input_ids'],target],1)
            with torch.autocast('cuda',dtype=torch.float16):
                emb,positions=self.io.embeddings(b,ids)
                prefix=self.model.model(inputs_embeds=emb[:,:cut],attention_mask=torch.ones_like(ids[:,:cut]),position_ids=positions[:,:,:cut],past_key_values=DynamicCache(),use_cache=True,return_dict=True)
                past=prefix.past_key_values.to_legacy_cache() if hasattr(prefix.past_key_values,'to_legacy_cache') else prefix.past_key_values;del prefix
                tail=emb[:,cut:];tail_positions=positions[:,:,cut:];total=ids.shape[1]
                def forward(count):
                    return self.model.model(inputs_embeds=tail.expand(count,-1,-1),attention_mask=torch.ones(count,total,device=ids.device,dtype=ids.dtype),position_ids=tail_positions.expand(-1,count,-1),past_key_values=self.cache_for_batch(past,count),use_cache=True,return_dict=True).last_hidden_state
                clean=forward(1);baseline_nll=self.nll(clean[:,:-1],target);baseline_prob=self.model.lm_head(clean[:,0]).float().softmax(-1)[0,target[0,0]];del clean
                for layer in range(self.model.config.num_hidden_layers):
                    for start in range(0,self.model.config.num_attention_heads,4):
                        head_ids=list(range(start,min(start+4,self.model.config.num_attention_heads)));funcs=[self.kernel['replace_activation_w_avg']([(layer,h,0)],self.mean.to(ids.device),self.model,self.model_config) for h in head_ids]
                        name=f'model.layers.{layer}.self_attn.o_proj'
                        def patch(module,args,output,funcs=funcs,name=name):
                            # One author intervention per batch row; numerical operations unchanged.
                            return torch.cat([fn(output[i:i+1],name,(args[0][i:i+1],)) for i,fn in enumerate(funcs)],dim=0)
                        hook=self.model.model.layers[layer].self_attn.o_proj.register_forward_hook(patch)
                        try:changed=forward(len(head_ids))
                        finally:hook.remove()
                        delta_nll=self.nll(changed[:,:-1],target)-baseline_nll
                        delta_prob=self.model.lm_head(changed[:,0]).float().softmax(-1)[:,target[0,0]]-baseline_prob
                        effects[t,layer,head_ids,0]=delta_prob.cpu();effects[t,layer,head_ids,1]=delta_nll.cpu();del changed
                    if (layer+1)%7==0:print(json.dumps({'event':'fv_causal_probe','stage':stage,'trial':t+1,'total_trials':trials,'layer':layer+1}),flush=True)
            evidence.append({'query_id':query['id'],'demo_ids':[x['id'] for x in demos],'synthetic_demo_label_permutation':True})
            del past,emb,positions,tail;torch.cuda.empty_cache()
        assert torch.isfinite(effects).all()
        return effects,evidence
    def begin_stage(self,stage):
        assert self.io is not None
        if stage==self.stage:return
        if self.stage:
            for layer in self.layers:layer.merge_reset()
        self.stage=stage;self.model.eval();self.io.clear();rows=self.io.stage_rows(stage);assert len(rows)==100;started=time.monotonic()
        with self.disabled(),torch.no_grad():
            self.mean,means=self.mean_activations(rows,stage,2 if self.preflight else 100)
            effects,causal=self.causal_heads(rows,stage,1 if self.preflight else 25)
            with torch.autocast('cuda',dtype=torch.float16):fv,heads=self.kernel['compute_function_vector'](self.mean,effects,self.model,self.model_config,n_top_heads=10,gen=True)
        self.function_vector=fv.detach();self.heads=[(int(l),int(h),float(score)) for l,h,score in heads]
        self.io.record(f'fv_stage_{stage:02d}.json',{'stage':stage,'preflight_only':self.preflight,'mean_activation_trials':len(means),'causal_trials':len(causal),'shots':10,'selected_heads':self.heads,'head_selection':'author dynamic causal answer-NLL head_score option; Qwen-native dimensions','mean_probe_ids':means,'causal_probe_ids':causal,'seconds':time.monotonic()-started,'historical_qa_replayed':0,'all_probes_current_training_pool':True,'in_memory_only':['function_vector','mean_head_activations'],'edit_layer':9})
        self.model.train()
        print(json.dumps({'event':'fv_auxiliary_ready','stage':stage,'seconds':time.monotonic()-started,'selected_heads':self.heads}),flush=True)
    def training_loss(self,batch,row,processor):
        b={k:v for k,v in batch.items() if k!='pixel_values'}
        if 'pixel_values' in batch:
            with torch.no_grad():b['_encoded_visual']=self.model.visual(batch['pixel_values'].to(self.model.visual.get_dtype()),grid_thw=batch['image_grid_thw'])
        target=self.io.target(row['answer']);depth=b['input_ids'].shape[1]-1;ids=torch.cat([b['input_ids'],target],1)
        # Author teacher: previous-stage base with adapters disabled + current task FV.
        with self.disabled(),torch.no_grad():
            intervention=self.kernel['add_function_vector'](self.edit_layer,self.function_vector,[depth],degree=1)
            def add(module,args,out):return intervention(out,f'module.base_model.model.model.layers.{self.edit_layer}.self_attn.o_proj')
            hook=self.model.model.layers[self.edit_layer].self_attn.o_proj.register_forward_hook(add)
            try:teacher=self.io.forward(b,ids).detach()
            finally:hook.remove()
            reference_heads,handles=self.capture_heads(depth,True,self.heads)
            try:self.io.forward(b,ids)
            finally:
                for h in handles:h.remove()
        student_heads,handles=self.capture_heads(depth,False,self.heads)
        try:student=self.io.forward(b,ids)
        finally:
            for h in handles:h.remove()
        loss=F.cross_entropy(self.model.lm_head(student[:,depth:-1]).float().flatten(0,1),target.flatten())
        kl1=self.kernel['KD_loss'](None,self.model.lm_head(student[:,depth]).float(),self.model.lm_head(teacher[:,depth]).float(),2)
        kl2=torch.zeros((),device=student.device)
        for start in range(0,student.shape[1],64):
            new=self.model.lm_head(student[:,start:start+64]).float()
            with torch.no_grad():previous=self.model.lm_head(teacher[:,start:start+64]).float()
            kl2=kl2+self.kernel['KD_loss'](None,new,previous,2)
        repair=torch.zeros((),device=student.device)
        for layer,head,_ in self.heads:
            item=F.mse_loss(student_heads[layer][:,head],reference_heads[layer][:,head].to(student.device))
            repair=repair+(1. if item.detach()>.01 else 0.)*item
        loss=loss+(20. if kl1.detach()>.1 else 0.)*kl1+(.1 if kl2.detach()>.1 else 0.)*kl2+repair
        self.last_components={'ce':float((loss-(20. if kl1.detach()>.1 else 0.)*kl1-(.1 if kl2.detach()>.1 else 0.)*kl2-repair).detach()),'kl_first':float(kl1.detach()),'kl_all':float(kl2.detach()),'head_pr':float(repair.detach())}
        print(json.dumps({'event':'fv_regularized_loss','stage':self.stage,'qa_id':row['id'],**self.last_components}),flush=True)
        return loss,target.shape[1]
    def method_state(self,stage):return {'method':self.name,'stage':stage,'upstream_revision':self.original_revision,'source_hashes':self.source_hashes,'causal_heads':self.heads,'edit_layer':9,'kl_coefficients':[20,.1],'pr_coefficient':1.,'rank':8,'alpha':16,'head_score_option':'current-task training-only dynamic causal NLL','auxiliary_in_ram_only':True}
