"""CL-MoE equations 5--12, adapted to a decoder-only Qwen2-VL backbone.

The released repository leaves task-index inference commented out and its
external merge script matches incorrect parameter prefixes. This port follows
the paper equations, with source-verified author expert classes. Qwen has no
encoder CLS: the frozen decoder's last question-token state replaces CLS.
"""
import json
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from native_auxiliary import ROOT, NativeIO, author_nodes


class CLMoELinear(nn.Module):
    def __init__(self,base,kernel):
        super().__init__()
        self.weight,self.bias=base.weight,base.bias
        self.A=kernel['CLMoEMOELinearA'](base.in_features,64,8)
        self.B=kernel['CLMoEMOELinearB'](64,base.out_features,8)
        self.router=nn.Linear(base.in_features,8,bias=False)
        self.dropout=nn.Dropout(.05)
        for component in (self.A,self.B,self.router):
            component.to(device=base.weight.device,dtype=torch.float32)
        for a,b in zip(self.A.loraA,self.B.loraB):
            nn.init.kaiming_uniform_(a.mlp.weight,a=5**.5)
            nn.init.zeros_(b.mlp.weight)
        self.enabled=True;self.collect=False;self.global_weight=None
        self.total=torch.zeros(8,dtype=torch.float64);self.count=0
        self.previous=None;self.seen=set();self.task_weights={}

    def forward(self,x):
        y=F.linear(x,self.weight,self.bias)
        if not self.enabled:return y
        with torch.autocast(x.device.type,enabled=False):
            z=x.float();w=torch.softmax(self.router(z),dim=-1)
            if self.collect:
                self.total+=w.detach().double().mean(dim=tuple(range(w.ndim-1))).cpu()
                self.count+=1
            if not self.training and self.global_weight is not None:
                w=.5*w+.5*self.global_weight.to(w.device,dtype=w.dtype)
            delta=torch.zeros_like(z[...,:1]).expand(*z.shape[:-1],self.weight.shape[0]).clone()
            for i in range(8):
                delta=delta+2*self.B.loraB[i](self.A.loraA[i](self.dropout(z)))*w[...,i,None]
        return y+delta.to(y.dtype)


class CLMoEMethod:
    def __init__(self,model,name,instructions,embedding=None):
        self.model,self.name=model,name;self.learning_rate=2e-4
        self.offload_activations=True;self.layers=[];self.anchors={};self.feature_cache=OrderedDict()
        self.kernel={'torch':torch,'nn':nn}
        relative='CLMoE/peft/tuners/clitmoelora.py'
        self.source_hash=author_nodes('CL_MoE',relative,{'CLMoEMOELinearA','CLMoEMOELinearB','CLMoEMOEExpert'},self.kernel)
        self.original_revision=json.loads((ROOT/'code/CL_MoE/source_manifest.json').read_text())['revision']
        for p in model.parameters():p.requires_grad=False
        targets={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
        for key,base in list(model.named_modules()):
            if key.startswith('model.layers.') and key.rsplit('.',1)[-1] in targets and isinstance(base,nn.Linear):
                parent,child=key.rsplit('.',1);layer=CLMoELinear(base,self.kernel)
                setattr(model.get_submodule(parent),child,layer);self.layers.append(layer)
        assert self.layers

    def bind_context(self,processor,manifest,data,run,preflight=False):
        self.io=NativeIO(self,processor,manifest,data,run);self.preflight=preflight

    def set_instruction(self,task):
        # Evaluation task labels are never used by the router.
        pass

    def begin_stage(self,stage):
        self.stage=stage
        for layer in self.layers:
            layer.previous=[(a.mlp.weight.detach().cpu().clone(),b.mlp.weight.detach().cpu().clone())
                            for a,b in zip(layer.A.loraA,layer.B.loraB)] if stage>1 else None
            layer.global_weight=None

    @torch.no_grad()
    def feature(self,row):
        key=row['question']
        if key in self.feature_cache:
            self.feature_cache.move_to_end(key);return self.feature_cache[key]
        tokens=self.io.processor.tokenizer(key,return_tensors='pt',truncation=True,max_length=512)
        ids=tokens['input_ids'].to(self.model.device)
        old=self.model.training;self.model.eval()
        flags=[layer.enabled for layer in self.layers]
        try:
            for layer in self.layers:layer.enabled=False
            with torch.autocast('cuda',dtype=self.model.dtype):
                hidden=self.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).last_hidden_state
                value=hidden[0,-1].float().cpu()
        finally:
            for layer,flag in zip(self.layers,flags):layer.enabled=flag
            self.model.train(old)
        self.feature_cache[key]=value
        while len(self.feature_cache)>256:self.feature_cache.popitem(last=False)
        return value

    def prepare_row(self,row,batch):
        if self.model.training or not self.anchors:return
        feature=self.feature(row)
        selected=min(self.anchors,key=lambda k:float(torch.linalg.vector_norm(feature-self.anchors[k])))
        for layer in self.layers:layer.global_weight=layer.task_weights[selected]

    @torch.no_grad()
    def end_stage(self,stage):
        # Final-router statistics use the entire current frozen TRAIN set only.
        old=self.model.training;self.model.eval()
        rows=self.io.data[stage,'train'];anchor=None
        if self.preflight:rows=rows[:2]
        for layer in self.layers:
            layer.total.zero_();layer.count=0;layer.global_weight=None;layer.collect=False
        for row in rows:
            feature=self.feature(row)
            anchor=feature.double() if anchor is None else anchor+feature.double()
            batch=self.io.batch(row,stage)
            for layer in self.layers:layer.collect=True
            with torch.autocast('cuda',dtype=self.model.dtype):self.io.forward(batch)
            for layer in self.layers:layer.collect=False
        self.anchors[stage]=(anchor/len(rows)).float()
        selected=[]
        for layer in self.layers:
            assert layer.count==len(rows)
            weights=(layer.total/layer.count).float();current=set(weights.topk(2).indices.tolist())
            layer.task_weights[stage]=weights
            if layer.previous is not None:
                for i,(a,b) in enumerate(zip(layer.A.loraA,layer.B.loraB)):
                    momentum=.3 if i in current and i not in layer.seen else .7
                    for p,previous in zip((a.mlp.weight,b.mlp.weight),layer.previous[i]):
                        p.mul_(1-momentum).add_(previous.to(p.device),alpha=momentum)
            layer.seen.update(current);layer.previous=None;selected.append(sorted(current))
        self.model.train(old)
        self.io.clear()
        self.io.record(f'clmoe_stage_{stage:02d}.json',{'train_only_count':len(rows),'top2_per_layer':selected,
                       'anchor_definition':'last question token of frozen Qwen decoder; no encoder CLS',
                       'auxiliary_full_current_train_forward_passes':1,'gamma':.7,'beta':.5})

    def method_state(self,stage):
        return {'method':self.name,'completed_stage':stage,'upstream_revision':self.original_revision,
                'source_hash':self.source_hash,'rank':64,'alpha':128,'experts':8,'top_k':2,'gamma':.7,'beta':.5,
                'implementation':'paper equations 5-12; source expert classes; Qwen2-VL adapted reproduction',
                'task_inference':'nearest frozen last-question-token centroid, no test task ID',
                'statistics':'equal per-example mean token routing weights on full current train only'}
