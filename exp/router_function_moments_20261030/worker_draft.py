"""DRAFT ONLY. Native-forward hook helper; no CLI, asset loader, queue or retry.
Not admitted. Caller must bind original CP/source/80 TRAIN rows before use.
Hooks + TorchDispatchMode capture actual original softmax and expert B outputs.
Internal t is reconstructed in original arithmetic, not directly exposed by module.
All extra base GEMMs, reconstruction/CPU copies/dispatch costs are observation cost.
"""
import time
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

class _GateCapture(TorchDispatchMode):
    def __init__(self, owner):
        super().__init__(); self.owner=owner
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result=func(*args,**(kwargs or {}))
        current=self.owner.current
        packet=getattr(func,'overloadpacket',None)
        name=getattr(packet,'__name__','')
        if current is not None and name=='_softmax':
            state=self.owner.state[current]
            # First softmax within the target module is its actual router.
            # PAS later computes usage summaries; do not capture those as gates.
            if 'raw_gate' not in state:
                e=self.owner.experts
                if result.shape[-1]!=e:
                    raise RuntimeError('unexpected first module softmax shape')
                shaped=result.reshape(1,self.owner.length,e)
                state['raw_gate']=shaped[:,self.owner.start:self.owner.end,:].detach().clone()
                state['gate_dispatch_operator']=str(func)
        return result

class NativeMomentHooks:
    """One model, one QA at a time; enabled only after original prepare_row().
    No forward override, weight edit, gate normalization or branch replacement.
    Caller writes every emitted row durably; no hidden activation retained after QA.
    """
    def __init__(self, *, model, method_name, statistics, emit,
                 closure_atol, closure_rtol):
        if method_name not in ('PASs_MoE','CL_MoE'):
            raise ValueError('only two preregistered mechanisms')
        self.model=model;self.method_name=method_name;self.statistics=statistics
        self.emit=emit;self.atol=closure_atol;self.rtol=closure_rtol
        self.enabled=False;self.current=None;self.state={};self.handles=[]
        self.experts=2 if method_name=='PASs_MoE' else 8
        self.layers={n:m for n,m in model.core.named_modules()
                     if (hasattr(m,'_current_E') if self.experts==2 else type(m).__name__=='CLMoELinear')}
        if len(self.layers)!=168:
            raise RuntimeError('expected complete 168-layer graph')
        for name,m in self.layers.items():
            self.state[name]={}
            self.handles.append(m.register_forward_pre_hook(lambda mod,args,n=name:self.pre(n,mod,args)))
            if self.experts==2:
                adapter=m.active_adapter
                if m.training or m.gate_router!='a' or m.merged or m.disable_adapters or m._current_E()!=2 or m.fan_in_fan_out:
                    raise RuntimeError('unsupported original PAS state')
                if m._expert_mask.detach().cpu().tolist()!=[1.,1.]:
                    raise RuntimeError('original active mask mismatch')
                self.handles.append(m.lora_B[adapter].register_forward_hook(
                    lambda mod,args,out,n=name:self.pas_b(n,out)))
            else:
                if m.training or not m.enabled or m.collect:
                    raise RuntimeError('unsupported original CL state')
                for e,expert in enumerate(m.B.loraB):
                    self.handles.append(expert.register_forward_hook(
                        lambda mod,args,out,n=name,e=e:self.cl_b(n,e,out)))
            self.handles.append(m.register_forward_hook(lambda mod,args,out,n=name:self.post(n,mod,out)))
    def begin_qa(self, *, qa_id, domain, prompt_length, answer_tokens):
        if self.enabled or self.current is not None or any(self.state.values()):
            raise RuntimeError('prior QA incomplete')
        if prompt_length<1 or answer_tokens<1:
            raise ValueError('nonempty prompt and answer')
        self.qa_id=qa_id;self.domain=domain
        self.start=prompt_length-1;self.end=prompt_length+answer_tokens
        self.length=self.end+1;self.calls={n:0 for n in self.layers}
        self.enabled=True
    def pre(self,name,m,args):
        if not self.enabled:return
        if self.current is not None or self.state[name]:raise RuntimeError('nested/stale layer')
        x=args[0]
        if x.ndim!=3 or x.shape[:2]!=(1,self.length):
            raise RuntimeError('loss-mask/input length alignment mismatch')
        self.current=name;self.state[name]={'x':x.detach(),'experts':{}}
    def pas_b(self,name,outputs):
        if not self.enabled:return
        if self.current!=name or len(outputs)!=2:raise RuntimeError('PAS hook ordering')
        for e,o in enumerate(outputs):
            self.state[name]['experts'][e]=o.reshape(1,self.length,-1)[:,self.start:self.end,:].detach().clone()
    def cl_b(self,name,e,output):
        if not self.enabled:return
        if self.current!=name:raise RuntimeError('CL hook ordering')
        self.state[name]['experts'][e]=output[:,self.start:self.end,:].detach().clone()
    def post(self,name,m,output):
        if not self.enabled:return
        if self.current!=name:raise RuntimeError('layer boundary mismatch')
        state=self.state[name];self.current=None
        if 'raw_gate' not in state or set(state['experts'])!=set(range(self.experts)):
            raise RuntimeError('actual gate/expert capture incomplete')
        began=time.monotonic()
        # Recompute original base on full original input to retain GEMM shape.
        # This is extra observation work, not claimed as native deployment cost.
        with torch.autocast('cuda',enabled=False):
            x=state['x']
            if x.dtype!=m.weight.dtype:
                raise RuntimeError('base dtype contract differs; original autocast needs separate source review')
            base_full=F.linear(x,m.weight,m.bias)
            base=base_full[:,self.start:self.end,:]
            gate=state['raw_gate']
            if self.experts==8:
                if m.global_weight is None:raise RuntimeError('missing original CL anchor gate')
                gate=.5*gate+.5*m.global_weight.to(gate.device,dtype=gate.dtype)
                scale=2
            else:scale=m.scaling[m.active_adapter]
            values=[state['experts'][i]*scale for i in range(self.experts)]
            delta=torch.zeros_like(values[0])
            expected=base
            for e in range(self.experts):
                branch=values[e]*gate[...,e,None]
                delta=delta+branch
                if self.experts==2:expected=expected+branch
            if self.experts==8:expected=base+delta.to(base.dtype)
            expected=expected.to(output.dtype)
            actual=output[:,self.start:self.end,:]
            error=(expected.double()-actual.double()).abs()
            allowed=self.atol+self.rtol*torch.maximum(expected.double().abs(),actual.double().abs())
            closure={'all_elements_within_tolerance':bool((error<=allowed).all()),
                     'bitwise_equal':bool(torch.equal(expected,actual)),
                     'max_absolute_error':float(error.max()),
                     'atol':self.atol,'rtol':self.rtol}
            u=torch.stack(values,dim=2)[0].detach()
            row=self.statistics.collect_qa(qa_id=self.qa_id,domain=self.domain,layer_id=name,
                experts=u,gates=gate[0].detach(),
                base=base[0].detach(),actual_update=delta[0].detach(),
                prediction_positions=list(range(self.start,self.end)),
                position_kinds=['answer']*(self.end-self.start-1)+['eos'],
                closure_atol=self.atol,closure_rtol=self.rtol)
        row['actual_native_layer_output_closure']=closure
        row['update_observation']='native-operation reconstruction from actual gate and B hooks; final-output closed, not direct internal delta tensor'
        row['gate_dispatch_operator']=state['gate_dispatch_operator']
        row['observation_hook_seconds']=time.monotonic()-began
        self.emit(row)
        self.state[name]={};self.calls[name]+=1
        if not closure['all_elements_within_tolerance']:
            raise RuntimeError('native final output closure failed; raw row retained')
    def end_qa(self):
        self.enabled=False
        if self.current is not None or any(self.state.values()) or set(self.calls.values())!={1}:
            raise RuntimeError('one original forward per QA and all 168 layers required')
    def dispatch(self):return _GateCapture(self)
    def close(self):
        self.enabled=False
        for h in self.handles:h.remove()
        self.handles=[];self.current=None;self.state.clear()


def forward_hidden_only(*, model, batch, answer, processor):
    """Duplicate original answer_loss input preparation, omit lm_head and CE.
    Original onevision bridge/vision binding must already be source-verified by caller.
    No generation, optimizer, backward, score, hidden/prediction persistence.
    Caller wraps torch.inference_mode + original CUDA autocast + hooks.dispatch().
    Caller prepares original CL question feature BEFORE begin_qa, and separately counts it.
    """
    target=processor.tokenizer(answer,add_special_tokens=False,truncation=True,max_length=512)['input_ids']
    if not target:raise ValueError('empty answer')
    target=target+[processor.tokenizer.eos_token_id]
    target_tensor=torch.tensor([target],device=batch['input_ids'].device)
    ids=torch.cat([batch['input_ids'],target_tensor],dim=1)
    mask=torch.ones_like(ids)
    embeddings=model.get_input_embeddings()(ids)
    grid=batch.get('image_grid_thw')
    if 'pixel_values' in batch:
        visual=model.visual(batch['pixel_values'].to(model.visual.get_dtype()),grid_thw=grid)
        image_mask=(ids==model.config.image_token_id).unsqueeze(-1).expand_as(embeddings)
        if image_mask.sum().item()!=visual.numel():raise ValueError('image-token feature mismatch')
        embeddings=embeddings.masked_scatter(image_mask,visual.to(embeddings.dtype))
    positions,_=model.get_rope_index(ids,grid,attention_mask=mask)
    result=model.model(inputs_embeds=embeddings,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True)
    del result
    return {'target_tokens_including_eos':len(target),'generations':0,'CE':0,'F1':0}

if __name__=='__main__':
    raise SystemExit('DRAFT ONLY: no CLI/model runner admitted; independently bind assets and review hooks first')
