"""DRAFT ONLY. Native-generation-prefix hooks; no CLI, asset loader, queue or retry.
Not admitted. Caller must bind original CP/source/80 TRAIN rows before use.
Hooks + TorchDispatchMode capture actual original softmax and expert B outputs.
Internal t is reconstructed in original arithmetic, not directly exposed by module.
All extra base GEMMs, reconstruction/CPU copies/dispatch costs are observation cost.
"""
import time,json,math,sys
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
        if current is not None and name in ('_softmax','softmax'):
            state=self.owner.state[current]
            # First softmax within the target module is its actual router.
            # PAS later computes usage summaries; do not capture those as gates.
            if 'raw_gate' not in state:
                e=self.owner.experts
                if tuple(result.shape) not in ((self.owner.length,e),(1,self.owner.length,e)):
                    raise RuntimeError('unexpected first module softmax shape')
                shaped=result.reshape(1,self.owner.length,e)
                state['raw_gate']=shaped[:,self.owner.start:self.owner.end,:].detach().clone()
                logits=args[0].reshape(1,self.owner.length,e)
                state['softmax_input']=logits[:,self.owner.start:self.owner.end,:].detach().clone()
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
    def begin_native_qa(self, *, qa_id, domain, prompt_length, old_tf_c):
        if self.enabled or self.current is not None or any(self.state.values()):
            raise RuntimeError('prior QA incomplete')
        if prompt_length<1:raise ValueError('nonempty question-only prompt')
        self.qa_id=qa_id;self.domain=domain;self.prompt_length=prompt_length
        self.old_tf_c=old_tf_c;self.step_index=0;self.step_metadata=[];self.pools={n:{} for n in self.layers}
        self.enabled=True
    @staticmethod
    def cache_length(cache):
        if cache is None:return 0
        if hasattr(cache,'get_seq_length'):return int(cache.get_seq_length())
        if isinstance(cache,(tuple,list)) and cache:return int(cache[0][0].shape[-2])
        raise RuntimeError('unreviewed cache type; no inference/rescue')
    def decoder_pre(self,module,args,kwargs):
        if not self.enabled:return
        if self.current is not None or any(self.state.values()):raise RuntimeError('incomplete previous decoder step')
        x=kwargs.get('inputs_embeds')
        if x is None:x=kwargs.get('input_ids')
        if x is None and args:x=args[0]
        if x is None or x.shape[0]!=1:raise RuntimeError('one native sequence required')
        L=int(x.shape[1]);cache=self.cache_length(kwargs.get('past_key_values'))
        if self.step_index==0:
            if L!=self.prompt_length or cache!=0:raise RuntimeError('native prefill contract')
            self.phase='prefill'
        else:
            if L!=1:raise RuntimeError('native cached decode must use one token')
            self.phase='decode'
        if self.step_index>=32:raise RuntimeError('more than32 original prediction forwards')
        self.length=L;self.start=L-1;self.end=L
        self.calls={n:0 for n in self.layers};self.cache_before=cache
        self.cache_position=kwargs.get('cache_position')
        if self.cache_position is None:raise RuntimeError('actual cache_position missing')
        self.absolute_prediction_position=int(self.cache_position[-1])
        expected=self.prompt_length-1 if self.step_index==0 else self.prompt_length+self.step_index-1
        if self.absolute_prediction_position!=expected:raise RuntimeError('native cache_position does not match prediction index')
        if self.step_index>0 and cache!=expected:raise RuntimeError('native cached decode length does not advance')
        self.step_index+=1
    def decoder_post(self,module,args,kwargs,output):
        if not self.enabled:return
        if self.current is not None or any(self.state.values()) or set(self.calls.values())!={1}:
            raise RuntimeError('one original decoder call must include all168 target layers')
        self.step_metadata.append(dict(prediction_index=self.step_index-1,phase=self.phase,
            actual_L=self.length,cache_length_before=self.cache_before,
            cache_position=self.cache_position.detach().cpu().tolist(),
            absolute_prediction_position=self.absolute_prediction_position))
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
            metadata=dict(layer_id=name,qa_id=self.qa_id,domain=self.domain,
                raw_gate_present='raw_gate' in state,
                captured_expert_indices=sorted(state['experts']),
                expected_expert_indices=list(range(self.experts)),
                gate_dispatch_operator=state.get('gate_dispatch_operator'),
                input_shape=list(state['x'].shape),
                selected_prediction_positions=[self.start,self.end])
            error=RuntimeError('actual gate/expert capture incomplete: '+json.dumps(metadata,sort_keys=True))
            error.capture_metadata=metadata
            raise error
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
            g=gate[0].detach();native_delta=delta[0].detach()
            ud=u.double();gd=g.double();ideal=(ud*gd[:,:,None]).sum(1)
            c=self.old_tf_c[name]
            if len(c)!=self.experts:raise RuntimeError('old TF reference dimension mismatch')
            cd=torch.tensor(c,dtype=torch.float64,device=u.device)
            oldR_direct=float(((ud*(gd-cd)[:,:,None]).sum(1)).square().sum())
            operations=2*self.experts+2;unit=2.**-24;gamma=operations*unit/(1-operations*unit)
            rounderr=(native_delta.double()-ideal).abs()
            roundbound=gamma*(ud*gd[:,:,None]).abs().sum(1)+operations*(2.**-126)
            arithmetic=dict(rule='gamma_(2E+2)*sum_abs_Ug+(2E+2)*FP32_min_normal',
                max_absolute_error=float(rounderr.max()),max_absolute_bound=float(roundbound.max()),
                violating_elements=int((rounderr>roundbound).sum()),
                all_elements_within_tolerance=bool((rounderr<=roundbound).all()))
        # Keep only selected detached GPU fragments until this QA ends; no full E2 moments per step.
        pool=self.pools[name].setdefault(self.phase,[])
        pool.append(dict(u=u,g=g,base=base[0].detach().clone(),delta=native_delta.clone(),
            position=self.absolute_prediction_position,oldR_direct=oldR_direct))
        row=dict(qa_id=self.qa_id,domain=self.domain,layer_id=name,phase=self.phase,
            prediction_index=self.step_index-1,decoder_input_L=self.length,
            cache_length_before=self.cache_before,absolute_prediction_position=self.absolute_prediction_position,
            actual_gate_tau=float(m.gate_tau) if self.experts==2 else None,
            raw_token_gate=state['raw_gate'][0,0].detach().cpu().tolist(),
            actual_softmax_input=state['softmax_input'][0,0].detach().cpu().tolist(),
            effective_gate=g[0].detach().cpu().tolist(),
            actual_global_weight=m.global_weight.detach().cpu().tolist() if self.experts==8 else None,
            old_tf_reference_R_direct=oldR_direct,routed_update_energy=float(ideal.square().sum()),
            gate_sum_error=float(gd.sum()-1),actual_native_layer_output_closure=closure,
            moment_arithmetic_closure=arithmetic,
            gate_dispatch_operator=state['gate_dispatch_operator'],observation_hook_seconds=time.monotonic()-began)
        self.emit(row)
        self.state[name]={};self.calls[name]+=1
        if not closure['all_elements_within_tolerance'] or not arithmetic['all_elements_within_tolerance']:
            raise RuntimeError('native output/arithmetic closure failed; original record retained')
    def end_native_qa(self, generated_new_tokens):
        self.enabled=False
        if self.current is not None or any(self.state.values()):raise RuntimeError('incomplete final layer')
        if self.step_index!=generated_new_tokens or len(self.step_metadata)!=generated_new_tokens:
            raise RuntimeError('one prediction forward per actual generated token required')
        if not 1<=generated_new_tokens<=32:raise RuntimeError('prefix count outside fixed limit')
        phase_rows=[]
        for name,phases in self.pools.items():
            if len(phases['prefill'])!=1 or len(phases.get('decode',[]))!=generated_new_tokens-1:
                raise RuntimeError('phase support mismatch')
            for phase,parts in phases.items():
                positions=[p['position'] for p in parts]
                if len(set(positions))!=len(positions):raise RuntimeError('duplicate absolute prediction positions')
                row=self.statistics.collect_qa(qa_id=self.qa_id,domain=self.domain,layer_id=name,
                    experts=torch.cat([p['u'] for p in parts],dim=0),
                    gates=torch.cat([p['g'] for p in parts],dim=0),
                    base=torch.cat([p['base'] for p in parts],dim=0),
                    actual_update=torch.cat([p['delta'] for p in parts],dim=0),
                    prediction_positions=positions,position_kinds=['native_prediction']*len(parts),
                    closure_atol=self.atol,closure_rtol=self.rtol)
                c=self.old_tf_c[name]
                terms=[row['a'],-2*math.fsum(c[i]*row['h'][i] for i in range(self.experts)),
                    math.fsum(c[i]*row['H'][i][j]*c[j] for i in range(self.experts) for j in range(self.experts))]
                raw=math.fsum(terms);direct=math.fsum(p['oldR_direct'] for p in parts)/len(parts)
                scale=64*sys.float_info.epsilon*math.fsum(abs(v) for v in terms)
                row.update(phase=phase,old_tf_c=c,old_tf_reference_R_terms=terms,
                    old_tf_reference_R_raw=raw,old_tf_reference_R_direct=direct,
                    old_tf_R_raw_minus_direct=raw-direct,old_tf_R_roundoff_scale=scale,
                    old_tf_R_closure_exceeds_scale=abs(raw-direct)>scale)
                if not row['closure']['all_elements_within_tolerance']:
                    raise RuntimeError('QA-phase arithmetic certificate failed')
                phase_rows.append(row)
            phases.clear() # release selected GPU fragments per layer/QA
        self.pools.clear()
        return dict(steps=list(self.step_metadata),phase_rows=phase_rows)
    def dispatch(self):return _GateCapture(self)
    def close(self):
        self.enabled=False
        for h in self.handles:h.remove()
        self.handles=[];self.current=None;self.state.clear()


# No teacher-forcing helper here: native generation only. No CLI/model admission.
