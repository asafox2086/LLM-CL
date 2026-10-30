"""Native capture/model construction helpers for fixed block12 exit pilot.
No CLI, model loader, data reader, CP export, generation or score.
"""
import copy,torch
import torch.nn.functional as F
BLOCK_INDEX=11 # Human block12, fixed before any pilot observations.

class BlockCapture:
    def __init__(self,model):
        block=model.core.language_model.model.layers[BLOCK_INDEX]
        self.active=False;self.cache={};self.count={};self.handles=[]
        self.handles.append(block.register_forward_pre_hook(lambda m,a:self.put('block_input',a[0])))
        self.handles.append(block.post_attention_layernorm.register_forward_pre_hook(lambda m,a:self.put('skip',a[0])))
        self.handles.append(block.mlp.down_proj.register_forward_pre_hook(lambda m,a:self.put('down_input',a[0])))
        self.handles.append(block.mlp.down_proj.register_forward_hook(lambda m,a,o:self.put('down_output',o)))
        self.handles.append(block.register_forward_hook(lambda m,a,o:self.put('block_output',o[0] if isinstance(o,tuple) else o)))
    def begin(self,prompt_length,answer_tokens):
        if self.active or self.cache:raise RuntimeError('stale capture')
        if prompt_length<1 or answer_tokens<1:raise ValueError('nonempty prompt/answer')
        self.start=prompt_length-1;self.end=prompt_length+answer_tokens;self.length=self.end+1
        self.positions=list(range(self.start,self.end));self.kinds=['answer']*answer_tokens+['eos']
        self.active=True;self.count={}
    def put(self,key,value):
        if not self.active:return
        if value.ndim!=3 or value.shape[:2]!=(1,self.length):raise ValueError('actual causal input length')
        self.count[key]=self.count.get(key,0)+1
        self.cache[key]=value[0,self.start:self.end,:].detach().clone()
    def take(self):
        self.active=False
        if set(self.cache)!={'block_input','skip','down_input','down_output','block_output'} or set(self.count.values())!={1}:raise RuntimeError('exactly one original forward capture required')
        out=self.cache;self.cache={};out['positions']=self.positions;out['kinds']=self.kinds
        # Actual native Qwen pre-norm block closes at its unnormalized skip addition.
        out['native_skip_add_bitwise']=bool(torch.equal(out['skip']+out['down_output'],out['block_output']))
        if not out['native_skip_add_bitwise']:raise RuntimeError('original block skip-add contract mismatch')
        return out
    def close(self):
        self.active=False
        for handle in self.handles:handle.remove()
        self.handles=[];self.cache={}

def make_mean_student(teacher_model,method_name,descriptions):
    """Copy original graph with dense replacements prebuilt in deepcopy memo.
    No teacher state/forward edit. Other parameter sharing is preserved by deepcopy.
    Consumer must also audit unchanged dtype/alias groups outside replaced subtrees.
    """
    if method_name not in ('PASs_MoE','CL_MoE'):raise ValueError('fixed two teachers')
    layers={n:m for n,m in teacher_model.core.named_modules() if (hasattr(m,'_current_E') if method_name=='PASs_MoE' else type(m).__name__=='CLMoELinear')}
    if len(layers)!=168 or set(layers)!=set(descriptions):raise ValueError('all frozen 168 projections')
    cpu_rng=torch.random.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state_all()
    memo={};folding={}
    for name,m in layers.items():
        c=descriptions[name]['all']['c_global_mean_gate']
        E=2 if method_name=='PASs_MoE' else 8
        if len(c)!=E or any(v<0 for v in c) or abs(sum(c)-1)>1e-5:raise ValueError('frozen mean coefficients; never renormalize')
        if m.weight.dtype!=torch.float16 or m.training:raise ValueError('original eval FP16 base')
        if method_name=='PASs_MoE':
            a=m.active_adapter
            if m.fan_in_fan_out or m.disable_adapters or m.merged or m._current_E()!=2 or m._expert_mask.detach().cpu().tolist()!=[1.,1.]:raise ValueError('PAS sealed state')
            pairs=[(m.lora_B[a].loraB[e].weight,m.As_weight(e)) for e in range(E)];scale=m.scaling[a]
        else:
            if not m.enabled or m.collect:raise ValueError('CL original state')
            pairs=[(m.B.loraB[e].mlp.weight,m.A.loraA[e].mlp.weight) for e in range(E)];scale=2.
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            w=m.weight.detach().float().clone()
            for coeff,(B,A) in zip(c,pairs):w.add_(B.detach().float()@A.detach().float(),alpha=float(coeff)*scale)
            dense=torch.nn.Linear(m.weight.shape[1],m.weight.shape[0],bias=m.bias is not None,device='meta',dtype=m.weight.dtype)
            dense.weight=torch.nn.Parameter(w.to(m.weight.dtype).detach().clone(),requires_grad=False)
            if m.bias is not None:dense.bias=torch.nn.Parameter(m.bias.detach().clone(),requires_grad=False)
            dense.requires_grad_(False);dense.eval()
        memo[id(m)]=dense;folding[name]=dict(c=list(c),scale=scale,shape=list(dense.weight.shape),dtype=str(dense.weight.dtype))
    student=copy.deepcopy(teacher_model,memo)
    student.eval();student.requires_grad_(False)
    # frozen_vision.install attaches a Python closure capturing the ORIGINAL model.
    # deepcopy does not rebind closure cells: restore this student's own bound
    # PackedVision.forward, which already supports the prepared 2D feature input.
    original_visual=getattr(student,'_llmcl_visual_original',None)
    if original_visual is None or getattr(original_visual,'__self__',None) is not student.visual:
        raise RuntimeError('student original visual method is not bound to student.visual')
    if getattr(original_visual.__func__,'__closure__',None) is not None:
        raise RuntimeError('expected closure-free original PackedVision.forward')
    student.visual.forward=original_visual
    if not (student.model is student.core.language_model.model and
            student.lm_head is student.core.language_model.lm_head and
            student.visual.core is student.core and
            student.visual.tower is student.core.vision_tower and
            student.visual.projector is student.core.multi_modal_projector and
            student.visual.image_newline is student.core.image_newline):
        raise RuntimeError('student wrapper/core/visual forward entry aliases do not close')
    if student.visual.forward.__self__ is not student.visual:
        raise RuntimeError('student visual forward remains bound to old model')
    # Identity/shape checks alone would miss an incorrect copied value. Compare
    # every registered non-replaced parameter/buffer at construction, on GPU.
    equal_counts={}
    for kind in ('parameters','buffers'):
        getter='named_'+kind
        teacher_values={n:t for n,t in getattr(teacher_model.core,getter)(remove_duplicate=False)
                        if not any(n==z or n.startswith(z+'.') for z in layers)}
        student_values={n:t for n,t in getattr(student.core,getter)(remove_duplicate=False)
                        if not any(n==z or n.startswith(z+'.') for z in layers)}
        if set(teacher_values)!=set(student_values):
            raise RuntimeError('non-replaced '+kind+' names changed')
        for name,t in teacher_values.items():
            u=student_values[name]
            if t.dtype!=u.dtype or t.shape!=u.shape or t.device!=u.device or not torch.equal(t,u):
                raise RuntimeError('non-replaced '+kind+' changed: '+name)
        equal_counts[kind+'_alias_occurrences']=len(teacher_values)
    student._llmcl_exit_forward_integrity_proof=dict(wrapper_core_aliases_verified=True,
        student_own_PackedVision_forward_restored=True,visual_forward_has_no_closure=True,
        nonreplaced_registered_values_equal=True,comparison_counts=equal_counts)
    if any(hasattr(m,'_current_E') or type(m).__name__=='CLMoELinear' for m in student.core.modules()):raise RuntimeError('conditional bank retained')
    if sum(p.numel() for p in student.core.parameters())!=893675552:raise RuntimeError('single original backbone count')
    if {p.data_ptr() for p in student.core.parameters()} & {p.data_ptr() for p in teacher_model.core.parameters()}:raise RuntimeError('student aliases teacher parameter storage')
    advanced=not torch.equal(torch.random.get_rng_state(),cpu_rng) or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),cuda_rng))
    torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
    student._llmcl_exit_construction_rng_proof=dict(snapshot_restored=True,advanced_during_construction=advanced,meta_linear_parameter_assignment=True)
    return student,folding

def kl_and_agreement(teacher_logits,student_logits,chunk_tokens=16):
    """Equal-token within one QA; caller then equal-QA aggregates. All vocab retained.
    No reference answer score. FP64 log-softmax, no truncation/top-k approximation.
    """
    if teacher_logits.shape!=student_logits.shape or teacher_logits.ndim!=2:raise ValueError('paired full-vocabulary logits')
    T=teacher_logits.shape[0]
    if T<1 or not teacher_logits.is_cuda or not student_logits.is_cuda:raise ValueError('nonempty GPU tensors')
    total=0.;agree=0;minimum=None
    for start in range(0,T,chunk_tokens):
        lt=F.log_softmax(teacher_logits[start:start+chunk_tokens].double(),dim=-1)
        ls=F.log_softmax(student_logits[start:start+chunk_tokens].double(),dim=-1)
        kl=(lt.exp()*(lt-ls)).sum(-1)
        if not bool(torch.isfinite(kl).all()):raise ArithmeticError('nonfinite KL')
        total+=float(kl.sum());agree+=int((lt.argmax(-1)==ls.argmax(-1)).sum())
        v=float(kl.min());minimum=v if minimum is None else min(minimum,v)
    return dict(teacher_to_student_KL=total/T,top1_agreement=agree/T,target_positions=T,minimum_raw_token_KL=minimum)

def activation_metrics(teacher,student):
    result={}
    for key in ('block_output','skip','down_output'):
        t=teacher[key].double();s=student[key].double()
        numerator=float((s-t).square().sum()/t.shape[0]);denominator=float(t.square().sum()/t.shape[0])
        result[key]=dict(mean_squared_vector_error=numerator,teacher_energy=denominator,relative_squared_error=numerator/denominator if denominator>0 else None)
    # Difference of actual teacher output and its ideal unrounded skip-add.
    native_roundoff=teacher['block_output'].double()-teacher['skip'].double()-teacher['down_output'].double()
    increment=teacher['block_output'].double()-teacher['block_input'].double()
    inc_energy=float(increment.square().sum()/increment.shape[0])
    result['teacher_block_increment_energy']=inc_energy
    result['block_error_over_teacher_block_increment']=result['block_output']['mean_squared_vector_error']/inc_energy if inc_energy>0 else None
    result['teacher_skip_add_roundoff_squared_norm']=float(native_roundoff.square().sum()/native_roundoff.shape[0])
    return result
