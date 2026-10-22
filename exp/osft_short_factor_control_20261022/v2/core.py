"""Original OSFT tensors and forward; freeze the complete shorter-side low basis."""
import importlib.util
from pathlib import Path
import sys
from types import MethodType

ROOT=Path(__file__).resolve().parents[3]
PILOT=ROOT/'exp/osft_increment_screen_20261019/v1'
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2'
WORK=Path(__file__).parent/'private'
spec=importlib.util.spec_from_file_location('short_factor_inherited_study',PILOT/'study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sha,write=study.sha,study.write

def digest(parameters):
    from training_validity import parameter_digest
    return parameter_digest(list(parameters))

def gradient_directory(run):
    p=run/'gradients';p.mkdir(parents=True,exist_ok=True)
    return p

def save_first_raw_gradients(model,run,stage):
    import torch
    path=gradient_directory(run)/f'S{stage:02d}_first_raw.pt'
    torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},path)
    return path

class ShortFixedOSFT:
    def __init__(self,model):
        sys.path.insert(0,str(PORT));from osft_adapter import OSFTMethod
        original=OSFTMethod(model,'OSFT',['unused'])
        self.__dict__.update(original.__dict__)
        names={id(v):n for n,v in model.named_modules()}
        self.layer_names=[names[id(v)] for v in self.layers]
        self.stage_records=[];self._before_weights=None;self._short_start=None
        self._apply_flags()
    def _apply_flags(self):
        for l in self.layers:
            tall=l.out_features>=l.in_features
            l.U_low.requires_grad_(tall);l.V_low.requires_grad_(not tall)
            l.S_low.requires_grad_(True)
            short=l.V_low if tall else l.U_low
            assert short.grad is None
    def shorts(self):
        return [l.V_low if l.out_features>=l.in_features else l.U_low for l in self.layers]
    def check_trainables(self):
        expected={id(p) for l in self.layers for p in
            ((l.U_low,l.S_low) if l.out_features>=l.in_features else (l.V_low,l.S_low))}
        actual={id(p) for p in self.model.parameters() if p.requires_grad}
        assert actual==expected and all(not p.requires_grad and p.grad is None for p in self.shorts())
        return expected
    def layout(self):
        rows=[]
        for name,l in zip(self.layer_names,self.layers):
            tall=l.out_features>=l.in_features
            rows.append(dict(name=name,shape=[l.out_features,l.in_features],frozen='V_low' if tall else 'U_low',
                trainable=sum(p.numel() for p in l.parameters() if p.requires_grad),
                frozen_short=(l.V_low if tall else l.U_low).numel(),S_low=l.S_low.numel()))
        self.check_trainables()
        return dict(modules=rows,target_modules=len(rows),trainable_coordinates=sum(r['trainable'] for r in rows),
            frozen_short_coordinates=sum(r['frozen_short'] for r in rows),S_low_coordinates=sum(r['S_low'] for r in rows))
    def set_instruction(self,task):
        from osft_adapter import OSFTMethod
        return OSFTMethod.set_instruction(self,task)
    def begin_stage(self,stage):
        from osft_adapter import OSFTMethod
        assert self._before_weights is None
        OSFTMethod.begin_stage(self,stage)
        self._apply_flags();self.check_trainables()
        self._short_start=digest(self.shorts())
    def post_backward(self,scale,progress):
        from osft_adapter import OSFTMethod
        self.check_trainables();OSFTMethod.post_backward(self,scale,progress)
    def _actual_weights(self):
        import torch
        for l in self.layers:
            with torch.no_grad(),torch.autocast(device_type=l.U_low.device.type,dtype=l.output_dtype,
                                                enabled=l.U_low.device.type=='cuda'):
                w=self.kernel['reconstruct_weight_matrix'](l.svd(),torch.float32,l.output_dtype)
            yield w.detach().cpu()
    def observe_before_first_step(self):
        assert self._before_weights is None
        self._before_weights=list(self._actual_weights())
    def after_update(self,step,updates):
        import torch,time
        self.check_trainables()
        with torch.no_grad():
            for l in self.layers:
                s=l.svd()
                if l.out_features<l.in_features:
                    self.kernel['project_parameter_to_orthogonal_space'](s,skip_u=True)
                else:
                    s['V_low']=l.V_low.detach().clone()
                    self.kernel['project_parameter_to_orthogonal_space'](s)
        self.check_trainables()
        if step==1:
            assert self._before_weights is not None
            begin=time.monotonic();rows=[]
            for name,before,after in zip(self.layer_names,self._before_weights,self._actual_weights()):
                assert torch.isfinite(before).all() and torch.isfinite(after).all()
                delta=after.float()-before.float()
                rows.append(dict(name=name,squared_change=float(delta.double().square().sum()),
                    squared_before=float(before.double().square().sum()),changed_coordinates=int((before!=after).sum())))
            self._before_weights=None
            self._first_step=dict(stage=self.stage,matrices=rows,after_step_observation_seconds=time.monotonic()-begin,
                squared_change=sum(r['squared_change'] for r in rows),
                squared_before=sum(r['squared_before'] for r in rows),
                scope='Actual original CUDA FP16-autocast reconstructed weights before update and after parameter projection; one first step per stage, not matched to old reference dynamics. after_step_observation_seconds excludes pre-step capture; full RUN and TRAIN include both halves.')
        if step==updates:
            after=digest(self.shorts());assert after==self._short_start
            self.stage_records.append(dict(stage=self.stage,short_before=self._short_start,short_after=after,
                short_frozen_exact=True,first_step=self._first_step,updates=updates))
    def method_state(self,stage):
        from osft_adapter import OSFTMethod
        return OSFTMethod.method_state(self,stage)|dict(short_side_frozen=True,layout=self.layout(),stage_records=self.stage_records)

def measured_optimizer(method,factory,record):
    """No optimizer reference on method/checkpoint; bound step remains scheduler-compatible."""
    opt=factory(method)
    expected=method.check_trainables()
    assert {id(p) for g in opt.param_groups for p in g['params']}==expected
    original_step=type(opt).step
    steps={}
    def step(self,*args,**kwargs):
        method.check_trainables()
        params=[p for g in self.param_groups for p in g['params']]
        assert {id(p) for p in params}==method.check_trainables()
        assert all(p.grad is not None for p in params)
        gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in params)
        value=original_step(self,*args,**kwargs)
        states=[self.state[p] for p in params]
        assert all('exp_avg' in s and 'exp_avg_sq' in s for s in states)
        moments=[s[k] for s in states for k in ('exp_avg','exp_avg_sq')]
        assert all(t.device.type=='cpu' for t in moments)
        steps[method.stage]=steps.get(method.stage,0)+1
        record(dict(stage=method.stage,step=steps[method.stage],parameter_tensors=len(params),
            gradient_bytes=gradient_bytes,moment_bytes=sum(t.numel()*t.element_size() for t in moments),
            all_moments_CPU=True,short_gradient_None=True,optimizer_parameter_IDs_match=True))
        return value
    opt.step=MethodType(step,opt)
    return opt

def cpu_gate():
    import copy,torch
    from transformers import Qwen2Config,Qwen2ForCausalLM,get_cosine_schedule_with_warmup
    sys.path.insert(0,str(PORT));from osft_adapter import OSFTMethod
    from runtime_utils import make_optimizer
    torch.set_num_threads(2);torch.manual_seed(41)
    cfg=Qwen2Config(vocab_size=31,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
    base=Qwen2ForCausalLM(cfg);ref=copy.deepcopy(base);model=copy.deepcopy(base)
    torch.manual_seed(42);original=OSFTMethod(ref,'OSFT',['unused']);rng=torch.get_rng_state().clone()
    torch.manual_seed(42);method=ShortFixedOSFT(model)
    assert torch.equal(rng,torch.get_rng_state()) and len(method.layers)==14
    assert set(ref.state_dict())==set(model.state_dict())
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in ref.state_dict().items())
    ref.eval();model.eval();ids=torch.tensor([[2,4,7,9]])
    assert torch.equal(ref(ids).logits,model(ids).logits)
    ref(ids).logits.float().square().mean().backward()
    model(ids).logits.float().square().mean().backward()
    assert all(p.grad is not None and torch.equal(p.grad,dict(ref.named_parameters())[n].grad)
               for n,p in model.named_parameters() if p.requires_grad)
    # Clear the admission backward before the actual stage lifecycle.
    model.zero_grad(set_to_none=True);method.begin_stage(1)
    short=digest(method.shorts());evidence=[]
    opt=measured_optimizer(method,make_optimizer,evidence.append)
    schedule=get_cosine_schedule_with_warmup(opt,0,1)
    model(ids).logits.float().square().mean().backward();method.post_backward(1.,0.)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run=Path(tmp)/'fresh_absent_run';assert not run.exists()
        path=save_first_raw_gradients(model,run,1)
        loaded=torch.load(path,map_location='cpu',weights_only=True)
        actual={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert loaded.keys()==actual.keys() and all(loaded[n].dtype==g.dtype and torch.equal(loaded[n],g) for n,g in actual.items())
    before=digest([p for p in model.parameters() if p.requires_grad])
    method.observe_before_first_step();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    opt.step();schedule.step();method.after_update(1,1)
    assert short==digest(method.shorts()) and before!=digest([p for p in model.parameters() if p.requires_grad])
    assert len(evidence)==1 and evidence[0]['parameter_tensors']==28
    method.begin_stage(2);method.check_trainables()
    assert len({r['frozen'] for r in method.layout()['modules']})==2
    assert not torch.cuda.is_initialized()
    return dict(passed=True,tiny_HF_layers=2,target_modules=14,initial_state_forward_RNG_exact=True,
        shared_trainable_raw_gradients_exact=True,both_short_side_branches=True,stage_refactor_flags_reapplied=True,
        frozen_short_full_digest_unchanged_after_optimizer_and_projection=True,finite_backward=True,
        HostAdamW_scheduler_bound_step_compatible=True,optimizer_evidence=evidence,
        first_raw_save_reload_from_absent_run_and_directory_exact=True,optimizer_steps=1,new_actual_model_requests=0)
