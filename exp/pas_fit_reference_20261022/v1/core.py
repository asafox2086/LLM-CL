"""Original PASs training method, with scalar callback evidence only."""
import importlib.util
from pathlib import Path
import sys
from types import MethodType

ROOT=Path(__file__).resolve().parents[3]
PILOT=ROOT/'exp/osft_increment_screen_20261019/v1'
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/edge_deployment_20261019/pas_fit_reference_20261022/v1'
WORK=Path(__file__).parent/'private'
spec=importlib.util.spec_from_file_location('pas_reference_inherited_study',PILOT/'study.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
sha,write=study.sha,study.write

def digest(parameters):
    from training_validity import parameter_digest
    return parameter_digest(list(parameters))

def gradient_directory(run):
    path=run/'gradients';path.mkdir(parents=True,exist_ok=True);return path

def save_first_raw_gradients(model,run,stage):
    import torch
    path=gradient_directory(run)/f'S{stage:02d}_first_raw.pt'
    torch.save({n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None},path)
    return path

def learner_digest(method):
    tensors=[getattr(method.learner,field)[k] for field in ['_I_agg','_A_ref','_B_ref']
             for k in sorted(getattr(method.learner,field))]
    return digest(tensors)

def make_method(model,instructions):
    sys.path.insert(0,str(PORT));from pass_adapter import PASMethod
    class ObservedPAS(PASMethod):
        def __init__(self,model,instructions):
            super().__init__(model,'PASs_MoE',instructions)
            self.stage_records=[];self.live_record=None
        def begin_stage(self,stage):
            assert self.live_record is None
            super().begin_stage(stage)
            self.live_record=dict(stage=stage,recording_true=0,recording_false=0,post_backward=0,
                after_update=0,prior_importance_modules=len(self.learner._I_agg))
            assert all(l.gate_tau==1.5 for l in self.layers)
        def recording(self,enabled):
            super().recording(enabled)
            if self.live_record is not None:self.live_record['recording_true' if enabled else 'recording_false']+=1
        def post_backward(self,scale,progress):
            super().post_backward(scale,progress)
            assert self.live_record is not None
            self.live_record['post_backward']+=1
        def after_update(self,step,total):
            super().after_update(step,total)
            self.live_record['after_update']+=1
        def end_stage(self,stage):
            assert self.live_record['stage']==stage
            self.live_record.update(recorded_usage_callbacks=self.learner._cur_I_steps,
                current_importance_modules=len(self.learner._cur_I_ema),
                regularization_calls_in_task=self.learner._loss_calls_in_task,
                tau_at_end=[l.gate_tau for l in self.layers])
            super().end_stage(stage)
            fields={field:getattr(self.learner,field) for field in ['_I_agg','_A_ref','_B_ref']}
            self.live_record.update(aggregated_modules={f:len(v) for f,v in fields.items()},
                reference_tensor_bytes=sum(t.numel()*t.element_size() for v in fields.values() for t in v.values()),
                reference_tensors_CPU=all(t.device.type=='cpu' for v in fields.values() for t in v.values()))
            self.stage_records.append(self.live_record);self.live_record=None
    return ObservedPAS(model,instructions)

def observed_optimizer(method,factory,records):
    optimizer=factory(method);original_step=optimizer.step
    assert optimizer.__class__.__name__=='HostAdamW' and method.name=='PASs_MoE'
    assert all(g['lr']==1e-4 and g['weight_decay']==0 for g in optimizer.param_groups)
    expected={id(p) for p in method.model.parameters() if p.requires_grad}
    assert {id(p) for g in optimizer.param_groups for p in g['params']}==expected
    step_number=0
    def step(_optimizer,*args,**kwargs):
        nonlocal step_number
        out=original_step(*args,**kwargs);step_number+=1
        parameters=[p for g in optimizer.param_groups for p in g['params']]
        moments=[v for state in optimizer.state.values() for n,v in state.items() if n in ('exp_avg','exp_avg_sq')]
        records.append(dict(stage=method.live_record['stage'],step=step_number,
            optimizer_parameter_IDs_match={id(p) for p in parameters}==expected,
            parameter_tensors=len(parameters),trainable_coordinates=sum(p.numel() for p in parameters),
            gradient_bytes=sum(p.grad.numel()*p.grad.element_size() for p in parameters if p.grad is not None),
            two_moment_bytes=sum(t.numel()*t.element_size() for t in moments),all_moments_CPU=all(t.device.type=='cpu' for t in moments)))
        return out
    optimizer.step=MethodType(step,optimizer)
    return optimizer

def cpu_gate():
    import copy,tempfile,torch
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    from torch import nn
    from transformers import Qwen2Config,Qwen2Model,get_cosine_schedule_with_warmup
    sys.path.insert(0,str(PORT))
    from pass_adapter import PASMethod
    from runtime_utils import make_optimizer
    from checkpoints import save,load
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.model=Qwen2Model(Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,
                num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32))
        def forward(self,ids):return self.model(input_ids=ids,use_cache=False).last_hidden_state
    class Processor:
        def save_pretrained(self,path):
            Path(path).mkdir(parents=True,exist_ok=True);write(Path(path)/'cpu_gate.json',dict(tiny=True))
    torch.manual_seed(42);base=Tiny();first=copy.deepcopy(base);second=copy.deepcopy(base)
    torch.manual_seed(42);original=PASMethod(first,'PASs_MoE',['unused'])
    torch.manual_seed(42);instrumented=make_method(second,['unused'])
    assert list(first.state_dict())==list(second.state_dict())
    assert all(torch.equal(v,second.state_dict()[k]) for k,v in first.state_dict().items())
    assert len(original.layers)==len(instrumented.layers)==7 and instrumented.name=='PASs_MoE'
    assert all(p.dtype==torch.float32 for p in second.parameters() if p.requires_grad)
    frozen=[p for p in second.parameters() if not p.requires_grad];frozen_before=digest(frozen)
    ids=torch.tensor([[1,2,3,4]])
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);original.state_root=root/'original_state';instrumented.state_root=root/'observed_state'
        original.save_checkpoints=instrumented.save_checkpoints=False
        original.model.eval();instrumented.model.eval();state=torch.get_rng_state()
        a=first(ids);after=torch.get_rng_state();torch.set_rng_state(state);b=second(ids)
        assert torch.equal(a,b) and torch.equal(after,torch.get_rng_state())
        optim_records=[]
        for stage in [1,2]:
            original.begin_stage(stage);instrumented.begin_stage(stage)
            oo=make_optimizer(original);io=observed_optimizer(instrumented,make_optimizer,optim_records)
            os=get_cosine_schedule_with_warmup(oo,0,2);isched=get_cosine_schedule_with_warmup(io,0,2)
            first.train();second.train()
            for step in [1,2]:
                oo.zero_grad(set_to_none=True);io.zero_grad(set_to_none=True);state=torch.get_rng_state()
                original.recording(True);loss=first(ids).square().mean();original.recording(False)
                loss.backward();original.post_backward(1.,(step-1)/2)
                after=torch.get_rng_state();torch.set_rng_state(state)
                instrumented.recording(True);loss2=second(ids).square().mean();instrumented.recording(False)
                loss2.backward();instrumented.post_backward(1.,(step-1)/2)
                assert torch.equal(loss,loss2) and torch.equal(after,torch.get_rng_state())
                aa=dict(first.named_parameters());bb=dict(second.named_parameters())
                assert aa.keys()==bb.keys()
                assert all((p.grad is None)==(bb[n].grad is None) and (p.grad is None or torch.equal(p.grad,bb[n].grad)) for n,p in aa.items())
                if stage==1 and step==1:
                    path=save_first_raw_gradients(second,root/'absent_run',1)
                    saved=torch.load(path,weights_only=True,map_location='cpu')
                    current={n:p.grad for n,p in second.named_parameters() if p.requires_grad and p.grad is not None}
                    assert saved.keys()==current.keys() and all(saved[n].dtype==g.dtype and torch.equal(saved[n],g) for n,g in current.items())
                torch.nn.utils.clip_grad_norm_(first.parameters(),1.);torch.nn.utils.clip_grad_norm_(second.parameters(),1.)
                oo.step();io.step();os.step();isched.step();original.after_update(step,2);instrumented.after_update(step,2)
                assert all(torch.equal(p,bb[n]) for n,p in aa.items())
            oo.zero_grad(set_to_none=True);io.zero_grad(set_to_none=True)
            original.end_stage(stage);instrumented.end_stage(stage)
            assert learner_digest(original)==learner_digest(instrumented)
            assert len(instrumented.learner._I_agg)==7
        assert digest(frozen)==frozen_before
        assert all(r['recorded_usage_callbacks']>0 and r['current_importance_modules']==7 for r in instrumented.stage_records)
        instrumented.model.eval();before=second(ids).detach();state_hash=learner_digest(instrumented)
        run=root/'checkpoint_run';run.mkdir()
        meta=save(instrumented,Processor(),run,2,dict(method='PASs_MoE',manifest_sha256='tiny',execution_source_sha256={}))
        restored=load(run);restored['method'].model.eval()
        assert torch.equal(before,restored['method'].model(ids)) and learner_digest(restored['method'])==state_hash
        assert restored['method'].stage_records==instrumented.stage_records
    assert not torch.cuda.is_initialized()
    return dict(passed=True,CUDA_uninitialized_before_and_after=True,tiny_HF_layers=1,target_modules=7,initial_state_forward_RNG_exact=True,
        four_updates_original_and_instrumented_loss_raw_gradients_weights_and_RNG_exact=True,
        two_stage_original_learner_states_exact=True,positive_usage_and_stage_history=True,
        frozen_base_digest_unchanged=True,first_gradient_absent_directory_save_reload_exact=True,
        full_graph_checkpoint_reload_forward_and_history_exact=True,optimizer_evidence=optim_records,
        new_actual_model_or_QA_or_GPU_requests=0)
