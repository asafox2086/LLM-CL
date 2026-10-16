"""PASs-MoE: unchanged authored router, experts and rank-importance learner."""
import ast
import hashlib
import json
import math
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from methods import ROOT,Method


class LoraLayer:
    """PEFT storage interface only; mathematical methods come from author source."""
    def __init__(self,in_features,out_features):
        self.in_features,self.out_features=in_features,out_features
        self.r,self.lora_alpha,self.scaling={},{},{}
        self.lora_A,self.lora_B,self.lora_dropout=nn.ModuleDict(),nn.ModuleDict(),nn.ModuleDict()
        self.disable_adapters=False


def components():
    folder=ROOT/'code/PASs_MoE'
    hashes={row['path']:row['sha256'] for row in json.loads((folder/'files_sha256.json').read_text())}
    namespace={'torch':torch,'nn':nn,'F':F,'os':os,'math':math,'warnings':warnings,
               'LoraLayer':LoraLayer,'BaseCLearner':object,
               'transpose':lambda weight,fan:weight.T if fan else weight}
    selections={
        'CoIN/peft/tuners/coinmoelora.py':{'CoINMOELoraLayer','CoINMOELoraLinear','CoINMOEExpert','CoINMOELinearA','CoINMOELinearB'},
        'cl_learner/moelora.py':{'moeloraLearner'}}
    for relative,names in selections.items():
        path=folder/'source'/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=hashes[relative]:
            raise ValueError('PASs-MoE upstream hash mismatch: '+relative)
        tree=ast.parse(path.read_text())
        definitions=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future]+definitions,type_ignores=[])),str(path),'exec'),namespace)
    return namespace,json.loads((folder/'source_manifest.json').read_text())['revision']


class PASMethod(Method):
    def __init__(self,model,name,instructions,embedding=None):
        self.model,self.name=model,name
        self.layers,self.embedding=[],None
        source,self.original_revision=components()
        native=source['CoINMOELoraLinear']
        class HostedLinear(native):
            def forward(self,x,**kwargs):
                # Original router uses -1e9 masks (valid for author's BF16),
                # so keep router/expert arithmetic FP32 on these FP16-only GPUs.
                if self.training and torch.is_grad_enabled():
                    with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast(x.device.type,enabled=False):
                        return super().forward(x,**kwargs)
                with torch.autocast(x.device.type,enabled=False):
                    return super().forward(x,**kwargs)
        for parameter in model.parameters():
            parameter.requires_grad=False
        targets={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
        candidates=[(key,value) for key,value in model.named_modules()
            if key.startswith('model.layers.') and key.rsplit('.',1)[-1] in targets and isinstance(value,nn.Linear)]
        for key,base in candidates:
            adapter=HostedLinear('default',base.in_features,base.out_features,r=128,lora_alpha=256,
                lora_dropout=.05,expert_num=2,gate_rank=16,gate_tau=1.5,gate_router='a',gate_tie='A_T')
            adapter.weight,adapter.bias=base.weight,base.bias
            adapter.to(base.weight.device)
            parent,child=key.rsplit('.',1)
            setattr(model.get_submodule(parent),child,adapter)
            self.layers.append(adapter)
        # Preserve the author's effective frozen projector after PEFT wrapping.
        self.state_root=None
        with __import__('tempfile').TemporaryDirectory() as empty:
            args=SimpleNamespace(moelora_orth_lambda=1e-4,moelora_B_reg_scale=.1,
                moelora_importance_top_p=.4,output_dir=empty)
            self.learner=source['moeloraLearner'](args,0,model)

    def set_instruction(self,task):
        pass

    def begin_stage(self,stage):
        self.learner.before_train(stage,self.model)
        for layer in self.layers:
            layer.gate_tau=1.5

    def recording(self,enabled):
        self.learner.enable_record(enabled)

    def post_backward(self,scale,progress):
        parameters=[p for p in self.model.parameters() if p.grad is not None]
        for parameter in parameters:
            parameter.grad.div_(scale)
        self.learner.post_backward(self.model,cl_task_progress=progress)
        for parameter in parameters:
            parameter.grad.mul_(scale)

    def after_update(self,step,total):
        # The authored callback observes max_steps=0 before trainer.train,
        # then initializes its full-task cosine schedule on first callback.
        tau=.7+.5*(1.5-.7)*(1+math.cos(math.pi*min(1,step/total)))
        for layer in self.layers:
            layer.gate_tau=tau

    def end_stage(self,stage):
        if self.state_root is None:
            raise ValueError('PASs-MoE state output directory is not configured')
        self.state_root.mkdir(parents=True,exist_ok=True)
        if not getattr(self,'save_checkpoints',False):
            # Keep mandatory references and importance in RAM between stages;
            # the authored persistence callback does not affect their update.
            save=self.learner._save_cl_state
            self.learner._save_cl_state=lambda task:None
            self.learner._cl_state_root=str(self.state_root)
            try:
                self.learner.after_train(stage,self.model)
            finally:
                self.learner._save_cl_state=save
            return
        # The atomic stage snapshot stores these full states. The author's
        # redundant latest/task copies are temporary to avoid tripling disk use.
        with __import__('tempfile').TemporaryDirectory(dir=self.state_root,prefix='author_state_') as temp:
            self.learner._cl_state_root=temp
            self.learner.after_train(stage,self.model)

    def method_state(self,stage):
        state=super().method_state(stage)
        state['learner']={key:getattr(self.learner,key) for key in
            ['_I_agg','_A_ref','_B_ref','_cur_I_ema','_cur_I_steps','_loss_calls_in_task']}
        state['gate_tau']=[layer.gate_tau for layer in self.layers]
        return state

    def restore_method_state(self,state):
        for key,value in state['learner'].items():
            setattr(self.learner,key,value)
        for layer,tau in zip(self.layers,state['gate_tau']):
            layer.gate_tau=tau

    def export(self,path):
        super().export(path)

    def restore(self,path):
        super().restore(path)
        state=torch.load(Path(path).parent/'training_state.pt',map_location='cpu',weights_only=False)
        self.restore_method_state(state['method_state'])
