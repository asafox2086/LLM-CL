"""Qwen2-VL adapters; SMoLoRA's expert classes and forward are author source."""
import ast
import hashlib
import json
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def make_method(model,name,instructions,embedding=None):
    if name=='TaskGradReuse':
        from task_gradient_adapter import TaskGradReuseMethod
        return TaskGradReuseMethod(model,name,instructions,embedding)
    if name=='CL_MoE':
        from clmoe_adapter import CLMoEMethod
        return CLMoEMethod(model,name,instructions,embedding)
    if name=='ER_LoRA':
        from replay_adapter import ERMethod
        return ERMethod(model,name,instructions,embedding)
    if name=='PaRSP':
        from parsp_adapter import PaRSPMethod
        return PaRSPMethod(model,name,instructions,embedding)
    if name=='FV_guided_Training':
        from fv_adapter import FVMethod
        return FVMethod(model,name,instructions,embedding)
    if name=='Hyper_LLaVA':
        from hyper_adapter import HyperMethod
        return HyperMethod(model,name,instructions,embedding)
    if name=='SpaRTA':
        from sparta_adapter import SpaRTAMethod
        return SpaRTAMethod(model,name,instructions,embedding)
    if name=='ALER':
        from aler_adapter import ALERMethod
        return ALERMethod(model,name,instructions,embedding)
    if name=='OSFT':
        from osft_adapter import OSFTMethod
        return OSFTMethod(model,name,instructions,embedding)
    if name=='PASs_MoE':
        from pass_adapter import PASMethod
        return PASMethod(model,name,instructions,embedding)
    return Method(model,name,instructions,embedding)


class SeqLinear(nn.Module):
    def __init__(self, base, rank=8, alpha=32):
        super().__init__()
        self.weight, self.bias = base.weight, base.bias
        self.A = nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=torch.float32)
        self.B = nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=torch.float32)
        nn.init.kaiming_uniform_(self.A.weight,a=5**0.5)
        nn.init.zeros_(self.B.weight)
        self.dropout = nn.Dropout(0.1)
        self.scale = alpha/rank
    def forward(self,x):
        base = F.linear(x,self.weight,self.bias)
        return base+(self.B(self.A(self.dropout(x.float())))*self.scale).to(base.dtype)


def author_components():
    manifest = json.loads((ROOT/'code/SMoLoRA/source_manifest.json').read_text())
    relative = 'PEFT_SMoLoRA/peft/tuners/smolora.py'
    source = ROOT/'code/SMoLoRA/source'/relative
    expected = next(row['sha256'] for row in json.loads((ROOT/'code/SMoLoRA/files_sha256.json').read_text()) if row['path']==relative)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
        raise ValueError('upstream SMoLoRA source integrity mismatch')
    tree = ast.parse(source.read_text())
    names = {'SMoLoraLinearA','SMoLoraLinearB','SMoExpert'}
    definitions = [node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
    linear = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='SMoLoraLinear')
    forward = next(node for node in linear.body if isinstance(node,ast.FunctionDef) and node.name=='forward')
    namespace = {'torch':torch,'nn':nn,'F':F,'transpose':lambda weight,fan:weight.T if fan else weight}
    # Compile only the reviewed mathematical classes/method, not upstream imports or entry points.
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions+[forward],type_ignores=[])),str(source),'exec'),namespace)
    return namespace,manifest['revision'],expected


class AuthorSMoLinear(nn.Module):
    def __init__(self,base,embedding,components):
        super().__init__()
        self.weight,self.bias = base.weight,base.bias
        self.in_features,self.out_features = base.in_features,base.out_features
        self.active_adapter='default'
        self.expert_num=8
        self.ins_emb=embedding
        self.ins_type=0
        self.fan_in_fan_out=False
        self.disable_adapters=False
        self.merged=False
        self.r={'default':128}
        self.scaling={'default':2.0}
        self.lora_A=nn.ModuleDict({'default':components['SMoLoraLinearA'](base.in_features,128,8)})
        self.lora_B=nn.ModuleDict({'default':components['SMoLoraLinearB'](128,base.out_features,8)})
        self.lora_dropout=nn.ModuleDict({'default':nn.Dropout(0.05)})
        self.lora_vu_gate=nn.ModuleDict({'default':nn.Linear(base.in_features,4,bias=False)})
        self.lora_ins_gate=nn.ModuleDict({'default':nn.Linear(384,4,bias=False)})
        self.lora_fc_A=nn.ModuleDict({'default':nn.Linear(base.out_features,1,bias=False)})
        self.lora_fc_B=nn.ModuleDict({'default':nn.Linear(base.out_features,1,bias=False)})
        for expert in self.lora_A['default'].loraA:
            nn.init.normal_(expert.mlp.weight,mean=0,std=0.01)
        for expert in self.lora_B['default'].loraB:
            nn.init.zeros_(expert.mlp.weight)
        self.to(base.weight.device)
        self._author_forward=components['forward']
    def forward(self,x,**kwargs):
        # The upstream forward materializes all eight experts, including those
        # masked by hard routing. Offload their saved autograd tensors to host RAM
        # to fit the original mechanism on 11 GiB cards. Decoder checkpoints retain
        # dropout RNG; the author forward and all expert parameters remain intact.
        if self.training and torch.is_grad_enabled() and x.requires_grad:
            # Place the hook inside the projection: an outer save_on_cpu context
            # is overridden by decoder checkpoint hooks during recomputation.
            # This local hook also applies on that recomputation path.
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                return self._author_forward(self,x,**kwargs)
        return self._author_forward(self,x,**kwargs)


def instruction_embeddings(instructions):
    from transformers import AutoTokenizer,AutoModel
    path = ROOT/'model/all-MiniLM-L6-v2'
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    encoder=AutoModel.from_pretrained(path,local_files_only=True).eval()
    data=tokenizer(instructions,padding=True,truncation=True,return_tensors='pt')
    with torch.no_grad():
        features=encoder(**data)[0]
        mask=data['attention_mask'].unsqueeze(-1).float()
        embedding=(features*mask).sum(1)/mask.sum(1).clamp_min(1e-9)
    return embedding.tolist()


class Method:
    def __init__(self,model,name,instructions,embedding=None):
        self.model,self.name=model,name
        self.layers=[]
        self.original_revision=None
        for parameter in model.parameters():
            parameter.requires_grad=False
        if name=='SMoLoRA':
            components,self.original_revision,self.source_hash=author_components()
            self.embedding=embedding if embedding is not None else instruction_embeddings(instructions)
            targets={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
        elif name=='SeqLoRA':
            targets={'q_proj','v_proj'}
            self.embedding=None
        else:
            raise ValueError('No verified adapter implemented for '+name)
        candidates=[(key,value) for key,value in model.named_modules()
                    if key.startswith('model.layers.') and key.rsplit('.',1)[-1] in targets and isinstance(value,nn.Linear)]
        for key,base in candidates:
            parent_name,child=key.rsplit('.',1)
            parent=model.get_submodule(parent_name)
            adapter=AuthorSMoLinear(base,self.embedding,components) if name=='SMoLoRA' else SeqLinear(base)
            setattr(parent,child,adapter)
            self.layers.append(adapter)
        if not self.layers:
            raise ValueError('Qwen2-VL language projection mapping failed')
        # Author add_adapter calls mark_only_lora_as_trainable after wrapping:
        # the visual projector is frozen despite a projector-lr CLI argument.
        # Preserve that effective source behavior, not the unused argument.

    def set_instruction(self,task):
        if self.name=='SMoLoRA':
            for layer in self.layers:
                layer.ins_type=task-1

    def begin_stage(self,stage):
        if self.name=='SMoLoRA':
            for layer in self.layers:
                for gate in [layer.lora_vu_gate,layer.lora_ins_gate]:
                    for parameter in gate.parameters():
                        parameter.requires_grad=(stage==1)
        self.set_instruction(stage)

    def changed_state(self):
        # Includes frozen gates and all experts, not just the current trainable subset.
        result,storage = {},{}
        for key,value in self.model.state_dict().items():
            if '.lora_' in key or key.endswith('.A.weight') or key.endswith('.B.weight'):
                identity=(value.data_ptr(),tuple(value.shape),value.dtype)
                if identity not in storage:
                    storage[identity]=value.detach().cpu().clone()
                result[key]=storage[identity]
        return result

    def export(self,path):
        torch.save(self.changed_state(),Path(path)/'changed_parameters.pt')
        (Path(path)/'method.json').write_text(json.dumps({'method':self.name,'upstream_revision':self.original_revision,
              'qwen2_vl_adaptation':True,'instruction_embeddings':self.embedding},indent=2)+'\n')

    def restore(self,path):
        state=torch.load(Path(path)/'changed_parameters.pt',map_location='cpu',weights_only=True)
        own=self.model.state_dict()
        if not state or any(key not in own or own[key].shape!=value.shape for key,value in state.items()):
            raise ValueError('incompatible changed parameter snapshot')
        self.model.load_state_dict(state,strict=False)

    def method_state(self,stage):
        return {'method':self.name,'completed_stage':stage,'instruction_embeddings':self.embedding,
                'precision':'author-native BF16 on Ampere GPU' if self.name=='SMoLoRA' else 'FP16 base, FP32 adapters',
                'upstream_revision':self.original_revision,'router_frozen_after_first_stage':self.name=='SMoLoRA'}
