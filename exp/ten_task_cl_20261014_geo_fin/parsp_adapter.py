"""Author PaRSP localization, region masks and nullspace gradients; no QA replay."""
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import gc,os,json,time,torch
from torch import nn
from tqdm import tqdm
from native_auxiliary import NativeIO,author_nodes,ROOT

class EighCompatibility:
    @staticmethod
    def eigh(matrix,*args,**kwargs):
        # MKL SSYEVD's float32 workspace query rounds a required lwork down for
        # the native 8960x8960 covariance. Same eigendecomposition in float64
        # avoids that backend error; return float32 eigenpairs to author code.
        if matrix.device.type=='cpu' and matrix.shape[-1]>4096:
            print(json.dumps({'event':'parsp_eigh_backend','shape':list(matrix.shape),'workaround':'float64 CPU eigensolver; float32 projection unchanged'}),flush=True)
            values,vectors=torch.linalg.eigh(matrix.double(),*args,**kwargs)
            return values.to(matrix.dtype),vectors.to(matrix.dtype)
        return torch.linalg.eigh(matrix,*args,**kwargs)
class TorchEighCompatibility:
    linalg=EighCompatibility()
    def __getattr__(self,name):return getattr(torch,name)

def author_components():
    env={'torch':torch,'nn':nn,'tqdm':tqdm,'defaultdict':defaultdict,'os':os,'gc':gc,'PreTrainedTokenizerBase':object}
    hashes={}
    def load(path,names,transform=None):hashes[path]=author_nodes('PaRSP',path,set(names),env,transform)
    load('parsp/utils/common.py',['get_prob'])
    load('parsp/methods/utils.py',['transfer_output_qwen','get_bsvalues_qwen'])
    # The author localizer consumes only the final token; transfer that slice only.
    load('parsp/models/wrapper.py',['QwenIntermediatesWrapper'],lambda s:s.replace('data.detach().cpu()','(data[:, -1:, :] if data.ndim==3 else data).detach().cpu()'))
    load('parsp/methods/localization.py',['NeuronLocalizationQwen'])
    load('parsp/methods/masking.py',['layer_division_qwen','neuron_division_qwen'])
    projection_env=dict(env);projection_env['torch']=TorchEighCompatibility()
    rel='parsp/methods/projection.py'
    hashes[rel]=author_nodes('PaRSP',rel,{'common_null_space_projector_sum_method_qwen'},projection_env)
    env['common_null_space_projector_sum_method_qwen']=projection_env['common_null_space_projector_sum_method_qwen']
    revision=json.loads((ROOT/'code/PaRSP/source_manifest.json').read_text())['revision']
    return env,revision,hashes

class ProbeView(nn.Module):
    def __init__(self,io):
        super().__init__();object.__setattr__(self,'io',io);self.model=io.model.model;self.lm_head=io.model.lm_head;self.config=io.model.config;self.pending=None
    @property
    def device(self):return self.io.device
    def forward(self,ids,**kwargs):
        hidden=self.io.forward(self.pending,ids)
        return (self.lm_head(hidden[:,-1:]).float(),None)

class ProbeTokenizer:
    def __init__(self,io,view,rows,stage):self.io,self.view,self.rows,self.stage=io,view,{str(i):row for i,row in enumerate(rows)},stage
    def __call__(self,text,**kwargs):
        row=self.rows[text.split('#',1)[0]];b=self.io.batch(row,self.stage);self.view.pending=b
        # Same final-answer-token attribution as author, with intact native image tokens.
        ids=torch.cat([b['input_ids'],self.io.target(row['answer'])[:,:-1]],dim=1)
        return SimpleNamespace(input_ids=ids)

class PaRSPMethod:
    def __init__(self,model,name,instructions,embedding=None):
        self.model,self.name=model,name;self.learning_rate=3e-5;self.host_optimizer=True;self.offload_activations=True;self.stage=0
        self.kernel,self.original_revision,self.source_hashes=author_components();self.handles=[];self.history={};self.io=None;self.selection=None
        for p in model.parameters():p.requires_grad=False
    def bind_context(self,processor,manifest,data,run,preflight=False):self.io=NativeIO(self,processor,manifest,data,run);self.preflight=preflight
    def set_instruction(self,task):pass
    def group_modules(self):
        for i,l in enumerate(self.model.model.layers):
            yield f'{i}:qkv',l.self_attn.q_proj,[l.self_attn.q_proj,l.self_attn.k_proj,l.self_attn.v_proj]
            yield f'{i}:o',l.self_attn.o_proj,[l.self_attn.o_proj]
            yield f'{i}:gateup',l.mlp.gate_proj,[l.mlp.gate_proj,l.mlp.up_proj]
            yield f'{i}:down',l.mlp.down_proj,[l.mlp.down_proj]
    @torch.no_grad()
    def covariance(self,rows,stage):
        sums={};counts=defaultdict(int);hooks=[]
        def hook(key):
            def capture(module,args):
                x=args[0].detach().reshape(-1,args[0].shape[-1]).float()
                assert torch.isfinite(x).all()
                with torch.autocast('cuda',enabled=False):c=(x.T@x).cpu()
                if key not in sums:sums[key]=c
                else:sums[key].add_(c)
                counts[key]+=x.shape[0]
            return capture
        for key,module,_ in self.group_modules():hooks.append(module.register_forward_pre_hook(hook(key)))
        try:
            for i,row in enumerate(rows):
                with torch.autocast('cuda',dtype=torch.float16):self.io.forward(self.io.batch(row,stage))
                if (i+1)%10==0:print(json.dumps({'event':'parsp_covariance','stage':stage,'rows':i+1,'total':len(rows)}),flush=True)
        finally:
            for h in hooks:h.remove()
        return {k:v/counts[k] for k,v in sums.items()}
    def begin_stage(self,stage):
        assert self.io is not None
        if stage==self.stage:return
        self.model.zero_grad(set_to_none=True)
        for h in self.handles:h.remove()
        self.handles=[];self.projectors={};self.model.eval();self.io.clear();rows=self.io.stage_rows(stage);assert len(rows)==100
        if self.preflight:rows=rows[:2]
        started=time.monotonic()
        with self.kernel['QwenIntermediatesWrapper'](ProbeView(self.io)) as wrapper:
            view=wrapper.model;tok=ProbeTokenizer(self.io,view,rows,stage);loc=self.kernel['NeuronLocalizationQwen'](wrapper,tok,str(self.model.device))
            prompts=[str(i)+'#' for i in range(len(rows))];answers=[row['answer'] for row in rows]
            attn,ffn=loc.get_top_attribute_layers(prompts,answers,layer_ratio=.38,max_length=4096)
            neurons_ffn=loc.get_top_attribute_neurons_ffn(prompts,answers,ffn,neuron_ratio=.4,batch_size=128,max_length=4096)
            neurons_attn=loc.get_top_attribute_neurons_attn(prompts,answers,attn,neuron_ratio=.4,batch_size=128,max_length=4096)
        # No external MMLU/classification calibration is introduced. Formal stage1
        # has no previous task, so the author's missing-covariance branch is a no-op.
        # Hardware preflight seeds throwaway covariances to exercise projection too.
        if self.preflight and not self.history:self.history=self.covariance(rows,stage)
        for i,layer in enumerate(self.model.model.layers):
            if i in attn:layer.self_attn.float()
            if i in ffn:layer.mlp.float()
        # Save exact author region-mask hook handles so masks do not accumulate across tasks.
        params=list(self.model.parameters());original={}
        for p in params:
            original[id(p)]=p.register_hook
            def register(fn,p=p):
                h=original[id(p)](fn);self.handles.append(h);return h
            p.register_hook=register
        try:self.kernel['neuron_division_qwen'](self.model,attn,ffn,neurons_ffn,neurons_attn)
        finally:
            for p in params:del p.register_hook
        for key,_,modules in self.group_modules():
            selected=[m for m in modules if m.weight.requires_grad]
            if not selected or key not in self.history:continue
            P=self.kernel['common_null_space_projector_sum_method_qwen'](self.history[key],rcond=1e-6,min_ratio=.2,use_cache=False,layer_name=key)
            self.projectors[key]=P
            for module in selected:
                def project(grad,P=P):
                    with torch.autocast('cuda',enabled=False):return grad@P.to(device=grad.device,dtype=grad.dtype)
                self.handles.append(module.weight.register_hook(project))
        self.stage=stage;self.selection={'stage':stage,'attention_layers':attn,'ffn_layers':ffn,'ffn_neurons':neurons_ffn,'attention_neurons':neurons_attn,'auxiliary_train_ids':[x['id'] for x in rows],'seconds':time.monotonic()-started,'layer_ratio':.38,'neuron_ratio':.4,'probe_count':len(rows),'nullspace_min_ratio':.2,'preflight_only':self.preflight,'protected_history_covariance':bool(self.projectors),'historical_qa_replayed':0,'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad)}
        self.io.record(f'parsp_stage_{stage:02d}.json',self.selection);self.model.train();print(json.dumps({'event':'parsp_localization_ready','stage':stage,'seconds':self.selection['seconds'],'trainable_parameters':self.selection['trainable_parameters']}),flush=True)
    @torch.no_grad()
    def end_stage(self,stage):
        self.model.zero_grad(set_to_none=True)
        started=time.monotonic();self.model.eval();cov=self.covariance(self.io.stage_rows(stage),stage)
        for k,v in cov.items():
            if k in self.history:self.history[k].add_(v)
            else:self.history[k]=v
        self.io.record(f'parsp_covariance_stage_{stage:02d}.json',{'stage':stage,'qa_count':100,'qa_ids':[x['id'] for x in self.io.stage_rows(stage)],'seconds':time.monotonic()-started,'covariance_groups':{k:list(v.shape) for k,v in cov.items()},'saved_to_disk':False,'historical_qa_replayed':0})
        # Positive scalar normalization cannot change author relative-eigenvalue projector.
        self.io.clear()
    def method_state(self,stage):return {'method':self.name,'stage':stage,'upstream_revision':self.original_revision,'history_covariance_in_ram':True,'author_region_mask_and_projector':True,'source_hashes':self.source_hashes}
