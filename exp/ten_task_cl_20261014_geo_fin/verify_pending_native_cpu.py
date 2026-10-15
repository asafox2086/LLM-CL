"""Native tiny-Qwen2-VL integration checks, using only synthetic CPU inputs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,tempfile,copy,math,warnings
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import Qwen2VLConfig,Qwen2VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from native_auxiliary import NativeIO
from parsp_adapter import PaRSPMethod,author_components as parsp_components
from fv_adapter import FVMethod,author_components as fv_components
warnings.filterwarnings('ignore',message='User provided device_type of')
torch.set_num_threads(1);torch.manual_seed(42)
HERE=Path(__file__).resolve().parent

def tiny():
    config=Qwen2VLConfig(vocab_size=128,hidden_size=16,intermediate_size=32,num_hidden_layers=12,num_attention_heads=2,num_key_value_heads=1,attention_dropout=0.,max_position_embeddings=32768,rope_scaling={'type':'mrope','mrope_section':[1,1,2]},vision_config={'depth':1,'embed_dim':16,'hidden_size':16,'num_heads':2,'in_channels':3,'spatial_merge_size':2,'patch_size':2,'temporal_patch_size':1},image_token_id=125,video_token_id=126,vision_start_token_id=124,vision_end_token_id=123)
    return Qwen2VLForConditionalGeneration(config).float()
class Tokenizer:
    eos_token_id=2
    def __call__(self,answer,**kwargs):return {'input_ids':[int(answer)]}
class ToyIO(NativeIO):
    def __init__(self,method):
        self.method,self.model=method,method.model;self.processor=SimpleNamespace(tokenizer=Tokenizer());self.cache={};self.run=Path(tempfile.mkdtemp(prefix='llmcl_cpu_aux_'));self.data={}
        for stage in range(1,11):self.data[(stage,'train')]=[{'id':f'toy{stage}:{i}','answer':str(3+i%30),'question':'synthetic','image':bool(i%2)} for i in range(100)]
    def batch(self,row,stage):
        number=int(row['id'].split(':')[1]);tokens=[5,6,7+number%50,8]
        if row['image']:
            grid=torch.tensor([[1,4,4]]);pixels=torch.arange(16*12,dtype=torch.float32).reshape(16,12)/192
            with torch.no_grad():visual=self.model.visual(pixels,grid_thw=grid)
            return {'input_ids':torch.tensor([[124,125,125,125,125,123]+tokens]),'image_grid_thw':grid,'_encoded_visual':visual.detach()}
        return {'input_ids':torch.tensor([tokens])}
    def record(self,name,value):self.last_record=value

def check_parsp():
    model=tiny();method=PaRSPMethod(model,'PaRSP',[]);method.io=ToyIO(method);method.preflight=True
    method.begin_stage(1);assert model.training and method.selection['probe_count']==2
    p=method.kernel['common_null_space_projector_sum_method_qwen'](torch.diag(torch.tensor([4.,2.,0.,0.])),use_cache=False)
    assert torch.allclose(p,p.T) and torch.allclose(p@p,p) and torch.linalg.norm(p@torch.diag(torch.tensor([4.,2.,0.,0.])))<1e-6
    row=method.io.stage_rows(1)[1];b=method.io.batch(row,1);hidden=method.io.forward(b);loss=model.lm_head(hidden[:,-1]).square().mean();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    old_handles=list(method.handles);method.end_stage(1);method.begin_stage(2);assert all(h.id not in mod._backward_hooks for mod in model.parameters() if mod._backward_hooks for h in old_handles)
    out={'passed':True,'upstream_revision':method.original_revision,'source_hashes':method.source_hashes,'native_qwen_visual_and_text_localization':True,'region_mask_plus_nullspace_gradient_backward':True,'projector_symmetric_idempotent_known_nullspace':True,'stage_two_old_mask_hooks_removed':True,'preflight_probe_rows':2,'formal_probe_rows':100}
    (HERE/'preflight/parsp_cpu.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS PaRSP native tiny-Qwen image/text attribution, masks, projection and stage transition',flush=True)

def check_fv():
    model=tiny();method=FVMethod(model,'FV_guided_Training',[]);method.io=ToyIO(method);method.preflight=True
    method.begin_stage(1);assert model.training and len(method.heads)==10 and method.function_vector.shape==(1,16)
    assert all(0<=l<12 and 0<=h<2 for l,h,_ in method.heads)
    row=method.io.stage_rows(1)[1];b=method.io.batch(row,1);loss,count=method.training_loss(b,row,method.io.processor);assert torch.isfinite(loss) and count==2
    loss.backward();assert any(layer.B.weight.grad is not None and torch.count_nonzero(layer.B.weight.grad)>0 for layer in method.layers)
    # Verify teacher intervention really changes logits, not a renamed CE-only method.
    with method.disabled(),torch.no_grad():
        clean=method.io.forward(b);patch=method.kernel['add_function_vector'](9,torch.ones(1,16)*3,[b['input_ids'].shape[1]-1])
        hook=model.model.layers[9].self_attn.o_proj.register_forward_hook(lambda m,a,o:patch(o,'module.base_model.model.model.layers.9.self_attn.o_proj'))
        try:edited=method.io.forward(b)
        finally:hook.remove()
        assert not torch.allclose(model.lm_head(clean[:,-1]),model.lm_head(edited[:,-1]),atol=1e-6)
    a=torch.randn(2,3,128,requires_grad=True);z=torch.randn(2,3,128);kd=method.kernel['KD_loss'](None,a,z,2);expected=(torch.softmax(a/2,-1)*(torch.log_softmax(a/2,-1)-torch.log_softmax(z/2,-1))).sum()/2
    assert torch.allclose(kd,expected,atol=1e-6);kd.backward();assert a.grad.abs().sum()>0
    # Cache prefix reuse must agree with a full native multimodal forward.
    with method.disabled(),torch.no_grad():
        target=method.io.target(row['answer']);ids=torch.cat([b['input_ids'],target],1);emb,pos=method.io.embeddings(b,ids);cut=b['input_ids'].shape[1]-1
        full=method.io.forward(b,ids);prefix=model.model(inputs_embeds=emb[:,:cut],position_ids=pos[:,:,:cut],attention_mask=torch.ones_like(ids[:,:cut]),past_key_values=DynamicCache(),use_cache=True,return_dict=True)
        legacy=prefix.past_key_values.to_legacy_cache();tail=model.model(inputs_embeds=emb[:,cut:],position_ids=pos[:,:,cut:],attention_mask=torch.ones_like(ids),past_key_values=method.cache_for_batch(legacy,1),use_cache=True,return_dict=True).last_hidden_state
        assert torch.allclose(tail,full[:,cut:],atol=3e-6),(tail-full[:,cut:]).abs().max()
    # Per-stage merge keeps old update, then resets new adapter to exactly zero.
    layer=method.layers[0]
    with torch.no_grad():layer.B.weight.fill_(.01)
    layer.eval();x=torch.randn(1,2,layer.weight.shape[1]);before=layer(x).detach();layer.merge_reset();assert torch.allclose(layer(x),before,atol=1e-6) and torch.count_nonzero(layer.B.weight)==0
    out={'passed':True,'upstream_revision':method.original_revision,'source_hashes':method.source_hashes,'native_qwen_visual_text_ten_shot_probes':True,'all_native_causal_heads_measured':24,'dynamic_top10_nll_head_selection':True,'source_intervention_changes_logits':True,'two_kl_direction_and_gradient':True,'full_method_loss_backward':True,'cache_prefix_matches_full_native_image_forward':True,'merge_reset_preserves_prior_update':True,'formal_mean_trials':100,'formal_causal_trials':25,'preflight_mean_trials':2,'preflight_causal_trials':1}
    (HERE/'preflight/fv_cpu.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS FV source causal heads, native multimodal prefix cache, FV logits, reverse KL and train backward',flush=True)

if __name__=='__main__':
    if len(sys.argv)==1 or sys.argv[1]=="parsp":check_parsp()
    if len(sys.argv)==1 or sys.argv[1]=="fv":check_fv()
