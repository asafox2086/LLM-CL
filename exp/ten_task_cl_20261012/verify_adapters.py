"""Tiny genuine Qwen2-VL text/image updates and exact resume checks, CPU only."""
import json
import tempfile
from pathlib import Path
import torch
from transformers import Qwen2VLConfig,Qwen2VLForConditionalGeneration
from methods import Method,ROOT
from run import answer_loss
from stage_checkpoint import save_stage,load_stage
from frozen_vision import install


class Tokenizer:
    eos_token_id=11
    def __call__(self,*args,**kwargs):
        return {'input_ids':[9,10]}
class Processor:
    tokenizer=Tokenizer()


def main():
    torch.set_num_threads(2)
    results={}
    for name in ['SeqLoRA','SMoLoRA']:
        torch.manual_seed(42)
        config=Qwen2VLConfig(vocab_size=256,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
            rope_scaling={'type':'mrope','mrope_section':[1,1,2]},
            image_token_id=200,video_token_id=201,vision_start_token_id=197,vision_end_token_id=198,
            vision_config={'depth':1,'embed_dim':32,'hidden_size':32,'num_heads':4,
                           'mlp_ratio':2,'patch_size':14,'temporal_patch_size':2,'spatial_merge_size':2})
        model=Qwen2VLForConditionalGeneration(config).float()
        text={'input_ids':torch.tensor([[2,3,4]]),'attention_mask':torch.ones(1,3,dtype=torch.long)}
        model.eval()
        with torch.no_grad():
            baseline=model(**text).logits.clone()
        method=Method(model,name,['instruction']*10,embedding=[[.1]*384]*10)
        assert not any(p.requires_grad for p in model.visual.parameters())
        method.begin_stage(1)
        model.eval()
        with torch.no_grad():
            initial=model(**text).logits
        assert torch.allclose(baseline,initial,atol=1e-6,rtol=1e-5)
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=1e-3)
        image={'input_ids':torch.tensor([[197,200,200,200,200,198,2,3]]),
               'attention_mask':torch.ones(1,8,dtype=torch.long),
               'pixel_values':torch.randn(16,3*2*14*14),'image_grid_thw':torch.tensor([[1,4,4]])}
        def step(batch):
            model.train()
            optimizer.zero_grad()
            loss,_=answer_loss(model,batch,'answer',Processor())
            assert torch.isfinite(loss)
            loss.backward()
            assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in params)
            optimizer.step()
            return float(loss)
        text_loss=step(text)
        image_loss=step(image)
        model.eval()
        with torch.no_grad():
            raw_loss,_=answer_loss(model,image,'answer',Processor())
            features=model.visual(image['pixel_values'],grid_thw=image['image_grid_thw'])
            install(model)
            cached=dict(image,pixel_values=features)
            cached_loss,_=answer_loss(model,cached,'answer',Processor())
            assert torch.equal(raw_loss,cached_loss)
        with tempfile.TemporaryDirectory(prefix='llmcl_adapter_') as temp:
            path=save_stage(temp,1,method.export,{'optimizer':optimizer.state_dict(),'scheduler':None,'scaler':None,
                            'method_state':method.method_state(1),'global_step':2},
                            {'synthetic':True},{'task_scores':[0.]*10})
            step(text)
            expected={k:v.clone() for k,v in model.state_dict().items()}
            load_stage(path,method.restore,optimizer)
            step(text)
            for key,value in model.state_dict().items():
                assert torch.equal(value,expected[key]),(name,key)
        results[name]={'tiny_qwen_text_update':True,'tiny_qwen_real_image_update':True,
                       'zero_adapter_matches_base':True,'next_adam_update_matches_after_resume':True,
                       'frozen_visual_cache_loss_matches_raw_image':True,
                       'text_loss':text_loss,'image_loss':image_loss,
                       'full_2b_gpu_preflight_verified':False}
    (ROOT/'exp/ten_task_cl_20261012/adapter_verification.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
