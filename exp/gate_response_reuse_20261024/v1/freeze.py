"""One new tiny carrier/gradient/serialization admission; metadata freeze."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,importlib,json,sys,time,traceback
from common import ROOT,HERE,PORT,BASE,REPORT,REFERENCE,SOURCES,sha,write,selection,metadata,parent_reference


def main():
    assert not any((REPORT/n).exists() for n in ('CPU_GATE_STARTED.json','READY.json','FREEZE_FAILURE.json'))
    review=json.loads((REPORT/'SOURCE_REVIEW.json').read_bytes())
    assert review['decision'].startswith('accept') and review['model_admission'] is False
    assert review['protocol_sha256']==sha(REPORT/'PROTOCOL.json')
    for n,v in review['sources'].items():assert sha(HERE/n)==v,n
    write(REPORT/'CPU_GATE_STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    tick=time.monotonic()
    import torch
    from torch.nn import functional as F
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from core import Carrier,initialize,install,digest,layout,cpu_state
    sys.path.insert(0,str(PORT));from host_adam import HostAdamW
    torch.set_num_threads(2);torch.manual_seed(42)
    parent=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    parent.requires_grad_(False);original_digest=digest(parent.state_dict().items());ids=torch.tensor([[2,5,7,9]])
    folder=REPORT/'private_cpu_gate';folder.mkdir(exist_ok=False);evidence={}
    for case in ('Static','Reuse'):
        model=copy.deepcopy(parent);object.__setattr__(model,'core',model)
        values=[initialize(layer.mlp,case,rank=3 if case=='Static' else 2) for layer in model.model.layers]
        install(model,case,values,master=True)
        before=digest(model.state_dict().items());opt=HostAdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,eps=1e-8,weight_decay=0.,betas=(.9,.999))
        x=torch.randn(1,4,16)
        for layer in model.model.layers:
            calls=[];handle=layer.mlp.gate_proj.register_forward_hook(lambda *unused:calls.append(1))
            block=layer.mlp;got=block(x);handle.remove();assert len(calls)==1
            gate=F.linear(x,block.gate_proj.weight)
            up=F.linear(F.linear(x,block.right),block.left)
            if block.row_scale is not None:up=up+gate*block.row_scale
            expected=F.linear(F.silu(gate)*up,block.down_proj.weight)
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
        for step in range(4):
            opt.zero_grad(set_to_none=True);loss=model(ids,labels=ids).loss;assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step()
        assert digest(model.state_dict().items())!=before
        model.requires_grad_(False);output=model(ids).logits.detach()
        package=folder/(case+'.pt');torch.save(dict(state=cpu_state(model),layout=layout(model)),package)
        loaded=torch.load(package,map_location='cpu',weights_only=False)
        reconstructed=copy.deepcopy(parent);object.__setattr__(reconstructed,'core',reconstructed)
        install(reconstructed,case,values);reconstructed.load_state_dict(loaded['state'],strict=True)
        assert layout(reconstructed)==loaded['layout']
        torch.testing.assert_close(reconstructed(ids).logits,output,rtol=0,atol=0)
        assert not any('up_proj' in n for n in loaded['state'])
        evidence[case]=dict(updates=4,one_gate_call=True,physical_up_absent=True,strict_full_Qwen_reload=True,layout=layout(model))
        del model,reconstructed,opt
    assert digest(parent.state_dict().items())==original_digest and not torch.cuda.is_initialized()
    write(REPORT/'CPU_GATE.json',dict(passed=True,seconds=time.monotonic()-tick,tests=evidence,CUDA_initialized=False,
        tiny_FP32_only_not_real_CUDA_FP16_or_quality=True,real_QA=0))
    _,_,selected=selection();m,s=metadata();parent_reference()
    paths=[HERE/n for n in SOURCES]+[REPORT/n for n in ('PROTOCOL.json','NECESSITY_REVIEW.json','SOURCE_REVIEW.json','CPU_GATE.json')]
    paths += list(PORT.glob('*.py'))+[PORT/'execution_policy.json']
    paths += [ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json',ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json']
    paths += [ROOT/t['files']['train']['path'] for t in m['tasks']]
    paths += [p for p in BASE.iterdir() if p.is_file() and p.name!='model.safetensors' and p.stat().st_size<32*1024**2]
    paths += [REFERENCE/n for n in ('VERIFIED.json','init_COMPLETE.json','READY.json','initializations.json','parent.jsonl')]
    runtime={}
    for module in ('transformers.models.qwen2.modeling_qwen2','transformers.models.llava_onevision.modeling_llava_onevision',
        'transformers.models.llava_onevision.processing_llava_onevision','transformers.models.llava_onevision.image_processing_llava_onevision',
        'transformers.models.siglip.modeling_siglip'):
        path=__import__('pathlib').Path(importlib.import_module(module).__file__).resolve();runtime[str(path)]=sha(path)
    files={str(p.relative_to(ROOT)):sha(p) for p in paths};files.update(runtime)
    write(REPORT/'READY.json',dict(files=files,selected=selected,uid=os.getuid(),
        boot_id=__import__('pathlib').Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        interpreter=str(__import__('pathlib').Path(sys.executable).resolve()),model_seconds={'init':1200,'Static':2400,'Reuse':2400},
        teacher_capture_QA=48,new_DEV_generations=200,TEST_QA=0,scientific_sources_frozen=True))


if __name__=='__main__':
    try:main()
    except Exception:
        write(REPORT/'FREEZE_FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True));raise
