"""One tiny structural/statistics/serialization gate, then metadata freeze; no real QA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,importlib,json,sys,time,traceback
from common import ROOT,HERE,PORT,BASE,REPORT,REFERENCE,SOURCES,sha,write,selection,metadata,parent_reference,BUDGETS


def main():
    assert not any((REPORT/n).exists() for n in ('CPU_GATE_STARTED.json','READY.json','FREEZE_FAILURE.json'))
    review=json.loads((REPORT/'SOURCE_REVIEW.json').read_bytes())
    assert review['decision'].startswith('accept') and review['model_admission'] is False
    assert review['protocol_sha256']==sha(REPORT/'PROTOCOL.json')
    for n,v in review['sources'].items():assert sha(HERE/n)==v,n
    write(REPORT/'CPU_GATE_STARTED.json',dict(pid=os.getpid(),unix=time.time()));tick=time.monotonic()
    import torch
    from torch.nn import functional as F
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from core import initialize,install,digest,layout,cpu_state,add_qa_moment
    torch.set_num_threads(2);torch.manual_seed(42);assert not torch.cuda.is_initialized()
    parent=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    parent.requires_grad_(False);original_digest=digest(parent.state_dict().items());ids=torch.tensor([[2,5,7,9]])
    x=torch.randn(1,4,16);parts={arm:[] for arm in ('WeightD','ResponseD')};checks=[]
    for layer in parent.model.layers:
        block=layer.mlp;z=F.silu(block.gate_proj(x))*block.up_proj(x);hh=block.down_proj(z)[0].detach()
        moment=torch.zeros(16,16);add_qa_moment(moment,hh[:1],qa_count=2);add_qa_moment(moment,hh[1:],qa_count=2)
        expected=(hh[:1].T@hh[:1]+hh[1:].T@hh[1:]/3)/2
        torch.testing.assert_close(moment,expected,rtol=1e-6,atol=1e-12)
        values,meta=initialize(block.down_proj.weight,moment,rank=3)
        ev,l=torch.linalg.eigh((moment+moment.T)*.5);l=l[:,-3:].flip(1)
        left,right=values['ResponseD']
        torch.testing.assert_close(left@right,l@(l.T@block.down_proj.weight),rtol=2e-5,atol=1e-7)
        for arm,pair in values.items():parts[arm].append(pair)
        checks.append(dict(QA_equal_weight_unequal_position_counts=True,balanced_same_projected_D=True,moment_dtype=meta['moment_dtype']))
    folder=REPORT/'private_cpu_gate';folder.mkdir(exist_ok=False);evidence={}
    for arm,values in parts.items():
        model=copy.deepcopy(parent);object.__setattr__(model,'core',model);install(model,values)
        model.requires_grad_(False);output=model(ids).logits.detach()
        for layer in model.model.layers:
            block=layer.mlp;got=block(x);z=F.silu(block.gate_proj(x))*block.up_proj(x)
            expected=F.linear(F.linear(z,block.right),block.left)
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
        package=folder/(arm+'.pt');torch.save(dict(state=cpu_state(model),layout=layout(model)),package)
        loaded=torch.load(package,map_location='cpu',weights_only=False)
        reconstructed=copy.deepcopy(parent);object.__setattr__(reconstructed,'core',reconstructed)
        install(reconstructed,values);reconstructed.load_state_dict(loaded['state'],strict=True)
        assert layout(reconstructed)==loaded['layout'] and not any('down_proj' in n for n in loaded['state'])
        torch.testing.assert_close(reconstructed(ids).logits,output,rtol=0,atol=0)
        assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        evidence[arm]=dict(optimizer_updates=0,physical_D_absent=True,strict_full_Qwen_reload=True,layout=layout(model))
    assert digest(parent.state_dict().items())==original_digest and not torch.cuda.is_initialized()
    write(REPORT/'CPU_GATE.json',dict(passed=True,seconds=time.monotonic()-tick,tests=evidence,statistics_checks=checks,
          CUDA_initialized=False,tiny_FP32_only_not_real_CUDA_FP16_or_quality=True,real_QA=0))
    _,_,selected=selection();m,s=metadata();parent_reference()
    paths=[HERE/n for n in SOURCES]+[REPORT/n for n in ('PROTOCOL.json','NECESSITY_REVIEW.json','SOURCE_REVIEW.json','CPU_GATE.json')]
    paths+=list(PORT.glob('*.py'))+[PORT/'execution_policy.json']
    paths+=[ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json',ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json']
    paths+=[ROOT/t['files']['train']['path'] for t in m['tasks']]
    paths+=[p for p in BASE.iterdir() if p.is_file() and p.name!='model.safetensors' and p.stat().st_size<32*1024**2]
    paths+=[REFERENCE/n for n in ('VERIFIED.json','init_COMPLETE.json','READY.json','initializations.json','parent.jsonl')]
    runtime={}
    for module in ('transformers.models.qwen2.modeling_qwen2','transformers.models.llava_onevision.modeling_llava_onevision',
        'transformers.models.llava_onevision.processing_llava_onevision','transformers.models.llava_onevision.image_processing_llava_onevision','transformers.models.siglip.modeling_siglip'):
        path=__import__('pathlib').Path(importlib.import_module(module).__file__).resolve();runtime[str(path)]=sha(path)
    files={str(p.relative_to(ROOT)):sha(p) for p in paths};files.update(runtime)
    write(REPORT/'READY.json',dict(files=files,selected=selected,uid=os.getuid(),
        boot_id=__import__('pathlib').Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        interpreter=str(__import__('pathlib').Path(sys.executable).resolve()),model_seconds=BUDGETS,
        teacher_capture_QA=48,new_DEV_generations=200,TEST_QA=0,scientific_sources_frozen=True))


if __name__=='__main__':
    try:main()
    except Exception:
        write(REPORT/'FREEZE_FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True));raise
