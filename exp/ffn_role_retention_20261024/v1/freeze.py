"""A single synthetic structural check, then metadata freeze; no real model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import ast,copy,json,sys,time
from pathlib import Path
from common import ROOT,HERE,PORT,REPORT,BASE,SPLIT,MANIFEST,ANCHOR,SOURCES,sha,write,data,train_ast


def main():
    assert not (REPORT/'CPU_GATE_STARTED.json').exists() and not (REPORT/'READY.json').exists()
    review=json.loads((REPORT/'SOURCE_REVIEW.json').read_bytes())
    assert review['decision'].startswith('accept') and review['no_model_admission'] is True
    for n,h in review['source_sha256'].items():assert sha(HERE/n)==h
    assert review['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
    write(REPORT/'CPU_GATE_STARTED.json',dict(pid=os.getpid(),unix=time.time()))
    started=time.monotonic()
    import torch
    from torch import nn
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from core import replace,restore,fold,digest,layout,state_cpu,FactorLinear
    sys.path.insert(0,str(PORT));from methods import Method
    from host_adam import HostAdamW
    from transformers import get_cosine_schedule_with_warmup
    torch.set_num_threads(2);torch.manual_seed(42)
    parent=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    initial_qv=[];tests={}
    ids=torch.tensor([[2,4,7,9]])
    for case in ('KeepGate','KeepUp'):
        model=copy.deepcopy(parent);object.__setattr__(model,'core',model)
        model.requires_grad_(False);stash,records=replace(model,case,rank=2)
        before=layout(model)['weights']['coefficients'];torch.manual_seed(42);method=Method(model,'SeqLoRA',['tiny'])
        train=[(n,p) for n,p in model.named_parameters() if p.requires_grad];initial_qv.append(digest(train))
        base_before=digest([(n,p) for n,p in model.named_parameters() if not p.requires_grad])
        model.train();opt=HostAdamW([p for _,p in train],lr=1e-4,weight_decay=0,eps=1e-8,betas=(.9,.999))
        schedule=get_cosine_schedule_with_warmup(opt,0,4)
        for step in range(4):
            opt.zero_grad(set_to_none=True);loss=model(ids,labels=ids).loss;assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in train)
            torch.nn.utils.clip_grad_norm_([p for _,p in train],1.);assert opt.param_groups[0]['lr']>0
            opt.step();schedule.step()
        assert digest(train)!=initial_qv[-1],'Tiny q/v optimizer did not update parameters'
        assert base_before==digest([(n,p) for n,p in model.named_parameters() if not p.requires_grad])
        model.eval();a=model(ids).logits.detach();folded=fold(model);b=model(ids).logits.detach()
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        assert len(folded)==4 and layout(model)['weights']['coefficients']==before
        folder=REPORT/'private_cpu_gate';folder.mkdir(exist_ok=True)
        package=folder/(case+'.pt')
        torch.save(dict(case=case,rank=2,state=state_cpu(model),config=model.config.to_dict(),layout=layout(model),state_digest=digest(model.state_dict().items())),package)
        loaded=torch.load(package,map_location='cpu',weights_only=False)
        reload=copy.deepcopy(parent);object.__setattr__(reload,'core',reload);reload.requires_grad_(False)
        zeros=[(torch.zeros(32,2),torch.zeros(2,16)) for _ in range(2)]
        replace(reload,case,rank=2,factors=zeros);reload.load_state_dict(loaded['state'],strict=True)
        assert digest(reload.state_dict().items())==loaded['state_digest'] and layout(reload)==loaded['layout']
        torch.testing.assert_close(reload(ids).logits,b,rtol=1e-5,atol=1e-6)
        tests[case]=dict(count=before,updates=4,loss=float(loss),fold_max_error=float((a-b).abs().max()),physical=True,serialized_strict_reload=True)
        del model,reload
    assert tests['KeepGate']['count']==tests['KeepUp']['count'] and initial_qv[0]==initial_qv[1]
    assert not torch.cuda.is_initialized()
    write(REPORT/'CPU_GATE.json',dict(passed=True,seconds=time.monotonic()-started,tests=tests,qv_initial_exact=True,CUDA_initialized=False,real_QA=0,real_model=0))
    m,s,fit,dev,plan=data();_,train_sha=train_ast()
    paths=[HERE/n for n in SOURCES]+[REPORT/n for n in ('PROTOCOL.md','NECESSITY_REVIEW.json','SOURCE_REVIEW.json','CPU_GATE.json')]
    paths += [p for p in PORT.glob('*.py')]+[PORT/'execution_policy.json',MANIFEST,SPLIT,ANCHOR]
    paths += [ROOT/t['files']['train']['path'] for t in m['tasks']]
    paths += [p for p in BASE.iterdir() if p.is_file() and p.name!='model.safetensors' and p.stat().st_size<32*1024**2]
    ready=dict(files={str(p.relative_to(ROOT)):sha(p) for p in paths},train_AST_sha256=train_sha,
        uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        interpreter=str(Path(sys.executable).resolve()),FIT_QA=1213,DEV_QA=100,TEST_QA=0,updates=87,
        rank=200,design_parent_coefficients=893675552,design_compact_coefficients=816728096,
        source_review_sha256=sha(REPORT/'SOURCE_REVIEW.json'),CPU_GATE_sha256=sha(REPORT/'CPU_GATE.json'))
    write(REPORT/'READY.json',ready)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        write(REPORT/'FREEZE_FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True));raise
