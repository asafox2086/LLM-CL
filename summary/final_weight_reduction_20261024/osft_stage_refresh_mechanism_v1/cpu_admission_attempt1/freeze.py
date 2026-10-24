"""One necessary tiny shared-prefix/branch/RNG/state admission, never real QA."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy,importlib,json,sys,time,traceback,random
from pathlib import Path
from common import ROOT,HERE,PORT,BASE,REPORT,SOURCES,sha,write,data,training_prefix,BUDGETS


def main():
    assert not any((REPORT/n).exists() for n in ('CPU_GATE_STARTED.json','READY.json','FREEZE_FAILURE.json'))
    review=json.loads((REPORT/'SOURCE_REVIEW.json').read_bytes())
    assert review['decision'].startswith('accept') and not review['model_admission']
    assert review['protocol_sha256']==sha(REPORT/'PROTOCOL.json')
    for n,v in review['sources'].items():assert sha(HERE/n)==v
    write(REPORT/'CPU_GATE_STARTED.json',dict(pid=os.getpid(),unix=time.time()));tick=time.monotonic()
    import numpy as np
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from core import StageMethod,digest,cpu_state,layout,rng_digest
    sys.path.insert(0,str(PORT));from runtime_utils import make_optimizer
    torch.set_num_threads(2);torch.manual_seed(42);random.seed(42);np.random.seed(42)
    assert not torch.cuda.is_initialized()
    parent=Qwen2ForCausalLM(Qwen2Config(vocab_size=31,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2)).eval()
    parent.requires_grad_(False);original=digest(parent.state_dict().items());ids=torch.tensor([[2,5,7,9]])
    prefix=copy.deepcopy(parent);object.__setattr__(prefix,'core',prefix);pm=StageMethod(prefix,'prefix')
    def train(model,method):
        opt=make_optimizer(method)
        for step in (1,2):
            opt.zero_grad(set_to_none=True);loss=model(ids,labels=ids).loss;assert torch.isfinite(loss)
            loss.backward();method.post_backward(1.,(step-1)/2)
            params=[p for p in model.parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step();method.after_update(step,2)
        opt.zero_grad(set_to_none=True)
    pm.begin_stage(1);train(prefix,pm)
    snapshot=cpu_state(prefix);shared_digest=digest(prefix.state_dict().items())
    rng=dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=[])
    folder=REPORT/'private_cpu_gate';folder.mkdir(exist_ok=False)
    path=folder/'shared_S1.pt';torch.save(dict(state=snapshot,method_stage=1,rng=rng,rng_digest=rng_digest(rng)),path)
    evidence={}
    for arm in ('Fixed','Refresh'):
        model=copy.deepcopy(parent);object.__setattr__(model,'core',model);method=StageMethod(model,arm)
        package=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(package['state'],strict=True);method.stage=package['method_stage'];assert method.stage==1
        assert digest(model.state_dict().items())==shared_digest
        random.setstate(package['rng']['python']);np.random.set_state(package['rng']['numpy']);torch.set_rng_state(package['rng']['cpu'])
        assert rng_digest(dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=[]))==package['rng_digest']
        records=[];method.write_boundary=lambda stage,row:records.append(row)
        before_ids=[id(p) for layer in method.layers for p in (layer.U_low,layer.S_low,layer.V_low)]
        method.begin_stage(2)
        assert len(records)==1 and records[0]['prior_method_stage']==1 and records[0]['actual_refactor_calls']==(14 if arm=='Refresh' else 0)
        after_ids=[id(p) for layer in method.layers for p in (layer.U_low,layer.S_low,layer.V_low)]
        assert (before_ids==after_ids)==(arm=='Fixed')
        assert all(p.requires_grad for layer in method.layers for p in (layer.U_low,layer.S_low,layer.V_low))
        before=digest(model.state_dict().items());train(model,method);assert before!=digest(model.state_dict().items())
        model.requires_grad_(False);model.eval();out=model(ids).logits.detach();final=folder/(arm+'.pt')
        torch.save(dict(state=cpu_state(model),layout=layout(model)),final)
        reconstructed=copy.deepcopy(parent);object.__setattr__(reconstructed,'core',reconstructed);re_method=StageMethod(reconstructed,arm)
        loaded=torch.load(final,map_location='cpu',weights_only=False);reconstructed.load_state_dict(loaded['state'],strict=True)
        reconstructed.requires_grad_(False);reconstructed.eval()
        assert layout(reconstructed)==loaded['layout']
        torch.testing.assert_close(out,reconstructed(ids).logits,rtol=0,atol=0)
        evidence[arm]=dict(shared_start_exact=True,RNG_restored=True,restored_method_stage=1,
            actual_S2_refactors=records[0]['actual_refactor_calls'],low_factors_still_train=True,new_stage_updates=2,
            actual_parameters_changed=True,strict_full_reload=True)
    assert digest(parent.state_dict().items())==original and not torch.cuda.is_initialized()
    write(REPORT/'CPU_GATE.json',dict(passed=True,seconds=time.monotonic()-tick,tests=evidence,shared_prefix_updates=2,
        CUDA_initialized=False,real_QA=0,tiny_FP32_not_real_FP16_or_quality=True))
    m,s,fit,dev,plan=data()
    paths=[HERE/n for n in SOURCES]+[REPORT/n for n in ('PROTOCOL.json','NECESSITY_REVIEW.json','SOURCE_REVIEW.json','CPU_GATE.json')]
    paths+=list(PORT.glob('*.py'))+[PORT/'execution_policy.json']
    paths+=[ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json',ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json']
    paths+=[ROOT/t['files']['train']['path'] for t in m['tasks']]
    paths+=[ROOT/n for n in ('code/OSFT/source/src/mini_trainer/osft_utils.py','code/OSFT/files_sha256.json','code/OSFT/source_manifest.json')]
    paths+=[p for p in BASE.iterdir() if p.is_file() and p.name!='model.safetensors' and p.stat().st_size<32*1024**2]
    runtime={}
    for module in ('transformers.models.qwen2.modeling_qwen2','transformers.models.llava_onevision.modeling_llava_onevision',
        'transformers.models.llava_onevision.processing_llava_onevision','transformers.models.llava_onevision.image_processing_llava_onevision','transformers.models.siglip.modeling_siglip'):
        path=Path(importlib.import_module(module).__file__).resolve();runtime[str(path)]=sha(path)
    files={str(p.relative_to(ROOT)):sha(p) for p in paths};files.update(runtime)
    write(REPORT/'READY.json',dict(files=files,training_plan=plan,training_ast_sha256=training_prefix()[1],model_seconds=BUDGETS,
        uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),interpreter=str(Path(sys.executable).resolve()),
        actual_FIT=2099,actual_updates=153,new_DEV_generations=480,TEST_QA=0,scientific_sources_frozen=True))


if __name__=='__main__':
    try:main()
    except Exception:
        write(REPORT/'FREEZE_FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True));raise
