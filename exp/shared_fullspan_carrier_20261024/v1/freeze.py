"""Separate tiny synthetic admission and Torch-free metadata sealing."""
import argparse,hashlib,importlib.metadata,importlib.util,io,json,os,sys,time
from pathlib import Path
from common import ROOT,HERE,PORT,REPORT,BASE,SOURCES,ORIGINAL_BASE_PROOF_SHA,data,loss_node,sha,write

def source_review():
    review=json.loads((REPORT/'SOURCE_REVIEW.json').read_bytes())
    assert review['decision']=='accept_fixed_carrier_sources_for_metadata_and_tiny_gate_only'
    assert review['protocol_sha256']==sha(REPORT/'PROTOCOL.md')
    assert review['necessity_review_sha256']==sha(REPORT/'NECESSITY_REVIEW.json')
    assert review['sources']=={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in SOURCES}
    return review

def gate():
    assert 'torch' not in sys.modules and not (REPORT/'CPU_GATE_STARTED.json').exists()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    source_review();tick=time.monotonic();write(REPORT/'CPU_GATE_STARTED.json',dict(pid=os.getpid(),synthetic_only=True))
    import torch
    from torch import nn
    from torch.nn import functional as F
    from core import Factor,PackedFactor,full_initialize,recover_group,pack_group
    assert not torch.cuda.is_initialized();torch.set_num_threads(2);torch.manual_seed(42)
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.gate_proj=nn.Linear(4,8,bias=False);self.up_proj=nn.Linear(4,8,bias=False);self.down_proj=nn.Linear(8,4,bias=False)
        def forward(self,x):return self.down_proj(F.silu(self.gate_proj(x))*self.up_proj(x))
    import copy
    mlps=nn.ModuleList([Tiny(),Tiny()]);original=copy.deepcopy(mlps)
    cached=[]
    for _ in range(2):
        sample=[]
        for m in original:
            x=torch.randn(1,6,4);sample.append((x,m(x).detach()))
        cached.append(sample)
    basis,factors=full_initialize(mlps,8)
    for i,m in enumerate(mlps):torch.testing.assert_close(m(cached[0][i][0]),cached[0][i][1],rtol=1e-4,atol=1e-5)
    parameters=[basis.weight,*[f.value for f in factors]]
    actual=sum(m(cached[0][i][0]).square().mean() for i,m in enumerate(mlps))
    direct=[]
    for i,m in enumerate(mlps):
        x=cached[0][i][0]
        g=F.linear(x,(basis.weight@m.gate_proj.value).T)
        u=F.linear(x,(basis.weight@m.up_proj.value).T)
        direct.append(F.linear(F.silu(g)*u,basis.weight@m.down_proj.value).square().mean())
    a=torch.autograd.grad(actual,parameters);b=torch.autograd.grad(sum(direct),parameters)
    for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-5)
    def audit(fs,opt,step):
        for f in fs:
            removed=~f.mask
            assert bool((f.value[removed]==0).all()) and bool((f.value.grad[removed]==0).all())
            for key in ('exp_avg','exp_avg_sq'):assert bool((opt.state[f.value][key][removed]==0).all())
            if step>=6:assert int(f.mask.sum())==f.mask.numel()//4
    events=[];recover_group(mlps,cached,torch.device('cpu'),events.append,epochs=4,update_interval=2,final_at=6,audit=audit)
    assert len(events)==8 and all(e['actual_optimizer_steps']==e['step'] for e in events)
    # Standard pickle cannot serialize local Tiny classes: gate only uses cloudpickle for this local synthetic fixture.
    import cloudpickle,pickle
    pack_group(mlps);x=torch.randn(1,6,4,dtype=torch.float16)
    with torch.no_grad():before=[m(x) for m in mlps]
    stream=io.BytesIO();torch.save(mlps,stream,pickle_module=cloudpickle);stream.seek(0)
    restored=torch.load(stream,pickle_module=pickle,weights_only=False)
    assert restored[0].gate_proj.basis is restored[1].down_proj.basis
    assert all(isinstance(getattr(m,n),PackedFactor) for m in restored for n in ('gate_proj','up_proj','down_proj'))
    assert not any(isinstance(m,Factor) for m in restored.modules())
    with torch.no_grad():
        for m,y in zip(restored,before):assert torch.equal(m(x),y)
    assert not torch.cuda.is_initialized()
    write(REPORT/'CPU_GATE.json',dict(passed=True,seconds=time.monotonic()-tick,synthetic_updates=8,
        shared_U_joint_gradient_reference=True,full_initialization_reference=True,mask_parameter_gradient_moments=True,
        actual_optimizer_witness=True,physical_values_bitmap_pickle_reload=True,shared_alias_reload=True,
        CPU_only_not_actual_GPU_proof=True,CUDA_initialized=False))

def freeze():
    assert 'torch' not in sys.modules and not (REPORT/'READY.json').exists()
    source_review();gate_record=json.loads((REPORT/'CPU_GATE.json').read_bytes());assert gate_record['passed']
    task,rows,plan=data()
    inherited_path=ROOT/'summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1/MODEL_READY.json'
    assert sha(inherited_path)==ORIGINAL_BASE_PROOF_SHA
    inherited=json.loads(inherited_path.read_bytes())
    dependencies={str((PORT/n).relative_to(ROOT)):sha(PORT/n) for n in ('runtime_utils.py','onevision_bridge.py','frozen_vision.py','retrieval.py','resource_monitor.py','methods.py','execution_policy.json')}
    assert dependencies[str((PORT/'execution_policy.json').relative_to(ROOT))]==inherited['dependencies'][str((PORT/'execution_policy.json').relative_to(ROOT))]
    original_screen=json.loads((ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/READY.json').read_bytes())
    assert sha(ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/READY.json')=='75b86e6272295b548c82cb73fd29dc60671e18b23314b625d46486232b0d1893'
    assert inherited['base_weights']==original_screen['base_files']
    environment={}
    for module,files in [('torch',['__init__.py']),('transformers',['modeling_utils.py','models/qwen2/modeling_qwen2.py','models/llava_onevision/modeling_llava_onevision.py','models/llava_onevision/processing_llava_onevision.py','models/llava_onevision/image_processing_llava_onevision.py','models/siglip/modeling_siglip.py'])]:
        spec=importlib.util.find_spec(module);folder=Path(spec.origin).parent
        for name in files:environment[str(folder/name)]=sha(folder/name)
    small={n:sha(ROOT/n) for n in inherited['base_small_files']}
    assert small==inherited['base_small_files']
    for n,h in inherited['base_weights'].items():assert (ROOT/n).is_file()
    evidence={str((REPORT/n).relative_to(ROOT)):sha(REPORT/n) for n in ('PROTOCOL.md','NECESSITY_REVIEW.json','SOURCE_REVIEW.json','CPU_GATE.json')}
    for p in (inherited_path,ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/READY.json',ROOT/plan['manifest_path'],ROOT/plan['S1_train_path'],ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'):
        evidence[str(p.relative_to(ROOT))]=sha(p)
    write(REPORT/'READY.json',dict(sources={str((HERE/n).relative_to(ROOT)):sha(HERE/n) for n in SOURCES},
        evidence=evidence,dependencies=dependencies,environment=environment,base_small_files=small,
        base_weights=inherited['base_weights'],inherited_weight_proof_sha256=sha(ROOT/'summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1/MODEL_READY.json'),
        full_loss_AST_sha256=loss_node()[1],data=plan,interpreter=str(Path(sys.executable).resolve()),
        versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','cloudpickle')},
        allowed_GPUs=[0,1,2,3],execution_host='OLD',execution_root=str(ROOT),budget_seconds=3600,execution_uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        metadata_only=True,no_automatic_continual_training=True))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('gate','freeze'));args=parser.parse_args()
    try:gate() if args.action=='gate' else freeze()
    except Exception:
        import traceback
        write(REPORT/('CPU_GATE_FAILURE.json' if args.action=='gate' else 'FREEZE_FAILURE.json'),dict(traceback=traceback.format_exc()));raise
