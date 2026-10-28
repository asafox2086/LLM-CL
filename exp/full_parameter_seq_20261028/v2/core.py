"""Unfrozen VLM sequential-FT core DRAFT; no executable experiment or admission.

Proposal: all unique VLM Parameters FP32, BF16 autocast, existing
HostAdamW (FP32 moments resident CPU, GPU Adam arithmetic), lr=1e-4, wd=0,
betas=(.9,.999), eps=1e-8; no scaler. At S19 end only, cast once to BF16,
then evaluate full final TEST and save that same graph. Earlier stage graphs
remain FP32 with BF16 AMP. No precision restoration or second model. No LoRA, frozen feature path, or cross-update visual cache.
Intended protocols: original nested pct01/pct10, both native backbones;
manifest/order/batch16/micro1/epoch1 and sparse-stage evaluator stay external.
This module deliberately does not create a queue, select GPUs, or run data.
"""
import ast
import hashlib
import os
from pathlib import Path
import sys

import torch

ROOT = Path(os.environ['LLMCL_ROOT']).resolve()
NATIVE = ROOT / 'exp/two_backbone_native_cl_20261017_v2'
PINS = {
 'runtime_utils.py':'4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240',
 'onevision_bridge.py':'f3b1dd38af3b58dbfa69073da698ea2bf39a3a11850fb1069a80e38d0e3669a9',
 'checkpoints.py':'06c53eab2c6810901f5206edbfb1df9defabdc747b1856ace0afa31e74b37a78',
 'host_adam.py':'5e733e6b87c12eb9331ff23e8919a142f1861a2beec82ec8b0361d878b9aec8c',
}
DTYPE = torch.bfloat16


def install_runtime():
    """Call only in a NEW isolated Python process; patches module globals only."""
    for name, expected in PINS.items():
        assert hashlib.sha256((NATIVE/name).read_bytes()).hexdigest() == expected
    assert 'runtime_utils' not in sys.modules and 'onevision_bridge' not in sys.modules
    sys.path.insert(0, str(NATIVE))
    import runtime_utils as rt
    import onevision_bridge as bridge
    import checkpoints
    rt.ROOT = ROOT
    tree = ast.parse((NATIVE/'runtime_utils.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'input_batch')
    # Preserve original image resize, prompt/retrieval/tokenizer/processor bytes.
    # Only Qwen's frozen prepare is removed; OneVision packing is replaced below.
    branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "'image_sizes' in batch")
    assert len(branch.orelse) == 3
    assert isinstance(branch.orelse[1], ast.ImportFrom) and branch.orelse[1].module == 'frozen_vision'
    assert ast.unparse(branch.orelse[2].value).startswith('prepare(')
    branch.orelse = [branch.orelse[0], ast.Assign(targets=[ast.Name(id='hit',ctx=ast.Store())],value=ast.Constant(False))]
    namespace = dict(rt.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                 str(Path(__file__).resolve())+':native-input-without-frozen-prepare','exec'),namespace)
    rt.input_batch = namespace['input_batch']
    # FP32 master parameters need explicit BF16 AMP inside native visual pack.
    tree_bridge = ast.parse((NATIVE/'onevision_bridge.py').read_text())
    cls = next(n for n in tree_bridge.body if isinstance(n,ast.ClassDef) and n.name=='PackedVision')
    forward = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    casts = [n for n in ast.walk(forward) if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.autocast']
    assert len(casts)==1
    next(k for k in casts[0].keywords if k.arg=='dtype').value = ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='bfloat16',ctx=ast.Load())
    ns = dict(bridge.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[forward],type_ignores=[])),str(__file__)+':trainable-packed-AMP','exec'),ns)
    bridge.PackedVision.forward = ns['forward']
    ev = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    count = 0
    for n in ast.walk(ev):
        if isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.autocast':
            kw = next(k for k in n.keywords if k.arg=='dtype')
            assert ast.unparse(kw.value)=='method.model.dtype'
            kw.value=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='bfloat16',ctx=ast.Load());count+=1
    assert count==1
    ns=dict(rt.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ev],type_ignores=[])),str(__file__)+':eval-BF16-AMP','exec'),ns)
    rt.evaluate=ns['evaluate']
    bridge.pack_native_batch = pack_native_trainable
    bridge.NATIVE_GATE_MODE = False
    return rt, bridge, checkpoints


def pack_native_trainable(batch, row):
    """Keep native size/token repair, retaining this QA's visual autograd graph."""
    import onevision_bridge as bridge
    model = bridge.ACTIVE
    assert isinstance(model, bridge.OneVisionBridge)
    assert not bridge.NATIVE_GATE_MODE
    batch = {k:v.to(model.device) for k,v in batch.items()}
    sizes = batch.pop('image_sizes')
    batch['image_grid_thw'] = sizes
    # Original PackedVision.forward supplies BF16 autocast and native feature
    # selection, projector, image_newline and pack_image_features unchanged.
    # This value is consumed once in answer_loss; never detached or cached.
    with torch.set_grad_enabled(model.training and torch.is_grad_enabled()):
        features = model.visual(batch['pixel_values'], grid_thw=sizes)
    assert features.ndim == 2 and features.shape[-1] == model.config.hidden_size
    if model.training and torch.is_grad_enabled():
        assert features.requires_grad
    batch['pixel_values'] = features
    ids = batch['input_ids']; token = model.config.image_token_id
    positions = (ids[0] == token).nonzero().flatten()
    assert len(positions) and int(positions[-1]-positions[0]+1) == len(positions)
    if len(positions) != len(features):
        begin,end = int(positions[0]), int(positions[-1])+1
        revised = torch.cat([ids[:,:begin],torch.full((1,len(features)),token,
                      device=ids.device,dtype=ids.dtype),ids[:,end:]],dim=1)
        assert torch.equal(ids[ids!=token],revised[revised!=token])
        batch['input_ids']=revised; batch['attention_mask']=torch.ones_like(revised)
    assert int((batch['input_ids']==token).sum()) == len(features)
    return batch, False


def unique_parameters(model):
    rows = list(model.parameters())
    assert len(rows) == len({id(p) for p in rows})
    return rows


class FullFTMethod:
    name = 'FullFT_Seq'
    original_revision = 'draft-full-VLM-FP32-HostAdam-BF16AMP-final-BF16'
    learning_rate = 1e-4

    def __init__(self, model, instructions):
        self.model = model
        self.instructions = list(instructions)
        self.stage = 0
        self.finalized = False
        assert all(p.is_cuda and p.dtype == torch.float32 for p in unique_parameters(model))
        for parameter in unique_parameters(model):
            parameter.requires_grad_(True)
        # The bridge registers core language, head, vision, projector and
        # image_newline aliases. No unregistered core Parameter can be omitted.
        core = getattr(model, 'core', model)
        assert {id(p) for p in core.parameters()} == {id(p) for p in model.parameters()}
        assert not any(type(m).__name__ in ('SeqLinear','AuthorSMoLinear') for m in model.modules())
        assert not any(k.startswith('_llmcl_visual_') for k in vars(model))
        self.initial_unique_parameter_count = sum(p.numel() for p in unique_parameters(model))

    def begin_stage(self, stage):
        assert not self.finalized
        self.stage = stage
        self.model.train()

    def set_instruction(self, stage):
        # Input prompt already uses the same manifest task instruction.
        assert 1 <= stage <= len(self.instructions)

    def end_stage(self, stage):
        assert self.stage == stage

    def training_loss(self, batch, row, processor):
        from runtime_utils import answer_loss
        return answer_loss(self.model, batch, row['answer'], processor)


def make_optimizer(method):
    parameters = unique_parameters(method.model)
    assert all(p.requires_grad and p.dtype == torch.float32 and p.is_cuda for p in parameters)
    from host_adam import HostAdamW
    return HostAdamW(parameters, lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0.)


def accumulate_update(method, optimizer, minibatch, task, processor, diagnostic=False):
    """Draft exact micro1 mean-CE + full-model clip1; scheduler stays external."""
    from runtime_utils import input_batch, answer_loss
    assert 0 < len(minibatch) <= 16
    model = method.model
    model.train(); optimizer.zero_grad(set_to_none=True)
    losses = []; answer_tokens = 0
    for row in minibatch:
        batch,_ = input_batch(row,task,processor,model.device)
        with torch.autocast('cuda',dtype=DTYPE):
            loss,n = answer_loss(model,batch,row['answer'],processor)
        assert torch.isfinite(loss)
        (loss/len(minibatch)).backward()
        losses.append(float(loss.detach())); answer_tokens += n
        del batch,loss
    parameters = unique_parameters(model)
    grad = torch.nn.utils.clip_grad_norm_(parameters,1.)
    assert torch.isfinite(grad)
    probes = {}
    if diagnostic:
        for name,p in model.named_parameters():
            if p.grad is None or not bool(torch.count_nonzero(p.grad)):
                continue
            group = ("projector" if ("projector" in name or "visual.merger" in name) else "newline" if "image_newline" in name else "vision" if name.startswith("visual.") else "head" if name.startswith("lm_head.") else "language")
            if group not in probes or p.numel()<probes[group][1].numel():
                probes[group]=(name,p)
        probes={g:(name,p,p.detach().clone()) for g,(name,p) in probes.items()}
    moment_h2d = sum(v.numel()*v.element_size() for p in parameters if p.grad is not None
                     for key,v in optimizer.state.get(p,{}).items() if key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'))
    # No GradScaler, skipped-step recovery, optimizer replacement or retries.
    optimizer.step()
    moment_dtypes = sorted({str(v.dtype) for state in optimizer.state.values()
                            for key,v in state.items() if key in ('exp_avg','exp_avg_sq')})
    assert moment_dtypes == ['torch.float32']
    assert all(not v.is_cuda for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v))
    finite_weights = finite_moments = None
    if diagnostic:
        finite_weights=all(bool(torch.isfinite(p).all()) for p in parameters)
        finite_moments=all(bool(torch.isfinite(v).all()) for st in optimizer.state.values() for key,v in st.items() if key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'))
        assert finite_weights and finite_moments
    module_probes={g:{'parameter':name,'nonzero_gradient':True,'parameter_changed':not torch.equal(p.detach(),before)} for g,(name,p,before) in probes.items()}
    return {'finite_gradient_norm':True,'finite_weights_diagnostic':finite_weights,'finite_moments_diagnostic':finite_moments,'module_probes':module_probes,'mean_loss':sum(losses)/len(losses),'answer_tokens':answer_tokens,
            'gradient_l2_preclip':float(grad),'optimizer_moment_dtypes':moment_dtypes,
            'all_unique_parameters':sum(p.numel() for p in parameters),
            'optimizer_CPU_state_bytes':sum(v.numel()*v.element_size() for st in optimizer.state.values() for v in st.values() if torch.is_tensor(v)),
            'moment_transfer_source_accounting_H2D_bytes':moment_h2d,
            'moment_transfer_source_accounting_D2H_bytes':sum(v.numel()*v.element_size() for p in parameters if p.grad is not None for k,v in optimizer.state[p].items() if k in ('exp_avg','exp_avg_sq','max_exp_avg_sq')),
            'transfer_note':'Source byte accounting; not a measured PCIe bandwidth or transfer duration.'}


def save_stage(method, processor, run, stage, config):
    """Original full-graph stage artifact, including processor/RNG/source config."""
    from checkpoints import save
    assert config['method'] == 'FullFT_Seq'
    expected = DTYPE if method.finalized else torch.float32
    assert config['training_parameter_dtype'] == 'torch.float32'
    assert config['final_deployment_dtype'] == 'torch.bfloat16'
    assert config['condition'] in ('pct01','pct10')
    assert all(p.dtype == expected for p in unique_parameters(method.model))
    assert method.finalized == (stage == 19)
    return save(method,processor,run,stage,config)


def finalize_for_actual_final_test(method):
    """Only after S19 optimizer deletion and before actual full final TEST."""
    assert method.stage == 19 and not method.finalized
    assert all(p.grad is None for p in unique_parameters(method.model))
    method.model.to(dtype=DTYPE)
    method.model.eval()
    method.finalized = True
    assert all(p.dtype == DTYPE for p in unique_parameters(method.model))
    return {'stage':19,'once_only':True,'deployment_dtype':'torch.bfloat16',
            'final_TEST_must_follow':True,'resume_training_forbidden':True}


UNVERIFIED = (
 'No model was instantiated and no input, checkpoint or optimizer state was executed.',
 'Need real image QA: vision tower/projector/language nonzero gradients AND actual updates; include image_newline/head in inventory.',
 'Need BF16 hardware/support, real peak for both backbones, finite updates, state dtype receipts and full checkpoint strict reload.',
 'Need real HostAdam CPU moment/transfer and activation offload cost plus FP32 parameter/gradient peak; no claim 24GiB fits.',
 'Draft runner exists; protocol/source freezing and actual pct01/pct10 two-backbone preflights/reload remain pending.',
 'Need actual final BF16 full TEST before assigning quality or deployment bytes/GPU; intermediate FP32 mothergraph/BF16AMP scores cannot be relabeled pure-BF16.',
)


def batch_digest(batch):
    """Exact prepared input tensor bytes; CPU serialization, no model CPU math."""
    h=hashlib.sha256()
    for name,value in sorted(batch.items()):
        assert torch.is_tensor(value)
        v=value.detach().cpu().contiguous()
        h.update(str((name,tuple(v.shape),str(v.dtype))).encode())
        raw=v.reshape(-1).view(torch.uint8).numpy().reshape(-1)
        for start in range(0,raw.size,8*1024**2):h.update(raw[start:start+8*1024**2].tobytes())
    return h.hexdigest()


def full_parameter_inventory(method):
    """Check the reloaded graph, without resetting any mask or constructing FT."""
    import json
    model=method.model;parameters=unique_parameters(model)
    assert not method.finalized
    assert all(p.is_cuda and p.dtype==torch.float32 and p.requires_grad for p in parameters)
    original=getattr(model,'core',model)
    assert {id(p) for p in original.parameters()}=={id(p) for p in parameters}
    assert not any(type(m).__name__ in ('SeqLinear','AuthorSMoLinear') for m in model.modules())
    assert not any('lora' in name.lower() for name,_ in model.named_parameters())
    assert not any(k.startswith('_llmcl_visual_') for k in vars(model))
    count=sum(p.numel() for p in parameters)
    assert count==method.initial_unique_parameter_count
    layout=[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'numel':p.numel(),
             'requires_grad':p.requires_grad} for n,p in model.named_parameters()]
    return {'unique_parameters':count,'registered_parameter_bytes':sum(p.numel()*p.element_size() for p in parameters),
            'parameter_layout_sha256':hashlib.sha256(json.dumps(layout,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'all_original_parameters_covered':True,'no_LoRA':True,'no_frozen_visual_cache':True}


def make_reload_reference(method, processor, row, task):
    """One last ALREADY-USED TRAIN QA, eval mode/BF16AMP, not TEST or F1."""
    import json
    from runtime_utils import input_batch,answer_loss
    from training_validity import parameter_digest
    method.model.eval();method.set_instruction(task['task'])
    assert row.get('images'), 'Reload visual-gradient probe needs the last used TRAIN image QA'
    with torch.no_grad(),torch.autocast('cuda',dtype=DTYPE):
        batch,details=input_batch(row,task,processor,method.model.device)
        loss,n=answer_loss(method.model,batch,row['answer'],processor)
    assert torch.isfinite(loss)
    stat=Path('/proc/self/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
    return {'schema':'fullft-checkpoint-probe-v1','preflight_only':True,'split':'train',
            'row':row,'task':task,'loss':float(loss),'answer_tokens':n,
            'input_sha256':batch_digest(batch),'input_details':details,
            'parameter_sha256':parameter_digest(unique_parameters(method.model)),
            'parameter_inventory':full_parameter_inventory(method),
            'producer_identity':{'pid':os.getpid(),'startticks':int(fields[19]),'uid':os.getuid()},
            'AMP_dtype':'torch.bfloat16','parameter_dtype':'torch.float32',
            'row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()}
