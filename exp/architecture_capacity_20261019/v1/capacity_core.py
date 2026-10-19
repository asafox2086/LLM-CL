"""Capacity controls over the frozen CL-MoE / SMoLoRA port, not a new method.

Authors split TOTAL rank between experts. Keep that distinction explicit.
Only independent trial processes import these generated classes; active
baseline source files/modules are never modified.
"""
import ast
import copy
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / 'exp/two_backbone_native_cl_20261017_v2'
sys.path.insert(0, str(PORT))

CASES = {
    'CL_original': dict(family='CL_MoE', experts=8, total_rank=64),
    'CL_shared_equal_rank': dict(family='CL_MoE', experts=1, total_rank=64),
    'CL_shared_small_rank': dict(family='CL_MoE', experts=1, total_rank=8),
    'SMo_original': dict(family='SMoLoRA', experts=8, total_rank=128),
    'SMo_two_equal_rank': dict(family='SMoLoRA', experts=2, total_rank=128),
    'SMo_two_small_rank': dict(family='SMoLoRA', experts=2, total_rank=16),
}


def source(name):
    p = PORT / name
    return p.read_text(), hashlib.sha256(p.read_bytes()).hexdigest()


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def compile_class(text, name, namespace):
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<capacity-control:' + name + '>', 'exec'), namespace)
    return namespace[name]


def make_method(model, case, instructions, embedding=None):
    import methods as original
    spec = CASES[case]
    e, r = spec['experts'], spec['total_rank']
    assert r % e == 0
    if spec['family'] == 'CL_MoE':
        import clmoe_adapter as authored
        text, _ = source('clmoe_adapter.py')
        modified = replace_once(text, "(base.in_features,64,8)", "(base.in_features,CAP_R,CAP_E)")
        modified = replace_once(modified, "(64,base.out_features,8)", "(CAP_R,base.out_features,CAP_E)")
        modified = replace_once(modified, "nn.Linear(base.in_features,8,bias=False)", "nn.Linear(base.in_features,CAP_E,bias=False)")
        modified = replace_once(modified, "torch.zeros(8,dtype=torch.float64)", "torch.zeros(CAP_E,dtype=torch.float64)")
        modified = replace_once(modified, "range(8)", "range(CAP_E)")
        modified = replace_once(modified, "weights.topk(2)", "weights.topk(min(2,CAP_E))")
        ns = dict(vars(authored), CAP_E=e, CAP_R=r)
        compile_class(modified, 'CLMoELinear', ns)
        cls = compile_class(modified, 'CLMoEMethod', ns)
        method = cls(model, spec['family'], instructions, embedding)
    else:
        text, _ = source('methods.py')
        modified = replace_once(text, "self.expert_num=8", "self.expert_num=CAP_E")
        modified = replace_once(modified, "self.r={'default':128}", "self.r={'default':CAP_R}")
        modified = replace_once(modified, "(base.in_features,128,8)", "(base.in_features,CAP_R,CAP_E)")
        modified = replace_once(modified, "(128,base.out_features,8)", "(CAP_R,base.out_features,CAP_E)")
        modified = replace_once(modified, "nn.Linear(base.in_features,4,bias=False)", "nn.Linear(base.in_features,CAP_E//2,bias=False)")
        modified = replace_once(modified, "nn.Linear(384,4,bias=False)", "nn.Linear(384,CAP_E//2,bias=False)")
        ns = dict(vars(original), CAP_E=e, CAP_R=r)
        compile_class(modified, 'AuthorSMoLinear', ns)
        cls = compile_class(modified, 'Method', ns)
        method = cls(model, spec['family'], instructions, embedding)
    method.capacity_spec = dict(spec, per_expert_rank=r//e,
        scale=2.0, positions=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        is_authored_hyperparameter_control=True, not_new_algorithm=True)
    return method


def parameter_inventory(model):
    """Unique actual parameter objects, not double-counted author aliases."""
    groups = {}
    for name, p in model.named_parameters():
        if '.loraA.' in name or '.loraB.' in name:
            key = 'expert_A_B'
        elif any(s in name for s in ['.router.', 'lora_vu_gate.', 'lora_ins_gate.']):
            key = 'routers'
        elif 'lora_fc_' in name:
            key = 'branch_fusion'
        else:
            key = 'base'
        value = groups.setdefault(key, dict(parameters=0, bytes=0, trainable=0))
        value['parameters'] += p.numel()
        value['bytes'] += p.numel() * p.element_size()
        value['trainable'] += p.numel() * int(p.requires_grad)
    return groups


def cpu_gate():
    """Real tiny HF graph / original-vs-control equivalence, not QA evidence."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    import methods as original
    import clmoe_adapter
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = Qwen2Config(vocab_size=127, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.)
    base = Qwen2ForCausalLM(config).eval()
    instructions = ['synthetic instruction ' + str(i) for i in range(19)]
    embedding = torch.randn(19, 384).tolist()
    records = {}
    for family, case in [('CL_MoE','CL_original'), ('SMoLoRA','SMo_original')]:
        a, b = copy.deepcopy(base), copy.deepcopy(base)
        torch.manual_seed(137)
        ref = (clmoe_adapter.CLMoEMethod(a, family, instructions) if family=='CL_MoE'
               else original.Method(a, family, instructions, embedding))
        torch.manual_seed(137)
        new = make_method(b, case, instructions, embedding)
        assert a.state_dict().keys() == b.state_dict().keys()
        assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
        # Nonzero B makes the control exercise routing/expert arithmetic too.
        with torch.no_grad():
            pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
            for name in pa:
                if '.loraB.' in name:
                    pa[name].normal_(0,.02);pb[name].copy_(pa[name])
        ids = torch.tensor([[3,9,5,17,8],[4,7,12,13,11]])
        a.eval();b.eval()
        x, y = a(ids).logits, b(ids).logits
        assert torch.equal(x,y)
        x.float().square().mean().backward();y.float().square().mean().backward()
        for name, p in a.named_parameters():
            q = dict(b.named_parameters())[name]
            assert (p.grad is None)==(q.grad is None)
            if p.grad is not None:assert torch.equal(p.grad,q.grad) and torch.isfinite(q.grad).all()
        records[case] = dict(original_state_exact=True, nonzero_expert_forward_exact=True,
                             all_available_raw_gradients_exact=True)
    inventories = {}
    for case in CASES:
        model = copy.deepcopy(base)
        torch.manual_seed(137)
        method = make_method(model, case, instructions, embedding)
        count_before = sum(p.numel() for p in model.parameters())
        method.begin_stage(1)
        inv = parameter_inventory(model)
        model.eval()
        out = model(torch.tensor([[2,3,7,8]])).logits
        assert torch.isfinite(out).all()
        out.square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
        method.begin_stage(19)
        assert sum(p.numel() for p in model.parameters())==count_before
        inventories[case] = dict(groups=inv, capacity=method.capacity_spec,
            no_parameter_growth_on_begin_stage=True, finite_tiny_HF_forward_backward=True)
    assert inventories['CL_original']['groups']['expert_A_B']['parameters']==inventories['CL_shared_equal_rank']['groups']['expert_A_B']['parameters']
    assert inventories['SMo_original']['groups']['expert_A_B']['parameters']==inventories['SMo_two_equal_rank']['groups']['expert_A_B']['parameters']
    assert not torch.cuda.is_initialized()
    return dict(original_equivalence=records, six_capacity_controls=inventories,
                CPU_only=True, TEST_QA=0, no_full_model_quality_or_memory_claim=True)
