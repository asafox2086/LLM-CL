"""The one tiny implementation gate; no original QA, pretrained model or CUDA."""
import copy
import io
import json
import pickle
import tempfile
from pathlib import Path


def run():
    import torch
    import cloudpickle
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from common import ASSETS, PORT, all_weight_layout, mask_digest, write
    from core import Packed2of4Linear, finalize_ffn_values
    from pruning import load_pruner, prune_fixed
    from runtime import SparseMethod, add_common_qv, measured_optimizer
    import sys
    sys.path.insert(0, str(PORT))
    from host_adam import HostAdamW
    from methods import SeqLinear
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True); torch.manual_seed(42)
    receipt = json.loads((ASSETS / 'SOURCE_RECEIPT.json').read_text())
    digest = receipt['files']['sparsegpt.py']['sha256']
    native = load_pruner(ASSETS / 'sparsegpt.py', digest, observe_mask=False, cpu_admission=True)
    observed = load_pruner(ASSETS / 'sparsegpt.py', digest, observe_mask=True, cpu_admission=True)
    where = {}
    def equal(left, right, label):
        if not torch.equal(left, right):
            finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            difference = float((left.float() - right.float()).abs().max())
            raise AssertionError(json.dumps(dict(where=where, label=label, finite=finite,
                max_abs_difference=difference, left_shape=list(left.shape), right_shape=list(right.shape))))
    cfg = Qwen2Config(vocab_size=31, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    base = Qwen2ForCausalLM(cfg).eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    dense = copy.deepcopy(base); packed = copy.deepcopy(base)
    torch.manual_seed(20261024)
    inputs = [torch.randint(0, 31, (1, 6)) for _ in range(16)]
    keep_masks = {}
    for block in range(2):
        where.update(block=block)
        targets = [f'model.layers.{block}.mlp.{name}' for name in ('gate_proj', 'up_proj', 'down_proj')]
        for model, klass in ((dense, native), (packed, observed)):
            solvers = {name: klass(model.get_submodule(name)) for name in targets}
            handles = []
            for name in targets:
                def capture(module, args, output, name=name):
                    solvers[name].add_batch(args[0].detach(), output.detach())
                handles.append(model.get_submodule(name).register_forward_hook(capture))
            with torch.no_grad():
                for ids in inputs:
                    model(ids)
            for handle in handles:
                handle.remove()
            for name in targets:
                solver = solvers[name]
                if klass is native:
                    solver.fasterprune(.5, prunen=2, prunem=4, blocksize=128, percdamp=.01)
                else:
                    keep = prune_fixed(solver)
                    keep_masks[name] = keep.clone()
                    old = model.get_submodule(name)
                    layer = Packed2of4Linear(old, keep)
                    equal(layer.materialize(), old.weight, 'physical_pack')
                    parent, child = name.rsplit('.', 1)
                    setattr(model.get_submodule(parent), child, layer)
            solvers.clear()
        with torch.no_grad():
            equal(dense(inputs[0]).logits, packed(inputs[0]).logits, 'observer_and_pack_complete_decoder')
    # A selected support may contain numerical zeros. Never infer it by nonzeros.
    zero = torch.nn.Linear(8, 4, bias=False)
    zero.weight.data.zero_(); support = torch.zeros_like(zero.weight, dtype=torch.bool)
    support.reshape(4, 2, 4)[..., 1:3] = True
    zero_packed = Packed2of4Linear(zero, support)
    equal(zero_packed.materialize(), zero.weight, 'zero_slot_support')
    assert bool((zero_packed.pattern == 3).all())
    add_common_qv(dense, expected_layers=2); add_common_qv(packed, expected_layers=2)
    dense.eval(); packed.eval()
    with torch.no_grad():
        equal(dense(inputs[0]).logits, packed(inputs[0]).logits, 'common_qv_initial_function')
    results = {}
    target = torch.tensor([5, 6])
    for case in ('Frozen', 'Plastic'):
        model = copy.deepcopy(packed); reference = copy.deepcopy(dense)
        method = SparseMethod(model, case, expected_layers=2)
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        ref_qv = [p for module in reference.modules() if isinstance(module, SeqLinear)
                  for p in (module.A.weight, module.B.weight)]
        for parameter in ref_qv:
            parameter.requires_grad_(True)
        ref_ffn = [reference.get_submodule(name).weight for name in keep_masks] if case == 'Plastic' else []
        for parameter in ref_ffn:
            parameter.requires_grad_(True)
        before_mask = mask_digest(model); witnesses = []
        for stage in (1, 2):
            where.update(case=case, stage=stage)
            method.begin_stage(stage)
            with tempfile.TemporaryDirectory() as folder:
                def commit(value):
                    path = Path(folder) / 'optimizer' / f'{len(witnesses):02d}.json'
                    write(path, value)
                    assert json.loads(path.read_text()) == value
                    witnesses.append(value)
                optimizer = measured_optimizer(method, commit)
                groups = [dict(params=ref_qv, lr=1e-4)]
                if ref_ffn:
                    groups.append(dict(params=ref_ffn, lr=1e-5))
                ref_optimizer = HostAdamW(groups, betas=(.9, .999), eps=1e-8, weight_decay=0.)
                for step in (1, 2):
                    where['step'] = step
                    model.train(); reference.train()
                    optimizer.zero_grad(set_to_none=True); ref_optimizer.zero_grad(set_to_none=True)
                    state = torch.get_rng_state().clone()
                    loss = torch.nn.functional.cross_entropy(model(inputs[step]).logits[0, -2:].float(), target) * .001
                    loss.backward(); packed_rng = torch.get_rng_state().clone()
                    torch.set_rng_state(state)
                    ref_loss = torch.nn.functional.cross_entropy(reference(inputs[step]).logits[0, -2:].float(), target) * .001
                    ref_loss.backward()
                    equal(loss, ref_loss, 'loss'); equal(packed_rng, torch.get_rng_state(), 'RNG')
                    for name, keep in keep_masks.items():
                        if case == 'Plastic':
                            layer = model.get_submodule(name); ref = reference.get_submodule(name)
                            ref.weight.grad.masked_fill_(~keep, 0)
                            equal(layer.values.grad, ref.weight.grad[keep].reshape_as(layer.values), name + ':gradient')
                    for name, layer in model.named_modules():
                        if isinstance(layer, SeqLinear):
                            other = reference.get_submodule(name)
                            equal(layer.A.weight.grad, other.A.weight.grad, name + ':Agradient')
                            equal(layer.B.weight.grad, other.B.weight.grad, name + ':Bgradient')
                    if step == 1:
                        path = Path(folder) / 'initially_missing' / 'gradients' / 'first.pt'
                        path.parent.mkdir(parents=True)
                        gradients = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
                        torch.save(gradients, path); restored = torch.load(path, weights_only=True)
                        assert restored.keys() == gradients.keys()
                        for name in gradients:
                            equal(restored[name], gradients[name], name + ':gradient_reload')
                    params = [p for p in model.parameters() if p.requires_grad]
                    ref_params = [p for p in reference.parameters() if p.requires_grad]
                    # Small synthetic loss keeps this exact arithmetic admission
                    # below clipping; it is not the actual training recipe.
                    assert float(torch.nn.utils.clip_grad_norm_(params, 1.)) < 1
                    assert float(torch.nn.utils.clip_grad_norm_(ref_params, 1.)) < 1
                    optimizer.step(); ref_optimizer.step()
                    for name, keep in keep_masks.items():
                        layer = model.get_submodule(name); ref = reference.get_submodule(name)
                        equal(layer.materialize(), ref.weight, name + ':updated_weight')
                    for name, layer in model.named_modules():
                        if isinstance(layer, SeqLinear):
                            other = reference.get_submodule(name)
                            equal(layer.A.weight, other.A.weight, name + ':Aupdated')
                            equal(layer.B.weight, other.B.weight, name + ':Bupdated')
                    assert mask_digest(model) == before_mask
            del optimizer, ref_optimizer
        model.eval()
        # Separate final-storage fixture: retain FP32 q/v, cast the base and the
        # prescribed packed slots to FP16. It is not a GPU FP16 math admission.
        qv_ids = {id(p) for module in model.modules() if isinstance(module, SeqLinear)
                  for p in (module.A.weight, module.B.weight)}
        value_ids = {id(module.values) for module in model.modules() if isinstance(module, Packed2of4Linear)}
        for parameter in model.parameters():
            parameter.requires_grad_(False); parameter.grad = None
            if id(parameter) not in qv_ids | value_ids:
                parameter.data = parameter.data.to(torch.float16)
        for module in model.modules():
            if isinstance(module, Packed2of4Linear):
                module.output_dtype = torch.float16
        finalize_ffn_values(model, expected_layers=2)
        with torch.no_grad():
            expected = model(inputs[0]).logits
        stream = io.BytesIO(); torch.save(method, stream, pickle_module=cloudpickle); stream.seek(0)
        recovered = torch.load(stream, pickle_module=pickle, weights_only=False)
        with torch.no_grad():
            equal(recovered.model(inputs[0]).logits, expected, 'whole_graph_reload')
        assert all_weight_layout(recovered.model) == all_weight_layout(model)
        assert not any(name.endswith(tuple('.mlp.' + t + '.weight' for t in ('gate_proj', 'up_proj', 'down_proj')))
                       for name in recovered.model.state_dict())
        assert len(witnesses) == 4 and all(w['parameter_IDs_match'] and w['all_moments_CPU'] for w in witnesses)
        results[case] = dict(own_updates=4, independent_dense_reference_updates=4,
            training_coordinates=witnesses[0]['coordinates'],
            optimizer_witnesses=4, complete_decoder_and_graph_reload_exact=True)
    assert not torch.cuda.is_initialized()
    return dict(passed=True, tiny_HF_CPU_FP32_only=True, cases=results,
        original_versus_observed_SparseGPT_and_packing_exact=True, explicit_zero_support_preserved=True,
        gradients_weights_RNG_equal_to_independent_dense_masked_reference=True,
        persistent_optimizer_and_gradient_roundtrip=True, CUDA_initialized=False,
        original_QA=0, pretrained_models=0, actual_VLM_quality_or_GPU_evidence=False)
