"""Count model coefficients for the full-data baseline table without training or inference."""
import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, LlavaOnevisionForConditionalGeneration, Qwen2VLForConditionalGeneration

from full_only_20261118 import MANIFEST_SHA, NATIVE_FILES, read, sha


NAMES = {"onevision": "LLaVA-OneVision-Qwen2-0.5B-OV", "qwen": "Qwen2-VL-2B-Instruct"}


def base_shapes(folder):
    shapes, hashes = {}, {}
    for shard in sorted(folder.glob("*.safetensors")):
        with shard.open("rb") as stream:
            header = stream.read(struct.unpack("<Q", stream.read(8))[0])
        hashes[shard.name] = hashlib.sha256(header).hexdigest()
        for name, metadata in json.loads(header).items():
            if name != "__metadata__":
                assert name not in shapes
                shapes[name] = tuple(metadata["shape"])
    assert shapes
    return shapes, hashes


def create_model(config, backbone, shapes):
    from onevision_bridge import OneVisionBridge
    cls = LlavaOnevisionForConditionalGeneration if backbone == "onevision" else Qwen2VLForConditionalGeneration
    core = cls(config)
    core.tie_weights()
    weights = dict(core.named_parameters())
    assert {k: tuple(p.shape) for k,p in weights.items()} == shapes
    model = OneVisionBridge(core) if backbone == "onevision" else core
    assert {id(p) for p in weights.values()} == {id(p) for p in model.parameters()}
    return model


def osft_count(linears):
    from osft_adapter import author_functions
    kernels, _, source_hash = author_functions()
    fixed = trainable = replaced = 0
    for layer in linears:
        out_size, in_size = layer.weight.shape
        rank = min(out_size, in_size)
        protected = int(.75 * rank)
        factors = kernels["create_svd_dict"](layer.weight, protected, decompose_existing=False)
        assert factors["rank_high"] == protected
        for part in ("U", "S", "V"):
            assert not factors[part+"_high"].requires_grad and factors[part+"_low"].requires_grad
            fixed += factors[part+"_high"].numel()
            trainable += factors[part+"_low"].numel()
        replaced += layer.weight.numel()
    return dict(added=fixed+trainable-replaced, trainable=trainable, fixed_factors=fixed,
                replaced=replaced, author_source=source_hash)


def sparta_count(model, linears, ntasks):
    from sparta_adapter import Router, author_components
    kernels, _, hashes = author_components()
    router = Router(model.config.hidden_size, kernels, ntasks)
    router_count = sum(p.numel() for p in router.parameters())
    assert router_count == 6 * 8 * ntasks * model.config.hidden_size
    adapters = 0
    for base in linears:
        adapter = kernels["DATAInjectedLinear"](base.in_features, base.out_features,
                                               base.bias is not None, r=2, r2=8)
        # The training port reuses the original frozen weight and bias for linear_data.
        adapters += sum(p.numel() for name,p in adapter.named_parameters() if not name.startswith("linear_data."))
    total = adapters + math.ceil(len(model.model.layers)/4) * router_count
    return dict(added=total, trainable=total, dual_rank_coefficients=adapters, author_sources=hashes)


def audit(root, backbone, variant):
    exp = root/"exp"
    design = read(root/"summary/full_baselines_20261118/DESIGN.json")
    kind = "native" if variant == "main" else "forever"
    port = exp/("full_baselines_20261118_" + kind)
    binding = design["ports"][kind]
    assert all(sha(port/name) == digest for name,digest in binding.items())
    assert sha(root/design["manifest"]) == design["manifest_sha256"] == MANIFEST_SHA
    if kind == "native":
        for name in ("methods.py","clmoe_adapter.py","pass_adapter.py","onevision_bridge.py",
                     "osft_adapter.py","sparta_adapter.py","parsp_adapter.py"):
            assert sha(port/name) == sha(exp/"full_only_20261118"/name) == NATIVE_FILES[name]
    sys.path.insert(0,str(port))
    from methods import make_method
    folder = root/"model"/NAMES[backbone]
    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    shapes, headers = base_shapes(folder)
    base_count = sum(math.prod(shape) for shape in shapes.values())
    instructions = [t["instruction"] for t in read(root/design["manifest"])["tasks"]]
    assert len(instructions) == 19
    rows = {}
    methods = ["SeqFT_LoRA","CL_MoE","SMoLoRA","PASs_MoE","OSFT","SpaRTA","PaRSP"]
    if variant == "forever":
        methods = ["FOREVER"]
    if variant == "main" and "fullft" in design["ports"]:
        ft = exp/"full_baselines_20261118_fullft"/"core.py"
        source = ft.read_text()
        assert sha(ft) == design["ports"]["fullft"]["core.py"]
        assert "for parameter in unique_parameters(model):\n            parameter.requires_grad_(True)" in source
        assert "assert all(p.requires_grad and p.dtype == torch.float32" in source
        rows["FullFT"] = dict(added=0, trainable=base_count, core_sha256=sha(ft))
    with torch.device("meta"):
        for name in methods:
            model = create_model(config, backbone, shapes)
            linears = [mod for k,mod in model.named_modules() if k.startswith("model.layers.")
                       and isinstance(mod,torch.nn.Linear)]
            assert len(linears) == 7 * len(model.model.layers)
            if name == "OSFT":
                rows[name] = osft_count(linears)
            elif name == "SpaRTA":
                rows[name] = sparta_count(model, linears, len(instructions))
            else:
                method = make_method(model, "SeqLoRA" if name=="SeqFT_LoRA" else name,
                                     instructions, [[0.] * 384] * len(instructions))
                if name in ("SMoLoRA","SeqFT_LoRA","PASs_MoE"):
                    method.begin_stage(19)
                added = sum(p.numel() for p in model.parameters()) - base_count
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                # PaRSP chooses which base layers to update using actual stage TRAIN examples.
                rows[name] = dict(added=added, trainable=None if name=="PaRSP" else trainable)
    return dict(backbone=backbone, base=base_count, safetensor_headers=headers,
                config_sha256=sha(folder/"config.json"), port_binding=binding, manifest=MANIFEST_SHA,
                source_sha256=sha(Path(__file__)), counts=rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("backbone", choices=NAMES)
    cli.add_argument("variant", choices=("main","forever"))
    args = cli.parse_args()
    print(json.dumps(audit(args.root,args.backbone,args.variant)))
