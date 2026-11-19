"""Count the uncompressed adapter tensors in the image-disjoint training graphs."""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter

import torch
from transformers import AutoConfig

from full_baseline_parameter_counts_20261119 import NAMES, base_shapes, create_model
from full_only_20261118 import read, sha
from image_disjoint_shared_20261119 import MANIFEST_SHA


SCOPES={"shared":"image_disjoint_shared_20261119","eight":"image_disjoint_experts_20261119",
        "one":"image_disjoint_rank64_20261119"}


def audit(root, arm):
    scope=SCOPES[arm]
    d=read(root/"summary"/scope/"DESIGN.json")
    assert sha(root/d["manifest"])==d["manifest_sha256"]==MANIFEST_SHA
    port=root/"exp"/scope
    sys.path.insert(0,str(port))
    from methods import make_method
    backbone=d["backbone"]
    folder=root/"model"/NAMES[backbone]
    cfg=AutoConfig.from_pretrained(folder,local_files_only=True)
    shapes,headers=base_shapes(folder)
    tasks=[t["instruction"] for t in read(root/d["manifest"])["tasks"]]
    assert len(tasks)==19
    with torch.device("meta"):
        model=create_model(cfg,backbone,shapes)
        before={id(p) for p in model.parameters()}
        linears={n:(p.in_features,p.out_features) for n,p in model.named_modules()
                 if n.startswith("model.layers.") and isinstance(p,torch.nn.Linear)}
        method=make_method(model,"SeqLoRA" if arm=="shared" else "CL_MoE",tasks,[[0.]*384]*19)
        if arm=="shared":
            method.begin_stage(19)
        names={n:p for n,p in model.named_parameters() if id(p) not in before}
        assert set(map(id,names.values()))=={id(p) for p in model.parameters() if p.requires_grad}
        weights=Counter()
        projected={}
        families={}
        for n,p in names.items():
            assert p.dtype==torch.float32
            component=next(k for k in (".A.",".B.",".router.") if k in n)
            proj=n.split(component,1)[0]
            assert proj in linears
            group="router" if component==".router." else "factors"
            weights[group]+=p.numel()*p.element_size()
            projected[proj]=linears[proj]
            families.setdefault(proj.rsplit(".",1)[1],Counter())[group]+=p.numel()
        per_rank=sum(a+b for a,b in projected.values())
        total_rank=8 if arm=="shared" else 64
        assert weights["factors"]==4*total_rank*per_rank
        gate_outputs={"shared":0,"one":1,"eight":8}[arm]
        assert weights["router"]==gate_outputs*4*sum(a for a,_ in projected.values())
    return dict(backbone=backbone,arm=arm,source=sha(Path(__file__)),design=sha(root/"summary"/scope/"DESIGN.json"),
                manifest=MANIFEST_SHA,headers=headers,
                method_sources={n:sha(port/n) for n in ("methods.py","clmoe_adapter.py","onevision_bridge.py")},
                projections=len(projected),coefficient_counts_by_projection=families,rank=total_rank,
                factor_bytes=weights["factors"],router_bytes=weights["router"],tensor_bytes=sum(weights.values()),
                dtype="float32",components="Added named model parameters, unmerged, original precision. Excludes frozen base, per-task runtime statistics, optimizer, serialization, and transport overhead.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("arm",choices=SCOPES)
    args=parser.parse_args()
    print(json.dumps(audit(args.root,args.arm)))
