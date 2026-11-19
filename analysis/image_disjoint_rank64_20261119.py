"""Same-host single rank-64 expert for the image-disjoint OneVision experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import full_only_20261118 as paired
import full_rank64_control_20261119 as single
from full_baselines_20261118 import SPACE
from image_disjoint_shared_20261119 import MANIFEST, MANIFEST_SHA, NAME as SHARED, bind


NAME="image_disjoint_rank64_20261119"


def prepare(root):
    scope=root/"summary"/NAME
    port=root/"exp"/NAME
    scope.mkdir(exist_ok=True)
    port.mkdir(exist_ok=True)
    old=root/"summary"/single.NAME
    d=paired.read(old/"DESIGN.json")
    assert paired.sha(Path(single.__file__))==d["source_sha256"]
    assert paired.sha(Path(paired.__file__))==d["reference_driver"]
    assert d["host"]=="old2080" and d["experts"]==1 and d["rank_per_expert"]==64
    assert paired.sha(root/MANIFEST)==MANIFEST_SHA
    original=root/"exp"/single.NAME
    for n,v in dict(paired.NATIVE_FILES,**{"clmoe_adapter.py":d["adapter_sha256"]}).items():
        assert paired.sha(original/n)==v
        bind(port/n,(original/n).read_bytes())
    assert paired.sha(old/"worker_body.py")==d["worker_sha256"]
    bind(scope/"worker_body.py",(old/"worker_body.py").read_bytes())
    policy=paired.read(original/"execution_policy.json")
    policy["conditions"]["full"].update(manifest=MANIFEST,manifest_sha256=MANIFEST_SHA)
    bind(port/"execution_policy.json",json.dumps(policy,indent=2).encode())
    sys.path.insert(0,str(port))
    import run
    _,_,digest,rows=run.load_protocol("full")
    counts=[len(rows[i,"train"]) for i in range(1,20)]
    totals=sum(counts),sum((n+15)//16 for n in counts),sum(len(rows[i,"test"]) for i in range(1,20))
    assert digest==MANIFEST_SHA and totals==(129462,8099,21454)
    d.update(manifest=MANIFEST,manifest_sha256=digest,train=totals[0],updates=totals[1],
             source_sha256=paired.sha(Path(__file__)),reference_single_driver=d["source_sha256"],
             policy_sha256=paired.sha(port/"execution_policy.json"),
             reference_run="exp/image_disjoint_experts_20261119/runs/onevision/CL_MoE")
    d.pop("previous_full",None)
    bind(scope/"DESIGN.json",json.dumps(d,indent=2).encode())
    print(json.dumps({k:d[k] for k in ("host","train","updates","test","trainable_parameters","comparison")}))


def eligible(root):
    shared=root/"summary"/SHARED
    if not (shared/"IDENTITY.json").exists():
        return None
    cards=paired.idle_cards()
    expert=root/"summary/image_disjoint_experts_20261119"
    ended=(root/"summary/full_image_disjoint_20261119/LEGACY_WORKER_ENDED.json").exists()
    pending=int(ended and not (expert/"IDENTITY.json").exists())
    if len(cards)<=pending:
        return None
    # A baseline and the expert successor both reserve this study's disk headroom.
    # Allow for already active baselines that may still need their final checkpoint.
    need=20*single.SNAPSHOT+1024**3
    base=root/"summary/image_disjoint_baselines_20261119"
    for p in base.glob("*_IDENTITY.json"):
        label=p.name.split("_IDENTITY")[0]
        if not (base/(label+"_EXIT.json")).exists():
            need+=SPACE["onevision"][label.split("_",1)[1]]*1024**3
    for name,method,budget in ((SHARED,"SeqFT_LoRA",60000000),
                               ("image_disjoint_experts_20261119","CL_MoE",3020000000)):
        if not (root/"summary"/name/"EXIT.json").exists():
            ck=root/"exp"/name/"runs/onevision"/method/"checkpoints"
            need+=max(0,budget-sum(p.stat().st_size for p in ck.glob("stage_*.pt")))
    return cards[pending] if paired.available(root)>=need else None


def main(root,mode):
    assert root==Path("/data2/liyapeng_grp/program/LLMCL")
    if mode=="prepare":
        return prepare(root)
    design=paired.read(root/"summary"/NAME/"DESIGN.json")
    assert design["reference_single_driver"]==paired.sha(Path(single.__file__))
    assert design["reference_driver"]==paired.sha(Path(paired.__file__))
    single.NAME=NAME
    single.__file__=__file__
    single.free_for_control=eligible
    paired.MANIFEST_SHA=MANIFEST_SHA
    return (single.worker if mode=="worker" else single.queue)(root)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","queue","worker"))
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    sys.exit(main(args.root,args.mode))
