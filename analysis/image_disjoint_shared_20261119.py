"""Full image-disjoint shared q/v LoRA on either benchmark backbone."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import full_only_20261118 as original


NAME = "image_disjoint_shared_20261119"
MANIFEST = "data/FiveDomainCL/image_disjoint_20261119/full/manifest.json"
MANIFEST_SHA = "a4a3110199c6558e7500f0fff4abd005545ae28d88e3fefa289d69548b659589"
METHOD = "SeqFT_LoRA"


def bind(path, data):
    if path.exists():
        assert path.read_bytes() == data, path
    else:
        with path.open("xb") as f:
            f.write(data)


def prepare(root, host):
    scope, port = root/"summary"/NAME, root/"exp"/NAME
    scope.mkdir(exist_ok=True)
    port.mkdir(exist_ok=True)
    source = root/"exp"/original.NAME
    reference = original.read(root/"summary"/original.NAME/"DESIGN.json")
    assert original.sha(Path(original.__file__)) == reference["source_sha256"] == "c2d65255934dc2d01a2d59ae0a77acb76a53aa50f07a0c27e26aa367b6af1d39"
    assert reference["host"] == host and original.sha(root/MANIFEST) == MANIFEST_SHA
    for name,digest in original.NATIVE_FILES.items():
        assert original.sha(source/name) == digest
        bind(port/name,(source/name).read_bytes())
    policy = original.read(source/"execution_policy.json")
    policy["conditions"]["full"].update(manifest=MANIFEST,manifest_sha256=MANIFEST_SHA)
    policy.update(methods=[METHOD],ownership={host:["full:"+METHOD]},checkpoint_all_stages_methods=[METHOD])
    bind(port/"execution_policy.json",json.dumps(policy,indent=2).encode())
    sys.path.insert(0,str(port))
    import run as native
    _, m, digest, rows = native.load_protocol("full")
    train = [len(rows[t["task"],"train"]) for t in m["tasks"]]
    budgets = (sum(train),sum((n+15)//16 for n in train),sum(len(rows[i,"test"]) for i in range(1,20)))
    assert digest == MANIFEST_SHA and budgets == (129462,8099,21454)
    design = dict(host=host,backbone=original.HOSTS[host],methods=[METHOD],manifest=MANIFEST,
                  manifest_sha256=digest,train=budgets[0],updates=budgets[1],test=budgets[2],
                  seed=42,epochs=1,policy_sha256=original.sha(port/"execution_policy.json"),
                  source_sha256=original.sha(Path(__file__)),reference_driver=reference["source_sha256"],
                  comparison="Shared q/v rank-8 LoRA. Full current-task TEST after acquisition, all TEST at S19. Preserve all 19 compact checkpoints and native resource events.")
    bind(scope/"DESIGN.json",json.dumps(design,indent=2).encode())
    print(json.dumps(design))


def worker(root,host):
    # The shared worker contains no expert-specific forward or stage statistics.
    original.NAME,original.MANIFEST_SHA = NAME,MANIFEST_SHA
    original.__file__ = __file__
    original.worker(root,host,METHOD)


def queue(root,host):
    scope = root/"summary"/NAME
    design = original.read(scope/"DESIGN.json")
    assert design["host"] == host and design["source_sha256"] == original.sha(Path(__file__))
    assert design["reference_driver"] == original.sha(Path(original.__file__))
    original.put(scope/"QUEUE_STARTED.json",dict(pid=os.getpid(),startticks=original.ticks(os.getpid()),unix=time.time()))
    legacy = root/"exp"/original.NAME/"runs"/original.HOSTS[host]/"CL_MoE"/"checkpoints"
    while True:
        # Allow the existing CL worker to finish its compatible first-stage snapshot
        # and leave room for the subsequent full-size expert trajectory.
        budget = 3020000000 if host=="old2080" else 6050000000
        remaining = max(0,budget-sum(f.stat().st_size for f in legacy.glob("stage_*.pt")))
        cards = original.idle_cards()
        capacity = original.available(root) >= remaining+110000000+800*1024**2
        allowed = host!="new3090" or len(original.occupied_by_project(root))<3
        if cards and capacity and allowed:
            gpu=cards[0]
            if gpu in original.idle_cards():
                break
        time.sleep(900)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,LLMCL_ROOT=str(root),LLMCL_HOST=host,
             LLMCL_BACKBONE=original.HOSTS[host],PYTHONDONTWRITEBYTECODE="1",
             PYTHONUNBUFFERED="1",OMP_NUM_THREADS="8",MKL_NUM_THREADS="8")
    with (scope/"train.log").open("x") as f:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker",str(root),host],
                              env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT)
        original.put(scope/"IDENTITY.json",dict(pid=proc.pid,startticks=original.ticks(proc.pid),gpu=gpu,unix=time.time()))
        rc=proc.wait()
    original.put(scope/"EXIT.json",dict(pid=proc.pid,rc=rc,unix=time.time()))
    return rc


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","worker","queue"))
    parser.add_argument("root",type=Path)
    parser.add_argument("host",choices=original.HOSTS)
    args=parser.parse_args()
    sys.exit(dict(prepare=prepare,worker=worker,queue=queue)[args.mode](args.root,args.host))
