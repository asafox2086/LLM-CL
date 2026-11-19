"""Continue CL-MoE from the identical first task on the corrected full-data sequence."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import full_only_20261118 as legacy
from image_disjoint_shared_20261119 import MANIFEST, MANIFEST_SHA, NAME as SHARED, bind


NAME="image_disjoint_experts_20261119"
HANDOFF="full_image_disjoint_20261119"


def replace_once(source,before,after):
    assert source.count(before)==1,before
    return source.replace(before,after)


def prepare(root,host):
    scope=root/"summary"/NAME
    port=root/"exp"/NAME
    scope.mkdir(exist_ok=True)
    port.mkdir(exist_ok=True)
    shared=legacy.read(root/"summary"/SHARED/"DESIGN.json")
    assert shared["reference_driver"]==legacy.sha(Path(legacy.__file__))
    assert shared["manifest_sha256"]==legacy.sha(root/MANIFEST)==MANIFEST_SHA
    old=legacy.read(root/"summary"/legacy.NAME/"DESIGN.json")
    assert old["host"]==host and old["backbone"]==shared["backbone"]
    assert legacy.read(root/old["manifest"])["tasks"][0]==legacy.read(root/MANIFEST)["tasks"][0]
    files={}
    for n,digest in legacy.NATIVE_FILES.items():
        path=root/"exp"/legacy.NAME/n
        assert legacy.sha(path)==digest
        data=path.read_bytes()
        if n=="run.py":
            data=replace_once(data,b"if not resume_obj: evaluate(method, processor, manifest['tasks'], data, run, 0)",
                b"assert not resume_obj\n    from image_disjoint_experts_20261119 import acquire_from_stage1\n"
                b"    acquire_from_stage1(method,processor,manifest,data,run,evaluate)")
            data=replace_once(data,b"range(resume_obj['stage']+1 if resume_obj else 1, len(manifest['tasks'])+1)",
                                  b"range(2, len(manifest['tasks'])+1)")
        bind(port/n,data)
        files[n]=legacy.sha(port/n)
    policy=legacy.read(root/"exp"/SHARED/"execution_policy.json")
    policy.update(methods=["CL_MoE"],ownership={host:["full:CL_MoE"]},checkpoint_all_stages_methods=["CL_MoE"])
    bind(port/"execution_policy.json",json.dumps(policy,indent=2).encode())
    sys.path.insert(0,str(port))
    import run
    _,m,digest,rows=run.load_protocol("full")
    counts=[len(rows[t["task"],"train"]) for t in m["tasks"]]
    assert digest==MANIFEST_SHA and sum(counts)==129462 and counts[0]==34602
    assert sum((n+15)//16 for n in counts[1:])==5936
    assert sum(len(rows[i,"test"]) for i in range(1,20))==21454
    design=dict(shared,methods=["CL_MoE"],source_sha256=legacy.sha(Path(__file__)),
                policy_sha256=legacy.sha(port/"execution_policy.json"),sources=files,
                original_manifest=old["manifest_sha256"],reused_stage1_updates=2163,new_updates=5936,
                comparison="Shared LoRA versus CL-MoE at acquisition and S19. At S19 also compare the same experts with their full-TRAIN mean gates.")
    bind(scope/"DESIGN.json",json.dumps(design,indent=2).encode())
    print(json.dumps({k:design[k] for k in ("host","train","updates","test","reused_stage1_updates","new_updates")}))


def acquire_from_stage1(method,processor,manifest,data,run,evaluate):
    import torch
    import runtime_utils as rt
    from training_validity import parameter_digest

    root=Path(os.environ["LLMCL_ROOT"])
    handoff=root/"summary"/HANDOFF
    record=legacy.read(handoff/"STAGE1_HANDOFF.json")
    ended=legacy.read(handoff/"LEGACY_WORKER_ENDED.json")
    design=legacy.read(root/"summary"/NAME/"DESIGN.json")
    source_run=root/"exp/full_only_20261118/runs"/design["backbone"]/"CL_MoE"
    assert Path(record["run"])==source_run and record["identity"]==ended["identity"]
    cp=source_run/"checkpoints/stage_01.pt"
    assert legacy.sha(cp)==record["snapshot"]["sha256"]==ended["stage1"]
    obj=torch.load(cp,map_location="cpu",weights_only=False)
    assert obj["method"]=="CL_MoE" and obj["stage"]==1 and obj["manifest_sha256"]==design["original_manifest"]
    params={n:p for n,p in method.model.named_parameters() if p.requires_grad}
    assert set(params)==set(obj["parameters"]) and set(obj["anchors"])=={1}
    with torch.no_grad():
        for n,p in params.items():
            value=obj["parameters"][n]
            assert p.shape==value.shape and p.dtype==value.dtype
            p.copy_(value)
    valid=legacy.read(source_run/"validity/stage_01.json")
    assert valid==record["validity"] and valid["updates"]==2163
    assert parameter_digest(list(params.values()))==valid["parameters_after_sha256"]
    method.stage=1
    method.anchors=obj["anchors"]
    assert len(method.layers)==len(obj["cla_layers"]) and len(method.layers) in (168,196)
    for layer,prior in zip(method.layers,obj["cla_layers"]):
        assert set(prior["task_weights"])=={1} and prior["seen"] and prior["seen"]<=set(range(8))
        assert prior["task_weights"][1].numel()==8
        layer.task_weights=prior["task_weights"]
        layer.seen=prior["seen"]
    rt.restore_rng(obj["rng"])
    del obj
    legacy.put(run/"STAGE1_PROVENANCE.json",dict(source_run=str(source_run),checkpoint=ended["stage1"],
               original_manifest=design["original_manifest"],manifest=MANIFEST_SHA,
               train_events=str(source_run/"logs/events.jsonl"),parameters_after=valid["parameters_after_sha256"]))
    rt.write(run/"validity/stage_01.json",valid)
    scores=evaluate(method,processor,[manifest["tasks"][0]],data,run,1)
    assert len(scores)==1 and (run/"checkpoints/stage_01.pt").is_file()
    rt.write(run/"stages/stage_01.json",dict(complete=True,weights_saved=True,stage=1,task_scores=scores,
             train_qa=34602,test_fraction=1.0,reused_training_from=str(source_run)))


def worker(root,host):
    d=legacy.read(root/"summary"/NAME/"DESIGN.json")
    assert d["reference_driver"]==legacy.sha(Path(legacy.__file__))
    code=replace_once(inspect.getsource(legacy.gated_comparison),"assert count==131018","assert count==129462")
    exec(compile(code,__file__,"exec"),vars(legacy))
    legacy.NAME=NAME
    legacy.MANIFEST_SHA=MANIFEST_SHA
    legacy.NATIVE_FILES=d["sources"]
    legacy.__file__=__file__
    legacy.worker(root,host,"CL_MoE")


def queue(root,host):
    scope=root/"summary"/NAME
    d=legacy.read(scope/"DESIGN.json")
    assert d["source_sha256"]==legacy.sha(Path(__file__)) and d["host"]==host
    legacy.put(scope/"QUEUE_STARTED.json",dict(pid=os.getpid(),startticks=legacy.ticks(os.getpid()),unix=time.time()))
    handoff=root/"summary"/HANDOFF
    while True:
        if (handoff/"LEGACY_WORKER_ENDED.json").is_file():
            cards=legacy.idle_cards()
            enough=legacy.available(root)>=(6800000000 if host=="new3090" else 3700000000)
            if cards and enough and (host!="new3090" or len(legacy.occupied_by_project(root))<3):
                # The shared study chooses the lowest idle card. Keep its admission first.
                shared_claim=root/"summary"/SHARED/"IDENTITY.json"
                if shared_claim.exists() or len(cards)>1:
                    gpu=cards[0] if shared_claim.exists() else cards[1]
                    if gpu in legacy.idle_cards():
                        break
        time.sleep(900)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,LLMCL_ROOT=str(root),LLMCL_HOST=host,
             LLMCL_BACKBONE=d["backbone"],PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",
             OMP_NUM_THREADS="8",MKL_NUM_THREADS="8")
    with (scope/"train.log").open("x") as f:
        proc=subprocess.Popen([sys.executable,__file__,"worker",str(root),host],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
        legacy.put(scope/"IDENTITY.json",dict(pid=proc.pid,startticks=legacy.ticks(proc.pid),gpu=gpu,unix=time.time()))
        rc=proc.wait()
    legacy.put(scope/"EXIT.json",dict(pid=proc.pid,rc=rc,unix=time.time()))
    return rc


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=("prepare","queue","worker"))
    parser.add_argument("root",type=Path)
    parser.add_argument("host",choices=legacy.HOSTS)
    args=parser.parse_args()
    sys.exit(dict(prepare=prepare,queue=queue,worker=worker)[args.mode](args.root,args.host))
