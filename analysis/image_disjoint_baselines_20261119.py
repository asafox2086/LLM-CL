"""Remaining full-data baselines on the same held-out images as shared LoRA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import full_baselines_20261118 as legacy
import full_only_20261118 as common
from image_disjoint_shared_20261119 import MANIFEST, MANIFEST_SHA, NAME as SHARED, bind


SCOPE = "image_disjoint_baselines_20261119"


def verify_data(port, kind, root, host):
    if kind == "fullft":
        # Use the actual input checks from main without constructing a CUDA model.
        src = (port/"run.py").read_text()
        start = src.index("    mp=a.manifest.resolve();")
        end = src.index("    run=a.run_dir.resolve();",start)
        import textwrap
        ns = dict(core=SimpleNamespace(ROOT=root), a=SimpleNamespace(condition="full",
                  manifest=root/MANIFEST,manifest_sha256=MANIFEST_SHA), hashlib=hashlib,json=json)
        exec(compile(textwrap.dedent(src[start:end]),str(port/"run.py"),"exec"),ns)
        rows = ns["data"]
    else:
        # Separate interpreters keep the dataset loader and helpers bound to their port.
        if str(port) not in sys.path:
            sys.path.insert(0,str(port))
        import run
        assert Path(run.__file__).resolve().parent == port
        _,_,digest,rows = run.load_protocol("full")
        assert digest == MANIFEST_SHA
    train = [len(rows[i,"train"]) for i in range(1,20)]
    totals = (sum(train),sum((n+15)//16 for n in train),sum(len(rows[i,"test"]) for i in range(1,20)))
    assert totals == (129462,8099,21454)
    print(json.dumps(dict(kind=kind,host=host,train=totals[0],main_updates=totals[1],test=totals[2])))


def prepare(root, host):
    scope = root/"summary"/SCOPE
    scope.mkdir(exist_ok=True)
    source_design = common.read(root/"summary"/legacy.SCOPE/"DESIGN.json")
    assert source_design["source"] == common.sha(Path(legacy.__file__))
    assert source_design["host"] == host and common.sha(root/MANIFEST)==MANIFEST_SHA
    ports = {}
    checks = {}
    for kind, source_files in source_design["ports"].items():
        port = root/"exp"/(SCOPE+"_"+kind)
        port.mkdir(exist_ok=True)
        for name,digest in source_files.items():
            old = root/"exp"/(legacy.SCOPE+"_"+kind)/name
            assert common.sha(old)==digest
            data = old.read_bytes()
            if name == "execution_policy.json":
                policy = json.loads(data)
                assert set(policy["conditions"]) == {"full"}
                policy["conditions"]["full"].update(manifest=MANIFEST,manifest_sha256=MANIFEST_SHA)
                data = json.dumps(policy,indent=2).encode()
            elif kind == "fullft" and name == "run.py":
                assert data.count(b"{'full':131018}") == 1
                data = data.replace(b"{'full':131018}",b"{'full':129462}")
            bind(port/name,data)
        env=dict(os.environ,LLMCL_ROOT=str(root),LLMCL_HOST=host,LLMCL_BACKBONE=common.HOSTS[host],
                 PYTHONDONTWRITEBYTECODE="1")
        output = subprocess.check_output([sys.executable,__file__,"verify",str(root),host,"--kind",kind],env=env,text=True)
        checks[kind]=json.loads(output.strip().splitlines()[-1])
        ports[kind]={n:common.sha(port/n) for n in source_files}
    design=dict(host=host,methods=legacy.inventory(host),manifest=MANIFEST,manifest_sha256=MANIFEST_SHA,
                train=129462,main_updates=8099,test=21454,seed=42,source=common.sha(Path(__file__)),
                legacy_driver=source_design["source"],ports=ports,input_checks=checks)
    bind(scope/"DESIGN.json",json.dumps(design,indent=2).encode())
    print(json.dumps(dict(host=host,methods=design["methods"],checks=checks)))


def eligible_gpu(root,host):
    shared=root/"summary"/SHARED
    if not (shared/"IDENTITY.json").exists():
        return None
    cards=common.idle_cards()
    ours=common.occupied_by_project(root)
    # The clean expert continuation needs its own card after the legacy S1 exits.
    handoff=root/"summary/full_image_disjoint_20261119"
    expert=root/"summary/image_disjoint_experts_20261119"
    legacy_run=root/"summary"/common.NAME/"CL_MoE_IDENTITY.json"
    old_gpu=common.read(legacy_run)["gpu"]
    legacy_holds=not (handoff/"LEGACY_WORKER_ENDED.json").exists() and old_gpu in ours
    missing=int(not legacy_holds and not (expert/"IDENTITY.json").exists())
    if host=="old2080" and not (root/"summary/image_disjoint_rank64_20261119/IDENTITY.json").exists():
        missing+=1
    if len(cards)<=missing or (host=="new3090" and len(ours)+missing>=3):
        return None
    return cards[-1]


def storage_for_research(root,host):
    backbone=common.HOSTS[host]
    studies=[(SHARED,"SeqFT_LoRA",110000000),
             ("image_disjoint_experts_20261119","CL_MoE",6050000000 if host=="new3090" else 3020000000)]
    if host=="old2080":
        studies.append(("image_disjoint_rank64_20261119","CL_MoE",3100000000))
    total=0
    for name,method,budget in studies:
        if (root/"summary"/name/"EXIT.json").exists():
            continue
        ck=root/"exp"/name/"runs"/backbone/method/"checkpoints"
        total+=max(0,budget-sum(p.stat().st_size for p in ck.glob("stage_*.pt")))
    return total


def queue(root,host):
    scope=root/"summary"/SCOPE
    design=common.read(scope/"DESIGN.json")
    assert design["host"]==host and design["source"]==common.sha(Path(__file__))
    assert design["legacy_driver"]==common.sha(Path(legacy.__file__))
    common.put(scope/"QUEUE_STARTED.json",dict(pid=os.getpid(),startticks=common.ticks(os.getpid()),unix=time.time()))
    pending=legacy.inventory(host)
    while pending:
        gpu=eligible_gpu(root,host)
        if gpu is None:
            time.sleep(900)
            continue
        space=common.available(root)-storage_for_research(root,host)
        job=next((j for j in pending if legacy.SPACE[j[0]][j[1]]*1024**3<=space),None)
        if job is None:
            time.sleep(900)
            continue
        backbone,method=job
        kind={"FullFT":"fullft","FOREVER":"forever"}.get(method,"native")
        port=root/"exp"/(SCOPE+"_"+kind)
        assert common.sha(root/MANIFEST)==MANIFEST_SHA
        assert all(common.sha(port/n)==v for n,v in design["ports"][kind].items())
        run=port/"runs"/backbone/"full"/method
        assert not run.exists()
        args=["--condition","full","--run-dir",str(run)]
        if kind=="fullft":
            args+=["--backbone",backbone,"--manifest",str(root/MANIFEST),"--manifest-sha256",MANIFEST_SHA]
        else:
            args+=["--method",method]
        role="new3090_onevision" if host=="new3090" and backbone=="onevision" else host
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,LLMCL_HOST=role,LLMCL_BACKBONE=backbone,
                 LLMCL_ROOT=str(root),OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",PYTHONUNBUFFERED="1")
        if eligible_gpu(root,host)!=gpu:
            continue
        label=backbone+"_"+method
        with (scope/(label+".log")).open("x") as log:
            proc=subprocess.Popen([sys.executable,str(port/"run.py"),*args],cwd=root,env=env,
                                  stdout=log,stderr=subprocess.STDOUT)
            common.put(scope/(label+"_IDENTITY.json"),dict(pid=proc.pid,startticks=common.ticks(proc.pid),
                       gpu=gpu,run=str(run),unix=time.time()))
            rc=proc.wait()
        common.put(scope/(label+"_EXIT.json"),dict(rc=rc,pid=proc.pid,unix=time.time()))
        pending.remove(job)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","verify","queue"))
    parser.add_argument("root",type=Path)
    parser.add_argument("host",choices=common.HOSTS)
    parser.add_argument("--kind",choices=("fullft","native","forever"))
    args=parser.parse_args()
    if args.action=="verify":
        verify_data(args.root/"exp"/(SCOPE+"_"+args.kind),args.kind,args.root,args.host)
    else:
        (prepare if args.action=="prepare" else queue)(args.root,args.host)
