"""OneVision full-data replication of the trained single rank-64 CL expert."""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import full_only_20261118 as paired
from full_baselines_20261118 import SPACE, inventory, versioned


NAME = "full_rank64_control_20261119"
SINGLE_SHA = "0eafa79719dd82d9000b3051b16c999d7fffd9a9e85bb957dfefc8b6969f66c5"
DRIVER_SHA = "c2d65255934dc2d01a2d59ae0a77acb76a53aa50f07a0c27e26aa367b6af1d39"
# 35,438,592 FP32 coefficients per snapshot, plus tensor metadata and CL state.
SNAPSHOT = 146 * 1024**2


def reserve_save(torch, root, stage, obj, tmp):
    reserved = root / "summary" / NAME / "reserved" / str(stage)
    assert reserved.stat().st_size == SNAPSHOT and not tmp.exists()
    os.rename(reserved, tmp)
    with tmp.open("r+b") as f:
        torch.save(obj, f)
        end = f.tell()
        assert 0 < end <= SNAPSHOT
        f.truncate(end)
        f.flush()
        os.fsync(f.fileno())


def bound_worker():
    # Reuse the exact reference worker, including training, acquisition tests, checkpoints
    # and S19 validation. A one-output softmax is identically one; omit its duplicate test.
    assert paired.sha(Path(paired.__file__)) == DRIVER_SHA
    src = inspect.getsource(paired.worker)
    edits = [
        ('    assert gpu in idle_cards()', '    assert gpu == os.environ["RANK64_CLAIMED_GPU"]'),
        ('        torch.save(payload,tmp)', '        reserve_save(torch,root,stage,payload,tmp)'),
        ('torch.save(dict(stage=1,method=method_name,parameters=state,rng=rt.rng(),manifest_sha256=MANIFEST_SHA),tmp)',
         'reserve_save(torch,root,0,dict(stage=1,method=method_name,parameters=state,rng=rt.rng(),manifest_sha256=MANIFEST_SHA),tmp)'),
        ('        if stage==19 and method_name=="CL_MoE":\n            gated_comparison(method,processor,tasks,data,run,rt,fv,port)\n',''),
        ('branches=[out]+([out/"fixed"] if method_name=="CL_MoE" else [])','branches=[out]'),
    ]
    for before, after in edits:
        assert src.count(before) == 1
        src = src.replace(before,after)
    return src


def prepare(root):
    scope, port = root/"summary"/NAME, root/"exp"/NAME
    scope.mkdir(exist_ok=True)
    port.mkdir(exist_ok=True)
    ref = paired.read(root/"summary"/paired.NAME/"DESIGN.json")
    assert (ref["host"],ref["backbone"],ref["train"],ref["test"],ref["updates"]) == (
        "old2080","onevision",131018,21454,8197)
    assert paired.sha(root/ref["manifest"]) == ref["manifest_sha256"] == paired.MANIFEST_SHA
    files = dict(paired.NATIVE_FILES, **{"clmoe_adapter.py": SINGLE_SHA})
    for fname, digest in files.items():
        orig = "cl_single_expert_training_20261105" if fname=="clmoe_adapter.py" else paired.NAME
        p = root/"exp"/orig/fname
        assert paired.sha(p) == digest
        versioned(port/fname,p.read_bytes())
    policy = paired.read(root/"exp"/paired.NAME/"execution_policy.json")
    policy.update(methods=["CL_MoE"], checkpoint_all_stages_methods=["CL_MoE"],
                  ownership={"old2080":["full:CL_MoE"]})
    versioned(port/"execution_policy.json",json.dumps(policy,indent=2).encode())
    versioned(scope/"worker_body.py",bound_worker().encode())
    design = dict(ref, methods=["CL_MoE"],source_sha256=paired.sha(Path(__file__)),
                  policy_sha256=paired.sha(port/"execution_policy.json"),
                  reference_driver=DRIVER_SHA,worker_sha256=paired.sha(scope/"worker_body.py"),
                  adapter_sha256=SINGLE_SHA, reference_adapter_sha256=paired.NATIVE_FILES["clmoe_adapter.py"],
                  reference_run="exp/full_only_20261118/runs/onevision/CL_MoE",
                  experts=1,rank_per_expert=64,total_rank=64,projections=168,trainable_parameters=35438592,
                  comparison="One rank-64 versus eight rank-8 experts at seed 42, same host and GPU model. "
                             "Primary outcome is change in eight-minus-one macro F1 from task acquisition "
                             "to sequence end. Use all five domains and matched image-cluster intervals. "
                             "Expert partition, effective routing, and consolidation vary jointly.",
                  fixed_reference="One softmax output equals one on all inputs; native branch only.")
    versioned(scope/"DESIGN.json",json.dumps(design,indent=2).encode())


def free_for_control(root):
    study = root/"summary"/paired.NAME
    baseline = root/"summary/full_baselines_20261118"
    cards = set(paired.idle_cards())
    pending = 0
    required = 20*SNAPSHOT + 1024**3
    for method, size in zip(paired.METHODS, (3020000000,60000000)):
        receipt = study/(method+"_IDENTITY.json")
        if (study/(method+"_EXIT.json")).exists():
            continue
        if receipt.exists():
            cards.discard(paired.read(receipt)["gpu"])
        else:
            pending += 1
        ck = root/"exp"/paired.NAME/"runs/onevision"/method/"checkpoints"
        required += max(0, size - sum(f.stat().st_size for f in ck.glob("stage_*.pt")))
    baseline_active = False
    for receipt in baseline.glob("*_IDENTITY.json"):
        label = receipt.name.split("_IDENTITY")[0]
        if not (baseline/(label+"_EXIT.json")).exists():
            item = paired.read(receipt)
            cards.discard(item["gpu"])
            required += SPACE["onevision"][label.split("_",1)[1]] * 1024**3
            baseline_active = True
    # The existing queues choose the lowest numbered idle cards and can admit two
    # priority workers and at most one serial baseline. Never rely on their old GPU list.
    waiting = [m for b,m in inventory("old2080") if not (baseline/(b+"_"+m+"_EXIT.json")).exists()]
    baseline_may_admit = bool(waiting) and not baseline_active
    if baseline_may_admit:
        required += max(SPACE["onevision"][m] for m in waiting)*1024**3
    if len(cards) <= pending + int(baseline_may_admit):
        return None
    return max(cards,key=int) if paired.available(root)>=required else None


def worker(root):
    import torch
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    if free_for_control(root) != gpu:
        return 75
    # Hold the CUDA context before opening a run or reserving disk space. An
    # admission race exits without loading a model or touching any other PID.
    claim = torch.empty(1,device="cuda")
    pids = subprocess.check_output(["nvidia-smi","--id="+gpu,"--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"],text=True).split()
    if set(pids) != {str(os.getpid())}:
        return 75
    scope = root/"summary"/NAME
    design = paired.read(scope/"DESIGN.json")
    assert design["worker_sha256"] == paired.sha(scope/"worker_body.py")
    assert design["source_sha256"] == paired.sha(Path(__file__))
    reserved = scope/"reserved"
    reserved.mkdir()
    try:
        for i in range(20):
            with (reserved/str(i)).open("xb") as f:
                os.posix_fallocate(f.fileno(),0,SNAPSHOT)
        paired.put(scope/"IDENTITY.json",dict(pid=os.getpid(),startticks=paired.ticks(os.getpid()),
                                               gpu=gpu,unix=time.time()))
        os.environ["RANK64_CLAIMED_GPU"] = gpu
        namespace = dict(vars(paired),NAME=NAME,__file__=__file__,reserve_save=reserve_save,
                         NATIVE_FILES=dict(paired.NATIVE_FILES, **{"clmoe_adapter.py":SINGLE_SHA}))
        exec(compile((scope/"worker_body.py").read_bytes(),str(scope/"worker_body.py"),"exec"),namespace)
        namespace["worker"](root,"old2080","CL_MoE")
    finally:
        # These numbered files are this process's untouched, zero-filled reservations.
        # Once written, a checkpoint has been renamed into the run and is never here.
        for i in range(20):
            p=reserved/str(i)
            if p.exists() and p.stat().st_size==SNAPSHOT:
                p.unlink()
    return 0


def queue(root):
    scope=root/"summary"/NAME
    assert paired.read(scope/"DESIGN.json")["source_sha256"] == paired.sha(Path(__file__))
    paired.put(scope/"QUEUE_STARTED.json",dict(pid=os.getpid(),startticks=paired.ticks(os.getpid()),unix=time.time()))
    while True:
        gpu=free_for_control(root)
        if gpu is not None:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,LLMCL_ROOT=str(root),LLMCL_HOST="old2080",
                     LLMCL_BACKBONE="onevision",OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",
                     PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
            with (scope/"train.log").open("a") as log:
                p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker",str(root)],
                                   cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                rc=p.wait()
            if rc!=75 or (scope/"IDENTITY.json").exists():
                paired.put(scope/"EXIT.json",dict(pid=p.pid,rc=rc,unix=time.time()))
                return rc
        time.sleep(900)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=("prepare","queue","worker"))
    cli.add_argument("root",type=Path)
    args=cli.parse_args()
    assert args.root == Path("/data2/liyapeng_grp/program/LLMCL")
    sys.exit(dict(prepare=prepare,queue=queue,worker=worker)[args.mode](args.root))
