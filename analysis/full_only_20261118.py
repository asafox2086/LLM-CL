"""Run the full-data CL-MoE and shared-LoRA comparison on unoccupied GPUs."""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
import types

NAME = "full_only_20261118"
METHODS = ("CL_MoE", "SeqFT_LoRA")
HOSTS = {"old2080": "onevision", "new3090": "qwen"}
NATIVE_FILES = {"training_validity.py":"0ec302d2fa4e8bb1a87b41c861c3aea78bf473e3df805b87d4dd450d21a2f735","parsp_adapter.py":"d1ebe1831e6d089d434bdde34b59db64eff17dc74d8d77f9ca4f080a51da8366","pass_adapter.py":"c8e23c85a93e97130d464b4b6d87110470035c5af37f9575fce705923c80c7c4","frozen_vision.py":"8211651e92d17019f6ce6cf9615df7f6b5fcaeecc765aec4acdf916c448a35bd","clmoe_adapter.py":"33946eddfc6424638ded0675ded68860273859c2f71993a6a60bd22556341be9","replay_adapter.py":"da07cfd8e44183ab3e29f1ab51ecbec0f6e7877ac6335efcf179158afff8d141","osft_adapter.py":"60b05f3b59f3878e65ee9543657b54a5aa5da13c14e139123e1c3591f3007b34","sparta_adapter.py":"8a00505b40663e5a233ddceb34e295d9b7b73d55f0679ac3d2b93318e77d7a44","native_auxiliary.py":"17f3345fb8d4b4cb3367de2a0b914b34bf69251ecc7f141ba1c7179006599682","pipeline.py":"a05c27c773dc69369cbef90f89a77c6224d1f5a5d1716c1f8576ac3ff823be17","hyper_adapter.py":"518574b8c3d1c46400d240cc984f0d1c83cb01d96a7d1e2bf788bea4d4610976","sparse_results.py":"07a7808beff8bc3330e92dc5b0fb5443401486846db764c912a5a0f57558fa3f","fv_adapter.py":"85473b19726249703f645a30142b6cfb046a37f798430ca1a255e48213641fab","methods.py":"e5bd8f51690b006274ed5adcfa4ef3b1a2c3b0f1d79476d948b71d1cfec41c5b","prepare_native.py":"200affb091e040e4f4492fb92389b9f1b54a3e71b33683e7de322ca6dd74cb4f","retrieval.py":"7d7a9dd6cb223bf96cd9ab3499a4ddf69f8fbb60940d80db84582ad518f5c133","checkpoints.py":"06c53eab2c6810901f5206edbfb1df9defabdc747b1856ace0afa31e74b37a78","checkpoint_probe.py":"71b38fb9f202fa2fa012e76a883e23322f68ef938c9f8ef9fac78d01886eb23f","host_adam.py":"5e733e6b87c12eb9331ff23e8919a142f1861a2beec82ec8b0361d878b9aec8c","run.py":"0e1f794ff1daf0694bdf960c3f89ccd4078764577db75271d8f78684282a39a9","freeze_splits.py":"5b4d58451ef3bcdb9fc033c0c9bc15b72ddc36877a64ca22d50f12e9e301510a","onevision_bridge.py":"f3b1dd38af3b58dbfa69073da698ea2bf39a3a11850fb1069a80e38d0e3669a9","resource_monitor.py":"a161c8a82bf762622795f64210c95164fb905c6fa80572b35ae5ee3dd7146533","aler_adapter.py":"4a4c58024727e5e5d01254a6b90b721e873ee6ea9f74d59308da6aa424217635","runtime_utils.py":"4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240"}
MANIFEST_SHA = "cf737555dab3646c9916dc63a66f1d52285eebbeed4aa414e807ae5b08a0d952"

def read(p):
    return json.loads(p.read_bytes())

def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()

def put(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

def prepare(root, host, policy):
    scope = root/"summary"/NAME
    orig = root/"exp/cl_full_data_research_20261105"
    port = root/"exp"/NAME
    for filename, digest in NATIVE_FILES.items():
        assert sha(orig/filename) == digest, filename
    info = policy["conditions"]["full"]
    manifest = root/info["manifest"]
    assert sha(manifest) == info["manifest_sha256"] == MANIFEST_SHA
    m = read(manifest)
    train = test = updates = 0
    missing = set()
    for t in m["tasks"]:
        split_ids = {}
        for split in ("train","test"):
            d = t["files"][split]
            source = root/d["path"]
            assert sha(source) == d["sha256"]
            ids = set()
            for line in source.read_bytes().splitlines():
                row = json.loads(line)
                assert row["id"] not in ids and row["answer"].strip()
                ids.add(row["id"])
                for image in row["images"]:
                    if not (root/image).is_file():
                        missing.add(image)
            assert len(ids) == d["count"]
            split_ids[split] = ids
        assert split_ids["train"].isdisjoint(split_ids["test"])
        n = len(split_ids["train"])
        train += n
        test += len(split_ids["test"])
        updates += math.ceil(n / 16)
    assert not missing, list(missing)[:5]
    assert (len(m["tasks"]),train,test,updates) == (19,131018,21454,8197)
    scope.mkdir(parents=True, exist_ok=True)
    port.mkdir(parents=True, exist_ok=True)
    for fn,digest in NATIVE_FILES.items():
        dest = port/fn
        if not dest.exists():
            shutil.copyfile(orig/fn, dest)
        assert sha(dest) == digest
    policy["conditions"] = {"full": dict(info, allowed_gpus=list(range(4)))}
    policy["hosts"] = {host:dict(policy["hosts"][host],reserved_gpus=[],conditions={"full":list(range(4))})}
    policy["methods"] = list(METHODS)
    policy["checkpoint_all_stages_methods"] = list(METHODS)
    policy["evaluation"] = "Full current task at acquisition; all 19 tasks at end. No initial TEST or next-task TEST."
    if (port/"execution_policy.json").exists():
        assert read(port/"execution_policy.json") == policy
    else:
        put(port/"execution_policy.json",policy)
    design = dict(host=host,backbone=HOSTS[host],methods=METHODS,condition="full",manifest=info["manifest"],
                  manifest_sha256=MANIFEST_SHA,train=train,test=test,updates=updates,epochs=1,seed=42,
                  policy_sha256=sha(port/"execution_policy.json"),source_sha256=sha(Path(__file__)),
                  early_evaluation="all current-task TEST QA after training that task",
                  final_evaluation="all original TEST QA; macro is an unweighted mean of five domain QA means",
                  fixed_reference="For frozen S19 experts, average effective gate over answer/EOS positions within each QA, then uniformly over all full TRAIN QA. Fixed weight per layer; same checkpoint and generation as native dynamic.",
                  checkpoints="All 19 compact snapshots before TEST. Frozen pretrained base, tokenizer and original code required for reconstruction.",
                  previous_full="Prior OneVision run stopped in stage 1 without a usable stage checkpoint. Fresh training in an independent directory.")
    put(scope/"DESIGN.json",design)
    print(json.dumps(dict(host=host,train=train,test=test,updates=updates,models_loaded=0)),flush=True)

def idle_cards():
    free = []
    output = subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used","--format=csv,noheader,nounits"],text=True)
    for line in output.splitlines():
        if not line.strip():
            continue
        gpu,used = [x.strip() for x in line.split(",")]
        if int(used) <= 64:
            pids = subprocess.check_output(["nvidia-smi","--id="+gpu,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
            if not pids.strip():
                free.append(gpu)
    return free

def occupied_by_project(root):
    used=set()
    listing=subprocess.check_output(["nvidia-smi","--query-gpu=index","--format=csv,noheader,nounits"],text=True)
    for gpu in listing.split():
        pids=subprocess.check_output(["nvidia-smi","--id="+gpu,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
        for word in pids.split():
            if not word.isdigit():
                continue
            proc=Path("/proc")/word
            try:
                if proc.stat().st_uid!=os.getuid():
                    continue
                cwd=(proc/"cwd").resolve(strict=True)
                if cwd==root or root in cwd.parents or str(root).encode()+b"/" in (proc/"cmdline").read_bytes():
                    used.add(gpu)
            except (FileNotFoundError,PermissionError,ProcessLookupError):
                continue
    return used

def available(root):
    device=subprocess.check_output(["df","--output=source",str(root)],text=True).splitlines()[-1].strip()
    text=subprocess.run(["quota","-w"],capture_output=True,text=True).stdout
    row=next(v.split() for v in text.splitlines() if v.split() and v.split()[0]==device)
    used,soft,hard=[int(x.rstrip("*")) for x in row[1:4]]
    return min(shutil.disk_usage(root).free,(min(n for n in (soft,hard) if n)-used)*1024)

def queue(root, host):
    scope=root/"summary"/NAME
    port=root/"exp"/NAME
    design=read(scope/"DESIGN.json")
    assert design["source_sha256"]==sha(Path(__file__)) and design["host"]==host
    assert sha(port/"execution_policy.json")==design["policy_sha256"]
    put(scope/"QUEUE_STARTED.json",dict(pid=os.getpid(),startticks=ticks(os.getpid()),unix=time.time(),host=host))
    pending=list(METHODS)
    active={}
    failed=[]
    estimate = dict(zip(METHODS,([3020000000,60000000] if host=="old2080" else [6050000000,110000000])))
    while pending or active:
        for method,(proc,log,gpu) in list(active.items()):
            rc=proc.poll()
            if rc is not None:
                log.close()
                put(scope/(method+"_EXIT.json"),dict(pid=proc.pid,rc=rc,unix=time.time()))
                del active[method]
                if rc!=0:
                    failed.append(method)
        for method in list(pending):
            cards=[g for g in idle_cards() if g not in {v[2] for v in active.values()}]
            if not cards or (host=="new3090" and len(occupied_by_project(root) | {v[2] for v in active.values()})>=3):
                break
            remaining=0
            for other in active:
                ck=port/"runs"/HOSTS[host]/other/"checkpoints"
                remaining+=max(0,estimate[other]-sum(f.stat().st_size for f in ck.glob("stage_*.pt")))
            if available(root) < estimate[method]+remaining+800*1024**2:
                continue
            gpu=cards[0]
            if gpu not in idle_cards():
                continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,LLMCL_ROOT=str(root),LLMCL_HOST=host,
                     LLMCL_BACKBONE=HOSTS[host],PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="8",MKL_NUM_THREADS="8")
            log=(scope/(method+".log")).open("x")
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker",str(root),host,method],cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env)
            put(scope/(method+"_IDENTITY.json"),dict(pid=proc.pid,startticks=ticks(proc.pid),gpu=gpu,unix=time.time()))
            active[method]=(proc,log,gpu)
            pending.remove(method)
        if pending or active:
            time.sleep(900)
    if failed:
        raise RuntimeError(failed)
    put(scope/"PAIR_FINISHED.json",dict(methods=list(METHODS),unix=time.time()))

def ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ",1)[1].split()[19]

def gated_comparison(method,processor,tasks,data,where,rt,fv,port):
    import torch
    model=method.model
    layers={k:l for k,l in model.named_modules() if type(l).__name__=="CLMoELinear"}
    assert len(layers) in (168,196)
    sums={k:torch.zeros(8,device=model.device,dtype=torch.float64) for k in layers}
    count=0
    positions=None
    seen=set()
    def observe(name,layer):
        def hook(router,inputs,logits):
            if positions is None:
                return
            start,end=positions
            assert name not in seen and logits.shape==(1,end+1,8)
            seen.add(name)
            with torch.autocast("cuda",enabled=False):
                w=logits[:,start:end].float().softmax(-1)
                if layer.global_weight is not None:
                    w=.5*w+.5*layer.global_weight.to(w.device,dtype=w.dtype)
                sums[name].add_(w.double().mean((0,1)))
        return hook
    hooks=[l.router.register_forward_hook(observe(k,l)) for k,l in layers.items()]
    model.eval()
    try:
        with torch.inference_mode():
            for task in tasks:
                i=task["task"]
                method.set_instruction(i)
                for row in data[i,"train"]:
                    batch,_=rt.input_batch(row,task,processor,model.device)
                    fv.prepare(batch,row,model,method.vision_original)
                    method.prepare_row(row,batch)
                    start=batch["input_ids"].shape[1]-1
                    size=len(processor.tokenizer(row["answer"],add_special_tokens=False,truncation=True,max_length=512)["input_ids"])+1
                    seen.clear()
                    positions=(start,start+size)
                    with torch.autocast("cuda",dtype=model.dtype):
                        _,actual=rt.answer_loss(model,batch,row["answer"],processor)
                    positions=None
                    assert size==actual and len(seen)==len(layers)
                    count+=1
                print(json.dumps(dict(event="full_train_gate_mean",task=i,qa=count)),flush=True)
    finally:
        for h in hooks:
            h.remove()
    assert count==131018
    coeff={k:(v/count).float() for k,v in sums.items()}
    for w in coeff.values():
        assert torch.isfinite(w).all() and (w>=0).all() and abs(w.sum().item()-1)<1e-4
    put(where/"FIXED_COEFFICIENTS.json",dict(train_qa=count,weights={k:v.cpu().tolist() for k,v in coeff.items()}))
    original={n:l.forward for n,l in layers.items()}
    first=next(iter(original.values())).__func__
    source=textwrap.dedent(inspect.getsource(first))
    assert sha(port/"clmoe_adapter.py")==NATIVE_FILES["clmoe_adapter.py"] and source.count("delta=torch.zeros_like")==1
    # Preserve the indentation of the native forward branch.
    raw=textwrap.dedent(inspect.getsource(first)).splitlines()
    lines=[]
    for line in raw:
        if "delta=torch.zeros_like" in line:
            lines.append(line[:len(line)-len(line.lstrip())]+"w=self._full_fixed.expand_as(w)")
        lines.append(line)
    ns=dict(first.__globals__)
    exec(compile("\n".join(lines),str(port/"clmoe_adapter.py")+":fixed","exec"),ns)
    try:
        for k,l in layers.items():
            l._full_fixed=coeff[k]
            l.forward=types.MethodType(ns["forward"],l)
        with torch.inference_mode():
            rt.evaluate(method,processor,tasks,data,where/"fixed",19)
    finally:
        for k,l in layers.items():
            l.forward=original[k]
            if hasattr(l,"_full_fixed"):
                del l._full_fixed

def worker(root,host,method_name):
    import torch
    scope=root/"summary"/NAME
    port=root/"exp"/NAME
    d=read(scope/"DESIGN.json")
    assert d["host"]==host and d["source_sha256"]==sha(Path(__file__))
    assert sha(port/"execution_policy.json")==d["policy_sha256"]
    assert all(sha(port/k)==v for k,v in NATIVE_FILES.items())
    gpu=os.environ["CUDA_VISIBLE_DEVICES"]
    assert gpu in idle_cards()
    sys.path.insert(0,str(port))
    import run as native
    import frozen_vision as fv
    import runtime_utils as rt
    out=port/"runs"/HOSTS[host]/method_name
    out.mkdir(parents=True,exist_ok=False)
    put(out/"STARTED.json",dict(pid=os.getpid(),startticks=ticks(os.getpid()),host=host,gpu=gpu,method=method_name))
    manifest=read(root/d["manifest"])
    native.required_tasks=lambda stage,total: [stage] if stage<total else list(range(1,total+1))
    def save_compact(method,processor,run,stage,config):
        return read(run/"checkpoints"/f"stage_{stage:02d}.json")
    def evaluate(method,processor,tasks,data,run,stage):
        if stage==0:
            return []
        path=run/"checkpoints"/f"stage_{stage:02d}.pt"
        path.parent.mkdir(exist_ok=True)
        params={k:v.detach().cpu().clone() for k,v in method.model.named_parameters() if v.requires_grad}
        assert params and not path.exists()
        payload=dict(parameters=params,stage=stage,method=method_name,rng=rt.rng(),manifest_sha256=MANIFEST_SHA,
                     anchors=getattr(method,"anchors",{}))
        if method_name=="CL_MoE":
            payload["cla_layers"]=[dict(task_weights=l.task_weights,seen=l.seen) for l in method.layers]
        tmp=path.with_suffix(".partial")
        torch.save(payload,tmp)
        os.replace(tmp,path)
        metadata=dict(stage=stage,path=str(path.relative_to(run)),bytes=path.stat().st_size,sha256=sha(path),
                      format="trainable_parameters_and_CL_state",parameters=sum(v.numel() for v in params.values()))
        put(path.with_suffix(".json"),metadata)
        rt.write(path.parent/"latest.json",metadata)
        del payload,params
        answers=rt.evaluate(method,processor,tasks,data,run,stage)
        if stage==19 and method_name=="CL_MoE":
            gated_comparison(method,processor,tasks,data,run,rt,fv,port)
        return answers
    make=native.make_method
    def make_with_first_stage_backup(*args,**kwargs):
        method=make(*args,**kwargs)
        if method_name=="CL_MoE":
            original_end_stage=method.end_stage
            def end_stage(stage):
                if stage==1:
                    cp=out/"first_stage_before_auxiliary.pt"
                    state={k:v.detach().cpu().clone() for k,v in method.model.named_parameters() if v.requires_grad}
                    tmp=cp.with_suffix(".partial")
                    torch.save(dict(stage=1,method=method_name,parameters=state,rng=rt.rng(),manifest_sha256=MANIFEST_SHA),tmp)
                    os.replace(tmp,cp)
                    put(out/"FIRST_STAGE_BEFORE_AUXILIARY.json",dict(sha256=sha(cp),updates=2163,phase="before stage-end statistics"))
                    del state
                return original_end_stage(stage)
            method.end_stage=end_stage
        return method
    native.make_method=make_with_first_stage_backup
    native.evaluate=evaluate
    native.save_checkpoint=save_compact
    native.aggregate_sparse=lambda *args:None
    sys.argv=[str(port/"run.py"),"--method",method_name,"--condition","full","--run-dir",str(out)]
    native.main()
    branches=[out]+([out/"fixed"] if method_name=="CL_MoE" else [])
    for branch in branches:
        total=0
        for task in manifest["tasks"]:
            i=task["task"]
            desc=task["files"]["test"]
            truth=[json.loads(l) for l in (root/desc["path"]).read_bytes().splitlines()]
            p=branch/f"evaluations/stage_19/T{i:02d}_test_predictions.jsonl"
            rows=[json.loads(l) for l in p.read_bytes().splitlines()]
            assert len(rows)==len(truth)==desc["count"]
            assert all(x["id"]==y["id"] and x["references"]==y["answers"] and x["generation_cap"]==task["max_new_tokens"] for x,y in zip(rows,truth))
            avg=sum(x["scores"]["token_f1"] for x in rows)/len(rows)
            assert abs(avg-read(p.with_suffix(".scores.json"))["token_f1"])<1e-8
            total+=len(rows)
        assert total==21454
    put(out/"COMPLETE.json",dict(train=d["train"],test_per_branch=21454,branches=len(branches),stages=19,unix=time.time()))

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=["prepare","queue","worker"])
    cli.add_argument("root",type=Path)
    cli.add_argument("host",choices=HOSTS)
    cli.add_argument("method",choices=METHODS,nargs="?")
    args=cli.parse_args()
    if args.mode=="prepare":
        prepare(args.root,args.host,read(Path(__file__).with_name("full_only_native_policy_20261118.json")))
    elif args.mode=="queue":
        queue(args.root,args.host)
    else:
        worker(args.root,args.host,args.method)
