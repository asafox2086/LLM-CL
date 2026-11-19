"""Read complete image-disjoint endpoints and training costs, paired on identical QA."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import full_data_paired_results_20261119 as paired
from full_only_20261118 import NATIVE_FILES, read, sha
from image_disjoint_shared_20261119 import MANIFEST, MANIFEST_SHA


SCOPES={"shared":"image_disjoint_shared_20261119","single":"image_disjoint_rank64_20261119",
        "eight":"image_disjoint_experts_20261119"}
METRIC="token_f1"


def terminal(root,arm,design,tasks):
    name=SCOPES[arm]
    method="SeqFT_LoRA" if arm=="shared" else "CL_MoE"
    scope=root/"summary"/name
    port=root/"exp"/name
    run=port/"runs"/design["backbone"]/method
    ident=read(scope/"IDENTITY.json")
    end=read(scope/"EXIT.json")
    start=read(run/"STARTED.json")
    assert end["rc"]==0 and end["pid"]==ident["pid"]==start["pid"]
    assert start["startticks"]==ident["startticks"] and start["method"]==method
    stat=Path("/proc",str(ident["pid"]),"stat")
    try:
        fields=stat.read_text().rsplit(") ",1)[1].split()
    except FileNotFoundError:
        fields=None
    assert fields is None or fields[19]!=str(ident["startticks"])
    done=read(run/"COMPLETE.json")
    assert (done["stages"],done["train"],done["test_per_branch"],done["branches"])==(19,129462,21454,2 if arm=="eight" else 1)
    status=read(run/"run_status.json")
    assert status["status"]=="completed" and status["completed_stage"]==19
    cfg=read(run/"config.json")
    assert (cfg["manifest_sha256"],cfg["condition"],cfg["method"],cfg["seed"],cfg["epochs"],cfg["batch_size"])==(MANIFEST_SHA,"full",method,42,1,16)
    assert (cfg["host"],cfg["backbone"])==(design["host"],design["backbone"])
    assert sha(root/"analysis"/(name+".py"))==design["source_sha256"]
    assert sha(port/"execution_policy.json")==design["policy_sha256"]
    sources=dict(NATIVE_FILES)
    if arm=="single":
        sources["clmoe_adapter.py"]=design["adapter_sha256"]
    if arm=="eight":
        sources=design["sources"]
    assert cfg["execution_source_sha256"]==sources
    assert all(sha(port/k)==v for k,v in sources.items())
    stages={}
    for t in tasks:
        i=t["task"]
        n=t["files"]["train"]["count"]
        v=read(run/f"validity/stage_{i:02}.json")
        s=read(run/f"stages/stage_{i:02}.json")
        assert v["passed"] and v["actual_parameters_changed"] and v["updates"]==(n+15)//16
        assert s["complete"] and s["weights_saved"] and s["train_qa"]==n
        meta=read(run/f"checkpoints/stage_{i:02}.json")
        assert meta["stage"]==i and (run/meta["path"]).stat().st_size==meta["bytes"]
        stages[i]=v["updates"]
    assert sum(stages.values())==8099 and sha(run/meta["path"])==meta["sha256"]
    if arm=="eight":
        assert read(run/"FIXED_COEFFICIENTS.json")["train_qa"]==129462
    return run,dict(identity=ident,exit=end,config_sha256=sha(run/"config.json"),
                    final=meta,complete_sha256=sha(run/"COMPLETE.json"))


def timing(root,run,arm,tasks):
    origins={run:set(range(1,20))}
    if arm=="eight":
        receipt=read(run/"STAGE1_PROVENANCE.json")
        source=Path(receipt["source_run"])
        assert source==root/"exp/full_only_20261118/runs"/run.parent.name/"CL_MoE"
        ended=read(root/"summary/full_image_disjoint_20261119/LEGACY_WORKER_ENDED.json")
        assert receipt["checkpoint"]==ended["stage1"]==sha(source/"checkpoints/stage_01.pt")
        assert receipt["manifest"]==MANIFEST_SHA
        origins={source:{1},run:set(range(2,20))}
    layout={}
    per_stage={}
    hashes={}
    for origin,required in origins.items():
        events=origin/"logs/events.jsonl"
        resource=origin/"resources.jsonl"
        hashes[str(origin.relative_to(root))]=[sha(events),sha(resource)]
        updates={i:[] for i in required}
        with events.open() as f:
            for line in f:
                event=json.loads(line)
                if event.get("event")=="train_update" and event["stage"] in required:
                    updates[event["stage"]].append(event)
        phases={i:{} for i in required}
        with resource.open() as f:
            for line in f:
                e=json.loads(line)
                i=e["stage"]
                tag=e["phase"]
                if i in required and tag in ("stage_before_auxiliary","stage_training_end_with_optimizer","stage_training_end_persistent"):
                    assert tag not in phases[i]
                    phases[i][tag]=e["unix"]
        for i in sorted(required):
            n=tasks[i-1]["files"]["train"]["count"]
            es=updates[i]
            assert len(es)==(n+15)//16
            assert all((e["step"],e["total_qa"],e["qa_seen"])==(k,n,min(16*k,n)) for k,e in enumerate(es,1))
            assert all(math.isfinite(e["seconds"]) and e["seconds"]>0 for e in es)
            layout[i]=[(e["qa_seen"],e["prompt_tokens"],e["target_tokens"]) for e in es]
            ts=phases[i]
            begin,after_steps,end=(ts[k] for k in ("stage_before_auxiliary","stage_training_end_with_optimizer","stage_training_end_persistent"))
            assert begin<=after_steps<=end
            per_stage[i]=dict(main_update_seconds=sum(e["seconds"] for e in es),
                              end_stage_seconds=end-after_steps,stage_seconds=end-begin)
    return layout,dict(stages=per_stage,files=hashes,
                       totals={k:sum(s[k] for s in per_stage.values()) for k in per_stage[1]})


def changes(matrix):
    # First four columns are paired acquisition/final F1 for the two methods.
    result={}
    common=(matrix[:,0]==100)&(matrix[:,2]==100)
    result["both_full_credit_at_acquisition"]=int(common.sum())
    for j,name in ((0,"first"),(2,"eight")):
        before,after=matrix[:,j],matrix[:,j+1]
        delta=after-before
        up=float(np.maximum(delta,0).mean())
        down=float(np.maximum(-delta,0).mean())
        assert np.isclose(up-down,after.mean()-before.mean())
        result[name]=dict(positive_points_per_question=up,lost_points_per_question=down,
                          net_points_per_question=up-down,improved=int((delta>0).sum()),
                          worsened=int((delta<0).sum()),unchanged=int((delta==0).sum()),
                          still_full_credit=int((after[common]==100).sum()),
                          mean_final_on_common_full_credit=float(after[common].mean()) if common.any() else None)
    return result


def compare(root,first,execute):
    sd=root/"summary"
    arms=(first,"eight")
    designs={a:read(sd/SCOPES[a]/"DESIGN.json") for a in arms}
    assert len({(d["backbone"],d["host"],d["manifest_sha256"]) for d in designs.values()})==1
    assert all(d["manifest"]==MANIFEST and d["manifest_sha256"]==MANIFEST_SHA for d in designs.values())
    original=read(sd/"full_only_20261118/DESIGN.json")
    tasks,truth,groups=paired.load_test(root,original)
    assert sha(root/MANIFEST)==MANIFEST_SHA
    clean=read(root/MANIFEST)["tasks"]
    for t,u in zip(tasks,clean):
        assert {k:v for k,v in t.items() if k not in ("files","training_order_sha256")}=={
            k:v for k,v in u.items() if k not in ("files","training_order_sha256")}
        assert t["files"]["test"]==u["files"]["test"]
    assert len(clean)==len(tasks)==19
    scope=sd/"image_disjoint_results_20261119"
    protocol=dict(first=first,eight="eight",backbone=original["backbone"],train=129462,updates=8099,
                  test=21454,manifest=MANIFEST_SHA,score=METRIC,source=sha(Path(__file__)),
                  helper_sha256=sha(Path(paired.__file__)),draws=5000,seed=20261119,
                  domains={d:dict(questions=len(g),image_clusters=int(g.max())+1) for d,g in groups.items()},
                  cost="Stage wall times include setup of each stage, updates, validation and consolidation up to the persistent snapshot, before evaluation. Timing includes the expert's reused S1 from its original worker. No inference throughput or device energy measurement.")
    if not execute:
        print(json.dumps(protocol))
        return
    scope.mkdir(exist_ok=True)
    out=scope/(first+"_vs_eight.json")
    if out.exists():
        raise FileExistsError(out)
    runs,receipts={},{}
    for a in arms:
        runs[a],receipts[a]=terminal(root,a,designs[a],clean)
    costs={};layouts={}
    for a in arms:
        layouts[a],costs[a]=timing(root,runs[a],a,clean)
    assert layouts[first]==layouts["eight"]
    values,files={},{}
    for t in clean:
        i=t["task"]
        paths=[(runs[first],i),(runs[first],19),(runs["eight"],i),(runs["eight"],19),(runs["eight"]/"fixed",19)]
        arrays={k:paired.scores(*k,t,truth[i],files,root) for k in dict.fromkeys(paths)}
        values.setdefault(t["domain"],[]).append(np.column_stack([arrays[k] for k in paths]))
    estimates=paired.paired_results(values,groups)
    for item in [estimates["macro"],*estimates["domains"].values()]:
        # The helper names its first method "shared". Give every output its actual label.
        for k in list(item):
            if "shared" in k:
                item[k.replace("shared",first)]=item.pop(k)
        if first=="single":
            for label in ("acquisition","final"):
                v=item["single_minus_expert_"+label]
                item["eight_minus_one_"+label]=dict(points=-v["points"],ci95=[-v["ci95"][1],-v["ci95"][0]])
            v=item["change_in_single_minus_expert"]
            item["change_in_eight_minus_one"]=dict(points=-v["points"],ci95=[-v["ci95"][1],-v["ci95"][0]])
    result=dict(protocol=protocol,terminal=receipts,training_costs=costs,scores=estimates,
                within_question_changes={d:changes(np.concatenate(v)) for d,v in values.items()},prediction_files=files)
    tmp=scope/f".{out.name}.{os.getpid()}"
    try:
        with tmp.open("x") as f:
            json.dump(result,f,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.link(tmp,out)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps(estimates["macro"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("first",choices=("shared","single"))
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    compare(args.root,args.first,args.execute)
