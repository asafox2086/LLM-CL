"""CPU verification of the shared three-stage prefix in cross4/cross5.

Two different later-task sequences are not a pure domain-count intervention.
This checks their stronger common-history invariant before interpreting them.
No GPU workers, choices or existing observers are changed.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from onevision_matched_task_domain_report import checked_rows, compare, transition

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016/matched_task_domain"
OUT = BASE / "common_prefix"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def exact_adapter(a, b):
    assert a.keys() == b.keys()
    assert all(torch.equal(a[n][f], b[n][f]) for n in a for f in ("A", "B")), "Shared-prefix parameter mismatch; no tolerance"


def self_check():
    a = {"toy": {"A": torch.ones((1,2)), "B": torch.zeros((2,1))}}
    b = {"toy": {"A": a["toy"]["A"].clone(), "B": a["toy"]["B"].clone()}}
    exact_adapter(a,b)
    b["toy"]["A"][0,0] = torch.nextafter(b["toy"]["A"][0,0], torch.tensor(float("inf")))
    try:
        exact_adapter(a,b)
    except AssertionError:
        pass
    else:
        raise AssertionError("One-ULP prefix mismatch was accepted")
    assert not torch.cuda.is_initialized()
    return {"same_prefix_exact":True,"one_ULP_mismatch_rejected":True,"CPU_only":True}


def build():
    sources={str(Path(__file__).relative_to(ROOT)):sha(__file__)}
    def check(p,digest=None):
        actual=sha(p);assert digest is None or actual==digest,str(p)
        sources[str(p.relative_to(ROOT))]=actual;return p
    plan=json.loads(check(BASE/"CPU_PLAN_CHECK.json").read_text());assert plan["passed"]
    spec=plan["spec"]
    for p,h in spec["source_sha256"].items():check(ROOT/p,h)
    lock=json.loads(check(ROOT/"summary/primary_metric_20261017/metric_lock.json").read_text())
    check(ROOT/"analysis/answer_f1.py",lock["metric_sha256"]);assert lock["option_mapping"] is False
    check(ROOT/"analysis/onevision_matched_task_domain_report.py")
    manifest=json.loads(check(ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json",spec["manifest_sha256"]).read_text());original={}
    for t in manifest["tasks"]:
        f=t["files"]["test"]
        original.update({r["id"]:r for r in map(json.loads,check(ROOT/f["path"],f["sha256"]).read_text().splitlines())})
    pairs=[];pending=[]
    for anchor in spec["anchors"]:
        for method in spec["methods"]:
            names=[f"T{anchor:02d}_{seq}__{method}" for seq in ["cross_short","cross_five_domains"]]
            files=[BASE/(name+"_complete.json") for name in names]
            if not all(p.exists() for p in files):
                pending.append({"anchor":anchor,"method":method});continue
            data=[]
            for name,path,seq in zip(names,files,["cross_short","cross_five_domains"]):
                proof=json.loads(check(path).read_text());assert proof["complete"] and proof["stage1_exact"] and proof["spec"]==spec
                run=(ROOT/proof["run"]).resolve();assert run.is_relative_to(ROOT/"exp/onevision_matched_task_domain_20261017")
                for rel,h in proof["artifact_sha256"].items():
                    p=(run/rel).resolve();assert p.is_relative_to(run);check(p,h)
                states=[torch.load(run/f"adapters/stage_{s:02d}.pt",map_location="cpu",weights_only=True)["adapter"] for s in (1,2,3)]
                grads=[torch.load(run/f"raw_step_gradients/{name}/stage_{s:02d}.pt",map_location="cpu",weights_only=True) for s in (1,2,3)]
                pred={s:checked_rows(run/f"evaluations/anchor_after{s}/anchor_image.jsonl",spec["anchor_probe_ids"][str(anchor)],original) for s in (1,3,5)}
                data.append({"states":states,"gradients":grads,"predictions":pred,"run":run,"name":name,"sequence":seq})
            a,b=data
            assert spec["sequences"][str(anchor)]["cross_short"][:3]==spec["sequences"][str(anchor)]["cross_five_domains"][:3]
            for sa,sb,ga,gb in zip(a["states"],b["states"],a["gradients"],b["gradients"]):
                exact_adapter(sa,sb)
                assert ga["parameter_layout"]==gb["parameter_layout"] and ga["train_ids"]==gb["train_ids"]
                assert torch.equal(ga["pre_intervention_gradient"],gb["pre_intervention_gradient"])
            for s in (1,3):
                assert [(r["id"],r["prediction"],r["nll"],r["scores"]) for r in a["predictions"][s]]==[(r["id"],r["prediction"],r["nll"],r["scores"]) for r in b["predictions"][s]]
            if method.startswith("EWC_"):
                for s in (1,2,3):
                    fa,fb=[torch.load(c["run"]/f"fishers/stage_{s:02d}.pt",map_location="cpu",weights_only=True) for c in data]
                    assert fa["ids"]==fb["ids"]
                    assert all(torch.equal(fa[k][n],fb[k][n]) for k in ("mean","diagonal") for n in fa[k])
            row={"anchor":anchor,"method":method,"shared_first3_tasks":spec["sequences"][str(anchor)]["cross_short"][:3],
                 "all3_states_and_first_gradients_exact":True,"stage1_and3_outputs_scores_NLL_exact":True,
                 "cross5_minus_cross4_after_shared_stage3":compare(b["predictions"][3],b["predictions"][5],a["predictions"][3],a["predictions"][5],original),
                 "cross4_stage3_to5":transition(a["predictions"][3],a["predictions"][5],original),
                 "cross5_stage3_to5":transition(b["predictions"][3],b["predictions"][5],original)}
            if method=="SeqFT_LoRA":
                full=[checked_rows(c["run"]/"evaluations/final_native_anchor/anchor_image.jsonl",spec["final_SeqFT_full_anchor_ids"][str(anchor)],original) for c in data]
                row["full_native_anchor_final_cross5_minus_cross4"]=transition(full[0],full[1],original)
            pairs.append(row)
    for p,h in sources.items():assert sha(ROOT/p)==h
    assert not torch.cuda.is_initialized()
    return {"CPU_only":True,"complete_pairs":len(pairs),"planned_pairs":12,"pairs":pairs,"pending":pending,"source_sha256":sources,"self_checks":self_check(),
            "limits":["Shared three-stage history exactly checked, but later task contents/order also differ, not a pure count effect.",
                      "32QA diagnostic and single seed/order; full native anchor final only SeqFT507/228, not21454.",
                      "Original score/ID/source preserved; no TEST tuning or incomplete prediction splice.",
                      "Conditional image intervals uncorrected, not seed/order replication."]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");args=p.parse_args();torch.set_num_threads(2)
    if args.check:print(json.dumps(self_check()));return
    result=build();OUT.mkdir(exist_ok=True)
    (OUT/"EVIDENCE.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"CPU_only":True,"complete_pairs":result["complete_pairs"],"planned_pairs":12}))


if __name__=="__main__":main()
