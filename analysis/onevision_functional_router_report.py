"""Read-only CPU evidence report for every predeclared routing intervention."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import traceback
from answer_f1 import answer_f1
from onevision_functional_router_queue import ROOT, STUDY, FULL, functional_plan, complete, sha, write


def build():
    spec = functional_plan()
    old = spec["parent_spec"]
    m = ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert sha(m) == old["manifest_sha256"]
    manifest = json.loads(m.read_text())
    originals, domains = {}, {}
    for t in manifest["tasks"]:
        f = t["files"]["test"]
        path = ROOT/f["path"]
        assert sha(path) == f["sha256"]
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        domains.setdefault(t["domain"], []).extend(r["id"] for r in rows)
        for r in rows:
            originals[r["id"]] = r
    proofs, cells, aggregate, activation = {}, [], [], []
    jobs = [("pilot", None)] + [("full", v) for v in FULL]
    for scope, view in jobs:
        if not complete(scope, view, spec):
            continue
        name = "pilot" if scope == "pilot" else "full_"+view
        p = STUDY/(name+"_complete.json")
        proofs[p.name] = sha(p)
        proof = json.loads(p.read_text())
        run = ROOT/proof["run"]
        conditions = spec["views"] if scope == "pilot" else [view]
        for condition in conditions:
            label = condition if scope == "pilot" else "final_full_input_router"
            generated_all = []
            for d in domains:
                path = run/f"evaluations/{label}/{d}_image.jsonl"
                rows = [json.loads(s) for s in path.read_text().splitlines()]
                ids = old["diagnostic_ids"][d] if scope == "pilot" else domains[d]
                assert [r["id"] for r in rows] == ids
                for r in rows:
                    assert r["references"] == originals[r["id"]]["answers"] and math.isfinite(r["nll"])
                    if "scores" in r:
                        assert r["scores"]["token_f1"] == answer_f1(r["prediction"], r["references"])
                generated = [r for r in rows if "scores" in r]
                assert scope == "pilot" or len(generated) == len(rows)
                generated_all.extend(generated)
                cells.append({"scope": scope, "view": condition, "domain": d, "QA_NLL": len(rows), "generated": len(generated),
                    "Answer_F1": sum(r["scores"]["token_f1"] for r in generated)/len(generated),
                    "NLL": sum(r["nll"] for r in rows)/len(rows),
                    "route_counts": dict(Counter(str(r["route_expert"]) for r in rows)),
                    "gain_min": min(r.get("route_residual_gain", 1.) for r in rows),
                    "gain_max": max(r.get("route_residual_gain", 1.) for r in rows)})
            tasks = {t["task"]: [r["scores"]["token_f1"] for r in generated_all if r["native_task"] == t["task"]] for t in manifest["tasks"]}
            assert all(tasks.values())
            aggregate.append({"scope": scope, "view": condition, "generated": len(generated_all),
                "native19_AP": sum(sum(v)/len(v) for v in tasks.values())/19,
                "domain_macro_F1": sum(c["Answer_F1"] for c in cells if c["scope"] == scope and c["view"] == condition)/5,
                "full_FWT": None, "full_BWT": None})
        if scope == "pilot":
            activation.append(json.loads((run/"calibration_summary.json").read_text()))
    return {"CPU_only": True, "completion_sha256": proofs, "cells": cells, "aggregate": aggregate, "calibration": activation,
            "pilot_completed": "pilot_complete.json" in proofs, "full_completed": sum("full_"+v+"_complete.json" in proofs for v in FULL),
            "limits": ["All four fixed-pool views; no TEST chooses calibration, router or full candidates.",
                       "16 new parameters, zero new LoRAs; inherited pool/projection/cache/temporary costs remain charged.",
                       "A routing gain would identify an access/aggregation component, not prove knowledge is sufficient.",
                       "Diagnostic500NLL/155generated differ from full21454; no full transfer metrics from final-only.",
                       "Adaptation of published principles, not original MVP/D-MoLE reproduction or proven novel method."],
            "report_source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}}


def once():
    r = build()
    write(STUDY/"VERIFIED_EVIDENCE.json", r)
    lines = ["# 训练质量校准与软聚合：固定专家池对照", "",
             f"四视图诊断完成：{int(r['pilot_completed'])}/1；完整确认：{r['full_completed']}/2。缺失为待测。", "",
             "| 范围 | 视图 | 生成题数 | 19任务AP（Answer F1） |", "|---|---|---:|---:|"]
    for a in r["aggregate"]:
        lines.append(f"| {a['scope']} | {a['view']} | {a['generated']} | {a['native19_AP']:.3f} |")
    lines += ["", "[全部领域、路由、融合强度与激活证据](VERIFIED_EVIDENCE.json)。原LoRA全部固定；只使用原池实际保留的TRAIN向量/QA做16参数质量校准，不补样、不使用测试答案，实际QA/前向数按证明记录。", ""]
    lines += ["- "+s for s in r["limits"]]
    (STUDY/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    write(STUDY/"observer_status.json", {"pid": os.getpid(), "pilot_completed": r["pilot_completed"], "full_completed": r["full_completed"], "CPU_only": True})
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--watch", action="store_true")
    args = p.parse_args()
    try:
        while True:
            r = once()
            if not args.watch or r["full_completed"] == 2:
                break
            time.sleep(45)
    except Exception:
        write(STUDY/"observer_needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
