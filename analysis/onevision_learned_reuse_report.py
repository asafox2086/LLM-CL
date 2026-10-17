"""Pure CPU read-only evidence observer; no ranking incomplete/full-mixed cells."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time
import traceback

from answer_f1 import answer_f1
from onevision_learned_reuse_queue import ROOT, REPORT as QUEUE, TRIAL, VARIANTS, trial_plan, complete, sha, write


def build():
    spec = trial_plan()
    manifest_path = ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert sha(manifest_path) == spec["manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    test, by_domain = {}, defaultdict(list)
    for task in manifest["tasks"]:
        info = task["files"]["test"]
        path = ROOT/info["path"]
        assert sha(path) == info["sha256"]
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        for r in rows:
            assert r["id"] not in test
            test[r["id"]] = dict(r, native_task=task["task"], domain=task["domain"])
            by_domain[task["domain"]].append(r["id"])
    assert len(test) == 21454
    cells, mechanics, proofs, raw, aggregate = [], [], {}, {}, []
    for variant in VARIANTS:
        if complete(variant, "train", spec):
            path = TRIAL/f"{variant}_train_complete.json"
            proof = json.loads(path.read_text())
            proofs[path.name] = sha(path)
            run = ROOT/proof["run"]
            ds = [json.loads((run/"decisions"/f"reuse_T{t:02d}.json").read_text()) for t in range(1, 20)]
            assert [d["stage"] for d in ds] == list(range(1, 20))
            assert all(d["train_ids"] == spec["train_ids"][str(d["stage"])] for d in ds)
            events = [json.loads(s) for s in (run/"events.jsonl").read_text().splitlines()]
            assert all(e["learning_rate_used"] > 0 and math.isfinite(e["gradient_l2_preclip"]) for e in events)
            mechanics.append({"variant": variant, "stages": 19, "experts_final": ds[-1]["experts"],
                "new_stages": [d["stage"] for d in ds if d["action"] == "new_from_base"],
                "cap_forced_reuse_stages": [d["stage"] for d in ds if d.get("cap_forced_reuse")],
                "metric_changed_stages": [d["stage"] for d in ds if d["metric_fit"]["parameter_changed"]],
                "metric_updates": sum(d["metric_fit"]["optimizer_updates"] for d in ds),
                "optimizer_qa": sum(d["current_optimizer_qa"] for d in ds),
                "extra_feature_forwards": sum(d["extra_feature_forwards"] for d in ds),
                "language_loss_forwards_TRAIN": sum(d["language_loss_forwards"] for d in ds),
                "decisions": ds, "memory_summary": json.loads((run/"memory_summary.json").read_text()),
                "TRAIN_windows": [json.loads((run/"TRAIN_windows"/f"reuse_T{t:02d}.json").read_text()) for t in range(1, 20)]})
        for phase in ["train", "full"]:
            if not complete(variant, phase, spec):
                continue
            path = TRIAL/f"{variant}_{phase}_complete.json"
            p = json.loads(path.read_text())
            proofs[path.name] = sha(path)
            run = ROOT/p["run"]
            label = "final_diagnostic" if phase == "train" else "final_full_input_router"
            for domain in by_domain:
                expected = spec["diagnostic_ids"][domain] if phase == "train" else by_domain[domain]
                rows = [json.loads(s) for s in (run/f"evaluations/{label}/{domain}_image.jsonl").read_text().splitlines()]
                assert [r["id"] for r in rows] == expected
                for r in rows:
                    assert r["references"] == test[r["id"]]["answers"] and math.isfinite(r["nll"])
                    if "prediction" in r:
                        assert r["scores"]["token_f1"] == answer_f1(r["prediction"], r["references"])
                    elif phase == "full":
                        raise AssertionError("Incomplete full generation")
                generated = [r for r in rows if "prediction" in r]
                route = Counter(str(r["route_expert"]) for r in rows)
                cells.append({"variant": variant, "phase": phase, "domain": domain, "NLL_QA": len(rows),
                    "generated_QA": len(generated), "Answer_F1": sum(r["scores"]["token_f1"] for r in generated)/len(generated),
                    "mean_NLL": sum(r["nll"] for r in rows)/len(rows), "route_counts": dict(route)})
                raw[variant, phase, domain] = {r["id"]: r for r in generated}
            generated_all = [r for domain in by_domain for r in raw[variant, phase, domain].values()]
            task_values = {str(t["task"]): [r["scores"]["token_f1"] for r in generated_all if r["native_task"] == t["task"]]
                           for t in manifest["tasks"]}
            assert all(task_values.values()), "Every native task needs generation before task AP"
            task_means = {t: sum(v)/len(v) for t, v in task_values.items()}
            aggregate.append({"variant": variant, "phase": phase, "generated_QA": len(generated_all),
                "native_task_F1": task_means, "native19_AP": sum(task_means.values())/19,
                "domain_macro_F1": sum(c["Answer_F1"] for c in cells if c["variant"] == variant and c["phase"] == phase)/5,
                "QA_weighted_F1": sum(r["scores"]["token_f1"] for r in generated_all)/len(generated_all),
                "full_FWT": None, "full_BWT": None, "full_transfer_status": "missing sparse cells; final-only not sufficient"})
    comparisons = []
    for phase in ["train", "full"]:
        for domain in by_domain:
            keys = [(v, phase, domain) for v in VARIANTS]
            if not all(k in raw for k in keys):
                continue
            a, b = (raw[k] for k in keys)
            assert a.keys() == b.keys()
            deltas = [b[i]["scores"]["token_f1"]-a[i]["scores"]["token_f1"] for i in a]
            comparisons.append({"phase": phase, "domain": domain, "QA": len(a),
                "learned_minus_fixed_Answer_F1": sum(deltas)/len(deltas),
                "improved": sum(d > 0 for d in deltas), "worsened": sum(d < 0 for d in deltas),
                "unchanged": sum(d == 0 for d in deltas),
                "different_routes": sum(a[i]["route_expert"] != b[i]["route_expert"] for i in a)})
    return {"CPU_only": True, "completion_sha256": proofs, "cells": cells, "aggregate": aggregate, "mechanics": mechanics,
            "comparisons": comparisons, "expected_train_runs": 2, "expected_full_runs": 2,
            "completed_train_runs": sum(c["variant"] in VARIANTS for c in mechanics),
            "completed_full_runs": sum(f"{v}_full_complete.json" in proofs for v in VARIANTS),
            "limitations": ["Assigned-expert labels and cosine novelty are not proof of task mastery.",
                "One expert produces no discriminative metric update; report actual gradients/changes.",
                "Kmax3 is a fixed budget, never a proved optimal expert count.",
                "Diagnostic500 NLL /155 generated are not full21454; report separate cells.",
                "Final-only evaluation cannot establish full FWT/BWT or within-expert forgetting.",
                "No domain/answer/TEST-based allocation; all valid variants get full confirmation.",
                "nativeAnyRes research is separate from formal common-image V2.",
                "No memory superiority without matched measurement including router/history/temporary/extra compute."],
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)), "analysis/answer_f1.py": sha(ROOT/"analysis/answer_f1.py")}}


def once():
    r = build()
    write(TRIAL/"VERIFIED_EVIDENCE.json", r)
    lines = ["# 可学习向量空间与LoRA优先复用", "",
             f"完整训练证明：{r['completed_train_runs']}/2；完整21454题确认：{r['completed_full_runs']}/2。待测不记零分。", "",
             "| 版本 | 范围 | 领域 | 生成题数 | Answer F1 |", "|---|---|---|---:|---:|"]
    for c in r["cells"]:
        lines.append(f"| {c['variant']} | {c['phase']} | {c['domain']} | {c['generated_QA']} | {c['Answer_F1']:.3f} |")
    lines += ["", "| 版本 | 最终LoRA数 | 新增阶段 | 投影实际改变阶段 |", "|---|---:|---|---|"]
    for m in r["mechanics"]:
        lines.append(f"| {m['variant']} | {m['experts_final']} | {m['new_stages']} | {m['metric_changed_stages']} |")
    lines += ["", "完整分配决策、训练NLL前后、实际LR/梯度、逐题得失、内存窗口与资源见[JSON](VERIFIED_EVIDENCE.json)。尚无GPU完成证明时，不宣称路由有效或性能提升。", ""]
    lines += ["- "+s for s in r["limitations"]]
    (TRIAL/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    write(TRIAL/"observer_status.json", {"pid": os.getpid(), "CPU_only": True,
         "completed_train_runs": r["completed_train_runs"], "completed_full_runs": r["completed_full_runs"]})
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            result = once()
            if not args.watch or result["completed_full_runs"] == 2:
                break
            time.sleep(45)
    except Exception:
        write(TRIAL/"observer_needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
