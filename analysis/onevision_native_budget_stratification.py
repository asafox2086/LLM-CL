"""Descriptive budget stratification and algorithm-specific initial outputs.

No new fit/inference or score selection. Domain/task composition is reported
beside the global association; never infer update-count causality from it.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"summary/onevision_diagnostics_20261016/native_acquisition_20261018_v3"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def initial():
    snapshot_path = OUT/"old2080_snapshot.json"
    snap = json.loads(snapshot_path.read_text())
    runs = snap["runs"]
    assert len(runs) == 6 and all(r["backbone"] == "onevision" for r in runs)
    n = min(r["completed_prefix"] for r in runs)
    reference = next(r for r in runs if r["method"] == "SeqFT_LoRA")
    totals = []

    def rows(run, tid):
        relative = run["run"]+f"/evaluations/stage_00/T{tid:02d}_test_predictions.jsonl"
        p = ROOT/relative
        assert sha(p) == snap["source_sha256"][relative]
        return [json.loads(s) for s in p.read_text().splitlines()]

    cached = {t: rows(reference, t) for t in range(1, n+1)}
    for run in runs:
        items = []
        for tid in range(1, n+1):
            a, b = cached[tid], rows(run, tid)
            assert [x["id"] for x in a] == [x["id"] for x in b]
            assert [x["references"] for x in a] == [x["references"] for x in b]
            items.append({"task": tid, "QA": len(a),
                "raw_prediction_differences": sum(x["prediction"] != y["prediction"] for x, y in zip(a, b)),
                "F1_differences": sum(x["scores"]["token_f1"] != y["scores"]["token_f1"] for x, y in zip(a, b))})
        totals.append({"method": run["method"], "tasks": items,
            "raw_prediction_differences": sum(x["raw_prediction_differences"] for x in items),
            "F1_differences": sum(x["F1_differences"] for x in items)})
    result = {"CPU_only": True, "common_prefix": n, "reference": reference["run"],
        "old_snapshot_sha256": sha(snapshot_path), "comparisons": totals,
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        "limit": "Algorithm initialization may reparameterize the same checkpoint (OSFT SVD); a measured difference is not unique proof of its numerical cause. Never force shared S0 or transplant another method's baseline."}
    (OUT/"INITIAL_FUNCTION_CHECK.json").write_text(json.dumps(result, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({"initial_differences": [{k: x[k] for k in ["method", "raw_prediction_differences", "F1_differences"]} for x in totals]}))


def stratify():
    path = OUT/"EVIDENCE.json"
    evidence = json.loads(path.read_text())
    assert evidence["CPU_only"] and evidence["new_GPU_operations"] == 0
    for host, expected in evidence["snapshot_sha256"].items():
        assert sha(OUT/(host+"_snapshot.json")) == expected
    prefix = evidence["common_completed_prefix"]
    summaries = []
    for run in evidence["runs"]:
        if run["completed_prefix"] < 4:
            continue
        rows = run["observations"][:prefix]
        scopes = [("all", rows)]+[(domain, [r for r in rows if r["domain"] == domain]) for domain in dict.fromkeys(r["domain"] for r in rows)]
        groups = {}
        for label, selected in scopes:
            groups[label] = {}
            for key, data in [("one_update", [r for r in selected if r["updates"] == 1]),
                              ("multiple_updates", [r for r in selected if r["updates"] > 1])]:
                groups[label][key] = {"tasks": [r["task"] for r in data], "count": len(data),
                    "before_F1": sum(r["before"]["F1"] for r in data)/len(data) if data else None,
                    "after_F1": sum(r["after"]["F1"] for r in data)/len(data) if data else None,
                    "acquisition_delta": sum(r["acquisition_delta"] for r in data)/len(data) if data else None}
        summaries.append({"backbone": run["backbone"], "method": run["method"], "groups": groups})
    global_lower = [r["backbone"]+"/"+r["method"] for r in summaries if r["groups"]["all"]["one_update"]["acquisition_delta"] < r["groups"]["all"]["multiple_updates"]["acquisition_delta"]]
    medical_reverse = [r["backbone"]+"/"+r["method"] for r in summaries if r["groups"]["medical"]["one_update"]["acquisition_delta"] > r["groups"]["medical"]["multiple_updates"]["acquisition_delta"]]
    result = {"CPU_only": True, "common_prefix": prefix, "new_GPU_operations": 0,
        "budget_split": "Frozen effective batch16: one actual update vs more than one; no TEST chosen threshold",
        "native_evidence_sha256": sha(path), "rows": summaries,
        "global_one_update_lower": global_lower, "medical_order_reversed": medical_reverse,
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        "limits": ["Descriptive task grouping; train count/steps/content/domain/order/prior headroom confounded.",
                   "Medical stratification holds domain label only, not content, difficulty or update budget.",
                   "No causal proof that more steps help, no significance or strict Simpson theorem claim.",
                   "Common prefix13 lacks finance/traffic, not completed five-domain method performance."]}
    (OUT/"STRATIFIED_BUDGET.json").write_text(json.dumps(result, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({"global_lower": global_lower, "medical_reverse": medical_reverse, "rows": summaries}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--initial", action="store_true")
    args = p.parse_args()
    if args.initial:
        initial()
    else:
        stratify()
