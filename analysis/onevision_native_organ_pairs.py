"""All six current OneVision methods, same Organ507 acquisition pairs.

Author strata and image-cluster intervals only. No GPU or TEST-selected view.
"""
import hashlib
import json
from pathlib import Path
from answer_f1 import answer_f1
from onevision_v2_language_snapshot import paired

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"summary/onevision_diagnostics_20261016/native_acquisition_20261018_v3"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path = OUT/"old2080_snapshot.json"
    snap = json.loads(path.read_text())
    assert len(snap["runs"]) == 6
    manifest_path = ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert sha(manifest_path) == snap["source_sha256"][str(manifest_path.relative_to(ROOT))]
    task = json.loads(manifest_path.read_text())["tasks"][3]
    f = task["files"]["test"]
    assert sha(ROOT/f["path"]) == f["sha256"]
    gold = [json.loads(s) for s in (ROOT/f["path"]).read_text().splitlines()]
    assert len(gold) == 507
    groups = {"all": list(range(507))}
    for lang in ["en", "zh"]:
        groups[lang] = [i for i, r in enumerate(gold) if r["metadata"]["q_lang"] == lang]
    for typ in ["OPEN", "CLOSED"]:
        groups[typ] = [i for i, r in enumerate(gold) if r["metadata"]["answer_type"] == typ]
        for lang in ["en", "zh"]:
            groups[lang+"_"+typ] = [i for i in groups[typ] if gold[i]["metadata"]["q_lang"] == lang]
    results = []
    for run in snap["runs"]:
        assert run["completed_prefix"] >= 4
        records = []
        for stage in [3, 4]:
            relative = run["run"]+f"/evaluations/stage_{stage:02d}/T04_test_predictions.jsonl"
            assert sha(ROOT/relative) == snap["source_sha256"][relative]
            rows = [json.loads(s) for s in (ROOT/relative).read_text().splitlines()]
            assert [r["id"] for r in rows] == [r["id"] for r in gold]
            for a, b in zip(rows, gold):
                assert a["references"] == b["answers"]
                assert a["scores"]["token_f1"] == answer_f1(a["prediction"], a["references"])
            records.append(rows)
        before, after = records
        output = {}
        for label, indices in groups.items():
            deltas = [after[i]["scores"]["token_f1"]-before[i]["scores"]["token_f1"] for i in indices]
            output[label] = {"QA": len(indices),
                "before_F1": sum(before[i]["scores"]["token_f1"] for i in indices)/len(indices),
                "after_F1": sum(after[i]["scores"]["token_f1"] for i in indices)/len(indices),
                "delta_F1": paired(deltas, [gold[i]["images"] for i in indices]),
                "increased": sum(d > 0 for d in deltas), "decreased": sum(d < 0 for d in deltas),
                "unchanged": sum(d == 0 for d in deltas)}
        results.append({"method": run["method"], "groups": output})
    result = {"CPU_only": True, "new_GPU_operations": 0, "methods": results,
        "old_snapshot_sha256": sha(path), "TEST_selected_methods_or_thresholds": False,
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
            "analysis/onevision_v2_language_snapshot.py": sha(ROOT/"analysis/onevision_v2_language_snapshot.py"),
            "analysis/answer_f1.py": sha(ROOT/"analysis/answer_f1.py")},
        "limits": ["All six methods and all nine author language/type groups reported, not selected favorable results.",
                   "Conditional uncorrected image intervals, single seed/order, not randomized cause isolation.",
                   "Current Organ transition before later domains, not final retention or proof of a unique training mechanism.",
                   "Different authored capacities/LR/auxiliary passes remain; near-zero mean gain not zero function update."]}
    (OUT/"ORGAN_PAIRS.json").write_text(json.dumps(result, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({r["method"]: {k: r["groups"][k] for k in ["all", "OPEN", "CLOSED"]} for r in results}))


if __name__ == "__main__":
    main()
