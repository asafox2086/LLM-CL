"""CPU audit of every completed native-task acquisition, not final AP ranking.

Comparisons use the common completed prefix among trained current runs. The
one-update boundary is the frozen effective batch16, never TEST-selected.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT/"exp/two_backbone_native_cl_20261017_v2"
OUT = ROOT/"summary/onevision_diagnostics_20261016/native_acquisition_20261018_v2"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot(host, paths):
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}

    def checked(p, expected=None):
        value = sha(p)
        assert expected is None or value == expected, str(p)
        sources[str(p.relative_to(ROOT))] = value
        return p

    lock = json.loads(checked(ROOT/"summary/primary_metric_20261017/metric_lock.json").read_text())
    checked(ROOT/"analysis/answer_f1.py", lock["metric_sha256"])
    assert lock["option_mapping"] is False and lock["additional_prefix_or_answer_extraction"] is False
    mp = ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    manifest = json.loads(checked(mp).read_text())
    gold, train = {}, {}
    for task in manifest["tasks"]:
        for split, target in [("test", gold), ("train", train)]:
            f = task["files"][split]
            target[task["task"]] = [json.loads(s) for s in checked(ROOT/f["path"], f["sha256"]).read_text().splitlines()]
            assert len(target[task["task"]]) == f["count"]
    assert sum(map(len, train.values())) == 1313 and sum(map(len, gold.values())) == 21454
    runs, event_prefixes = [], []
    for relative in paths:
        run = (ROOT/relative).resolve()
        assert run.is_relative_to(FRAME/"runs") and not (run/"gpu_preflight.json").exists()
        config = json.loads(checked(run/"config.json").read_text())
        assert config["manifest_sha256"] == sha(mp) and config["epochs"] == 1 and config["seed"] == 42
        assert config["condition"] == "pct01" and config["image_area_cap"] == 401408
        for name, expected in config["execution_source_sha256"].items():
            checked(FRAME/name, expected)
            checked(run/"runner_source"/name, expected)
        log = run/"logs/events.jsonl"
        raw = log.read_bytes() if log.exists() else b""
        raw = raw[:raw.rfind(b"\n")+1]
        event_prefixes.append({"path": str(log.relative_to(ROOT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
        events = [json.loads(s) for s in raw.splitlines()]
        completed = 0
        updates_by_stage = {}
        for stage in range(1, 20):
            path = run/f"stages/stage_{stage:02d}.json"
            if not path.exists():
                break
            proof = json.loads(checked(path).read_text())
            valid = json.loads(checked(run/f"validity/stage_{stage:02d}.json").read_text())
            assert proof["complete"] and proof["stage"] == stage and proof["train_qa"] == len(train[stage])
            assert valid["passed"] and valid["actual_parameters_changed"]
            assert valid["parameters_before_sha256"] != valid["parameters_after_sha256"]
            expected = math.ceil(len(train[stage])/16)
            ee = [e for e in events if e.get("event") == "train_update" and e["stage"] == stage]
            assert len(ee) == valid["updates"] == valid["positive_lr_updates"] == expected
            assert ee[-1]["qa_seen"] == len(train[stage])
            assert all(math.isfinite(e["loss"]) and math.isfinite(e["gradient_l2_preclip"]) and e["gradient_l2_preclip"] > 0 for e in ee)
            assert all(any(lr > 0 for lr in e["learning_rate_used"]) for e in ee)
            updates_by_stage[stage] = [{k: e[k] for k in ["step", "qa_seen", "loss", "gradient_l2_preclip", "learning_rate_used"]} for e in ee]
            completed = stage

        def cell(stage, tid):
            path = run/f"evaluations/stage_{stage:02d}/T{tid:02d}_test_predictions.jsonl"
            rows = [json.loads(s) for s in checked(path).read_text().splitlines()]
            summary = json.loads(checked(path.with_suffix(".scores.json")).read_text())
            assert len(rows) == summary["count"] == len(gold[tid])
            for row, original in zip(rows, gold[tid]):
                assert row["id"] == original["id"] and row["references"] == original["answers"]
                assert row["scores"]["token_f1"] == answer_f1(row["prediction"], row["references"])
            mean = sum(r["scores"]["token_f1"] for r in rows)/len(rows)
            assert abs(mean-summary["token_f1"]) < 1e-10
            strata = {}
            for kind in ["OPEN", "CLOSED"]:
                selected = [r for r, o in zip(rows, gold[tid]) if o.get("metadata", {}).get("answer_type") == kind]
                if selected:
                    strata[kind] = {"QA": len(selected), "F1": sum(r["scores"]["token_f1"] for r in selected)/len(selected)}
            return {"F1": mean, "strata": strata, "semantic_initial": hashlib.sha256(json.dumps([[r[k] for k in ["id", "prediction", "references", "scores"]] for r in rows], ensure_ascii=False, sort_keys=True).encode()).hexdigest()}

        observations = []
        for task in manifest["tasks"][:completed]:
            tid = task["task"]
            before, after = cell(tid-1, tid), cell(tid, tid)
            initial = cell(0, tid) if tid > 1 else before
            final = cell(19, tid) if completed == 19 else None
            observations.append({"task": tid, "native_label": task["native_label"], "domain": task["domain"],
                "TRAIN_QA": len(train[tid]), "TEST_QA": len(gold[tid]), "updates": len(updates_by_stage[tid]),
                "initial": initial, "before": before, "after": after, "final": final,
                "acquisition_delta": after["F1"]-before["F1"],
                "retention_delta": final["F1"]-after["F1"] if final is not None else None,
                "actual_updates": updates_by_stage[tid]})
        resource = run/"resources.jsonl"
        resource_errors = []
        if resource.exists():
            raw_resource = resource.read_bytes()
            raw_resource = raw_resource[:raw_resource.rfind(b"\n")+1]
            rr, offset = [], 0
            for line_number, line in enumerate(raw_resource.splitlines(keepends=True), 1):
                try:
                    rr.append(json.loads(line))
                except json.JSONDecodeError as error:
                    resource_errors.append({"line": line_number, "offset": offset,
                        "length": len(line), "sha256": hashlib.sha256(line).hexdigest(),
                        "text": line.decode("utf-8", errors="replace"), "error": str(error)})
                offset += len(line)
            event_prefixes.append({"path": str(resource.relative_to(ROOT)), "bytes": len(raw_resource),
                                   "sha256": hashlib.sha256(raw_resource).hexdigest()})
            # A corrupt resource log is not repaired or selectively used for
            # memory claims. Independent training/prediction gates stay exact.
            parameter_rows = [] if resource_errors else [x for x in rr if "trainable_parameters" in x and x.get("stage") in [1, 4]]
        else:
            parameter_rows = []
        runs.append({"run": relative, "backbone": config["backbone"], "method": config["method"],
            "completed_prefix": completed, "observations": observations,
            "resource_comparison_excluded": bool(resource_errors), "resource_log_errors": resource_errors,
            "stage1_or4_parameter_observations": [{k: x[k] for k in ["phase", "stage", "pid", "trainable_parameters", "trainable_parameter_bytes"]} for x in parameter_rows]})
    for p, h in sources.items():
        assert sha(ROOT/p) == h, "Sealed source changed"
    for p in event_prefixes:
        assert hashlib.sha256((ROOT/p["path"]).read_bytes()[:p["bytes"]]).hexdigest() == p["sha256"]
    result = {"CPU_only": True, "host": host, "runs": runs, "source_sha256": sources,
              "event_prefixes": event_prefixes, "new_GPU_operations": 0}
    target = OUT/(host+"_snapshot.json")
    assert not target.exists(), "Dated snapshot immutable; a future extension needs a new output"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({"host": host, "completed_prefixes": {r["method"]: r["completed_prefix"] for r in runs}}))


def merge():
    hosts = [json.loads((OUT/(h+"_snapshot.json")).read_text()) for h in ["old2080", "new3090"]]
    a, b = [h["source_sha256"] for h in hosts]
    common = sorted(set(a)&set(b))
    assert all(a[k] == b[k] for k in common) and "analysis/answer_f1.py" in common
    runs = [r for h in hosts for r in h["runs"]]
    assert len(runs) == len({(r["backbone"], r["method"]) for r in runs}) == 8
    # Organ4 is the predeclared anchor already required by v1's prefix>=4
    # gate; partial new-run startup must not destroy its common-task cohort.
    eligible = [r for r in runs if r["completed_prefix"] >= 4]
    prefix = min(r["completed_prefix"] for r in eligible)
    assert prefix >= 4
    reference = {row["task"]: row for row in eligible[0]["observations"][:prefix]}
    summaries = []
    for r in eligible:
        rows = r["observations"][:prefix]
        assert len(rows) == prefix and [x["task"] for x in rows] == list(range(1, prefix+1))
        for row in rows:
            ref = reference[row["task"]]
            assert (row["TRAIN_QA"], row["TEST_QA"], row["updates"]) == (ref["TRAIN_QA"], ref["TEST_QA"], ref["updates"])
        by_budget = {}
        for group, selected in [("one_update", [x for x in rows if x["updates"] == 1]),
                                ("multiple_updates", [x for x in rows if x["updates"] > 1])]:
            assert selected
            by_budget[group] = {"tasks": [x["task"] for x in selected],
                "mean_acquisition_delta": sum(x["acquisition_delta"] for x in selected)/len(selected),
                "negative_task_deltas": [x["task"] for x in selected if x["acquisition_delta"] < 0]}
        summaries.append({"backbone": r["backbone"], "method": r["method"], "common_prefix": prefix,
            "groups": by_budget, "overall_acquisition_delta": sum(x["acquisition_delta"] for x in rows)/prefix,
            "Organ": next(x for x in rows if x["task"] == 4)})
    result = {"CPU_only": True, "new_GPU_operations": 0, "common_completed_prefix": prefix,
        "eligibility": "At least four completed positive-update stages to reach predeclared Organ4 anchor; common minimum prefix, determined only by completion, not TEST score",
        "pending_before_Organ4_runs": [r["run"] for r in runs if r["completed_prefix"] < 4],
        "runs": runs, "summaries": summaries, "shared_source_paths_exact": common,
        "snapshot_sha256": {h: sha(OUT/(h+"_snapshot.json")) for h in ["old2080", "new3090"]},
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        "limits": ["Delta acquisition is a diagnostic, not final AP/FWT/BWT or a new algorithm.",
                   "One-update boundary is fixed batch16; tasks also differ in content/difficulty/domain. No data-size/step causal claim.",
                   "Methods retain authored capacities/LR/auxiliaries; these are not matched-capacity mechanism interventions.",
                   "Common prefix may omit finance/traffic; do not call it a completed five-domain comparison.",
                   "Pending fresh Qwen ER excluded from comparison only because no training complete; no old-run splice.",
                   "Formal common401408 FP16 is separate from nativeAnyRes research; no TEST tuning."]}
    result["limits"].append("Malformed resource records preserved; affected run excluded from resource comparison, never repaired or spliced across PIDs. Primary score/update gates unchanged.")
    (OUT/"EVIDENCE.json").write_text(json.dumps(result, indent=2, ensure_ascii=True)+"\n")
    lines = ["# 原生任务习得：共同完成前缀", "",
        f"8个当前run中{len(eligible)}个已完成预定Organ4锚，共同比较T01—T{prefix:02d}；其余未到锚点的run待测。只读原输出、完整金标/F1及真实更新证明。", "",
        "学习增量=刚学完当前任务F1−学习前F1，不是最终AP。一次／多次更新分组由固定batch16决定，不按分数选阈值。", "",
        "| 骨干／方法 | 一次更新任务平均学习增量 | 多次更新任务平均学习增量 |", "|---|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['backbone']}／{r['method']} | {r['groups']['one_update']['mean_acquisition_delta']:+.3f} | {r['groups']['multiple_updates']['mean_acquisition_delta']:+.3f} |")
    lines += ["", "## 共同任务的逐项学习增量", "", "| 任务／TRAIN／更新 | "+" | ".join(r["backbone"]+"／"+r["method"] for r in eligible)+" |", "|---|"+"---:|"*len(eligible)]
    for tid in range(1, prefix+1):
        ref = reference[tid]
        values = [f"{r['observations'][tid-1]['acquisition_delta']:+.3f}" for r in eligible]
        lines.append(f"| T{tid:02d} {ref['native_label']}／{ref['TRAIN_QA']}／{ref['updates']} | "+" | ".join(values)+" |")
    lines += ["", "完整初始/学习前/刚学完/最终、作者题型、实际loss/梯度/LR和SHA见[EVIDENCE](EVIDENCE.json)。主表不填本诊断分数。", ""]
    lines += ["- "+s for s in result["limits"]]
    (OUT/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"prefix": prefix, "eligible": len(eligible), "summaries": summaries}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=["old2080", "new3090"])
    parser.add_argument("--run", action="append")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge()
    else:
        assert args.host and args.run
        snapshot(args.host, args.run)
