"""New sealed formal-V2 cells: immutable collector, separate follow-up output.

The restarted Qwen ER run is pending, never spliced with the interrupted run.
No GPU, new training, inference, choices, prompts or metric changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import onevision_v2_language_snapshot as original

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"summary/onevision_diagnostics_20261016/baseline_language_followup_20261018"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(host, paths):
    result = original.build(paths, host)
    result["source_sha256"][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    result["scope"] = "Current six runs; Qwen ER fresh restart, interrupted run never concatenated"
    path = OUT/(host+"_snapshot.json")
    assert not path.exists(), "Preserve this dated snapshot; new follow-up requires a new output"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({"CPU_only": True, "host": host,
        "sealed_cells": sum(c["scores"] is not None for r in result["runs"] for c in r["cells"])}))


def merge():
    snapshots, hashes = {}, {}
    for host in ["old2080", "new3090"]:
        path = OUT/(host+"_snapshot.json")
        hashes[str(path.relative_to(ROOT))] = sha(path)
        snapshots[host] = json.loads(path.read_text())
        assert snapshots[host]["CPU_only"] and snapshots[host]["host_label"] == host
        assert snapshots[host]["Organ_TRAIN_QA"] == 26 and snapshots[host]["Organ_TEST_QA"] == 507
    a, b = [s["source_sha256"] for s in snapshots.values()]
    common = sorted(set(a)&set(b))
    assert common and all(a[p] == b[p] for p in common)
    assert "analysis/answer_f1.py" in common and "analysis/onevision_v2_language_snapshot.py" in common
    assert "exp/two_backbone_native_cl_20261017_v2/run.py" in common
    assert sha(ROOT/"analysis/onevision_v2_language_snapshot.py") == a["analysis/onevision_v2_language_snapshot.py"]
    runs = [r for s in snapshots.values() for r in s["runs"]]
    assert len(runs) == len({(r["backbone"], r["method"]) for r in runs}) == 6
    initial = {}
    for backbone, count in [("onevision", 4), ("qwen", 2)]:
        selected = [r for r in runs if r["backbone"] == backbone]
        assert len(selected) == count
        accepted = [next(c for c in r["cells"] if c["stage"] == 0) for r in selected]
        verified = [c for c in accepted if c["scores"] is not None]
        assert verified and len({c["semantic_fields_sha256"] for c in verified}) == 1
        initial[backbone] = {"verified_current_runs": len(verified), "expected_runs": count,
            "exact_initial_outputs_among_available": True, "missing_not_assumed_equal": count-len(verified)}
    accepted_cells = sum(c["scores"] is not None for r in runs for c in r["cells"])
    acquired = [{"backbone": r["backbone"], "method": r["method"], "transition": t}
                for r in runs for t in r["transitions"] if t["label"] == "current_Organ_acquisition"]
    retention = [{"backbone": r["backbone"], "method": r["method"], "transition": t}
                for r in runs for t in r["transitions"] if t["label"] == "post_acquisition_retention"]
    previous = json.loads((ROOT/"summary/onevision_diagnostics_20261016/baseline_language_profile/VERIFIED_EVIDENCE.json").read_text())
    old_cells = {(r["run"], c["stage"]) for r in previous["runs"] for c in r["cells"] if c["scores"] is not None}
    new_cells = [(r["run"], c["stage"]) for r in runs for c in r["cells"] if c["scores"] is not None and (r["run"], c["stage"]) not in old_cells]
    assert new_cells, "Do not repeat an unchanged snapshot as new progress"
    result = {"CPU_only": True, "new_GPU_operations": 0, "accepted_cells": accepted_cells,
        "newly_sealed_cells": new_cells, "runs": runs, "initial_exact_checks": initial,
        "current_acquisition": acquired, "retention": retention,
        "host_snapshot_sha256": hashes, "shared_source_and_protocol_paths_exact": common,
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        "limits": ["S3->S4 is an observed stage transition, not a randomized no-update intervention.",
                   "S4->S19 combines later medical and cross-domain tasks, not pure domain-count causality.",
                   "Language groups are author annotations, not verified translation pairs.",
                   "Single seed; image intervals condition on fits, not seed/order uncertainty.",
                   "Qwen vs OneVision changes architecture/pretraining as well as size; not knowledge/size isolation.",
                   "New Qwen ER pending; never splice the interrupted run or assume its scores.",
                   "Formal common401408 FP16 only; never fill nativeAnyRes research anchors with these cells.",
                   "ER loss contains replay; raw loss levels across methods not target-only fit comparisons.",
                   "Organ507 cell alone not full21454 or all-method ranking; full completion verified separately."]}
    (OUT/"VERIFIED_EVIDENCE.json").write_text(json.dumps(result, indent=2, ensure_ascii=True)+"\n")
    lines = ["# 新封存V2基线：习得与保持分开", "",
        "原只读核验器验证完整507题、原金标/元数据/Answer F1、每个前置阶段实际正LR/有限梯度/参数SHA改变及冻结源码；没有新GPU训练/推理。新Qwen ER从头运行，旧中断run不拼入。", "",
        f"本次接受{accepted_cells}个完整格，其中{len(new_cells)}个为新增；当前任务习得配对{len(acquired)}组、S4→S19保持配对{len(retention)}组。", "",
        "| 骨干／方法 | S0英文／中文 | S3学习前英文／中文 | S4刚学完英文／中文 | S19最终英文／中文 |",
        "|---|---:|---:|---:|---:|"]
    for r in runs:
        values = []
        for c in r["cells"]:
            g = c["scores"]
            values.append("待测" if g is None else f"{g['q_lang=en']['Answer_F1']:.3f}/{g['q_lang=zh']['Answer_F1']:.3f}")
        lines.append("| "+r["backbone"]+"／"+r["method"]+" | "+" | ".join(values)+" |")
    for title, values in [("S3→S4：当前任务习得", acquired), ("S4→S19：后续净变化", retention)]:
        lines += ["", "## "+title, "", "| 骨干／方法 | 英文变化及95%图像区间 | 中文变化及95%图像区间 |", "|---|---:|---:|"]
        for row in values:
            cells = []
            for lang in ["en", "zh"]:
                d = row["transition"]["groups"]["q_lang="+lang]["delta_F1"]
                lo, hi = d["image_cluster_95CI"]
                cells.append(f"{d['mean']:+.3f} [{lo:+.3f},{hi:+.3f}]")
            lines.append("| "+row["backbone"]+"／"+row["method"]+" | "+" | ".join(cells)+" |")
    lines += ["", "原TRAIN26题（中文15/英文11）；完整逐题增减数、全部语言/题型分层、实际loss/梯度/LR/阶段证明与SHA见[JSON](VERIFIED_EVIDENCE.json)。", ""]
    lines += ["- "+s for s in result["limits"]]
    (OUT/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"accepted_cells": accepted_cells, "new_cells": len(new_cells),
        "acquisition": [{"backbone": a["backbone"], "method": a["method"],
            "en": a["transition"]["groups"]["q_lang=en"]["delta_F1"],
            "zh": a["transition"]["groups"]["q_lang=zh"]["delta_F1"]} for a in acquired], "retention": retention}))


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
