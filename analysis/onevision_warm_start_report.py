"""Read-only raw-F1 report for sealed cold/warm initialization controls."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

from answer_f1 import answer_f1
import onevision_warm_start_queue as queue

ROOT, BASE, REPORT = queue.ROOT, queue.BASE, queue.STUDY
DOMAINS = ["natural", "medical", "geography", "finance", "traffic"]


def build():
    spec = queue.warm_plan()
    cold_spec = queue.previous.previous.previous.trial_plan()
    manifest_file = ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert queue.sha(manifest_file) == spec["manifest_sha256"]
    gold, domain_ids = {}, {d: [] for d in DOMAINS}
    for task in json.loads(manifest_file.read_text())["tasks"]:
        info = task["files"]["test"]
        p = ROOT / info["path"]
        assert queue.sha(p) == info["sha256"]
        for line in p.read_text().splitlines():
            row = json.loads(line)
            assert row["id"] not in gold
            gold[row["id"]] = (row["answers"], task["task"])
            domain_ids[task["domain"]].append(row["id"])
    assert len(gold) == 21454
    results, proofs, training, resources = [], {}, {}, {}
    for variant in ("learned_cosine_K3", queue.VARIANT):
        folder = BASE / "learned_reuse" if variant == "learned_cosine_K3" else REPORT
        for phase in ("train", "full"):
            if variant == queue.VARIANT:
                complete = queue.warm_complete(phase, spec)
            else:
                complete = queue.previous.previous.previous.complete(variant, phase, cold_spec)
            if not complete:
                continue
            proof_file = folder / f"{variant}_{phase}_complete.json"
            proof = json.loads(proof_file.read_text())
            proofs[str(proof_file.relative_to(ROOT))] = queue.sha(proof_file)
            run = ROOT / proof["run"]
            label = "final_diagnostic" if phase == "train" else "final_full_input_router"
            scores, ids_gold = {t: [] for t in range(1, 20)}, []
            domains, outside = {}, {"in_diagnostic500": [], "outside_diagnostic500": []}
            selected500 = {i for values in spec["diagnostic_ids"].values() for i in values}
            for domain in DOMAINS:
                rel = f"evaluations/{label}/{domain}_image.jsonl"
                p = run / rel
                assert queue.sha(p) == proof["artifact_sha256"][rel]
                rows = [json.loads(s) for s in p.read_text().splitlines()]
                assert [r["id"] for r in rows] == (spec["diagnostic_ids"][domain] if phase == "train" else domain_ids[domain])
                values = []
                for row in rows:
                    references, task = gold[row["id"]]
                    assert row["references"] == references and row["native_task"] == task
                    assert math.isfinite(row["nll"])
                    if "scores" not in row:
                        assert phase == "train"
                        continue
                    value = answer_f1(row["prediction"], references)
                    assert value == row["scores"]["token_f1"]
                    values.append(value)
                    scores[task].append(value)
                    ids_gold.append((row["id"], references))
                    outside["in_diagnostic500" if row["id"] in selected500 else "outside_diagnostic500"].append(value)
                if phase == "train":
                    assert [r["id"] for r in rows if "scores" in r] == [i for i in spec["diagnostic_ids"][domain] if i in set(spec["generation_ids"][domain])]
                else:
                    assert len(values) == len(rows)
                domains[domain] = sum(values) / len(values)
            assert len(ids_gold) == (155 if phase == "train" else 21454) and all(scores.values())
            results.append({"variant": variant, "phase": phase, "generated": len(ids_gold), "domains": domains,
                            "AP": sum(sum(v) / len(v) for v in scores.values()) / 19,
                            "FWT": None, "BWT": None,
                            "QA_weighted_diagnostic_partitions": {k: {"QA": len(v), "F1": sum(v)/len(v) if v else None} for k, v in outside.items()}})
            resources[f"{variant}/{phase}"] = json.loads((run / "memory_summary.json").read_text())
            if phase == "train":
                events = [json.loads(s) for s in (run / "events.jsonl").read_text().splitlines()]
                expected = sum(math.ceil(len(v) / 16) for v in spec["train_ids"].values())
                assert len(events) == expected
                assert all(math.isfinite(e["loss"]) and math.isfinite(e["gradient_l2_preclip"]) and
                           e["learning_rate_used"] > 0 for e in events)
                assert sum(e["qa_seen"] - (events[i-1]["qa_seen"] if i and events[i-1]["label"] == e["label"] else 0)
                           for i, e in enumerate(events)) == 1313
                training[variant] = {"actual_updates": len(events), "positive_LR_finite_gradients": True,
                                     "TRAIN_QA": 1313, "experts": proof["experts"],
                                     "runtime_initialization_gates": 19 if variant == queue.VARIANT else None,
                                     "runtime_geometry_gates": 19 if variant == queue.VARIANT else None}
    return {"CPU_only": True, "raw_Answer_F1": True, "formal_V2": False, "TEST_selection": False,
            "warm_train_complete": queue.VARIANT in training,
            "cold_full_complete": any(r["variant"] == "learned_cosine_K3" and r["phase"] == "full" for r in results),
            "warm_full_complete": any(r["variant"] == queue.VARIANT and r["phase"] == "full" for r in results),
            "results": results, "training": training, "resources": resources, "completion_sha256": proofs,
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): queue.sha(Path(__file__))},
            "limits": ["155 diagnostic generated QA are separate from complete21454; no fullFWT/BWT from final-only.",
                       "Only new-expert initialization changes; geometry/allocation must match cold runtime gates.",
                       "A result here does not establish initialization x unique-data-size interaction or knowledge sufficiency.",
                       "CPU/GPU resident, optimizer, temporary clones/history/probes/metric and compute remain charged.",
                       "Outside diagnostic500 is not claimed untouched by every earlier study."]}


def once():
    result = build()
    queue.write(REPORT / "VERIFIED_EVIDENCE.json", result)
    lines = ["# 初始化单变量控制：封存结果", "",
             f"新增温启动训练完成：{int(result['warm_train_complete'])}/1；温启动完整确认：{int(result['warm_full_complete'])}/1；原冷启动完整确认：{int(result['cold_full_complete'])}/1。缺失是待测。", "",
             "| 范围 | 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP | FWT | BWT |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["results"]:
        lines.append("| " + ("155诊断" if row["phase"] == "train" else "21454完整") + " | " + row["variant"]
                     + " | " + " | ".join(f"{row['domains'][d]:.3f}" for d in DOMAINS)
                     + f" | {row['AP']:.3f} | — | — |")
    lines += ["", "只接受完整SHA证明，逐题核对冻结原ID/标注并复算原始Answer F1。冷启动既有诊断不是新增温启动结果；未完成不能写成0分或收益。", "",
              "[协议与六栏](PROTOCOL.md) · [全部证明、资源与实际更新](VERIFIED_EVIDENCE.json)", ""]
    lines += ["- " + x for x in result["limits"]]
    (REPORT / "VERIFIED_RESULTS.md").write_text("\n".join(lines) + "\n")
    queue.write(REPORT / "observer_status.json", {"pid": os.getpid(), "CPU_only": True,
                **{k: result[k] for k in ("warm_train_complete", "warm_full_complete", "cold_full_complete")}})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    try:
        previous = None
        while True:
            keys = [BASE / "learned_reuse" / f"learned_cosine_K3_{p}_complete.json" for p in ("train", "full")]
            keys += [REPORT / f"{queue.VARIANT}_{p}_complete.json" for p in ("train", "full")]
            stamp = tuple(queue.sha(p) if p.exists() else None for p in keys)
            if stamp != previous:
                result = once()
                previous = stamp
                print(json.dumps({"CPU_only": True, "warm_train_complete": result["warm_train_complete"],
                                  "warm_full_complete": result["warm_full_complete"], "cold_full_complete": result["cold_full_complete"]}), flush=True)
            if not args.watch or result["warm_full_complete"] and result["cold_full_complete"]:
                break
            time.sleep(45)
    except Exception:
        queue.write(REPORT / "observer_needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
