"""Read-only full-native-task F1/interaction report for balanced TRAIN support."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import json
import math
from pathlib import Path
import time
import traceback

from answer_f1 import answer_f1
from onevision_acquisition_budget_bridge import interval
import onevision_cold_data_queue as queue

ROOT, REPORT = queue.ROOT, queue.STUDY


def build():
    spec = queue.study_plan()
    manifest_path = ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert queue.sha(manifest_path) == spec["manifest_sha256"]
    originals = {}
    for task in json.loads(manifest_path.read_text())["tasks"]:
        if task["task"] not in spec["targets"]:
            continue
        info = task["files"]["test"]
        path = ROOT / info["path"]
        assert queue.sha(path) == info["sha256"]
        for line in path.read_text().splitlines():
            row = json.loads(line)
            originals[row["id"]] = row
    results, proofs, predictions, resources, initial = [], {}, {}, {}, {}
    for case in spec["cases"]:
        index = case["index"]
        if not queue.study_complete(index, spec):
            continue
        proof_path = REPORT / f"case_{index:02d}_complete.json"
        proof = json.loads(proof_path.read_text())
        proofs[proof_path.name] = queue.sha(proof_path)
        run = ROOT / proof["run"]
        gate = json.loads((run / "initial_state_gate.json").read_text())
        assert gate["actual_AB_tensor_sha256"] == gate["source_AB_tensor_sha256"]
        row_result = {"case_index": index, "task": case["task"], "start": case["start"],
                      "unique_TRAIN": case["unique_TRAIN"], "effective_batch": case["effective_batch"],
                      "TEST_QA": len(case["TEST_ids"]), "before_after": {}}
        for label in ("initial", "endpoint_1", "endpoint_4"):
            rows = [json.loads(s) for s in (run / "evaluations" / label / "native_image.jsonl").read_text().splitlines()]
            assert [r["id"] for r in rows] == case["TEST_ids"]
            for row in rows:
                assert row["references"] == originals[row["id"]]["answers"]
                assert row["native_task"] == case["task"] and math.isfinite(row["nll"])
                assert row["scores"]["token_f1"] == answer_f1(row["prediction"], row["references"])
            row_result["before_after"][label] = {"Answer_F1": sum(r["scores"]["token_f1"] for r in rows)/len(rows),
                                                  "NLL": sum(r["nll"] for r in rows)/len(rows)}
            predictions[case["task"], case["unique_TRAIN"], case["start"], label] = rows
            if label == "initial":
                fields = ("id", "native_task", "prediction", "references", "scores", "nll", "prompt_tokens", "target_tokens", "target_truncated", "visual_tokens")
                canonical = [{k: r[k] for k in fields if k in r} for r in rows]
                key = (case["task"], case["start"])
                value = (gate["actual_AB_tensor_sha256"], canonical)
                if key in initial:
                    assert initial[key] == value, "Same-init support conditions have unequal initial function"
                initial[key] = value
        events = json.loads((run / "training.json").read_text())["records"]
        assert len(events) == 4 and [e["update"] for e in events] == [1, 2, 3, 4]
        assert all(e["LR"] == 1e-4 and math.isfinite(e["loss"]) and math.isfinite(e["gradient_L2_preclip"])
                   and e["gradient_L2_preclip"] > 0 and e["cumulative_exposures"] == e["update"] * case["effective_batch"] for e in events)
        assert events[-1]["A_delta_L2"] + events[-1]["B_delta_L2"] > 0
        if case["start"] == "cold_base":
            assert events[0]["A_delta_L2"] == 0 and events[0]["B_delta_L2"] > 0
        row_result["actual_training"] = {"updates": 4, "QA_exposures": events[-1]["cumulative_exposures"],
                                          "constant_positive_LR_finite_gradients": True, "records": events}
        window = json.loads((run / "TRAIN_window.json").read_text())
        monitor = [json.loads(s) for s in (run / "resources.jsonl").read_text().splitlines()]
        monitor = [v for v in monitor if v["pid"] == window["pid"] and window["before"]["unix"] <= v["unix"] <= window["after"]["unix"]]
        keys = ["VmRSS_bytes", "Pss_bytes", "cpu_uss_bytes", "gpu_allocated_bytes", "gpu_reserved_bytes", "gpu_process_nvml_bytes"]
        resources[str(index)] = {"TRAIN_window": window,
            "pure_TRAIN_periodic_sample_peaks": {k: max((v[k] for v in monitor if v.get(k) is not None), default=None) for k in keys},
            "sampled_CPU_peak_is_not_exact_instantaneous_peak": True,
            "memory_summary": json.loads((run / "memory_summary.json").read_text())}
        results.append(row_result)
    gaps, interactions = [], []
    for task in spec["targets"]:
        sizes = sorted({c["unique_TRAIN"] for c in spec["cases"] if c["task"] == task})
        for n in sizes:
            for label in ("initial", "endpoint_1", "endpoint_4"):
                keys = [(task, n, kind, label) for kind in spec["starts"]]
                if not all(k in predictions for k in keys):
                    continue
                cold, warm = (predictions[k] for k in keys)
                delta = [b["scores"]["token_f1"]-a["scores"]["token_f1"] for a, b in zip(cold, warm)]
                gaps.append({"task": task, "unique_TRAIN": n, "label": label, "contrast": "warm_minus_cold",
                             **interval(delta, [originals[r["id"]]["images"] for r in cold])})
        for n in sizes[:-1]:
            for label in ("endpoint_1", "endpoint_4"):
                keys = [(task, m, kind, label) for m in (n, sizes[-1]) for kind in spec["starts"]]
                if not all(k in predictions for k in keys):
                    continue
                few_c, few_w, all_c, all_w = (predictions[k] for k in keys)
                delta = [(b["scores"]["token_f1"]-a["scores"]["token_f1"]) - (d["scores"]["token_f1"]-c["scores"]["token_f1"])
                         for a, b, c, d in zip(few_c, few_w, all_c, all_w)]
                interactions.append({"task": task, "few_unique": n, "all_unique": sizes[-1], "label": label,
                                     "contrast": "(warm-cold)_few_minus_(warm-cold)_all",
                                     **interval(delta, [originals[r["id"]]["images"] for r in few_c])})
    return {"CPU_only": True, "completed_trajectories": len(results), "required_trajectories": 18,
            "results": results, "warm_minus_cold": gaps, "support_interactions": interactions,
            "resources": resources, "completion_sha256": proofs,
            "initial_function_comparisons_checked": len(results) - len(initial),
            "all18_initial_functions_exact_across_support": True if len(results) == 18 else None,
            "source_sha256": {str(Path(__file__).relative_to(ROOT)): queue.sha(Path(__file__))},
            "scope": "full native228/267/100 tasks; NOT full21454 or19stage CL or formalV2",
            "limits": ["Single nested TRAIN hash selection, seed42; support amount changes content/answer coverage too.",
                       "Equal exposure/effective batch within anchor, not equal tokens or equal compute across unique supports/tasks.",
                       "Warm uses original nativeT01/history347TRAIN; not the older basis-budget warm control.",
                       "Known3 anchors, no TEST-based case or budget selection; all18/both endpoints retained.",
                       "Intervals are image-cluster/uncorrected and fixed-weight, not multiple-training-seed evidence.",
                       "No fullAP/FWT/BWT or universal knowledge sufficiency from these isolated native tasks."]}


def once():
    result = build()
    queue.write(REPORT / "VERIFIED_EVIDENCE.json", result)
    lines = ["# 独特TRAIN量 × 初始化 × 更新数", "",
             f"有效轨迹完成{result['completed_trajectories']}/18，每条均报告初始、1步、4步；缺失为待测。", "",
             "| 原生任务 | 初始化 | 独特TRAIN | 每步曝光 | 原TEST | 初始F1 | 1步F1 | 4步F1 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["results"]:
        lines.append(f"| T{row['task']:02d} | {row['start']} | {row['unique_TRAIN']} | {row['effective_batch']} | {row['TEST_QA']} | "
                     + " | ".join(f"{row['before_after'][k]['Answer_F1']:.3f}" for k in ("initial", "endpoint_1", "endpoint_4")) + " |")
    lines += ["", "全部原native任务测试，不是完整五域主表。逐题原ID/金标/Answer F1复算、实际更新/原始梯度摘要和内存见[证据](VERIFIED_EVIDENCE.json)。", "",
              "预定交互为：(温−冷)少样本 − (温−冷)全原TRAIN，正值表示该对照下小支持集的继承优势更大。所有方向、任务和区间均保留，不按结果选方法或超参；未齐格不能解释为0效应。", "",
              "[协议与六栏](PROTOCOL.md)", ""]
    lines += ["- " + s for s in result["limits"]]
    (REPORT / "VERIFIED_RESULTS.md").write_text("\n".join(lines) + "\n")
    queue.write(REPORT / "observer_status.json", {"pid": os.getpid(), "CPU_only": True,
                "completed_trajectories": result["completed_trajectories"], "required_trajectories": 18})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    try:
        previous = None
        while True:
            paths = [REPORT / f"case_{i:02d}_complete.json" for i in range(18)]
            stamp = tuple(queue.sha(p) if p.exists() else None for p in paths)
            if stamp != previous:
                result = once()
                previous = stamp
                print(json.dumps({"CPU_only": True, "completed_trajectories": result["completed_trajectories"]}), flush=True)
            if not args.watch or result["completed_trajectories"] == 18:
                break
            time.sleep(45)
    except Exception:
        queue.write(REPORT / "observer_needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
