"""CPU-only recovery of actual TRAIN resource windows from completed pilots.

Does not edit/import active GPU workers or parse test predictions/scores.
Original mixed-process maxima remain intact for provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016"
PILOTS = BASE / "two_module_prototype"
REPORT = BASE / "two_module_resource_windows"
KEYS = ("VmRSS_bytes", "Pss_bytes", "cpu_uss_bytes", "gpu_allocated_bytes",
        "gpu_reserved_bytes", "gpu_process_nvml_bytes")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def window(rows, before, after):
    assert before["pid"] == after["pid"] and before["unix"] < after["unix"]
    assert before in rows and after in rows
    # The frozen worker resets the CUDA peak immediately before this snapshot.
    # Refuse the peak claim if the saved endpoint contradicts that boundary.
    assert before["gpu_peak_allocated_bytes"] == before["gpu_allocated_bytes"]
    assert before["gpu_peak_reserved_bytes"] == before["gpu_reserved_bytes"]
    selected = [row for row in rows if row["pid"] == before["pid"]
                and before["unix"] <= row["unix"] <= after["unix"]]
    assert selected
    peaks = {k: max((row[k] for row in selected if row.get(k) is not None), default=None)
             for k in KEYS}
    growth = {k: after[k] - before[k] if before.get(k) is not None and after.get(k) is not None
              else None for k in KEYS}
    assert after["gpu_peak_allocated_bytes"] >= max(before["gpu_allocated_bytes"],
                                                    after["gpu_allocated_bytes"])
    return {"same_PID": before["pid"], "from_unix": before["unix"], "to_unix": after["unix"],
            "seconds": after["unix"] - before["unix"], "samples": len(selected),
            "before": before, "after": after, "growth_bytes": growth,
            "TRAIN_window_sampled_max_bytes": peaks,
            "TRAIN_window_exact_CUDA_allocator_peak_bytes": {
                "allocated": after["gpu_peak_allocated_bytes"],
                "reserved": after["gpu_peak_reserved_bytes"]},
            "CPU_NVML_peaks_are_sampled_not_exact": True,
            "final_and_initial_TEST_excluded": True,
            "cross_PID_difference_is_causal_cost": False}


def self_check():
    def row(t, pid, value):
        return {"pid": pid, "unix": t, "gpu_allocated_bytes": value,
                "gpu_reserved_bytes": value + 1, "gpu_peak_allocated_bytes": value,
                "gpu_peak_reserved_bytes": value + 1}
    before, after = row(100, 7, 10), row(102, 7, 12)
    after.update(gpu_peak_allocated_bytes=30, gpu_peak_reserved_bytes=31)
    rows = [row(99, 7, 100), before, row(101, 7, 16), row(101, 8, 4000), after, row(103, 7, 1000)]
    result = window(rows, before, after)
    assert result["TRAIN_window_sampled_max_bytes"]["gpu_allocated_bytes"] == 16
    assert result["TRAIN_window_exact_CUDA_allocator_peak_bytes"]["allocated"] == 30
    assert result["TRAIN_window_sampled_max_bytes"]["Pss_bytes"] is None
    for wrong in [dict(before, gpu_peak_allocated_bytes=99), dict(before, pid=8)]:
        try:
            window([*rows, wrong], wrong, after)
        except AssertionError:
            pass
        else:
            raise AssertionError("Must refuse invalid reset/PID evidence")
    return {"excludes_initial_final_and_other_PID": True,
            "sampled_maximum_separate_from_reset_allocator_peak": True,
            "missing_metric_stays_null": True,
            "invalid_reset_and_PID_rejected": True}


def collect():
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}
    plan_path = PILOTS / "CPU_PILOT_PLAN.json"
    plan = json.loads(plan_path.read_text())["spec"]
    sources[str(plan_path.relative_to(ROOT))] = sha(plan_path)
    for relative, digest in plan["source_sha256"].items():
        assert sha(ROOT / relative) == digest
        sources[relative] = digest
    worker = (ROOT / "analysis/onevision_two_module_pilot.py").read_text()
    assert "torch.cuda.reset_peak_memory_stats();x.monitor.snapshot('pilot_training_before'" in worker
    results, checkpoint_hashes = [], {}
    for task, variant in plan["cases"]:
        path = PILOTS / f"T{task:02d}_{variant}_complete.json"
        if not path.exists():
            continue  # Never inspect mutable/incomplete runs.
        proof = json.loads(path.read_text())
        assert proof["complete"] and proof["spec"] == plan and proof["formal_CL_result"] is False
        directory = ROOT / proof["run"]
        assert directory.resolve().is_relative_to((ROOT / "exp/onevision_two_module_20261017/pilot").resolve())
        sources[str(path.relative_to(ROOT))] = sha(path)
        for relative, digest in proof["artifact_sha256"].items():
            artifact = directory / relative
            assert artifact.resolve().is_relative_to(directory.resolve()) and sha(artifact) == digest
        # Bind raw endpoints to the completed proof. Scores are deliberately unread.
        for name in ["resources.jsonl", "memory_summary.json", "training.json", "model_counts.json"]:
            sources[str((directory / name).relative_to(ROOT))] = sha(directory / name)
        rows = [json.loads(line) for line in (directory / "resources.jsonl").read_text().splitlines()]
        summary = json.loads((directory / "memory_summary.json").read_text())
        before, after = summary["before"], summary["after"]
        assert before["phase"] == "pilot_training_before" and after["phase"] == "pilot_training_after"
        result = window(rows, before, after)
        assert result["growth_bytes"] == summary["growth_bytes"]
        training = json.loads((directory / "training.json").read_text())
        checkpoint_hashes[task, variant] = {k: proof["artifact_sha256"][k]
                                            for k in ("final_adapter.pt", "raw_gradient.pt", "optimizer_after.pt")}
        result.update(task=task, variant=variant, valid_training=proof["valid_training"],
                      run=proof["run"], original_mixed_process_sampled_max_bytes=summary["observed_peak_bytes"],
                      model_counts=json.loads((directory / "model_counts.json").read_text()),
                      selected_strength=training["selection"]["chosen_strength"],
                      accepted_step_fraction=training["actual_step_fraction"],
                      method_window_probe_QA_forwards=training["method_window_probe_QA_forwards"],
                      all_logged_probe_QA_forwards=training["extra_TRAIN_probe_QA_forwards"],
                      original_trainer_admission_QA=training["original_trainer_admission_QA"])
        results.append(result)
    paired = []
    for row in results:
        if row["variant"] == "control":
            continue
        controls = [r for r in results if r["task"] == row["task"] and r["variant"] == "control"]
        if not controls:
            continue
        control = controls[0]
        matched_bytes = {k: v == checkpoint_hashes[row["task"], "control"][k]
                         for k, v in checkpoint_hashes[row["task"], row["variant"]].items()}
        paired.append({"task": row["task"], "variant": row["variant"],
                       "same_completed_file_bytes_as_control": matched_bytes,
                       "GPU_allocated_before_difference_bytes": row["before"]["gpu_allocated_bytes"] - control["before"]["gpu_allocated_bytes"],
                       "GPU_allocated_after_difference_bytes": row["after"]["gpu_allocated_bytes"] - control["after"]["gpu_allocated_bytes"],
                       "GPU_growth_difference_bytes": row["growth_bytes"]["gpu_allocated_bytes"] - control["growth_bytes"]["gpu_allocated_bytes"],
                       "admission_warmup_QA_variant_control": [row["original_trainer_admission_QA"], control["original_trainer_admission_QA"]],
                       "module_memory_causality_identified": False})
    return {"CPU_only": True, "self_checks": self_check(), "completed_pilots": len(results),
            "planned_pilots": 12, "cases": results, "paired_to_control": paired, "source_sha256": sources,
            "memory_superiority_proven": False, "accuracy_causality_proven": False,
            "limits": ["Same-PID endpoint growth is measured, but independent process differences do not identify marginal module memory causality.",
                       "Original trainer admission and initial TEST are outside the reset method window; their compute must still be reported separately.",
                       "Control alone executes an original-trainer admission before the window; warmup histories differ, so before/after growth differences cannot be attributed solely to A/B modules.",
                       "CUDA after endpoint follows optimizer cleanup/empty_cache, so tiny endpoint growth does not imply a tiny training peak.",
                       "CPU/NVML periodic sampling may miss brief spikes; only CUDA reset allocator peaks are exact counters.",
                       "Training memory window includes TRAIN A/B probe computations and export; no initial/final TEST.",
                       "Independent small-task pilots do not establish whole19-stage or formalV2 resource superiority."]}


def render(result):
    mib = lambda value: f"{value / 1048576:.3f}"
    lines = ["# 原型训练窗口：修正峰值统计范围", "",
             "只读已完成证明绑定的原始资源日志，活动/排队GPU源码不变；TEST文件仅校验哈希，不解析答案或分数。原mixed maxima原样保留在JSON中，不能当作纯TRAIN峰值。", "",
             "| 任务/条件 | GPU allocated前→后MiB | 增长bytes | TRAIN重置allocator峰值MiB | TRAIN采样GPU峰值MiB | TRAIN采样NVML峰值MiB | TRAIN窗口秒 | 方法窗口额外评分题次 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["cases"]:
        lines.append(f"| T{r['task']:02d}/{r['variant']} | {mib(r['before']['gpu_allocated_bytes'])}→{mib(r['after']['gpu_allocated_bytes'])} | "
                     f"{r['growth_bytes']['gpu_allocated_bytes']} | {mib(r['TRAIN_window_exact_CUDA_allocator_peak_bytes']['allocated'])} | "
                     f"{mib(r['TRAIN_window_sampled_max_bytes']['gpu_allocated_bytes'])} | {mib(r['TRAIN_window_sampled_max_bytes']['gpu_process_nvml_bytes'])} | "
                     f"{r['seconds']:.3f} | {r['method_window_probe_QA_forwards']} |")
    lines += ["", "每行来自同一PID的before/after与其间资源样本。GPU allocator峰值读取after端点，并验证源代码先reset以及before peak等于当前占用；周期采样可漏过峰值，不能替代此计数。CPU RSS/PSS/USS和NVML仍只承诺采样最大值，原始bytes/端点全部写入[RESOURCE_WINDOWS.json](RESOURCE_WINDOWS.json)。", "",
              "训练后先清理优化器/缓存再取after，增长很小不代表峰值很小；参数、候选、原始梯度的CPU负担也不能只看GPU allocated。原控制额外运行过一次原训练器作为准入，这部分QA/计算在JSON单列，不包含在方法窗口，也没有被抹掉。", "",
              "## 相同参数也能有不同的增长值", "",
              "控制组在窗口前已经执行过原训练器准入，其他条件没有这一预热，必须单列这一差别。若两个条件最终参数、原始梯度和优化器文件SHA相同，却有不同的before与增长，不能把增长差直接归因为算法新增内存；本审计未定位具体运行时buffer，也不宣称泄漏。", ""]
    for p in result["paired_to_control"]:
        lines.append(f"- T{p['task']:02d}/{p['variant']}：相对control的before差{p['GPU_allocated_before_difference_bytes']}bytes、after差{p['GPU_allocated_after_difference_bytes']}bytes、增长差{p['GPU_growth_difference_bytes']}bytes；final_adapter/raw_gradient/optimizer三文件字节完全相同：{all(p['same_completed_file_bytes_as_control'].values())}；准入QA分别{p['admission_warmup_QA_variant_control']}。")
    lines += ["",
              "这是口径修正，尚未证明某方法比LoRA更省内存或更高F1。四组目标QA/参数/初始函数与原生输入的门槛仍独立验证。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    result = collect()
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "RESOURCE_WINDOWS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORT / "VERIFIED_RESULTS.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"CPU_only": True, "complete_pilots": result["completed_pilots"],
                      "cases": [{"task": r["task"], "variant": r["variant"],
                                 "actual_TRAIN_allocator_peak_bytes": r["TRAIN_window_exact_CUDA_allocator_peak_bytes"]["allocated"]}
                                for r in result["cases"]]}))
