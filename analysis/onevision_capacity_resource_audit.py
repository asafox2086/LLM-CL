"""Read-only, CPU-only resource accounting of completed capacity diagnostics.

Do not turn serialized tensor bytes into an estimate of live process RSS.
No test predictions, model forwards, or mutable active-run files are read.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016"
OUT = BASE / "capacity_resource_evidence"
KEYS = ("VmRSS_bytes", "Pss_bytes", "cpu_uss_bytes", "gpu_allocated_bytes",
        "gpu_reserved_bytes", "gpu_process_nvml_bytes")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def tensors(tree):
    import torch
    if isinstance(tree, torch.Tensor):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from tensors(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from tensors(value)


def payload(tree):
    """Logical tensor payload and deduplicated backing-storage bytes, separately."""
    items = list(tensors(tree))
    unique = {}
    for value in items:
        assert value.device.type == "cpu", "Audit cannot allocate CUDA storage"
        storage = value.untyped_storage()
        key = (str(value.device), storage.data_ptr(), storage.nbytes())
        unique[key] = storage.nbytes()
    return {"tensor_references": len(items),
            "logical_elements": sum(t.numel() for t in items),
            "logical_bytes": sum(t.numel() * t.element_size() for t in items),
            "unique_backing_storage_bytes": sum(unique.values()),
            "dtype_names": sorted({str(t.dtype) for t in items}),
            "loaded_device": "CPU; load location is not original residency proof"}


def maxima(rows):
    return {k: max((row[k] for row in rows if row.get(k) is not None), default=None)
            for k in KEYS}


def self_check():
    import torch
    a = torch.arange(10, dtype=torch.float32)
    got = payload({"same": a, "alias": a, "view": a[2:5]})
    assert got["logical_bytes"] == 92 and got["unique_backing_storage_bytes"] == 40
    got = payload([a, a.clone()])
    assert got["logical_bytes"] == got["unique_backing_storage_bytes"] == 80
    got = maxima([{"gpu_allocated_bytes": 3}, {"gpu_allocated_bytes": 7}])
    assert got["gpu_allocated_bytes"] == 7 and got["Pss_bytes"] is None
    assert not torch.cuda.is_initialized()
    return {"alias_and_view_storage_deduplication": True,
            "independent_clone_billed_separately": True,
            "missing_resource_is_null": True, "CUDA_initialized": False}


def audit(cap, source_map):
    import torch
    proof_path = BASE / f"capacity/cap{cap}_complete.json"
    proof = json.loads(proof_path.read_text())
    assert proof["complete"] and proof["cap"] == cap
    source_map[str(proof_path.relative_to(ROOT))] = sha(proof_path)
    for relative, expected in proof["source_sha256"].items():
        assert sha(ROOT / relative) == expected, relative
        source_map[relative] = expected
    run = ROOT / proof["run"]

    def checked(relative):
        p = run / relative
        digest = sha(p)
        assert digest == proof["artifact_sha256"][relative], relative
        source_map[str(p.relative_to(ROOT))] = digest
        return p

    saved = torch.load(checked("latest.pt"), map_location="cpu", weights_only=True)
    assert saved["stage"] == 19 and saved["cap"] == cap
    assert saved["source_sha256"] == proof["source_sha256"]
    assert len(saved["pool"]) == proof["retained_experts"] == cap
    pool = payload(saved["pool"])
    prototypes = payload(saved["prototypes"])
    assert pool["logical_elements"] == cap * 540672
    assert pool["logical_bytes"] == pool["unique_backing_storage_bytes"] == cap * 2162688
    assert pool["dtype_names"] == ["torch.float32"]
    assert len(saved["prototypes"]) == len(saved["assignments"]) == 19
    assert all(e in saved["pool"] for e in saved["assignments"].values())
    events = [json.loads(line) for line in checked("events.jsonl").read_text().splitlines()]
    resources = [json.loads(line) for line in checked("resources.jsonl").read_text().splitlines()]
    summary = json.loads(checked("memory_summary.json").read_text())
    before, after = summary["before"], summary["after"]
    assert before["pid"] == after["pid"] == summary["pid"]
    assert before in resources and after in resources and before["unix"] < after["unix"]
    assert before["phase"] == "capacity_training_before"
    assert after["phase"] == "capacity_training_after"
    same_process = [row for row in resources if row["pid"] == summary["pid"]
                    and row["unix"] >= summary["process_started_unix"]]
    window = [row for row in same_process if before["unix"] <= row["unix"] <= after["unix"]]
    assert summary["observed_peak_bytes"] == maxima(same_process)
    growth = {k: after[k] - before[k] if after.get(k) is not None and before.get(k) is not None
              else None for k in KEYS}
    assert growth == summary["growth_bytes"]
    decisions, updates, optimizer_qa = [], 0, 0
    train_seconds = 0.0
    for stage in range(1, 20):
        d = json.loads(checked(f"decisions/cap{cap}_T{stage:02d}.json").read_text())
        assert d["stage"] == stage
        assert d["persistent_lora_parameters"] == 540672 * d["retained_experts"]
        group = [e for e in events if e["event"] == "train"
                 and e["label"] == f"cap{cap}_T{stage:02d}"]
        n = d["current_optimizer_qa"]
        assert len(group) == math.ceil(n / 16)
        assert [e["qa_seen"] for e in group] == [min(16 * i, n) for i in range(1, len(group) + 1)]
        assert all(e["total_qa"] == n and e["learning_rate_used"] > 0 for e in group)
        assert all(math.isfinite(e["loss"]) and math.isfinite(e["gradient_l2_preclip"]) for e in group)
        assert group[-1]["qa_seen"] == n
        updates += len(group)
        optimizer_qa += n
        train_seconds += sum(e["seconds"] for e in group)
        decisions.append({k: d[k] for k in ("stage", "retained_experts", "auxiliary_train_forwards",
                                          "prototype_actual_new_features", "prototype_seconds",
                                          "gradient_probe_train_qa")})
    assert optimizer_qa == 1313 and updates == 93
    assert before["trainable_parameters"] == after["trainable_parameters"] == 540672
    assert not torch.cuda.is_initialized()
    return {"cap": cap, "proof_run": proof["run"], "pid": summary["pid"],
            "retained_modules": len(saved["pool"]), "retained_pool": pool,
            "router_prototypes": prototypes,
            "loaded_checkpoint_all_tensors": payload(saved),
            "model_monitor_trainable_parameters": after["trainable_parameters"],
            "before": before, "after": after, "growth_bytes": growth,
            "training_span_including_intermediate_TEST_sampled_max_bytes": maxima(window),
            "whole_process_sampled_max_bytes": maxima(same_process),
            "post_training_samples_in_original_peak_scope": sum(row["unix"] > after["unix"] for row in same_process),
            "training_span_seconds_including_auxiliary_TEST_and_saves": after["unix"] - before["unix"],
            "logged_optimizer_step_seconds_not_all_training_overhead": train_seconds,
            "optimizer_updates": updates, "target_optimizer_QA": optimizer_qa,
            "auxiliary_train_scoring_QA_forwards": sum(d["auxiliary_train_forwards"] for d in decisions),
            "prototype_cache_misses": sum(d["prototype_actual_new_features"] for d in decisions),
            "gradient_probe_QA_logged_count_not_forward_equivalent": sum(d["gradient_probe_train_qa"] for d in decisions),
            "per_stage_resources": decisions,
            "deployment_process_memory_measured": False,
            "pure_TRAIN_peak_separately_identified": False,
            "original_checkpoint_file_bytes_not_RAM": (run / "latest.pt").stat().st_size}


def render(result):
    rows = result["caps"]
    mib = lambda b: f"{b / 1048576:.3f}"
    lines = ["# 容量实验的内存与计算口径：离线只读核验", "",
             "四组已经完成的容量诊断，逐项验证原完成证明、源码与权重/日志SHA。只在CPU读取已完成记录，不读TEST答案/预测，不运行模型或改动队列。", "",
             "## 1. 多个专家实际计入了哪里", "",
             "| 实际LoRA数 | 监控可训练参数 | 保存专家总参数 | 专家FP32存储MiB | 原型存储MiB | GPU allocated前→后MiB | GPU allocated增长MiB |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['retained_modules']} | {r['model_monitor_trainable_parameters']} | {r['retained_pool']['logical_elements']} | "
                     f"{mib(r['retained_pool']['unique_backing_storage_bytes'])} | {mib(r['router_prototypes']['unique_backing_storage_bytes'])} | "
                     f"{mib(r['before']['gpu_allocated_bytes'])}→{mib(r['after']['gpu_allocated_bytes'])} | {mib(r['growth_bytes']['gpu_allocated_bytes'])} |")
    lines += ["", "结论：监控中的可训练参数只反映当前激活的模型适配器，不能代替专家池总容量。原脚本用CPU适配器快照保存pool，再restore到当前模型；CPU加载本身不是原运行设备证据，必须结合冻结源码。四组GPU端点增长相同，不能推出总内存增长相同。池张量字节是精确存储负担，不是进程RSS，不能把它直接与GPU缓存相加当作新的实测总内存。", "",
              "## 2. 原始进程占用保留，评价与训练峰值分开解释", "",
              "| LoRA数 | CPU RSS前→后MiB | CPU PSS前→后MiB | CPU USS前→后MiB | GPU reserved前→后MiB | GPU NVML前→后MiB |",
              "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        cells = [f"{mib(r['before'][k])}→{mib(r['after'][k])}" for k in
                 ("VmRSS_bytes", "Pss_bytes", "cpu_uss_bytes", "gpu_reserved_bytes", "gpu_process_nvml_bytes")]
        lines.append(f"| {r['cap']} | " + " | ".join(cells) + " |")
    lines += ["", "每行前后属于同一PID，可报告该行增长；跨行是不同进程，不能相减来估计增添一个专家的因果内存开销。原容量训练阶段夹有诊断TEST，原observed_peak扫描还包含最终评价；本审计另列截止TRAIN-after的原始采样峰值，但它仍包含中间评价，不能称纯训练峰值。CUDA allocator峰值没有在该TRAIN窗口开始处重新reset，不能用端点的进程历史峰值冒充窗口峰值。RSS/PSS/USS/NVML采样可能漏掉短时尖峰。", "",
              "## 3. 少数据不等于少计算", "",
              "| LoRA数 | 目标QA/优化器更新 | 额外TRAIN评分题次 | 原型新特征缓存未命中 | 梯度探针记录题数 | 优化器步日志秒数 | 训练区间含评价/保存秒数 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['cap']} | {r['target_optimizer_QA']}/{r['optimizer_updates']} | {r['auxiliary_train_scoring_QA_forwards']} | "
                     f"{r['prototype_cache_misses']} | {r['gradient_probe_QA_logged_count_not_forward_equivalent']} | "
                     f"{r['logged_optimizer_step_seconds_not_all_training_overhead']:.3f} | {r['training_span_seconds_including_auxiliary_TEST_and_saves']:.3f} |")
    lines += ["", "不同列不能简单相加：原型缓存未命中是特征提取事件，不等于完整图文loss前向；梯度探针还包含反向；优化器步计时不包含全部TRAIN评分/特征开销。所有组目标QA与93个正LR更新一致，但额外TRAIN评分并不相同。不能只以1313条训练数据或0.5B语言基座宣称整套流程轻量。", "",
              "## 4. 六栏结论（问题12）", "",
              "- **探究的问题**：多LoRA是否必然体现在监控的GPU增长与可训练参数中？少样本实验的资源成本能否只看这两项？",
              "- **做的实验**：CPU只读已完成cap1/2/3/5，校验权重、19阶段决策、93次更新和同PID资源端点；区分逻辑tensor字节、去重storage、实际进程占用与计算事件。",
              "- **结果**：专家总参数随实际容量增长，但激活模型的可训练计数和GPU allocated增长未变；完整数值见表。额外TRAIN评分/原型/梯度探针需要另计。",
              "- **证明了什么**：当前计数不能作为总专家容量或部署内存的代理；离线/CPU专家可把容量负担从GPU指标中隐藏。未证明泄漏、额外专家免费或任何方法内存更优；这是资源归因，非准确率退化因果。",
              "- **改进的方法**：允许有界多LoRA，但以匹配设置的CPU/GPU常驻、增长、纯TRAIN峰值及额外计算联合验证；专家/历史/路由全计费。现有活动源码不热改，未来新版本需独立TRAIN窗口。",
              "- **改进的效果**：完成资源口径纠偏，尚无新增性能或内存优势结果；最终方法仍须生成F1与实际成本同时证明。", "",
              "完整逐字节端点、范围、成本与SHA见[RESOURCE_AUDIT.json](RESOURCE_AUDIT.json)。本审计不替代原48组任务/领域归因，也不构成正式V2方法排名。", ""]
    return "\n".join(lines)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicitly hide all GPUs"
    import torch
    assert not torch.cuda.is_initialized()
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}
    checks = self_check()
    rows = [audit(cap, sources) for cap in (1, 2, 3, 5)]
    assert len({r["pid"] for r in rows}) == 4
    assert len({r["growth_bytes"]["gpu_allocated_bytes"] for r in rows}) == 1
    assert len({r["router_prototypes"]["unique_backing_storage_bytes"] for r in rows}) == 1
    assert not torch.cuda.is_initialized()
    result = {"passed": True, "CPU_only": True, "self_checks": checks,
              "caps": rows, "source_sha256": sources,
              "claims_memory_superiority": False, "claims_accuracy_causality": False}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RESOURCE_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "VERIFIED_RESULTS.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({"CPU_only": True, "caps": [{"cap": r["cap"], "pool_bytes": r["retained_pool"]["logical_bytes"],
                      "extra_TRAIN_scoring_QA": r["auxiliary_train_scoring_QA_forwards"]} for r in rows]}))


if __name__ == "__main__":
    main()
