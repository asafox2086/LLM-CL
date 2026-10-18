"""Read only the newly sealed TRAIN calibration; never open TEST predictions.

Image-group exclusion applies only to this 16-parameter head. The frozen
experts, feature map and memory already saw TRAIN, so this is not validation
of the entire pipeline. Nothing produced here replaces the deployed head.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

import onevision_functional_router_core as core

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    torch.set_num_threads(1)
    tick = time.monotonic()
    attempt = BASE / "functional_router/pilot_attempt.json"
    pointer = json.loads(attempt.read_text())
    run = ROOT / pointer["run"]
    assert run.resolve().is_relative_to((ROOT / "exp/onevision_functional_router_20261018/pilot").resolve())
    calibration = run / "calibration.pt"
    summary_path = run / "calibration_summary.json"
    summary = json.loads(summary_path.read_text())
    assert sha(calibration) == summary["calibration_sha256"]
    saved = torch.load(calibration, map_location="cpu", weights_only=True)
    spec = saved["spec"]
    assert spec == pointer["spec"] and saved["TEST_used"] is False
    assert summary["sealed_before_intervention_TEST"] and summary["actual_changed"]
    assert spec["LoRA_optimizer_updates"] == spec["additional_LoRAs"] == 0
    for name, expected in spec["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    parent_proof = BASE / "learned_reuse/learned_cosine_K3_train_complete.json"
    assert sha(parent_proof) == spec["parent_completion_sha256"]
    parent_run = ROOT / json.loads(parent_proof.read_text())["run"]
    parent_state = parent_run / "latest.pt"
    assert sha(parent_state) == spec["parent_state_sha256"]
    state = torch.load(parent_state, map_location="cpu", weights_only=True)
    ids = saved["TRAIN_ids"]
    assert ids == spec["calibration_TRAIN_ids"] == [x["id"] for x in state["memory"]]
    assert len(ids) == len(set(ids)) == summary["TRAIN_QA"]
    manifest = ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert sha(manifest) == spec["parent_spec"]["manifest_sha256"]
    rows, file_hashes = {}, {}
    # Only TRAIN files are opened. No TEST QA, prediction or answer is read.
    for task in json.loads(manifest.read_text())["tasks"]:
        info = task["files"]["train"]
        path = ROOT / info["path"]
        assert sha(path) == info["sha256"]
        file_hashes[info["path"]] = sha(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["id"] not in rows
            rows[row["id"]] = row
    assert len(rows) == 1313 and set(ids) <= set(rows)
    groups = {}
    for i, ident in enumerate(ids):
        images = rows[ident]["images"]
        assert images and isinstance(images, list)
        key = json.dumps(sorted(images), ensure_ascii=False)
        groups.setdefault(key, []).append(i)
    cos, nll = saved["cosines"], saved["TRAIN_NLL"]
    assert cos.shape == (len(ids), 3) and nll.shape == (len(ids), 4)
    assert torch.isfinite(cos).all() and torch.isfinite(nll).all()
    head, logs = core.fit(cos, nll)
    assert torch.equal(head, saved["weight"]), "Actual head must exactly reproduce"
    assert torch.equal(logs["raw_first_gradient"], saved["fit"]["raw_first_gradient"])
    assert torch.equal(logs["raw_last_gradient"], saved["fit"]["raw_last_gradient"])
    assert logs["events"] == saved["fit"]["events"]
    original, fitted = [], []
    for c in cos:
        expert, _, _ = core.choose("original_hard", c, head, state["threshold"])
        original.append(0 if expert is None else expert + 1)
        expert, _, _ = core.choose("quality_hard", c, head, state["threshold"])
        fitted.append(0 if expert is None else expert + 1)
    heldout = [-1] * len(ids)
    folds = []
    for key, indices in sorted(groups.items()):
        remaining = [i for i in range(len(ids)) if i not in indices]
        assert remaining and not set(remaining) & set(indices)
        fold_head, fold_logs = core.fit(cos[remaining], nll[remaining])
        for i in indices:
            expert, _, _ = core.choose("quality_hard", cos[i], fold_head, state["threshold"])
            heldout[i] = 0 if expert is None else expert + 1
        folds.append({"image_group_sha256": hashlib.sha256(key.encode()).hexdigest(),
                      "excluded_TRAIN_ids": [ids[i] for i in indices],
                      "head_training_QA": len(remaining), "CPU_steps": len(fold_logs["events"]),
                      "head_changed": fold_logs["changed"]})
    assert all(0 <= j <= 3 for j in heldout)
    best = nll.argmin(1)

    def stats(choices):
        chosen = nll[torch.arange(len(ids)), torch.tensor(choices)]
        minimum = nll.min(1).values
        return {"mean_teacher_forced_TRAIN_NLL": float(chosen.mean()),
                "mean_regret_to_best_of_four_TRAIN_NLL": float((chosen - minimum).mean()),
                "exact_minimum_NLL_count": int((chosen == minimum).sum()),
                "route_counts_base_e0_e1_e2": [choices.count(j) for j in range(4)]}

    result = {"CPU_only": True, "new_GPU_operations": 0, "TEST_opened": False,
              "TRAIN_QA": len(ids), "distinct_image_groups": len(groups),
              "exact_reproduced_head_and_gradient_and_loss": True,
              "original_geometry_hard": stats(original), "quality_hard_fit_same_TRAIN": stats(fitted),
              "quality_hard_leave_image_group_out_head_only": stats(heldout),
              "fixed_each_base_expert_mean_TRAIN_NLL": nll.mean(0).tolist(),
              "best_of_four_TRAIN_mean_NLL_reference_only": float(nll.min(1).values.mean()),
              "changed_routes_fit_vs_original": sum(a != b for a, b in zip(fitted, original)),
              "changed_routes_excluded_vs_fit": sum(a != b for a, b in zip(heldout, fitted)),
              "paired_TRAIN_rows": [{"id": ident, "original_class": original[i], "fit_class": fitted[i],
                                     "excluded_class": heldout[i], "minimum_NLL_class": int(best[i]),
                                     "four_actual_NLL": nll[i].tolist()} for i, ident in enumerate(ids)],
              "exclusion_folds": folds, "additional_LM_forwards": 0,
              "analysis_CPU_refits": 1 + len(folds), "analysis_CPU_steps": core.STEPS * (1 + len(folds)),
              "analysis_seconds": time.monotonic() - tick,
              "calibration_sha256": sha(calibration), "summary_sha256": sha(summary_path),
              "parent_state_sha256": sha(parent_state), "manifest_sha256": sha(manifest),
              "TRAIN_files_sha256": file_hashes,
              "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                "analysis/onevision_functional_router_core.py": sha(ROOT / "analysis/onevision_functional_router_core.py")},
              "limits": ["43 retained TRAIN rows, not random full TRAIN or TEST; experts/metric already saw these inputs.",
                         "Exclusion applies only to head fitting, not base, experts, metric or memory construction.",
                         "Reference best-NLL class is TRAIN-only diagnostic, never a deployed oracle.",
                         "Teacher-forced NLL does not establish generated Answer F1 or semantic mastery.",
                         "No soft residual NLL is inferred from convex expert losses.",
                         "No new head replaces the frozen GPU worker, no thresholds or settings selected."]}
    target = BASE / "functional_router/calibration_audit"
    write(target / "EVIDENCE.json", result)
    lines = ["# TRAIN质量校准：实际干预与排除同图拟合检查", "",
             f"仅原记忆{len(ids)}条TRAIN、{len(groups)}组图像；没有读取TEST，没有新增模型前向。原16参数/400步及首末梯度、loss轨迹精确重现。", "",
             "| 调用方式 | TRAIN平均NLL | 距四候选最低NLL的平均差 | 选中最低NLL数量 |",
             "|---|---:|---:|---:|"]
    for label, name in [("原几何硬阈值", "original_geometry_hard"), ("原质量头同样本拟合", "quality_hard_fit_same_TRAIN"),
                        ("仅质量头排除同图后推断", "quality_hard_leave_image_group_out_head_only")]:
        x = result[name]
        lines.append(f"| {label} | {x['mean_teacher_forced_TRAIN_NLL']:.6f} | {x['mean_regret_to_best_of_four_TRAIN_NLL']:.6f} | {x['exact_minimum_NLL_count']}/{len(ids)} |")
    lines += ["", f"拟合后相对原几何调用改变{result['changed_routes_fit_vs_original']}/{len(ids)}题；排除同图后相对拟合头改变{result['changed_routes_excluded_vs_fit']}/{len(ids)}题。", "",
              "这只是TRAIN记忆上的调用可学性/稳定性：原专家和向量投影已看过这些训练输入，不能当完整流程未见样本验证，更不能当测试F1收益或专业知识充分的证明。全部排除头均为离线分析，不替换活动GPU头，不选部署参数。", "",
              f"分析额外CPU拟合{result['analysis_CPU_refits']}次/{result['analysis_CPU_steps']}步，计入分析成本；额外LM前向为0。当前GPU四视图和两个完整候选仍全部保留；领域整体性能与资源成本才是方法评价标准。", "",
              "[证据、全部43题和排除组](EVIDENCE.json)。"]
    (target / "VERIFIED_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ["TRAIN_QA", "distinct_image_groups", "original_geometry_hard", "quality_hard_fit_same_TRAIN", "quality_hard_leave_image_group_out_head_only", "changed_routes_fit_vs_original", "changed_routes_excluded_vs_fit", "analysis_seconds"]}))


if __name__ == "__main__":
    main()
