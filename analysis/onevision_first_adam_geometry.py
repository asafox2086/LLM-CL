"""Read-only CPU audit of completed pilot first AdamW updates.

No model forward, TEST scores, optimizer execution or live run reads. The first
Adam formula is prior optimizer algebra, not a new CL mechanism or F1 proof.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016"
REPORT = BASE / "first_adam_geometry"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cosine(torch, a, b):
    return float(torch.dot(a, b) / torch.linalg.vector_norm(a) / torch.linalg.vector_norm(b))


def residual_factors(torch, before, after):
    result = {}
    for name in before:
        a, b = before[name]["A"].double(), before[name]["B"].double()
        da, db = after[name]["A"].double() - a, after[name]["B"].double() - b
        # BA_after-BA_before = dB*A + B*dA + dB*dA, without subtracting
        # nearly equal dense BA matrices or cancelling large Gram terms.
        result[name] = (torch.cat((db, b, db), dim=1), torch.cat((a, da, da), dim=0))
    return result


def residual_inner(left, right, scale):
    return sum(float(((left[n][0].T @ right[n][0]) *
                      (left[n][1] @ right[n][1].T)).sum()) * scale**2 for n in left)


def self_check(torch):
    rng = torch.Generator().manual_seed(42)
    make = lambda shape: torch.randn(shape, generator=rng, dtype=torch.float64)
    for mode in ("both", "A_only", "B_only", "zero"):
        start = {"toy": {"A": make((2, 5)), "B": make((7, 2))}}
        after = {"toy": {"A": start["toy"]["A"] + (make((2, 5)) if mode in ("both", "A_only") else 0),
                         "B": start["toy"]["B"] + (make((7, 2)) if mode in ("both", "B_only") else 0)}}
        dense = 4 * (after["toy"]["B"] @ after["toy"]["A"] - start["toy"]["B"] @ start["toy"]["A"])
        factored = residual_factors(torch, start, after)
        recovered = 4 * factored["toy"][0] @ factored["toy"][1]
        torch.testing.assert_close(recovered, dense, rtol=1e-12, atol=1e-12)
        assert math.isclose(residual_inner(factored, factored, 4), float((dense * dense).sum()), rel_tol=1e-12, abs_tol=1e-12)
        other = {"toy": {"A": make((2, 5)), "B": make((7, 2))}}
        other_dense = 4 * (other["toy"]["B"] @ other["toy"]["A"] - start["toy"]["B"] @ start["toy"]["A"])
        cross = residual_inner(factored, residual_factors(torch, start, other), 4)
        assert math.isclose(cross, float((dense * other_dense).sum()), rel_tol=1e-12, abs_tol=1e-12)
    return {"dense_vs_residual_factorization_modes": 4, "FP64_rtol_atol": 1e-12,
            "dense_cross_inner_products_checked": 4,
            "supports_actual_finite_BA_displacement_not_only_linearization": True}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    checks = self_check(torch)
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}

    def checked(path, expected=None):
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        sources[str(path.relative_to(ROOT))] = digest
        return path

    plan = json.loads(checked(BASE / "two_module_prototype/CPU_PILOT_PLAN.json").read_text())["spec"]
    for name, digest in plan["source_sha256"].items():
        checked(ROOT / name, digest)
    start = torch.load(checked(ROOT / plan["past_natural_checkpoint"], plan["past_natural_sha256"]),
                       map_location="cpu", weights_only=True)["adapter"]
    shared_function = {n: (v["B"].double(), v["A"].double()) for n, v in start.items()}
    shared_function_norm = math.sqrt(residual_inner(shared_function, shared_function, 4))
    data, pending = [], []
    for task in (2, 11, 12):
        proof_path = BASE / f"two_module_prototype/T{task:02d}_control_complete.json"
        if not proof_path.exists():
            pending.append(task)
            continue
        proof = json.loads(checked(proof_path).read_text())
        assert proof["complete"] and proof["valid_training"] and proof["spec"] == plan
        run = (ROOT / proof["run"]).resolve()
        assert run.is_relative_to((ROOT / "exp/onevision_two_module_20261017/pilot").resolve())

        def artifact(name):
            return checked(run / name, proof["artifact_sha256"][name])

        training = json.loads(artifact("training.json").read_text())
        raw = torch.load(artifact("raw_gradient.pt"), map_location="cpu", weights_only=True)
        optimizer = torch.load(artifact("optimizer_after.pt"), map_location="cpu", weights_only=True)
        export = torch.load(artifact("final_adapter.pt"), map_location="cpu", weights_only=True)
        final = export["adapter"]
        assert len(start) == len(final) == 48 and start.keys() == final.keys()
        assert len(optimizer["param_groups"]) == 1 and training["actual_step_fraction"] == 1
        group = optimizer["param_groups"][0]
        assert group["weight_decay"] == 0 and not group["amsgrad"] and not group["maximize"]
        assert group["lr"] == training["actual_learning_rate_used"] == 1e-4
        assert group["betas"] == (.9, .999) and group["eps"] == 1e-8
        assert set(raw["TRAIN_ids"]) == set(plan["TRAIN_ids"][str(task)])
        assert len(raw["TRAIN_ids"]) == training["original_target_TRAIN_QA"]
        gradient = raw["gradient"].double()
        assert gradient.numel() == plan["trainable_parameters"] == 540672
        assert len(raw["layout"]) == len(group["params"]) == 96
        updates, formulas, first_moment_gradients = [], [], []
        visited, offset = set(), 0
        for index, (name, shape) in enumerate(raw["layout"]):
            module, factor, suffix = name.rsplit(".", 2)
            assert suffix == "weight" and (module, factor) not in visited
            visited.add((module, factor))
            before, after = start[module][factor], final[module][factor]
            assert list(before.shape) == list(after.shape) == shape
            assert before.dtype == after.dtype == torch.float32
            state = optimizer["state"][group["params"][index]]
            assert float(state["step"]) == 1
            m, v = state["exp_avg"], state["exp_avg_sq"]
            assert m.shape == v.shape == before.shape and torch.isfinite(m).all() and torch.isfinite(v).all()
            formula = -group["lr"] / (1 - group["betas"][0]) * m / (
                v.sqrt() / math.sqrt(1 - group["betas"][1]) + group["eps"])
            updates.append((after - before).flatten().double())
            formulas.append(formula.flatten().double())
            first_moment_gradients.append((m.double() / (1 - group["betas"][0])).flatten())
            offset += before.numel()
        assert offset == gradient.numel()
        delta, formula = torch.cat(updates), torch.cat(formulas)
        from_m = torch.cat(first_moment_gradients)
        assert torch.isfinite(delta).all() and torch.linalg.vector_norm(delta) > 0
        assert int((gradient * delta > 0).sum()) == 0
        coefficient = min(1., 1. / (training["actual_gradient_L2_preclip"] + 1e-6))
        clipped = gradient * coefficient
        factors = residual_factors(torch, start, final)
        scale = export["alpha"] / export["rank"]
        assert scale == 4
        total = {n: (v["B"].double(), v["A"].double()) for n, v in final.items()}
        value = {"task": task, "TRAIN_QA": len(raw["TRAIN_ids"]),
                 "raw_gradient_L2_FP64": float(torch.linalg.vector_norm(gradient)),
                 "raw_gradient_L2_GPU_record": training["actual_gradient_L2_preclip"],
                 "actual_parameter_displacement_L2": float(torch.linalg.vector_norm(delta)),
                 "actual_BA_displacement_Frobenius": math.sqrt(residual_inner(factors, factors, scale)),
                 "shared_initial_total_BA_Frobenius": shared_function_norm,
                 "final_total_BA_Frobenius": math.sqrt(residual_inner(total, total, scale)),
                 "median_absolute_coordinate_step_over_LR": float((delta.abs() / group["lr"]).median()),
                 "fraction_coordinates_step_at_least_99percent_LR": float((delta.abs() >= .99 * group["lr"]).double().mean()),
                 "recorded_moment_formula_max_absolute_step_error": float((delta - formula).abs().max()),
                 "moment_recovered_vs_clipped_gradient_relative_L2_error": float(torch.linalg.vector_norm(from_m - clipped) / torch.linalg.vector_norm(clipped)),
                 "uphill_raw_gradient_coordinates": 0, "optimizer_step_count_each_parameter": 1,
                 "formula_is_bit_exact_GPU_reproduction": False}
        data.append((value, gradient, delta, factors, total))
    comparisons = []
    for a, b in itertools.combinations(data, 2):
        x, y = a[0], b[0]
        comparisons.append({"tasks": [x["task"], y["task"]],
                            "second_over_first_raw_gradient_norm": y["raw_gradient_L2_FP64"] / x["raw_gradient_L2_FP64"],
                            "second_over_first_actual_step_norm": y["actual_parameter_displacement_L2"] / x["actual_parameter_displacement_L2"],
                            "raw_gradient_cosine": cosine(torch, a[1], b[1]),
                            "actual_parameter_displacement_cosine": cosine(torch, a[2], b[2]),
                            "actual_BA_displacement_cosine": residual_inner(a[3], b[3], 4) / x["actual_BA_displacement_Frobenius"] / y["actual_BA_displacement_Frobenius"],
                            "final_total_BA_cosine_including_shared_initial": residual_inner(a[4], b[4], 4) / x["final_total_BA_Frobenius"] / y["final_total_BA_Frobenius"]})
    for path, digest in sources.items():
        assert sha(ROOT / path) == digest
    assert not torch.cuda.is_initialized()
    result = {"CPU_only": True, "TEST_predictions_read": False, "GPU_or_optimizer_executed": False,
              "CPU_geometry_checks": checks,
              "cases": [d[0] for d in data], "comparisons": comparisons, "pending_control_tasks": pending,
              "source_sha256": sources,
              "limits": ["First Adam normalization is known optimizer algebra, not a new mathematical discovery.",
                         "Actual raw target gradients and saved first moments/weights are used; no old-task gradient is available here.",
                         "Component saturation and smaller angular similarity do not prove harmful transfer or cause a generated F1 deficit.",
                         "Distinct target QA, unique counts and label lengths remain; only start and optimizer settings match.",
                         "No TEST score used for choosing capacity, route, optimizer, step size or training budget.",
                         "No claimed bit-exact CPU/GPU optimizer replay; formula error explicitly retained.",
                         "Raw gradient norm, actual parameter displacement and BA function displacement are distinct quantities."]}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "EVIDENCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 单步任务：梯度大，不等于实际学得多", "",
             "只读完成control的原始梯度、首步AdamW状态和前后参数；不读TEST预测、不做GPU前向或新训练。", "",
             "| 任务 | TRAIN QA | 原始梯度L2 | 实际参数位移L2 | 坐标步长≥99% LR比例 | BA实际位移范数 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["cases"]:
        lines.append(f"| T{row['task']:02d} | {row['TRAIN_QA']} | {row['raw_gradient_L2_FP64']:.6f} | {row['actual_parameter_displacement_L2']:.6f} | {row['fraction_coordinates_step_at_least_99percent_LR']:.2%} | {row['actual_BA_displacement_Frobenius']:.6f} |")
    lines += ["", "首步AdamW（fresh状态、weight_decay=0）的理想化坐标位移为 −lr·g_clip/(|g_clip|+eps)。这里所有参数step=1；实际文件也包含FP32舍入，不称CPU/GPU逐比特复现。", ""]
    for c in comparisons:
        lines.append(f"T{c['tasks'][0]:02d}→T{c['tasks'][1]:02d}：原始梯度范数比例{c['second_over_first_raw_gradient_norm']:.4f}，实际参数位移范数比例{c['second_over_first_actual_step_norm']:.4f}；梯度余弦{c['raw_gradient_cosine']:.4f}，实际参数位移余弦{c['actual_parameter_displacement_cosine']:.4f}，实际BA位移余弦{c['actual_BA_displacement_cosine']:.4f}。")
        lines.append(f"两个最终完整LoRA矩阵（含共同旧功能）余弦{c['final_total_BA_cosine_including_shared_initial']:.6f}，但减去同一个阶段起点后的本任务BA增量余弦仅{c['actual_BA_displacement_cosine']:.6f}。完整矩阵相似不等于新学方向重复；这也不证明新增独立专家必有生成收益。")
    lines += ["", "- **探究的问题：** 小任务只有一个fresh Adam更新时，能否由梯度范数直接判断学习量或干扰？",
              "- **做的实验：** 同一旧自然LoRA起点，读取完成任务的全部540672维原始梯度、96份真实优化器状态与48层A/B；另算实际scale·BA位移。",
              "- **结果：** 见表和配对；公式重建与实际参数位移的全部误差保留在JSON。",
              "- **证明了什么：** 这些任务的原始梯度大小与实际位移大小并不成比例，实际位移方向也不同于原始梯度方向；不能把较大梯度范数直接叫作更多学习或更强干扰。",
              "- **改进的方法：** 归因同时报告真实优化器位移和BA函数变化，再做有限旧任务loss与生成质量控制；不依据本次TEST改优化器/超参数。",
              "- **改进的效果：** 修正机制解释口径，无新模型性能提升；不能仅凭这些几何量证明遗忘原因或专业知识充分。", "",
              f"未完成control任务：{pending}。先验代数、实际参数证据和性能因果分开；完整任务/领域48轨迹仍需运行。[全部数值与SHA](EVIDENCE.json)。", ""]
    (REPORT / "VERIFIED_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"CPU_only": True, "cases": result["cases"], "comparisons": comparisons, "pending": pending}))


if __name__ == "__main__":
    main()
