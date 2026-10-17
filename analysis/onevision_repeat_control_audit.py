"""Read-only CPU audit: do protection controls differ without a domain switch?

Only a complete six-method repeated-anchor panel is admitted. The GPU workers,
their sources, training choices, original predictions and observers are untouched.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from onevision_matched_task_domain_report import paired, checked_rows

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016/matched_task_domain"
METHODS = ["SeqFT_LoRA", "ER_LoRA", "EWC_lambda1", "EWC_lambda100",
           "anchor_gradient_noop", "anchor_gradient_projection"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(a, b):
    assert a.keys() == b.keys()
    return all(torch.equal(a[n][f], b[n][f]) for n in a for f in ("A", "B"))


def quadratic(adapter, fisher):
    terms = []
    for name, diagonal in fisher["diagonal"].items():
        module, factor, suffix = name.rsplit(".", 2)
        assert suffix == "weight" and factor in ("A", "B")
        p, c, f = adapter[module][factor].double(), fisher["mean"][name].double(), diagonal.double()
        assert p.shape == c.shape == f.shape and torch.isfinite(f).all() and (f >= 0).all()
        terms.append((f * (p-c).square()).sum() / 2)
    return float(torch.stack(terms).sum())


def self_check():
    a = {"toy": {"A": torch.ones((1, 2)), "B": torch.zeros((2, 1))}}
    f = {"mean": {"toy.A.weight": a["toy"]["A"].clone(), "toy.B.weight": a["toy"]["B"].clone()},
         "diagonal": {"toy.A.weight": torch.ones((1, 2)), "toy.B.weight": torch.ones((2, 1))}}
    assert quadratic(a, f) == 0
    b = {"toy": {"A": a["toy"]["A"].clone(), "B": a["toy"]["B"].clone()}}
    b["toy"]["B"][0, 0] = .5
    assert quadratic(b, f) == .125 and not same(a, b)
    b["toy"]["B"][0, 0] = 0
    b["toy"]["A"][0, 0] = torch.nextafter(b["toy"]["A"][0, 0], torch.tensor(float("inf")))
    assert not same(a, b), "Equality audit must not silently accept a small parameter change"
    assert not torch.cuda.is_initialized()
    return {"center_zero": True, "finite_displacement_positive": True,
            "one_ULP_parameter_difference_rejected": True, "no_CUDA": True}


def build(anchor):
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__),
               "analysis/onevision_matched_task_domain_report.py": sha(ROOT / "analysis/onevision_matched_task_domain_report.py")}

    def check(path, digest=None):
        actual = sha(path)
        assert digest is None or actual == digest, str(path)
        key = str(path.relative_to(ROOT))
        if key in sources:
            assert sources[key] == actual
        sources[key] = actual
        return path

    plan = json.loads(check(BASE / "CPU_PLAN_CHECK.json").read_text())
    assert plan["passed"] and plan["GPU_used"] is False
    lock = json.loads(check(ROOT / "summary/primary_metric_20261017/metric_lock.json").read_text())
    check(ROOT / "analysis/answer_f1.py", lock["metric_sha256"])
    assert lock["option_mapping"] is False
    spec = plan["spec"]
    assert spec["sequences"][str(anchor)]["repeat_anchor"] == [anchor]*5
    for path, digest in spec["source_sha256"].items():
        check(ROOT / path, digest)
    manifest = json.loads(check(ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json", spec["manifest_sha256"]).read_text())
    original = {}
    for task in manifest["tasks"]:
        f = task["files"]["test"]
        original.update({row["id"]: row for row in map(json.loads, check(ROOT / f["path"], f["sha256"]).read_text().splitlines())})
    cases, summaries = {}, []
    for method in METHODS:
        name = f"T{anchor:02d}_repeat_anchor__{method}"
        proof_path = BASE / (name + "_complete.json")
        assert proof_path.exists(), "Incomplete repeated-task panel; do not infer from partial training"
        proof = json.loads(check(proof_path).read_text())
        assert proof["complete"] and proof["spec"] == spec and proof["stage1_exact"]
        run = (ROOT / proof["run"]).resolve()
        assert run.is_relative_to(ROOT / "exp/onevision_matched_task_domain_20261017")
        for path, digest in proof["artifact_sha256"].items():
            p = (run/path).resolve()
            assert p.is_relative_to(run)
            check(p, digest)

        def load(relative):
            assert relative in proof["artifact_sha256"]
            return torch.load(run / relative, map_location="cpu", weights_only=True)

        states = [load(f"adapters/stage_{s:02d}.pt")["adapter"] for s in range(1, 6)]
        raw = [load(f"raw_step_gradients/{name}/stage_{s:02d}.pt") for s in range(1, 6)]
        training = [json.loads((run / f"training/{name}/stage_{s:02d}.json").read_text()) for s in range(1, 6)]
        assert all([i for e in t["events"] for i in e["train_ids"]] == spec["TRAIN_ids"][str(anchor)] for t in training)
        assert all(len(t["events"]) == 4 and all(e["lr_used"] == 1e-4 for e in t["events"]) for t in training)
        ids = spec["anchor_probe_ids"][str(anchor)]
        endpoints = {s: checked_rows(run / f"evaluations/anchor_after{s}/anchor_image.jsonl", ids, original) for s in (1, 3, 5)}
        current = [checked_rows(run / f"evaluations/plasticity_after{s}/current_image.jsonl", spec["plasticity_ids"][str(anchor)], original) for s in range(2, 6)]
        cases[method] = {"states": states, "raw": raw, "endpoints": endpoints, "current": current,
                         "training": training, "run": str(run.relative_to(ROOT))}
        summaries.append({"method": method, "F1_S1_S3_S5": [sum(r["scores"]["token_f1"] for r in endpoints[s])/len(ids) for s in (1, 3, 5)],
                          "NLL_S1_S3_S5": [sum(r["nll"] for r in endpoints[s])/len(ids) for s in (1, 3, 5)],
                          "current_F1_stage_observation_mean": sum(r["scores"]["token_f1"] for rows in current for r in rows)/32,
                          "target_QA_exposures": 40,
                          "replay_QA_exposures": sum(len(e["replay_ids"]) for t in training for e in t["events"]),
                          "positive_penalty_updates_after_stage1": sum(e["penalty"] > 0 for t in training[1:] for e in t["events"]),
                          "projection_flags": sum(e["projection_applied"] for t in training for e in t["events"])})
    seq = cases["SeqFT_LoRA"]
    for case in cases.values():
        assert same(seq["states"][0], case["states"][0])
        assert [(r["id"], r["prediction"], r["nll"], r["scores"]) for r in seq["endpoints"][1]] == [(r["id"], r["prediction"], r["nll"], r["scores"]) for r in case["endpoints"][1]]
    identity = []
    for method in METHODS[-2:]:
        case = cases[method]
        assert all(same(a, b) for a, b in zip(seq["states"], case["states"]))
        assert all(torch.equal(a["pre_intervention_gradient"], b["pre_intervention_gradient"]) for a, b in zip(seq["raw"], case["raw"]))
        for s in (1, 3, 5):
            assert [(r["id"], r["prediction"], r["nll"], r["scores"]) for r in seq["endpoints"][s]] == [(r["id"], r["prediction"], r["nll"], r["scores"]) for r in case["endpoints"][s]]
        assert all(e["anchor_gradient_dot"] >= 0 for t in case["training"][1:] for e in t["events"])
        identity.append({"method": method, "all5_states_exact_SeqFT": True, "all5_first_raw_gradients_exact": True,
                         "all3_anchor_predictions_scores_NLL_exact": True, "all16_later_update_dots_nonnegative": True})

    # The EWC/SeqFT equality is at their actual same state; no gradients from a
    # different historical run are transplanted into these EWC trajectories.
    first_equal = {}
    for method in METHODS[2:4]:
        case = cases[method]
        first_equal[method] = torch.equal(case["raw"][1]["pre_intervention_gradient"], seq["raw"][1]["pre_intervention_gradient"])
        assert first_equal[method]
        assert not same(case["states"][1], seq["states"][1]), "EWC later updates must actually change the stage2 endpoint"

    # Use the original latest stage1 Fisher, whose center exactly equals every
    # common stage1 state, as a frozen surrogate evaluated along the SeqFT path.
    # This is not a new EWC model, actual TRAIN loss, or generated-quality score.
    p = BASE / f"T{anchor:02d}_repeat_anchor__EWC_lambda1_complete.json"
    proof = json.loads(p.read_text()); fisher = torch.load(ROOT / proof["run"] / "fishers/stage_01.pt", map_location="cpu", weights_only=True)
    p100 = json.loads((BASE / f"T{anchor:02d}_repeat_anchor__EWC_lambda100_complete.json").read_text())
    other = torch.load(ROOT / p100["run"] / "fishers/stage_01.pt", map_location="cpu", weights_only=True)
    assert fisher.keys() == other.keys()
    assert all(torch.equal(fisher[k][n], other[k][n]) for k in ("mean", "diagonal") for n in fisher[k])
    assert fisher["ids"] == spec["TRAIN_ids"][str(anchor)][:4]
    qs = [quadratic(s, fisher) for s in seq["states"]]
    assert qs[0] == 0 and all(q > 0 for q in qs[1:])
    reference = cases["anchor_gradient_noop"]["raw"][1]
    assert reference["reference_train_ids"] == spec["TRAIN_ids"][str(anchor)]
    old_gradient_norm = float(torch.linalg.vector_norm(reference["reference_gradient"].double()))
    assert math.isfinite(old_gradient_norm) and old_gradient_norm > 0
    images = [original[r["id"]]["images"] for r in seq["endpoints"][5]]
    differences = []
    for method in METHODS[1:]:
        rows = cases[method]["endpoints"][5]; control = seq["endpoints"][5]
        differences.append({"method": method,
                            "F1_vs_SeqFT": paired([a["scores"]["token_f1"]-b["scores"]["token_f1"] for a,b in zip(rows,control)],images),
                            "NLL_vs_SeqFT": paired([a["nll"]-b["nll"] for a,b in zip(rows,control)],images)})
    for path, digest in sources.items():
        assert sha(ROOT/path) == digest, "Input changed during CPU audit"
    assert not torch.cuda.is_initialized()
    result = {"CPU_only": True, "anchor": anchor, "complete_panel_methods": METHODS, "all_same_original8_TRAIN_each_stage": True,
              "common_stage1_states_and_outputs_exact": True, "summaries": summaries, "identity_controls": identity,
              "EWC_stage2_first_raw_gradient_exact_SeqFT": first_equal,
              "EWC_stage2_endpoint_actually_differs_from_SeqFT": True,
              "same_stage1_Fisher_exact_between_lambda1_and100": True,
              "stage1_old8TRAIN_eval_gradient_L2": old_gradient_norm,
              "stage1_Fisher_quadratic_along_SeqFT_states_lambda1": qs,
              "same_center_surrogate_is_not_actual_old_loss": True,
              "final_method_differences": differences, "source_sha256": sources, "self_checks": self_check(),
              "limits": ["Repeated same8QA is a no-domain-switch control, not the actual19-task1313QA CL setting.",
                         "Fresh AdamW and boundary placement are held fixed, not independently ablated.",
                         "The anchor has32QA; paired outputs record actual image clusters. Only SeqFT has full507/228 final.",
                         "NLL is auxiliary, F1 remains the sole primary. F1 intervals touching0 do not establish significant harm.",
                         "Positive displacement penalty is expected algebra, not new mathematics or proof forgetting occurs.",
                         "Old TRAIN gradient is at the shared stage1 state; no own later EWC TRAIN gradients invented.",
                         "Conditional paired image intervals uncorrected; one anchor/order/seed, no general method ranking.",
                         "All six variants included; no TEST selection, code/threshold/data/checkpoint edits or new GPU jobs."]}
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--check", action="store_true"); p.add_argument("--anchor", type=int, default=4, choices=[4,2]); args = p.parse_args()
    torch.set_num_threads(2)
    if args.check:
        print(json.dumps(self_check())); return
    result = build(args.anchor)
    out = BASE / "repeat_control_evidence"; out.mkdir(exist_ok=True)
    path = out / f"T{args.anchor:02d}_EVIDENCE.json"
    temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n"); temp.replace(path)
    lines = ["# 同一任务重复训练：保护是否需要、是否生效", "",
             "六组共同首阶段参数/输出精确相同；原8条训练题重复5阶段，每阶段4更新，共40目标QA曝光。",
             "此处没有跨领域切换；F1唯一主指标，NLL辅助。", "",
             "| 方法 | 锚F1 S1/S3/S5 | 锚NLL S1/S3/S5 | 当前题阶段均值F1 |", "|---|---:|---:|---:|"]
    for row in result["summaries"]:
        f1="/".join(f"{v:.3f}" for v in row["F1_S1_S3_S5"]); nll="/".join(f"{v:.4f}" for v in row["NLL_S1_S3_S5"])
        lines.append(f"| {row['method']} | {f1} | {nll} | {row['current_F1_stage_observation_mean']:.3f} |")
    lines += ["", "## 机制与效果分开", "",
              "- noop/projection全部5个参数状态、首批原始梯度和三次锚评测精确等于SeqFT；16次后续更新点积非负，投影未触发，不是一次有效投影的负结果。",
              "- EWC两系数的stage2首批梯度精确等于SeqFT，但stage2末参数已不同；后续机制确实改变了轨迹，不能说EWC始终不生效。",
              f"- 共同首阶段的旧8TRAIN eval梯度范数为{result['stage1_old8TRAIN_eval_gradient_L2']:.6f}；这不是旧训练损失已经收敛的证据。",
              "- 固定stage1 Fisher的二次距离沿SeqFT轨迹离开0，同时锚NLL下降；该中心距离不能直接解释成实际旧功能损失。正二次距离是已知代数，不是新发现。", "",
              "## 相对SeqFT的最终配对差异", "",
              "| 方法 | F1差/95%图像区间 | NLL差/95%图像区间 |", "|---|---:|---:|"]
    for row in result["final_method_differences"]:
        values=[]
        for key in ["F1_vs_SeqFT", "NLL_vs_SeqFT"]:
            v=row[key]; lo,hi=v["image_cluster_95CI"]; values.append(f"{v['mean']:+.4f} [{lo:+.4f},{hi:+.4f}]")
        lines.append(f"| {row['method']} | {values[0]} | {values[1]} |")
    lines += ["", "EWCλ100的NLL劣于SeqFT，但其F1差区间包含0；不宣布显著F1损害。ER的额外回放也不代表本控制内稳定优于SeqFT。", "",
              "当前只证明：保护未触发与触发后未获主指标收益是不同情况；没有域切换也能出现约束与继续习得的张力。跨域是否进一步加剧，须等同域/跨域完整配对。", "",
              "全部原始SHA、参数等价、曝光及区间见[JSON]("+path.name+")。", ""]
    lines += ["- "+s for s in result["limits"]]
    (out/f"T{args.anchor:02d}_VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"CPU_only":True,"six_method_panel_complete":True,"anchor":args.anchor,"output":str(path.relative_to(ROOT))}))


if __name__ == "__main__":
    main()
