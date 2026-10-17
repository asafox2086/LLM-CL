"""Read completed pilots only: initial/1/4-update acquisition across3 fixed tasks."""
import json
import os
from pathlib import Path

from onevision_acquisition_budget_bridge import interval, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016"
REPORT = BASE / "acquisition_budget_bridge"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    from answer_f1 import answer_f1
    import onevision_two_module_queue as validator
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__),
               "analysis/onevision_acquisition_budget_bridge.py": sha(ROOT / "analysis/onevision_acquisition_budget_bridge.py")}

    def checked(path, expected=None):
        digest = sha(path)
        assert expected is None or digest == expected, str(path)
        sources[str(path.relative_to(ROOT))] = digest
        return path

    plan = json.loads(checked(BASE / "two_module_prototype/CPU_PILOT_PLAN.json").read_text())["spec"]
    root_proof = json.loads(checked(BASE / "basis_budget/complete.json").read_text())
    assert root_proof["complete"]
    budget = ROOT / root_proof["run"]
    protocol = json.loads(checked(budget / "protocol.json").read_text())
    assert protocol["past_adapter_sha256"] == plan["past_natural_sha256"]
    assert protocol["seed"] == plan["seed"] == 42
    for relative, digest in {**root_proof["source_sha256"], **plan["source_sha256"]}.items():
        checked(ROOT / relative, digest)
    manifest = json.loads(checked(ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json", plan["manifest_sha256"]).read_text())
    metric = json.loads(checked(ROOT / "summary/primary_metric_20261017/metric_lock.json").read_text())
    checked(ROOT / "analysis/answer_f1.py", metric["metric_sha256"])
    cases, pending = [], []
    for task in (2, 11, 12):
        if not validator.pilot_complete((task, "control"), plan):
            pending.append(task)
            continue
        p = json.loads(checked(BASE / f"two_module_prototype/T{task:02d}_control_complete.json").read_text())
        current = ROOT / p["run"]
        previous = budget / f"T{task:02d}_past_AB"
        b = json.loads(checked(previous / "complete.json").read_text())
        for relative, digest in b["source_sha256"].items():
            checked(ROOT / relative, digest)
        for relative, digest in b["artifact_sha256"].items():
            checked(previous / relative, digest)
        file = manifest["tasks"][task - 1]["files"]["test"]
        original = list(map(json.loads, checked(ROOT / file["path"], file["sha256"]).read_text().splitlines()))
        original_ids = [r["id"] for r in original]

        def rows(directory, proof, name):
            path = checked(directory / name, proof["artifact_sha256"][name])
            values = list(map(json.loads, path.read_text().splitlines()))
            assert [r["id"] for r in values] == original_ids
            for row, raw in zip(values, original):
                assert row["references"] == raw["answers"]
                assert row["scores"]["token_f1"] == answer_f1(row["prediction"], row["references"])
            return values

        one = torch.load(checked(current / "final_adapter.pt", p["artifact_sha256"]["final_adapter.pt"]), map_location="cpu", weights_only=True)["adapter"]
        old = torch.load(previous / "update_1.pt", map_location="cpu", weights_only=True)["adapter"]
        assert one.keys() == old.keys() and len(one) == 48
        assert all(torch.equal(one[k][f], old[k][f]) for k in one for f in ("A", "B"))
        a = rows(current, p, "evaluations/initial/target_image.jsonl")
        c = rows(current, p, "evaluations/final/target_image.jsonl")
        old_one = rows(previous, b, "evaluations/endpoint_1/native_image.jsonl")
        d = rows(previous, b, "evaluations/endpoint_4/native_image.jsonl")
        assert all(x["prediction"] == y["prediction"] and x["nll"] == y["nll"] and x["scores"]["token_f1"] == y["scores"]["token_f1"] for x, y in zip(c, old_one))
        tr = json.loads(checked(previous / "training.json").read_text())
        actual = json.loads(checked(current / "training.json", p["artifact_sha256"]["training.json"]).read_text())
        n = actual["original_target_TRAIN_QA"]
        assert n == len(plan["TRAIN_ids"][str(task)])
        assert protocol["target_train_ids"][str(task)] == plan["TRAIN_ids"][str(task)]
        assert actual["pre_update_loss"] == tr["records"][0]["loss"]
        assert actual["actual_gradient_L2_preclip"] == tr["records"][0]["gradient_l2_preclip"]
        assert all(r["TRAIN_ids"] == tr["records"][0]["TRAIN_ids"] and r["cumulative_QA_exposures"] == n * r["update"] and r["learning_rate_used"] == 1e-4 for r in tr["records"])
        stages = [{"label": label, "QA": len(part), "Answer_F1": sum(r["scores"]["token_f1"] for r in part) / len(part),
                   "mean_NLL": sum(r["nll"] for r in part) / len(part), "target_QA_exposures": exposure}
                  for label, part, exposure in [("initial", a, 0), ("one_update", c, n), ("four_updates", d, n * 4)]]
        contrasts = []
        for label, x, y in [("initial_to_one", a, c), ("one_to_four", c, d), ("initial_to_four", a, d)]:
            change = [v["scores"]["token_f1"] - u["scores"]["token_f1"] for u, v in zip(x, y)]
            images = [raw["images"] for raw in original]
            contrasts.append({"label": label, "F1_difference_pp": interval(change, images),
                              "NLL_difference_nats": interval([v["nll"] - u["nll"] for u, v in zip(x, y)], images),
                              "F1_up_down_same_QA": [sum(v > 0 for v in change), sum(v < 0 for v in change), sum(v == 0 for v in change)]})
        cases.append({"task": task, "TRAIN_unique_QA": n, "stages": stages, "contrasts": contrasts,
                      "first_update_48layer_AB_exact": True, "first_update_all_TEST_outputs_and_NLL_exact": True})
    assert not torch.cuda.is_initialized()
    result = {"CPU_only": True, "completed_control_tasks": len(cases), "cases": cases,
              "pending_control_tasks": pending, "source_sha256": sources,
              "limits": ["Initial-to-one and one-to-four are fixed task/seed interventions, not later19-stage CL or independent-seed replicates.",
                         "Increasing updates increases exposure and compute; no new unique data or parameters, but not a free method improvement.",
                         "F1 point drops with intervals including0 are not proven population harm or equivalence.",
                         "Old natural347TRAIN cost shared; no claim of absence of prior domains/knowledge or unique mechanism.",
                         "All3 tasks predeclared and retained, missing control results stay pending; no TEST-based setting choice.",
                         "F1/NLL and per-question gains/losses differ; no universal cure from more updates."]}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "ALL_TASKS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# 三个固定任务：初始、1步与4步习得", "",
             "新完整pilot控制与旧预算轨迹的首步参数、loss/梯度范数及逐题输出精确桥接；只接受完成证明。当前没有新增GPU训练，不改正式基线或活动源码。", "",
             "| 任务 | 原TRAIN/TEST | 初始F1 | 1步F1 | 4步F1 | 初始→1步pp及95%区间 | 1→4步pp及95%区间 |", "|---|---:|---:|---:|---:|---|---|"]
    for c in cases:
        a, b = c["contrasts"][:2]
        fmt = lambda q: f"{q['mean_difference']:+.3f} [{q['image_cluster_95CI'][0]:+.3f},{q['image_cluster_95CI'][1]:+.3f}]"
        lines.append(f"| T{c['task']:02d} | {c['TRAIN_unique_QA']}/{c['stages'][0]['QA']} | " + " | ".join(f"{p['Answer_F1']:.3f}" for p in c["stages"]) + f" | {fmt(a['F1_difference_pp'])} | {fmt(b['F1_difference_pp'])} |")
    lines += ["", "待完成控制任务：" + str(pending) + "。区间为原图像聚类10000次bootstrap、seed42、固定权重、未校正；所有原始NLL/变好变差题数/曝光与哈希见[ALL_TASKS.json](ALL_TASKS.json)。", "",
              "低分在后续任务之前存在，不能全部解释为后续遗忘；同一事实池/先验和模型下额外优化会影响本任务质量，未证明全部低分因预算或知识充分。1→4步多3倍额外曝光，计算成本单列。对含0的小变化不能宣称显著退化或已证明零效果。", ""]
    (REPORT / "ALL_TASKS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"CPU_only": True, "completed": len(cases), "pending": pending, "cases": cases}))


if __name__ == "__main__":
    main()
