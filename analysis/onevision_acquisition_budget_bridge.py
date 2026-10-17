"""CPU-only exact bridge from the new pilot to completed budget controls."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016"
REPORT = BASE / "acquisition_budget_bridge"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values, images):
    groups = defaultdict(list)
    assert len(values) == len(images)
    for value, image in zip(values, images):
        groups[tuple(image)].append(value)
    blocks = list(groups.values())
    rng, samples = random.Random(42), []
    for _ in range(10000):
        chosen = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        samples.append(sum(map(sum, chosen)) / sum(map(len, chosen)))
    samples.sort()
    return {"QA": len(values), "recipient_image_clusters": len(blocks),
            "mean_difference": sum(values) / len(values), "image_cluster_95CI": [samples[249], samples[9749]],
            "bootstrap_replicates": 10000, "seed": 42, "multiple_comparison_corrected": False}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    from answer_f1 import answer_f1
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}

    def record(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return path

    pilot_path = record(BASE / "two_module_prototype/T02_control_complete.json")
    pilot = json.loads(pilot_path.read_text())
    assert pilot["complete"] and pilot["valid_training"]
    for relative, digest in pilot["spec"]["source_sha256"].items():
        assert sha(ROOT / relative) == digest
        sources[relative] = digest
    pilot_run = ROOT / pilot["run"]
    assert pilot_run.resolve().is_relative_to((ROOT / "exp/onevision_two_module_20261017/pilot").resolve())
    for relative, digest in pilot["artifact_sha256"].items():
        assert sha(pilot_run / relative) == digest
    budget_root_proof = json.loads(record(BASE / "basis_budget/complete.json").read_text())
    assert budget_root_proof["complete"]
    budget_root = ROOT / budget_root_proof["run"]
    budget_protocol = json.loads(record(budget_root / "protocol.json").read_text())
    assert budget_protocol["past_adapter_sha256"] == pilot["spec"]["past_natural_sha256"]
    assert budget_protocol["manifest_sha256"] == pilot["spec"]["manifest_sha256"]
    assert budget_protocol["seed"] == pilot["spec"]["seed"] == 42
    assert budget_protocol["target_train_ids"]["2"] == pilot["spec"]["TRAIN_ids"]["2"]
    cases = {}
    for task in (2, 11, 12):
        directory = budget_root / f"T{task:02d}_past_AB"
        proof = json.loads(record(directory / "complete.json").read_text())
        for relative, digest in proof["source_sha256"].items():
            assert sha(ROOT / relative) == digest
            sources[relative] = digest
        for relative, digest in proof["artifact_sha256"].items():
            assert sha(directory / relative) == digest
        cases[task] = (directory, proof)
    directory, proof = cases[2]
    one = torch.load(record(directory / "update_1.pt"), map_location="cpu", weights_only=True)["adapter"]
    current = torch.load(record(pilot_run / "final_adapter.pt"), map_location="cpu", weights_only=True)["adapter"]
    assert one.keys() == current.keys() and len(one) == 48
    assert all(torch.equal(one[k][f], current[k][f]) for k in one for f in ("A", "B"))
    training = json.loads(record(directory / "training.json").read_text())
    actual = json.loads(record(pilot_run / "training.json").read_text())
    assert training["records"][0]["loss"] == actual["pre_update_loss"]
    assert training["records"][0]["gradient_l2_preclip"] == actual["actual_gradient_L2_preclip"]
    assert set(training["records"][0]["TRAIN_ids"]) == set(pilot["spec"]["TRAIN_ids"]["2"])
    assert training["constant_learning_rate"] == actual["actual_learning_rate_used"] == 1e-4
    assert training["four_update_exposure_multiplier"] == 4
    assert training["fresh_optimizer_at_trajectory_start"]
    assert [r["update"] for r in training["records"]] == [1, 2, 3, 4]
    assert all(r["TRAIN_ids"] == training["records"][0]["TRAIN_ids"]
               and r["cumulative_QA_exposures"] == 11 * r["update"]
               and r["learning_rate_used"] == 1e-4 for r in training["records"])
    lock = json.loads(record(ROOT / "summary/primary_metric_20261017/metric_lock.json").read_text())
    assert sha(record(ROOT / "analysis/answer_f1.py")) == lock["metric_sha256"]
    manifest = json.loads(record(ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json").read_text())
    original = {}
    for task in manifest["tasks"]:
        f = task["files"]["test"]
        assert sha(record(ROOT / f["path"])) == f["sha256"]
        original.update({r["id"]: r for r in map(json.loads, (ROOT / f["path"]).read_text().splitlines())})

    def rows(path, proof, run):
        assert sha(record(path)) == proof["artifact_sha256"][str(path.relative_to(run))]
        values = list(map(json.loads, path.read_text().splitlines()))
        assert len({row["id"] for row in values}) == len(values)
        for row in values:
            assert row["references"] == original[row["id"]]["answers"]
            if "prediction" in row:
                assert row["scores"]["token_f1"] == answer_f1(row["prediction"], row["references"])
        return values

    initial = rows(pilot_run / "evaluations/initial/target_image.jsonl", pilot, pilot_run)
    final = rows(pilot_run / "evaluations/final/target_image.jsonl", pilot, pilot_run)
    old_one = rows(directory / "evaluations/endpoint_1/native_image.jsonl", proof, directory)
    four = rows(directory / "evaluations/endpoint_4/native_image.jsonl", proof, directory)
    assert len(initial) == len(final) == len(old_one) == len(four) == 228
    assert [r["id"] for r in initial] == [r["id"] for r in final] == [r["id"] for r in old_one] == [r["id"] for r in four]
    assert all(a["nll"] == b["nll"] and a["prediction"] == b["prediction"] and a["scores"]["token_f1"] == b["scores"]["token_f1"]
               for a, b in zip(final, old_one))
    stages = []
    for name, values in [("initial", initial), ("one_update", final), ("four_updates", four)]:
        stages.append({"stage": name, "QA": len(values), "mean_NLL": sum(r["nll"] for r in values) / len(values),
                       "Answer_F1": sum(r["scores"]["token_f1"] for r in values) / len(values)})
    comparisons = []
    for name, before, after in [("initial_to_one", initial, final), ("one_to_four", final, four), ("initial_to_four", initial, four)]:
        changes = [b["scores"]["token_f1"] - a["scores"]["token_f1"] for a, b in zip(before, after)]
        images = [original[r["id"]]["images"] for r in before]
        comparisons.append({"comparison": name, "F1_difference_pp": interval(changes, images),
                            "NLL_difference_nats": interval([b["nll"] - a["nll"] for a, b in zip(before, after)], images),
                            "F1_up_down_same_QA": [sum(v > 0 for v in changes), sum(v < 0 for v in changes), sum(v == 0 for v in changes)]})
    other_budget = []
    for task in (2, 11, 12):
        d, p = cases[task]
        a, b = [rows(d / f"evaluations/endpoint_{step}/native_image.jsonl", p, d) for step in (1, 4)]
        assert [r["id"] for r in a] == [r["id"] for r in b]
        changes = [y["scores"]["token_f1"] - x["scores"]["token_f1"] for x, y in zip(a, b)]
        other_budget.append({"task": task, "one_update_F1": sum(x["scores"]["token_f1"] for x in a) / len(a),
                             "four_update_F1": sum(x["scores"]["token_f1"] for x in b) / len(b),
                             "one_to_four": interval(changes, [original[r["id"]]["images"] for r in a])})
    assert not torch.cuda.is_initialized()
    result = {"CPU_only": True, "actual_48_layer_AB_exact": True, "actual_228_predictions_NLL_F1_exact": True,
              "same_actual_first_update_loss_gradient_norm": True, "target_unique_TRAIN_QA": 11,
              "one_four_QA_exposures": [11, 44], "common_past_natural_TRAIN_QA_cost": 347,
              "target_TRAIN_budget_changed_not_new_data_or_parameters": True,
              "T02_stages": stages, "T02_paired": comparisons, "all_three_past_AB_budget_controls": other_budget,
              "source_sha256": sources, "limits": ["One fixed modality task/order/seed for the new initial-to-one bridge; no universal domain-count or knowledge sufficiency claim.",
              "Four updates mean4passes/44 exposures and extra compute, not a fair1epoch method win or a proposed new algorithm.",
              "Correct answer NLL and free-generation AnswerF1 measure different behavior; point-score decline does not establish population harm if CI includes0.",
              "All three fixed budget tasks reported; rural reverses the direction, so extra updates are not a universal cure.",
              "No later target task was trained in the initial-to-one comparison, but inherited natural training history is shared; not an experiment with no past domain.",
              "Recipient-image bootstrap is fixed-weight and uncorrected; not multi-seed uncertainty or untouched confirmation."]}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "EVIDENCE.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 先学会，还是后来忘掉：一阶段习得与既有预算轨迹的精确衔接", "",
             "新GPU原型control与原完成预算实验在同一Modality任务上的全部48层A/B、首步loss/梯度范数和228题预测/NLL/F1精确一致。只读CPU核验，活动GPU源码与队列不变。", "",
             "| 同一继承起点的阶段 | 完整Modality QA | 答案NLL | Answer F1 | 目标TRAIN曝光 |", "|---|---:|---:|---:|---:|"]
    for r, exposure in zip(stages, [0, 11, 44]):
        lines.append(f"| {r['stage']} | {r['QA']} | {r['mean_NLL']:.4f} | {r['Answer_F1']:.3f} | {exposure} |")
    lines += ["", "| 配对比较 | F1差pp及图像聚类95%区间 | NLL差及区间 | F1上升/下降/持平QA |", "|---|---|---|---|"]
    for p in comparisons:
        a, b = p["F1_difference_pp"], p["NLL_difference_nats"]
        lines.append(f"| {p['comparison']} | {a['mean_difference']:+.3f} [{a['image_cluster_95CI'][0]:+.3f},{a['image_cluster_95CI'][1]:+.3f}] | "
                     f"{b['mean_difference']:+.4f} [{b['image_cluster_95CI'][0]:+.4f},{b['image_cluster_95CI'][1]:+.4f}] | {p['F1_up_down_same_QA']} |")
    lines += ["", "## 同时保留反向任务，不能只挑医学", "",
              "| 原任务/起点pastAB | 1步F1 | 4步F1 | 配对差pp及95%区间 |", "|---|---:|---:|---|"]
    for p in other_budget:
        z = p["one_to_four"]
        lines.append(f"| T{p['task']:02d} | {p['one_update_F1']:.3f} | {p['four_update_F1']:.3f} | {z['mean_difference']:+.3f} [{z['image_cluster_95CI'][0]:+.3f},{z['image_cluster_95CI'][1]:+.3f}] |")
    lines += ["", "## 六栏结论", "",
              "- **探究的问题**：低分是后续跨域遗忘，还是当前极少更新就没形成足够生成收益？",
              "- **做的实验**：固定自然域训练347QA后的旧LoRA和Modality11条TRAIN，新增初始/1步完整228题，并精确衔接同轨迹4步；不增加知识、参数或独特样本。原三任务预算对照全部保留。",
              "- **结果**：首步答案NLL明显下降，但F1点估计29.600→29.263；10题升、32题降、186持平。继续到4步F1达51.956。具体区间见上表，不把含0的小下降说成显著损害。",
              "- **证明了什么**：本例的一步任务质量问题出现在后续任务之前；只讨论保护旧任务不足以解释它，单步NLL改善也不能替代生成质量证据。固定数据/参数下额外优化确能改变此任务结果，但尚未独立识别其具体表示/概率机制，不能否定所有知识或跨域因素。",
              "- **改进的方法**：已完成同数据更多更新的机制控制；不是新CL算法，也未改正式1epoch基线。更多更新的计算/曝光另计，不用测试选最佳步数。",
              "- **改进的效果**：Modality局部收益明显、医学KG也提高，但rural变差，预算不是统一解药。新A/B原型需检查是否实际改动后再解释收益/负结果；原48严格任务/域归因仍必需。", "",
              "[源SHA与逐项区间](EVIDENCE.json)。这是原生单任务机制诊断，不是完整21454方法排名、FWT/BWT或多seed结论。", ""]
    (REPORT / "VERIFIED_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"CPU_only": True, "exact_AB_layers": 48, "exact_QA": 228,
                      "T02_stages": stages, "T02_paired": comparisons}))


if __name__ == "__main__":
    main()
