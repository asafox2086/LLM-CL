"""CPU-only activation/effect audit of completed deterministic CL controls.

Read immutable completion proofs; never edits or imports GPU trainers, starts
training, changes choices, or accepts partial trajectories as results.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import hashlib
import json
import math
import time
import traceback
from pathlib import Path

import torch
from answer_f1 import answer_f1
from onevision_first_adam_geometry import residual_factors, residual_inner

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016/matched_task_domain"
OUT = BASE / "mechanism_activation"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def norm(tensor):
    return float(torch.linalg.vector_norm(tensor.double()))


def ewc_terms(adapter, layout, fishers, coefficient):
    total, latest = [], []
    for name, shape in layout:
        module, factor, suffix = name.rsplit(".", 2)
        p = adapter[module][factor].double()
        assert suffix == "weight" and list(p.shape) == shape
        terms = []
        for f in fishers:
            diagonal, center = f["diagonal"][name].double(), f["mean"][name].double()
            assert diagonal.shape == center.shape == p.shape
            assert torch.isfinite(diagonal).all() and torch.isfinite(center).all() and (diagonal >= 0).all()
            terms.append(coefficient * diagonal * (p - center))
        total.append(torch.stack(terms).sum(dim=0).flatten())
        latest.append(terms[-1].flatten())
    return torch.cat(total), torch.cat(latest)


def self_check():
    adapter = {"toy": {"A": torch.ones((1, 2)), "B": torch.full((2, 1), .5)}}
    layout = [(f"toy.{k}.weight", list(v.shape)) for k, v in adapter["toy"].items()]
    latest = {"diagonal": {n: torch.ones(shape) for n, shape in layout},
              "mean": {n: adapter["toy"][n.split(".")[1]].clone() for n, _ in layout}}
    earlier = {"diagonal": latest["diagonal"], "mean": {n: torch.zeros(shape) for n, shape in layout}}
    total, own = ewc_terms(adapter, layout, [latest], 100)
    assert torch.count_nonzero(total) == torch.count_nonzero(own) == 0
    total, own = ewc_terms(adapter, layout, [earlier, latest], 100)
    assert norm(total) > 0 and torch.count_nonzero(own) == 0
    assert not torch.cuda.is_initialized()
    return {"latest_center_zero_does_not_imply_earlier_terms_zero": True,
            "CPU_only_no_optimizer_or_TEST_selection": True}


def build():
    assert not torch.cuda.is_initialized()
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}

    def checked(path, digest=None):
        actual = sha(path)
        assert digest is None or actual == digest, str(path)
        sources[str(path.relative_to(ROOT))] = actual
        return path

    checked(ROOT / "analysis/onevision_first_adam_geometry.py")
    checked(ROOT / "analysis/answer_f1.py")
    plan = json.loads(checked(BASE / "CPU_PLAN_CHECK.json").read_text())
    assert plan["passed"] and plan["GPU_used"] is False
    spec = plan["spec"]
    for name, digest in spec["source_sha256"].items():
        checked(ROOT / name, digest)
    manifest = json.loads(checked(ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json", spec["manifest_sha256"]).read_text())
    original = {}
    for task in manifest["tasks"]:
        f = task["files"]["test"]
        original.update({r["id"]: r for r in map(json.loads, checked(ROOT / f["path"], f["sha256"]).read_text().splitlines())})
    results, raw_by_case = [], {}
    for anchor, sequence, method in spec["cases"]:
        name = f"T{anchor:02d}_{sequence}__{method}"
        proof_path = BASE / f"{name}_complete.json"
        if not proof_path.exists():
            continue
        proof = json.loads(checked(proof_path).read_text())
        assert proof["complete"] and proof["spec"] == spec and proof["stage1_exact"]
        run = (ROOT / proof["run"]).resolve()
        assert run.is_relative_to((ROOT / "exp/onevision_matched_task_domain_20261017").resolve())

        def artifact(relative):
            p = (run / relative).resolve()
            assert p.is_relative_to(run)
            return checked(p, proof["artifact_sha256"][relative])

        def load(relative):
            return torch.load(artifact(relative), map_location="cpu", weights_only=True)

        states = [load(f"adapters/stage_{stage:02d}.pt")["adapter"] for stage in range(1, 6)]
        gate = json.loads(artifact("stage1_exact_gate.json").read_text())
        assert gate["passed"] and gate["no_tolerance"]
        canonical = torch.load(checked(BASE / f"canonical_anchor{anchor}.pt", gate["canonical_sha256"]), map_location="cpu", weights_only=True)
        assert canonical["source_sha256"] == spec["source_sha256"]
        assert states[0].keys() == canonical["adapter"].keys()
        assert all(torch.equal(states[0][n][f], canonical["adapter"][n][f]) for n in states[0] for f in ("A", "B"))

        def scores(label, group, ids):
            rows = [json.loads(line) for line in artifact(f"evaluations/{label}/{group}_image.jsonl").read_text().splitlines()]
            assert [r["id"] for r in rows] == ids
            for row in rows:
                assert row["references"] == original[row["id"]]["answers"]
                assert answer_f1(row["prediction"], row["references"]) == row["scores"]["token_f1"]
            return sum(r["scores"]["token_f1"] for r in rows) / len(rows), rows

        s1, initial_rows = scores("anchor_after1", "anchor", spec["anchor_probe_ids"][str(anchor)])
        s5, _ = scores("anchor_after5", "anchor", spec["anchor_probe_ids"][str(anchor)])
        fishers = [load(f"fishers/stage_{stage:02d}.pt") for stage in range(1, 6)] if method.startswith("EWC_") else []
        stages, plasticity, first = [], [], {}
        for stage, task in enumerate(spec["sequences"][str(anchor)][sequence], 1):
            training = json.loads(artifact(f"training/{name}/stage_{stage:02d}.json").read_text())
            events = training["events"]
            assert len(events) == 4 and [i for e in events for i in e["train_ids"]] == spec["TRAIN_ids"][str(task)]
            assert all(e["lr_used"] == 1e-4 and math.isfinite(e["loss"]) and math.isfinite(e["gradient_l2_preclip"]) for e in events)
            raw = load(f"raw_step_gradients/{name}/stage_{stage:02d}.pt")
            g = raw["pre_intervention_gradient"].double()
            assert g.numel() == 540672 and torch.isfinite(g).all()
            first[stage] = g
            row = {"stage": stage, "task": task, "first_raw_pre_intervention_gradient_L2": norm(g),
                   "target_QA_exposures": sum(len(e["train_ids"]) for e in events),
                   "replay_QA_exposures": sum(len(e["replay_ids"]) for e in events),
                   "Fisher_QA": training["fisher_extra_qa_next"],
                   "reference_gradient_QA_exposures": sum(e["reference_extra_QA"] for e in events),
                   "projection_flagged_updates": sum(e["projection_applied"] for e in events),
                   "negative_reference_dot_updates": sum(e["anchor_gradient_dot"] is not None and e["anchor_gradient_dot"] < 0 for e in events),
                   "logged_positive_penalty_updates": sum(e["penalty"] > 0 for e in events),
                   "all_update_losses": [e["loss"] for e in events],
                   "all_logged_preclip_norms": [e["gradient_l2_preclip"] for e in events]}
            if stage > 1:
                previous, after = states[stage-2], states[stage-1]
                displacement = torch.cat([(after[n][f] - previous[n][f]).double().flatten() for n in previous for f in ("A", "B")])
                assert norm(displacement) > 0
                factors = residual_factors(torch, previous, after)
                row["actual_four_update_parameter_displacement_L2"] = norm(displacement)
                row["actual_four_update_BA_displacement_Frobenius"] = math.sqrt(residual_inner(factors, factors, 4))
                if fishers:
                    total, latest = ewc_terms(previous, raw["parameter_layout"], fishers[:stage-1], 100 if method == "EWC_lambda100" else 1)
                    assert torch.count_nonzero(latest) == 0
                    row["EWC_first_latest_term_gradient_L2"] = norm(latest)
                    row["EWC_first_all_terms_gradient_L2_FP64"] = norm(total)
                    row["EWC_gradient_is_reconstructed_from_own_weights_and_Fisher"] = True
                    if stage == 2:
                        assert torch.count_nonzero(total) == 0
                ref = raw["reference_gradient"]
                if ref is not None:
                    ref = ref.double()
                    row["first_raw_new_old_gradient_cosine"] = float(torch.dot(g, ref) / torch.linalg.vector_norm(g) / torch.linalg.vector_norm(ref))
                    row["reference_gradient_is_own_anchor_TRAIN_not_EWC_gradient"] = True
                p, _ = scores(f"plasticity_after{stage}", "current", spec["plasticity_ids"][str(task)])
                plasticity.append(p)
            stages.append(row)
        result = {"case": name, "anchor": anchor, "sequence": sequence, "method": method, "run": proof["run"],
                  "anchor_F1_stage1": s1, "anchor_F1_stage5": s5, "anchor_F1_change_pp": s5-s1,
                  "current_task_F1_stage_observation_mean": sum(plasticity)/len(plasticity),
                  "current_task_F1_is_not_old_anchor_retention": True, "stages": stages}
        results.append(result)
        raw_by_case[anchor, sequence, method] = first, initial_rows
    pairs = []
    for case in results:
        key = case["anchor"], case["sequence"], "SeqFT_LoRA"
        if case["method"] == "SeqFT_LoRA" or key not in raw_by_case:
            continue
        first, old_rows = raw_by_case[case["anchor"], case["sequence"], case["method"]]
        control, initial = raw_by_case[key]
        assert [(r["id"], r["prediction"], r["nll"], r["scores"]["token_f1"]) for r in old_rows] == [(r["id"], r["prediction"], r["nll"], r["scores"]["token_f1"]) for r in initial]
        first_stage2_equal = torch.equal(first[2], control[2])
        if case["method"].startswith("EWC_"):
            assert first_stage2_equal, "Same-state EWC stage2 first combined gradient differs from SeqFT although penalty gradient0"
        baseline = next(v for v in results if (v["anchor"], v["sequence"], v["method"]) == key)
        pairs.append({"anchor": case["anchor"], "sequence": case["sequence"], "method": case["method"],
                      "stage2_first_raw_gradient_exact_SeqFT": first_stage2_equal,
                      "anchor_F1_difference_vs_SeqFT_pp": case["anchor_F1_stage5"]-baseline["anchor_F1_stage5"],
                      "current_task_F1_difference_vs_SeqFT_pp": case["current_task_F1_stage_observation_mean"]-baseline["current_task_F1_stage_observation_mean"],
                      "same_initial_weights_and_outputs": True})
    for name, digest in sources.items():
        assert sha(ROOT / name) == digest
    assert not torch.cuda.is_initialized()
    result = {"CPU_only": True, "cases": results, "complete_cases": len(results), "planned_cases": 48,
              "method_pairs": pairs, "source_sha256": sources, "self_checks": self_check(),
              "limits": ["Projection counts are logged flags, not a claim that every flag changes final accuracy.",
                         "Only first minibatch raw gradients are stored; other updates have scalar logs, not full raw vectors.",
                         "EWC term reconstructed in FP64 at each run own state; no target-only GPU gradient invented.",
                         "Zero latest penalty does not imply harmful transfer; own first reference cosines may be positive.",
                         "Actual parameter/BA displacement is finite, but its norm does not measure generated learning quantity.",
                         "Two medical anchors and fixed tasks/order/seed; neither pure domain-count nor universal method superiority.",
                         "Current scores are32 stage observations, possibly repeated IDs; anchor test32, not full21454.",
                         "Paired uncertainty and full-anchor SeqFT controls are in the original verified score report; no TEST tuning."]}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "EVIDENCE.json", result)
    lines = ["# 严格任务/领域对照：机制是否激活、激活后是否有效", "",
             f"完整轨迹 {len(results)}/48。只读通过SHA和共同首阶段门槛的完整轨迹；0个完成表示待测。", "",
             "| 锚/序列/机制 | 旧锚F1 S1→S5 | 当前任务F1 | 回放曝光 | 投影标记 | 非零penalty更新 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in results:
        sums = lambda k: sum(s[k] for s in row["stages"])
        lines.append(f"| T{row['anchor']:02d}/{row['sequence']}/{row['method']} | {row['anchor_F1_stage1']:.3f}→{row['anchor_F1_stage5']:.3f} | {row['current_task_F1_stage_observation_mean']:.3f} | {sums('replay_QA_exposures')} | {sums('projection_flagged_updates')} | {sums('logged_positive_penalty_updates')} |")
    lines += ["", "逐阶段EWC最新项/全部项、首批原始新旧梯度、实际四步参数/BA位移、额外TRAIN信息和同SeqFT的旧/新F1差见[EVIDENCE.json](EVIDENCE.json)。", "",
              "机制激活与最终作用分开：正则项首步0不是遗忘因果；投影标记不是成功保护；回放访问额外旧QA，不能称同总计算量。任何一组保旧结果都要与当前学习并列。", "",
              "当前没有足够结果时不写因果结论。配对图像区间、序列比较与SeqFT完整锚测试见[原严格结果](../VERIFIED_RESULTS.md)。", ""]
    (OUT / "VERIFIED_RESULTS.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.watch:
        import fcntl
        lock = (OUT / "observer.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = None
    while True:
        signature = sorted((p.name, p.stat().st_mtime_ns) for p in BASE.glob("T*_complete.json"))
        if signature != previous:
            result = build()
            previous = signature
            write(OUT / "observer_status.json", {"pid": os.getpid(), "CPU_only": True,
                  "phase": "complete" if result["complete_cases"] == 48 else "waiting_completed_GPU_cases",
                  "completed_cases": result["complete_cases"]})
            print(json.dumps({"complete_cases": result["complete_cases"], "CPU_only": True}), flush=True)
            if result["complete_cases"] == 48:
                return
        if not args.watch:
            return
        time.sleep(45)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        write(OUT / "observer_needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc(),
              "CPU_only": True, "GPU_workers_not_modified_or_stopped": True})
        raise
