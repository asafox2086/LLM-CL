"""Bind actual cold-start experts to completed matched initialization controls.

No GPU, optimizer, new predictions or TEST-dependent choices. Full228 warm
control is explicitly not claimed equal to the native sequential checkpoint.
"""
import json
import os
from pathlib import Path

import torch

from answer_f1 import answer_f1
from onevision_acquisition_budget_bridge import interval, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016"


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    trial = read(BASE / "learned_reuse/learned_cosine_K3_train_complete.json")
    native = read(BASE / "deterministic_native/complete.json")
    budget = read(BASE / "basis_budget/complete.json")
    assert trial["complete"] and native["complete"] and budget["complete"]
    tr, nr, br = (ROOT / x["run"] for x in [trial, native, budget])

    def artifact(run, proof, name):
        p = run / name
        assert sha(p) == proof["artifact_sha256"][name], name
        inputs[str(p.relative_to(ROOT))] = sha(p)
        return p

    def adapter(path):
        return torch.load(path, map_location="cpu", weights_only=True)["adapter"]

    def equal(a, b):
        return a.keys() == b.keys() and all(torch.equal(a[k][v], b[k][v]) for k in a for v in ("A", "B"))

    allocation, counts, stages = [], {}, {}
    for task in range(1, 20):
        p = artifact(tr, trial, f"decisions/reuse_T{task:02d}.json")
        d = json.loads(p.read_text())
        expert, n = d["selected_expert"], len(d["train_ids"])
        counts[expert] = counts.get(expert, 0) + n
        stages.setdefault(expert, []).append(task)
        allocation.append({"stage": task, "expert": expert, "TRAIN_QA": n, "action": d["action"]})
    assert sum(counts.values()) == 1313 and stages["e1"] == [2]
    trained = {}
    bridges = []
    for task in (1, 2, 3):
        trained[task] = adapter(artifact(tr, trial, f"candidates/reuse_T{task:02d}.pt"))
        isolated = adapter(artifact(nr, native, f"adapters/matched_native_T{task:02d}.pt"))
        sequential = adapter(artifact(nr, native, f"adapters/sequential_T{task:02d}.pt"))
        exact_isolated, exact_sequential = equal(trained[task], isolated), equal(trained[task], sequential)
        assert exact_isolated and (exact_sequential if task == 1 else not exact_sequential)
        bridges.append({"stage": task, "new_expert_exact_native_isolated": exact_isolated,
                        "new_expert_exact_native_sequential": exact_sequential})
    unchanged = []
    for task in range(2, 20):
        state_path = artifact(tr, trial, f"states/reuse_T{task:02d}.pt")
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        assert equal(state["pool"]["e1"], trained[2])
        unchanged.append(task)
    events = [json.loads(x) for x in artifact(tr, trial, "events.jsonl").read_text().splitlines()]
    e1_events = [x for x in events if x["label"] == "reuse_T02"]
    assert len(e1_events) == 1 and e1_events[0]["qa_seen"] == 11 and e1_events[0]["learning_rate_used"] == 1e-4
    manifest_path = ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    manifest = read(manifest_path)
    assert sha(manifest_path) == trial["spec"]["manifest_sha256"]
    info = manifest["tasks"][1]["files"]["test"]
    path = ROOT / info["path"]
    assert sha(path) == info["sha256"]
    inputs[str(path.relative_to(ROOT))] = sha(path)
    original = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    assert len(original) == 228
    controls, full_rows = [], {}
    for kind in ("random_A_zero_B", "past_A_zero_B", "past_AB"):
        run = br / ("T02_" + kind)
        proof = read(run / "complete.json")
        for name, expected in proof["source_sha256"].items():
            assert sha(ROOT / name) == expected
        checkpoint = artifact(run, proof, "update_1.pt")
        weights = adapter(checkpoint)
        if kind == "random_A_zero_B":
            assert equal(weights, trained[2]), "New e1 must exactly match the full228 fresh control"
        rows_path = artifact(run, proof, "evaluations/endpoint_1/native_image.jsonl")
        rows = list(map(json.loads, rows_path.read_text().splitlines()))
        assert [r["id"] for r in rows] == list(original)
        for row in rows:
            assert row["references"] == original[row["id"]]["answers"]
            assert row["scores"]["token_f1"] == answer_f1(row["prediction"], row["references"])
        training = read(artifact(run, proof, "training.json"))
        first = training["records"][0]
        assert first["update"] == 1 and len(first["TRAIN_ids"]) == first["cumulative_QA_exposures"] == 11
        assert first["learning_rate_used"] == 1e-4
        full_rows[kind] = rows
        controls.append({"start": kind, "QA": len(rows), "TRAIN_QA": 11, "updates": 1,
                         "Answer_F1": sum(r["scores"]["token_f1"] for r in rows) / len(rows),
                         "NLL": sum(r["nll"] for r in rows) / len(rows),
                         "exact_actual_e1_checkpoint": equal(weights, trained[2]),
                         "exact_native_sequential_checkpoint": equal(weights, adapter(artifact(nr, native, "adapters/sequential_T02.pt")))})
    native_rows = [json.loads(x) for x in artifact(nr, native, "evaluations/matched_native_T02/medical_image.jsonl").read_text().splitlines()]
    fresh = {r["id"]: r for r in full_rows["random_A_zero_B"]}
    overlap = [x for x in native_rows if x["id"] in fresh]
    assert len(overlap) == 11
    for row in overlap:
        other = fresh[row["id"]]
        assert row["nll"] == other["nll"]
        if "scores" in row:
            assert row["prediction"] == other["prediction"] and row["scores"]["token_f1"] == other["scores"]["token_f1"]
    left, right = full_rows["random_A_zero_B"], full_rows["past_AB"]
    contrast = interval([b["scores"]["token_f1"] - a["scores"]["token_f1"] for a, b in zip(left, right)],
                        [original[r["id"]]["images"] for r in left])
    result = {"CPU_only": True, "new_GPU_operations": 0, "TEST_selected_cases_or_parameters": False,
              "expert_TRAIN_QA": counts, "expert_stages": stages, "actual_allocation": allocation,
              "initial_new_expert_bridges": bridges, "e1_exact_first_state_at_stages": unchanged,
              "e1_actual_training_event": e1_events, "full228_controls": controls,
              "fresh_e1_native_full_overlap_exact_NLL": len(overlap),
              "full228_past_AB_minus_fresh": contrast, "input_sha256": inputs,
              "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
              "limits": ["Actual e1 is unchanged after its first stage; this excludes later parameter overwriting for that expert only.",
                         "Warm control is NOT bitwise the native sequential checkpoint; never splice its full scores into a native CL main table.",
                         "11 target TRAIN/one update matched; inherited natural347 TRAIN history is disclosed and not free.",
                         "The overall pool already trained natural347, but the new expert skipped its parameters; no full warm-started-pool intervention is measured.",
                         "Prior output behavior, knowledge, visual alignment and nonzero-B optimization differ together; not isolated professional-knowledge causality.",
                         "Single task/seed; no proof all methods, all domains or the final6.659 AP gap share this cause."]}
    out = BASE / "expert_cold_start_20261018"
    out.mkdir(exist_ok=True)
    (out / "EVIDENCE.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: result[k] for k in ["expert_TRAIN_QA", "expert_stages", "initial_new_expert_bridges", "full228_controls", "full228_past_AB_minus_fresh"]}))


if __name__ == "__main__":
    main()
