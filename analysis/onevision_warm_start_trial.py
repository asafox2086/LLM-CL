"""Initialization-only warm inheritance control for the frozen cosine stream.

The immutable original loop is reused through process-local hooks. Allocation,
features, metric learning, IDs, optimizer and budget must match the cold run.
Only a newly allocated expert's A/B initialization can change. No TEST choices.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import copy
import hashlib
import json
from pathlib import Path
import traceback

import torch
import onevision_learned_reuse_trial as original

ROOT, BASE = original.ROOT, original.BASE
REPORT = BASE / "warm_start_20261018"
VARIANT = "warm_inherit_K3_v1"
COLD = "learned_cosine_K3"
COLD_REPORT = original.REPORT
ORIGINAL_PLAN = original.plan
ORIGINAL_ALLOCATE = original.core.allocate
ORIGINAL_FIT = original.core.fit
ORIGINAL_EXPERIMENT = original.MetricExperiment
sha, write = original.digest, original.write


def same_adapter(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k][v], b[k][v]) for k in a for v in ("A", "B"))


def adapter_hash(a):
    h = hashlib.sha256()
    for key in sorted(a):
        for letter in ("A", "B"):
            t = a[key][letter].detach().contiguous().cpu()
            h.update(json.dumps([key, letter, str(t.dtype), list(t.shape)]).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def source_expert(decision):
    """Same TRAIN vote/mean-similarity/key tie break as original allocate()."""
    keys = decision["expert_keys"]
    sim, winners = decision["cosines"], decision["row_winners"]
    assert keys and sim and len(sim) == len(winners)
    return min(range(len(keys)), key=lambda i: (-winners.count(keys[i]),
               -float(torch.tensor(sim)[:, i].mean()), keys[i]))


def select_start(initial, pool, decision, destination):
    if decision["action"] == "new_from_base" and pool:
        source = decision["expert_keys"][source_expert(decision)]
        assert source in pool and destination not in pool
        # A new expert owns its tensors; later training must not alias old pool.
        return copy.deepcopy(pool[source]), source, "inherit_existing_complete_AB"
    if decision["action"] == "new_from_base":
        assert not pool and destination == "e0"
        return initial, None, "untouched_base_first_expert"
    assert destination in pool
    return pool[destination], destination, "reuse_existing_unchanged_policy"


def geometry_equal(a, b):
    if a["stage"] != b["stage"] or a["pool"].keys() != b["pool"].keys():
        return False
    if a["threshold"] != b["threshold"] or not torch.equal(a["weight"], b["weight"]):
        return False
    if len(a["memory"]) != len(b["memory"]):
        return False
    return all(x["id"] == y["id"] and x["expert"] == y["expert"] and
               torch.equal(x["feature"], y["feature"]) for x, y in zip(a["memory"], b["memory"]))


def plan():
    cold, values = ORIGINAL_PLAN()
    # No patch is installed by plan/import. Verify the original sealed run.
    assert original.REPORT == COLD_REPORT
    assert original.verify(cold, COLD, "train")
    path = COLD_REPORT / f"{COLD}_train_complete.json"
    proof = json.loads(path.read_text())
    spec = copy.deepcopy(cold)
    spec.update(variants=[VARIANT], initialization_only_control=True,
                cold_parent_completion_sha256=sha(path), cold_parent_run=proof["run"],
                initialization="new expert inherits complete A+B of TRAIN geometric vote winner; first expert untouched base",
                allocation_unchanged="original learned-cosine K3 decisions, geometry and metric exact at all19 stages",
                inference="unchanged original learned-cosine input router; no quality-head/soft-gain intervention",
                parameter_count_unchanged=True, extra_LM_forwards_for_source_choice=0,
                metric_fit_always_learned=True, full_confirmation_required_independent_of_diagnostic=True,
                new_variant_compatibility="process-local metric-fit flag preserves original learned=True branch",
                run_namespace=f"exp/onevision_learned_reuse_20261018/{VARIANT}",
                warm_quality_factorial="not implemented by this control")
    spec["source_sha256"][str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    return spec, values, proof


def checked_cold(proof, name):
    p = ROOT / proof["run"] / name
    assert sha(p) == proof["artifact_sha256"][name]
    return p


def verify(spec, phase):
    path = REPORT / f"{VARIANT}_{phase}_complete.json"
    if not path.exists():
        return False
    proof = json.loads(path.read_text())
    assert proof["complete"] and proof["spec"] == spec and proof["variant"] == VARIANT and proof["phase"] == phase
    run = ROOT / proof["run"]
    assert run.resolve().is_relative_to((ROOT / spec["run_namespace"] / phase).resolve())
    for name, expected in proof["artifact_sha256"].items():
        p = run / name
        assert p.resolve().is_relative_to(run.resolve()) and sha(p) == expected
    assert proof["train_qa"] == 1313 and proof["stages"] == 19 and proof["experts"] == 3
    if phase == "train":
        for stage in range(1, 20):
            init = json.loads((run / "initialization_gates" / f"T{stage:02d}.json").read_text())
            geom = json.loads((run / "geometry_gates" / f"S{stage:02d}.json").read_text())
            assert init["passed"] and geom["passed"] and init["allocation_exact_cold"]
        assert json.loads((run / "first_task_exact_gate.json").read_text())["passed"]
        for stage in (2, 3):
            init = json.loads((run / "initialization_gates" / f"T{stage:02d}.json").read_text())
            assert init["mode"] == "inherit_existing_complete_AB" and init["different_from_base"]
    else:
        assert proof["full_test_qa"] == 21454 and proof["reload_exact"]
    return True


def install(spec, values, proof, phase):
    """Hooks affect only this new worker process, never files/other processes."""
    pending = {}

    def allocate(*args):
        expert, decision = ORIGINAL_ALLOCATE(*args)
        assert not pending
        pending.update(expert=expert, decision=copy.deepcopy(decision))
        return expert, decision

    def fit(memory, weight, learned):
        # Original loop compares variant name to COLD. Our disjoint namespace
        # makes its argument False; explicitly preserve the learned control.
        assert learned is False
        return ORIGINAL_FIT(memory, weight, True)

    class WarmExperiment(ORIGINAL_EXPERIMENT):
        def set_metric(self, state, live):
            stage = state["stage"]
            if stage:
                path = checked_cold(proof, f"states/reuse_T{stage:02d}.pt")
                ref = torch.load(path, map_location="cpu", weights_only=True)
                assert geometry_equal(state, ref), "Warm initializer changed supposedly frozen geometry/allocation"
                if stage == 1:
                    assert same_adapter(state["pool"]["e0"], ref["pool"]["e0"])
                write(self.run / "geometry_gates" / f"S{stage:02d}.json",
                      {"passed": True, "stage": stage, "cold_state_sha256": sha(path),
                       "projection_threshold_memory_feature_and_pool_keys_exact": True,
                       "pool_AB_may_differ_after_stage1": True, "no_tolerance": True})
            else:
                assert not state["pool"] and not state["memory"] and state["threshold"] == .5
                assert torch.equal(state["weight"], original.core.initialize(spec["metric_input_dim"]))
            super().set_metric(state, live)

        def configure(self, shared, rank=None, finetune_shared=False):
            assert phase == "train" and self.metric is not None and not self.metric_live
            assert rank is None and finetune_shared and pending
            stage, expert, decision = self.metric["stage"] + 1, pending["expert"], pending["decision"]
            cold_path = checked_cold(proof, f"decisions/reuse_T{stage:02d}.json")
            ref = json.loads(cold_path.read_text())
            assert ref["selected_expert"] == expert and ref["train_ids"] == spec["train_ids"][str(stage)]
            assert all(ref[k] == v for k, v in decision.items()), "Original allocation differs from cold paired run"
            expected_argument = self.original_initial if decision["action"] == "new_from_base" else self.metric["pool"][expert]
            assert same_adapter(shared, expected_argument)
            start, source, mode = select_start(self.original_initial, self.metric["pool"], decision, expert)
            expected = adapter_hash(start)
            super().configure(start, rank=rank, finetune_shared=finetune_shared)
            assert same_adapter(self.initial, start) and adapter_hash(self.initial) == expected
            write(self.run / "initialization_gates" / f"T{stage:02d}.json",
                  {"passed": True, "stage": stage, "destination": expert, "source": source, "mode": mode,
                   "allocation_exact_cold": True, "cold_decision_sha256": sha(cold_path),
                   "TRAIN_ids": spec["train_ids"][str(stage)], "initial_AB_sha256": expected,
                   "different_from_base": not same_adapter(start, self.original_initial),
                   "source_selection": "original TRAIN geometric vote/mean/key, no gold/domain/TEST",
                   "extra_LM_forwards": 0, "complete_AB_not_A_only": True,
                   "temporary_CPU_clone_parameters": 540672 if mode == "inherit_existing_complete_AB" else 0})
            pending.clear()

        def train(self, label, rows):
            initial = self.initial
            result = super().train(label, rows)
            assert not same_adapter(initial, result), "No effective update from actual warm initialization"
            return result

    original.REPORT = REPORT
    original.VARIANTS = [VARIANT]
    original.plan = lambda: (spec, values)
    original.MetricExperiment = WarmExperiment
    original.core.allocate = allocate
    original.core.fit = fit


def checks(spec, values, proof):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(1)
    tests = []
    initial = {"layer": {"A": torch.ones(1, 2), "B": torch.zeros(2, 1)}}
    pool = {"e0": {"layer": {"A": torch.full((1, 2), 2.), "B": torch.ones(2, 1)}}}
    first = {"action": "new_from_base"}
    start, source, mode = select_start(initial, {}, first, "e0")
    tests += [same_adapter(start, initial), source is None, mode == "untouched_base_first_expert"]
    decision = {"action": "new_from_base", "expert_keys": ["e0"], "cosines": [[.2]], "row_winners": ["e0"]}
    start, source, mode = select_start(initial, pool, decision, "e1")
    tests += [same_adapter(start, pool["e0"]), source == "e0", mode == "inherit_existing_complete_AB"]
    tests.append(all(start[k][v].data_ptr() != pool["e0"][k][v].data_ptr() for k in start for v in ("A", "B")))
    start["layer"]["B"].zero_()
    tests.append(bool(pool["e0"]["layer"]["B"].eq(1).all()))
    reuse = dict(decision, action="reuse")
    start, source, mode = select_start(initial, pool, reuse, "e0")
    tests += [start is pool["e0"], mode == "reuse_existing_unchanged_policy"]
    decisions = []
    for stage in range(1, 20):
        d = json.loads(checked_cold(proof, f"decisions/reuse_T{stage:02d}.json").read_text())
        source = d["selected_expert"] if d["action"] == "reuse" else (
            None if stage == 1 else d["expert_keys"][source_expert(d)])
        if stage > 1:
            # The new-expert source is exactly the original forced-reuse rule.
            # Check it from the actual recorded cosine matrix, not domain labels.
            sim = torch.tensor(d["cosines"])
            which = sim.max(1).indices
            chosen = min(range(len(d["expert_keys"])), key=lambda i:
                         (-int((which == i).sum()), -float(sim[:, i].mean()), d["expert_keys"][i]))
            tests.append(d["expert_keys"][source_expert(d)] == d["expert_keys"][chosen])
        decisions.append({"stage": stage, "action": d["action"], "expert": d["selected_expert"],
                          "source_if_warm": source, "TRAIN_QA": len(d["train_ids"])})
    tests += [sum(d["TRAIN_QA"] for d in decisions) == 1313,
              [d["stage"] for d in decisions if d["action"] == "new_from_base"] == [1, 2, 3]]
    example = {"stage": 1, "pool": pool, "threshold": .5, "weight": torch.ones(2, 2),
               "memory": [{"id": "a", "expert": "e0", "feature": torch.ones(2)}]}
    tests.append(geometry_equal(example, copy.deepcopy(example)))
    changed = copy.deepcopy(example)
    changed["pool"]["e0"]["layer"]["B"].zero_()
    tests.append(geometry_equal(example, changed))
    changed["weight"][0, 0] = 2
    tests.append(not geometry_equal(example, changed))
    changed = copy.deepcopy(example)
    changed["memory"][0]["feature"][0] = 2
    tests.append(not geometry_equal(example, changed))
    # Real original metric fit can be invoked through the compatibility hook
    # without GPU and must exactly equal learned=True, not fixed=False.
    memory = [{"id": str(i), "expert": f"e{i % 2}", "feature": torch.tensor([1., float(i), 2.])} for i in range(4)]
    w = original.core.initialize(3)
    expected, _ = ORIGINAL_FIT(memory, w, True)
    install(spec, values, proof, "train")
    actual, log = original.core.fit(memory, w, False)
    tests += [torch.equal(actual, expected), log["optimizer_updates"] == 40,
              not torch.equal(actual, w)]
    assert all(tests)
    result = {"passed": True, "CPU_only": True, "checks": len(tests), "GPU_started": False,
              "performance_evidence": False, "decisions_from_actual_cold_TRAIN": decisions,
              "spec": spec, "all19_runtime_geometry_and_initialization_gates_pending": True}
    write(REPORT / "CPU_PLAN_CHECK.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("spec", "decisions_from_actual_cold_TRAIN")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--phase", choices=["train", "full"], default="train")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    spec, values, proof = plan()
    if args.check:
        checks(spec, values, proof)
        return
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    assert json.loads((REPORT / "CPU_PLAN_CHECK.json").read_text())["spec"] == spec
    if verify(spec, args.phase):
        return
    if args.phase == "full":
        assert verify(spec, "train")
    install(spec, values, proof, args.phase)
    original.run_trial(VARIANT, args.phase)
    assert verify(spec, args.phase)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        phase = "full" if "full" in os.sys.argv else "train"
        write(REPORT / f"{VARIANT}_{phase}_needs_attention.json",
              {"pid": os.getpid(), "traceback": traceback.format_exc(), "action": "Preserve evidence; no blind retry"})
        raise
