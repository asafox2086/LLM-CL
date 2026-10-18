"""Warm-start pool x TRAIN quality calibration; no old worker is changed.

No LLM/LoRA parameter training. Only16 CPU head parameters learn from TRAIN
NLL at the actual retained memory (at most48 vectors). TEST never chooses a head.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import fcntl
import json
from pathlib import Path
import time
import traceback

import torch
from torch import nn
import torch.nn.functional as F

import onevision_functional_router_core as core
import onevision_learned_reuse_core as metric
import onevision_learned_reuse_trial as previous
import onevision_warm_start_trial as warm
from onevision_research_worker import atomic_torch, digest
from onevision_knowledge_intervention import reset_seed, case_complete, valid_checkpoint
from run import freeze, write

ROOT = previous.ROOT
REPORT = previous.BASE/"warm_quality_20261018"
PARENT = warm.VARIANT


def plan():
    old, values, _ = warm.plan()
    assert warm.verify(old, "train"), "Wait for sealed warm19; no GPU work before dependency"
    proof_path = warm.REPORT/f"{PARENT}_train_complete.json"
    proof = json.loads(proof_path.read_text())
    parent = ROOT/proof["run"]
    state = torch.load(parent/"latest.pt", map_location="cpu", weights_only=True)
    assert state["stage"] == 19 and len(state["pool"]) == 3 and 3 <= len(state["memory"]) <= 48
    train_ids = {i for ids in old["train_ids"].values() for i in ids}
    assert {r["id"] for r in state["memory"]} <= train_ids
    sources = dict(old["source_sha256"])
    for name in ["onevision_functional_router_core.py", "onevision_functional_router_probe.py",
                 "onevision_warm_quality_probe.py"]:
        sources["analysis/"+name] = digest(ROOT/"analysis"/name)
    spec = {"source_sha256": sources, "parent_variant": PARENT,
            "parent_completion_sha256": digest(proof_path), "parent_state_sha256": digest(parent/"latest.pt"),
            "parent_spec": old, "views": core.VIEWS, "calibration_TRAIN_ids": [r["id"] for r in state["memory"]],
            "head_parameters": 16, "head_steps": core.STEPS, "head_lr": core.LR, "head_decay": core.DECAY,
            "head_target": "softmax(-original TRAIN NLL) over base+three experts; no task/domain pseudo-label",
            "LoRA_optimizer_updates": 0, "additional_LoRAs": 0, "all_four_diagnostic_views_required": True,
            "full_candidates": ["quality_hard"], "all_full_candidates_required": True,
            "factorial_cell": "inherited_complete_AB + TRAIN_quality_hard",
            "LoRA_training_reused_exact_warm19": True,
            "no_TEST_view_selection": True,
            "run_namespace": "exp/onevision_warm_quality_20261018",
            "new_TRAIN_NLL_forwards": 4*len(state["memory"]), "TEST_used_for_calibration_or_choice": False,
            "not_original_published_method": "Controlled lightweight adaptation of D-MoLE recognition and MVP aggregation principles"}
    return spec, values, state, parent


class ScaledLoRA(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.weight, self.bias, self.A, self.B, self.drop = old.weight, old.bias, old.A, old.B, old.drop
        self.enabled, self.gain = old.enabled, 1.

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if not self.enabled or self.gain == 0:
            return y
        with torch.autocast("cuda", enabled=False):
            delta = 4*self.B(self.A(self.drop(x.float())))
            if self.gain != 1.:
                delta = delta*self.gain
        return y+delta.to(y.dtype)


class Probe(previous.MetricExperiment):
    def __init__(self, manifest, run):
        self.view = "original_hard"
        self.head = None
        self.last_gain = None
        super().__init__(manifest, run)
        for key, old in list(self.layers.items()):
            block = ScaledLoRA(old)
            parent, name = key.rsplit(".", 1)
            setattr(self.model.get_submodule(parent), name, block)
            self.layers[key] = block
        for p in self.model.parameters():
            p.requires_grad = False

    def gain(self, value):
        if self.last_gain != value:
            for layer in self.layers.values():
                layer.gain = value
            self.last_gain = value

    def activate_route(self, row):
        if not self.metric_live:
            self.gain(1.)
            return
        state = self.metric
        keys, scores = metric.similarities(self.feature(row)[None], state["memory"], state["weight"])
        index, gain, probs = core.choose(self.view, scores[0], self.head, state["threshold"])
        expert = keys[index] if index is not None else None
        cache_key = (state["stage"], expert)
        if self.active_key != cache_key:
            self.restore(self.original_initial if expert is None else state["pool"][expert])
            self.active_key = cache_key
        self.gain(gain)
        # Original fields match parent exactly for the original_hard reload gate.
        self.last_route = {"route_mode": "learned_or_fixed_cosine", "route_expert": expert,
                           "route_cosine": float(scores[0].max()), "route_threshold": state["threshold"]}
        if self.view != "original_hard":
            self.last_route.update(route_policy=self.view, route_residual_gain=gain, route_quality_probabilities=probs)


def artifacts_complete(scope, view, spec):
    name = "pilot" if scope == "pilot" else "full_"+view
    path = REPORT/(name+"_complete.json")
    if not path.exists():
        return False
    p = json.loads(path.read_text())
    assert p["complete"] and p["spec"] == spec and p["LoRA_optimizer_updates"] == 0
    run = ROOT/p["run"]
    assert run.resolve().is_relative_to((ROOT/"exp/onevision_warm_quality_20261018").resolve())
    for name, expected in p["artifact_sha256"].items():
        f = run/name
        assert f.resolve().is_relative_to(run.resolve()) and digest(f) == expected
    assert json.loads((run/"original_exact_gate.json").read_text())["passed"]
    if scope == "full":
        assert p["full_QA"] == 21454 and json.loads((run/"candidate_reload_exact.json").read_text())["passed"]
    return True


def semantic(rows):
    return [{k: v for k, v in r.items() if k != "generation_seconds"} for r in rows]


def readrows(run, label, domain):
    return [json.loads(s) for s in (run/f"evaluations/{label}/{domain}_image.jsonl").read_text().splitlines()]


def run_probe(scope, view=None):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = (REPORT/"worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    reset_seed()
    spec, (manifest, data, domains, tests, full, native, parent_proof), state, parent = plan()
    assert json.loads((REPORT/"CPU_PLAN_CHECK.json").read_text())["spec"] == spec
    assert scope == "pilot" or view in spec["full_candidates"]
    if artifacts_complete(scope, view, spec):
        return
    name = "pilot" if scope == "pilot" else "full_"+view
    assert not (REPORT/(name+"_needs_attention.json")).exists(), "No blind retry"
    pointer = REPORT/(name+"_attempt.json")
    if pointer.exists():
        v = json.loads(pointer.read_text())
        assert v["spec"] == spec
        run = ROOT/v["run"]
    else:
        run = ROOT/"exp/onevision_warm_quality_20261018"/name/time.strftime("%Y%m%d_%H%M%S")
        run.mkdir(parents=True)
        write(pointer, {"spec": spec, "run": str(run.relative_to(ROOT))})
        write(run/"protocol.json", spec)
    x = Probe(manifest, run)
    generation = {d: [r for r in rows if r["id"] in set(spec["parent_spec"]["generation_ids"][d])] for d, rows in tests.items()}
    train = {r["id"]: r for t in range(1, 20) for r in data[t, "train"]}

    def evaluate(label, selected, generated, condition):
        x.view = condition
        x.set_metric(state, True)
        if not case_complete(run, label, selected):
            x.evaluate(label, selected, generated)

    x.monitor.snapshot("functional_inference_before", model=x.model)
    # This gate precedes all interventions. Gain1 must reproduce the entire
    # original500 NLL/155 predictions/ref/F1, not merely approximate a toy layer.
    evaluate("original_hard", tests, generation, "original_hard")
    for d in tests:
        assert semantic(readrows(run, "original_hard", d)) == semantic(readrows(parent, "final_diagnostic", d))
    write(run/"original_exact_gate.json", {"passed": True, "QA": 500, "generated": 155, "gain": 1., "no_tolerance": True})
    calibration = run/"calibration.pt"
    if scope == "pilot":
        if valid_checkpoint(calibration):
            saved = torch.load(calibration, map_location="cpu", weights_only=True)
            assert saved["spec"] == spec
        else:
            rows = [train[i] for i in spec["calibration_TRAIN_ids"]]
            x.set_metric(state, False)
            keys = sorted(state["pool"])
            losses, cosine_rows = [], []
            tick = time.monotonic()
            for row, memory in zip(rows, state["memory"]):
                f = x.feature(row)
                assert torch.equal(f, memory["feature"]), "Frozen TRAIN features must reproduce exactly"
                _, cos = metric.similarities(f[None], state["memory"], state["weight"])
                cosine_rows.append(cos[0])
                observed = []
                for expert in [None, *keys]:
                    x.restore(x.original_initial if expert is None else state["pool"][expert])
                    x.model.eval()
                    with torch.no_grad():
                        loss, _ = x.loss(row)
                    assert torch.isfinite(loss)
                    observed.append(float(loss))
                losses.append(observed)
            cos, nll = torch.stack(cosine_rows), torch.tensor(losses)
            x.head, logs = core.fit(cos, nll)
            saved = {"spec": spec, "weight": x.head, "cosines": cos, "TRAIN_NLL": nll, "fit": logs,
                     "TRAIN_ids": spec["calibration_TRAIN_ids"], "experts": keys, "seconds": time.monotonic()-tick,
                     "actual_LM_TRAIN_NLL_forwards": len(rows)*4, "TEST_used": False}
            atomic_torch(calibration, saved)
            write(run/"calibration_summary.json", {"parameters": x.head.numel(), "actual_changed": logs["changed"],
                "TRAIN_QA": len(rows), "NLL_forwards": len(rows)*4, "seconds": saved["seconds"],
                "CPU_head_steps": len(logs["events"]), "loss_first": logs["events"][0]["loss"], "loss_last": logs["events"][-1]["loss"],
                "calibration_sha256": digest(calibration), "sealed_before_intervention_TEST": True})
        x.head = saved["weight"]
        for condition in core.VIEWS[1:]:
            evaluate(condition, tests, generation, condition)
    else:
        assert artifacts_complete("pilot", None, spec)
        p = json.loads((REPORT/"pilot_complete.json").read_text())
        pilot = ROOT/p["run"]
        saved = torch.load(pilot/"calibration.pt", map_location="cpu", weights_only=True)
        assert saved["spec"] == spec
        x.head = saved["weight"]
        evaluate("candidate_reload", tests, generation, view)
        for d in tests:
            assert semantic(readrows(run, "candidate_reload", d)) == semantic(readrows(pilot, view, d))
        write(run/"candidate_reload_exact.json", {"passed": True, "no_tolerance": True, "calibration_sha256": digest(pilot/"calibration.pt")})
        evaluate("final_full_input_router", full, full, view)
        previous.stratify_full_results(run, full, {r["id"] for rows in tests.values() for r in rows})
    x.monitor.snapshot("functional_inference_after", model=x.model)
    previous.finish_resources(x, run, "functional_inference_before", "functional_inference_after", "inference + TRAIN NLL calibration and16parameter CPU fit; no LoRA optimizer")
    files = [p for p in run.rglob("*") if p.is_file() and p.suffix in [".json", ".jsonl", ".pt"]]
    write(REPORT/(name+"_complete.json"), {"complete": True, "spec": spec, "scope": scope, "view": view,
        "run": str(run.relative_to(ROOT)), "LoRA_optimizer_updates": 0, "extra_LoRAs": 0,
        "full_QA": 21454 if scope == "full" else None, "all_views": core.VIEWS if scope == "pilot" else [view],
        "parent_state_sha256": digest(parent/"latest.pt"), "artifact_sha256": {str(p.relative_to(run)): digest(p) for p in files}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--view", choices=["quality_hard"])
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.check:
        spec, *_ = plan()
        write(REPORT/"CPU_PLAN_CHECK.json", {"passed": True, "CPU_only": True, "GPU_started": False, "spec": spec})
        print(json.dumps({"passed": True, "CPU_only": True, "head_parameters": 16, "LoRA_updates": 0, "TRAIN_NLL_forwards": spec["new_TRAIN_NLL_forwards"]}))
    else:
        try:
            run_probe(args.scope, args.view)
        except Exception:
            name = "pilot" if args.scope == "pilot" else "full_"+str(args.view)
            write(REPORT/(name+"_needs_attention.json"), {"pid": os.getpid(), "traceback": traceback.format_exc()})
            raise
