"""Bounded learned-cosine LoRA reuse trial, separate from all frozen old workers.

Both predeclared variants receive the same native1313 TRAIN/19 stages. No TEST
selects allocation, thresholds, prompts, steps, memory, or final confirmation.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
from collections import OrderedDict
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

import onevision_learned_reuse_core as core
from onevision_full_input_repair import RepairedExperiment
from onevision_capacity_probe import finish_resources, stratify_full_results
from onevision_knowledge_intervention import reset_seed, case_complete, valid_checkpoint
from onevision_research_worker import atomic_torch, digest, ResearchExperiment
from onevision_deterministic_native import checked_events, quarantine_partial_events
from run import Experiment, freeze, hashed, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"summary/onevision_diagnostics_20261016"
REPORT = BASE/"learned_reuse"
VARIANTS = ["fixed_cosine_K3", "learned_cosine_K3"]
CAP = 3


def source_hashes():
    names = ["onevision_learned_reuse_core.py", "onevision_learned_reuse_trial.py",
             "onevision_full_input_repair.py", "onevision_capacity_probe.py", "onevision_capacity_queue.py",
             "onevision_knowledge_intervention.py", "onevision_research_worker.py",
             "onevision_deterministic_native.py", "onevision_matched_native_controls.py",
             "onevision_trained_visual_probe.py"]
    paths = [ROOT/"analysis"/name for name in names]
    paths += list((ROOT/"exp/onevision_diagnostics_20261016").glob("*.py"))
    paths += [ROOT/"exp/five_domain_native_cl_20261016_sparse"/s for s in ["runtime_utils.py", "resource_monitor.py"]]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def id_digest(rows):
    return hashlib.sha256(json.dumps([r["id"] for r in rows], ensure_ascii=False).encode()).hexdigest()


def plan():
    manifest, data, domains, tests, _, _, protocol = freeze()
    native_proof = BASE/"deterministic_native/complete.json"
    parent = json.loads(native_proof.read_text())
    native = ROOT/parent["run"]
    native_spec = json.loads((native/"protocol.json").read_text())
    assert digest(native/"protocol.json") == parent["artifact_sha256"]["protocol.json"]
    train = [r for t in range(1, 20) for r in data[t, "train"]]
    full = {d: [r for t in tasks for r in data[t, "test"]] for d, tasks in domains.items()}
    assert len(train) == 1313 and sum(map(len, full.values())) == 21454
    assert not {r["id"] for r in train} & {r["id"] for rows in full.values() for r in rows}
    config = json.loads((ROOT/"model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json").read_text())
    dim = 2*config["text_config"]["hidden_size"]
    spec = {"variants": VARIANTS, "cap": CAP, "source_sha256": source_hashes(),
            "manifest_sha256": protocol["source_manifest_sha256"], "model_revision": protocol["model_revision"],
            "native_completion_sha256": digest(native_proof), "seed": 42, "stages": 19, "train_qa": len(train),
            "train_ids": {str(t): [r["id"] for r in data[t, "train"]] for t in range(1, 20)},
            "full_test_qa": 21454, "full_ids_sha256": {d: id_digest(r) for d, r in full.items()},
            "diagnostic_ids": {d: [r["id"] for r in rows] for d, rows in tests.items()},
            "generation_ids": native_spec["generation_ids"],
            "metric_input_dim": dim, "metric_output_dim": core.DIM, "metric_parameters": dim*core.DIM,
            "metric_memory_per_expert": core.MEMORY, "metric_steps": core.STEPS, "metric_lr": core.LR,
            "temperature": core.TEMPERATURE, "TRAIN_calibration_new_cost": core.NEW_COST,
            "allocation": "majority TRAIN Q+image novelty; new from untouched base iff >0.5 novel and room; otherwise reuse top-vote expert",
            "router_learning": "TRAIN input-vector expert-assignment CE with leave-one-out positive prototypes; no domain labels",
            "threshold": "TRAIN leave-one-out similarities, reuse-preferring calibration; heuristic not mastery proof",
            "inference": "one input-selected expert, or untouched base below TRAIN threshold; no answer/task/domain routing",
            "training": "rank8 alpha32 q/v dropout0.1 nativeAnyRes BF16, original one epoch batch16 lr1e-4 floor warmup; no model replay",
            "selection_reads_TEST": False, "all_variants_full_confirmation_required": True,
            "inference_control": "exact reload500NLL/155generated before all21454; never splice old predictions",
            "reference": "native cap1 is a sequential reference, not claimed equal total extra compute or formalV2"}
    return spec, (manifest, data, domains, tests, full, native, parent)


class MetricExperiment(RepairedExperiment):
    def __init__(self, manifest, run):
        self.metric_live = False
        self.metric = None
        super().__init__(manifest, run)
        self.feature_cache = OrderedDict()
        self.metric_feature_misses = 0
        self.language_loss_forwards = 0

    def feature(self, row):
        # No instruction, gold, task/domain identity or answer format used here.
        key = hashlib.sha256(json.dumps([row["question"], row["images"]], ensure_ascii=False).encode()).hexdigest()
        if key not in self.feature_cache:
            self.metric_feature_misses += 1
            self.model.eval()
            aux = {"_task": 1, "question": row["question"], "images": row["images"]}
            with torch.no_grad():
                batch = self.inputs(aux)
                ids = self.processor.tokenizer(row["question"].replace("<image>", ""), add_special_tokens=False)["input_ids"]
                assert ids and "_features" in batch
                q = self.model.get_input_embeddings()(torch.tensor([ids], device="cuda"))[0].float().mean(0).cpu()
                v = batch["_features"].float().mean(0).cpu()
                vector = F.normalize(torch.cat([F.normalize(q, dim=0), F.normalize(v, dim=0)]), dim=0)
                assert torch.isfinite(vector).all()
                self.feature_cache[key] = vector.detach()
                while len(self.feature_cache) > 64:
                    self.feature_cache.popitem(last=False)
        self.feature_cache.move_to_end(key)
        return self.feature_cache[key]

    def set_metric(self, state, live):
        self.metric = state
        self.metric_live = live
        self.active_key = None

    def activate_route(self, row):
        if not self.metric_live:
            self.last_route = {"route_mode": "fixed_training_or_probe"}
            return
        state = self.metric
        expert, similarity = core.infer(self.feature(row), state["memory"], state["weight"], state["threshold"])
        key = (state["stage"], expert)
        if self.active_key != key:
            self.restore(self.original_initial if expert is None else state["pool"][expert])
            self.active_key = key
        self.last_route = {"route_mode": "learned_or_fixed_cosine", "route_expert": expert,
                           "route_cosine": similarity, "route_threshold": state["threshold"]}

    def loss(self, row, view="image", wrong=None):
        self.language_loss_forwards += 1
        return super().loss(row, view, wrong)

    def train(self, label, rows):
        # Same optimizer, order, loss, clipping and schedule as frozen run.py.
        # Additional raw-gradient logging does not draw RNG or alter tensors.
        assert not self.metric_live
        self.model.train()
        self.monitor.snapshot(label+"_before", model=self.model)
        named = [(k, p) for k, p in self.model.named_parameters() if p.requires_grad]
        parameters = [p for _, p in named]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
        order = sorted(rows, key=lambda r: hashed(r, "training"))
        self.model.zero_grad(set_to_none=True)
        updates = math.ceil(len(order)/16)
        from transformers import get_cosine_schedule_with_warmup
        warmup = min(int(.03*updates), max(0, updates-1))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, updates)
        for start in range(0, len(order), 16):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            minibatch, losses = order[start:start+16], []
            for row in minibatch:
                loss, details = self.loss(row)
                assert torch.isfinite(loss)
                (loss/len(minibatch)).backward()
                losses.append(float(loss.detach()))
            raw = torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for p in parameters])
            assert torch.isfinite(raw).all() and raw.norm() > 0
            atomic_torch(self.run/"raw_update_gradients"/label/f"step{start//16+1}.pt", {
                "train_ids": [r["id"] for r in minibatch], "preclip_gradient": raw,
                "layout": [{"name": k, "shape": list(p.shape), "count": p.numel()} for k, p in named]})
            grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            assert torch.isfinite(grad)
            used_lr = optimizer.param_groups[0]["lr"]
            assert used_lr > 0
            optimizer.step()
            scheduler.step()
            record = {"event": "train", "label": label, "qa_seen": start+len(minibatch), "total_qa": len(order),
                      "loss": sum(losses)/len(losses), "gradient_l2_preclip": float(grad),
                      "learning_rate_used": used_lr, "warmup_updates": warmup,
                      "seconds": time.monotonic()-tick, "gpu_allocated_bytes": torch.cuda.memory_allocated(), **details}
            with (self.run/"events.jsonl").open("a") as out:
                out.write(json.dumps(record)+"\n")
            print(json.dumps(record), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scheduler, loss, raw
        gc.collect()
        torch.cuda.empty_cache()
        self.monitor.snapshot(label+"_after", model=self.model)
        result = self.snapshot()
        atomic_torch(self.run/"adapters"/(label+".pt"), {"adapter": result})
        return result


def verify(spec, variant, phase):
    path = REPORT/f"{variant}_{phase}_complete.json"
    if not path.exists():
        return False
    p = json.loads(path.read_text())
    assert p["complete"] and p["spec"] == spec and p["variant"] == variant and p["phase"] == phase
    run = ROOT/p["run"]
    assert run.is_relative_to(ROOT/"exp/onevision_learned_reuse_20261018")
    for name, value in p["artifact_sha256"].items():
        q = run/name
        assert q.resolve().is_relative_to(run.resolve()) and digest(q) == value
    assert p["train_qa"] == 1313 and p["stages"] == 19
    if phase == "full":
        assert p["full_test_qa"] == 21454 and p["reload_exact"]
    return True


def run_trial(variant, phase):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1" and variant in VARIANTS and phase in ["train", "full"]
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = (REPORT/(variant+"_worker.lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    reset_seed()
    spec, (manifest, data, domains, tests, full, native, parent) = plan()
    assert json.loads((REPORT/"CPU_PLAN_CHECK.json").read_text())["spec"] == spec
    if verify(spec, variant, phase):
        return
    assert not (REPORT/f"{variant}_{phase}_needs_attention.json").exists(), "No blind retry"
    pointer = REPORT/f"{variant}_{phase}_attempt.json"
    if pointer.exists():
        attempt = json.loads(pointer.read_text())
        assert attempt["spec"] == spec
        run = ROOT/attempt["run"]
    else:
        run = ROOT/"exp/onevision_learned_reuse_20261018"/variant/phase/time.strftime("%Y%m%d_%H%M%S")
        run.mkdir(parents=True)
        write(pointer, {"spec": spec, "run": str(run.relative_to(ROOT))})
        write(run/"protocol.json", spec)
    x = MetricExperiment(manifest, run)
    reference_initial = torch.load(native/"initial_adapter.pt", map_location="cpu", weights_only=True)["adapter"]
    assert all(torch.equal(x.original_initial[k][b], reference_initial[k][b]) for k in reference_initial for b in ["A", "B"])
    generation = {d: [r for r in rows if r["id"] in set(spec["generation_ids"][d])] for d, rows in tests.items()}
    assert sum(map(len, generation.values())) == 155
    train_by_id = {r["id"]: r for t in range(1, 20) for r in data[t, "train"]}

    def evaluate(label, selected, generated, state):
        x.set_metric(state, True)
        if not case_complete(run, label, selected):
            x.evaluate(label, selected, generated)

    def seal(state, extra):
        files = [p for p in run.rglob("*") if p.is_file() and p.suffix in [".json", ".jsonl", ".pt"]
                 and "interrupted" not in p.parts]
        write(REPORT/f"{variant}_{phase}_complete.json", {
            "complete": True, "spec": spec, "variant": variant, "phase": phase, "run": str(run.relative_to(ROOT)),
            "train_qa": 1313, "stages": 19, "experts": len(state["pool"]),
            "artifact_sha256": {str(p.relative_to(run)): digest(p) for p in files}, **extra})

    if phase == "full":
        assert verify(spec, variant, "train")
        old = json.loads((REPORT/f"{variant}_train_complete.json").read_text())
        training_run = ROOT/old["run"]
        state = torch.load(training_run/"latest.pt", map_location="cpu", weights_only=True)
        assert state["stage"] == 19 and state["spec"] == spec
        x.monitor.snapshot("confirmation_before", model=x.model)
        evaluate("reload_diagnostic", tests, generation, state)
        fields = ["id", "native_task", "view", "nll", "references", "prompt_tokens", "target_tokens", "target_truncated",
                  "visual_tokens", "route_mode", "route_expert", "route_cosine", "route_threshold", "prediction", "scores"]
        for d in tests:
            old_rows = [json.loads(s) for s in (training_run/f"evaluations/final_diagnostic/{d}_image.jsonl").read_text().splitlines()]
            new_rows = [json.loads(s) for s in (run/f"evaluations/reload_diagnostic/{d}_image.jsonl").read_text().splitlines()]
            assert [{k: r[k] for k in fields if k in r} for r in old_rows] == [{k: r[k] for k in fields if k in r} for r in new_rows]
        write(run/"reload_exact_gate.json", {"passed": True, "QA": 500, "generated": 155,
                                             "no_tolerance": True, "parent_state_sha256": digest(training_run/"latest.pt")})
        evaluate("final_full_input_router", full, full, state)
        stratify_full_results(run, full, {r["id"] for rows in tests.values() for r in rows})
        x.monitor.snapshot("confirmation_after", model=x.model)
        finish_resources(x, run, "confirmation_before", "confirmation_after", "inference only; no optimizer updates")
        seal(state, {"full_test_qa": 21454, "reload_exact": True, "optimizer_updates": 0,
                     "parent_training_completion_sha256": digest(REPORT/f"{variant}_train_complete.json")})
        return

    # Actual model-feature invariance: metadata/gold never enter the router.
    original = data[1, "train"][0]
    a = x.feature(original).clone()
    x.feature_cache.clear()
    b = x.feature(dict(original, _task=999, domain="fictional", answer="changed", answers=["changed"], context="changed"))
    assert torch.equal(a, b) and a.numel() == spec["metric_input_dim"]
    write(run/"router_input_gate.json", {"passed": True, "QA": original["id"], "gold_task_domain_context_invariance": True})
    latest = run/"latest.pt"
    if valid_checkpoint(latest):
        state = torch.load(latest, map_location="cpu", weights_only=True)
        assert state["spec"] == spec and state["variant"] == variant
    else:
        state = {"stage": 0, "pool": {}, "memory": [], "weight": core.initialize(spec["metric_input_dim"]),
                 "threshold": .5, "spec": spec, "variant": variant}
    x.monitor.snapshot("reuse_training_before", model=x.model)
    for tid in range(state["stage"]+1, 20):
        rows, label = data[tid, "train"], f"reuse_T{tid:02d}"
        x.set_metric(state, False)
        feature_start, forward_start = x.metric_feature_misses, x.language_loss_forwards
        feature_tick = time.monotonic()
        features = torch.stack([x.feature(r) for r in rows])
        feature_seconds = time.monotonic()-feature_tick
        expert, decision = core.allocate(features, state["memory"], state["weight"], state["threshold"], len(state["pool"]), CAP)
        start = x.original_initial if decision["action"] == "new_from_base" else state["pool"][expert]
        x.configure(start, finetune_shared=True)
        current = sorted(rows, key=lambda r: hashed(r, "reuse-current-probe"))[:4]
        old_ids = [m["id"] for m in state["memory"] if m["expert"] == expert][:4]
        old_rows = [train_by_id[i] for i in old_ids]
        def probe(qa):
            x.model.eval()
            with torch.no_grad():
                return [{"id": r["id"], "nll": float(x.loss(r)[0])} for r in qa]
        before = {"current": probe(current), "old": probe(old_rows)}
        candidate = run/"candidates"/(label+".pt")
        if valid_checkpoint(candidate):
            saved = torch.load(candidate, map_location="cpu", weights_only=True)
            assert saved["spec"] == spec and saved["expert"] == expert
            assert checked_events(run, label, len(rows)) == saved["events_sha256"]
            trained = saved["adapter"]
            x.restore(trained)
        else:
            quarantine_partial_events(run, label)
            x.cache.clear()
            reset_seed()
            torch.cuda.reset_peak_memory_stats()
            window_before = x.monitor.snapshot(label+"_pure_TRAIN_before", model=x.model)
            trained = x.train(label, rows)
            window_after = x.monitor.snapshot(label+"_pure_TRAIN_after", model=x.model)
            assert any(not torch.equal(start[k][b], trained[k][b]) for k in trained for b in ["A", "B"])
            write(run/"TRAIN_windows"/(label+".json"), {"pid": os.getpid(), "before": window_before, "after": window_after,
                "cuda_peak_allocated_bytes": window_after["gpu_peak_allocated_bytes"],
                "cuda_peak_reserved_bytes": window_after["gpu_peak_reserved_bytes"],
                "scope": "model optimizer epoch + raw-gradient persistence; no scoring/router-fit/TEST; monitor CPU peaks sampled separately"})
            atomic_torch(candidate, {"spec": spec, "expert": expert, "adapter": trained,
                                    "events_sha256": checked_events(run, label, len(rows))})
        if tid == 1:
            path = native/"adapters/sequential_T01.pt"
            assert digest(path) == parent["artifact_sha256"][str(path.relative_to(native))]
            expected = torch.load(path, map_location="cpu", weights_only=True)["adapter"]
            exact = all(torch.equal(trained[k][b], expected[k][b]) for k in trained for b in ["A", "B"])
            write(run/"first_task_exact_gate.json", {"passed": exact, "native_reference": True, "no_tolerance": True})
            assert exact, "Logging/renderer/training changed native first-task update"
        after = {"current": probe(current), "old": probe(old_rows)}
        ResearchExperiment.gradients(x, label, {"current": current, **({"old": old_rows} if old_rows else {})})
        memory = core.update_memory(state["memory"], rows, features, expert)
        metric_tick = time.monotonic()
        weight, fit_log = core.fit(memory, state["weight"], variant == "learned_cosine_K3")
        if fit_log["optimizer_updates"]:
            atomic_torch(run/"metric_gradients"/(label+".pt"), {
                "first": fit_log.pop("raw_gradient_first"), "last": fit_log.pop("raw_gradient_last"),
                "input_weight": state["weight"], "output_weight": weight,
                "memory_ids": [r["id"] for r in memory], "TRAIN_only": True})
        threshold, calibration = core.calibrate(memory, weight)
        metric_seconds = time.monotonic()-metric_tick
        pool = dict(state["pool"])
        pool[expert] = trained
        state = {"stage": tid, "pool": pool, "memory": memory, "weight": weight, "threshold": threshold,
                 "spec": spec, "variant": variant}
        assert len(pool) <= CAP and len(memory) <= core.MEMORY*CAP
        decision.update(stage=tid, selected_expert=expert, experts=len(pool),
            train_ids=[r["id"] for r in rows], before_nll=before, after_nll=after,
            metric_fit=fit_log, calibration=calibration, next_threshold=threshold,
            extra_feature_forwards=x.metric_feature_misses-feature_start,
            feature_seconds=feature_seconds, metric_fit_calibration_seconds=metric_seconds,
            language_loss_forwards=x.language_loss_forwards-forward_start,
            current_optimizer_qa=len(rows), scoring_controls_do_not_choose_allocation=True,
            retained_lora_parameters=540672*len(pool), metric_parameters=weight.numel(),
            memory_vector_bytes=sum(m["feature"].numel()*4 for m in memory),
            bound_is_not_optimal_expert_count=True)
        write(run/"decisions"/(label+".json"), decision)
        atomic_torch(run/"states"/(label+".pt"), state)
        atomic_torch(latest, state)
        x.monitor.snapshot(label+"_stage_after", model=x.model)
        write(REPORT/(variant+"_status.json"), {"phase": "training", "stage": tid, "pid": os.getpid(),
                                               "experts": len(pool), "run": str(run.relative_to(ROOT))})
        del features
    x.monitor.snapshot("reuse_training_after", model=x.model)
    evaluate("final_diagnostic", tests, generation, state)
    finish_resources(x, run, "reuse_training_before", "reuse_training_after", "TRAIN feature extraction, model epochs, TRAIN probes and CPU router fit; final TEST excluded from before/after, global peak includes diagnostic")
    seal(state, {"full_test_qa": None, "diagnostic_qa": 500, "diagnostic_generated": 155,
                 "full_confirmation_pending": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--phase", choices=["train", "full"], default="train")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    if args.check:
        spec, _ = plan()
        write(REPORT/"CPU_PLAN_CHECK.json", {"passed": True, "CPU_only": True, "spec": spec,
                                            "GPU_started": False, "performance_evidence": False})
        print(json.dumps({"passed": True, "CPU_only": True, "train_qa": 1313, "full_test_qa": 21454,
                          "metric_parameters": spec["metric_parameters"], "variants": VARIANTS}))
        return
    assert args.variant
    try:
        run_trial(args.variant, args.phase)
    except Exception:
        write(REPORT/f"{args.variant}_{args.phase}_needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc(),
              "action": "Preserve evidence, no blind retry or test-driven gate relaxation"})
        raise


if __name__ == "__main__":
    main()
