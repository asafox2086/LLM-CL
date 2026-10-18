"""TRAIN-support x initialization x update-budget causal control on GPU1.

All18 trajectories and both1/4-update endpoints are predeclared. Within an
anchor/update budget, exposure count and per-row frequency are exactly matched.
This is an isolated diagnostic, never a replacement for frozen1313/19-stage CL.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
from collections import Counter
import copy
import fcntl
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch
from onevision_full_input_repair import RepairedExperiment
from onevision_learned_reuse_trial import source_hashes
from onevision_knowledge_intervention import reset_seed, case_complete
from onevision_research_worker import atomic_torch, digest
from onevision_capacity_probe import finish_resources
from run import freeze, write

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016"
REPORT = BASE / "cold_data_factorial_20261018"
FAMILY = ROOT / "exp/onevision_cold_data_factorial_20261018"
TASKS = (2, 11, 12)  # The same three previously predeclared budget anchors.
STARTS = ("cold_base", "warm_native_T01")


def select_rows(rows, n):
    order = sorted(rows, key=lambda r: hashlib.sha256(("42:cold-data-TRAIN-subset:" + r["id"]).encode()).digest())
    assert len({r["id"] for r in rows}) == len(rows) and 0 < n <= len(rows)
    return order[:n]


def balanced_exposures(rows, batch):
    assert rows and batch % len(rows) == 0
    return rows * (batch // len(rows))


def same_adapter(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k][v], b[k][v]) for k in a for v in ("A", "B"))


def adapter_hash(a):
    h = hashlib.sha256()
    for key in sorted(a):
        for letter in ("A", "B"):
            t = a[key][letter].detach().cpu().contiguous()
            h.update(json.dumps([key, letter, str(t.dtype), list(t.shape)]).encode())
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def plan():
    manifest, data, _, _, _, _, protocol = freeze()
    path = BASE / "deterministic_native/complete.json"
    proof = json.loads(path.read_text())
    assert proof["complete"]
    native = ROOT / proof["run"]
    for name in ("initial_adapter.pt", "adapters/sequential_T01.pt"):
        assert digest(native / name) == proof["artifact_sha256"][name]
    sources = source_hashes()
    sources[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    cases = []
    for task in TASKS:
        rows = data[task, "train"]
        sizes = (1, 4, len(rows))
        assert len(set(sizes)) == 3 and len(rows) in (6, 11, 12)
        batch = math.lcm(*sizes)
        for n in sizes:
            chosen = select_rows(rows, n)
            for kind in STARTS:
                cases.append({"index": len(cases), "task": task, "start": kind, "unique_TRAIN": n,
                              "TRAIN_ids": [r["id"] for r in chosen], "effective_batch": batch,
                              "exposure_ids_per_update": [r["id"] for r in balanced_exposures(chosen, batch)],
                              "images_unique": len({tuple(r["images"]) for r in chosen}),
                              "endpoint_updates": [1, 4], "endpoint_exposures": [batch, 4 * batch],
                              "TEST_ids": [r["id"] for r in data[task, "test"]]})
    assert len(cases) == 18
    spec = {"source_sha256": sources, "manifest_sha256": protocol["source_manifest_sha256"],
            "model_revision": protocol["model_revision"], "native_completion_sha256": digest(path),
            "initial_state_sha256": proof["artifact_sha256"]["initial_adapter.pt"],
            "warm_state_sha256": proof["artifact_sha256"]["adapters/sequential_T01.pt"],
            "warm_source": "original deterministic native sequential T01; NOT old basis-budget past_AB",
            "historical_natural_TRAIN": len(data[1, "train"]), "seed": 42, "cases": cases,
            "targets": list(TASKS), "starts": list(STARTS), "rank": 8, "alpha": 32,
            "trainable_parameters": 540672, "one_resident_LoRA": True,
            "training": "fresh AdamW lr1e-4 weight_decay0, constant LR four balanced full effective batches; save1/4 prefixes",
            "nativeAnyRes_BF16": True, "formal_V2": False, "official1313_manifest_changed": False,
            "routing_or_new_knowledge": False, "TEST_selects_anything": False,
            "all18_and_both_endpoints_required": True,
            "unique_support_selection": "TRAIN-only seed42 hash nested prefixes1/4/all; no answer/type/domain stratification",
            "evaluation": "initial and both endpoints, all original native TEST228/267/100; no full21454CL metrics",
            "scope": "isolated initialization x uniqueTRAIN-support x optimizer-updates; subset composition and task identity bounded",
            "total_training_QA_forwards": sum(4 * c["effective_batch"] for c in cases),
            "total_target_generation_QA": sum(3 * len(c["TEST_ids"]) for c in cases),
            "additional_TRAIN_probe_forwards_per_case": "all original task TRAIN before/after; no optimizer on held-out support",
            "extra_cost_disclosed": True}
    return spec, (manifest, data, native, proof)


def complete(index, spec):
    path = REPORT / f"case_{index:02d}_complete.json"
    if not path.exists():
        return False
    proof = json.loads(path.read_text())
    assert proof["complete"] and proof["spec"] == spec and proof["case"] == spec["cases"][index]
    run = ROOT / proof["run"]
    assert run.resolve().is_relative_to(FAMILY.resolve())
    for name, expected in proof["artifact_sha256"].items():
        p = run / name
        assert p.resolve().is_relative_to(run.resolve()) and digest(p) == expected
    assert proof["updates"] == 4 and proof["valid_training"] and proof["extra_LoRAs"] == 0
    for label in ("initial", "endpoint_1", "endpoint_4"):
        assert json.loads((run / "evaluations" / label / "durable_complete.json").read_text())["complete"]
    return True


def probe(x, rows):
    x.model.eval()
    with torch.no_grad():
        return [{"id": r["id"], "nll": float(x.loss(r)[0])} for r in rows]


def run_case(index, spec, values):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"
    manifest, data, native, parent = values
    case = spec["cases"][index]
    assert not (REPORT / f"case_{index:02d}_needs_attention.json").exists(), "No blind retry"
    if complete(index, spec):
        return
    lock = (REPORT / f"case_{index:02d}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pointer = REPORT / f"case_{index:02d}_attempt.json"
    # A failed experiment must not silently mix optimizer trajectories. An
    # actual interrupted run is preserved and requires a traceable recovery.
    assert not pointer.exists(), "Existing incomplete trajectory: preserve and inspect; no restart/splice"
    run = FAMILY / f"case_{index:02d}" / time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    write(pointer, {"spec": spec, "case": case, "run": str(run.relative_to(ROOT)), "pid": os.getpid()})
    write(run / "protocol.json", {"spec": spec, "case": case})
    torch.set_num_threads(6)
    torch.use_deterministic_algorithms(True)
    reset_seed()
    x = RepairedExperiment(manifest, run)
    initial = torch.load(native / "initial_adapter.pt", map_location="cpu", weights_only=True)["adapter"]
    assert same_adapter(x.original_initial, initial)
    warm = torch.load(native / "adapters/sequential_T01.pt", map_location="cpu", weights_only=True)["adapter"]
    start = initial if case["start"] == "cold_base" else warm
    x.configure(start, finetune_shared=True)
    assert same_adapter(x.initial, start)
    write(run / "initial_state_gate.json", {"passed": True, "start": case["start"],
          "actual_AB_tensor_sha256": adapter_hash(x.initial), "source_AB_tensor_sha256": adapter_hash(start),
          "parent_completion_sha256": spec["native_completion_sha256"],
          "exact_all48_AB": True, "all_original_model_and_input_sources_bound": True})
    atomic_torch(run / "initial_state.pt", {"adapter": x.initial, "actual_start": case["start"]})
    task = case["task"]
    test = {"native": data[task, "test"]}
    x.evaluate("initial", test, test)
    before_probe = probe(x, data[task, "train"])
    chosen = select_rows(data[task, "train"], case["unique_TRAIN"])
    exposure = balanced_exposures(chosen, case["effective_batch"])
    assert [r["id"] for r in exposure] == case["exposure_ids_per_update"]
    x.cache.clear()
    reset_seed()
    x.model.train()
    named = [(k, p) for k, p in x.model.named_parameters() if p.requires_grad]
    assert sum(p.numel() for _, p in named) == 540672
    parameters = [p for _, p in named]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0)
    torch.cuda.reset_peak_memory_stats()
    before = x.monitor.snapshot("factorial_pure_TRAIN_before", model=x.model)
    events = []
    for update in range(1, 5):
        optimizer.zero_grad(set_to_none=True)
        losses, tokens = [], 0
        for row in exposure:
            loss, details = x.loss(row)
            assert torch.isfinite(loss)
            (loss / len(exposure)).backward()
            losses.append(float(loss.detach()))
            tokens += details["target_tokens"]
        raw = torch.cat([p.grad.detach().float().cpu().flatten() if p.grad is not None else torch.zeros(p.numel()) for _, p in named])
        assert torch.isfinite(raw).all() and raw.norm() > 0
        atomic_torch(run / "raw_gradients" / f"step_{update}.pt", {
            "preclip_gradient": raw, "layout": [{"name": k, "shape": list(p.shape), "count": p.numel()} for k, p in named],
            "TRAIN_exposure_ids": case["exposure_ids_per_update"], "update": update})
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.)
        assert torch.isfinite(gradient) and optimizer.param_groups[0]["lr"] == 1e-4
        optimizer.step()
        weights = x.snapshot()
        assert not same_adapter(start, weights)
        if update in (1, 4):
            atomic_torch(run / f"endpoint_{update}.pt", {"adapter": weights, "updates": update})
            # Save the exact optimizer state with each retained prefix. This
            # does not resume/splice a failed run and is charged to TRAIN cost.
            saved = copy.deepcopy(optimizer.state_dict())
            for state in saved["state"].values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.detach().cpu()
            atomic_torch(run / f"optimizer_{update}.pt", saved)
            del saved
        event = {"update": update, "unique_TRAIN": len(chosen), "cumulative_exposures": update * len(exposure),
                 "per_unique_exposures": update * len(exposure) // len(chosen), "target_tokens": tokens,
                 "loss": sum(losses) / len(losses), "gradient_L2_preclip": float(gradient),
                 "LR": 1e-4, "A_delta_L2": math.sqrt(sum(float((weights[k]["A"].double()-start[k]["A"].double()).square().sum()) for k in start)),
                 "B_delta_L2": math.sqrt(sum(float((weights[k]["B"].double()-start[k]["B"].double()).square().sum()) for k in start))}
        events.append(event)
        write(run / "training.json", {"case": case, "records": events, "constant_LR": True,
                                      "same_optimizer_1_to_4_prefix": True})
        print(json.dumps({"event": "factorial_train", "case_index": index, **event}), flush=True)
    if case["start"] == "cold_base":
        assert events[0]["A_delta_L2"] == 0 and events[0]["B_delta_L2"] > 0
    optimizer.zero_grad(set_to_none=True)
    del optimizer, raw, loss, weights
    gc.collect()
    torch.cuda.empty_cache()
    after = x.monitor.snapshot("factorial_pure_TRAIN_after", model=x.model)
    write(run / "TRAIN_window.json", {"pid": os.getpid(), "before": before, "after": after,
          "cuda_peak_allocated_bytes": after["gpu_peak_allocated_bytes"],
          "cuda_peak_reserved_bytes": after["gpu_peak_reserved_bytes"],
          "scope": "four optimizer steps + raw-gradient/optimizer/parameter persistence; excludes initial/final TEST and TRAIN NLL probes"})
    finish_resources(x, run, "factorial_pure_TRAIN_before", "factorial_pure_TRAIN_after",
                     "same-PID TRAIN before/after; global sampled peaks also include initial TEST; see reset-CUDA TRAIN_window")
    write(run / "TRAIN_probe_NLL.json", {"before": before_probe, "after": probe(x, data[task, "train"]),
          "all_original_task_TRAIN": True, "not_used_for_update_or_choice": True})
    for update in (1, 4):
        state = torch.load(run / f"endpoint_{update}.pt", map_location="cpu", weights_only=True)["adapter"]
        x.configure(state, finetune_shared=False)
        x.evaluate(f"endpoint_{update}", test, test)
        assert case_complete(run, f"endpoint_{update}", test)
    files = [p for p in run.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl", ".pt")]
    write(REPORT / f"case_{index:02d}_complete.json", {"complete": True, "spec": spec, "case": case,
          "run": str(run.relative_to(ROOT)), "valid_training": True, "updates": 4,
          "extra_LoRAs": 0, "artifact_sha256": {str(p.relative_to(run)): digest(p) for p in files}})
    assert complete(index, spec)


def checks(spec):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    tests = []
    for case in spec["cases"]:
        counts = Counter(case["exposure_ids_per_update"])
        tests += [len(counts) == case["unique_TRAIN"], len(set(counts.values())) == 1,
                  len(case["exposure_ids_per_update"]) == case["effective_batch"],
                  set(counts) == set(case["TRAIN_ids"]), len(set(case["TEST_ids"])) == len(case["TEST_ids"]),
                  not set(counts) & set(case["TEST_ids"])]
    for task in TASKS:
        cases = [c for c in spec["cases"] if c["task"] == task]
        tests += [len(cases) == 6, len({c["effective_batch"] for c in cases}) == 1,
                  len({tuple(c["TEST_ids"]) for c in cases}) == 1]
        for n in sorted({c["unique_TRAIN"] for c in cases}):
            a, b = [c for c in cases if c["unique_TRAIN"] == n]
            tests += [a["exposure_ids_per_update"] == b["exposure_ids_per_update"],
                      a["TRAIN_ids"] == b["TRAIN_ids"]]
        prefixes = [c["TRAIN_ids"] for c in cases if c["start"] == "cold_base"]
        tests.append(all(a == b[:len(a)] for a, b in zip(prefixes, prefixes[1:])))
    try:
        balanced_exposures([{}, {}, {}], 16)
    except AssertionError:
        tests.append(True)
    else:
        tests.append(False)
    tests += [spec["total_training_QA_forwards"] == 1632,
              spec["total_target_generation_QA"] == 10710,
              spec["historical_natural_TRAIN"] == 347]
    assert all(tests)
    result = {"passed": True, "CPU_only": True, "checks": len(tests), "GPU_started": False,
              "performance_evidence": False, "actual_GPU_initial_and_update_gates_pending": True, "spec": spec}
    write(REPORT / "CPU_PLAN_CHECK.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "spec"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        spec, values = plan()
        if args.check:
            checks(spec)
        else:
            assert args.case_index is not None and 0 <= args.case_index < 18
            assert json.loads((REPORT / "CPU_PLAN_CHECK.json").read_text())["spec"] == spec
            run_case(args.case_index, spec, values)
    except Exception:
        name = "CPU_check_failure.json" if args.check else f"case_{args.case_index}_needs_attention.json"
        write(REPORT / name, {"pid": os.getpid(), "traceback": traceback.format_exc(), "no_blind_retry": True})
        raise
