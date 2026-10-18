"""CPU-only admission for a fixed-pool lightweight literature adaptation.

Every prior job survives. Foreign GPU1 occupancy always blocks admission.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import onevision_learned_reuse_queue as previous

original = previous.original
ROOT, BASE = previous.ROOT, previous.BASE
REPORT = BASE/"functional_router_queue"
STUDY = BASE/"functional_router"
FULL = ["quality_hard", "quality_soft"]
ORDER = ["ready_sharedS1", "original_cosine_train2", "fixed_pool_functional_pilot4views",
         "languagecue4", "matched48_SeqFT_first", "original_cosine_full2", "functional_full2", "all_original_remaining_jobs"]
sha, write = previous.sha, previous.write


def functional_plan():
    g = json.loads((STUDY/"CPU_PLAN_CHECK.json").read_text())
    c = json.loads((STUDY/"CPU_MECHANICS.json").read_text())
    assert g["passed"] and g["CPU_only"] and c["passed"] and c["CPU_only"]
    s = g["spec"]
    assert s["head_parameters"] == 16 and s["LoRA_optimizer_updates"] == 0 and s["additional_LoRAs"] == 0
    assert not s["TEST_used_for_calibration_or_choice"] and s["full_candidates"] == FULL
    for name, expected in s["source_sha256"].items():
        assert sha(ROOT/name) == expected
    for name, expected in c["source_sha256"].items():
        assert sha(ROOT/name) == expected
    parent = BASE/"learned_reuse/learned_cosine_K3_train_complete.json"
    assert sha(parent) == s["parent_completion_sha256"]
    return s


def complete(scope, view, spec):
    p = STUDY/("pilot_complete.json" if scope == "pilot" else "full_"+view+"_complete.json")
    if not p.exists():
        return False
    v = json.loads(p.read_text())
    assert v["complete"] and v["spec"] == spec and v["LoRA_optimizer_updates"] == 0 and v["extra_LoRAs"] == 0
    run = original.verify_artifacts(v, "exp/onevision_functional_router_20261018")
    assert json.loads((run/"original_exact_gate.json").read_text())["passed"]
    if scope == "pilot":
        assert v["all_views"] == spec["views"]
        c = json.loads((run/"calibration_summary.json").read_text())
        assert c["parameters"] == 16 and c["TRAIN_QA"] == len(spec["calibration_TRAIN_ids"])
        assert c["NLL_forwards"] == spec["new_TRAIN_NLL_forwards"]
    else:
        assert v["full_QA"] == 21454 and json.loads((run/"candidate_reload_exact.json").read_text())["passed"]
    return True


def policy(action, old_train, old_full, pilot, full):
    if action[0] in ["wait", "acquired_anchor"] or old_train != 2:
        return action
    if not pilot:
        return "functional_pilot", None
    if old_full == 2:
        for i, done in enumerate(full):
            if not done:
                return "functional_full", i
    return action


def next_action(specs, shared, cue, trial, functional):
    assert not list(STUDY.glob("*needs_attention.json")), "Preserve failure; no blind retry"
    inherited = REPORT/"INHERITED_WORKER.json"
    if inherited.exists() and not (REPORT/"INHERITED_COMPLETE.json").exists():
        old = json.loads(inherited.read_text())
        stat = Path("/proc")/str(old["worker_pid"])/"stat"
        if stat.exists() and int(stat.read_text().rsplit(")", 1)[1].split()[19]) == old["worker_startticks"]:
            return ("wait", None), {"inherited_worker": old, "waiting_for_natural_completion": True}
        # The previous CPU owner may retire while its unchanged GPU worker
        # finishes. A missing completion proof is a failure, never a retry.
        assert old["kind"] == "matched"
        assert original.previous.prior.complete_case(specs[0]["cases"][old["target"]], specs[0]), "Inherited worker ended without valid proof"
        write(REPORT/"INHERITED_COMPLETE.json", {"verified": True, "original_worker": old,
                                               "no_training_or_prediction_splice": True})
    # While a foreign process occupies GPU1, do not repeatedly rehash hundreds
    # of MB of immutable old gradients/checkpoints merely to report waiting.
    # All old SHA gates still run before any actual GPU admission when idle.
    owners = subprocess.check_output(["nvidia-smi", "--id=1", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    used, util = map(int, subprocess.check_output(["nvidia-smi", "--id=1", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).split(","))
    if owners or used >= 500 or util >= 10:
        return ("wait", None), {"GPU1_PIDs": owners, "used_MiB": used, "utilization": util,
                               "completion_rehash_deferred_until_idle": True, "no_GPU_admitted": True}
    action, state = previous.next_action(specs, shared, cue, trial)
    pilot = complete("pilot", None, functional)
    full = [complete("full", v, functional) for v in FULL]
    state.update(functional_pilot_complete=pilot, functional_full_completed=sum(full))
    return policy(action, state.get("cosine_train_completed", 0), state.get("cosine_full_completed", 0), pilot, full), state


def command_for(kind, target, specs):
    if kind == "functional_pilot":
        return ["analysis/onevision_functional_router_probe.py", "--scope", "pilot"], STUDY/"pilot_nohup.log"
    if kind == "functional_full":
        view = FULL[target]
        return ["analysis/onevision_functional_router_probe.py", "--scope", "full", "--view", view], STUDY/("full_"+view+"_nohup.log")
    return previous.command_for(kind, target, specs)


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    original.REPORT, original.ORDER = REPORT, ORDER
    original.choose = previous.previous.previous.prior.choose
    specs = original.plans()
    shared = previous.previous.previous.shared_plan()
    cue = previous.previous.cue_plan()
    trial, functional = previous.trial_plan(), functional_plan()
    if check:
        tests = [policy(("wait", None), 2, 0, False, [False]*2) == ("wait", None),
                 policy(("acquired_anchor", 2), 2, 0, False, [False]*2) == ("acquired_anchor", 2),
                 policy(("cosine_train", 1), 1, 0, False, [False]*2) == ("cosine_train", 1),
                 policy(("matched", 18), 2, 0, False, [False]*2) == ("functional_pilot", None),
                 policy(("matched", 18), 2, 0, True, [False]*2) == ("matched", 18),
                 policy(("cosine_full", 0), 2, 0, True, [False]*2) == ("cosine_full", 0),
                 policy(("training", "control"), 2, 2, True, [False]*2) == ("functional_full", 0),
                 policy(("training", "control"), 2, 2, True, [True, False]) == ("functional_full", 1),
                 policy(("training", "control"), 2, 2, True, [True]*2) == ("training", "control"),
                 command_for("matched", 18, specs) == previous.command_for("matched", 18, specs)]
        assert all(tests)
        action, state = next_action(specs, shared, cue, trial, functional)
        write(REPORT/"CPU_VALIDATION.json", {"passed": True, "CPU_only": True, "checks": len(tests),
            "entrypoint_sha256": sha(Path(__file__)), "plan_sha256": sha(STUDY/"CPU_PLAN_CHECK.json"),
            "order": ORDER, "current_action": action, "state": state, "GPU_started": False})
        print(json.dumps({"passed": True, "CPU_only": True, "checks": len(tests), "action": action}))
        return
    a = json.loads((REPORT/"AUTHORIZATION.json").read_text())
    assert a["entrypoint_sha256"] == sha(Path(__file__)) and a["order"] == ORDER
    assert a["plan_sha256"] == sha(STUDY/"CPU_PLAN_CHECK.json") and a["user_authorized_literature_lightweight_adaptation"]
    lock = (REPORT/"coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        (kind, target), state = next_action(specs, shared, cue, trial, functional)
        write(REPORT/"queue_state.json", {"pid": os.getpid(), "phase": kind, "target": target, "state": state, "GPU1_only": True})
        if kind == "wait":
            time.sleep(30)
            continue
        if kind == "complete":
            return
        command, log = command_for(kind, target, specs)
        with log.open("a") as output:
            p = subprocess.Popen(["/usr/bin/nohup", str(ROOT/".vision-env/bin/python"), "-u", *command], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", PYTHONUNBUFFERED="1"),
                stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            write(REPORT/"queue_state.json", {"pid": os.getpid(), "phase": "running", "kind": kind, "target": target, "worker_pid": p.pid, "GPU1_only": True})
            code = p.wait()
        assert code == 0, "Keep failed artifacts; no blind retry"
        if kind == "functional_pilot":
            assert complete("pilot", None, functional)
        elif kind == "functional_full":
            assert complete("full", FULL[target], functional)
        elif kind.startswith("cosine_"):
            assert previous.complete(previous.VARIANTS[target], "train" if kind == "cosine_train" else "full", trial)
        elif kind == "language_cue":
            assert previous.previous.cue_complete(previous.previous.CASES[target], cue)
        elif kind == "acquired_anchor":
            assert previous.previous.previous.shared_complete(target, shared)
        elif kind == "full":
            assert original.previous.prior.complete_full(target)
        elif kind == "pilot":
            assert original.previous.pilot_complete(specs[1]["cases"][target], specs[1])
        elif kind == "matched":
            assert original.previous.prior.complete_case(specs[0]["cases"][target], specs[0])
        elif kind == "training":
            assert original.full_training(target, specs[2]) in ["valid", "invalid"]
        else:
            assert original.confirmation_complete(target, kind, specs[3])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        write(REPORT/"needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
