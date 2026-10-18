"""Prioritize both predeclared full functional confirmations, not TEST winners.

New CPU entrypoint only. Immutable model workers, baseline jobs and all previous
causal controls survive. Inherited GPU worker must finish naturally with proof.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import onevision_functional_router_queue as previous

ROOT, BASE, STUDY = previous.ROOT, previous.BASE, previous.STUDY
REPORT = BASE / "functional_confirmation_queue"
ORDER = ["ready_sharedS1", "original_cosine_train2", "fixed_pool_functional_pilot4views",
         "both_predeclared_functional_full2", "languagecue4", "matched48_SeqFT_first",
         "original_cosine_full2", "all_original_remaining_jobs"]
sha, write = previous.sha, previous.write


def choose(action, pilot_done, full_done):
    assert len(full_done) == 2
    if action[0] in ("wait", "acquired_anchor") or not pilot_done:
        return action
    for i, done in enumerate(full_done):
        if not done:
            return "functional_full", i
    return action


def startticks(pid):
    p = Path("/proc") / str(pid) / "stat"
    return int(p.read_text().rsplit(")", 1)[1].split()[19]) if p.exists() else None


def verify_worker(kind, target, specs, trial, functional):
    if kind == "functional_pilot":
        return previous.complete("pilot", None, functional)
    if kind == "functional_full":
        return previous.complete("full", previous.FULL[target], functional)
    if kind.startswith("cosine_"):
        return previous.previous.complete(previous.previous.VARIANTS[target],
                                          "train" if kind == "cosine_train" else "full", trial)
    if kind == "language_cue":
        return previous.previous.previous.cue_complete(previous.previous.previous.CASES[target],
                                                        previous.previous.previous.cue_plan())
    if kind == "acquired_anchor":
        return previous.previous.previous.previous.shared_complete(target,
                                                                   previous.previous.previous.previous.shared_plan())
    if kind == "full":
        return previous.original.previous.prior.complete_full(target)
    if kind == "pilot":
        return previous.original.previous.pilot_complete(specs[1]["cases"][target], specs[1])
    if kind == "matched":
        return previous.original.previous.prior.complete_case(specs[0]["cases"][target], specs[0])
    if kind == "training":
        return previous.original.full_training(target, specs[2]) in ("valid", "invalid")
    return previous.original.confirmation_complete(target, kind, specs[3])


def next_action(specs, shared, cue, trial, functional):
    inherited_path = REPORT / "INHERITED_WORKER.json"
    if inherited_path.exists() and not (REPORT / "INHERITED_COMPLETE.json").exists():
        inherited = json.loads(inherited_path.read_text())
        if startticks(inherited["worker_pid"]) == inherited["worker_startticks"]:
            return ("wait", None), {"inherited_worker": inherited, "waiting_for_natural_completion": True}
        assert verify_worker(inherited["kind"], inherited["target"], specs, trial, functional), \
            "Inherited worker ended without valid proof; no retry"
        write(REPORT / "INHERITED_COMPLETE.json", {"verified": True, "original_worker": inherited,
                                                   "no_source_or_prediction_splice": True})
    # Old implementation retains all foreign-GPU, failure and source SHA gates.
    action, state = previous.next_action(specs, shared, cue, trial, functional)
    if action[0] in ("wait", "acquired_anchor"):
        return action, state
    pilot = state["functional_pilot_complete"]
    full = [previous.complete("full", v, functional) for v in previous.FULL]
    state["predeclared_functional_full_completed"] = sum(full)
    return choose(action, pilot, full), state


def checks():
    tests = []
    for action in [("wait", None), ("acquired_anchor", 4)]:
        tests.append(choose(action, True, [False, False]) == action)
    for action in [("functional_pilot", None), ("cosine_train", 1), ("matched", 24)]:
        tests.append(choose(action, False, [False, False]) == action)
    for action in [("matched", 24), ("cosine_full", 0), ("complete", None), ("training", "control")]:
        tests.append(choose(action, True, [False, False]) == ("functional_full", 0))
        tests.append(choose(action, True, [True, False]) == ("functional_full", 1))
        tests.append(choose(action, True, [True, True]) == action)
    tests.append(previous.FULL == ["quality_hard", "quality_soft"])
    assert all(tests)
    return len(tests)


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    n = checks()
    if check:
        result = {"passed": True, "CPU_only": True, "GPU_started": False, "checks": n,
                  "entrypoint_sha256": sha(Path(__file__)),
                  "unchanged_previous_queue_sha256": sha(ROOT / "analysis/onevision_functional_router_queue.py"),
                  "order": ORDER, "all_previous_jobs_retained": True, "no_TEST_choice": True}
        write(REPORT / "CPU_POLICY_CHECK.json", result)
        print(json.dumps(result))
        return
    authorization = json.loads((REPORT / "AUTHORIZATION.json").read_text())
    assert authorization["entrypoint_sha256"] == sha(Path(__file__))
    assert authorization["unchanged_previous_queue_sha256"] == sha(ROOT / "analysis/onevision_functional_router_queue.py")
    assert authorization["order"] == ORDER and authorization["all_previous_jobs_retained"]
    assert authorization["delegated_GPU1_plan"] and authorization["no_TEST_choice"]
    lock = (REPORT / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous.REPORT = REPORT
    previous.original.REPORT, previous.original.ORDER = REPORT, ORDER
    previous.original.choose = previous.previous.previous.previous.prior.choose
    specs = previous.original.plans()
    shared = previous.previous.previous.previous.shared_plan()
    cue = previous.previous.previous.cue_plan()
    trial, functional = previous.previous.trial_plan(), previous.functional_plan()
    assert authorization["functional_plan_sha256"] == sha(STUDY / "CPU_PLAN_CHECK.json")
    while True:
        (kind, target), state = next_action(specs, shared, cue, trial, functional)
        write(REPORT / "queue_state.json", {"pid": os.getpid(), "phase": kind, "target": target,
                                            "state": state, "GPU1_only": True})
        if kind == "wait":
            time.sleep(30)
            continue
        if kind == "complete":
            return
        command, log = previous.command_for(kind, target, specs)
        with log.open("a") as output:
            p = subprocess.Popen(["/usr/bin/nohup", str(ROOT / ".vision-env/bin/python"), "-u", *command], cwd=ROOT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="6",
                                          MKL_NUM_THREADS="6", PYTHONUNBUFFERED="1"),
                                 stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            write(REPORT / "queue_state.json", {"pid": os.getpid(), "phase": "running", "kind": kind,
                                                "target": target, "worker_pid": p.pid, "GPU1_only": True})
            code = p.wait()
        assert code == 0, "Preserve failure, no blind retry"
        assert verify_worker(kind, target, specs, trial, functional)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        write(REPORT / "needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
