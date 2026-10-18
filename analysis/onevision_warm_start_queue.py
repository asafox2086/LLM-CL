"""CPU admission of an initialization-only control, preserving all old jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import onevision_functional_confirmation_queue as previous

ROOT, BASE = previous.ROOT, previous.BASE
REPORT = BASE / "warm_start_queue"
STUDY = BASE / "warm_start_20261018"
VARIANT = "warm_inherit_K3_v1"
ORDER = ["ready_sharedS1", "original_cosine_train2", "fixed_pool_functional_pilot4views",
         "both_predeclared_functional_full2", "warm_initialization_only_train19",
         "original_learned_cosine_full", "warm_initialization_only_full21454",
         "languagecue4", "matched48_SeqFT_first", "remaining_original_cosine_full",
         "all_original_remaining_jobs"]
sha, write, startticks = previous.sha, previous.write, previous.startticks


def warm_plan():
    p = json.loads((STUDY / "CPU_PLAN_CHECK.json").read_text())
    assert p["passed"] and p["CPU_only"] and not p["GPU_started"] and p["checks"] == 37
    s = p["spec"]
    assert s["variants"] == [VARIANT] and s["initialization_only_control"]
    assert s["train_qa"] == 1313 and s["full_test_qa"] == 21454 and s["cap"] == 3
    assert s["parameter_count_unchanged"] and s["extra_LM_forwards_for_source_choice"] == 0
    assert s["metric_fit_always_learned"] and not s["selection_reads_TEST"]
    assert s["full_confirmation_required_independent_of_diagnostic"]
    for name, expected in s["source_sha256"].items():
        assert sha(ROOT / name) == expected
    cold = BASE / "learned_reuse/learned_cosine_K3_train_complete.json"
    assert sha(cold) == s["cold_parent_completion_sha256"]
    return s


def warm_complete(phase, spec):
    path = STUDY / f"{VARIANT}_{phase}_complete.json"
    if not path.exists():
        return False
    p = json.loads(path.read_text())
    assert p["complete"] and p["spec"] == spec and p["variant"] == VARIANT and p["phase"] == phase
    run = previous.previous.original.verify_artifacts(p, spec["run_namespace"] + "/" + phase)
    assert p["stages"] == 19 and p["train_qa"] == 1313 and p["experts"] == 3
    assert json.loads((run / "renderer_gpu_gate.json").read_text())["passed"]
    if phase == "train":
        assert json.loads((run / "first_task_exact_gate.json").read_text())["passed"]
        assert json.loads((run / "router_input_gate.json").read_text())["passed"]
        for i in range(1, 20):
            g = json.loads((run / "geometry_gates" / f"S{i:02d}.json").read_text())
            init = json.loads((run / "initialization_gates" / f"T{i:02d}.json").read_text())
            assert g["passed"] and init["passed"] and init["allocation_exact_cold"]
        for i in (2, 3):
            init = json.loads((run / "initialization_gates" / f"T{i:02d}.json").read_text())
            assert init["mode"] == "inherit_existing_complete_AB" and init["different_from_base"]
    else:
        assert p["full_test_qa"] == 21454 and p["reload_exact"] and p["optimizer_updates"] == 0
        assert p["parent_training_completion_sha256"] == sha(STUDY / f"{VARIANT}_train_complete.json")
    return True


def choose(action, functional_done, train_done, cold_full_done, warm_full_done):
    if action[0] in ("wait", "acquired_anchor") or not all(functional_done):
        return action
    if not train_done:
        return "warm_train", None
    if not cold_full_done:
        return "cosine_full", 1
    if not warm_full_done:
        return "warm_full", None
    return action


def inherited_status(inherited, ticks, verify):
    if ticks == inherited["worker_startticks"]:
        return "live"
    assert verify(), "Inherited GPU worker ended without complete proof; never restart automatically"
    return "complete"


def next_action(specs, shared, cue, trial, functional, warm):
    inherited = REPORT / "INHERITED_WORKER.json"
    if inherited.exists() and not (REPORT / "INHERITED_COMPLETE.json").exists():
        old = json.loads(inherited.read_text())
        state = inherited_status(old, startticks(old["worker_pid"]),
                                 lambda: previous.verify_worker(old["kind"], old["target"], specs, trial, functional))
        if state == "live":
            return ("wait", None), {"inherited_worker": old, "waiting_for_natural_completion": True}
        write(REPORT / "INHERITED_COMPLETE.json", {"verified": True, "original_worker": old,
                                                   "no_GPU_source_or_prediction_splice": True})
    assert not list(STUDY.glob("*needs_attention.json")), "Preserve warm failure; no retry"
    action, state = previous.next_action(specs, shared, cue, trial, functional)
    if action[0] in ("wait", "acquired_anchor"):
        return action, state
    done = [previous.previous.complete("full", v, functional) for v in previous.previous.FULL]
    trained, confirmed = warm_complete("train", warm), warm_complete("full", warm)
    cold_full = previous.previous.previous.complete("learned_cosine_K3", "full", trial)
    state.update(warm_train_completed=trained, warm_full_completed=confirmed,
                 cold_learned_full_completed=cold_full)
    return choose(action, done, trained, cold_full, confirmed), state


def command_for(kind, target, specs):
    if kind in ("warm_train", "warm_full"):
        phase = "train" if kind == "warm_train" else "full"
        return ["analysis/onevision_warm_start_trial.py", "--phase", phase], STUDY / f"{VARIANT}_{phase}_nohup.log"
    return previous.previous.command_for(kind, target, specs)


def checks():
    tests = []
    for action in [("wait", None), ("acquired_anchor", 4)]:
        tests.append(choose(action, [True, True], False, False, False) == action)
    for done in ([False, False], [True, False], [False, True]):
        action = ("functional_full", 0)
        tests.append(choose(action, done, False, False, False) == action)
    for action in [("matched", 24), ("cosine_full", 0), ("training", "control"), ("complete", None)]:
        tests += [choose(action, [True, True], False, False, False) == ("warm_train", None),
                  choose(action, [True, True], True, False, False) == ("cosine_full", 1),
                  choose(action, [True, True], True, True, False) == ("warm_full", None),
                  choose(action, [True, True], True, True, True) == action]
    item = {"worker_startticks": 123}
    tests += [inherited_status(item, 123, lambda: False) == "live",
              inherited_status(item, None, lambda: True) == "complete",
              inherited_status(item, 456, lambda: True) == "complete"]
    try:
        inherited_status(item, None, lambda: False)
    except AssertionError:
        tests.append(True)
    else:
        tests.append(False)
    tests += [command_for("warm_train", None, None)[0] == ["analysis/onevision_warm_start_trial.py", "--phase", "train"],
              command_for("cosine_full", 1, None) == previous.previous.command_for("cosine_full", 1, None),
              command_for("functional_full", 1, None) == previous.previous.command_for("functional_full", 1, None)]
    assert all(tests)
    return len(tests)


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    n, warm = checks(), warm_plan()
    if check:
        result = {"passed": True, "CPU_only": True, "checks": n, "GPU_started": False,
                  "entrypoint_sha256": sha(Path(__file__)),
                  "unchanged_previous_queue_sha256": sha(ROOT / "analysis/onevision_functional_confirmation_queue.py"),
                  "warm_plan_sha256": sha(STUDY / "CPU_PLAN_CHECK.json"), "order": ORDER,
                  "all_previous_jobs_retained": True, "no_TEST_choice": True}
        write(REPORT / "CPU_POLICY_CHECK.json", result)
        print(json.dumps(result))
        return
    auth = json.loads((REPORT / "AUTHORIZATION.json").read_text())
    assert auth["entrypoint_sha256"] == sha(Path(__file__))
    assert auth["unchanged_previous_queue_sha256"] == sha(ROOT / "analysis/onevision_functional_confirmation_queue.py")
    assert auth["warm_plan_sha256"] == sha(STUDY / "CPU_PLAN_CHECK.json")
    assert auth["order"] == ORDER and auth["all_previous_jobs_retained"] and auth["delegated_GPU1_plan"]
    lock = (REPORT / "coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Preserve old workers. Only CPU report destinations change in this process.
    previous.REPORT = REPORT
    previous.previous.REPORT = REPORT
    original = previous.previous.original
    original.REPORT, original.ORDER = REPORT, ORDER
    original.choose = previous.previous.previous.previous.previous.prior.choose
    specs = original.plans()
    shared = previous.previous.previous.previous.previous.shared_plan()
    cue = previous.previous.previous.previous.cue_plan()
    trial, functional = previous.previous.previous.trial_plan(), previous.previous.functional_plan()
    while True:
        (kind, target), state = next_action(specs, shared, cue, trial, functional, warm)
        write(REPORT / "queue_state.json", {"pid": os.getpid(), "phase": kind, "target": target,
                                            "state": state, "GPU1_only": True})
        if kind == "wait":
            time.sleep(30)
            continue
        if kind == "complete":
            return
        command, log = command_for(kind, target, specs)
        with log.open("a") as output:
            p = subprocess.Popen(["/usr/bin/nohup", str(ROOT / ".vision-env/bin/python"), "-u", *command],
                                 cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="6",
                                                   MKL_NUM_THREADS="6", PYTHONUNBUFFERED="1"),
                                 stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            write(REPORT / "queue_state.json", {"pid": os.getpid(), "phase": "running", "kind": kind,
                                                "target": target, "worker_pid": p.pid, "GPU1_only": True})
            code = p.wait()
        assert code == 0, "Keep failed artifacts; no blind retry"
        if kind in ("warm_train", "warm_full"):
            assert warm_complete("train" if kind == "warm_train" else "full", warm)
        else:
            assert previous.verify_worker(kind, target, specs, trial, functional)


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
