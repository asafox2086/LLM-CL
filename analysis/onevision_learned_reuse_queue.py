"""New CPU entrypoint: learned-cosine trials plus every unchanged prior job.

No imported worker file is modified. Foreign GPU1 occupancy always wins.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import onevision_language_causal_queue as previous

original = previous.original
ROOT, BASE = previous.ROOT, previous.BASE
REPORT = BASE/"learned_reuse_queue"
TRIAL = BASE/"learned_reuse"
VARIANTS = ["fixed_cosine_K3", "learned_cosine_K3"]
ORDER = ["cap1", "pilot12", "ready_shared_acquired_anchor", "cosine_reuse_train2",
         "fixed_language_cue4runs_all8fullcells", "matched48_SeqFT_first", "cosine_reuse_full2",
         "full19x4", "final21454x4", "cap3", "shared_initial21454", "sparse_fullx4"]
sha, write = previous.sha, previous.write


def trial_plan():
    p = json.loads((TRIAL/"CPU_PLAN_CHECK.json").read_text())
    mechanics = json.loads((TRIAL/"CPU_MECHANICS.json").read_text())
    assert p["passed"] and p["CPU_only"] and mechanics["passed"] and mechanics["CPU_only"]
    assert mechanics["synthetic_not_model_results"]
    spec = p["spec"]
    assert spec["variants"] == VARIANTS and spec["train_qa"] == 1313 and spec["cap"] == 3
    assert not spec["selection_reads_TEST"] and spec["all_variants_full_confirmation_required"]
    for name, expected in spec["source_sha256"].items():
        assert sha(ROOT/name) == expected
    assert mechanics["source_sha256"] == {s: sha(ROOT/s) for s in mechanics["source_sha256"]}
    return spec


def complete(variant, phase, spec):
    path = TRIAL/f"{variant}_{phase}_complete.json"
    if not path.exists():
        return False
    p = json.loads(path.read_text())
    assert p["complete"] and p["spec"] == spec and p["variant"] == variant and p["phase"] == phase
    run = original.verify_artifacts(p, "exp/onevision_learned_reuse_20261018")
    assert p["train_qa"] == 1313 and p["stages"] == 19 and p["experts"] <= spec["cap"]
    assert json.loads((run/"renderer_gpu_gate.json").read_text())["passed"]
    if phase == "train":
        assert json.loads((run/"first_task_exact_gate.json").read_text())["passed"]
        assert json.loads((run/"router_input_gate.json").read_text())["passed"]
        assert len(list((run/"decisions").glob("reuse_T*.json"))) == 19
    else:
        assert p["full_test_qa"] == 21454 and p["optimizer_updates"] == 0 and p["reload_exact"]
        assert p["parent_training_completion_sha256"] == sha(TRIAL/f"{variant}_train_complete.json")
    return True


def policy(action, foundation, shared4, matched, training, full):
    assert len(training) == len(full) == 2
    if action[0] in ["wait", "acquired_anchor"] or not foundation or not shared4:
        return action
    for i, done in enumerate(training):
        if not done:
            return "cosine_train", i
    if matched == 48 and action[0] != "language_cue":
        for i, done in enumerate(full):
            if not done:
                return "cosine_full", i
    return action


def next_action(specs, shared, cue, trial):
    assert not list(TRIAL.glob("*needs_attention.json")), "Keep failed trial; no blind retry"
    action, state = previous.next_action(specs, shared, cue)
    train = [complete(v, "train", trial) for v in VARIANTS]
    full = [complete(v, "full", trial) for v in VARIANTS]
    state["cosine_train_completed"] = sum(train)
    state["cosine_full_completed"] = sum(full)
    if action[0] == "wait":
        return action, state
    foundation = state.get("cap1", False) and state.get("pilots", 0) == 12
    shared4 = previous.previous.shared_complete(4, shared)
    return policy(action, foundation, shared4, state.get("matched", 0), train, full), state


def command_for(kind, target, specs):
    if kind in ["cosine_train", "cosine_full"]:
        phase, variant = ("train" if kind == "cosine_train" else "full"), VARIANTS[target]
        return ["analysis/onevision_learned_reuse_trial.py", "--variant", variant, "--phase", phase], TRIAL/f"{variant}_{phase}_nohup.log"
    return previous.command_for(kind, target, specs)


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    original.REPORT, original.ORDER, original.choose = REPORT, ORDER, previous.previous.prior.choose
    specs, shared, cue, trial = original.plans(), previous.previous.shared_plan(), previous.cue_plan(), trial_plan()
    if check:
        original.check(specs)
        no, yes = [False]*2, [True]*2
        tests = [policy(("wait", None), True, True, 12, no, no) == ("wait", None),
                 policy(("acquired_anchor", 4), True, False, 12, no, no) == ("acquired_anchor", 4),
                 policy(("full", 1), False, True, 12, no, no) == ("full", 1),
                 policy(("matched", 18), True, False, 12, no, no) == ("matched", 18),
                 policy(("language_cue", 0), True, True, 12, no, no) == ("cosine_train", 0),
                 policy(("matched", 18), True, True, 12, [True, False], no) == ("cosine_train", 1),
                 policy(("language_cue", 0), True, True, 12, yes, no) == ("language_cue", 0),
                 policy(("matched", 18), True, True, 12, yes, no) == ("matched", 18),
                 policy(("training", "control"), True, True, 48, yes, no) == ("cosine_full", 0),
                 policy(("complete", None), True, True, 48, yes, [True, False]) == ("cosine_full", 1),
                 policy(("training", "control"), True, True, 48, yes, yes) == ("training", "control"),
                 command_for("matched", 18, specs) == previous.command_for("matched", 18, specs)]
        assert all(tests)
        action, state = next_action(specs, shared, cue, trial)
        write(REPORT/"CPU_VALIDATION.json", {"passed": True, "CPU_only": True, "inherited_checks": 11,
             "new_checks": len(tests), "GPU_started": False, "current_action": action, "state": state,
             "entrypoint_sha256": sha(Path(__file__)), "plan_sha256": sha(TRIAL/"CPU_PLAN_CHECK.json"), "order": ORDER})
        print(json.dumps({"passed": True, "CPU_only": True, "checks": 23, "current_action": action}))
        return
    auth = json.loads((REPORT/"AUTHORIZATION.json").read_text())
    assert auth["authorization_basis"] == "explicit_user_try_learned_cosine_reuse_and_delegated_GPU1_plan"
    assert auth["entrypoint_sha256"] == sha(Path(__file__)) and auth["order"] == ORDER
    assert auth["plan_sha256"] == sha(TRIAL/"CPU_PLAN_CHECK.json")
    lock = (REPORT/"coordinator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        (kind, target), state = next_action(specs, shared, cue, trial)
        write(REPORT/"queue_state.json", {"phase": kind, "target": target, "pid": os.getpid(), "state": state, "GPU1_only": True})
        if kind == "wait":
            time.sleep(30)
            continue
        if kind == "complete":
            return
        command, log = command_for(kind, target, specs)
        with log.open("a") as output:
            child = subprocess.Popen(["/usr/bin/nohup", str(ROOT/".vision-env/bin/python"), "-u", *command], cwd=ROOT,
                 env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="6", MKL_NUM_THREADS="6", PYTHONUNBUFFERED="1"),
                 stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
            write(REPORT/"queue_state.json", {"phase": "running", "kind": kind, "target": target,
                                              "pid": os.getpid(), "worker_pid": child.pid, "GPU1_only": True})
            code = child.wait()
        assert code == 0, "Preserve failure, no blind retry"
        if kind.startswith("cosine_"):
            assert complete(VARIANTS[target], "train" if kind == "cosine_train" else "full", trial)
        elif kind == "language_cue":
            assert previous.cue_complete(previous.CASES[target], cue)
        elif kind == "acquired_anchor":
            assert previous.previous.shared_complete(target, shared)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        write(REPORT/"needs_attention.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
