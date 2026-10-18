"""Small CPU-only admission adapter; all old GPU/queue sources stay frozen."""
import argparse
import json
from pathlib import Path
import traceback

import onevision_warm_start_queue as previous

ROOT, BASE = previous.ROOT, previous.BASE
REPORT = BASE / "cold_data_queue"
STUDY = BASE / "cold_data_factorial_20261018"
ORDER = previous.ORDER[:5] + ["all18_uniqueTRAIN_initialization_trajectories_both1and4_endpoints"] + previous.ORDER[5:]
sha, write, startticks = previous.sha, previous.write, previous.startticks
OLD_NEXT, OLD_COMMAND = previous.next_action, previous.command_for
OLD_VERIFY = previous.previous.verify_worker


def study_plan():
    p = json.loads((STUDY / "CPU_PLAN_CHECK.json").read_text())
    assert p["passed"] and p["CPU_only"] and not p["GPU_started"] and p["checks"] == 142
    spec = p["spec"]
    assert spec["targets"] == [2, 11, 12] and len(spec["cases"]) == 18
    assert spec["all18_and_both_endpoints_required"] and not spec["TEST_selects_anything"]
    assert spec["total_training_QA_forwards"] == 1632 and spec["total_target_generation_QA"] == 10710
    assert not spec["official1313_manifest_changed"] and not spec["formal_V2"]
    for name, expected in spec["source_sha256"].items():
        assert sha(ROOT / name) == expected
    assert sha(BASE / "deterministic_native/complete.json") == spec["native_completion_sha256"]
    return spec


def study_complete(index, spec):
    path = STUDY / f"case_{index:02d}_complete.json"
    if not path.exists():
        return False
    p = json.loads(path.read_text())
    assert p["complete"] and p["spec"] == spec and p["case"] == spec["cases"][index]
    run = previous.previous.previous.original.verify_artifacts(p, "exp/onevision_cold_data_factorial_20261018")
    assert p["valid_training"] and p["updates"] == 4 and p["extra_LoRAs"] == 0
    assert json.loads((run / "initial_state_gate.json").read_text())["passed"]
    assert json.loads((run / "renderer_gpu_gate.json").read_text())["passed"]
    events = json.loads((run / "training.json").read_text())["records"]
    assert len(events) == 4
    for label in ("initial", "endpoint_1", "endpoint_4"):
        folder = run / "evaluations" / label
        assert json.loads((folder / "durable_complete.json").read_text())["complete"]
        rows = [json.loads(s) for s in (folder / "native_image.jsonl").read_text().splitlines()]
        assert [r["id"] for r in rows] == p["case"]["TEST_ids"]
    return True


def choose(action, warm_train, done):
    assert len(done) == 18
    if action[0] in ("wait", "acquired_anchor", "functional_full", "functional_pilot", "cosine_train", "warm_train") or not warm_train:
        return action
    for index, complete in enumerate(done):
        if not complete:
            return "data_factorial", index
    return action


def command_for(kind, target, specs):
    if kind == "data_factorial":
        return ["analysis/onevision_cold_data_factorial.py", "--case-index", str(target)], STUDY / f"case_{target:02d}_nohup.log"
    return OLD_COMMAND(kind, target, specs)


def checks():
    tests = []
    no, yes = [False] * 18, [True] * 18
    for action in [("wait", None), ("acquired_anchor", 4), ("functional_full", 1), ("warm_train", None),
                   ("cosine_train", 0), ("functional_pilot", None)]:
        tests.append(choose(action, True, no) == action)
    for action in [("cosine_full", 1), ("warm_full", None), ("matched", 24), ("complete", None)]:
        tests += [choose(action, False, no) == action, choose(action, True, no) == ("data_factorial", 0),
                  choose(action, True, [True] * 5 + [False] * 13) == ("data_factorial", 5),
                  choose(action, True, yes) == action]
    tests += [command_for("warm_full", None, None) == OLD_COMMAND("warm_full", None, None),
              command_for("functional_full", 1, None) == OLD_COMMAND("functional_full", 1, None),
              command_for("data_factorial", 17, None)[0] == ["analysis/onevision_cold_data_factorial.py", "--case-index", "17"]]
    assert all(tests)
    return len(tests)


def install(spec):
    def next_action(*args):
        assert not list(STUDY.glob("*needs_attention.json")), "Preserve factorial failure; no retry"
        action, state = OLD_NEXT(*args)
        if action[0] in ("wait", "acquired_anchor") or not state.get("warm_train_completed", False):
            return action, state
        done = [study_complete(i, spec) for i in range(18)]
        state["cold_data_factorial_completed"] = sum(done)
        return choose(action, True, done), state

    def verify(kind, target, *args):
        return study_complete(target, spec) if kind == "data_factorial" else OLD_VERIFY(kind, target, *args)

    previous.REPORT, previous.ORDER = REPORT, ORDER
    previous.next_action = next_action
    previous.command_for = command_for
    previous.previous.verify_worker = verify


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    n, spec = checks(), study_plan()
    if check:
        result = {"passed": True, "CPU_only": True, "checks": n, "GPU_started": False,
                  "wrapper_entrypoint_sha256": sha(Path(__file__)),
                  "unchanged_warm_queue_sha256": sha(ROOT / "analysis/onevision_warm_start_queue.py"),
                  "study_plan_sha256": sha(STUDY / "CPU_PLAN_CHECK.json"),
                  "order": ORDER, "all_previous_jobs_retained": True, "no_TEST_choice": True}
        write(REPORT / "CPU_POLICY_CHECK.json", result)
        print(json.dumps(result))
        return
    auth = json.loads((REPORT / "AUTHORIZATION.json").read_text())
    assert auth["wrapper_entrypoint_sha256"] == sha(Path(__file__))
    assert auth["study_plan_sha256"] == sha(STUDY / "CPU_PLAN_CHECK.json")
    assert auth["entrypoint_sha256"] == sha(ROOT / "analysis/onevision_warm_start_queue.py")
    assert auth["order"] == ORDER and auth["all_previous_jobs_retained"]
    install(spec)
    # The immutable ancestor verifies its own source, source ancestry and warm
    # plan too. AUTHORIZATION explicitly records both wrapper and ancestor SHA.
    previous.main(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        write(REPORT / "needs_attention.json", {"traceback": traceback.format_exc()})
        raise
