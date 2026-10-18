"""One-time CPU-only handoff, preserving the actual full-test GPU worker."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import onevision_warm_start_queue as q


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    unit = "llmcl-onevision-causal-priority.service"
    old_report = q.BASE / "functional_confirmation_queue"
    assert command("systemctl", "--user", "show", unit, "-p", "KillMode", "--value") == "process"
    old_pid = int(command("systemctl", "--user", "show", unit, "-p", "MainPID", "--value"))
    assert old_pid == 356616 and q.startticks(old_pid) == 14682713
    old = json.loads((old_report / "queue_state.json").read_text())
    assert old["phase"] == "running" and old["kind"] == "functional_full" and old["target"] == 0
    child = old["worker_pid"]
    assert child == 358091 and q.startticks(child) is not None
    assert q.checks() == 28
    warm = q.warm_plan()
    functional = q.previous.previous.functional_plan()
    assert functional["full_candidates"] == ["quality_hard", "quality_soft"]
    q.REPORT.mkdir(parents=True, exist_ok=True)
    assert not (q.REPORT / "ACTIVATION.json").exists(), "One-time handoff, no blind retry"
    inherited = {"kind": old["kind"], "target": old["target"], "worker_pid": child,
                 "worker_startticks": q.startticks(child), "unchanged_GPU_source": True,
                 "retired_CPU_pid": old_pid}
    q.write(q.REPORT / "INHERITED_WORKER.json", inherited)
    action, state = q.next_action(None, None, None, None, None, None)
    assert action == ("wait", None) and state["waiting_for_natural_completion"]
    assert callable(q.previous.previous.previous.previous.cue_complete)
    assert callable(q.previous.previous.previous.previous.previous.shared_complete)
    assert callable(q.previous.previous.previous.previous.previous.prior.choose)
    # The new function must not swallow failures or treat an observation timeout
    # as permission to retry. These are pure CPU negative controls.
    assert q.inherited_status(inherited, inherited["worker_startticks"], lambda: False) == "live"
    assert q.inherited_status(inherited, None, lambda: True) == "complete"
    try:
        q.inherited_status(inherited, None, lambda: False)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing proof incorrectly admitted a retry")
    q.write(q.REPORT / "INHERITANCE_CPU_CHECK.json", {"passed": True, "CPU_only": True,
            "checks": 7, "actual_worker_live": inherited, "GPU_started": False,
            "actual_action": action, "both_existing_full_confirmations_before_warm": True})
    auth = {"delegated_GPU1_plan": True,
            "basis": "Human requests causal cold-start diagnosis and delegates GPU1 plan. Add initialization-only19-stage control, preserve both already admitted full confirmations and every original job.",
            "entrypoint_sha256": q.sha(q.ROOT / "analysis/onevision_warm_start_queue.py"),
            "unchanged_previous_queue_sha256": q.sha(q.ROOT / "analysis/onevision_functional_confirmation_queue.py"),
            "warm_plan_sha256": q.sha(q.STUDY / "CPU_PLAN_CHECK.json"), "order": q.ORDER,
            "all_previous_jobs_retained": True, "no_TEST_choice": True, "GPU_source_edits": False,
            "worker_CPU_checks": 37, "CPU_policy_checks": 28, "inheritance_checks": 7}
    q.write(q.REPORT / "AUTHORIZATION.json", auth)
    previous_activation = json.loads((old_report / "ACTIVATION.json").read_text())
    ids = {int(p) for p in previous_activation["protected_after"]} | {child, 336398, 336411, 340149}
    gpu_ids = command("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits").splitlines()
    ids.update(int(p.strip()) for p in gpu_ids if p.strip().isdigit())
    protected = {str(p): q.startticks(p) for p in sorted(ids) if q.startticks(p) is not None}
    q.write(q.REPORT / "ACTIVATION_BEFORE.json", {"old_CPU_pid": old_pid, "old_phase": old,
            "protected_before": protected, "authorization": auth, "activation_source_sha256": q.sha(Path(__file__))})
    # Verified KillMode=process sends stop only to the CPU MainPID. GPU children
    # continue naturally, then their existing immutable proof must validate.
    subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    assert q.startticks(old_pid) is None
    assert all(q.startticks(int(pid)) == ticks for pid, ticks in protected.items())
    dropin = Path.home() / ".config/systemd/user" / (unit + ".d") / "80-warm-start.conf"
    assert not dropin.exists()
    text = ("[Service]\nExecStart=\nExecStart=/usr/bin/nohup " + str(q.ROOT / ".assets-env/bin/python")
            + " -u " + str(q.ROOT / "analysis/onevision_warm_start_queue.py")
            + "\nStandardOutput=append:" + str(q.REPORT / "coordinator_nohup.log")
            + "\nKillMode=process\nRestart=no\n")
    dropin.write_text(text)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "start", unit], check=True)
    new_pid = int(command("systemctl", "--user", "show", unit, "-p", "MainPID", "--value"))
    assert new_pid and new_pid != old_pid
    deadline = time.monotonic() + 15
    while not (q.REPORT / "queue_state.json").exists() and time.monotonic() < deadline:
        time.sleep(.5)
    phase = json.loads((q.REPORT / "queue_state.json").read_text())
    assert phase["pid"] == new_pid and phase["phase"] == "wait"
    assert phase["state"]["waiting_for_natural_completion"]
    after = {p: q.startticks(int(p)) for p in protected}
    assert after == protected
    result = {"UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "old_CPU_pid": old_pid, "old_startticks": 14682713,
              "new_CPU_pid": new_pid, "new_startticks": q.startticks(new_pid),
              "protected_before": protected, "protected_after": after, "phase": phase,
              "no_GPU_source_edits": True, "no_GPU_process_stopped": True,
              "dropin": text, "authorization": auth, "activation_source_sha256": q.sha(Path(__file__))}
    q.write(q.REPORT / "ACTIVATION.json", result)
    q.write(old_report / "queue_state.json", {"phase": "redirect",
            "authoritative": str((q.REPORT / "queue_state.json").relative_to(q.ROOT)),
            "retired_CPU_pid": old_pid, "new_CPU_pid": new_pid, "original_GPU_worker_preserved": child})
    print(json.dumps({"activated": True, "new_CPU_pid": new_pid,
                      "new_startticks": result["new_startticks"], "protected": protected,
                      "actual_phase": phase["phase"], "GPU_worker_preserved": child}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        q.REPORT.mkdir(parents=True, exist_ok=True)
        q.write(q.REPORT / "ACTIVATION_FAILURE.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
