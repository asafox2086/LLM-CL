"""One-time CPU handoff for balanced TRAIN-support controls; no GPU stop."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import onevision_cold_data_queue as q


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    unit = "llmcl-onevision-causal-priority.service"
    old_report = q.BASE / "warm_start_queue"
    old_pid = int(command("systemctl", "--user", "show", unit, "-p", "MainPID", "--value"))
    assert old_pid == 363113 and q.startticks(old_pid) == 14840433
    assert command("systemctl", "--user", "show", unit, "-p", "KillMode", "--value") == "process"
    phase = json.loads((old_report / "queue_state.json").read_text())
    assert phase["phase"] == "wait" and phase["state"]["waiting_for_natural_completion"]
    inherited = phase["state"]["inherited_worker"]
    assert inherited["kind"] == "functional_full" and inherited["target"] == 0
    child = inherited["worker_pid"]
    assert child == 358091 and q.startticks(child) == inherited["worker_startticks"] == 14710479
    assert q.checks() == 25
    spec = q.study_plan()
    assert spec["all18_and_both_endpoints_required"]
    q.REPORT.mkdir(parents=True, exist_ok=True)
    assert not (q.REPORT / "ACTIVATION.json").exists(), "One-time admission, no blind retry"
    inherited = dict(inherited, retired_CPU_pid=old_pid, unchanged_GPU_source=True)
    q.write(q.REPORT / "INHERITED_WORKER.json", inherited)
    q.install(spec)
    assert q.previous.checks() == 28
    action, waiting = q.previous.next_action(None, None, None, None, None, None)
    assert action == ("wait", None) and waiting["waiting_for_natural_completion"]
    assert q.previous.inherited_status(inherited, inherited["worker_startticks"], lambda: False) == "live"
    assert q.previous.inherited_status(inherited, None, lambda: True) == "complete"
    try:
        q.previous.inherited_status(inherited, None, lambda: False)
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing proof admitted retry")
    q.write(q.REPORT / "INHERITANCE_CPU_CHECK.json", {"passed": True, "CPU_only": True,
            "checks": 6, "ancestor_policy_checks": 28, "actual_live_worker": inherited,
            "next_action": action, "GPU_started": False, "all_old_commands_retained": True})
    auth = {"delegated_GPU1_plan": True, "basis": "Human requests causal lightweight/few-data diagnosis. Agent admits balanced unique-TRAIN x init x updates on original three anchors; no TEST selection.",
            "wrapper_entrypoint_sha256": q.sha(q.ROOT / "analysis/onevision_cold_data_queue.py"),
            "entrypoint_sha256": q.sha(q.ROOT / "analysis/onevision_warm_start_queue.py"),
            "unchanged_previous_queue_sha256": q.sha(q.ROOT / "analysis/onevision_functional_confirmation_queue.py"),
            "warm_plan_sha256": q.sha(q.previous.STUDY / "CPU_PLAN_CHECK.json"),
            "study_plan_sha256": q.sha(q.STUDY / "CPU_PLAN_CHECK.json"),
            "order": q.ORDER, "all_previous_jobs_retained": True, "no_TEST_choice": True,
            "worker_CPU_checks": 142, "new_CPU_policy_checks": 25, "ancestor_policy_checks": 28,
            "inheritance_checks": 6, "GPU_source_edits": False,
            "entrypoint_names": {"actual": "analysis/onevision_cold_data_queue.py", "immutable_ancestor_loop": "analysis/onevision_warm_start_queue.py"}}
    q.write(q.REPORT / "AUTHORIZATION.json", auth)
    previous = json.loads((old_report / "ACTIVATION.json").read_text())
    ids = {int(p) for p in previous["protected_after"]} | {child, 363745, 336398, 336411, 340149}
    gpu = command("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits").splitlines()
    ids.update(int(p.strip()) for p in gpu if p.strip().isdigit())
    protected = {str(p): q.startticks(p) for p in sorted(ids) if q.startticks(p) is not None}
    q.write(q.REPORT / "ACTIVATION_BEFORE.json", {"old_CPU_pid": old_pid, "old_state": phase,
            "protected_before": protected, "authorization": auth, "source_sha256": q.sha(Path(__file__))})
    subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    assert q.startticks(old_pid) is None
    assert all(q.startticks(int(pid)) == ticks for pid, ticks in protected.items())
    dropin = Path.home() / ".config/systemd/user" / (unit + ".d") / "90-cold-data.conf"
    assert not dropin.exists()
    text = ("[Service]\nExecStart=\nExecStart=/usr/bin/nohup " + str(q.ROOT / ".assets-env/bin/python")
            + " -u " + str(q.ROOT / "analysis/onevision_cold_data_queue.py")
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
    new = json.loads((q.REPORT / "queue_state.json").read_text())
    assert new["pid"] == new_pid and new["phase"] == "wait" and new["state"]["waiting_for_natural_completion"]
    after = {pid: q.startticks(int(pid)) for pid in protected}
    assert protected == after
    result = {"UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "old_CPU_pid": old_pid, "old_startticks": 14840433, "new_CPU_pid": new_pid,
              "new_startticks": q.startticks(new_pid), "protected_before": protected, "protected_after": after,
              "actual_phase": new, "no_GPU_source_edits": True, "no_GPU_process_stopped": True,
              "dropin": text, "authorization": auth, "activation_source_sha256": q.sha(Path(__file__))}
    q.write(q.REPORT / "ACTIVATION.json", result)
    q.write(old_report / "queue_state.json", {"phase": "redirect",
            "authoritative": str((q.REPORT / "queue_state.json").relative_to(q.ROOT)),
            "retired_CPU_pid": old_pid, "new_CPU_pid": new_pid, "GPU_worker_preserved": child})
    print(json.dumps({"activated": True, "new_CPU_pid": new_pid, "new_startticks": result["new_startticks"],
                      "protected": protected, "actual_phase": new["phase"], "inherited_GPU": child}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        q.REPORT.mkdir(parents=True, exist_ok=True)
        q.write(q.REPORT / "ACTIVATION_FAILURE.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
