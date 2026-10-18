"""One-time CPU dispatcher handoff; explicitly preserves the running GPU job."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import onevision_functional_confirmation_queue as q


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    unit = "llmcl-onevision-causal-priority.service"
    old_report = q.BASE / "functional_router_queue"
    assert command("systemctl", "--user", "show", unit, "-p", "KillMode", "--value") == "process"
    old_pid = int(command("systemctl", "--user", "show", unit, "-p", "MainPID", "--value"))
    assert old_pid == 340143 and q.startticks(old_pid) == 14237139
    old_phase = json.loads((old_report / "queue_state.json").read_text())
    assert old_phase["phase"] == "running" and old_phase["kind"] == "functional_pilot"
    assert old_phase["worker_pid"] == 348381
    child_ticks = q.startticks(348381)
    assert child_ticks == 14466517
    assert q.checks() == 18
    spec = q.previous.functional_plan()
    assert spec["full_candidates"] == ["quality_hard", "quality_soft"]
    q.REPORT.mkdir(parents=True, exist_ok=True)
    assert not (q.REPORT / "ACTIVATION.json").exists(), "One-time handoff, no blind retry"
    inherited = {"kind": old_phase["kind"], "target": old_phase["target"],
                 "worker_pid": 348381, "worker_startticks": child_ticks,
                 "unchanged_GPU_source": True, "retired_CPU_pid": old_pid}
    q.write(q.REPORT / "INHERITED_WORKER.json", inherited)
    action, state = q.next_action(None, None, None, None, None)
    assert action == ("wait", None) and state["waiting_for_natural_completion"]
    assert callable(q.previous.previous.previous.cue_complete)
    assert callable(q.previous.previous.previous.previous.shared_complete)
    assert q.previous.command_for("functional_full", 0, None)[0] == [
        "analysis/onevision_functional_router_probe.py", "--scope", "full", "--view", "quality_hard"]
    q.write(q.REPORT / "INHERITANCE_CPU_CHECK.json", {"passed": True, "CPU_only": True, "checks": 4,
            "actual_worker_live": inherited, "next_action": action, "GPU_started": False})
    authorization = {"delegated_GPU1_plan": True,
        "basis": "Human delegates GPU1 plan and requests lightweight domain aggregate performance. Agent advances BOTH predeclared full confirmations before remaining controls, without TEST-dependent selection.",
        "entrypoint_sha256": q.sha(q.ROOT / "analysis/onevision_functional_confirmation_queue.py"),
        "unchanged_previous_queue_sha256": q.sha(q.ROOT / "analysis/onevision_functional_router_queue.py"),
        "functional_plan_sha256": q.sha(q.STUDY / "CPU_PLAN_CHECK.json"),
        "order": q.ORDER, "all_previous_jobs_retained": True, "no_TEST_choice": True,
        "GPU_source_edits": False, "CPU_policy_checks": 18, "inheritance_checks": 4}
    q.write(q.REPORT / "AUTHORIZATION.json", authorization)
    protect_ids = [348381, 336398, 336411, 340149, 184250, 217515, 249343, 265271,
                   336392, 336774, 336779, 337648, 338009]
    protected = {str(pid): q.startticks(pid) for pid in protect_ids if q.startticks(pid) is not None}
    q.write(q.REPORT / "ACTIVATION_BEFORE.json", {"old_CPU_pid": old_pid, "old_phase": old_phase,
            "protected_before": protected, "authorization": authorization,
            "activation_source_sha256": q.sha(Path(__file__))})
    # Only the CPU MainPID receives stop; KillMode=process has been verified.
    subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    assert q.startticks(old_pid) is None
    assert all(q.startticks(int(pid)) == ticks for pid, ticks in protected.items())
    dropin = (Path.home() / ".config/systemd/user" / (unit + ".d") / "70-functional-confirmation.conf")
    assert not dropin.exists()
    text = ("[Service]\nExecStart=\nExecStart=/usr/bin/nohup " + str(q.ROOT / ".assets-env/bin/python")
            + " -u " + str(q.ROOT / "analysis/onevision_functional_confirmation_queue.py")
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
    after = {pid: q.startticks(int(pid)) for pid in protected}
    assert protected == after
    result = {"UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "old_CPU_pid": old_pid, "old_startticks": 14237139,
              "new_CPU_pid": new_pid, "new_startticks": q.startticks(new_pid),
              "protected_before": protected, "protected_after": after, "phase": phase,
              "no_GPU_source_edits": True, "no_GPU_process_stopped": True,
              "dropin": text, "authorization": authorization,
              "activation_source_sha256": q.sha(Path(__file__))}
    q.write(q.REPORT / "ACTIVATION.json", result)
    q.write(old_report / "queue_state.json", {"phase": "redirect",
            "authoritative": str((q.REPORT / "queue_state.json").relative_to(q.ROOT)),
            "retired_CPU_pid": old_pid, "new_CPU_pid": new_pid, "original_GPU_worker_preserved": 348381})
    print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        q.write(q.REPORT / "ACTIVATION_FAILURE.json", {"pid": os.getpid(), "traceback": traceback.format_exc()})
        raise
