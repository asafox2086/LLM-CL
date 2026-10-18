"""One CPU-owner handoff, preserving the unchanged matched GPU worker.

Does not stop a GPU process or edit an active scientific worker/source.
The new owner accepts an inherited worker only after its SHA completion proof.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"summary/onevision_diagnostics_20261016"
REPORT = BASE/"functional_router_queue"
UNIT = "llmcl-onevision-causal-priority.service"
OBSERVER = "llmcl-onevision-functional-router-report.service"


def cmd(*args):
    return subprocess.check_output(args, text=True).strip()


def ticks(pid):
    try:
        return int((Path("/proc")/str(pid)/"stat").read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        return None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True)+"\n")
    tmp.replace(p)


def main():
    assert not (REPORT/"ACTIVATION.json").exists(), "Single admission, inspect existing activation"
    check = json.loads((REPORT/"CPU_VALIDATION.json").read_text())
    inherited_check = json.loads((REPORT/"CPU_INHERIT_CHECK.json").read_text())
    entry = ROOT/"analysis/onevision_functional_router_queue.py"
    assert check["passed"] and inherited_check["passed"]
    assert check["entrypoint_sha256"] == inherited_check["entrypoint_sha256"] == sha(entry)
    assert cmd("systemctl", "--user", "show", UNIT, "-p", "KillMode", "--value") == "process"
    old_pid = int(cmd("systemctl", "--user", "show", UNIT, "-p", "MainPID", "--value"))
    old_ticks = ticks(old_pid)
    assert old_ticks is not None
    assert "onevision_learned_reuse_queue.py" in (Path("/proc")/str(old_pid)/"cmdline").read_text()
    protected = set(int(s) for s in cmd("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits").splitlines())
    for line in cmd("systemctl", "--user", "list-units", "--type=service", "--state=running", "--no-legend", "--plain").splitlines():
        unit = line.split()[0]
        if unit.startswith("llmcl-") and unit != UNIT:
            pid = int(cmd("systemctl", "--user", "show", unit, "-p", "MainPID", "--value"))
            if pid:
                protected.add(pid)
    before = {str(pid): ticks(pid) for pid in sorted(protected)}
    old_state_path = BASE/"learned_reuse_queue/queue_state.json"
    write(REPORT/"PREVIOUS_STATE.json", json.loads(old_state_path.read_text()))
    authorization = {
        "user_authorized_literature_lightweight_adaptation": True,
        "basis": "Latest human asks to examine published multi-domain methods and adapt them for lightweight data-limited gains; GPU1 plan already delegated.",
        "entrypoint_sha256": sha(entry), "plan_sha256": check["plan_sha256"], "order": check["order"],
        "no_TEST_choice": True, "all_original_jobs_retained": True,
        "CPU_check_count": check["checks"]+inherited_check["checks"],
    }
    write(REPORT/"AUTHORIZATION.json", authorization)
    # KillMode=process terminates only this CPU dispatcher; its child lives.
    subprocess.run(["systemctl", "--user", "stop", UNIT], check=True)
    assert ticks(old_pid) != old_ticks
    state = json.loads(old_state_path.read_text())
    if state.get("phase") == "running":
        worker = state["worker_pid"]
        assert state["kind"] == "matched", "Unexpected inherited job: preserve and inspect"
        assert ticks(worker) is not None, "Worker ended during handoff: inspect its proof before admitting"
        write(REPORT/"INHERITED_WORKER.json", {
            "kind": state["kind"], "target": state["target"], "worker_pid": worker,
            "worker_startticks": ticks(worker), "unchanged_GPU_source": True,
            "retired_CPU_pid": old_pid,
        })
    else:
        assert state.get("phase") == "wait"
    write(REPORT/"PREVIOUS_FINAL_STATE.json", state)
    config = Path.home()/".config/systemd/user"
    dropin = config/(UNIT+".d")/"60-functional-router.conf"
    dropin.parent.mkdir(parents=True, exist_ok=True)
    dropin.write_text("[Service]\nExecStart=\nExecStart=/usr/bin/nohup "+str(ROOT/".assets-env/bin/python")+
        " -u "+str(entry)+"\nStandardOutput=append:"+str(REPORT/"coordinator_nohup.log")+
        "\nKillMode=process\nRestart=no\n")
    observer_source = ROOT/"analysis/onevision_functional_router_report.py"
    observer_text = "[Unit]\nDescription=Read-only CPU evidence for fixed-pool functional routing\n[Service]\nType=simple\nWorkingDirectory="+str(ROOT)+"\nEnvironment=CUDA_VISIBLE_DEVICES=\nEnvironment=PYTHONUNBUFFERED=1\nExecStart=/usr/bin/nohup "+str(ROOT/".assets-env/bin/python")+" -u "+str(observer_source)+" --watch\nStandardInput=null\nStandardOutput=append:"+str(BASE/"functional_router/observer_nohup.log")+"\nStandardError=inherit\nKillMode=process\nRestart=no\n[Install]\nWantedBy=default.target\n"
    (config/OBSERVER).write_text(observer_text)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "start", UNIT], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", OBSERVER], check=True)
    time.sleep(3)
    new_pid = int(cmd("systemctl", "--user", "show", UNIT, "-p", "MainPID", "--value"))
    observer_pid = int(cmd("systemctl", "--user", "show", OBSERVER, "-p", "MainPID", "--value"))
    assert ticks(new_pid) is not None and ticks(observer_pid) is not None
    after = {pid: ticks(int(pid)) for pid in before}
    assert all(after[p] in [before[p], None] for p in before), "Protected PID reused; inspect"
    assert not (REPORT/"needs_attention.json").exists()
    assert not (BASE/"functional_router/observer_needs_attention.json").exists()
    current = json.loads((REPORT/"queue_state.json").read_text())
    assert current["pid"] == new_pid and current["phase"] == "wait"
    write(old_state_path, {"phase": "managed_by_functional_router_queue", "authoritative_queue":
        str((REPORT/"queue_state.json").relative_to(ROOT)), "retired_CPU_pid": old_pid,
        "old_GPU_worker_preserved": state.get("worker_pid")})
    write(REPORT/"ACTIVATION.json", {"UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entrypoint_sha256": sha(entry), "old_CPU_pid": old_pid, "old_startticks": old_ticks,
        "new_CPU_pid": new_pid, "new_startticks": ticks(new_pid),
        "observer_pid": observer_pid, "observer_startticks": ticks(observer_pid),
        "observer_source_sha256": sha(observer_source), "protected_before": before,
        "protected_after": after, "phase": current, "order": check["order"],
        "no_GPU_source_edits": True, "no_GPU_process_stopped": True,
        "dropin": dropin.read_text(), "observer_unit": observer_text,
        "activation_script_sha256": sha(Path(__file__)), "user_authorization": authorization})
    print(json.dumps({"activated": True, "CPU_pid": new_pid, "observer_pid": observer_pid,
                      "inherited_worker": state.get("worker_pid"), "phase": current["phase"]}))


if __name__ == "__main__":
    main()
