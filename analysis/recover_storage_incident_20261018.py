"""One reviewed infrastructure recovery; immutable trainers/settings untouched.

Archive failure evidence, verify disk writes and the SeqFT stage9 graph/hash,
then re-admit original Qwen jobs through the original CPU pipeline. ER has no
intermediate graph: restart in a distinct run, never fabricate resumption.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT/"exp/two_backbone_native_cl_20261017_v2"
REPORT = ROOT/"summary/storage_recovery_20261018"
STATE = ROOT/"summary/two_backbone_native_cl_20261017_v2/new3090_run_status.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024**2), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def cmd(*args):
    return subprocess.check_output(args, text=True).strip()


def process(pid):
    p = Path("/proc")/str(pid)/"stat"
    if not p.exists():
        return None
    s = p.read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "startticks": int(s[19]), "state": s[0]}


def audit():
    s = json.loads(STATE.read_text())
    sources = {p.name: sha(p) for p in FRAME.glob("*.py")}
    probes = []
    for directory in [ROOT, ROOT/"exp", ROOT/"summary", FRAME/"runs/qwen/pct01"]:
        v = os.statvfs(directory)
        assert v.f_bavail*v.f_frsize > 100*1024**3, "Need reviewed free-space headroom"
        p = directory/("storage_recovery_probe_"+str(time.time_ns()))
        payload = os.urandom(1024**2)
        with p.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        assert p.read_bytes() == payload
        p.unlink()
        probes.append({"directory": str(directory.relative_to(ROOT)), "free_bytes": v.f_bavail*v.f_frsize,
                       "write_read_fsync_1MiB_pass": True})
    checked = {}
    for name in ["SeqFT_LoRA", "ER_LoRA"]:
        j = s["jobs"]["pct01:"+name]
        run = ROOT/j["run_dir"]
        config = json.loads((run/"config.json").read_text())
        assert config["execution_source_sha256"] == sources
        assert config["manifest_sha256"] == "36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777"
        assert "No space left on device" in (run/"supervisor.log").read_text(errors="replace")
        assert not process(j["pid"]), "Do not duplicate a live method"
        if name == "SeqFT_LoRA":
            meta = json.loads((run/"checkpoints/latest.json").read_text())
            assert meta["stage"] == 9 and meta["method"] == name
            file = (run/meta["path"]).resolve()
            assert file.is_relative_to((run/"checkpoints").resolve())
            assert file.stat().st_size == meta["bytes"] and sha(file) == meta["sha256"]
            checked[name] = {"action": "original_resume_from_durable_stage9", "checkpoint": meta}
        else:
            assert not (run/"checkpoints/latest.json").exists()
            pre = ROOT/j["preflight_run"]
            a = json.loads((pre/"gpu_preflight.json").read_text())
            b = json.loads((pre/"checkpoint_reload_proof.json").read_text())
            assert a["passed"] and b["passed"]
            assert a["execution_source_sha256"] == b["execution_source_sha256"] == sources
            checked[name] = {"action": "fresh_training_run_no_intermediate_checkpoint", "preflight_valid": True}
    return s, {"passed": True, "source_sha256": sources, "write_probes": probes, "jobs": checked,
               "recovery_source_sha256": sha(Path(__file__)), "trainer_code_or_hyperparameters_changed": False}


def apply():
    assert not (REPORT/"APPLIED.json").exists(), "One reviewed recovery, no blind repetition"
    s, evidence = audit()
    REPORT.mkdir(parents=True, exist_ok=True)
    write(REPORT/"AUDIT.json", evidence)
    unit = "llmcl-two-backbone-native.service"
    assert cmd("systemctl", "--user", "show", unit, "-p", "KillMode", "--value") == "process"
    for gpu in [2, 3]:
        assert not cmd("nvidia-smi", "--id="+str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits")
    pids = [int(p) for p in cmd("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits").splitlines()]
    before = {str(p): process(p) for p in pids}
    subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    assert int(cmd("systemctl", "--user", "show", unit, "-p", "MainPID", "--value")) == 0
    s = json.loads(STATE.read_text())
    write(REPORT/"new3090_state_before.json", s)
    for key, j in s["jobs"].items():
        if j["condition"] != "pct01":
            continue
        assert j["status"].startswith("failed"), (key, j["status"])
        old = dict(j)
        run = ROOT/j["run_dir"]
        p = run/"supervisor.log"
        if p.exists():
            dest = REPORT/"failure_logs"/(j["method"]+".log")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
        j["storage_recovery"] = {"audit": str((REPORT/"AUDIT.json").relative_to(ROOT)),
                                 "previous_status": old, "source_and_split_unchanged": True}
        if j["method"] == "SeqFT_LoRA":
            j["status"] = "ready_resume"
            j["resume_count"] = 0
            # Preserve old retries separately, retaining only the checkpoint9
            # trajectory as the continuation prefix. Do not blend stage10 retries.
            events = run/"logs/events.jsonl"
            if events.exists():
                (REPORT/"SeqFT_events_before.jsonl").write_bytes(events.read_bytes())
                keep = []
                for line in events.read_text().splitlines():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if value.get("stage", 0) <= 9:
                        keep.append(line)
                events.write_text("\n".join(keep)+"\n")
            for p in (run/"checkpoints").glob("stage_*"):
                stage = int(p.name.split("_")[1].split(".")[0])
                if stage > 9 and p.is_file():
                    dest = run/"interrupted_checkpoints_storage_20261018"/p.name
                    assert not dest.exists() and p.resolve().is_relative_to(run.resolve())
                    dest.parent.mkdir(exist_ok=True)
                    p.rename(dest)
        elif j["method"] == "ER_LoRA":
            j["status"] = "ready_training"
        else:
            # Prior admission is incomplete, not a failed scientific result.
            # Full-disk time window and original files retained; no gate relaxed.
            j["status"] = "waiting"
        j.pop("pid", None)
    write(STATE, s)
    for p, original in before.items():
        assert process(int(p))["startticks"] == original["startticks"]
    subprocess.run(["systemctl", "--user", "start", unit], check=True)
    write(REPORT/"APPLIED.json", {"applied": True, "audit_sha256": sha(REPORT/"AUDIT.json"),
          "pipeline_pid": int(cmd("systemctl", "--user", "show", unit, "-p", "MainPID", "--value")),
          "protected_GPU_processes": before, "unchanged_trainer_sources": evidence["source_sha256"],
          "ER_restart_distinct_run": True, "SeqFT_stage10_old_events_archived": True,
          "memory_note": "Resumed SeqFT spans PIDs: do not subtract memory endpoints across PIDs; use raw same-PID windows."})
    print(json.dumps({"applied": True, "SeqFT": "resume9", "ER": "fresh_run", "other_admissions": "fresh_preflight"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    if a.apply:
        apply()
    else:
        _, result = audit()
        print(json.dumps(result))
