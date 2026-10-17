"""CPU dispatcher adding acquired-stage1 references to the unchanged study.

Inference only, no outcome-selected model choices. Keep all48 controls and the
anchor-first order; never start while any other process occupies GPU1.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import onevision_causal_anchor_first_queue as prior

original = prior.original
ROOT = original.ROOT
BASE = original.BASE
REPORT = BASE / "causal_measurement_queue"
SHARED = BASE / "shared_anchor_initial"
ORDER = ["cap1", "pilot12", "matched48_with_shared_acquired_anchor", "full19x4",
         "final21454x4", "cap3", "shared_initial21454", "sparse_fullx4"]


def policy(base_action, foundation_ready, ready, complete):
    assert set(ready) == set(complete) == {4, 2}
    if base_action[0] == "wait" or not foundation_ready:
        return base_action
    for anchor in [4, 2]:
        if ready[anchor] and not complete[anchor]:
            return "acquired_anchor", anchor
    return base_action


def shared_plan():
    gate = json.loads((SHARED / "CPU_PLAN_CHECK.json").read_text())
    assert gate["passed"] and gate["CPU_only"]
    spec = gate["spec"]
    for p, h in spec["source_sha256"].items():
        assert original.previous.prior.sha(ROOT / p) == h
    return spec


def shared_complete(anchor, spec):
    path = SHARED / f"T{anchor:02d}_complete.json"
    if not path.exists():
        return False
    p = json.loads(path.read_text())
    assert p["complete"] and p["spec"] == spec and p["anchor"] == anchor
    assert p["optimizer_updates"] == 0 and p["parameters_unchanged"]
    assert p["full_anchor_QA"] == len(spec["inputs"][str(anchor)]["full_ids"])
    run = original.verify_artifacts(p, "exp/onevision_shared_anchor_initial_20261017")
    gate = json.loads((run / "reload_exact_gate.json").read_text())
    assert gate["passed"] and gate["no_tolerance"] and gate["QA"] == 32
    canonical = BASE / "matched_task_domain" / f"canonical_anchor{anchor}.pt"
    assert p["canonical_sha256"] == gate["canonical_sha256"] == original.previous.prior.sha(canonical)
    parent = BASE / "matched_task_domain" / f"T{anchor:02d}_repeat_anchor__SeqFT_LoRA_complete.json"
    assert p["parent_completion_sha256"] == original.previous.prior.sha(parent)
    rows = [json.loads(s) for s in (run / "evaluations/full_acquired_anchor/anchor_image.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == spec["inputs"][str(anchor)]["full_ids"]
    return True


def next_action(specs, shared):
    failures = list(SHARED.glob("*needs_attention.json"))
    assert not failures, "Shared-stage1 failure: stop and preserve evidence"
    action, state = original.snapshot(specs)
    ready = {}
    for anchor, index in [(4, 0), (2, 24)]:
        assert specs[0]["cases"][index] == [anchor, "repeat_anchor", "SeqFT_LoRA"]
        ready[anchor] = original.previous.prior.complete_case(specs[0]["cases"][index], specs[0])
    complete = {a: shared_complete(a, shared) for a in [4, 2]}
    foundation = state.get("cap1", False) and state.get("pilots", 0) == 12
    state["shared_acquired_anchor"] = {str(a): {"parent_ready":ready[a], "complete":complete[a]} for a in [4,2]}
    return policy(action, foundation, ready, complete), state


def command_for(kind, target, specs):
    if kind == "acquired_anchor":
        return ["analysis/onevision_shared_anchor_initial.py", "--anchor", str(target)], SHARED / f"T{target:02d}_nohup.log"
    if kind == "full":
        return ["analysis/onevision_full_input_repair.py", "--cap", str(target)], BASE / "full_input_repair" / f"cap{target}_full_confirm_nohup.log"
    if kind == "pilot":
        task, variant = specs[1]["cases"][target]
        return ["analysis/onevision_two_module_pilot.py", "--case", str(target)], original.previous.PILOT / f"T{task:02d}_{variant}_nohup.log"
    if kind == "matched":
        a, seq, method = specs[0]["cases"][target]
        return ["analysis/onevision_matched_task_domain.py", "--case", str(target)], BASE / "matched_task_domain" / f"T{a:02d}_{seq}__{method}_nohup.log"
    if kind == "training":
        return ["analysis/onevision_two_module_fullstream.py", "--variant", target], original.FULL / (target + "_nohup.log")
    assert kind in ("initial", "final", "sparse")
    return ["analysis/onevision_two_module_confirmation.py", "--variant", target, "--phase", kind], original.CONFIRM / (target + "_" + kind + "_nohup.log")


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    original.REPORT = REPORT
    original.ORDER = ORDER
    original.choose = prior.choose
    specs, shared = original.plans(), shared_plan()
    assert shared["parent_spec"] == specs[0]
    if check:
        original.check(specs)
        no, yes = {4:False,2:False}, {4:True,2:True}
        ready4, done4 = {4:True,2:False}, {4:True,2:False}
        checks = [policy(("wait",None),True,yes,no)==("wait",None),
                  policy(("matched",18),True,ready4,no)==("acquired_anchor",4),
                  policy(("matched",18),True,ready4,done4)==("matched",18),
                  policy(("matched",30),True,yes,done4)==("acquired_anchor",2),
                  policy(("matched",30),True,yes,yes)==("matched",30),
                  policy(("full",1),False,yes,no)==("full",1),
                  policy(("complete",None),True,yes,no)==("acquired_anchor",4),
                  command_for("acquired_anchor",4,specs)[0]==["analysis/onevision_shared_anchor_initial.py","--anchor","4"],
                  command_for("matched",18,specs)[0]==["analysis/onevision_matched_task_domain.py","--case","18"],
                  prior.PRIORITY[:8]==[0,6,12,18,24,30,36,42]]
        assert all(checks)
        action,state = next_action(specs,shared)
        original.previous.prior.write(REPORT/"CPU_VALIDATION.json", {
            "passed":True,"CPU_only":True,"inherited_checks":11,"additional_checks":len(checks),
            "entrypoint_sha256":original.previous.prior.sha(Path(__file__)),
            "shared_plan_sha256":original.previous.prior.sha(SHARED/"CPU_PLAN_CHECK.json"),
            "immutable_GPU_source_sha256":shared["source_sha256"],"matched_priority":prior.PRIORITY,
            "current_action":action,"state":state,"GPU_started":False})
        print(json.dumps({"CPU_only":True,"checks":21,"GPU_started":False,"current_action":action}));return
    auth=json.loads((REPORT/"AUTHORIZATION.json").read_text())
    assert auth["approved_by_user"] and auth["order"]==ORDER and auth["matched_priority"]==prior.PRIORITY
    assert auth["entrypoint_sha256"]==original.previous.prior.sha(Path(__file__))
    assert auth["shared_plan_sha256"]==original.previous.prior.sha(SHARED/"CPU_PLAN_CHECK.json")
    lock=(REPORT/"coordinator.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        (kind,target),state=next_action(specs,shared)
        original.previous.prior.write(REPORT/"queue_state.json",{"phase":kind,"target":target,"pid":os.getpid(),"state":state})
        if kind=="wait":time.sleep(30);continue
        if kind=="complete":return
        command,log=command_for(kind,target,specs)
        with log.open("a") as output:
            child=subprocess.Popen(["/usr/bin/nohup",str(ROOT/".vision-env/bin/python"),"-u",*command],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="1",OMP_NUM_THREADS="6",MKL_NUM_THREADS="6",PYTHONUNBUFFERED="1"),
                stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            original.previous.prior.write(REPORT/"queue_state.json",{"phase":"running","kind":kind,"target":target,"pid":os.getpid(),"worker_pid":child.pid,"GPU1_only":True})
            code=child.wait()
        assert code==0,"Worker failed; evidence retained, no blind retry"
        if kind=="acquired_anchor":assert shared_complete(target,shared)
        elif kind=="full":assert original.previous.prior.complete_full(target)
        elif kind=="pilot":assert original.previous.pilot_complete(specs[1]["cases"][target],specs[1])
        elif kind=="matched":assert original.previous.prior.complete_case(specs[0]["cases"][target],specs[0])
        elif kind=="training":assert original.full_training(target,specs[2]) in ("valid","invalid")
        else:assert original.confirmation_complete(target,kind,specs[3])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");args=p.parse_args()
    try:main(args.check)
    except Exception:
        REPORT.mkdir(parents=True,exist_ok=True)
        original.previous.prior.write(REPORT/"needs_attention.json",{"pid":os.getpid(),"traceback":traceback.format_exc()})
        raise
