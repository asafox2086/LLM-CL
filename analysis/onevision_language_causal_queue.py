"""New frozen CPU dispatcher for a fixed-weight language-confound intervention.

Preserves the acquired-reference priority and every original48/full-method job.
No active/previous worker source is edited. Foreign GPU1 use always wins.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import onevision_causal_measurement_queue as previous

original=previous.original
ROOT=original.ROOT
BASE=original.BASE
REPORT=BASE/"language_causal_queue"
CUE=BASE/"language_cue"
CASES=[(seq,scope) for scope in ["probe","full"] for seq in ["within_medical","cross_short"]]
ORDER=["cap1","pilot12","ready_shared_acquired_anchor","fixed_language_cue4runs_all8fullcells",
       "matched48_SeqFT_first","full19x4","final21454x4","cap3","shared_initial21454","sparse_fullx4"]
sha=original.previous.prior.sha
write=original.previous.prior.write


def cue_plan():
    g=json.loads((CUE/"CPU_PLAN_CHECK.json").read_text());assert g["passed"] and g["CPU_only"]
    s=g["spec"]
    for p,h in s["source_sha256"].items():assert sha(ROOT/p)==h
    assert s["full_all8_cells_required_regardless_probe_results"] and s["optimizer_updates"]==0
    assert s["full_unique_QA"]==507 and s["full_condition_observations"]==4056
    return s


def cue_complete(case,spec):
    seq,scope=case;p=CUE/f"T04_{seq}_{scope}_complete.json"
    if not p.exists():return False
    v=json.loads(p.read_text())
    assert v["complete"] and v["spec"]==spec and v["sequence"]==seq and v["scope"]==scope
    assert v["optimizer_updates"]==0 and v["parameters_unchanged"] and v["views"]==spec["views"]
    n=len(spec["probe_ids"] if scope=="probe" else spec["full_ids"])
    assert v["QA_per_view"]==n
    run=original.verify_artifacts(v,"exp/onevision_language_cue_20261017")
    gate=json.loads((run/"original_exact_gate.json").read_text())
    assert gate["passed"] and gate["no_tolerance"] and gate["QA"]==n
    for view in spec["views"]:
        assert json.loads((run/"input_gates"/(view+".json")).read_text())["CPU_input_ids_pixels_image_sizes_exact"]
        assert json.loads((run/"parameter_gates"/(view+".json")).read_text())["all_A_B_exact"]
        rows=[json.loads(t) for t in (run/f"evaluations/{view}/anchor_image.jsonl").read_text().splitlines()]
        assert [r["id"] for r in rows]==(spec["probe_ids"] if scope=="probe" else spec["full_ids"])
    return True


def policy(action,foundation,shared4,done):
    assert len(done)==4
    if action[0] in ("wait","acquired_anchor") or not foundation or not shared4:return action
    for i,finished in enumerate(done):
        if not finished:return "language_cue",i
    return action


def next_action(specs,shared,cue):
    assert not list(CUE.glob("*needs_attention.json")),"Preserve cue failure; no blind retry"
    action,state=previous.next_action(specs,shared)
    complete=[cue_complete(c,cue) for c in CASES]
    state["language_cue_completed_runs"]=sum(complete)
    state["language_cue_required_runs"]=4
    if action[0]=="wait":return action,state
    ready=previous.shared_complete(4,shared)
    foundation=state.get("cap1",False) and state.get("pilots",0)==12
    return policy(action,foundation,ready,complete),state


def command_for(kind,target,specs):
    if kind=="language_cue":
        seq,scope=CASES[target]
        return ["analysis/onevision_language_cue_probe.py","--sequence",seq,"--scope",scope],CUE/f"T04_{seq}_{scope}_nohup.log"
    return previous.command_for(kind,target,specs)


def main(check=False):
    REPORT.mkdir(parents=True,exist_ok=True)
    original.REPORT=REPORT;original.ORDER=ORDER;original.choose=previous.prior.choose
    specs,shared,cue=original.plans(),previous.shared_plan(),cue_plan()
    assert cue["parent_spec"]==shared["parent_spec"]==specs[0]
    if check:
        original.check(specs)
        no=[False]*4;yes=[True]*4
        checks=[policy(("wait",None),True,True,no)==("wait",None),
                policy(("acquired_anchor",4),True,False,no)==("acquired_anchor",4),
                policy(("matched",18),True,False,no)==("matched",18),
                policy(("full",1),False,True,no)==("full",1),
                policy(("matched",18),True,True,no)==("language_cue",0),
                policy(("matched",18),True,True,[True,True,False,False])==("language_cue",2),
                policy(("matched",18),True,True,[True,True,True,False])==("language_cue",3),
                policy(("matched",18),True,True,yes)==("matched",18),
                policy(("complete",None),True,True,no)==("language_cue",0),
                command_for("matched",18,specs)==previous.command_for("matched",18,specs),
                CASES==[("within_medical","probe"),("cross_short","probe"),("within_medical","full"),("cross_short","full")],
                previous.prior.PRIORITY[:8]==[0,6,12,18,24,30,36,42]]
        assert all(checks)
        action,state=next_action(specs,shared,cue)
        write(REPORT/"CPU_VALIDATION.json",{"passed":True,"CPU_only":True,"inherited_checks":11,
            "new_checks":len(checks),"entrypoint_sha256":sha(Path(__file__)),"cases":CASES,
            "cue_plan_sha256":sha(CUE/"CPU_PLAN_CHECK.json"),"current_action":action,"state":state,"GPU_started":False})
        print(json.dumps({"CPU_only":True,"checks":23,"current_action":action,"GPU_started":False}));return
    auth=json.loads((REPORT/"AUTHORIZATION.json").read_text())
    assert auth["authorization_basis"]=="user_delegated_GPU1_research_plan"
    assert auth["entrypoint_sha256"]==sha(Path(__file__)) and auth["order"]==ORDER
    assert auth["cue_plan_sha256"]==sha(CUE/"CPU_PLAN_CHECK.json")
    lock=(REPORT/"coordinator.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        (kind,target),state=next_action(specs,shared,cue)
        write(REPORT/"queue_state.json",{"phase":kind,"target":target,"pid":os.getpid(),"state":state,"GPU1_only":True})
        if kind=="wait":time.sleep(30);continue
        if kind=="complete":return
        command,log=command_for(kind,target,specs)
        with log.open("a") as output:
            child=subprocess.Popen(["/usr/bin/nohup",str(ROOT/".vision-env/bin/python"),"-u",*command],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES="1",OMP_NUM_THREADS="6",MKL_NUM_THREADS="6",PYTHONUNBUFFERED="1"),
                stdout=output,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write(REPORT/"queue_state.json",{"phase":"running","kind":kind,"target":target,"pid":os.getpid(),"worker_pid":child.pid,"GPU1_only":True})
            code=child.wait()
        assert code==0,"Worker failure: keep evidence and stop"
        if kind=="language_cue":assert cue_complete(CASES[target],cue)
        elif kind=="acquired_anchor":assert previous.shared_complete(target,shared)
        elif kind=="full":assert original.previous.prior.complete_full(target)
        elif kind=="pilot":assert original.previous.pilot_complete(specs[1]["cases"][target],specs[1])
        elif kind=="matched":assert original.previous.prior.complete_case(specs[0]["cases"][target],specs[0])
        elif kind=="training":assert original.full_training(target,specs[2]) in ("valid","invalid")
        else:assert original.confirmation_complete(target,kind,specs[3])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--check",action="store_true");args=p.parse_args()
    try:main(args.check)
    except Exception:
        write(REPORT/"needs_attention.json",{"pid":os.getpid(),"traceback":traceback.format_exc()});raise
