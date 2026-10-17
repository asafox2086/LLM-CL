"""New CPU dispatcher: complete all predeclared SeqFT domain controls first.

Only scheduling inside the unchanged48-case study changes. No GPU source,
choice, sample, model setting, original result or observer is modified.
"""
import argparse
import json
import os
import traceback
from pathlib import Path

import onevision_causal_before_full_method_queue as previous

original = previous.original
ROOT = original.ROOT
REPORT = original.BASE / "causal_anchor_first_queue"
ORDER = previous.ORDER
PRIMARY = [0, 6, 12, 18, 24, 30, 36, 42]
PRIORITY = PRIMARY + [i for i in range(48) if i not in PRIMARY]


def choose(cap1, pilots, training, final, matched, cap3, initial, sparse, busy=False):
    action = previous.choose(cap1, pilots, training, final, matched, cap3, initial, sparse, busy)
    if action[0] != "matched":
        return action
    return next(("matched", i) for i in PRIORITY if not matched[i])


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    original.REPORT = REPORT
    original.ORDER = ORDER
    original.choose = choose
    specs = original.plans()
    assert len(PRIORITY) == len(set(PRIORITY)) == 48 and set(PRIORITY) == set(range(48))
    assert [i for i, case in enumerate(specs[0]["cases"]) if case[2] == "SeqFT_LoRA"] == PRIMARY
    assert specs[0]["cases"][24] == [2, "repeat_anchor", "SeqFT_LoRA"]
    if check:
        original.check(specs)
        m = [True]*10+[False]*38
        t, f, p, s = ["pending"]*4, [False]*4, [True]*12, [False]*4
        checks = []
        for expected in [12, 18, 24, 30, 36, 42]:
            checks.append(choose(True,p,t,f,m,False,False,s) == ("matched",expected))
            m[expected] = True
        checks.append(choose(True,p,t,f,m,False,False,s) == ("matched",10))
        checks.append(choose(True,p,t,f,m,False,False,s,True) == ("wait",None))
        checks.append(choose(True,p,t,f,[True]*48,False,False,s) == ("training","control"))
        checks.append(choose(True,[False]*12,t,f,[False]*48,False,False,s) == ("pilot",0))
        assert all(checks)
        original.previous.prior.write(REPORT/"CPU_PRIORITY_VALIDATION.json", {
            "passed":True,"CPU_only":True,"inherited_checks":11,"priority_checks":len(checks),
            "all48_exactly_once":True,"primary_case_indices":PRIMARY,"matched_priority":PRIORITY,
            "entrypoint_sha256":original.previous.prior.sha(Path(__file__)),
            "immutable_library_sha256":original.previous.prior.sha(ROOT/"analysis/onevision_two_module_fullqueue.py"),
            "no_TEST_score_used":True,"GPU_started":False})
        print(json.dumps({"CPU_only":True,"checks":21,"GPU_started":False}));return
    auth = json.loads((REPORT/"AUTHORIZATION.json").read_text())
    assert auth["approved_by_user"] and auth["order"] == ORDER and auth["matched_priority"] == PRIORITY
    assert auth["entrypoint_sha256"] == original.previous.prior.sha(Path(__file__))
    original.main()


if __name__ == "__main__":
    p = argparse.ArgumentParser();p.add_argument("--check",action="store_true");args=p.parse_args()
    try:
        main(args.check)
    except Exception:
        original.previous.prior.write(REPORT/"needs_attention.json",{"pid":os.getpid(),"traceback":traceback.format_exc()})
        raise
