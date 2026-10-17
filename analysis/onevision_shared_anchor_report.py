"""CPU observer: full-anchor final minus the exact acquired-stage1 model.

All missing cells remain unreported, not zero. No training, routing, choices,
old predictions or existing report sources are modified.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import fcntl
import json
import math
import time
import traceback
from pathlib import Path

import torch
from onevision_matched_task_domain_report import sha, checked_rows, transition, checkpoint

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"summary/onevision_diagnostics_20261016"
REPORT=BASE/"shared_anchor_initial"
MATCHED=BASE/"matched_task_domain"


def write(p,x):
    t=p.with_suffix(p.suffix+".observer.tmp");t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n");t.replace(p)


def exact_rows(a,b):
    fields=["id","references","prediction","scores","nll"]
    assert [[r[k] for k in fields] for r in a]==[[r[k] for k in fields] for r in b],"Exact same-state predictions/NLL mismatch"


def self_check():
    a=[{"id":"x","references":["a"],"prediction":"a","scores":{"token_f1":100},"nll":.5}]
    b=[dict(a[0],prediction="different")]
    for changed in [b,[dict(a[0],nll=math.nextafter(.5,1.))]]:
        try:exact_rows(a,changed)
        except AssertionError:pass
        else:raise AssertionError("Different prediction or one-ULP NLL was accepted")
    exact_rows(a,a);assert not torch.cuda.is_initialized()
    return {"equal_score_unequal_prediction_rejected":True,"one_ULP_NLL_rejected":True,"CPU_only":True}


def build():
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    def check(p,h=None):
        actual=sha(p);assert h is None or actual==h,str(p)
        sources[str(p.relative_to(ROOT))]=actual;return p
    gate=json.loads(check(REPORT/"CPU_PLAN_CHECK.json").read_text());assert gate["passed"] and gate["CPU_only"]
    spec=gate["spec"];parent=spec["parent_spec"]
    for p,h in spec["source_sha256"].items():check(ROOT/p,h)
    check(ROOT/"analysis/onevision_matched_task_domain_report.py")
    lock=json.loads(check(ROOT/"summary/primary_metric_20261017/metric_lock.json").read_text());assert not lock["option_mapping"]
    check(ROOT/"analysis/answer_f1.py",lock["metric_sha256"])
    original={}
    for values in spec["inputs"].values():
        f=values["test_file"]
        original.update({r["id"]:r for r in map(json.loads,check(ROOT/f["path"],f["sha256"]).read_text().splitlines())})
    def artifacts(proof,root):
        run=(ROOT/proof["run"]).resolve();assert run.is_relative_to(ROOT/root)
        for p,h in proof["artifact_sha256"].items():
            path=(run/p).resolve();assert path.is_relative_to(run);check(path,h)
        return run
    def same_state(a,b):
        assert a.keys()==b.keys() and all(torch.equal(a[n][f],b[n][f]) for n in a for f in ("A","B"))
    starts=[];comparisons=[]
    for anchor in [4,2]:
        p=REPORT/f"T{anchor:02d}_complete.json"
        if not p.exists():continue
        proof=json.loads(check(p).read_text());assert proof["complete"] and proof["spec"]==spec and proof["parameters_unchanged"]
        assert proof["anchor"]==anchor and proof["optimizer_updates"]==0
        run=artifacts(proof,"exp/onevision_shared_anchor_initial_20261017")
        canonical=MATCHED/f"canonical_anchor{anchor}.pt";check(canonical,proof["canonical_sha256"])
        c=checkpoint(canonical);assert c["source_sha256"]==parent["source_sha256"]
        rg=json.loads((run/"reload_exact_gate.json").read_text());assert rg["passed"] and rg["no_tolerance"] and rg["QA"]==32
        ids=spec["inputs"][str(anchor)]["full_ids"]
        initial=checked_rows(run/"evaluations/full_acquired_anchor/anchor_image.jsonl",ids,original)
        assert proof["full_anchor_QA"]==len(initial)
        probes=spec["inputs"][str(anchor)]["reload_gate_ids"]
        reload=checked_rows(run/"evaluations/reload_gate/anchor_image.jsonl",probes,original)
        by_id={r["id"]:r for r in initial};exact_rows(reload,[by_id[i] for i in probes])
        parent_path=MATCHED/f"T{anchor:02d}_repeat_anchor__SeqFT_LoRA_complete.json"
        pp=json.loads(check(parent_path,proof["parent_completion_sha256"]).read_text());assert pp["complete"] and pp["spec"]==parent
        parent_run=artifacts(pp,"exp/onevision_matched_task_domain_20261017")
        old=checked_rows(parent_run/"evaluations/anchor_after1/anchor_image.jsonl",probes,original);exact_rows(reload,old)
        same_state(checkpoint(parent_run/"adapters/stage_01.pt")["adapter"],c["adapter"])
        starts.append({"anchor":anchor,"full_QA":len(initial),"Answer_F1":sum(r["scores"]["token_f1"] for r in initial)/len(initial),
                       "NLL":sum(r["nll"] for r in initial)/len(initial),"original_and_reload_and_full_overlap32_exact":True,
                       "canonical_sha256":proof["canonical_sha256"],"run":proof["run"]})
        for sequence in parent["sequences"][str(anchor)]:
            p=MATCHED/f"T{anchor:02d}_{sequence}__SeqFT_LoRA_complete.json"
            if not p.exists():continue
            final_proof=json.loads(check(p).read_text());assert final_proof["complete"] and final_proof["spec"]==parent and final_proof["stage1_exact"]
            final_run=artifacts(final_proof,"exp/onevision_matched_task_domain_20261017")
            same_state(checkpoint(final_run/"adapters/stage_01.pt")["adapter"],c["adapter"])
            final=checked_rows(final_run/"evaluations/final_native_anchor/anchor_image.jsonl",ids,original)
            comparisons.append({"anchor":anchor,"sequence":sequence,"full_acquired_to_final":transition(initial,final,original),
                                "same_acquired_stage1_parameters_exact":True,"final_run":final_proof["run"]})
    for p,h in sources.items():assert sha(ROOT/p)==h
    assert not torch.cuda.is_initialized()
    result={"CPU_only":True,"complete_acquired_anchors":len(starts),"planned_acquired_anchors":2,
            "completed_full_reference_pairs":len(comparisons),"planned_full_reference_pairs":8,
            "acquired_starts":starts,"comparisons":comparisons,"source_sha256":sources,"self_checks":self_check(),
            "primary_metric":"Answer F1; original raw token_f1, no answer mapping",
            "limits":["Full native medical anchors507/228 only; not full21454 or19-task FWT/BWT.",
                      "Common acquired stage1 differs from untrained base and original19-stage one-epoch protocol.",
                      "Follow-up on already observed test sets, not untouched confirmation or TEST-selected model choices.",
                      "Intervals conditional on fixed trained models; no seed/order replication, uncorrected image bootstrap.",
                      "Final-minus-acquired measures net change; cannot uniquely attribute it to knowledge, alignment or interference.",
                      "Missing starts/finals omitted, not zero; original runs and weights remain immutable."]}
    write(REPORT/"VERIFIED_RESULTS.json",result)
    lines=["# 完整旧锚：最终成绩减去真实共同取得点", "",
           f"已核验共同起点{len(starts)}/2，完整净变化配对{len(comparisons)}/8。未完成不填0。F1唯一主指标，NLL辅助。", "",
           "| 锚/序列 | 完整题数 | 共同S1 F1→最终F1 | F1净变化及95%图像区间 |", "|---|---:|---:|---:|"]
    for row in comparisons:
        v=row["full_acquired_to_final"];delta=v["delta_Answer_F1"];lo,hi=delta["image_cluster_95CI"]
        lines.append(f"| T{row['anchor']:02d}/{row['sequence']} | {v['QA']} | {v['Answer_F1_before']:.3f}→{v['Answer_F1_after']:.3f} | {delta['mean']:+.3f} [{lo:+.3f},{hi:+.3f}] |")
    lines += ["", "先区分跨域自身净变化与同域净变化，再解释两者差距；不能把相对同域的差距全部叫遗忘。", "",
              "逐题质量变化、配对NLL、参数/输入/预测SHA见[JSON](VERIFIED_RESULTS.json)。", ""]
    lines += ["- "+v for v in result["limits"]]
    (REPORT/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");p.add_argument("--check",action="store_true");args=p.parse_args()
    torch.set_num_threads(2);REPORT.mkdir(exist_ok=True)
    if args.check:print(json.dumps(self_check()));return
    if args.watch:
        lock=(REPORT/"observer.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=None
    while True:
        files=[*REPORT.glob("T*_complete.json"),*MATCHED.glob("T*__SeqFT_LoRA_complete.json")]
        signature=sorted((str(p),p.stat().st_mtime_ns) for p in files)
        if signature!=previous:
            r=build();previous=signature
            write(REPORT/"observer_status.json",{"pid":os.getpid(),"CPU_only":True,"completed_starts":r["complete_acquired_anchors"],"completed_pairs":r["completed_full_reference_pairs"],"phase":"complete" if r["completed_full_reference_pairs"]==8 else "waiting_completed_proofs"})
            print(json.dumps({"complete_starts":r["complete_acquired_anchors"],"complete_pairs":r["completed_full_reference_pairs"]}),flush=True)
            if r["completed_full_reference_pairs"]==8:return
        if not args.watch:return
        time.sleep(45)


if __name__=="__main__":
    try:main()
    except Exception:
        REPORT.mkdir(exist_ok=True);write(REPORT/"reporter_needs_attention.json",{"pid":os.getpid(),"traceback":traceback.format_exc(),"CPU_only":True});raise
