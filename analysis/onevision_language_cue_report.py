"""Read-only CPU report: every fixed cue, raw Answer F1, no best-cue selection."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import json
import math
import time
from pathlib import Path

from onevision_language_causal_queue import CASES, CUE, ROOT, cue_complete, cue_plan, sha, write
from onevision_matched_task_domain_report import checked_rows, paired
from onevision_matched_language_profile import has_han


def build():
    spec=cue_plan()
    f=spec["files"]["test"];assert sha(ROOT/f["path"])==f["sha256"]
    original={r["id"]:r for r in map(json.loads,(ROOT/f["path"]).read_text().splitlines())}
    raw={};cells=[];proofs={}
    for seq,scope in CASES:
        if not cue_complete((seq,scope),spec):continue
        p=CUE/f"T04_{seq}_{scope}_complete.json";proofs[str(p.relative_to(ROOT))]=sha(p)
        v=json.loads(p.read_text());run=ROOT/v["run"]
        ids=spec["probe_ids"] if scope=="probe" else spec["full_ids"]
        groups={"all":list(range(len(ids)))}
        for field,values in [("q_lang",["en","zh"]),("answer_type",["OPEN","CLOSED"])]:
            for value in values:groups[field+"="+value]=[i for i,ident in enumerate(ids) if original[ident]["metadata"][field]==value]
        for view in spec["views"]:
            rows=checked_rows(run/f"evaluations/{view}/anchor_image.jsonl",ids,original)
            raw[seq,scope,view]=rows
            entry={"sequence":seq,"scope":scope,"view":view,"groups":{}}
            for g,inds in groups.items():
                rr=[rows[i] for i in inds]
                entry["groups"][g]={"QA":len(rr),"Answer_F1":math.fsum(r["scores"]["token_f1"] for r in rr)/len(rr),
                    "predictions_contain_Han":sum(has_han(r["prediction"]) for r in rr),
                    "empty_predictions":sum(not r["prediction"].strip() for r in rr)}
            cells.append(entry)
    changes=[];gap_interactions=[]
    for scope in ["probe","full"]:
        ids=spec["probe_ids"] if scope=="probe" else spec["full_ids"]
        for seq in ["within_medical","cross_short"]:
            if (seq,scope,"original") not in raw:continue
            for view in spec["views"][1:]:
                for lang in ["all","en","zh"]:
                    inds=[i for i,ident in enumerate(ids) if lang=="all" or original[ident]["metadata"]["q_lang"]==lang]
                    treatment,control=raw[seq,scope,view],raw[seq,scope,"original"]
                    delta=[treatment[i]["scores"]["token_f1"]-control[i]["scores"]["token_f1"] for i in inds]
                    changes.append({"scope":scope,"sequence":seq,"view":view,"language":lang,
                        "cue_minus_original_F1":paired(delta,[original[ids[i]]["images"] for i in inds])})
        if not all((seq,scope,"original") in raw for seq in ["within_medical","cross_short"]):continue
        for view in spec["views"][1:]:
            for lang in ["all","en","zh"]:
                inds=[i for i,ident in enumerate(ids) if lang=="all" or original[ident]["metadata"]["q_lang"]==lang]
                delta=[]
                for i in inds:
                    d={seq:raw[seq,scope,view][i]["scores"]["token_f1"]-raw[seq,scope,"original"][i]["scores"]["token_f1"] for seq in ["within_medical","cross_short"]}
                    delta.append(d["cross_short"]-d["within_medical"])
                gap_interactions.append({"scope":scope,"view":view,"language":lang,
                    "cross_minus_within_cue_effect_F1":paired(delta,[original[ids[i]]["images"] for i in inds])})
    return {"CPU_only":True,"complete_runs":len(proofs),"required_runs":4,"cells":cells,"cue_effects":changes,
        "cross_within_cue_interactions":gap_interactions,"completion_sha256":proofs,
        "report_source_sha256":{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            "analysis/onevision_matched_task_domain_report.py":sha(ROOT/"analysis/onevision_matched_task_domain_report.py"),
            "analysis/onevision_matched_language_profile.py":sha(ROOT/"analysis/onevision_matched_language_profile.py")},
        "scope":"post-hoc fixed-weight instruction intervention; all predefined views, not new CL method or main baseline setting",
        "limits":["Probe32 overlaps full507 and is not independent replication; never pooled.",
            "All8 full cells required regardless probe results; no TEST-based winner or setting selection.",
            "Literal Han presence is not language accuracy; existing raw Answer F1 unchanged.",
            "Paired image intervals condition on these fixed weights; single seed, uncorrected multiple comparisons.",
            "Recovery without new knowledge would isolate an elicitation component, not prove all knowledge intact.",
            "No improvement would reject only these fixed cues, not every alignment or knowledge explanation."]}


def once():
    r=build();write(CUE/"VERIFIED_EVIDENCE.json",r)
    lines=["# 固定权重：语言/简答指令干预", "",f"完整运行证明：{r['complete_runs']}/4。缺失为待测，不是零分。", "",
           "| 范围 | 权重 | 指令视图 | 英文F1 | 中文F1 | 中文预测含汉字 |", "|---|---|---|---:|---:|---:|"]
    for c in r["cells"]:
        en,zh=c["groups"]["q_lang=en"],c["groups"]["q_lang=zh"]
        lines.append(f"| {c['scope']} | {c['sequence']} | {c['view']} | {en['Answer_F1']:.3f} | {zh['Answer_F1']:.3f} | {zh['predictions_contain_Han']}/{zh['QA']} |")
    lines += ["", "原始QA、图像、权重与Answer F1均不改；四种指令全部报告。完整范围507题，每个权重4视图；probe32仅诊断。没有GPU完成证明时不报告干预效果。", "",
              "全部配对效应/区间、跨域与同域的指令效应差及源SHA见[JSON](VERIFIED_EVIDENCE.json)。", ""]
    lines += ["- "+x for x in r["limits"]]
    (CUE/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    write(CUE/"observer_status.json",{"pid":os.getpid(),"CPU_only":True,"complete_runs":r["complete_runs"],"required_runs":4})
    return r


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--watch",action="store_true");args=p.parse_args()
    while True:
        r=once()
        if not args.watch or r["complete_runs"]==4:break
        time.sleep(30)
