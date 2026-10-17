"""CPU-only author-language/answer-type profiling of complete anchor outputs.

Raw Answer F1 stays unchanged. No translation, option mapping, prefix stripping,
semantic rescoring, data filtering, model choice or GPU work is performed.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from onevision_matched_task_domain_report import checked_rows, paired

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"summary/onevision_diagnostics_20261016/matched_task_domain"


def has_han(text):
    return any("\u3400"<=c<="\u9fff" for c in text)


def interaction(delta,langs,images):
    groups=collections.defaultdict(list)
    for d,l,im in zip(delta,langs,images):groups[tuple(im)].append((d,l))
    def statistic(clusters):
        values={l:[d for group in clusters for d,lang in group if lang==l] for l in ["en","zh"]}
        assert all(values.values()),"Undefined language contrast in bootstrap draw"
        return math.fsum(values["zh"])/len(values["zh"])-math.fsum(values["en"])/len(values["en"])
    clusters=list(groups.values());rng=random.Random(42)
    boot=sorted(statistic([clusters[rng.randrange(len(clusters))] for _ in clusters]) for _ in range(2000))
    return {"zh_minus_en_difference_in_changes_pp":statistic(clusters),"image_cluster_95CI":[boot[49],boot[1949]],
            "image_clusters":len(clusters),"scope":"descriptive effect modification across author language cohorts; not causal mediation"}


def self_check():
    toy=interaction([1.,4.,1.,4.],["en","zh","en","zh"],[["a"],["a"],["b"],["b"]])
    assert toy["zh_minus_en_difference_in_changes_pp"]==3 and toy["image_cluster_95CI"]==[3,3]
    assert has_han("Answer: 肺") and not has_han("The answer is lungs")
    assert not torch.cuda.is_initialized()
    return {"shared_image_language_interaction_preserved":True,"literal_Han_flag_only":True,"CPU_only":True}


def build(anchor):
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    def check(p,h=None):
        actual=sha(p);assert h is None or h==actual,str(p)
        sources[str(p.relative_to(ROOT))]=actual;return p
    plan=json.loads(check(BASE/"CPU_PLAN_CHECK.json").read_text());assert plan["passed"]
    spec=plan["spec"]
    for p,h in spec["source_sha256"].items():check(ROOT/p,h)
    check(ROOT/"analysis/onevision_matched_task_domain_report.py")
    lock=json.loads(check(ROOT/"summary/primary_metric_20261017/metric_lock.json").read_text());assert not lock["option_mapping"]
    check(ROOT/"analysis/answer_f1.py",lock["metric_sha256"])
    m=json.loads(check(ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json",spec["manifest_sha256"]).read_text())
    f=m["tasks"][anchor-1]["files"]["test"]
    original_rows=[json.loads(s) for s in check(ROOT/f["path"],f["sha256"]).read_text().splitlines()]
    ids=[r["id"] for r in original_rows];assert ids==spec["final_SeqFT_full_anchor_ids"][str(anchor)]
    original={r["id"]:r for r in original_rows}
    assert {r["metadata"]["q_lang"] for r in original_rows}=={"en","zh"}
    assert {r["metadata"]["answer_type"] for r in original_rows}=={"OPEN","CLOSED"}
    groups={"all":list(range(len(ids)))}
    for lang in ["en","zh"]:groups[f"q_lang={lang}"]=[i for i,r in enumerate(original_rows) if r["metadata"]["q_lang"]==lang]
    for typ in ["OPEN","CLOSED"]:groups[f"answer_type={typ}"]=[i for i,r in enumerate(original_rows) if r["metadata"]["answer_type"]==typ]
    for lang in ["en","zh"]:
        for typ in ["OPEN","CLOSED"]:
            groups[f"q_lang={lang}/answer_type={typ}"]=[i for i,r in enumerate(original_rows) if r["metadata"]["q_lang"]==lang and r["metadata"]["answer_type"]==typ]
    train={}
    for task in m["tasks"]:
        t=task["task"]
        if str(t) not in spec["TRAIN_ids"]:continue
        ff=task["files"]["train"]
        train[t]={r["id"]:r for r in map(json.loads,check(ROOT/ff["path"],ff["sha256"]).read_text().splitlines())}
    observations={};profiles=[];train_profiles=[];repeat_reference=None;diagnostic=[];diagnostic_final={};common_fields=None
    def artifacts(proof,parent):
        run=(ROOT/proof["run"]).resolve();assert run.is_relative_to(ROOT/parent)
        for rel,h in proof["artifact_sha256"].items():
            path=(run/rel).resolve();assert path.is_relative_to(run);check(path,h)
        return run
    def profile(label,rows):
        observations[label]=rows;entry={"condition":label,"groups":{}}
        for group,inds in groups.items():
            rr=[rows[i] for i in inds]
            if not rr:entry["groups"][group]=None;continue
            counts=collections.Counter(r["prediction"] for r in rr)
            entry["groups"][group]={"QA":len(rr),"Answer_F1":math.fsum(r["scores"]["token_f1"] for r in rr)/len(rr),
                "NLL":math.fsum(r["nll"] for r in rr)/len(rr),"predictions_contain_Han":sum(has_han(r["prediction"]) for r in rr),
                "original_gold_contains_Han":sum(any(has_han(a) for a in original[r["id"]]["answers"]) for r in rr),
                "empty_predictions":sum(not r["prediction"].strip() for r in rr),
                "literal_prefix_hits":sum(r["prediction"].lower().startswith(("the answer is","answer:")) for r in rr),
                "raw_distinct_predictions":len(counts),"raw_top6_predictions":counts.most_common(6)}
        profiles.append(entry)
    for sequence,tasks in spec["sequences"][str(anchor)].items():
        name=f"T{anchor:02d}_{sequence}__SeqFT_LoRA";path=BASE/(name+"_complete.json")
        if not path.exists():continue
        p=json.loads(check(path).read_text());assert p["complete"] and p["stage1_exact"] and p["spec"]==spec
        run=artifacts(p,"exp/onevision_matched_task_domain_20261017")
        rows=checked_rows(run/"evaluations/final_native_anchor/anchor_image.jsonl",ids,original);profile(sequence,rows)
        probe_ids=spec["anchor_probe_ids"][str(anchor)]
        probes={stage:checked_rows(run/f"evaluations/anchor_after{stage}/anchor_image.jsonl",probe_ids,original) for stage in [1,3,5]}
        if common_fields is not None:
            fields=["id","references","prediction","nll","scores"]
            assert [[r[k] for k in fields] for r in probes[1]]==common_fields
        else:
            common_fields=[[r[k] for k in ["id","references","prediction","nll","scores"]] for r in probes[1]]
        entry={"sequence":sequence,"QA":len(probe_ids),"groups":{},"common_stage1_predictions_scores_NLL_exact":True,
               "common_stage1_fields_sha256":hashlib.sha256(json.dumps(common_fields,ensure_ascii=False,sort_keys=True).encode()).hexdigest()}
        for lang in ["en","zh"]:
            inds=[i for i,ident in enumerate(probe_ids) if original[ident]["metadata"]["q_lang"]==lang]
            if not inds:entry["groups"][lang]=None;continue
            stage_scores={str(s):{"QA":len(inds),"Answer_F1":math.fsum(probes[s][i]["scores"]["token_f1"] for i in inds)/len(inds),
                                 "predictions_contain_Han":sum(has_han(probes[s][i]["prediction"]) for i in inds)} for s in [1,3,5]}
            delta=[probes[5][i]["scores"]["token_f1"]-probes[1][i]["scores"]["token_f1"] for i in inds]
            entry["groups"][lang]={"stages":stage_scores,"stage1_to5_F1":paired(delta,[original[probe_ids[i]]["images"] for i in inds])}
        diagnostic.append(entry);diagnostic_final[sequence]=probes[5]
        if sequence=="repeat_anchor":
            repeat_reference=checked_rows(run/"evaluations/anchor_after1/anchor_image.jsonl",spec["anchor_probe_ids"][str(anchor)],original)
        by_stage=[]
        for stage,task in enumerate(tasks,1):
            tr=json.loads((run/f"training/{name}/stage_{stage:02d}.json").read_text())
            actual=[i for e in tr["events"] for i in e["train_ids"]];assert actual==spec["TRAIN_ids"][str(task)]
            rr=[train[task][i] for i in actual]
            by_stage.append({"stage":stage,"task":task,"QA_exposures":len(rr),
                "author_q_lang":dict(collections.Counter(r["metadata"].get("q_lang","not_annotated") for r in rr)),
                "questions_contain_Han":sum(has_han(r["question"]) for r in rr),
                "gold_answers_contain_Han":sum(any(has_han(a) for a in r["answers"]) for r in rr)})
        train_profiles.append({"sequence":sequence,"stages":by_stage,
                               "post_shared_stage1_Han_question_exposures":sum(r["questions_contain_Han"] for r in by_stage[1:])})
    shared=ROOT/"summary/onevision_diagnostics_20261016/shared_anchor_initial"/f"T{anchor:02d}_complete.json"
    if shared.exists():
        p=json.loads(check(shared).read_text());assert p["complete"] and p["spec"]["parent_spec"]==spec and p["optimizer_updates"]==0 and p["parameters_unchanged"]
        run=artifacts(p,"exp/onevision_shared_anchor_initial_20261017")
        check(BASE/f"canonical_anchor{anchor}.pt",p["canonical_sha256"])
        initial=checked_rows(run/"evaluations/full_acquired_anchor/anchor_image.jsonl",ids,original)
        probes=spec["anchor_probe_ids"][str(anchor)]
        reload=checked_rows(run/"evaluations/reload_gate/anchor_image.jsonl",probes,original)
        full_map={r["id"]:r for r in initial};fields=["id","references","prediction","scores","nll"]
        assert repeat_reference is not None
        assert [[r[k] for k in fields] for r in reload]==[[r[k] for k in fields] for r in repeat_reference]==[[full_map[i][k] for k in fields] for i in probes]
        profile("common_acquired_stage1",initial)
    contrasts=[];diagnostic_contrasts=[]
    pairs=[("cross_short","within_medical"),("cross_five_domains","within_medical"),("cross_short","repeat_anchor"),("cross_five_domains","cross_short"),("within_medical","repeat_anchor")]
    if "common_acquired_stage1" in observations:pairs += [(s,"common_acquired_stage1") for s in spec["sequences"][str(anchor)]]
    images=[r["images"] for r in original_rows];langs=[r["metadata"]["q_lang"] for r in original_rows]
    for treatment,control in pairs:
        if treatment not in observations or control not in observations:continue
        a,b=observations[treatment],observations[control]
        delta=[x["scores"]["token_f1"]-y["scores"]["token_f1"] for x,y in zip(a,b)]
        item={"treatment":treatment,"control":control,"strata":{},"arithmetic_contributions_pp":{}}
        for group,inds in groups.items():
            if not inds:item["strata"][group]=None;continue
            item["strata"][group]=paired([delta[i] for i in inds],[images[i] for i in inds])
        for lang in ["en","zh"]:
            inds=groups[f"q_lang={lang}"]
            item["arithmetic_contributions_pp"][lang]=math.fsum(delta[i] for i in inds)/len(ids)
        assert abs(math.fsum(item["arithmetic_contributions_pp"].values())-math.fsum(delta)/len(ids))<1e-10
        item["language_interaction"]=interaction(delta,langs,images)
        contrasts.append(item)
        if treatment in diagnostic_final and control in diagnostic_final:
            aa,bb=diagnostic_final[treatment],diagnostic_final[control]
            dc={"treatment":treatment,"control":control,"scope":"same original32QA; not full507 or independent replication","groups":{}}
            for lang in ["en","zh"]:
                inds=[i for i,row in enumerate(aa) if original[row["id"]]["metadata"]["q_lang"]==lang]
                if inds:dc["groups"][lang]=paired([aa[i]["scores"]["token_f1"]-bb[i]["scores"]["token_f1"] for i in inds],[original[aa[i]["id"]]["images"] for i in inds])
            diagnostic_contrasts.append(dc)
    for p,h in sources.items():assert sha(ROOT/p)==h
    assert not torch.cuda.is_initialized()
    return {"CPU_only":True,"anchor":anchor,"full_QA":len(ids),"completed_final_conditions":len(train_profiles),
            "shared_acquired_stage1_complete":shared.exists(),"profiles":profiles,"TRAIN_language_profiles":train_profiles,
            "contrasts":contrasts,"diagnostic32_trajectories":diagnostic,"diagnostic32_final_contrasts":diagnostic_contrasts,
            "source_sha256":sources,"self_checks":self_check(),
            "metric":"existing raw Answer F1 unchanged",
            "limits":["Post-hoc author-language/type strata; all predefined groups included, not a new main metric.",
                      "Han/prefix flags are literal character statistics, not language-following accuracy or semantic correctness.",
                      "No translation, option mapping, prefix stripping, alternative scoring, training/ID/prediction edits.",
                      "QA-weighted contribution is arithmetic decomposition, not causal percentage explained by language.",
                      "Language cohorts differ in questions/answers; image clustering does not establish translation-pair equivalence.",
                      "TRAIN task/domain/language/content shifts are confounded; prompt intervention and common fullS1 still needed.",
                      "Single anchor/seed/sequence; paired uncorrected image intervals condition on fixed models.",
                      "Missing full commonS1 is not replaced by32QA; no full21454 or general method superiority claims."]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--anchor",type=int,default=4,choices=[4,2]);p.add_argument("--check",action="store_true");args=p.parse_args();torch.set_num_threads(2)
    if args.check:print(json.dumps(self_check()));return
    r=build(args.anchor);out=BASE/"language_profile";out.mkdir(exist_ok=True)
    name=f"T{args.anchor:02d}_EVIDENCE.json";(out/name).write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n")
    lines=["# 最终完整旧锚：领域替换与语言分布混杂", "",
           "作者q_lang/answer_type只用于分层，原始Answer F1、问答与预测完全不改；没有翻译、答案映射或前缀剥离。", "",
           "| 条件 | 英文QA/F1 | 中文QA/F1 | 中文题预测含汉字 | 中文题空输出 |", "|---|---:|---:|---:|---:|"]
    for x in r["profiles"]:
        en,zh=x["groups"]["q_lang=en"],x["groups"]["q_lang=zh"]
        lines.append(f"| {x['condition']} | {en['QA']}/{en['Answer_F1']:.3f} | {zh['QA']}/{zh['Answer_F1']:.3f} | {zh['predictions_contain_Han']}/{zh['QA']} | {zh['empty_predictions']} |")
    lines += ["", "## 跨域−同域分层差异", ""]
    for x in r["contrasts"]:
        if x["control"]!="within_medical":continue
        for lang in ["en","zh"]:
            v=x["strata"][f"q_lang={lang}"];lo,hi=v["image_cluster_95CI"]
            lines.append(f"- {x['treatment']}, {lang}: F1差{v['mean']:+.3f}，95%图像区间[{lo:+.3f},{hi:+.3f}]，QA加权贡献{x['arithmetic_contributions_pp'][lang]:+.3f}pp。")
        inter=x["language_interaction"];lines.append(f"- 中文差−英文差：{inter['zh_minus_en_difference_in_changes_pp']:+.3f}pp，图像区间{inter['image_cluster_95CI']}；是分层异质性，不是语言因果效应。")
    lines += ["", "## 原32题同队列轨迹：与全锚起点分开", "",
              "| 条件/语言 | 题数 | S1/S3/S5 F1 | S1/S3/S5预测含汉字 |", "|---|---:|---:|---:|"]
    for x in r["diagnostic32_trajectories"]:
        for lang,g in x["groups"].items():
            if not g:continue
            ss=g["stages"];f1="/".join(f"{ss[str(s)]['Answer_F1']:.3f}" for s in [1,3,5]);hh="/".join(str(ss[str(s)]["predictions_contain_Han"]) for s in [1,3,5])
            lines.append(f"| {x['sequence']}/{lang} | {ss['1']['QA']} | {f1} | {hh} |")
    lines += ["", "32题属于完整队列的子集，其语言样本数和题目不同；本表不能补造完整507题S1。全部小样本配对区间见JSON，点估计下降不自动代表显著遗忘。", ""]
    lines += ["", "## 实际TRAIN语言构成", ""]
    for x in r["TRAIN_language_profiles"]:
        lines.append(f"- {x['sequence']}: 共同S1之后32目标QA曝光，问题含汉字{x['post_shared_stage1_Han_question_exposures']}次；逐阶段作者语言/ID源见JSON。")
    lines += ["", "没有汉字不等于回答一定错误；中文题上的英语回答也不能未经语义验证就称知识正确。完整共同S1仍须实测，不能把最终差距全说成已丢失的中文能力。", "",
              "下一可证伪问题是在相同权重/原问题/同一F1下，既有question_language指令能否恢复输出及成绩；尚无该干预结果，不据此宣称知识充足。", "",
              f"[全部分层、原始输出分布、源SHA]({name})", ""]
    lines += ["- "+x for x in r["limits"]]
    (out/f"T{args.anchor:02d}_VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"CPU_only":True,"anchor":args.anchor,"full_QA":r['full_QA'],"complete_final_conditions":r['completed_final_conditions'],"fullS1_complete":r['shared_acquired_stage1_complete']}))


if __name__=="__main__":main()
