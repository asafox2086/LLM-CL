"""Combine SHA-bound CPU snapshots without importing or changing GPU runners."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"summary/onevision_diagnostics_20261016/baseline_language_profile"


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources={};hosts={}
    for host in ["old2080","new3090"]:
        p=REPORT/(host+"_snapshot.json");sources[str(p.relative_to(ROOT))]=sha(p)
        hosts[host]=json.loads(p.read_text());assert hosts[host]["CPU_only"] and hosts[host]["host_label"]==host
        assert hosts[host]["Organ_TRAIN_QA"]==26 and hosts[host]["Organ_TEST_QA"]==507
    old,new=hosts["old2080"]["source_sha256"],hosts["new3090"]["source_sha256"]
    common=set(old)&set(new)
    for key in common:assert old[key]==new[key],key
    assert "analysis/answer_f1.py" in common and "analysis/onevision_v2_language_snapshot.py" in common
    assert sha(ROOT/"analysis/onevision_v2_language_snapshot.py")==old["analysis/onevision_v2_language_snapshot.py"]
    expected=["exp/two_backbone_native_cl_20261017_v2/","data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/"]
    shared_verified=[k for k in sorted(common) if any(k.startswith(prefix) for prefix in expected)]
    assert any(k.endswith("run.py") for k in shared_verified) and any(k.endswith("manifest.json") for k in shared_verified)
    runs=[r for v in hosts.values() for r in v["runs"]];assert len(runs)==6
    initial={}
    for backbone,n in [("onevision",4),("qwen",2)]:
        rr=[r for r in runs if r["backbone"]==backbone];assert len(rr)==n
        hashes=[next(c for c in r["cells"] if c["stage"]==0)["semantic_fields_sha256"] for r in rr]
        assert len(set(hashes))==1,"Initial original outputs differ; do not assume shared functional start"
        initial[backbone]={"methods":n,"exact_initial_prediction_reference_score_fields":True,"semantic_sha256":hashes[0]}
    acquisition=[]
    for r in runs:
        for transition in r["transitions"]:
            if transition["label"]=="current_Organ_acquisition":
                acquisition.append({"backbone":r["backbone"],"method":r["method"],"groups":transition["groups"]})
    evidence={"CPU_only":True,"host_snapshot_sha256":sources,"shared_paths_SHA_exact":shared_verified,
        "initial_function_output_checks":initial,"runs":runs,"current_Organ_acquisition":acquisition,
        "all6_final_Organ_cells_complete":all(next(c for c in r["cells"] if c["stage"]==19)["scores"] is not None for r in runs),
        "collector_sha256":old["analysis/onevision_v2_language_snapshot.py"],"reporter_sha256":sha(Path(__file__)),
        "source_verification_scope":"Collectors verified original hosts at snapshot_unix; merger verifies snapshot SHA consistency.",
        "limits":hosts["old2080"]["limits"]+["Qwen2B vs OneVision0.5B changes more than parameter count; not a causal size/language/knowledge isolation.",
            "SeqFT and ER acquisition are observational two-method checks, not all-method final superiority or significance comparison."]}
    (REPORT/"VERIFIED_EVIDENCE.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    lines=["# 正式V2基线：先前能力、当前习得与后续保持分开", "",
        "只读取同一正式V2运行已封存完整Organ507题；同ID/原金标，复算唯一Answer F1。没有新训练/推理/翻译/重评分，也不替代nativeAnyRes研究的共同S1。", "",
        "## 同一测试队列的三个关键时点", "",
        "S0为未微调，S3是Organ学习前（已学自然/其他医学任务），S4刚学完Organ，S19为最后阶段。待测保持空值。", "",
        "| 骨干/方法 | S0英文/中文F1 | S3学习前英文/中文F1 | S4刚学完英文/中文F1 | S19英文/中文F1 |", "|---|---:|---:|---:|---:|"]
    for r in runs:
        values=[]
        for c in r["cells"]:
            g=c["scores"];values.append("待测" if g is None else f"{g['q_lang=en']['Answer_F1']:.3f}/{g['q_lang=zh']['Answer_F1']:.3f}")
        lines.append("| "+r["backbone"]+" / "+r["method"]+" | "+" | ".join(values)+" |")
    lines += ["", "## 当前Organ阶段实际学到了多少", "",
        "原TRAIN26题（中文15/英文11），有效batch16、2次实际更新、LR为1e-4/5e-5、warmup0。两方法均记录有限非零梯度、前后参数SHA变化。所有前置阶段有效性与实际日志均核验，不只信passed字段。", "",
        "| 方法 | 英文S3→S4变化/95%图像区间 | 中文S3→S4变化/95%图像区间 |", "|---|---:|---:|"]
    for a in acquisition:
        values=[]
        for lang in ["en","zh"]:
            d=a["groups"]["q_lang="+lang]["delta_F1"];lo,hi=d["image_cluster_95CI"]
            values.append(f"{d['mean']:+.3f} [{lo:+.3f},{hi:+.3f}]")
        lines.append("| "+a["method"]+" | "+" | ".join(values)+" |")
    final_count=sum(next(c for c in r["cells"] if c["stage"]==19)["scores"] is not None for r in runs)
    lines += ["", "## 六栏判断", "",
        "- **探究的问题：** 中文低分发生在后续跨域之前，还是已经学会后才丢失？这一现象是否只有SeqFT？",
        "- **做的实验：** 对两机6个实际运行收集S0/S3/S4/S19已封存同队列，逐ID/参考/元数据与原F1精确核验；原始源码快照、清单、训练日志完整前缀、正LR及参数变化留SHA。全部作者语言/题型分层保留，不只报告有利子组。",
        f"- **结果：** OneVision四方法未训练输出逐题精确一致，Qwen两方法亦一致。当前可核验Organ习得配对{len(acquisition)}组、最终点{final_count}/6；每条语言变化及区间见上表，缺失不填0。",
        "- **证明了什么：** 两个已训练基线在后续跨域前就有明显语言分组差距，但中文并非完全不能学习。S0→S4不能全部算当前任务收益，必须看S3→S4；这个实测增量仍不能区分知识、语言/格式适配。英文/中文组不是翻译配对，不能用组间分数直接测语言能力。尚无最终遗忘量，不能把nativeAnyRes跨域终点移入本表。",
        "- **改进的方法：** 先保存学习前/刚学完的完整参照，再判断最终下降；GPU1的固定权重四指令是另一设置的独立机制控制。本审计不改变任何基线算法、训练或评测。",
        "- **改进的效果：** 获得更明确的习得/保持分界，尚无新算法提升；ER训练loss包含回放，不能把它与SeqFT的loss大小直接比较为目标任务拟合优劣。", "",
        "全部9个语言/题型分层、逐题增减数、日志及SHA见[证据JSON](VERIFIED_EVIDENCE.json)。不报告未完成的主表/FWT/BWT或全部方法优劣。", ""]
    lines += ["- "+x for x in evidence["limits"]]
    (REPORT/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"CPU_only":True,"baseline_runs":6,"accepted_acquisition_pairs":len(acquisition),"all_final_cells_complete":evidence["all6_final_Organ_cells_complete"]}))


if __name__=="__main__":main()
