"""CPU TRAIN-only subgroup gradients and measured finite loss transitions.

Use the complete deterministic19 run, never transplant gradients to strict48,
EWC, V2, or another parameter point. No new GPU work or method is introduced.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import collections
import hashlib
import json
import math
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"summary/onevision_diagnostics_20261016"


def main():
    torch.set_num_threads(2)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    def checked(p,expected=None):
        h=sha(p);assert expected is None or h==expected,str(p)
        sources[str(p.relative_to(ROOT))]=h;return p
    proof=json.loads(checked(BASE/"deterministic_native/complete.json").read_text())
    assert proof["complete"] and not proof["formal_full_test"]
    for p,h in proof["source_sha256"].items():checked(ROOT/p,h)
    run=(ROOT/proof["run"]).resolve();assert run.is_relative_to(ROOT/"exp/onevision_deterministic_native_20261016")
    def tensor(rel):
        p=checked(run/rel,proof["artifact_sha256"][rel])
        return torch.load(p,map_location="cpu",weights_only=True)
    protocol=json.loads(checked(run/"protocol.json",proof["artifact_sha256"]["protocol.json"]).read_text())
    manifest_path=ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    manifest=json.loads(checked(manifest_path,protocol["manifest_sha256"]).read_text());train={}
    for task in manifest["tasks"]:
        f=task["files"]["train"]
        for row in map(json.loads,checked(ROOT/f["path"],f["sha256"]).read_text().splitlines()):
            assert row["id"] not in train;train[row["id"]]=(row,task["task"])
    assert len(train)==1313
    raw=[tensor(f"raw_probe_gradients/sequential_T{s:02d}/medical.pt") for s in range(1,20)]
    ids=raw[0]["ids"];assert len(ids)==4
    languages=[train[i][0]["metadata"]["q_lang"] for i in ids]
    tasks=[train[i][1] for i in ids];assert collections.Counter(languages)=={"en":2,"zh":2}
    assert all(r["ids"]==ids and r["parameter_layout"]==raw[0]["parameter_layout"] and r["train_only"] and r["eval_mode_no_dropout"] for r in raw)
    layout=raw[0]["parameter_layout"]
    for r in raw:
        assert list(r["per_example_preclip_gradient"].shape)==[4,540672]
        assert torch.isfinite(r["per_example_preclip_gradient"]).all()
        assert len(r["nll"])==4 and all(math.isfinite(v) for v in r["nll"])
    checkpoints=[tensor(f"adapters/sequential_T{s:02d}.pt") for s in range(1,20)]
    for stage,c in enumerate(checkpoints,1):
        assert c["source_sha256"]==proof["source_sha256"]
        expected={ident for ident,(_,task) in train.items() if task==stage}
        assert set(c["train_ids_in_order"])==expected and len(c["train_ids_in_order"])==len(expected)
    states=[c["adapter"] for c in checkpoints]
    rows=[]
    for stage in range(2,20):
        before,after=states[stage-2:stage];a,b=raw[stage-2:stage];parts=[]
        for key,info in layout.items():
            module,branch,suffix=key.rsplit(".",2);assert suffix=="weight" and branch in ["A","B"]
            d=after[module][branch].double()-before[module][branch].double()
            assert list(d.shape)==info["shape"] and d.numel()==info["count"]
            assert sum(x.numel() for x in parts)==info["offset"]
            parts.append(d.flatten())
        delta=torch.cat(parts);assert delta.numel()==540672 and torch.isfinite(delta).all()
        gradients=a["per_example_preclip_gradient"].double()
        taylor=gradients@delta
        actual=torch.tensor(b["nll"],dtype=torch.float64)-torch.tensor(a["nll"],dtype=torch.float64)
        groups={};means={}
        for lang in ["en","zh"]:
            inds=[i for i,l in enumerate(languages) if l==lang]
            g=gradients[inds].mean(0);assert g.norm()>0;means[lang]=g
            groups[lang]={"TRAIN_QA":len(inds),"ids":[ids[i] for i in inds],
                "mean_gradient_L2":float(g.norm()),"Taylor_delta_nll":float(taylor[inds].mean()),
                "actual_delta_nll":float(actual[inds].mean()),"per_QA_actual_delta_nll":[float(actual[i]) for i in inds]}
        row={"stage":stage,"current_domain":manifest["tasks"][stage-1]["domain"],
            "all_four_probes_acquired_before_transition":max(tasks)<stage,
            "aggregate_Taylor_delta_nll":float(taylor.mean()),"aggregate_actual_delta_nll":float(actual.mean()),
            "groups":groups,"actual_displacement_L2":float(delta.norm()),
            "gradient_language_cosine":float(torch.dot(means["en"],means["zh"])/means["en"].norm()/means["zh"].norm())}
        row["aggregate_improves_but_group_worsens"]=row["aggregate_actual_delta_nll"]<0 and any(g["actual_delta_nll"]>0 for g in groups.values())
        row["Taylor_and_measured_loss_disagree_for_a_group"]=any((g["Taylor_delta_nll"]>0)!=(g["actual_delta_nll"]>0) for g in groups.values())
        assert abs(sum(g["actual_delta_nll"] for g in groups.values())/2-row["aggregate_actual_delta_nll"])<1e-12
        rows.append(row)
    eligible=[r for r in rows if r["all_four_probes_acquired_before_transition"]]
    masked=[r["stage"] for r in eligible if r["aggregate_improves_but_group_worsens"]]
    cross=[r for r in rows if r["stage"]>=12]
    assert all(r["current_domain"]!="medical" for r in cross)
    cross_totals={}
    for lang in ["en","zh"]:
        inds=[i for i,l in enumerate(languages) if l==lang]
        endpoint=sum(raw[-1]["nll"][i]-raw[10]["nll"][i] for i in inds)/2
        total=sum(r["groups"][lang]["actual_delta_nll"] for r in cross)
        assert abs(endpoint-total)<1e-12
        cross_totals[lang]=total
    for p,h in sources.items():assert sha(ROOT/p)==h
    assert not torch.cuda.is_initialized()
    result={"CPU_only":True,"no_TEST_files_or_scores_read":True,"run":proof["run"],
        "TRAIN_ids":ids,"author_languages":languages,"native_task_membership":tasks,
        "all18_transitions":rows,"all_probes_previously_acquired_stages":[r["stage"] for r in eligible],
        "aggregate_improvement_masks_group_loss_stages":masked,"cross_T11_to_T19_actual_TRAIN_NLL_changes":cross_totals,
        "source_sha256":sources,"protocol_version_retained":protocol,
        "limits":["Two fixed TRAIN QA per language, different native tasks; no population language effect, confidence interval or held-out accuracy claim.",
            "All18 transitions reported; only stages after every probe task is acquired enter old-function masking count.",
            "Same exact saved parameter points only, never reused as strict48/V2/EWC gradients.",
            "Taylor is a local prediction; measured NLL before/after is separately stored and may have a different sign.",
            "Opposite subgroup movements do not identify language as cause; question/task/answer content remains confounded.",
            "These are eval-mode TRAIN probe gradients, not historical optimizer gradients or proof projection would fix accuracy.",
            "No randomized method/weight intervention; not a new mathematical discovery, algorithm improvement, or domain-count causal law.",
            "Cross-segment Chinese TRAIN NLL improves overall here; cannot generalize a Chinese-only cross-domain degradation story."]}
    out=BASE/"train_language_geometry";out.mkdir(exist_ok=True)
    (out/"EVIDENCE.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=["# 平均旧功能改善会否掩盖子组受损：TRAIN-only实测", "",
        "来源为既有确定性19阶段轨迹，4条冻结医学TRAIN探针（英文2/中文2），无新GPU运行、不读TEST、不改变评分。与严格48组、正式V2权重不同。", "",
        "| 阶段/当前域 | 四题全部已学 | 平均实际ΔNLL | 英文实际ΔNLL | 中文实际ΔNLL | 平均改善但有组变差 |", "|---|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| T{r['stage']:02d}/{r['current_domain']} | {r['all_four_probes_acquired_before_transition']} | {r['aggregate_actual_delta_nll']:+.6f} | {r['groups']['en']['actual_delta_nll']:+.6f} | {r['groups']['zh']['actual_delta_nll']:+.6f} | {r['aggregate_improves_but_group_worsens']} |")
    lines += ["", "ΔNLL<0为这些固定TRAIN探针改善，>0为变差，不是Answer F1。原始梯度、实际参数位移Taylor项和有限NLL逐项分列于JSON。", "",
        "## 六栏判断", "",
        "- **探究的问题：** 旧任务平均loss或平均梯度方向是否代表所有旧子功能？",
        "- **做的实验：** 校验原完整证明、19个状态/梯度、4个原TRAIN ID与语言，全部18段都报告。只有四题所属任务都已学习后，才计旧功能受损；不把尚未学过的探针称遗忘。",
        f"- **结果：** 四题均已学的{len(eligible)}段中，平均NLL改善但至少一语言组变差的阶段为{masked}。有英文受损，也有中文受损，不是中文专属现象。",
        f"- **反方向证据：** 整个T11→T19后续跨域段，这2条英文TRAIN NLL总变{cross_totals['en']:+.6f}，2条中文{cross_totals['zh']:+.6f}；由首尾NLL与各段相加精确核验。该轨迹中文TRAIN总体改善，不能据此前K18另一轨迹的中文测试低分宣称所有跨域都伤中文。",
        "- **证明了什么：** 在本固定探针上，平均旧loss改善不保证每个子组保留；这是已知平均化性质的本例验证。语言与任务/答案内容混杂，2题一组不足证明语言是原因，不能解释所有持续学习方法或最终F1。Taylor与有限NLL有反向情形，局部线性符号不能替代真实有限变化。",
        "- **改进的方法/效果：** 原因判定需同时保留逐题/子组与整体有限变化；没有新增保护方法或模型收益。GPU1既有受控推理和严格48实验继续等待，不用这个CPU审计代替因果验证。", "",
        "[完整18段、同权重梯度/参数/原数据SHA](EVIDENCE.json)", ""]
    lines += ["- "+x for x in result["limits"]]
    (out/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"CPU_only":True,"TRAIN_QA":4,"transitions":18,"previously_acquired_transitions":len(eligible),"masked_stages":masked,"cross_segment_changes":cross_totals}))


if __name__=="__main__":main()
