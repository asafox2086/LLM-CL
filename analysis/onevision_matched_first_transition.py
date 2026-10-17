"""TRAIN-only first-transition geometry at the exact matched canonical state.

The shared old gradient is independently equal in two completed noop controls.
It is not a gradient measured at later EWC states or a finite loss change.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"summary/onevision_diagnostics_20261016/matched_task_domain"


def main():
    import hashlib
    torch.set_num_threads(2)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    sources={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))}
    def checked(p,h=None):
        actual=sha(p);assert h is None or h==actual,str(p)
        sources[str(p.relative_to(ROOT))]=actual;return p
    gate=json.loads(checked(BASE/"CPU_PLAN_CHECK.json").read_text());assert gate["passed"]
    spec=gate["spec"]
    for p,h in spec["source_sha256"].items():checked(ROOT/p,h)
    def case(sequence,method):
        name=f"T04_{sequence}__{method}";p=BASE/(name+"_complete.json")
        proof=json.loads(checked(p).read_text());assert proof["complete"] and proof["stage1_exact"] and proof["spec"]==spec
        run=(ROOT/proof["run"]).resolve();assert run.is_relative_to(ROOT/"exp/onevision_matched_task_domain_20261017")
        def load(relative):return torch.load(checked(run/relative,proof["artifact_sha256"][relative]),map_location="cpu",weights_only=True)
        return name,load
    canonical=BASE/"canonical_anchor4.pt";checked(canonical)
    state=torch.load(canonical,map_location="cpu",weights_only=True)["adapter"]
    references=[]
    for sequence in ["repeat_anchor","within_medical"]:
        name,load=case(sequence,"anchor_gradient_noop")
        a=load("adapters/stage_01.pt")["adapter"]
        assert a.keys()==state.keys() and all(torch.equal(a[n][f],state[n][f]) for n in a for f in ["A","B"])
        raw=load(f"raw_step_gradients/{name}/stage_02.pt")
        assert raw["reference_train_ids"]==spec["TRAIN_ids"]["4"]
        references.append(raw)
    assert references[0]["parameter_layout"]==references[1]["parameter_layout"]
    assert torch.equal(references[0]["reference_gradient"],references[1]["reference_gradient"])
    old=references[0]["reference_gradient"].double();layout=references[0]["parameter_layout"]
    assert old.numel()==540672 and torch.isfinite(old).all() and old.norm()>0
    rows=[]
    for sequence in spec["sequences"]["4"]:
        if not (BASE/f"T04_{sequence}__SeqFT_LoRA_complete.json").exists():continue
        name,load=case(sequence,"SeqFT_LoRA")
        a=load("adapters/stage_01.pt")["adapter"];b=load("adapters/stage_02.pt")["adapter"]
        assert a.keys()==state.keys() and all(torch.equal(a[n][f],state[n][f]) for n in a for f in ["A","B"])
        raw=load(f"raw_step_gradients/{name}/stage_02.pt");assert raw["parameter_layout"]==layout
        task=spec["sequences"]["4"][sequence][1]
        assert raw["train_ids"]==spec["TRAIN_ids"][str(task)][:2]
        g=raw["pre_intervention_gradient"].double();assert torch.isfinite(g).all()
        delta=[]
        for name,shape in layout:
            module,factor,_=name.rsplit(".",2)
            assert list(a[module][factor].shape)==shape
            delta.append((b[module][factor].double()-a[module][factor].double()).flatten())
        displacement=torch.cat(delta)
        # Ideal real-arithmetic fresh AdamW, not an actual saved first GPU step.
        clipped=g*min(1.,1./(float(g.norm())+1e-6))
        ideal=-1e-4*clipped/(clipped.abs()+1e-8)
        rows.append({"sequence":sequence,"stage2_task":task,
                     "shared_stage1_A_B_exact":True,"new_minibatch_raw_gradient_L2":float(g.norm()),
                     "current_TRAIN_vs_old8TRAIN_eval_gradient_cosine":float(torch.dot(g,old)/g.norm()/old.norm()),
                     "old_gradient_dot_ideal_fresh_Adam_first_step":float(torch.dot(old,ideal)),
                     "old_gradient_dot_actual4_update_displacement":float(torch.dot(old,displacement)),
                     "actual4_update_displacement_L2":float(displacement.norm())})
    for p,h in sources.items():assert sha(ROOT/p)==h
    assert not torch.cuda.is_initialized()
    result={"CPU_only":True,"no_TEST_predictions_or_scores_read":True,"old8TRAIN_gradient_L2":float(old.norm()),
            "two_independent_same_state_old_references_exact":True,"cases":rows,"source_sha256":sources,
            "limits":["Old reference eval-mode8QA versus new train-mode/dropout2QA; defined probes, not all-task gradients.",
                      "Shared gradient used only at proven identical canonical weights, never later EWC/other states.",
                      "Positive cosine only rejects a negative dot for this first pair; later conflict remains possible.",
                      "Ideal Adam is known algebra; actual only four-update endpoint displacement saved here.",
                      "Taylor terms are local linear predictions, not measured finite TRAIN/test NLL or F1 effects.",
                      "Single Organ anchor/seed and frozen examples; not a domain-count law or novelty claim."]}
    out=BASE/"first_transition_geometry";out.mkdir(exist_ok=True)
    (out/"EVIDENCE.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    lines=["# 严格共同起点：首次跨域梯度是否已经冲突", "",
           "旧8TRAIN eval梯度在两个独立noop轨迹中逐元素相同，三个SeqFT起点A/B也精确相同；没有移植不同权重的梯度。", "",
           "| 序列 | 新/旧梯度余弦 | 理想首步Adam旧loss Taylor项 | 真实四更新位移旧loss Taylor项 |", "|---|---:|---:|---:|"]
    for x in rows:lines.append(f"| {x['sequence']} | {x['current_TRAIN_vs_old8TRAIN_eval_gradient_cosine']:.6f} | {x['old_gradient_dot_ideal_fresh_Adam_first_step']:+.6f} | {x['old_gradient_dot_actual4_update_displacement']:+.6f} |")
    positive=sum(x["current_TRAIN_vs_old8TRAIN_eval_gradient_cosine"]>0 for x in rows)
    lines += ["", f"已完成{len(rows)}条，其中{positive}条首次点积为正。正点积的配对不能用这次首批负梯度冲突解释；较弱正相关也不证明最终损害由梯度夹角造成，需后续更新与真实有限性能控制。", "",
              "负Taylor项只说明该局部模型预计旧TRAIN loss下降；没有测量这个有限位移下的旧TRAIN loss，更不是F1或语义知识保留证明。", "",
              "[原始源SHA/数值](EVIDENCE.json)", ""]
    lines += ["- "+x for x in result["limits"]]
    (out/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"CPU_only":True,"completed_cases":len(rows),"all_current_first_cosines_positive":all(x['current_TRAIN_vs_old8TRAIN_eval_gradient_cosine']>0 for x in rows)}))


if __name__=="__main__":main()
