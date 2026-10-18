"""Compile all admitted diagnostic methods on the exact same 155 answers.

Separate from the formal common-image full21454 main table. Never uses TEST
scores to select a method or alter the two predeclared full confirmations.
"""
import hashlib
import json
from pathlib import Path

from answer_f1 import answer_f1
import onevision_functional_router_queue as queue

ROOT, BASE = queue.ROOT, queue.BASE
DOMAINS = ["natural", "medical", "geography", "finance", "traffic"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = queue.functional_plan()
    assert queue.complete("pilot", None, spec)
    report = json.loads((queue.STUDY / "VERIFIED_EVIDENCE.json").read_text())
    proof_path = queue.STUDY / "pilot_complete.json"
    assert report["pilot_completed"] and report["completion_sha256"][proof_path.name] == sha(proof_path)
    pilot_proof = json.loads(proof_path.read_text())
    reference_path = BASE / "learned_reuse/MATCHED_REFERENCE_20261018.json"
    reference = json.loads(reference_path.read_text())
    expected = {r["variant"]: r for r in reference["results"]}
    candidates = [("SeqFT-LoRA（原生研究参考）", "sequential_LoRA_cap1", BASE / "capacity/cap1_complete.json", "final_input_router"),
                  ("固定相似度K3", "fixed_cosine_K3", BASE / "learned_reuse/fixed_cosine_K3_train_complete.json", "final_diagnostic"),
                  ("可学习相似度K3", "learned_cosine_K3", BASE / "learned_reuse/learned_cosine_K3_train_complete.json", "final_diagnostic")]
    for view, name in [("quality_hard", "TRAIN质量校准＋硬调用"), ("geometry_soft", "几何软残差"),
                       ("quality_soft", "TRAIN质量校准＋软残差")]:
        candidates.append((name, view, proof_path, view))
    manifest_path = ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    assert sha(manifest_path) == spec["parent_spec"]["manifest_sha256"]
    gold = {}
    for task in json.loads(manifest_path.read_text())["tasks"]:
        info = task["files"]["test"]
        p = ROOT / info["path"]
        assert sha(p) == info["sha256"]
        for line in p.read_text().splitlines():
            row = json.loads(line)
            gold[row["id"]] = (row["answers"], task["task"])
    results, hashes, common = [], {}, None
    for name, variant, proof_file, label in candidates:
        proof = json.loads(proof_file.read_text())
        assert proof["complete"]
        hashes[str(proof_file.relative_to(ROOT))] = sha(proof_file)
        if variant in expected:
            assert sha(proof_file) == expected[variant]["proof_sha256"]
        run = ROOT / proof["run"]
        ids_gold, scores, domain_scores = [], {i: [] for i in range(1, 20)}, {}
        for d in DOMAINS:
            relative = f"evaluations/{label}/{d}_image.jsonl"
            p = run / relative
            assert sha(p) == proof["artifact_sha256"][relative]
            if variant in expected:
                assert sha(p) == expected[variant]["diagnostic_files_sha256"][relative]
            hashes[str(p.relative_to(ROOT))] = sha(p)
            rows = [json.loads(s) for s in p.read_text().splitlines()]
            domain_scores[d] = []
            for row in rows:
                assert row["references"] == gold[row["id"]][0]
                if "scores" not in row:
                    continue
                score = answer_f1(row["prediction"], row["references"])
                assert score == row["scores"]["token_f1"]
                ids_gold.append((row["id"], row["references"]))
                scores[gold[row["id"]][1]].append(score)
                domain_scores[d].append(score)
        assert len(ids_gold) == len({r[0] for r in ids_gold}) == 155 and all(scores.values())
        if common is None:
            common = ids_gold
        assert ids_gold == common
        ap = sum(sum(v) / len(v) for v in scores.values()) / 19
        if variant in expected:
            assert ap == expected[variant]["native19_AP"]
        else:
            assert ap == next(x["native19_AP"] for x in report["aggregate"] if x["scope"] == "pilot" and x["view"] == variant)
        results.append({"method": name, "variant": variant, "domains": {d: sum(v)/len(v) for d, v in domain_scores.items()},
                        "generated_per_domain": {d: len(v) for d, v in domain_scores.items()}, "AP": ap,
                        "FWT": None, "BWT": None})
    output = {"CPU_only": True, "same_155_original_ids_gold": True, "raw_Answer_F1": True,
              "formal_V2_main_table": False, "TEST_selection": False,
              "results": results, "input_sha256": hashes,
              "reference_report_sha256": sha(reference_path),
              "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
              "limits": ["155 generated diagnostic QA, NOT complete21454; nativeAnyRes, NOT formalV2.",
                         "Quality/aggregation variants adapt D-MoLE/MVP principles, not original paper reproductions.",
                         "All methods and both predeclared full candidates retained, regardless scores.",
                         "No full transfer metrics inferred from final diagnostics."]}
    target = queue.STUDY
    (target / "MAIN_TABLE_DIAGNOSTIC.json").write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 原生研究主表格式：同155题诊断，不能当完整主实验", "",
             "基座LLaVA-OneVision-Qwen2-0.5B-OV；相同1313TRAIN/19阶段、相同155生成题/原金标/Answer F1。自然10、医学73、遥感32、金融10、交通30题。AP为原19任务等权，五域列为各域诊断生成题均值。", "",
             "| 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP ↑ | FWT | BWT |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append("| " + row["method"] + " | " + " | ".join(f"{row['domains'][d]:.2f}" for d in DOMAINS)
                     + f" | {row['AP']:.2f} | — | — |")
    lines += ["", "质量硬调用AP相对原可学习相似度+5.679pp，软组合+5.362pp；两者仍低于同题顺序LoRA。固定知识/LoRA池下的调用与残差确实影响此诊断整体性能，不证明已超过简单基线、知识充分或全量收益。", "",
              "这是借鉴D-MoLE/MVP的改造，不能把该表当原论文方法复现成绩。两个预定完整21454候选均继续，原方法及归因全部保留。正式V2完整主表独立见[双骨干主表](../../two_backbone_pct01_20261017/MAIN_TABLE.md)。", "",
              "[相同题目/证明/逐域复算](MAIN_TABLE_DIAGNOSTIC.json) · [完整四视图证据](VERIFIED_EVIDENCE.json)。"]
    (target / "MAIN_TABLE_DIAGNOSTIC.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(results, ensure_ascii=True))


if __name__ == "__main__":
    main()
