"""Bind the same diagnostic cohort to the original sequential LoRA reference."""
import hashlib
import json
from pathlib import Path
from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/"summary/onevision_diagnostics_20261016"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    variants = {v: REPORT/f"learned_reuse/{v}_train_complete.json" for v in ["fixed_cosine_K3", "learned_cosine_K3"]}
    variants["sequential_LoRA_cap1"] = REPORT/"capacity/cap1_complete.json"
    manifest = json.loads((ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json").read_text())
    gold, tasks = {}, {}
    for t in manifest["tasks"]:
        f = t["files"]["test"]
        assert sha(ROOT/f["path"]) == f["sha256"]
        for s in (ROOT/f["path"]).read_text().splitlines():
            row = json.loads(s)
            gold[row["id"]], tasks[row["id"]] = row, t["task"]
    reference = None
    output = []
    for name, path in variants.items():
        proof = json.loads(path.read_text())
        assert proof["complete"]
        run = ROOT/proof["run"]
        label = "final_input_router" if name == "sequential_LoRA_cap1" else "final_diagnostic"
        rows, hashes = [], {}
        for domain in ["natural", "medical", "geography", "finance", "traffic"]:
            p = run/f"evaluations/{label}/{domain}_image.jsonl"
            relative = str(p.relative_to(run))
            hashes[relative] = sha(p)
            assert hashes[relative] == proof["artifact_sha256"][relative]
            rows.extend(r for r in map(json.loads, p.read_text().splitlines()) if "scores" in r)
        assert len(rows) == 155 and len({r["id"] for r in rows}) == 155
        ids_gold = [(r["id"], r["references"]) for r in rows]
        if reference is None:
            reference = ids_gold
        assert ids_gold == reference
        scores = {i: [] for i in range(1, 20)}
        for row in rows:
            assert row["references"] == gold[row["id"]]["answers"]
            score = answer_f1(row["prediction"], row["references"])
            assert score == row["scores"]["token_f1"]
            scores[tasks[row["id"]]].append(score)
        assert all(scores.values())
        output.append({"variant": name, "generated_QA": 155,
            "native19_AP": sum(sum(v)/len(v) for v in scores.values())/19,
            "proof_sha256": sha(path), "diagnostic_files_sha256": hashes})
    report = {"CPU_only": True, "same_155_original_ids_and_gold": True,
        "sole_raw_Answer_F1": True, "no_TEST_selection": True, "results": output,
        "limitation": "Diagnostic155 only, not formalV2 or complete21454; full confirmations retained regardless scores.",
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                          "analysis/answer_f1.py": sha(ROOT/"analysis/answer_f1.py")}}
    (REPORT/"learned_reuse/MATCHED_REFERENCE_20261018.json").write_text(json.dumps(report, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({"same_ids_gold": True, "results": [{"variant": v["variant"], "AP": v["native19_AP"]} for v in output]}))


if __name__ == "__main__":
    main()
