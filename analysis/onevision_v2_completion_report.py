"""Read-only CPU re-score of completed formal V2 baselines, including transfers."""
import hashlib
import json
import math
from pathlib import Path
import time
from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT/"exp/two_backbone_native_cl_20261017_v2"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    manifest_path = ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    gold, sources = {}, {str(manifest_path.relative_to(ROOT)): sha(manifest_path)}
    for t in manifest["tasks"]:
        f = t["files"]["test"]
        p = ROOT/f["path"]
        assert sha(p) == f["sha256"]
        gold[t["task"]] = [json.loads(s) for s in p.read_text().splitlines()]
        assert len(gold[t["task"]]) == f["count"]
    results = []
    for method in ["SeqFT_LoRA", "ER_LoRA"]:
        base = FRAME/"runs/onevision/pct01"/method
        matches = [p.parent for p in base.glob("training_*/run_status.json") if json.loads(p.read_text()).get("status") == "completed"]
        assert len(matches) == 1
        run = matches[0]
        status = json.loads((run/"run_status.json").read_text())
        config = json.loads((run/"config.json").read_text())
        assert status["completed_stage"] == 19 and config["manifest_sha256"] == sha(manifest_path)
        assert config["seed"] == 42 and config["epochs"] == 1 and config["method"] == method
        cells, proofs = {}, {}
        for stage in range(20):
            tids = range(1, 20) if stage in [0, 19] else [stage, stage+1]
            for tid in tids:
                p = run/f"evaluations/stage_{stage:02d}/T{tid:02d}_test_predictions.jsonl"
                rows = [json.loads(s) for s in p.read_text().splitlines()]
                assert len(rows) == len(gold[tid])
                scores = []
                for row, original in zip(rows, gold[tid]):
                    assert row["id"] == original["id"] and row["references"] == original["answers"]
                    score = answer_f1(row["prediction"], row["references"])
                    assert row["scores"]["token_f1"] == score and math.isfinite(score)
                    scores.append(score)
                value = sum(scores)/len(scores)
                summary = json.loads(p.with_suffix(".scores.json").read_text())
                assert summary["count"] == len(rows) and abs(summary["token_f1"]-value) < 1e-12
                cells[stage, tid] = value
                proofs[str(p.relative_to(run))] = sha(p)
        assert len(cells) == 74
        final = [cells[19, i] for i in range(1, 20)]
        aggregate = {"AP": sum(final)/19,
            "FWT": sum(cells[j-1, j]-cells[0, j] for j in range(2, 20))/18,
            "BWT": sum(cells[19, j]-cells[j, j] for j in range(1, 19))/18}
        original_metrics = json.loads((run/"metrics_by_score.json").read_text())["token_f1"]
        assert all(abs(value-original_metrics[key]) < 1e-12 for key, value in aggregate.items())
        domains = {}
        original_domains = json.loads((run/"domain_results.json").read_text())["token_f1"]
        for d in dict.fromkeys(t["domain"] for t in manifest["tasks"]):
            tasks = [t for t in manifest["tasks"] if t["domain"] == d]
            count = sum(len(gold[t["task"]]) for t in tasks)
            value = sum(cells[19, t["task"]]*len(gold[t["task"]]) for t in tasks)/count
            assert abs(original_domains[d]["final"]-value) < 1e-12
            domains[d] = {"F1": value, "QA": count}
        checkpoint = run/status["final_checkpoint"]
        assert checkpoint.stat().st_size == status["final_parameter_file_bytes"]
        results.append({"method": method, "backbone": "LLaVA-OneVision-Qwen2-0.5B-OV",
            "formal_common_image_protocol": True, "run": str(run.relative_to(ROOT)), "status": status,
            "aggregate": aggregate, "domains": domains, "final_native19": final, "full_QA": sum(d["QA"] for d in domains.values()),
            "rescored_cells": len(cells), "prediction_sha256": proofs,
            "checkpoint_sha256": sha(checkpoint), "config_sha256": sha(run/"config.json"),
            "metrics_sha256": sha(run/"metrics_by_score.json")})
    report = ROOT/"summary/two_backbone_pct01_20261017/COMPLETED_20261018.json"
    report.write_text(json.dumps({"UTC": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "CPU_only": True, "raw_Answer_F1": True, "TEST_used_to_select_method": False,
        "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)), "analysis/answer_f1.py": sha(ROOT/"analysis/answer_f1.py"), **sources},
        "results": results}, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({"completed": len(results), "cells_each": 74, "full_QA_each": 21454,
                      "metrics": [x["aggregate"] for x in results]}))


if __name__ == "__main__":
    main()
