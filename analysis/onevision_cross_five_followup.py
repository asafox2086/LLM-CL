"""Read-only completed four/five-domain follow-up; no GPU or choice changes.

Report all author-language strata, the exact common prefix, and both generated
quality and auxiliary NLL. A fifth domain also changes content/order here.
"""
import collections
import hashlib
import json
import math
from pathlib import Path
import random
from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"summary/onevision_diagnostics_20261016"
OUT = BASE/"matched_task_domain/common_prefix"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def paired(values, images):
    assert values and len(values) == len(images) and all(math.isfinite(x) for x in values)
    groups = collections.defaultdict(list)
    for value, image in zip(values, images):
        groups[tuple(image)].append(value)
    clusters = list(groups.values())
    rng, boot = random.Random(42), []
    for _ in range(2000):
        chosen = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        boot.append(sum(map(sum, chosen))/sum(map(len, chosen)))
    boot.sort()
    return {"mean": sum(values)/len(values), "image_cluster_95CI": [boot[49], boot[1949]],
            "QA": len(values), "image_clusters": len(clusters)}


def main():
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
               "analysis/answer_f1.py": sha(ROOT/"analysis/answer_f1.py")}

    def checked(p, expected=None):
        h = sha(p)
        assert expected is None or h == expected
        sources[str(p.relative_to(ROOT))] = h
        return p

    plan = json.loads(checked(BASE/"matched_task_domain/CPU_PLAN_CHECK.json").read_text())["spec"]
    for p, h in plan["source_sha256"].items():
        checked(ROOT/p, h)
    manifest = json.loads(checked(ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json", plan["manifest_sha256"]).read_text())
    task = manifest["tasks"][3]
    assert task["task"] == 4 and task["native_label"] == "Organ"
    info = task["files"]["test"]
    gold = [json.loads(s) for s in checked(ROOT/info["path"], info["sha256"]).read_text().splitlines()]
    assert len(gold) == 507
    prefix = json.loads(checked(OUT/"EVIDENCE.json").read_text())
    gates = [p for p in prefix["pairs"] if p["anchor"] == 4 and p["method"] == "SeqFT_LoRA"]
    assert len(gates) == 1 and gates[0]["all3_states_and_first_gradients_exact"]
    for p, h in prefix["source_sha256"].items():
        checked(ROOT/p, h)
    endpoints = {}
    for sequence in ["repeat_anchor", "within_medical", "cross_short", "cross_five_domains"]:
        proof_path = BASE/f"matched_task_domain/T04_{sequence}__SeqFT_LoRA_complete.json"
        proof = json.loads(checked(proof_path).read_text())
        assert proof["complete"] and proof["stage1_exact"] and proof["spec"] == plan
        run = ROOT/proof["run"]
        relative = "evaluations/final_native_anchor/anchor_image.jsonl"
        rows = [json.loads(s) for s in checked(run/relative, proof["artifact_sha256"][relative]).read_text().splitlines()]
        assert len(rows) == 507
        for row, original in zip(rows, gold):
            assert row["id"] == original["id"] and row["references"] == original["answers"]
            assert row["scores"]["token_f1"] == answer_f1(row["prediction"], row["references"])
            assert math.isfinite(row["nll"])
        endpoints[sequence] = rows
    groups = {"all": list(range(507))}
    for lang in ["en", "zh"]:
        groups[lang] = [i for i, row in enumerate(gold) if row["metadata"]["q_lang"] == lang]
    assert sum(len(groups[g]) for g in ["en", "zh"]) == 507
    levels = {}
    for name, rows in endpoints.items():
        levels[name] = {}
        for group, indices in groups.items():
            selected = [rows[i] for i in indices]
            levels[name][group] = {"QA": len(selected),
                "Answer_F1": sum(row["scores"]["token_f1"] for row in selected)/len(selected),
                "NLL": sum(row["nll"] for row in selected)/len(selected),
                "zero_F1_QA": sum(row["scores"]["token_f1"] == 0 for row in selected),
                "predictions_containing_Han": sum(any("\u3400" <= c <= "\u9fff" for c in row["prediction"]) for row in selected)}
    contrasts = {}
    for before, after in [("cross_short", "cross_five_domains"), ("within_medical", "cross_five_domains")]:
        key = after+"_minus_"+before
        contrasts[key] = {}
        for group, indices in groups.items():
            a, b = endpoints[before], endpoints[after]
            values = [b[i]["scores"]["token_f1"]-a[i]["scores"]["token_f1"] for i in indices]
            images = [gold[i]["images"] for i in indices]
            contrasts[key][group] = {"F1": paired(values, images),
                "NLL": paired([b[i]["nll"]-a[i]["nll"] for i in indices], images),
                "increased": sum(v > 0 for v in values), "decreased": sum(v < 0 for v in values),
                "unchanged": sum(v == 0 for v in values),
                "zero_F1_in_both": sum(a[i]["scores"]["token_f1"] == b[i]["scores"]["token_f1"] == 0 for i in indices)}
    acquired_path = BASE/"shared_anchor_initial/VERIFIED_RESULTS.json"
    acquired = json.loads(checked(acquired_path).read_text())
    acquired_refs = [v for v in acquired["comparisons"] if v["anchor"] == 4]
    assert {v["sequence"] for v in acquired_refs} == set(endpoints)
    result = {"CPU_only": True, "new_GPU_inference_or_training": False, "anchor": "T04 Organ",
        "full_QA": 507, "same_first3_states_and_raw_gradients": True, "levels": levels,
        "contrasts": contrasts, "full_acquired_reference": acquired_refs,
        "source_sha256": sources, "not_TEST_selected": True,
        "limits": ["One anchor/seed/order; fifth-domain sequence changes task content/order too, not pure count causality.",
                   "Intervals conditional on fitted runs, uncorrected image clusters, not seed/order uncertainty.",
                   "CI including zero proves neither equivalence nor absence of a small effect.",
                   "Language cohorts are not verified translations; no language-only causal claim.",
                   "507 native anchor not full21454 or a formal method ranking; NLL not sole main accuracy."]}
    (OUT/"FOLLOWUP_20261018.json").write_text(json.dumps(result, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({"full_QA": 507, "levels": levels,
                      "four_to_five": contrasts["cross_five_domains_minus_cross_short"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
