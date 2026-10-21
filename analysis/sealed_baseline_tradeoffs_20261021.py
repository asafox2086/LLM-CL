"""Read sealed predictions and training records; no model or optimizer imports."""
import argparse
import base64
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def signature(value):
    return sha_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def compare(left, right):
    assert len(left) == len(right)
    differences = [a - b for a, b in zip(left, right)]
    return {
        "QA": len(differences),
        "mean_difference_pp": sum(differences) / len(differences),
        "higher": sum(v > 0 for v in differences),
        "lower": sum(v < 0 for v in differences),
        "equal": sum(v == 0 for v in differences),
        "perfect_both": sum(a == b == 100 for a, b in zip(left, right)),
        "perfect_left_only": sum(a == 100 and b < 100 for a, b in zip(left, right)),
        "perfect_right_only": sum(b == 100 and a < 100 for a, b in zip(left, right)),
        "not_a_routing_rule_or_oracle": True,
    }


def collect(root, inputs):
    started = time.monotonic()
    lock = json.loads(inputs.read_text())
    assert sha_bytes((root / "analysis/answer_f1.py").read_bytes()) == lock["score_sha256"]
    sys.path.insert(0, str(root / "analysis"))
    from answer_f1 import answer_f1
    manifest_path = root / lock["manifest_path"]
    assert sha_bytes(manifest_path.read_bytes()) == lock["manifest_sha256"]
    tasks = json.loads(manifest_path.read_text())["tasks"]
    assert [t["task"] for t in tasks] == list(range(1, 20))
    assert sum(t["files"]["train"]["count"] for t in tasks) == 1313
    assert sum(t["files"]["test"]["count"] for t in tasks) == 21454
    gold_signatures = {}
    all_scores, final_text, methods = {}, {}, {}
    for method, item in lock["methods"].items():
        proof_bytes = base64.b64decode(item["proof_b64"])
        assert sha_bytes(proof_bytes) == item["proof_sha256"]
        proof = json.loads(proof_bytes)
        assert proof["candidate"]["method"] == method
        assert proof["candidate"]["backbone"] == "onevision"
        assert proof["positive_finite_training_stages"] == 19
        assert proof["rescored_cells"] == 74 and proof["final_QA"] == 21454
        run = root / proof["candidate"]["run"]
        config_bytes = (run / "config.json").read_bytes()
        assert sha_bytes(config_bytes) == proof["config_sha256"]
        config = json.loads(config_bytes)
        assert config["manifest_sha256"] == lock["manifest_sha256"]
        cells, state_scores, texts = {}, {}, {}
        for task in tasks:
            t = task["task"]
            for stage in sorted({0, t, 19}):
                rel = f"evaluations/stage_{stage:02d}/T{t:02d}_test_predictions.jsonl"
                data = (run / rel).read_bytes()
                digest = sha_bytes(data)
                assert digest == proof["prediction_sha256"][rel], (method, rel)
                rows = [json.loads(line) for line in data.splitlines()]
                ids = [row["id"] for row in rows]
                assert len(ids) == len(set(ids)) == task["files"]["test"]["count"]
                gold = signature([(row["id"], row["dataset"], row["references"]) for row in rows])
                if t not in gold_signatures:
                    gold_signatures[t] = gold
                assert gold == gold_signatures[t], (method, stage, t)
                values = [answer_f1(row["prediction"], row["references"]) for row in rows]
                assert all(v == row["scores"]["token_f1"] for v, row in zip(values, rows))
                assert all(math.isfinite(v) and 0 <= v <= 100 for v in values)
                state_scores[stage, t] = values
                cells[rel] = {"sha256": digest, "QA": len(rows), "F1": sum(values) / len(values),
                              "gold_identity_sha256": gold}
                if stage == 19:
                    texts[t] = [sha_bytes(row["prediction"].encode()) for row in rows]
        acquired = [cells[f"evaluations/stage_{t:02d}/T{t:02d}_test_predictions.jsonl"]["F1"] for t in range(1, 20)]
        final = [cells[f"evaluations/stage_19/T{t:02d}_test_predictions.jsonl"]["F1"] for t in range(1, 20)]
        ap = sum(final) / 19
        bwt = sum(f - a for f, a in zip(final[:-1], acquired[:-1])) / 18
        assert math.isclose(ap, proof["aggregate"]["AP"], abs_tol=1e-10)
        assert math.isclose(bwt, proof["aggregate"]["BWT"], abs_tol=1e-10)
        assert math.isclose(ap, sum(acquired) / 19 + 18 / 19 * bwt, abs_tol=1e-10)
        domain_results = {}
        task_results = []
        for task in tasks:
            t = task["task"]
            initial, learned, last = [state_scores[s, t] for s in [0, t, 19]]
            task_results.append({
                "task": t, "domain": task["domain"], "native_label": task["native_label"],
                "train_QA": task["files"]["train"]["count"], "test_QA": len(initial),
                "initial_F1": sum(initial) / len(initial), "acquired_F1": acquired[t - 1],
                "final_F1": final[t - 1],
                "later_change_pp": final[t - 1] - acquired[t - 1],
                "initial_perfect_lost_at_final": sum(a == 100 and c < 100 for a, c in zip(initial, last)),
                "new_perfect_at_final": sum(a < 100 and c == 100 for a, c in zip(initial, last)),
                "acquired_improvement_then_regressed": sum(b > a and c < b for a, b, c in zip(initial, learned, last)),
            })
        for domain in proof["domains"]:
            values = [v for task in tasks if task["domain"] == domain for v in state_scores[19, task["task"]]]
            mean = sum(values) / len(values)
            assert math.isclose(mean, proof["domains"][domain]["F1"], abs_tol=1e-10)
            domain_results[domain] = {"F1": mean, "QA": len(values)}
        events_path = run / "logs/events.jsonl"
        events_bytes = events_path.read_bytes()
        assert sha_bytes(events_bytes) == proof["source_sha256"][str(events_path.relative_to(root))]
        events = [json.loads(line) for line in events_bytes.splitlines()]
        assert len(events) == 93 and all(e["event"] == "train_update" for e in events)
        assert all(math.isfinite(e["loss"]) and math.isfinite(e["gradient_l2_preclip"])
                   and e["gradient_l2_preclip"] > 0 and all(v > 0 for v in e["learning_rate_used"]) for e in events)
        resource_path = run / "resources.jsonl"
        counts, model_bytes, invalid, first, last_resource = set(), set(), [], None, None
        with resource_path.open() as handle:
            for number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    invalid.append(number)
                    continue
                if first is None:
                    first = row
                last_resource = row
                if "trainable_parameters" in row:
                    counts.add(row["trainable_parameters"])
                if "model_parameter_bytes" in row:
                    model_bytes.add(row["model_parameter_bytes"])
        methods[method] = {
            "proof_sha256": item["proof_sha256"], "run": proof["candidate"]["run"],
            "config_sha256": proof["config_sha256"], "checkpoint_sha256_from_full_proof": proof["checkpoint_sha256"],
            "cells": cells, "task_results": task_results, "domains": domain_results,
            "AP19": ap, "BWT": bwt, "FWT_from_full_proof": proof["aggregate"]["FWT"],
            "acquired19_mean": sum(acquired) / 19, "later_change_accounting_term": 18 / 19 * bwt,
            "domain_macro_F1": sum(d["F1"] for d in domain_results.values()) / 5,
            "positive_finite_main_updates": len(events), "first_actual_LR": events[0]["learning_rate_used"],
            "observed_trainable_parameter_counts": sorted(counts),
            "observed_model_parameter_bytes": sorted(model_bytes),
            "resource_invalid_line_numbers": invalid,
            "resources_sha256": sha_bytes(resource_path.read_bytes()),
            "resource_endpoints_not_deployment_or_isolated_peak": True,
            "config_numeric": {k: config.get(k) for k in ["model_dtype", "autocast_dtype", "image_area_cap", "batch_size", "micro_batch_size", "seed"]},
        }
        all_scores[method] = state_scores
        final_text[method] = texts
        print(json.dumps({"method_complete": method, "AP19": ap, "acquired19_mean": sum(acquired) / 19,
                          "trainable_parameters": sorted(counts)}), flush=True)
    pairs = {}
    for left, right in itertools.combinations(methods, 2):
        result = {"tasks": {}, "domains": {}}
        for task in tasks:
            t = task["task"]
            item = compare(all_scores[left][19, t], all_scores[right][19, t])
            item["raw_text_identical"] = sum(a == b for a, b in zip(final_text[left][t], final_text[right][t]))
            result["tasks"][str(t)] = item
        for domain in methods[left]["domains"]:
            def pooled(method):
                return [v for task in tasks if task["domain"] == domain for v in all_scores[method][19, task["task"]]]
            result["domains"][domain] = compare(pooled(left), pooled(right))
        l, r = methods[left], methods[right]
        result["AP_difference_pp"] = l["AP19"] - r["AP19"]
        result["acquisition_difference_pp"] = l["acquired19_mean"] - r["acquired19_mean"]
        result["later_change_accounting_difference_pp"] = l["later_change_accounting_term"] - r["later_change_accounting_term"]
        assert math.isclose(result["AP_difference_pp"], result["acquisition_difference_pp"] + result["later_change_accounting_difference_pp"], abs_tol=1e-10)
        result["not_causal_decomposition"] = True
        pairs[left + "_minus_" + right] = result
    return {
        "source_sha256": sha_bytes(Path(__file__).read_bytes()), "inputs_sha256": sha_bytes(inputs.read_bytes()),
        "manifest_sha256": lock["manifest_sha256"], "score_sha256": lock["score_sha256"],
        "methods": methods, "pairs": pairs, "CPU_only": True,
        "new_model_forwards": 0, "new_optimizer_updates": 0,
        "same_ordered_ids_datasets_all_references_across_methods_and_states": True,
        "TEST_used_for_routing_or_hyperparameters": False,
        "scope": "posthoc description of four sealed formal runs; not intervention, independent DEV, equivalence or deployment proof",
        "seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Do not overwrite an analysis snapshot"
    result = collect(args.root, args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"complete": True, "output": str(args.output), "sha256": sha_bytes(args.output.read_bytes())}), flush=True)
