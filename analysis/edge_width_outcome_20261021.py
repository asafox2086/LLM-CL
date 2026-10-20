"""Summarize sealed CPU width controls; no models, new inference, or selection."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "summary/edge_deployment_20261019/structured_width_20261020/v1"
    verified = json.loads((folder / "VERIFIED_RESULTS.json").read_text())
    all_complete = json.loads((folder / "ALL_COMPLETE.json").read_text())
    assert all_complete["cases"] == ["dense", "weight", "fluctuation"]
    assert verified["goal_achieved"] is False
    assert verified["ready_sha256"] == sha(folder / "READY.json")
    records = {}
    matched_inputs = None
    for case in all_complete["cases"]:
        path = folder / (case + "_COMPLETE.json")
        data = json.loads(path.read_text())
        v = verified["all_cases"][case]
        assert data["complete"] and data["fresh_parameter_digest_exact"]
        assert sha(path) == v["source_complete_sha256"]
        assert data["ready_sha256"] == verified["ready_sha256"]
        assert data["new_optimizer_updates"] == data["official_TEST_QA"] == 0
        requests = data["requests"]
        speed = [r for r in requests if r["speed_only"]]
        quality = [r for r in requests if not r["speed_only"]]
        assert len(speed) == len(quality) == 5
        assert all(r["output_token_count"] == 8 for r in speed)
        inputs = [(r["id"], r["domain"], r["speed_only"], r["input"]) for r in requests]
        if matched_inputs is None:
            matched_inputs = inputs
        else:
            assert inputs == matched_inputs
        assert all(r["input"]["visual_cache_hit"] is False for r in requests)
        mean_f1 = sum(r["raw_AnswerF1"] for r in quality) / 5
        assert mean_f1 == v["mean_F1_five_seen_TRAIN_probes"]
        speed_peak = max(r["generation_peak_RSS_bytes"] for r in speed)
        assert speed_peak == v["maximum_observed_generation_RSS"]
        records[case] = {
            "complete_sha256": sha(path),
            "package_bytes": data["package"]["actual_package_bytes"],
            "five_seen_TRAIN_mean_F1": mean_f1,
            "loaded_RSS_bytes": data["memory_after_load"]["rss"],
            "loaded_PSS_bytes": data["memory_after_load"]["pss"],
            "loaded_USS_bytes": data["memory_after_load"]["uss"],
            "forced8_generation_peak_RSS_bytes": speed_peak,
            "all10_generation_peak_RSS_bytes": max(r["generation_peak_RSS_bytes"] for r in requests),
            "forced8_TTFT_sum_seconds": sum(r["TTFT_seconds"] for r in speed),
            "forced8_decode_tps_by_probe": [r["decode_tokens_per_second"] for r in speed],
            "free64_output_tokens_by_probe": [r["output_token_count"] for r in quality],
            "case_seconds": data["seconds"],
            "warmup_calibration_NLL": data["warmup_calibration_NLL"],
        }
    dense = records["dense"]
    for case in ["weight", "fluctuation"]:
        item = records[case]
        item["differences_from_dense"] = {
            "five_seen_TRAIN_F1_pp": item["five_seen_TRAIN_mean_F1"] - dense["five_seen_TRAIN_mean_F1"],
            **{name + "_relative_reduction": 1 - item[name] / dense[name] for name in [
                "package_bytes", "loaded_RSS_bytes", "loaded_PSS_bytes", "loaded_USS_bytes",
                "forced8_generation_peak_RSS_bytes", "all10_generation_peak_RSS_bytes",
                "forced8_TTFT_sum_seconds",
            ]},
        }
    prepare = json.loads((folder / "prepare_COMPLETE.json").read_text())
    output = {
        "source_sha256": sha(Path(__file__)),
        "verified_results_sha256": sha(folder / "VERIFIED_RESULTS.json"),
        "all_complete_sha256": sha(folder / "ALL_COMPLETE.json"),
        "same_original_inputs_across_cases": True,
        "records": records,
        "new_model_forwards": 0,
        "new_optimizer_updates": 0,
        "generation_requests_in_three_original_cases": 30,
        "warmup_requests_in_three_original_cases": 3,
        "prepare_seconds": prepare["seconds"],
        "total_prepare_and_case_seconds": prepare["seconds"] + sum(r["case_seconds"] for r in records.values()),
        "decision": "STOP fixed simultaneous vision/language half-width without recovery; no full TEST",
        "inference": "Input-aware ordering retains more function than weight-only on these probes, but neither preserves reference quality.",
        "limitations": [
            "Five teacher-seen TRAIN probes, one per domain; not domain F1, DEV or formal AP.",
            "Joint vision/language removal does not identify which component caused quality loss.",
            "No recovery training; this does not reject structured pruning with recovery or all compact alternatives.",
            "Single serial CPU pass, shared server load, no uncertainty or phone/GPU evidence.",
            "Original reporter peak covers forced8 only; all10 peak added here excludes warmup/load/prepare and is not full-run peak.",
            "Teacher saw original DEV; it cannot be used to select compression by that DEV score.",
        ],
        "goal_achieved": False,
    }
    (folder / "OUTCOME.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
