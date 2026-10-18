"""CPU context for new sealed results: capacity/LR and realized replay memory.

Replay draw identities are reconstructed from frozen source and verified TRAIN
loops, not claimed to be separately logged per-row GPU measurements.
"""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"summary/onevision_diagnostics_20261016/baseline_language_followup_20261018"
FRAME = ROOT/"exp/two_backbone_native_cl_20261017_v2"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    snap = json.loads((OUT/"old2080_snapshot.json").read_text())
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    prefixes, resources, replay = [], [], None

    def checked(p, expected=None):
        value = sha(p)
        assert expected is None or value == expected
        sources[str(p.relative_to(ROOT))] = value
        return p

    def prefix_rows(p):
        content = p.read_bytes()
        end = content.rfind(b"\n")+1
        raw = content[:end]
        prefixes.append({"path": str(p.relative_to(ROOT)), "bytes": end, "sha256": hashlib.sha256(raw).hexdigest()})
        return [json.loads(s) for s in raw.splitlines()]

    manifest_path = ROOT/"data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json"
    manifest = json.loads(checked(manifest_path).read_text())
    train = {}
    for t in manifest["tasks"]:
        f = t["files"]["train"]
        train[t["task"]] = [json.loads(s) for s in checked(ROOT/f["path"], f["sha256"]).read_text().splitlines()]
    assert sum(map(len, train.values())) == 1313
    for row in snap["runs"]:
        run = ROOT/row["run"]
        config = json.loads(checked(run/"config.json").read_text())
        assert config["manifest_sha256"] == sha(manifest_path)
        for name, expected in config["execution_source_sha256"].items():
            checked(FRAME/name, expected)
            checked(run/"runner_source"/name, expected)
        rr = prefix_rows(run/"resources.jsonl")
        selected = [x for x in rr if x.get("stage") == 4 and "trainable_parameters" in x]
        assert selected and len({x["pid"] for x in selected}) == 1
        counts = {x["trainable_parameters"] for x in selected}
        assert len(counts) == 1
        resources.append({"method": row["method"], "same_stage4_PID": selected[0]["pid"],
            "trainable_parameters": counts.pop(), "stage4_exact_resource_windows": selected,
            "actual_learning_rates": [e["learning_rate_used"] for e in row["stage4_training_events"]]})
        if row["method"] != "ER_LoRA":
            continue
        assert json.loads(checked(run/"run_status.json").read_text())["status"] == "completed"
        events = prefix_rows(run/"logs/events.jsonl")
        stages, draws = [], []
        for stage in range(1, 20):
            record = json.loads(checked(run/f"auxiliary/er_stage_{stage:02d}_memory.json").read_text())
            history = [(t, x) for t in range(1, stage) for x in train[t]]
            ordered = sorted(history, key=lambda item: hashlib.sha256(("42:ER:buffer:"+item[1]["id"]).encode()).digest())[:5000]
            assert record["ids"] == [x["id"] for _, x in ordered]
            assert record["capacity"] == 5000 and record["past_train_available"] == len(history)
            assert len(ordered) == len(history), "Capacity eviction occurred; do not claim none"
            proof = json.loads(checked(run/f"stages/stage_{stage:02d}.json").read_text())
            validity = json.loads(checked(run/f"validity/stage_{stage:02d}.json").read_text())
            assert proof["complete"] and validity["passed"] and validity["actual_parameters_changed"]
            updates = [e for e in events if e.get("event") == "train_update" and e["stage"] == stage]
            assert len(updates) == math.ceil(len(train[stage])/16) == validity["positive_lr_updates"]
            current = sorted(train[stage], key=lambda x: hashlib.sha256(f"42:{stage}:{x['id']}".encode()).digest())
            assert hashlib.sha256("".join(x["id"]+"\n" for x in current).encode()).hexdigest() == manifest["tasks"][stage-1]["training_order_sha256"]
            assert updates[-1]["qa_seen"] == len(current)
            for e in updates:
                assert math.isfinite(e["loss"]) and math.isfinite(e["gradient_l2_preclip"]) and e["gradient_l2_preclip"] > 0
                assert any(x > 0 for x in e["learning_rate_used"])
            stage_draws = []
            if ordered:
                for current_row in current:
                    index = int.from_bytes(hashlib.sha256(f"42:ER:draw:{stage}:{current_row['id']}".encode()).digest()[:8], "big") % len(ordered)
                    tid, old = ordered[index]
                    d = {"stage": stage, "current_id": current_row["id"], "replayed_task": tid,
                         "replayed_id": old["id"], "metadata": {
                             k: old.get("metadata", {})[k] for k in ["q_lang", "answer_type"] if k in old.get("metadata", {})}}
                    draws.append(d)
                    stage_draws.append(d)
            stages.append({"stage": stage, "all_past_QA": len(history), "recorded_memory": len(record["ids"]),
                "capacity_evictions": 0, "optimizer_QA": len(current), "reconstructed_replay_QA": len(stage_draws)})
        after = [d for d in draws if d["stage"] > 4]
        old_anchor = [d for d in after if d["replayed_task"] == 4]
        coverage = []
        for t in range(1, 20):
            selected_draws = [d for d in draws if d["replayed_task"] == t]
            coverage.append({"task": t, "original_TRAIN_QA": len(train[t]), "replay_exposures": len(selected_draws),
                             "unique_replayed_QA": len({d["replayed_id"] for d in selected_draws})})
        replay = {"method": "ER_LoRA", "run": row["run"], "capacity": 5000,
            "max_recorded_history": max(x["recorded_memory"] for x in stages), "all19_memories_exact": True,
            "capacity_evictions": 0, "stages": stages, "per_task_coverage": coverage,
            "after_Organ_total_draws": len(after), "after_Organ_anchor_draws": len(old_anchor),
            "after_Organ_anchor_unique_QA": len({d["replayed_id"] for d in old_anchor}),
            "Organ_TRAIN_author_types": dict(collections.Counter(x["metadata"]["answer_type"] for x in train[4])),
            "after_Organ_anchor_draw_author_types": dict(collections.Counter(d["metadata"]["answer_type"] for d in old_anchor)),
            "after_Organ_anchor_unique_author_types": {typ: len({d["replayed_id"] for d in old_anchor if d["metadata"]["answer_type"] == typ}) for typ in ["OPEN", "CLOSED"]},
            "all_draws": draws,
            "evidence_type": "Frozen-source deterministic draw reconstruction plus all recorded buffers and verified one-pass positive-update TRAIN loops; not separately logged per-row GPU draw IDs"}
    assert replay
    for p, expected in sources.items():
        assert sha(ROOT/p) == expected
    for item in prefixes:
        assert hashlib.sha256((ROOT/item["path"]).read_bytes()[:item["bytes"]]).hexdigest() == item["sha256"]
    report = {"CPU_only": True, "new_GPU_operations": 0, "stage4_resources": resources,
        "ER_memory_and_draws": replay, "source_sha256": sources, "resource_and_event_prefixes": prefixes,
        "limits": ["No capacity eviction excludes that mechanism in this run, not all replay coverage/gradient/retention causes.",
                   "Reconstructed exposure is not semantic information coverage or proof it caused the observed OPEN F1 decline.",
                   "Different authored method capacities and learning rates remain; these observations do not isolate a router mechanism.",
                   "Parameter counts are not actual process RAM or proof of resource superiority; raw same-PID resource windows supplied."]}
    (OUT/"MECHANISM_CONTEXT.json").write_text(json.dumps(report, indent=2, ensure_ascii=True)+"\n")
    print(json.dumps({"parameters": {x["method"]: x["trainable_parameters"] for x in resources},
        "replay": {k: v for k, v in replay.items() if k not in ["stages", "per_task_coverage", "all_draws"]}}))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
