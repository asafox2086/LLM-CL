"""TRAIN-only accounting of the frozen matched-study replay policy.

No Torch/model loading, no test prediction reading, no new GPU jobs. Planned
coverage is separated from replay IDs actually verified in completed GPU runs.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "summary/onevision_diagnostics_20261016/matched_task_domain"
OUT = BASE / "replay_exposure"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_and_replay(history, current, stage):
    memory = sorted(set(history), key=lambda i: hashlib.sha256(("42:memory:"+i).encode()).digest())[:8]
    replay = [memory[int.from_bytes(hashlib.sha256(f"42:{stage}:{i}".encode()).digest()[:8], "big") % len(memory)]
              for i in current] if memory else []
    return memory, replay


def self_check():
    memory, replay = memory_and_replay([], ["new"], 1)
    assert not memory and not replay
    history = ["old1", "old2", "old1"]
    a = memory_and_replay(history, ["new1", "new2"], 2)
    b = memory_and_replay(list(reversed(history)), ["new1", "new2"], 2)
    assert a == b and len(a[0]) == 2 and set(a[1]) <= set(history)
    assert not set(a[0]) & {"new1", "new2"}, "New QA cannot enter history before its stage"
    assert "torch" not in sys.modules
    return {"empty_history": True, "deduplicated_order_invariant": True,
            "no_future_QA_in_memory": True, "stdlib_CPU_only": True}


def build():
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}

    def checked(path, digest=None):
        actual = sha(path)
        assert digest is None or actual == digest, str(path)
        sources[str(path.relative_to(ROOT))] = actual
        return path

    plan = json.loads(checked(BASE / "CPU_PLAN_CHECK.json").read_text())
    assert plan["passed"] and plan["GPU_used"] is False
    spec = plan["spec"]
    for p, h in spec["source_sha256"].items():
        checked(ROOT / p, h)
    manifest = json.loads(checked(ROOT / "data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json", spec["manifest_sha256"]).read_text())
    rows, task_of, domain_of = {}, {}, {}
    # Read only TRAIN metadata. Neither QA answers nor question text chooses a
    # replay slot: the existing source's fixed hashes remain the sole policy.
    for task in manifest["tasks"]:
        t = task["task"]
        if str(t) not in spec["TRAIN_ids"]:
            continue
        f = task["files"]["train"]
        selected = set(spec["TRAIN_ids"][str(t)])
        for row in map(json.loads, checked(ROOT / f["path"], f["sha256"]).read_text().splitlines()):
            if row["id"] in selected:
                i = row["id"]
                assert i not in rows
                rows[i], task_of[i], domain_of[i] = row, t, task["domain"]
        assert selected <= rows.keys()
    profiles, completed = [], []
    for anchor in spec["anchors"]:
        anchor_ids = set(spec["TRAIN_ids"][str(anchor)])
        for sequence, tasks in spec["sequences"][str(anchor)].items():
            name = f"T{anchor:02d}_{sequence}__ER_LoRA"
            proof_path = BASE / (name + "_complete.json")
            proof = None
            if proof_path.exists():
                proof = json.loads(checked(proof_path).read_text())
                assert proof["complete"] and proof["stage1_exact"] and proof["spec"] == spec
                completed.append(name)
            history, stages = [], []
            for stage, task in enumerate(tasks, 1):
                current = spec["TRAIN_ids"][str(task)]
                assert len(current) == len(set(current)) == 8
                memory, replay = memory_and_replay(history, current, stage)
                assert set(memory) <= set(history) and set(replay) <= set(memory)
                if task not in tasks[:stage-1]:
                    assert not set(memory) & set(current)
                actual_verified = False
                if proof:
                    relative = f"training/{name}/stage_{stage:02d}.json"
                    p = (ROOT / proof["run"] / relative).resolve()
                    assert p.is_relative_to(ROOT / "exp/onevision_matched_task_domain_20261017")
                    t = json.loads(checked(p, proof["artifact_sha256"][relative]).read_text())
                    events = t["events"]
                    assert len(events) == 4
                    assert [i for e in events for i in e["train_ids"]] == current
                    assert [i for e in events for i in e["replay_ids"]] == replay
                    assert all(e["memory_size"] == len(memory) for e in events)
                    assert t["new_plus_replay_exposures"] == 8+len(replay)
                    actual_verified = True
                stages.append({"stage": stage, "task": task, "domain": domain_of[current[0]],
                               "memory_QA_IDs": memory, "memory_QA": len(memory),
                               "memory_unique_image_paths": len({tuple(rows[i]["images"]) for i in memory}),
                               "memory_anchor_QA": sum(i in anchor_ids for i in memory),
                               "memory_tasks": dict(Counter(task_of[i] for i in memory)),
                               "memory_domains": dict(Counter(domain_of[i] for i in memory)),
                               "replay_QA_IDs_in_order": replay, "replay_QA_exposures": len(replay),
                               "replay_unique_QA": len(set(replay)),
                               "replay_anchor_exposures": sum(i in anchor_ids for i in replay),
                               "replay_unique_anchor_QA": len(set(replay) & anchor_ids),
                               "replay_unique_image_paths": len({tuple(rows[i]["images"]) for i in replay}),
                               "replay_tasks": dict(Counter(task_of[i] for i in replay)),
                               "replay_domains": dict(Counter(domain_of[i] for i in replay)),
                               "actual_completed_GPU_log_verified": actual_verified})
                history.extend(current)
            assert sum(s["replay_QA_exposures"] for s in stages) == 32
            profiles.append({"anchor": anchor, "sequence": sequence, "case": name,
                             "GPU_case_complete": proof is not None, "stages": stages,
                             "target_QA_exposures": 40, "replay_QA_exposures": 32,
                             "replay_anchor_exposures": sum(s["replay_anchor_exposures"] for s in stages),
                             "replay_unique_anchor_QA": len({i for s in stages for i in s["replay_QA_IDs_in_order"]} & anchor_ids)})
    assert len(profiles) == 8
    for p, digest in sources.items():
        assert sha(ROOT/p) == digest, "Read-only inputs changed during audit"
    assert "torch" not in sys.modules
    return {"CPU_only": True, "TRAIN_only_policy_and_metadata": True,
            "no_TEST_predictions_or_scores_read": True, "no_GPU_or_sampling_change": True,
            "planned_ER_profiles": 8, "actual_completed_ER_cases_verified": len(completed),
            "completed_cases": completed, "profiles": profiles, "self_checks": self_check(), "source_sha256": sources,
            "limits": ["Planned IDs are determined by frozen TRAIN policy, not completed GPU results.",
                       "Only complete ER cases have actual replay-log validation; incomplete runs are not read.",
                       "Eight-slot hash memory is this mechanism control, not every formal baseline's buffer policy.",
                       "Anchor-QA coverage is not semantic knowledge coverage; other tasks/images can transfer.",
                       "Image-path counts are not decoded-pixel/semantic deduplication.",
                       "Coverage differences do not prove final F1 causality or justify TEST-selected sampling.",
                       "No answer contents, gradient scores or future/test data choose replay in this audit."]}


def main():
    result = build(); OUT.mkdir(exist_ok=True)
    p = OUT / "EXPOSURE.json"; t = p.with_suffix(".tmp")
    t.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n"); t.replace(p)
    lines = ["# 固定8条缓冲区，实际保护了多少旧锚题目", "",
             f"TRAIN-only CPU审计：8条ER计划，{result['actual_completed_ER_cases_verified']}条完成GPU日志已精确核验。未完成行仅是冻结规则的计划覆盖，不是模型结果。", "",
             "| 锚/序列 | 实际GPU完成 | 回放总曝光 | 旧锚回放曝光/独特QA | 各阶段旧锚曝光S2–5 |",
             "|---|---|---:|---:|---|"]
    for p in result["profiles"]:
        counts="/".join(str(s["replay_anchor_exposures"]) for s in p["stages"][1:])
        lines.append(f"| T{p['anchor']:02d}/{p['sequence']} | {p['GPU_case_complete']} | {p['replay_QA_exposures']} | {p['replay_anchor_exposures']}/{p['replay_unique_anchor_QA']} | {counts} |")
    lines += ["", "每阶段缓冲区的QA/任务/领域、图像路径、实际抽中与独特旧锚题目见[JSON](EXPOSURE.json)。", "",
              "数量相同不等于历史信息相同：跨序列对照必须同时报告覆盖。这里不改变采样，不把覆盖差异直接解释为准确率原因。", ""]
    lines += ["- "+x for x in result["limits"]]
    (OUT/"VERIFIED_RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"CPU_only":True,"planned":8,"actual_verified":result["actual_completed_ER_cases_verified"]}))


if __name__ == "__main__":
    main()
