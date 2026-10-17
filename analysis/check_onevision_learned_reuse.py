"""CPU mechanics tests on synthetic vectors, not experimental model evidence."""
import json
import torch
import onevision_learned_reuse_core as c


def main():
    torch.set_num_threads(2)
    rows = [{"id": str(i)} for i in range(8)]
    x = torch.eye(8)
    w = c.initialize(8)
    checks = {}
    checks["deterministic_initial"] = torch.equal(w, c.initialize(8))
    memory = c.update_memory([], rows[:4], x[:4], "e0")
    single, log = c.fit(memory, w, True)
    checks["single_expert_no_false_learning_claim"] = torch.equal(single, w) and log["optimizer_updates"] == 0
    memory = c.update_memory(memory, rows[4:], x[4:], "e1")
    fixed, log = c.fit(memory, w, False)
    checks["fixed_head_unchanged"] = torch.equal(fixed, w) and not log["parameter_changed"]
    learned, log = c.fit(memory, w, True)
    checks["two_expert_learning_actual"] = log["parameter_changed"] and log["optimizer_updates"] == 40
    checks["finite_nonzero_metric_gradients"] = all(e["gradient_l2"] > 0 for e in log["events"])
    checks["synthetic_loss_decreases"] = log["events"][-1]["loss"] < log["events"][0]["loss"]
    tau, calibration = c.calibrate(memory, learned)
    checks["TRAIN_threshold_finite"] = -1 <= tau <= 1 and len(calibration["negative_cosines"]) == 8
    expert, decision = c.allocate(x[:1], [], w, 0., 0, 3)
    checks["first_new_from_base"] = expert == "e0" and decision["action"] == "new_from_base"
    expert, decision = c.allocate(-x[:2], memory, learned, 1., 2, 3)
    checks["novel_can_expand"] = expert == "e2" and decision["action"] == "new_from_base"
    _, decision = c.allocate(-x[:2], memory, learned, 1., 2, 2)
    checks["cap_respected_explicit_forced_reuse"] = decision["action"] == "reuse" and decision["cap_forced_reuse"]
    expert, _ = c.infer(-x[0], memory, learned, 1.)
    checks["unseen_inference_can_use_base"] = expert is None
    many = [{"id": "m"+str(i)} for i in range(80)]
    bounded = c.update_memory([], many, x[0].repeat(80, 1), "e0")
    checks["memory_bounded"] = len(bounded) == c.MEMORY
    try:
        c.update_memory(memory, rows[:1], x[:1], "e0")
        checks["duplicate_id_rejected"] = False
    except AssertionError:
        checks["duplicate_id_rejected"] = True
    assert all(checks.values()), checks
    print(json.dumps({"passed": True, "CPU_only": True, "synthetic_not_model_results": True, "checks": checks,
                      "synthetic_metric_loss_first": log["events"][0]["loss"],
                      "synthetic_metric_loss_last": log["events"][-1]["loss"]}))


if __name__ == "__main__":
    main()
