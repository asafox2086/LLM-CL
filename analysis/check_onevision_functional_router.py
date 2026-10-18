"""Synthetic CPU checks; no claim about the actual model's Answer F1."""
import json
import torch
from onevision_functional_router_core import fit, probabilities, choose
from onevision_functional_router_probe import ScaledLoRA


def main():
    torch.set_num_threads(2)
    x = torch.tensor([[.9, .1, .1], [.1, .9, .1], [.1, .1, .9], [-.7, -.7, -.7]])
    losses = torch.tensor([[5., .1, 5., 5.], [5., 5., .1, 5.], [5., 5., 5., .1], [.1, 5., 5., 5.]])
    w, log = fit(x, losses)
    checks = {"16_parameters": w.numel() == 16, "actual_fit": log["changed"],
              "loss_decreases": log["events"][-1]["loss"] < log["events"][0]["loss"],
              "positive_gradient": log["events"][0]["grad_l2"] > 0,
              "hard_original_reject": choose("original_hard", x[0], w, 1.)[0] is None,
              "hard_original_accept": choose("original_hard", x[0], w, .5)[0] == 0,
              "soft_keeps_expert": choose("geometry_soft", x[-1], w, 1.)[0] is not None,
              "soft_range": all(0 <= choose("quality_soft", row, w, .5)[1] <= 1 for row in x),
              "trained_base_choice": choose("quality_hard", x[-1], w, .5)[0] is None,
              "trained_expert_choice": [choose("quality_hard", row, w, .5)[0] for row in x[:3]] == [0, 1, 2],
              "finite_probability": all(torch.isfinite(probabilities(row, w)).all() for row in x)}
    class Old(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(6, 8))
            self.bias = None
            self.A, self.B = torch.nn.Linear(8, 2, bias=False), torch.nn.Linear(2, 6, bias=False)
            self.drop, self.enabled = torch.nn.Dropout(.1), True
    original = Old().eval()
    scaled = ScaledLoRA(original).eval()
    z = torch.randn(3, 8)
    base = torch.nn.functional.linear(z, original.weight)
    delta = 4*original.B(original.A(z))
    checks["actual_layer_gain1_exact"] = torch.equal(scaled(z), base+delta)
    scaled.gain = 0.
    checks["actual_layer_gain0_base_exact"] = torch.equal(scaled(z), base)
    scaled.gain = .5
    checks["actual_layer_half_residual_exact"] = torch.equal(scaled(z), base+delta*.5)
    checks["no_GPU_initialization"] = not torch.cuda.is_initialized()
    assert all(checks.values()), checks
    print(json.dumps({"passed": True, "CPU_only": True, "synthetic_not_model_performance": True, "checks": checks}))


if __name__ == "__main__":
    main()
