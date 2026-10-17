"""Exercise the actual HF scheduler/AdamW against all frozen stage budgets."""
import hashlib
import json
import math
import sys
from pathlib import Path
import torch
from transformers import get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'exp/two_backbone_native_cl_20261017'
sys.path.insert(0, str(EXP))
from training_validity import warmup_updates, parameter_digest, validate_stage
from run import load_protocol


def main():
    cases = []
    for condition in ['pct01', 'full']:
        _, manifest, _, data = load_protocol(condition)
        for task in manifest['tasks']:
            updates = math.ceil(len(data[task['task'], 'train']) / 16)
            result = {'condition': condition, 'task': task['task'], 'updates': updates}
            for name, warm in [('legacy', math.ceil(.03 * updates)), ('corrected', warmup_updates(updates))]:
                p = torch.nn.Parameter(torch.tensor([1.]))
                optimizer = torch.optim.AdamW([p], lr=1e-4, weight_decay=0)
                scheduler = get_cosine_schedule_with_warmup(optimizer, warm, updates)
                before = parameter_digest([p])
                used = []
                for _ in range(updates):
                    optimizer.zero_grad()
                    p.sum().backward()
                    used.append(optimizer.param_groups[0]['lr'])
                    optimizer.step()
                    scheduler.step()
                after = parameter_digest([p])
                result[name] = {'warmup': warm, 'positive_lr_updates': sum(x > 0 for x in used),
                                'first_lr_used': used[0], 'parameters_changed': before != after}
                if name == 'corrected':
                    assert validate_stage(updates, sum(x > 0 for x in used), before, after)['passed']
            cases.append(result)
    assert sum(c['legacy']['positive_lr_updates'] == 0 for c in cases if c['condition'] == 'pct01') == 9
    assert all(c['legacy']['positive_lr_updates'] > 0 for c in cases if c['condition'] == 'full')
    p = torch.nn.Parameter(torch.tensor(1., dtype=torch.bfloat16))
    before = parameter_digest([p])
    p.data.add_(1)
    assert before != parameter_digest([p])
    for args in [(1, 0, 'a', 'b'), (1, 1, 'a', 'a')]:
        try:
            validate_stage(*args)
        except RuntimeError:
            pass
        else:
            raise AssertionError('Invalid stage admitted')
    proof = {'actual_HF_scheduler_AdamW_stage_budget_cases': 38,
             'all_corrected_have_positive_lr_and_changed_parameters': True,
             'legacy_pct01_zero_update_stages': 9, 'legacy_full_has_positive_steps': True,
             'invalid_stage_guard_checked': True, 'BF16_scalar_parameter_digest_checked': True,
             'CPU_only_not_GPU_accuracy_proof': True, 'cases': cases,
             'sources_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')},
             'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out = ROOT / 'summary/two_backbone_native_cl_20261017/scheduler_validation.json'
    out.write_text(json.dumps(proof, indent=2) + '\n')
    print('Shared scheduler: all38 actual stage-budget cases passed')


if __name__ == '__main__':
    main()
