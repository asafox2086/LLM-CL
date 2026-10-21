"""Read sealed LoRA_qv logs only; no model/QA/optimizer execution."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
OUT = ROOT/'summary/edge_deployment_20261019/update_geometry_literature_20261022/SHORT_STAGE_ACCOUNTING.json'


def main():
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    ready = json.loads((SCREEN/'READY.json').read_text())
    split = json.loads((SCREEN/'SPLIT.json').read_text())
    complete = SCREEN/'LoRA_qv_COMPLETE.json'
    assert digest(complete) == '09f7594e05dd45734e6b412ae183e067e144fc341c115c163d011d9cc19946bf'
    assert json.loads(complete.read_text())['complete'] is True
    assert digest(SCREEN/'SPLIT.json') == ready['split_sha256']
    sources = ['exp/two_backbone_native_cl_20261017_v2/'+n for n in
               ['run.py', 'runtime_utils.py', 'host_adam.py', 'osft_adapter.py']]
    assert all(digest(ROOT/n) == ready['sources'][n] for n in sources)
    events_path = ROOT/'exp/osft_increment_screen_20261019/v1/private/LoRA_qv/logs/events.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    planned = {t: math.ceil(len(split['fit_ids'][str(t)])/16) for t in range(1, 20)}
    actual = dict(collections.Counter(e['stage'] for e in events))
    assert actual == planned and len(events) == 87
    assert all(e['event'] == 'train_update' and any(x > 0 for x in e['learning_rate_used']) for e in events)
    for t, count in actual.items():
        assert [e['step'] for e in events if e['stage'] == t] == list(range(1, count+1))
    first_steps = [e for e in events if e['step'] == 1]
    assert len(first_steps) == 19
    assert all(math.isfinite(e['gradient_l2_preclip']) and math.isfinite(e['gradient_l2_postclip']) for e in events)
    result = dict(
        kind='existing_complete_reference_training_budget_not_quality_causality',
        analysis_sha256=digest(Path(__file__)),
        complete_sha256=digest(complete), split_sha256=digest(SCREEN/'SPLIT.json'),
        events_sha256=digest(events_path),
        frozen_sources_sha256={n:digest(ROOT/n) for n in sources},
        actual_positive_updates_by_stage=actual, total_positive_updates=len(events),
        fresh_optimizer_first_steps=len(first_steps),
        fresh_optimizer_first_step_fraction=len(first_steps)/len(events),
        one_update_stages=[t for t, n in actual.items() if n == 1],
        updates_with_preclip_l2_above_1=sum(e['gradient_l2_preclip'] > 1 for e in events),
        preclip_l2_min=min(e['gradient_l2_preclip'] for e in events),
        preclip_l2_max=max(e['gradient_l2_preclip'] for e in events),
        postclip_l2_max=max(e['gradient_l2_postclip'] for e in events),
        optimizer_source_facts=dict(recreated_each_stage=True, LoRA_factors_reset_each_stage=False,
                                   OSFT_refactors_each_later_stage=True, both_weight_decay=0,
                                   both_betas=[.9,.999], both_epsilon=1e-8,
                                   same_numeric_LR=False, OSFT_host_state=True),
        new_model_requests=0, new_backward_calls=0, new_optimizer_steps=0,
        partial_OSFT_results_read=False,
        scope='Counts and clipping describe the complete qv run; not a small-data causal or parameter-necessity result.')
    assert not OUT.exists(), 'Do not replace a sealed accounting record'
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
