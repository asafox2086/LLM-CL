"""Supplemental paired description, only after the original four-case screen ends."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCREEN = ROOT/'summary/edge_deployment_20261019/osft_increment_screen/v1'
OUT = SCREEN/'paired_analysis_20261022/v1'
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
CASES = ('LoRA_qv', 'OSFT_all', 'LoRA_all', 'LoRA_extra')
PAIRS = (('LoRA_all', 'LoRA_qv'), ('LoRA_extra', 'LoRA_qv'),
         ('LoRA_all', 'OSFT_all'), ('LoRA_extra', 'OSFT_all'),
         ('OSFT_all', 'LoRA_qv'), ('LoRA_all', 'LoRA_extra'))
REPLICATES = 10000
SEED = 20261022


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def known_groups(split):
    wanted = [i for ids in split['dev_ids'].values() for i in ids]
    groups = {d:[] for d in DOMAINS}
    seen = []
    for gid, g in sorted(split['groups'].items()):
        if g['dev']:
            assert g['domain'] in groups and g['ids']
            groups[g['domain']].append((gid, tuple(g['ids'])))
            seen.extend(g['ids'])
    assert sorted(seen) == sorted(wanted) and len(seen) == len(set(seen)) == 100
    assert all(sum(len(ids) for _, ids in groups[d]) == 20 for d in DOMAINS)
    return groups


def percentile(values, p):
    ordered = sorted(values)
    x = p*(len(ordered)-1)
    lo, hi = math.floor(x), math.ceil(x)
    return ordered[lo] + (x-lo)*(ordered[hi]-ordered[lo])


def comparison(values):
    return dict(higher=sum(x>0 for x in values), lower=sum(x<0 for x in values),
                equal=sum(x==0 for x in values))


def main():
    # This is not another queue owner. A missing end marker is a clean deferral.
    if not (SCREEN/'SCREEN_COMPLETE.json').exists():
        print(json.dumps(dict(status='pending_original_SCREEN_COMPLETE', new_model_requests=0)))
        return
    if hasattr(os, 'nice'):
        os.nice(10)
    started = time.monotonic()
    assert not (OUT/'RESULTS.json').exists(), 'Retain the completed analysis; do not overwrite'
    config = json.loads((OUT/'DESIGN.json').read_text())
    assert sha(Path(__file__)) == config['analysis_sha256']
    assert sha(OUT/'PLAN.md') == config['plan_sha256']
    assert sha(SCREEN/'SPLIT.json') == config['split_sha256']
    assert sha(SCREEN/'READY.json') == config['ready_sha256']
    assert (config['replicates'], config['seed']) == (REPLICATES, SEED)
    ready = json.loads((SCREEN/'READY.json').read_text())
    assert all(sha(ROOT/p) == h for p, h in ready['sources'].items())
    split = json.loads((SCREEN/'SPLIT.json').read_text())
    groups = known_groups(split)
    assert all(len(v) < 256 for v in groups.values())
    sys.path.insert(0, str(ROOT/'exp/osft_increment_screen_20261019/v1'))
    spec = importlib.util.spec_from_file_location('frozen_screen_dispatch', ROOT/'exp/osft_increment_screen_20261019/v1/dispatch.py')
    dispatch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dispatch)
    screen = json.loads((SCREEN/'SCREEN_COMPLETE.json').read_text())
    assert screen['complete'] and screen['TEST_QA'] == 0
    verified = {c:dispatch.verify(c, ready) for c in CASES}
    assert all(verified.values()) and verified == screen['all_four_results']
    scores = {}
    for c, proof in verified.items():
        rows = [json.loads(line) for p in proof['prediction_sha256']
                for line in (ROOT/p).read_text().splitlines()]
        scores[c] = {r['id']:r['scores']['token_f1'] for r in rows}
        assert len(rows) == len(scores[c]) == 100
        assert all(math.isfinite(v) and 0 <= v <= 100 for v in scores[c].values())
    ids_by_domain = {d:[i for _, ids in groups[d] for i in ids] for d in DOMAINS}
    point = {c:{d:sum(scores[c][i] for i in ids_by_domain[d])/20 for d in DOMAINS} for c in CASES}
    for c in CASES:
        assert all(math.isclose(point[c][d],verified[c]['domains'][d]['F1'],abs_tol=1e-12,rel_tol=0) for d in DOMAINS)
    # Sums and counts preserve QA weighting even for unequal group sizes.
    sums = {d:[(len(ids), [sum(scores[c][i] for i in ids) for c in CASES])
               for _, ids in groups[d]] for d in DOMAINS}
    boot = {f'{a}_minus_{b}':{d:[] for d in (*DOMAINS,'macro')} for a,b in PAIRS}
    rng = random.Random(SEED)
    draws_sha = hashlib.sha256()
    for _ in range(REPLICATES):
        replicate = {c:{} for c in CASES}
        for d in DOMAINS:
            chosen = [rng.randrange(len(sums[d])) for _ in sums[d]]
            draws_sha.update(bytes(chosen))  # All five domains have fewer than 256 groups.
            count = sum(sums[d][j][0] for j in chosen)
            for k,c in enumerate(CASES):
                replicate[c][d] = sum(sums[d][j][1][k] for j in chosen)/count
        for a,b in PAIRS:
            v = boot[f'{a}_minus_{b}']
            for d in DOMAINS:
                v[d].append(replicate[a][d]-replicate[b][d])
            v['macro'].append(sum(replicate[a][d]-replicate[b][d] for d in DOMAINS)/5)
    pairs = {}
    for a,b in PAIRS:
        key = f'{a}_minus_{b}'
        pairs[key] = dict(
            domain_difference_pp={d:point[a][d]-point[b][d] for d in DOMAINS},
            macro_difference_pp=verified[a]['domain_macro']-verified[b]['domain_macro'],
            QA_comparison={d:comparison([scores[a][i]-scores[b][i] for i in ids_by_domain[d]]) for d in DOMAINS},
            known_group_comparison={d:comparison([sum(scores[a][i]-scores[b][i] for i in ids) for _,ids in groups[d]]) for d in DOMAINS},
            exploratory_group_resampling_percentile_interval_pp={d:[percentile(v,.025),percentile(v,.975)] for d,v in boot[key].items()})
    result = dict(
        supplemental_after_qv_seen=True, analysis_sha256=sha(Path(__file__)),
        design_sha256=sha(OUT/'DESIGN.json'), screen_complete_sha256=sha(SCREEN/'SCREEN_COMPLETE.json'),
        complete_sha256={c:sha(SCREEN/(c+'_COMPLETE.json')) for c in CASES},
        split_sha256=sha(SCREEN/'SPLIT.json'), python=platform.python_version(),
        replicates=REPLICATES, seed=SEED, paired_group_draw_sha256=draws_sha.hexdigest(),
        point_domain_F1=point, point_macro_F1={c:verified[c]['domain_macro'] for c in CASES}, pairs=pairs,
        original_candidates_unchanged=screen['possible_full_retrain_candidates'],
        intervals_are_not_tests_or_equivalence_or_guaranteed_coverage=True,
        not_training_seed_uncertainty=True, unknown_cross_group_dependence_not_removed=True,
        new_model_requests=0, new_backward_calls=0, new_optimizer_steps=0,
        analysis_seconds=time.monotonic()-started)
    dispatch.write(OUT/'RESULTS.json', result)
    print(json.dumps(dict(status='complete', path=str((OUT/'RESULTS.json').relative_to(ROOT)),
                          sha256=sha(OUT/'RESULTS.json'), new_model_requests=0)))


if __name__ == '__main__':
    main()
