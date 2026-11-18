"""Compare two trained algorithms on the same QA immediately and at sequence end."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


RUNS = {
    'onevision': ('training_20261017_161154', 'training_20261017_161424'),
    'qwen': ('training_20261019_verified_resume_branch_v2', 'training_20261022_162837'),
}
METHODS = ('SeqFT_LoRA', 'CL_MoE')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(root, backbone):
    manifest_path = root / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    exposure = root / 'summary/budget_matched_20261117/IMAGE_EXPOSURE_AUDIT.json'
    assert digest(exposure) == '0e2f8d769d50738df96b28c1b673c16a96edee9fa6a1b18ae45de4a185c24683'
    image_hash = json.loads(exposure.read_bytes())['image_content_hashes']
    base = root / 'exp/two_backbone_native_cl_20261017_v2/runs' / backbone / 'pct01'
    runs = [base / method / name for method, name in zip(METHODS, RUNS[backbone])]
    proof = {}
    for method, run in zip(METHODS, runs):
        cfg = json.loads((run / 'config.json').read_bytes())
        assert cfg['manifest_sha256'] == digest(manifest_path) and cfg['seed'] == 42
        assert cfg['condition'] == 'pct01' and cfg['method'] == method
    by_domain = defaultdict(list)
    for task in manifest['tasks']:
        i = task['task']
        desc = task['files']['test']
        test = root / desc['path']
        assert digest(test) == desc['sha256']
        truth = [json.loads(s) for s in test.read_bytes().splitlines()]
        assert len(truth) == desc['count']
        columns = []
        for run in runs:
            for stage in (i, 19):
                path = run / f'evaluations/stage_{stage:02}/T{i:02}_test_predictions.jsonl'
                pred = [json.loads(s) for s in path.read_bytes().splitlines()]
                assert len(pred) == len(truth)
                assert all(r['id'] == t['id'] and r['references'] == t['answers'] and
                           r['generation_cap'] == task['max_new_tokens'] for r,t in zip(pred,truth))
                proof[str(path.relative_to(root))] = digest(path)
                scores = np.array([x['scores']['token_f1'] for x in pred])
                assert np.isfinite(scores).all() and scores.min() >= 0 and scores.max() <= 100
                native = json.loads(path.with_suffix('.scores.json').read_bytes())
                assert abs(scores.mean() - native['token_f1']) < 1e-8
                columns.append(scores)
        for j, row in enumerate(truth):
            imgs = [image_hash[p] for p in row['images']]
            by_domain[task['domain']].append((imgs, [float(c[j]) for c in columns]))
    assert len(by_domain) == 5 and sum(map(len, by_domain.values())) == 21454
    rng = np.random.default_rng(20261118)
    reports, domain_draws, domain_means, seen = {}, [], [], set()
    for domain, rows in sorted(by_domain.items()):
        parent = list(range(len(rows)))
        def find(i):
            while i != parent[i]:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        first = {}
        for j, (images, _) in enumerate(rows):
            for image in images:
                parent[find(j)] = find(first.setdefault(image, j))
        assert seen.isdisjoint(first)
        seen.update(first)
        groups = defaultdict(lambda: np.zeros(5))
        for j, (_, scores) in enumerate(rows):
            groups[find(j)] += [1, *scores]
        totals = np.array(list(groups.values()))
        n = len(totals)
        draws = []
        for _ in range(50):
            counts = rng.multinomial(n, np.full(n, 1/n), size=100)
            sums = counts @ totals
            draws.extend(sums[:, 1:] / sums[:, :1])
        points = totals.sum(0)[1:] / len(rows)
        reports[domain] = summarize(points, np.asarray(draws))
        reports[domain].update(questions=len(rows), image_clusters=n)
        domain_means.append(points)
        domain_draws.append(draws)
    macro = summarize(np.mean(domain_means,axis=0), np.mean(domain_draws,axis=0))
    anchors = {'qwen': (48.972,48.072),'onevision': (42.186,42.774)}
    for actual, expected in zip((macro['small_end'],macro['expert_end']),anchors[backbone]):
        assert abs(actual - expected) < .00051
    return dict(backbone=backbone, macro=macro, domains=reports, predictions_sha256=proof,
                manifest_sha256=digest(manifest_path), source_sha256=digest(Path(__file__)),
                units='QA token F1; equal five-domain macro; task 19 has zero subsequent change',
                uncertainty='5000 paired image-component draws by domain, conditional on the trained checkpoints',
                contrast='SeqFT q/v rank8 minus CL-MoE; method comparison, not an isolated parameter-count intervention')


def summarize(points, draws):
    names = ('small_immediate','small_end','expert_immediate','expert_end')
    report = dict(zip(names, points.tolist()))
    for label, coeff in [('immediate_gap',[1,0,-1,0]), ('end_gap',[0,1,0,-1]),
                         ('gap_change_during_later_tasks',[-1,1,1,-1])]:
        report[label] = float(points @ coeff)
        report[label+'_ci'] = np.quantile(draws @ coeff,[.025,.975]).tolist()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('backbone',choices=RUNS)
    args = parser.parse_args()
    result = analyze(args.root,args.backbone)
    out = args.root / 'summary/parameter_efficiency_retention_20261118' / (args.backbone+'.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(macro=result['macro'],domains={k: [v['immediate_gap'],v['end_gap'],v['gap_change_during_later_tasks']] for k,v in result['domains'].items()},sha256=digest(out))))
