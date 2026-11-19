"""Compare acquisition, retention and conditional routing on identical full-data test items."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from full_only_20261118 import HOSTS, MANIFEST_SHA, METHODS, NAME, NATIVE_FILES, read, sha


IMAGE_INDEX_SHA = "0e2f8d769d50738df96b28c1b673c16a96edee9fa6a1b18ae45de4a185c24683"
DRIVER_SHA = "c2d65255934dc2d01a2d59ae0a77acb76a53aa50f07a0c27e26aa367b6af1d39"
COLUMNS = ("shared_acquisition", "shared_final", "expert_acquisition", "expert_final", "fixed_final")
CONTRASTS = {
    "shared_minus_expert_acquisition": [1, 0, -1, 0, 0],
    "shared_minus_expert_final": [0, 1, 0, -1, 0],
    "change_in_shared_minus_expert": [-1, 1, 1, -1, 0],
    "dynamic_minus_fixed_final": [0, 0, 0, 1, -1],
}


def clusters(image_lists):
    parent = list(range(len(image_lists)))

    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, images in enumerate(image_lists):
        assert images
        for image in images:
            parent[find(i)] = find(first.setdefault(image, i))
    return np.unique([find(i) for i in range(len(parent))], return_inverse=True)[1]


def load_test(root, design):
    manifest = root / design["manifest"]
    assert sha(manifest) == design["manifest_sha256"] == MANIFEST_SHA
    image_index = root / "summary/budget_matched_20261117/IMAGE_EXPOSURE_AUDIT.json"
    assert sha(image_index) == IMAGE_INDEX_SHA
    images = read(image_index)["image_content_hashes"]
    tasks = read(manifest)["tasks"]
    assert [t["task"] for t in tasks] == list(range(1, 20))
    by_domain, assignments, truth, all_ids = {}, {}, {}, set()
    for task in tasks:
        i, domain = task["task"], task["domain"]
        desc = task["files"]["test"]
        test = root / desc["path"]
        assert sha(test) == desc["sha256"]
        rows = [json.loads(line) for line in test.read_bytes().splitlines()]
        ids = {row["id"] for row in rows}
        assert len(ids) == len(rows) == desc["count"] and all_ids.isdisjoint(ids)
        all_ids.update(ids)
        truth[i] = rows
        group = by_domain.setdefault(domain, [])
        group.extend([[images[p] for p in row["images"]] for row in rows])
    assert len(all_ids) == 21454 and len(by_domain) == 5
    seen = set()
    for domain, lists in by_domain.items():
        used = {image for row in lists for image in row}
        assert used.isdisjoint(seen)
        seen.update(used)
        assignments[domain] = clusters(lists)
    return tasks, truth, assignments


def check_terminal(scope, port, design):
    assert set(read(scope / "PAIR_FINISHED.json")["methods"]) == set(METHODS)
    runs, receipts = {}, {}
    for method in METHODS:
        run = port / "runs" / design["backbone"] / method
        ident = read(scope / (method + "_IDENTITY.json"))
        end = read(scope / (method + "_EXIT.json"))
        start = read(run / "STARTED.json")
        assert end["rc"] == 0 and end["pid"] == ident["pid"] == start["pid"]
        assert start["startticks"] == ident["startticks"] and start["method"] == method
        stat = Path("/proc") / str(ident["pid"]) / "stat"
        try:
            ticks = stat.read_text().rsplit(") ", 1)[1].split()[19]
        except FileNotFoundError:
            ticks = None
        assert ticks != str(ident["startticks"])
        complete = read(run / "COMPLETE.json")
        assert (complete["train"], complete["test_per_branch"], complete["stages"]) == (131018, 21454, 19)
        assert complete["branches"] == (2 if method == "CL_MoE" else 1)
        cfg = read(run / "config.json")
        assert (cfg["condition"], cfg["manifest_sha256"], cfg["seed"], cfg["backbone"], cfg["method"]) == (
            "full", MANIFEST_SHA, 42, design["backbone"], method)
        assert all(cfg["execution_source_sha256"][name] == digest for name,digest in NATIVE_FILES.items())
        status = read(run / "run_status.json")
        assert status["status"] == "completed" and status["completed_stage"] == 19
        snapshot = read(run / "checkpoints/stage_19.json")
        assert snapshot["stage"] == 19 and sha(run / snapshot["path"]) == snapshot["sha256"]
        updates = [read(run / f"validity/stage_{i:02}.json") for i in range(1, 20)]
        assert all(u["passed"] and u["actual_parameters_changed"] for u in updates)
        assert sum(u["updates"] for u in updates) == 8197
        runs[method] = run
        receipts[method] = dict(config_sha256=sha(run/"config.json"), checkpoint=snapshot,
                                complete_sha256=sha(run/"COMPLETE.json"), exit=end)
    return runs, receipts


def scores(run, stage, task, truth, hashes, root):
    p = run / f"evaluations/stage_{stage:02}/T{task['task']:02}_test_predictions.jsonl"
    predictions = [json.loads(line) for line in p.read_bytes().splitlines()]
    assert len(predictions) == len(truth)
    assert all((x["id"], x["references"], x["generation_cap"]) ==
               (y["id"], y["answers"], task["max_new_tokens"]) for x,y in zip(predictions, truth))
    values = np.asarray([r["scores"]["token_f1"] for r in predictions])
    assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 100))
    native = p.with_suffix(".scores.json")
    assert abs(values.mean() - read(native)["token_f1"]) < 1e-8
    for f in (p, native):
        hashes[str(f.relative_to(root))] = sha(f)
    return values


def describe(point, draws):
    result = dict(zip(COLUMNS, point.tolist()))
    for name, weights in CONTRASTS.items():
        result[name] = dict(points=float(point @ weights),
                            ci95=np.quantile(draws @ weights, [.025, .975]).tolist())
    return result


def paired_results(values, groups):
    rng = np.random.default_rng(20261119)
    means, samples, reports = [], [], {}
    for domain, indices in sorted(groups.items()):
        matrix = np.concatenate(values[domain])
        assert matrix.shape == (len(indices), len(COLUMNS))
        counts = np.bincount(indices)
        clusters_sum = np.column_stack([counts] + [
            np.bincount(indices, weights=col) for col in matrix.T])
        n = len(counts)
        parts = []
        for _ in range(50):
            totals = rng.multinomial(n, np.full(n, 1/n), size=100) @ clusters_sum
            parts.append(totals[:, 1:] / totals[:, :1])
        draw = np.concatenate(parts)
        mean = matrix.mean(axis=0)
        means.append(mean)
        samples.append(draw)
        reports[domain] = describe(mean, draw)
    return dict(macro=describe(np.mean(means, axis=0), np.mean(samples, axis=0)), domains=reports)


def main(root, execute):
    scope, port = root/"summary"/NAME, root/"exp"/NAME
    design = read(scope/"DESIGN.json")
    assert design["source_sha256"] == sha(root/"analysis/full_only_20261118.py") == DRIVER_SHA
    assert design["backbone"] == HOSTS[design["host"]]
    tasks, truth, groups = load_test(root, design)
    protocol = dict(backbone=design["backbone"], manifest=MANIFEST_SHA, index=IMAGE_INDEX_SHA,
                    source=sha(Path(__file__)), columns=COLUMNS, contrasts=CONTRASTS,
                    draws=5000, seed=20261119, domains={
                        k: dict(questions=len(v), image_clusters=int(v.max())+1) for k,v in groups.items()})
    if not execute:
        print(json.dumps(protocol))
        return
    out = scope/"FULL_PAIRED_RESULTS.json"
    if out.exists():
        raise FileExistsError(out)
    runs, receipts = check_terminal(scope, port, design)
    values, proof = {}, {}
    for task in tasks:
        i = task["task"]
        shared, experts = runs["SeqFT_LoRA"], runs["CL_MoE"]
        paths = [(shared,i), (shared,19), (experts,i), (experts,19), (experts/"fixed",19)]
        unique = {key: scores(*key, task, truth[i], proof, root) for key in dict.fromkeys(paths)}
        values.setdefault(task["domain"], []).append(np.column_stack([unique[key] for key in paths]))
    result = dict(protocol=protocol, terminal=receipts, files=proof, **paired_results(values, groups))
    # Publish the complete JSON once, even if two readers finish at the same time.
    tmp = scope/f".paired_results_{os.getpid()}.tmp"
    try:
        with tmp.open("x") as f:
            json.dump(result,f,indent=2,allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp,out)
    finally:
        tmp.unlink(missing_ok=True)
    print(json.dumps(result["macro"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("root", type=Path)
    cli.add_argument("--execute", action="store_true")
    args = cli.parse_args()
    main(args.root, args.execute)
