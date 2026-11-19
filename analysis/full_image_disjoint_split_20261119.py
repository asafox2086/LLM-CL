"""Keep the official test QA, exclude training QA sharing any byte-identical test image."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys


PARENT = Path("data/FiveDomainCL/five_domain_native_cl_20261016_sparse_labeled_20261017/full/manifest.json")
PARENT_SHA = "cf737555dab3646c9916dc63a66f1d52285eebbeed4aa414e807ae5b08a0d952"
OUT = Path("data/FiveDomainCL/image_disjoint_20261119/full")


def save(path, data):
    if path.exists():
        assert path.read_bytes() == data, path
        return
    with path.open("xb") as f:
        f.write(data)


def source(root, desc):
    data = (root / desc["path"]).read_bytes()
    assert sha256(data).hexdigest() == desc["sha256"]
    lines = data.splitlines(keepends=True)
    assert len(lines) == desc["count"]
    return lines


def order_digest(rows, stage):
    ordered = sorted((r["id"] for r in rows), key=lambda i: sha256(f"42:{stage}:{i}".encode()).digest())
    return sha256("".join(i+"\n" for i in ordered).encode()).hexdigest()


def freeze_ids(root, manifest, manifest_sha):
    """Write the same frozen ID list and split lock consumed by every training port."""
    parent = (root/PARENT).parent
    parent_lock = json.loads((parent.parent/"split_lock.json").read_bytes())["conditions"]["full"]
    original = (parent/"frozen_ids.json").read_bytes()
    assert parent_lock["manifest_sha256"] == PARENT_SHA
    assert sha256(original).hexdigest() == parent_lock["frozen_ids_sha256"]
    original_ids = json.loads(original)
    ids, files = {}, {}
    for task in manifest["tasks"]:
        for split, desc in task["files"].items():
            key = task["task_id"]+"/"+split
            data_ids = {json.loads(line)["id"] for line in source(root,desc)}
            ids[key] = [i for i in original_ids[key] if i in data_ids]
            assert len(ids[key]) == len(data_ids) == desc["count"]
            if split == "test":
                assert ids[key] == original_ids[key]
            files[desc["path"]] = desc["sha256"]
    serialized_ids = (json.dumps(ids,indent=2)+"\n").encode()
    target = root/OUT
    save(target/"frozen_ids.json",serialized_ids)
    condition = dict(manifest_sha256=manifest_sha,frozen_ids_sha256=sha256(serialized_ids).hexdigest(),files=files)
    lock = dict(experiment_id=manifest["experiment_id"],variant_id=manifest["variant_id"],
                seed=42,method_independent=True,conditions={"full":condition})
    save(target.parent/"split_lock.json",(json.dumps(lock,indent=2)+"\n").encode())


def main(root):
    original = (root / PARENT).read_bytes()
    assert sha256(original).hexdigest() == PARENT_SHA
    manifest = json.loads(original)
    target = root / OUT
    target.mkdir(parents=True, exist_ok=True)
    digests = {}

    def image_hash(p):
        if p not in digests:
            digests[p] = sha256((root / p).read_bytes()).hexdigest()
        return digests[p]

    tests = {}
    for task in manifest["tasks"]:
        rows = (json.loads(l) for l in source(root, task["files"]["test"]))
        tests.setdefault(task["domain"], []).extend({image_hash(p) for p in row["images"]} for row in rows)
    held_out = set().union(*(s for rows in tests.values() for s in rows))
    seen, kept_images = set(), set()
    counts = {}
    for task in manifest["tasks"]:
        i = task["task"]
        desc = task["files"]["train"]
        lines = source(root, desc)
        rows = [json.loads(l) for l in lines]
        assert order_digest(rows, i) == task["training_order_sha256"]
        retained, content = [], []
        for row, line in zip(rows, lines):
            images = {image_hash(p) for p in row["images"]}
            seen.update(images)
            if images.isdisjoint(held_out):
                retained.append(row)
                content.append(line)
                kept_images.update(images)
        assert retained
        if len(rows) != len(retained):
            path = OUT / f"T{i:02d}_train.jsonl"
            data = b"".join(content)
            save(root/path, data)
            desc.update(path=str(path),sha256=sha256(data).hexdigest(),count=len(retained),
                        dataset_counts=dict(Counter(r["dataset"] for r in retained)))
            task["training_order_sha256"] = order_digest(retained,i)
        counts[str(i)] = dict(domain=task["domain"],original=len(rows),clean=len(retained))
    assert held_out.isdisjoint(kept_images)
    clean = sum(c["clean"] for c in counts.values())
    assert sum(c["original"] for c in counts.values()) == 131018 and sum(map(len,tests.values())) == 21454
    manifest.update(parent_manifest_sha256=PARENT_SHA,variant_id="image_disjoint_20261119",
                    effective_train_count=clean,parent_1pct_ids_preserved=False,
                    exclusion_reason="Parent exclusions plus all TRAIN QA sharing image bytes with any TEST QA.")
    encode = lambda obj: (json.dumps(obj,indent=2,ensure_ascii=False)+"\n").encode()
    encoded = encode(manifest)
    report = dict(parent_sha256=PARENT_SHA,manifest_sha256=sha256(encoded).hexdigest(),
                  image_identity="SHA-256 of file bytes; transformed copies not tested",
                  test_qa=21454,train_original=131018,train_clean=clean,train_removed=131018-clean,
                  updates_clean=sum((c["clean"]+15)//16 for c in counts.values()),
                  test_exposed_before={d:sum(bool(images & seen) for images in rows) for d,rows in tests.items()},
                  test_exposed_after={d:sum(bool(images & kept_images) for images in rows) for d,rows in tests.items()},
                  stages=counts)
    save(target/"audit.json", encode(report))
    save(target/"manifest.json", encoded)
    freeze_ids(root,manifest,report["manifest_sha256"])
    print(json.dumps(report))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
