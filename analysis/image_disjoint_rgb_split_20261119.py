"""Remove located RGB collisions from TRAIN while preserving every original TEST row."""
from collections import Counter
import json
from pathlib import Path
import sys

import full_image_disjoint_split_20261119 as split
from full_only_20261118 import read,sha
from image_disjoint_rgb_audit_20261119 import Capture,PixelsCaptured
from image_disjoint_shared_20261119 import MANIFEST,MANIFEST_SHA,NAME


OUT=Path("data/FiveDomainCL/image_disjoint_rgb_20261119/full")


def main(root):
    scope=root/"summary/full_image_disjoint_20261119"
    match=read(scope/"RGB_MATCHES.json");audit=read(scope/"RGB_INPUT_AUDIT.json")
    assert match["audit"]==sha(scope/"RGB_INPUT_AUDIT.json")
    assert sha(root/MANIFEST)==audit["manifest"]==match["manifest"]==MANIFEST_SHA
    assert audit["rgb_shared"]==1
    manifest=read(root/MANIFEST)
    port=root/"exp"/NAME
    assert sha(port/"runtime_utils.py")==match["runtime"]
    sys.path.insert(0,str(port))
    import runtime_utils as rt
    assert rt.ROOT==root
    overlap=set(match["train_images"])
    for name,digest in {**match["train_images"],**match["test_images"]}.items():
        assert sha(root/name)==digest
        try:
            rt.input_batch(dict(question="",images=[name]),dict(instruction=""),Capture(),"cpu")
        except PixelsCaptured as result:
            assert result.args[0]==match["rgb_sha256"]
        else:
            raise AssertionError(name)
    (root/OUT).mkdir(parents=True,exist_ok=True)
    stages={}
    for task in manifest["tasks"]:
        i=task["task"];desc=task["files"]["train"];old=desc["count"]
        lines=split.source(root,desc);rows=[json.loads(l) for l in lines]
        assert split.order_digest(rows,i)==task["training_order_sha256"]
        excluded={r["id"] for r in rows if overlap.intersection(r["images"])}
        assert excluded==set(match["train_ids"].get(str(i),[]))
        if excluded:
            selected=[(r,l) for r,l in zip(rows,lines) if r["id"] not in excluded]
            blob=b"".join(l for _,l in selected);dest=OUT/f"T{i:02d}_train.jsonl"
            split.save(root/dest,blob)
            desc.update(path=str(dest),count=len(selected),sha256=split.sha256(blob).hexdigest(),
                        dataset_counts=dict(Counter(r["dataset"] for r,_ in selected)))
            task["training_order_sha256"]=split.order_digest([r for r,_ in selected],i)
        stages[i]=dict(before=old,after=desc["count"],removed=len(excluded))
    train=sum(s["after"] for s in stages.values());updates=sum((s["after"]+15)//16 for s in stages.values())
    test=sum(t["files"]["test"]["count"] for t in manifest["tasks"])
    removed=sum(len(v) for v in match["train_ids"].values())
    assert train+removed==129462 and test==21454
    manifest.update(parent_manifest_sha256=MANIFEST_SHA,effective_train_count=train,
                    variant_id=OUT.parent.name,exclusion_reason="Parent exclusions plus all training QA with a shared native processor-input RGB image.")
    encoder=lambda v:(json.dumps(v,indent=2,ensure_ascii=False)+"\n").encode()
    data=encoder(manifest);digest=split.sha256(data).hexdigest()
    split.OUT=OUT
    split.freeze_ids(root,manifest,digest)
    split.save(root/OUT/"manifest.json",data)
    report=dict(manifest=str(OUT/"manifest.json"),manifest_sha256=digest,parent=MANIFEST_SHA,
                train=train,updates=updates,test=test,removed=removed,stages=stages,
                difference_stages=[i for i,s in stages.items() if s["removed"]],
                train_test_shared_common_input_rgb=0,matches=sha(scope/"RGB_MATCHES.json"),
                source=sha(Path(__file__)),lock=sha((root/OUT).parent/"split_lock.json"))
    split.save(root/OUT/"audit.json",encoder(report))
    print(json.dumps(report))


if __name__=="__main__":
    main(Path(sys.argv[1]))
