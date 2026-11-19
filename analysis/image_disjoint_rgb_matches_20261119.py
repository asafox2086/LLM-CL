"""Locate every training QA contributing to the one processor-input RGB collision."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image
from image_disjoint_rgb_audit_20261119 import Capture,PixelsCaptured
from image_disjoint_shared_20261119 import MANIFEST,MANIFEST_SHA,NAME
from full_only_20261118 import read,sha


def main(root):
    scope=root/"summary/full_image_disjoint_20261119"
    audit=read(scope/"RGB_INPUT_AUDIT.json")
    assert audit["rgb_shared"]==1 and audit["test_questions_with_shared_rgb"]["finance"]==1
    assert sha(root/MANIFEST)==audit["manifest"]==MANIFEST_SHA
    port=root/"exp"/NAME
    assert sha(port/"runtime_utils.py")==audit["runtime"]
    sys.path.insert(0,str(port))
    import runtime_utils as rt
    assert rt.ROOT==root
    rows={}
    train=set();finance=set();test=set()
    for t in read(root/MANIFEST)["tasks"]:
        for split,desc in t["files"].items():
            p=root/desc["path"]
            assert sha(p)==desc["sha256"]
            with p.open() as f:
                values=[json.loads(line) for line in f]
            records=[(v["id"],v["images"]) for v in values]
            rows[t["task"],split]=records
            paths={p for _,images in records for p in images}
            if split=="train":
                train.update(paths)
                if t["domain"]=="finance":finance.update(paths)
            elif t["domain"]=="finance":test.update(paths)
    def pixels(p):
        try:
            rt.input_batch(dict(question="",images=[p]),dict(instruction=""),Capture(),"cpu")
        except PixelsCaptured as found:
            return found.args[0]
        raise AssertionError(p)
    def size(p):
        with Image.open(root/p) as im:
            w,h=im.size
        if w*h>401408:
            scale=math.sqrt(401408/(w*h))
            return max(1,int(w*scale)),max(1,int(h*scale))
        return w,h
    known={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        def fill(paths):
            ordered=sorted(set(paths)-known.keys())
            known.update(zip(ordered,pool.map(pixels,ordered)))
        fill(test|finance)
        test_keys={known[p] for p in test}
        collision=test_keys&{known[p] for p in finance}
        other=sorted(train-known.keys())
        sizes=dict(zip(other,pool.map(size,other)))
        candidates={size(p) for p in test if not collision or known[p] in collision}
        fill(p for p,s in sizes.items() if s in candidates)
    hits=test_keys&{known[p] for p in train if p in known}
    assert len(hits)==1
    key=hits.pop()
    matches={p for p,k in known.items() if k==key}
    excluded={}
    affected_test={}
    for (i,split),records in rows.items():
        affected=[rid for rid,images in records if matches.intersection(images)]
        if affected:(excluded if split=="train" else affected_test)[str(i)]=affected
    assert sum(map(len,affected_test.values()))==1 and excluded
    report=dict(manifest=MANIFEST_SHA,audit=sha(scope/"RGB_INPUT_AUDIT.json"),source=sha(Path(__file__)),
                runtime=audit["runtime"],rgb_sha256=key,train_ids=excluded,test_ids=affected_test,
                train_images={p:sha(root/p) for p in sorted(matches&train)},
                test_images={p:sha(root/p) for p in sorted(matches&test)})
    with (scope/"RGB_MATCHES.json").open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps(report))


if __name__=="__main__":
    main(Path(sys.argv[1]))
