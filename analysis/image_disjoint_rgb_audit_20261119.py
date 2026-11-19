"""Check for pixel-identical images across splits at the native image processor's input."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import PIL
from full_only_20261118 import NATIVE_FILES, read, sha
from image_disjoint_shared_20261119 import MANIFEST, MANIFEST_SHA, NAME


class PixelsCaptured(Exception):
    pass


class Capture:
    def apply_chat_template(self,*args,**kwargs):
        return ""

    def __call__(self,text,images,return_tensors):
        image,=images
        assert image.mode=="RGB"
        digest=hashlib.sha256(struct.pack(">II",image.width,image.height))
        digest.update(image.tobytes())
        raise PixelsCaptured(digest.hexdigest())


def main(root):
    scope=root/"summary/full_image_disjoint_20261119"
    output=scope/"RGB_INPUT_AUDIT.json"
    assert not output.exists()
    port=root/"exp"/NAME
    assert sha(port/"runtime_utils.py")==NATIVE_FILES["runtime_utils.py"]
    sys.path.insert(0,str(port))
    import runtime_utils as rt
    assert rt.ROOT==root and sha(root/MANIFEST)==MANIFEST_SHA
    manifest=read(root/MANIFEST)
    rows={}
    paths=set()
    for t in manifest["tasks"]:
        for split,desc in t["files"].items():
            path=root/desc["path"]
            assert sha(path)==desc["sha256"]
            with path.open() as f:
                data=[json.loads(line)["images"] for line in f]
            assert len(data)==desc["count"] and all(data)
            rows[t["task"],split]=data
            paths.update(image for images in data for image in images)
    def scan(path):
        try:
            rt.input_batch(dict(question="",images=[path]),dict(instruction=""),Capture(),"cpu")
        except PixelsCaptured as found:
            return found.args[0]
        raise AssertionError("Processor did not receive an RGB image")
    ordered=sorted(paths)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pixels=dict(zip(ordered,pool.map(scan,ordered)))
    train={pixels[p] for t in manifest["tasks"] for images in rows[t["task"],"train"] for p in images}
    total=Counter()
    matched=Counter()
    seen_test=set()
    for t in manifest["tasks"]:
        d=t["domain"]
        for images in rows[t["task"],"test"]:
            keys={pixels[p] for p in images}
            seen_test.update(keys)
            total[d]+=1
            matched[d]+=bool(keys&train)
    assert sum(total.values())==21454
    report=dict(manifest=MANIFEST_SHA,pillow=PIL.__version__,runtime=sha(port/"runtime_utils.py"),
                source=sha(Path(__file__)),file_paths=len(pixels),rgb_train=len(train),rgb_test=len(seen_test),
                rgb_shared=len(train&seen_test),test_questions=dict(total),test_questions_with_shared_rgb=dict(matched),
                input="Actual runtime_utils.input_batch RGB image after common area-cap resize, immediately before the backbone processor. Exact equality of width, height and RGB bytes. Excludes similarity and later processor transformations.",
                new_model_runs=0,unix=time.time())
    tmp=output.with_suffix(".tmp")
    with tmp.open("x") as f:
        json.dump(report,f,indent=2)
        f.flush();os.fsync(f.fileno())
    os.link(tmp,output)
    tmp.unlink()
    print(json.dumps(report))


if __name__=="__main__":
    main(Path(sys.argv[1]))
