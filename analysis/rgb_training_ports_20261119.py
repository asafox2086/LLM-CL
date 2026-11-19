"""Materialize versioned training entries against the pixel-disjoint full manifest."""
from pathlib import Path
import json
import sys

from full_only_20261118 import read,sha
from image_disjoint_shared_20261119 import bind


SOURCES={
    "shared":"95b6ff7946a0f39e299d300015381e555da0f9ea773f982c82bf2e657d8a5814",
    "experts":"b4063e8181e406758e13c63b58752d59a9e02795774067dfbdfbda6a0619d9aa",
    "baselines":"5354f2bb075c6501978a7a16db9f5db97b41a41822568bbb8dd2b63998bbb041",
    "rank64":"cd0ae1b8c68a8ea286fc60bc42aa96b3f4e26d63d5b8fe54e97024c00988b5dc",
    "paired_results":"3bbef8f72acae2795f3c267dfe8a8f861bea7c34cc4d50f81a95fc2d7efec56c",
}


def main(root):
    old_manifest="data/FiveDomainCL/image_disjoint_20261119/full/manifest.json"
    new_manifest="data/FiveDomainCL/image_disjoint_rgb_20261119/full/manifest.json"
    report=read((root/new_manifest).with_name("audit.json"))
    assert report["manifest_sha256"]==sha(root/new_manifest)=="9ed3798ab2b9aca66cda39530c94e69333706e2ec6bf1fbb9a2897ae023a6563"
    assert report["parent"]==sha(root/old_manifest) and report["difference_stages"]==[16]
    assert (report["train"],report["updates"],report["test"])==(129460,8099,21454)
    built={}
    for arm,expected in SOURCES.items():
        original=root/"analysis"/f"image_disjoint_{arm}_20261119.py"
        assert sha(original)==expected
        code=original.read_text()
        # Only scope names, split binding and exact QA-count checks change.
        for name in (*SOURCES,"results"):
            code=code.replace(f"image_disjoint_{name}_20261119",f"rgb_clean_{name}_20261119")
        code=code.replace(old_manifest,new_manifest).replace(report["parent"],report["manifest_sha256"])
        code=code.replace("129462","129460")
        if arm=="baselines":
            # The frozen predecessor FullFT port still has its original size assertion.
            assert 'b"{\'full\':131018}",b"{\'full\':129460}"' in code
        if arm=="shared":
            assert new_manifest in code and report["manifest_sha256"] in code
        target=root/"analysis"/f"rgb_clean_{arm}_20261119.py"
        compile(code,str(target),"exec")
        bind(target,code.encode())
        built[target.name]=sha(target)
    scope=root/"summary/rgb_cutover_20261119"
    scope.mkdir(exist_ok=True)
    binding=dict(source=sha(Path(__file__)),source_files=SOURCES,entries=built,manifest=report["manifest_sha256"])
    bind(scope/"PORTS.json",json.dumps(binding,indent=2).encode())
    print(json.dumps(binding))


if __name__=="__main__":
    main(Path(sys.argv[1]))
