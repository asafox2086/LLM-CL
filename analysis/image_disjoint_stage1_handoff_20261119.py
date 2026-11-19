"""Retire the legacy worker once its reusable, post-consolidation S1 checkpoint is durable."""
import ctypes
import json
import os
from pathlib import Path
import platform
import select
import signal
import sys
import time

from full_only_20261118 import NAME, read, sha, put, ticks
from image_disjoint_shared_20261119 import MANIFEST, MANIFEST_SHA


def checked(result):
    if result < 0:
        number=ctypes.get_errno()
        raise OSError(number,os.strerror(number))
    return result


def main(root):
    scope=root/"summary/full_image_disjoint_20261119"
    old=root/"summary"/NAME
    design=read(old/"DESIGN.json")
    old_manifest=read(root/design["manifest"])
    assert sha(root/MANIFEST)==MANIFEST_SHA
    assert old_manifest["tasks"][0]==read(root/MANIFEST)["tasks"][0]
    assert old_manifest["tasks"][0]["files"]["train"]["count"]==34602
    run=root/"exp"/NAME/"runs"/design["backbone"]/"CL_MoE"
    identity=read(old/"CL_MoE_IDENTITY.json")
    p=identity["pid"]
    proc=Path("/proc")/str(p)
    assert platform.machine()=="x86_64"
    libc=ctypes.CDLL(None,use_errno=True)
    process=checked(libc.syscall(434,p,0))
    assert ticks(p)==identity["startticks"] and proc.stat().st_uid==os.getuid()
    args=(proc/"cmdline").read_bytes().split(b"\0")
    assert args[1:]==[os.fsencode(root/"analysis/full_only_20261118.py"),b"worker",os.fsencode(root),
                       design["host"].encode(),b"CL_MoE",b""]
    assert read(run/"STARTED.json")["startticks"]==identity["startticks"]
    assert read(run/"config.json")["manifest_sha256"]==design["manifest_sha256"]
    directory=run/"checkpoints"
    metadata=directory/"stage_01.json"
    notify=checked(libc.inotify_init1(os.O_CLOEXEC))
    mask=0x100|0x80|0x08  # create, move into directory, close after write
    checked(libc.inotify_add_watch(notify,os.fsencode(run),mask))
    watching=False
    put(scope/"HANDOFF_WAITING.json",dict(pid=os.getpid(),startticks=ticks(os.getpid()),
                                        identity=identity,unix=time.time()))
    while True:
        if directory.is_dir() and not watching:
            checked(libc.inotify_add_watch(notify,os.fsencode(directory),mask))
            watching=True
        if metadata.is_file():
            try:
                snapshot=read(metadata)
                break
            except json.JSONDecodeError:
                pass
        ready,_,_=select.select([process,notify],[],[])
        if process in ready:
            raise RuntimeError("Legacy worker exited before publishing S1")
        os.read(notify,65536)
    assert snapshot["stage"]==1 and snapshot["path"]=="checkpoints/stage_01.pt"
    cp=run/snapshot["path"]
    assert cp.stat().st_size==snapshot["bytes"] and sha(cp)==snapshot["sha256"]
    validity=read(run/"validity/stage_01.json")
    assert validity["passed"] and validity["actual_parameters_changed"] and validity["updates"]==2163
    # This worker cannot load the corrected T02. Its immutable checkpoint is
    # the handoff; all later training must use a separately bound clean run.
    put(scope/"STAGE1_HANDOFF.json",dict(snapshot=snapshot,run=str(run),identity=identity,
                                        validity=validity,manifest=MANIFEST_SHA,unix=time.time()))
    checked(libc.syscall(424,process,signal.SIGTERM,None,0))
    assert select.select([process],[],[],60)[0]
    put(scope/"LEGACY_WORKER_ENDED.json",dict(identity=identity,unix=time.time(),stage1=sha(cp)))


if __name__=="__main__":
    main(Path(sys.argv[1]))
