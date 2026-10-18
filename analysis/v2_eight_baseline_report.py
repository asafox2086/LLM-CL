"""Scope-only wrapper over the frozen V2 collector; never changes training.

FV-guided Training/ALER were removed before formal training by the user.
FOREVER is a separately declared, not-yet-admitted comparison. Existing
completion proofs retain their original collector SHA and numerical values.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import v2_verified_baseline_report as original

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/two_backbone_pct01_20261017/completion_observer_eight_20261018'
EXCLUDED = {'FV_guided_Training', 'ALER'}
COLLECTOR_SHA = 'd7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a'
WRAPPER_SHA = original.sha(Path(__file__))
assert original.SOURCE_SHA == COLLECTOR_SHA
original.METHODS = [m for m in original.METHODS if m not in EXCLUDED]
original.LABELS = {m: original.LABELS[m] for m in original.METHODS}
original.REPORT = REPORT
rpc_original = original.rpc
write_original = original.write


def rpc(host, mode, relative=None):
    result = rpc_original(host, mode, relative)
    if mode == 'discover':
        result['candidates'] = [c for c in result['candidates'] if c['method'] in original.METHODS]
    return result


def write(path, value):
    if path.name in {'VERIFIED_INDEX.json', 'observer_status.json'}:
        value = dict(value, required_pairs=16, scope_wrapper_sha256=WRAPPER_SHA,
                     removed_before_formal_training=sorted(EXCLUDED),
                     pending_not_admitted=['FOREVER'], methods_in_scope=original.METHODS)
    write_original(path, value)


original.rpc = rpc
original.write = write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['check', 'once', 'watch'], required=True)
    args = parser.parse_args()
    assert len(original.METHODS) == 8 and EXCLUDED.isdisjoint(original.METHODS)
    assert 'CL_MoE' in original.METHODS and 'ER_LoRA' in original.METHODS
    if args.mode == 'check':
        print(json.dumps({'CPU_only': True, 'original_checks': original.checks(),
                          'required_pairs': 16, 'pending_not_admitted': ['FOREVER']}))
        return
    REPORT.mkdir(parents=True, exist_ok=True)
    assert not (REPORT/'needs_attention.json').exists(), 'Review failure before restarting'
    lock = (REPORT/'observer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        while True:
            assert original.sha(Path(__file__)) == WRAPPER_SHA, 'No hot edit'
            try:
                count = original.refresh()
            except (ConnectionError, subprocess.TimeoutExpired) as error:
                write(REPORT/'transport_wait.json', {'error': str(error), 'no_GPU_restart': True})
                if args.mode == 'once':
                    raise SystemExit(2)
            else:
                if args.mode == 'once' or count == 16:
                    break
            time.sleep(60)
    except Exception:
        write(REPORT/'needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc(),
                                            'no_training_restarted': True})
        raise


if __name__ == '__main__':
    main()
