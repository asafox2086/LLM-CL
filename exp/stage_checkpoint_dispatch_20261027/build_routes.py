"""Build a new dispatcher route; never edit the frozen legacy route or queue."""
from pathlib import Path
import argparse
import ast
import hashlib
import json


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    here = root / 'exp/stage_checkpoint_dispatch_20261027'
    legacy = root / 'exp/two_backbone_pct10_20261025'
    native = root / 'exp/two_backbone_pct10_native_checkpoint_20261027'
    forever = root / 'exp/two_backbone_pct10_forever_checkpoint_20261027'
    report = root / 'summary/stage_checkpoint_dispatch_20261027'
    assert not (report / 'READY.json').exists()
    originals = {}
    for name in ('dispatch_dynamic.py', 'forever_gate_supervisor.py'):
        raw = (legacy / name).read_text()
        originals[name] = sha(legacy / name)
        raw = raw.replace('two_backbone_pct10_forever_dynamic_20261025', forever.name)
        raw = raw.replace('summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_DISPATCH_READY.json',
                          'summary/stage_checkpoint_dispatch_20261027/READY.json')
        if name == 'dispatch_dynamic.py':
            raw = raw.replace("READY_PATH = REPORT / 'FOREVER_DYNAMIC_DISPATCH_READY.json'",
                              "READY_PATH = ROOT / 'summary/stage_checkpoint_dispatch_20261027/READY.json'")
            raw = raw.replace("ROOT / 'exp/two_backbone_pct10_20261025/dispatch.py'",
                              "ROOT / 'exp/stage_checkpoint_dispatch_20261027/dispatch.py'")
        compile(ast.parse(raw), name, 'exec')
        with (here / name).open('x', encoding='utf8', newline='\n') as stream:
            stream.write(raw)
    # The old preflight supervisor verifies both generated source bytes and
    # immutable author helper bytes. Preserve those checks in the new build.
    build = json.loads((forever / 'BUILD_RECEIPT.json').read_bytes())
    assert 'author_helpers' in build and 'destination_py_sha256' in build
    ready = dict(original_dispatch_sha256=sha(here / 'dispatch.py'),
        dynamic_sources={p.name: sha(p) for p in forever.glob('*.py')},
        native_sources={p.name: sha(p) for p in native.glob('*.py')},
        supervisor_sources={name: sha(here / name) for name in ('dispatch_dynamic.py', 'forever_gate_supervisor.py')},
        dynamic_policy_sha256=sha(forever / 'execution_policy.json'),
        native_policy_sha256=sha(native / 'execution_policy.json'),
        build_receipt_sha256=sha(forever / 'BUILD_RECEIPT.json'),
        entry_sha256=sha(here / 'entry.py'), original_route_sources=originals,
        legacy_science_sources_modified=False, models_started=0)
    report.mkdir(parents=True, exist_ok=True)
    with (report / 'READY.json').open('x') as stream:
        json.dump(ready, stream, indent=2)
    print(json.dumps({'ready': str(report / 'READY.json'), 'sha256': sha(report / 'READY.json')}))


if __name__ == '__main__':
    main()
