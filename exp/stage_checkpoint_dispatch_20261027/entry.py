"""Use the existing allocator and slot classes with new checkpoint-capable runners.

The active research slot and its sources stay unchanged. This file is also the
phase-child entry, so a child cannot fall back to the legacy runner by accident.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ('resource-root', 'resource-output', 'resource-python',
                 'resource-onevision-binding', 'resource-qwen-binding',
                 'conditional-root', 'conditional-output', 'conditional-python',
                 'position-admission', 'resource-adapter'):
        parser.add_argument('--' + name, type=Path)
    args, remaining = parser.parse_known_args()
    ready_path = Path(os.environ['LLMCL_ROOT']) / 'summary/stage_checkpoint_dispatch_20261027/READY.json'
    release = json.loads(ready_path.read_bytes())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == release['entry_sha256']
    native = HERE.parent / 'two_backbone_pct10_native_checkpoint_20261027'
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native.glob('*.py')} == release['native_sources']
    assert hashlib.sha256((native / 'execution_policy.json').read_bytes()).hexdigest() == release['native_policy_sha256']
    dynamic = load('stage_checkpoint_dynamic', HERE / 'dispatch_dynamic.py')
    module, ready = dynamic.prepare()
    original_phase = module.phase
    module.phase = lambda request: dynamic.forever_phase(module, original_phase, request)
    module.__file__ = str(Path(__file__).resolve())
    if '--phase-request' in remaining:
        sys.argv = [str(Path(__file__).resolve()), *remaining]
        module.main()
        return
    # NEW continues to respect its separate 1% ownership handoff. Do not deploy
    # this OLD entry on NEW until that handoff has been explicitly performed.
    assert '--machine' not in remaining or remaining[remaining.index('--machine') + 1] == 'old'
    assert all(getattr(args, x.replace('-', '_')) is not None for x in
               ('conditional-root', 'conditional-output', 'conditional-python',
                'position-admission', 'resource-adapter'))
    root = Path(os.environ['LLMCL_ROOT']).resolve()
    position_path = root / 'exp/finance_position_diagnostic_20261026/queue_adapter.py'
    conditional_path = args.conditional_root / 'exp/conditional_shared_budget_20261026/v1/queue_adapter.py'
    conditional = load('stage_checkpoint_existing_conditional', conditional_path)
    position = load('stage_checkpoint_existing_position', position_path)
    assert hashlib.sha256(position_path.read_bytes()).hexdigest() == conditional.POSITION_SHA256
    checked_root, admission = position.validate(args.position_admission)
    assert checked_root == root
    position_slot = position.Slot(args.position_admission, admission)
    conditional_slot = conditional.ConditionalSlot(position, root,
        args.conditional_root, args.conditional_output, args.conditional_python)
    resource = load('stage_checkpoint_existing_resource', args.resource_adapter)
    assert hashlib.sha256(conditional_path.read_bytes()).hexdigest() == resource.CONDITIONAL_SHA
    resource_slot = resource.ResourceSlot(conditional_slot, args, conditional_path)
    original_pipe = module.pipe
    def pipe(*values, **keywords):
        return resource_slot.wrap(conditional_slot.wrap(
            position_slot.wrap(original_pipe(*values, **keywords)), position_slot))
    module.pipe = pipe
    sys.argv = [str(Path(__file__).resolve()), *remaining]
    module.main()
    conditional_slot.drain_after_normal_return()
    resource_slot.drain()


if __name__ == '__main__':
    main()
