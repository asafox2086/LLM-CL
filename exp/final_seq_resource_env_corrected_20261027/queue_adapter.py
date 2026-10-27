"""Append the unchanged resource measurement after the current type-aux slot.

No new allocator, worker, retry or resource definition. Preserve the old failed
output. New output and .vision-env are explicit; all original type/paired and
downstream arguments are forwarded unchanged. This file never runs a model.
"""
import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

TYPE_SHA = '9dc9e3baef0eee4d81f9a7bdc6a4a2618cf8326c2bfc6a949619f1620b81dae7'
RESOURCE_SHA = '6a833c438c335109cf000a64f7ef4f004887625a1efaf3aa54e712c79f3f4a2b'
WORKER_SHA = 'aedc7e7b97d4f5be22a98d022c758d3c3c285dbc805e087b2a952c09de302400'
CONDITIONAL_SHA = '1eaea047c6b6399b922387a80fc47b00a90728c021e50e006d11ee6de1f7910c'
FAILURE_SHA = '13533134b412f45723277c1c39284cc9bfa25101d7c86cd92199048b2eb79415'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def option(argv, name):
    assert argv.count(name) == 1, name
    value = Path(argv[argv.index(name)+1]); assert value.is_absolute()
    return value


def load(name, path, expected):
    assert sha(path) == expected, str(path)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def corrected_class(original, outer_path):
    """Keep original init/claim/supervisor/identity/release/drain contracts."""
    class CorrectedResourceSlot(original):
        def __init__(self, parent, args, conditional_path):
            super().__init__(parent, args, conditional_path)
            # The unchanged supervisor checks every descriptor and copies this
            # extra provenance into request, STARTED and terminal receipts.
            self.sources['environment_outer_wrapper'] = dict(path=str(outer_path), sha256=sha(outer_path))

        def wrap(self, pipeline):
            # The older failed-closed measurement already set the original
            # _final_resource_wrapped flag. Do not clear it or replace its hook.
            if getattr(pipeline, '_env_corrected_final_resource_wrapped', False): return pipeline
            original_idle = pipeline.idle  # TypeAuxSlot.wrap has already run.
            def idle(gpu, minimum):
                if self.held_gpu() == gpu: return False
                available = original_idle(gpu, minimum)
                if (available and not self.path.exists() and self.parent.path.exists()
                        and self.parent.record().get('released') is True):
                    self.launch(gpu)
                    return False
                return available
            pipeline.idle = idle
            pipeline._env_corrected_final_resource_wrapped = True
            self.pipelines.append(pipeline)
            return pipeline
    return CorrectedResourceSlot


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--env-resource-root', type=Path,
        default=Path('/data2/liyapeng_grp/program/LLMCL'))
    parser.add_argument('--env-resource-output', type=Path)
    parser.add_argument('--env-resource-python', type=Path)
    parser.add_argument('--env-resource-source', type=Path)
    parser.add_argument('--env-resource-onevision-binding', type=Path)
    parser.add_argument('--env-resource-qwen-binding', type=Path)
    args, remaining = parser.parse_known_args()
    assert '--phase-request' not in remaining and '--resource-request' not in remaining
    root = args.env_resource_root; assert root.is_absolute() and root.is_dir()
    assert Path(os.environ['LLMCL_ROOT']).is_absolute()  # The allocator's ROOT stays untouched.
    old = root/'.deployment/final_seq_resource_metadata_20261026'
    output = args.env_resource_output or root/'.deployment/final_seq_resource_env_corrected_20261027/runs'
    python = args.env_resource_python or root/'.vision-env/bin/python'
    source = args.env_resource_source or old/'exp/final_seq_resource_20261026/queue_adapter.py'
    assert output == root/'.deployment/final_seq_resource_env_corrected_20261027/runs'
    assert python == root/'.vision-env/bin/python' and python.is_file()
    assert source.is_absolute() and sha(source.parent/'worker.py') == WORKER_SHA
    resource = load('preserved_final_resource_measurement', source, RESOURCE_SHA)
    old_slot = resource.read(old/'runs.slot.json')
    assert old_slot['status'] == 'failed_closed' and old_slot['released'] is True
    assert old_slot['terminal_sha256'] == FAILURE_SHA and sha(old_slot['terminal_path']) == FAILURE_SHA
    assert str(output) != old_slot['output']  # No old output/claim/slot is rewritten.
    resource_args = SimpleNamespace(resource_root=root, resource_output=output, resource_python=python,
        resource_onevision_binding=args.env_resource_onevision_binding or old/'onevision_binding.json',
        resource_qwen_binding=args.env_resource_qwen_binding or old/'qwen_binding.json')
    for name in ('onevision','qwen'):
        expected = old_slot['bindings'][name]
        actual = getattr(resource_args,'resource_'+name+'_binding')
        assert str(actual) == expected['path'] and sha(actual) == expected['sha256']
    conditional_path = option(remaining,'--conditional-root')/'exp/conditional_shared_budget_20261026/v1/queue_adapter.py'
    assert sha(conditional_path) == CONDITIONAL_SHA
    type_path = option(remaining,'--type-aux-root')/'exp/type_auxiliary_supervision_20261027/v1/queue_adapter.py'
    type_queue = load('current_type_aux_capacity_owner', type_path, TYPE_SHA)
    appended = corrected_class(resource.ResourceSlot, Path(__file__).resolve())
    original = type_queue.TypeAuxSlot
    class AppendedTypeAuxSlot(original):
        def __init__(self, *values, **keywords):
            super().__init__(*values, **keywords)
            self.corrected_resource = appended(self, resource_args, conditional_path)
        def wrap(self, pipeline, prior_slot):
            return self.corrected_resource.wrap(super().wrap(pipeline, prior_slot))
        def drain_after_normal_return(self):
            super().drain_after_normal_return()
            self.corrected_resource.drain()  # Original shared dispatch_old.lock/idle loop.
    type_queue.TypeAuxSlot = AppendedTypeAuxSlot
    sys.argv = [str(type_path), *remaining]
    type_queue.main()


if __name__ == '__main__': main()
