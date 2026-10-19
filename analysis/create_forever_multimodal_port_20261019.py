"""Create a separate auditable FOREVER draft, leaving active V2 files untouched."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/'exp/two_backbone_native_cl_20261017_v2'
TARGET = ROOT/'exp/two_backbone_forever_20261019_v1'
HELPERS = ['analysis/forever_multimodal_core_20261019.py',
           'analysis/forever_multimodal_replay_20261019.py']


def main():
    assert not TARGET.exists(), 'Never overwrite a prepared runner'
    original = {p.name: p.read_bytes() for p in ORIGINAL.glob('*.py')}
    policy_raw = (ORIGINAL/'execution_policy.json').read_bytes()
    before = {name: hashlib.sha256(raw).hexdigest() for name, raw in original.items()}
    TARGET.mkdir()
    for name, raw in original.items():
        (TARGET/name).write_bytes(raw)
    run = original['run.py'].decode()
    a = run.index('METHODS = ')
    b = run.index('\n\n', a)
    run = run[:a]+"METHODS = ['FOREVER']"+run[b:]
    run = run.replace("choices=['full', 'pct01']", "choices=['pct01']")
    needle = "    host_policy=policy['hosts'][host]\n"
    assert run.count(needle) == 1
    run = run.replace(needle, needle+"    assert not host.endswith('_gate') or args.preflight_only\n")
    # Existing shared primary optimizer/data/evaluation code is unchanged.
    run = run.replace("'primary_score': 'rougeL'", "'primary_score': 'token_f1'")
    needle = "    if resume_obj:\n        for key in ['method','condition','manifest_sha256','execution_source_sha256']:"
    assert run.count(needle) == 1
    replacement = needle.replace("'execution_source_sha256']", "'execution_source_sha256','external_helper_sha256']")
    run = run.replace(needle,
        "    config['external_helper_sha256'] = {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in "+repr(HELPERS)+"}\n"+replacement)
    needle = "        (snapshot/'execution_policy.json').write_bytes((HERE/'execution_policy.json').read_bytes())"
    assert run.count(needle) == 1
    run = run.replace(needle, needle+"\n        for f in config['external_helper_sha256']:\n            p=snapshot/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/f).read_bytes())")
    needle = "        gc.collect()\n        torch.cuda.empty_cache()\n        monitor.snapshot('training_after', model=model)"
    assert run.count(needle) == 1
    run = run.replace(needle, "        if args.method=='FOREVER':\n            method.preflight_replay()\n"+needle)
    (TARGET/'run.py').write_text(run)
    methods = original['methods.py'].decode()
    needle = 'def make_method(model,name,instructions,embedding=None):\n'
    assert methods.count(needle) == 1
    methods = methods.replace(needle, needle+"    if name=='FOREVER':\n        from forever_adapter import ForeverMethod\n        return ForeverMethod(model,name,instructions,embedding)\n")
    (TARGET/'methods.py').write_text(methods)
    (TARGET/'forever_adapter.py').write_bytes(
        (ROOT/'analysis/forever_multimodal_adapter_20261019.py').read_bytes())
    policy = json.loads(policy_raw)
    policy['experiment_id'] = TARGET.name
    policy['methods'] = ['FOREVER']
    policy['hosts']['old2080_qwen_gate'] = {
        **policy['hosts']['old2080'], 'backbone': 'qwen'}
    (TARGET/'execution_policy.json').write_text(json.dumps(policy,indent=2)+'\n')
    for p in TARGET.glob('*.py'):
        ast.parse(p.read_text())
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in ORIGINAL.glob('*.py')}
    assert policy_raw == (ORIGINAL/'execution_policy.json').read_bytes()
    evidence = {
        'status': 'ISOLATED_DRAFT_NOT_GPU_ADMITTED',
        'original_V2_source_sha256': before,
        'copied_common_files_byte_identical': sorted(set(original)-{'run.py','methods.py'}),
        'derived_files': ['run.py','methods.py'],
        'new_adapter': 'forever_adapter.py',
        'helper_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in HELPERS},
        'active_original_files_unchanged': True,
        'GPU_launch': False,
    }
    (TARGET/'COPY_PROVENANCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'prepared': str(TARGET), 'original_source_unchanged': True,
                      'GPU_admitted': False}))


if __name__ == '__main__':
    main()
