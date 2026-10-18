"""Recover a completed non-feature factorial case without retrying its training.

V2 incorrectly required a MetricExperiment receipt from a RepairedExperiment
worker. Preserve the failed wrapper and all artifacts. Only that exact terminal
failure is resolved after validating the sealed experiment; other failures stop.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'summary/onevision_diagnostics_20261016/feature_identity_repair_20261018'
REPORT = STUDY/'queue_v4'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def resolved_attention():
    auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    for name, expected in auth['immutable_input_sha256'].items():
        assert sha(ROOT/name) == expected, name
    allowed = STUDY/'worker_427417_needs_attention.json'
    actual = set(STUDY.glob('worker_*_needs_attention.json'))
    assert actual == {allowed}, 'Unknown worker failure: preserve and investigate'
    failure = json.loads(allowed.read_text())
    assert failure['pid'] == 427417
    assert failure['traceback'].endswith('AssertionError: Worker did not run a repaired experiment\n')
    base = ROOT/'summary/onevision_diagnostics_20261016/cold_data_factorial_20261018'
    proof = json.loads((base/'case_00_complete.json').read_text())
    plan = json.loads((base/'CPU_PLAN_CHECK.json').read_text())['spec']
    assert proof['complete'] and proof['valid_training'] and proof['spec'] == plan
    assert proof['case'] == plan['cases'][0] and proof['updates'] == 4
    assert not (base/'case_00_needs_attention.json').exists()
    run = ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_cold_data_factorial_20261018/case_00').resolve())
    for name, expected in proof['artifact_sha256'].items():
        p = run/name
        assert p.resolve().is_relative_to(run.resolve()) and sha(p) == expected, name
    records = json.loads((run/'training.json').read_text())['records']
    assert [r['update'] for r in records] == [1, 2, 3, 4]
    assert all(r['LR'] == 1e-4 and math.isfinite(r['loss']) and
               math.isfinite(r['gradient_L2_preclip']) and r['gradient_L2_preclip'] > 0 for r in records)
    for label in ('initial', 'endpoint_1', 'endpoint_4'):
        assert json.loads((run/'evaluations'/label/'durable_complete.json').read_text())['complete']
    # This class has the official image-token renderer gate; it does not call
    # MetricExperiment.feature, so demanding that feature's receipt is wrong.
    tree = ast.parse((ROOT/'analysis/onevision_cold_data_factorial.py').read_text())
    init = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
            isinstance(n.func, ast.Name) and n.func.id == 'RepairedExperiment']
    assert len(init) == 1
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'feature' for n in ast.walk(tree))
    return True


def compiled_queue():
    source = ROOT/'analysis/onevision_identity_repair_queue_v2.py'
    text = source.read_text()
    edits = [
        ("REPORT = STUDY/'queue_v2'", "REPORT = STUDY/'queue_v4'"),
        ("assert not list(STUDY.glob('worker_*_needs_attention.json')), 'Preserve repair failure'",
         "assert resolved_attention(), 'Preserve unresolved repair failure'"),
        ("if Path(command[0]).name in WRAPPED:",
         "if Path(command[0]).name in WRAPPED and kind != 'data_factorial':"),
        ("expected = [ENTRY, '--worker', before[0], '--', *before[1:]]",
         "expected = before if kind == 'data_factorial' else [ENTRY, '--worker', before[0], '--', *before[1:]]"),
        ("assert after == expected and log.parent == STUDY",
         "assert after == expected and (kind == 'data_factorial' or log.parent == STUDY)"),
    ]
    for before, after in edits:
        assert text.count(before) == 1
        text = text.replace(before, after)
    namespace = {'__name__': 'identity_queue_v4_body', '__file__': str(source),
                 'resolved_attention': resolved_attention}
    exec(compile(text, str(Path(__file__)), 'exec'), namespace)
    return namespace


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    assert resolved_attention()
    auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert sha(Path(__file__)) == auth['recovery_entrypoint_sha256']
    assert auth['no_model_or_data_changes'] and auth['all_original_jobs_retained']
    namespace = compiled_queue()
    namespace['main'](check)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps({'traceback': traceback.format_exc(),
            'no_blind_retry': True}, indent=2)+'\n')
        raise
