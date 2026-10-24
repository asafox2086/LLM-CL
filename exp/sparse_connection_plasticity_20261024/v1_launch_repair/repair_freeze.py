"""One stdlib-only prelaunch parser repair; inherit, never repeat, the tiny gate."""
import ast
import copy
import datetime
import json
import os
import sys
import time
from pathlib import Path
from common import HERE, REPORT, ROOT, parse_inventory, process_identity, sha, training_ast, write

OLD_HERE = ROOT / 'exp/sparse_connection_plasticity_20261024/v1'
OLD_REPORT = ROOT / 'summary/final_weight_reduction_20261024/sparse_connection_plasticity_v1'


def dump(node):
    return ast.dump(node, include_attributes=False)


def compare_sources():
    old_ready = json.loads((OLD_REPORT / 'READY.json').read_text())
    result = {}
    for relative, digest in old_ready['sources'].items():
        old = ROOT / relative; new = HERE / old.name
        assert sha(old) == digest
        if old.name != 'common.py':
            assert old.read_bytes() == new.read_bytes(), old.name
            result[old.name] = 'byte_identical'
    before = ast.parse((OLD_HERE / 'common.py').read_text())
    after = ast.parse((HERE / 'common.py').read_text())
    old_funcs = {n.name: n for n in before.body if isinstance(n, ast.FunctionDef)}
    new_funcs = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
    assert set(new_funcs) == set(old_funcs) | {'parse_inventory'}
    for name in old_funcs.keys() - {'inventory'}:
        assert dump(old_funcs[name]) == dump(new_funcs[name]), name
    original = old_funcs['inventory']
    repaired = new_funcs['inventory']
    assert [dump(n) for n in original.body[:4]] == [dump(n) for n in repaired.body[:4]]
    assert len(repaired.body) == 5
    assert dump(repaired.body[-1]) == dump(ast.parse('return parse_inventory(gpu, apps)').body[0])
    parser = new_funcs['parse_inventory']
    actual = copy.deepcopy(parser.body[1:])
    loops = [n for n in actual if isinstance(n, ast.For)]
    assert len(loops) == 2
    expected_skip = ast.parse('if not line.strip():\n    continue').body[0]
    for loop in loops:
        assert dump(loop.body[0]) == dump(expected_skip)
        del loop.body[0]
    assert [dump(n) for n in actual] == [dump(n) for n in original.body[4:]]
    old_top = [n for n in before.body if not isinstance(n, ast.FunctionDef)]
    new_top = [n for n in after.body if not isinstance(n, ast.FunctionDef)]
    def normalize(nodes):
        out = copy.deepcopy(nodes)
        for node in out:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('REPORT', 'ASSETS') for t in node.targets):
                node.value = ast.Constant('operational_path_only')
        return [dump(n) for n in out]
    assert normalize(old_top) == normalize(new_top)
    result['common.py'] = 'only_REPORT_ASSETS_paths_and_blank_line_parser_extraction'
    return old_ready, result


def regression(observed):
    gpu = observed['GPU_query']; apps = observed['compute_query']
    assert gpu['returncode'] == apps['returncode'] == 0
    cards = parse_inventory(gpu['stdout'], apps['stdout'])
    assert len(cards) == 4 and all(not x['owners'] for x in cards.values())
    # These are parser fixtures, never a device availability observation.
    plain = '\n'.join(line for line in gpu['stdout'].splitlines() if line.strip())
    assert parse_inventory(plain, '') == cards
    assert parse_inventory(' \n\t\n' + plain + '\n \n', '\n\t\n') == cards
    rejects = []
    uuid = cards[0]['uuid']
    for name, g, a in (
        ('gpu_missing_column', '0, ' + uuid + ', 8', ''),
        ('gpu_extra_column', '0, ' + uuid + ', 8, 0, extra', ''),
        ('gpu_nonnumeric_index', 'bad, ' + uuid + ', 8, 0', ''),
        ('gpu_nonnumeric_memory', '0, ' + uuid + ', bad, 0', ''),
        ('gpu_nonnumeric_utilization', '0, ' + uuid + ', 8, bad', ''),
        ('owner_missing_column', plain, '123, ' + uuid),
        ('owner_extra_column', plain, '123, ' + uuid + ', 1, extra'),
        ('owner_nonnumeric_PID', plain, 'bad, ' + uuid + ', 1'),
    ):
        try:
            parse_inventory(g, a)
        except ValueError:
            rejects.append(name)
        else:
            raise AssertionError('Malformed nonempty row accepted: ' + name)
    owned = parse_inventory(plain, str(os.getpid()) + ', ' + uuid + ', 1')
    assert owned[0]['owners'] == [process_identity()]
    return dict(real_failure_text_parsed=True, historical_card_rows=4,
        empty_and_whitespace_rows_ignored=True, malformed_nonempty_rows_rejected=rejects,
        actual_current_process_identity_preserved=True,
        no_subprocess_replacement_or_monkeypatch=True,
        not_a_current_hardware_or_free_card_observation=True)


def main():
    assert 'torch' not in sys.modules
    assert not any((REPORT / n).exists() for n in ('READY.json', 'PARSER_REPAIR_CHECK.json', 'REPAIR_FREEZE_STARTED.json'))
    assert not (HERE / 'private').exists() and not (REPORT / 'CONTROLLER_OWNER.json').exists()
    review_path = REPORT / 'REPAIR_SOURCE_REVIEW.json'
    review = json.loads(review_path.read_text())
    assert review['decision'] == 'accept_prelaunch_parser_repair_freeze_only'
    for relative, digest in review['files_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    assert review['old_ready_sha256'] == sha(OLD_REPORT / 'READY.json')
    old_ready, inheritance = compare_sources()
    # Check previous small source/dependency/evidence bindings, not big model payloads.
    for category in ('sources', 'dependencies', 'evidence'):
        for relative, digest in old_ready[category].items():
            assert sha(ROOT / relative) == digest, relative
    gate = json.loads((OLD_REPORT / 'CPU_GATE.json').read_text())
    assert gate['passed'] and not gate['CUDA_initialized']
    assert training_ast()[1] == old_ready['training_ast_sha256']
    assert sha(REPORT / 'PROTOCOL.md') == sha(OLD_REPORT / 'PROTOCOL.md')
    observed_path = OLD_REPORT / 'ACTIVATION_PRECHECK_FAILURE_OBSERVED.json'
    observed = json.loads(observed_path.read_text())
    assert not observed['controller_owner_exists'] and not observed['model_private_exists'] and observed['started_phases'] == []
    # Actual current state must still have no old or new model attempt.
    for folder, public in ((OLD_HERE, OLD_REPORT), (HERE, REPORT)):
        assert not (folder / 'private').exists() and not (public / 'CONTROLLER_OWNER.json').exists()
        assert not list((public / 'phases').glob('*/STARTED.json'))
    write(REPORT / 'REPAIR_FREEZE_STARTED.json', dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **process_identity(), source_review_sha256=sha(review_path), no_model_or_Torch=True))
    begin = time.monotonic()
    check = regression(observed)
    check.update(parser_seconds=time.monotonic() - begin, source_inheritance=inheritance,
        original_gate_sha256=sha(OLD_REPORT / 'CPU_GATE.json'), original_gate_rerun=False,
        scientific_functions_and_other_ten_sources_unchanged=True,
        actual_model_attempts=0, original_QA=0, Torch_imported='torch' in sys.modules)
    assert not check['Torch_imported']
    write(REPORT / 'PARSER_REPAIR_CHECK.json', check)
    ready = copy.deepcopy(old_ready)
    ready['sources'] = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(HERE.glob('*.py'))}
    extras = [OLD_REPORT / n for n in ('READY.json', 'ADMISSION_REVIEW.json', 'ACTIVATION_PRECHECK_FAILURE_OBSERVED.json')]
    extras += [REPORT / n for n in ('PROTOCOL.md', 'REPAIR_SPEC.md', 'REPAIR_SOURCE_REVIEW.json', 'PARSER_REPAIR_CHECK.json')]
    ready['evidence'].update({p.relative_to(ROOT).as_posix(): sha(p) for p in extras})
    ready['prelaunch_repair'] = dict(original_READY_sha256=sha(OLD_REPORT / 'READY.json'),
        original_gate_inherited_not_rerun=True, scientific_protocol_unchanged=True,
        old_entry_must_never_activate=True, new_admission_required=True,
        no_scientific_or_model_budget_reset=True)
    assert {k: ready[k] for k in old_ready if k not in ('sources', 'evidence')} == {
        k: old_ready[k] for k in old_ready if k not in ('sources', 'evidence')}
    write(REPORT / 'READY.json', ready)
    print(json.dumps(dict(ready_sha256=sha(REPORT / 'READY.json'), repair_check_sha256=sha(REPORT / 'PARSER_REPAIR_CHECK.json'),
        no_activation=True, inherited_gate_not_rerun=True, model_attempts=0)))


if __name__ == '__main__':
    main()
