"""Pure metadata/view/group-loss helpers; no Torch or model imports."""
import ast
import hashlib
import json
from pathlib import Path
import re

DECISION_SHA = 'f2088a3fbed06d1dd2e6a328ece573fb1b74746c98aaaf1dfc60d4b764ad9cda'
PACK_SHA = '698f6e898d296ea7f6fedb4089cefbc4989125c19fa2acb20125470f88468eb0'
SCOPE_SHA = 'f79c334d05abd7549a30e222d4b320d640c2daba9ffde19387e3c9e6f92b21b8'
POSITION_SHA = '6dc74f36d072029f6a0b054c4e671181d78fd269bac9c381407afd6e3a5b8bd1'
ADAPTATION_SHA = '251a2d490f9ce29069de141686a16da549c28de82f7c1ce58fab2df8da31252a'
INHERITED_WORKER_SHA = '912d5e08edb2642268a483f98f01da7440f7fed730e7b2e91d32609127e190c8'
LABEL_MAP = dict(zip('ABCD', 'BCDA'))


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def bound(ref, ledger, expected=None):
    path = Path(ref['path']); assert path.is_absolute()
    raw = path.read_bytes(); actual = digest(raw)
    assert actual == ref['sha256'] and (expected is None or actual == expected), str(path)
    ledger[str(path)] = actual
    return json.loads(raw)


def bind_sources(root, source_dir, binding):
    assert binding['schema'] == 'paired-option-supervision-v1'
    assert set(binding['sources']) == {'worker.py', 'helpers.py'}
    for name, expected in binding['sources'].items(): assert sha(source_dir/name) == expected
    limits = binding['limits']
    assert set(limits) == {'train', 'eval'} and all(type(x) is int and x > 0 for x in limits.values())
    port = 'exp/two_backbone_native_cl_20261017_v2'
    required = {port+'/'+p.name for p in (root/port).glob('*.py')}
    required.update({
        'exp/conditional_shared_budget_20261026/v1/worker.py',
        'exp/conditional_shared_budget_20261026/v1/adaptation.py',
        'exp/finance_position_diagnostic_20261026/worker.py',
        'exp/osft_increment_screen_20261019/v1/study.py',
        'exp/position_coordinate_control_20261022/v1/core.py',
        'exp/ffn_static_delete_20261025/v1/core.py',
        'exp/paired_image_support_20261024/v1/common.py',
        *('exp/plain_r8_gpu_control_20261025/v1/'+x for x in
          ('control_adaptation.py', 'control_deployment.py', 'common.py', 'readouts.py'))})
    assert required <= set(binding['inherited_sources'])
    for name, expected in binding['inherited_sources'].items():
        path = Path(name); assert not path.is_absolute() and '..' not in path.parts
        assert path.suffix == '.py' and sha(root/path) == expected, name
    assert sha(root/'exp/conditional_shared_budget_20261026/v1/worker.py') == INHERITED_WORKER_SHA
    assert sha(root/'exp/conditional_shared_budget_20261026/v1/adaptation.py') == ADAPTATION_SHA
    assert sha(root/'exp/finance_position_diagnostic_20261026/worker.py') == POSITION_SHA
    return dict(limits)


def cyclic_operator(path):
    """Compile the exact sealed pure functions, without importing its worker."""
    assert sha(path) == POSITION_SHA
    names = {'digest', 'ending', 'cyclic_context'}
    functions = [n for n in ast.parse(Path(path).read_bytes()).body
                 if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in functions} == names
    namespace = dict(hashlib=hashlib, re=re, ORDER=(3, 0, 1, 2))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['cyclic_context']


def cyclic_row(row, transform):
    answer = row['answer']; answers = row['answers']
    assert isinstance(answer, str) and answer in LABEL_MAP
    assert isinstance(answers, list) and answers and set(answers) == {answer}
    context, proof = transform(row['context'])
    result = dict(row, context=context, answer=LABEL_MAP[answer], answers=[LABEL_MAP[x] for x in answers])
    assert result['question'] == row['question'] and result['images'] == row['images']
    assert result['metadata'] == row['metadata'] and result['id'] == row['id']
    return result, dict(proof, original_answer_sha256=digest(answer.encode()),
        cyclic_answer_sha256=digest(result['answer'].encode()), gold_mapping=LABEL_MAP)


def eligibility(root, manifest, fit, dev, binding, ledger):
    decision = bound(binding['decision'], ledger, DECISION_SHA)
    scope = bound(binding['scope'], ledger, SCOPE_SHA)
    receipt = bound(binding['inputs_receipt'], ledger,
                    decision['evidence_sha256']['OPTION_ELIGIBILITY_INPUTS.json'])
    pack = bound(binding['masked_pack'], ledger, PACK_SHA)
    assert decision['counts'] == {'accepted': 45, 'excluded_fixed': 7, 'uncertain': 1}
    ids = decision['accepted_id_hashes']; assert len(ids) == len(set(ids)) == 45
    assert digest(canonical(ids)) == decision['accepted_ids_sha256']
    assert decision['fixed_old_label_to_new_label'] == LABEL_MAP
    assert scope['decision_sha256'] == DECISION_SHA and scope['eligible_total'] == 45
    assert scope['split_sha256'] == 'd8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
    assert (scope['eligible_FIT'], scope['eligible_DEV'], scope['unmatched']) == (33, 12, 0)
    assert receipt['private_pack_sha256'] == PACK_SHA
    assert pack['original_gold_and_predictions_absent'] and len(pack['records']) == 53
    assert pack['intended_old_label_to_new_label'] == LABEL_MAP
    masked = {x['id_sha256']: x for x in pack['records']}; assert len(masked) == 53
    task = manifest['tasks'][15]
    assert task['task'] == 16 and task['domain'] == 'finance' and task['max_new_tokens'] == 512
    entry = task['files']['train']; path = root/entry['path']
    assert entry['sha256'] == receipt['train_file_sha256'] and entry['path'] == receipt['train_file']
    raw = path.read_bytes(); assert digest(raw) == entry['sha256']
    lines = raw.splitlines()
    raw_rows = [json.loads(line) for line in lines]
    line_hashes = {json.loads(line)['id']: digest(line) for line in lines}
    assert len(raw_rows) == len(line_hashes) == entry['count']
    raw_by_id = {r['id']: r for r in raw_rows}
    transform = cyclic_operator(root/'exp/finance_position_diagnostic_20261026/worker.py')
    fit_ids = {r['id'] for rows in fit.values() for r in rows}; dev_ids = {r['id'] for rows in dev.values() for r in rows}
    assert not fit_ids.intersection(dev_ids)
    selected, changes, proofs = {'fit': {}, 'dev': {}}, {}, {}
    for row in fit[16]+dev[16]:
        key = digest(row['id'].encode())
        if key not in ids: continue  # Fixed, independently sealed eligibility; never gold/F1 selection.
        item = masked[key]
        assert canonical(row) == canonical(raw_by_id[row['id']])
        assert item['source_row_sha256'] == line_hashes[row['id']] and item['original_screen_candidate']
        assert item['question'] == row['question'] and item['context'] == row['context']
        assert item['image_identifiers_sha256'] == digest(json.dumps(row['images']).encode())
        assert row['dataset'] == 'FinMME' and row['metadata']['question_type'] == 'single_choice'
        assert [x['label'] for x in item['options']] == list('ABCD')
        changed, proof = cyclic_row(row, transform)
        role = 'fit' if row['id'] in fit_ids else 'dev'
        selected[role][row['id']] = key; changes[row['id']] = changed
        proofs[row['id']] = dict(id_sha256=key, source_row_sha256=line_hashes[row['id']], role=role, transform=proof)
    assert len(selected['fit']) == 33 and len(selected['dev']) == 12
    assert set(selected['fit'].values()) | set(selected['dev'].values()) == set(ids)
    assert set(selected['fit']).isdisjoint(selected['dev'])
    return selected, changes, proofs


def group_mean(forward, views):
    """Call each real forward exactly once; QA mean first, minibatch mean later."""
    assert len(views) in (1, 2)
    outputs = [forward(view) for view in views]
    loss = outputs[0][0] if len(outputs) == 1 else (outputs[0][0]+outputs[1][0])*.5
    return loss, sum(x[1] for x in outputs)
