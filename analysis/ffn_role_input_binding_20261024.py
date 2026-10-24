"""Once after VERIFIED, compare recorded prepared-input digests; no QA scoring."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root):
    tick = time.monotonic()
    report = root / 'summary/final_weight_reduction_20261024/ffn_role_retention_v1'
    work = root / 'exp/ffn_role_retention_20261024/v1/private'
    target = report / 'INPUT_BINDING.json'
    assert not target.exists(), 'Input comparison already consumed; do not repeat'
    verified = json.loads((report / 'VERIFIED.json').read_bytes())
    assert verified['complete'] and verified['TEST_QA'] == 0
    records = {}
    sources = {}
    for case in ('init', 'KeepGate', 'KeepUp'):
        complete_path = report / (case + '_COMPLETE.json')
        assert sha(complete_path) == verified['case_complete_sha256'][case]
        complete = json.loads(complete_path.read_bytes())
        for phase in (('parent',) if case == 'init' else ('initial', 'final')):
            key = case + '_' + phase
            path = work / case / (phase + '.jsonl')
            digest = sha(path)
            assert digest == complete['files'][case + '/' + phase + '.jsonl']
            assert digest == verified['rows'][key]['prediction_sha256']
            metadata = []
            for line in path.read_text().splitlines():
                row = json.loads(line)
                metadata.append(dict(id=row['id'], input_sha256=row['input_sha256']))
            assert [r['id'] for r in metadata] == [r['id'] for r in verified['rows'][key]['per_id']]
            assert len(metadata) == 100
            records[key] = metadata
            sources[key] = digest
    parent = records['init_parent']
    comparisons = {}
    for key, rows in records.items():
        assert [r['id'] for r in rows] == [r['id'] for r in parent]
        differences = [a['id'] for a, b in zip(rows, parent) if a['input_sha256'] != b['input_sha256']]
        comparisons[key] = dict(compared=100, different_ids=differences, identical=not differences)
    result = dict(complete=True, source_VERIFIED_sha256=sha(report / 'VERIFIED.json'),
                  consumer_sha256=sha(Path(__file__)), prediction_sha256=sources,
                  prepared_input_metadata=records, comparisons_to_parent=comparisons,
                  all_five_prepared_input_sequences_identical=all(r['identical'] for r in comparisons.values()),
                  scope='Existing prepared input tensors as recorded before generation; no proof of absence of unknown data duplicates or pretraining overlap.',
                  new_model_calls=0, new_QA_generation=0, score_calls=0, bootstrap_calls=0, TEST_QA=0,
                  seconds_before_write=time.monotonic() - tick)
    with target.open('x', encoding='utf8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    print(json.dumps(dict(INPUT_BINDING_sha256=sha(target),
                         all_five_prepared_input_sequences_identical=result['all_five_prepared_input_sequences_identical'],
                         comparisons_to_parent=comparisons)))


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
