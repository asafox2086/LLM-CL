"""CPU negative tests for the frozen report; no active source changes."""
import json
from pathlib import Path

import onevision_full_confirm_evidence as report


def blocked(cap):
    try:
        report.audit(cap)
    except AssertionError:
        return
    raise AssertionError('Unproven confirmation was accepted')


def main():
    original_complete = report.validator.complete_full
    original_reader = report.checked_json
    calls = []
    try:
        report.validator.complete_full = lambda _: False
        def forbidden(path, sources):
            calls.append(str(path));raise RuntimeError('Read after missing full proof')
        report.checked_json = forbidden
        for cap in [1, 3]:
            blocked(cap)
        assert not calls
        report.validator.complete_full = lambda _: True
        def failed_state(path, sources):
            calls.append(str(path))
            if path.name == 'metric_lock.json':
                return original_reader(path, sources)
            assert path.name.endswith('_STATE_AUDIT.json')
            return {'passed': False, 'CPU_only': True}
        report.checked_json = failed_state
        for cap in [1, 3]:
            blocked(cap)
        assert len(calls) == 4 and all('image.jsonl' not in path for path in calls)
    finally:
        report.validator.complete_full = original_complete;report.checked_json = original_reader
    report.validator.write(report.REPORT/'CPU_GUARD_VALIDATION.json', {'passed': True, 'CPU_only': True,
         'negative_checks': 4, 'missing_full_proof_rejected_before_any_result_read': True,
         'failed_state_gate_rejected_before_TEST_prediction_read': True,
         'source_sha256': {str(p.relative_to(report.ROOT)): report.validator.sha(p)
                           for p in [Path(__file__), Path(report.__file__)]},
         'monkeypatch_scope': 'This short CPU test process only; active service/source unmodified',
         'actual_full21454_ingestion_still_pending': True})
    print(json.dumps({'passed': True, 'CPU_negative_checks': 4, 'GPU_used': False}))


if __name__ == '__main__':
    main()
