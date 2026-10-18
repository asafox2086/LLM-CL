"""Proof-triggered four-cell report using the new full-inference namespace."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
from pathlib import Path
import time
import traceback

import onevision_functional_router_queue as functional_queue

ROOT = functional_queue.ROOT
STUDY = ROOT/'summary/onevision_diagnostics_20261016/feature_identity_repair_20261018'
functional_queue.STUDY = STUDY/'functional_router'
# Import after binding the new namespace; all old report sources stay intact.
import onevision_cold_quality_report as original


def stamp():
    new = STUDY/'functional_router/full_quality_hard_complete.json'
    return original.stamp(), original.queue.sha(new) if new.exists() else None


def once():
    result = original.once()
    inspected = []
    for p in [STUDY/'functional_router/full_quality_hard_complete.json',
              original.BASE/'warm_start_20261018/warm_inherit_K3_v1_train_complete.json',
              original.BASE/'warm_start_20261018/warm_inherit_K3_v1_full_complete.json',
              original.REPORT/'pilot_complete.json', original.REPORT/'full_quality_hard_complete.json']:
        if not p.exists():
            continue
        proof = json.loads(p.read_text())
        run = ROOT/proof['run']
        receipt = run/'feature_identity_repair_execution.json'
        assert receipt.exists(), 'New execution must include the repair GPU gate'
        payload = json.loads(receipt.read_text())
        assert payload['GPU_gate']['passed'] and payload['old_partial_predictions_reused'] is False
        assert payload['repair_source_sha256'] == original.queue.sha(ROOT/'analysis/onevision_feature_identity_repair.py')
        assert proof['artifact_sha256']['feature_identity_repair_execution.json'] == original.queue.sha(receipt)
        inspected.append({'proof': str(p.relative_to(ROOT)), 'execution_receipt_sha256': original.queue.sha(receipt)})
    result['feature_identity_repair'] = {'metadata_only': True, 'old_failed_full_not_reused': True,
        'proofs_checked': inspected, 'reporter_sha256': original.queue.sha(Path(__file__))}
    original.queue.write(original.REPORT/'FOUR_CELL_EVIDENCE.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    try:
        last = None
        while True:
            current = stamp()
            if current != last:
                result = once()
                last = current
                print(json.dumps({'CPU_only': True, 'complete_full_cells': result['complete_full_cells'],
                                  'warm19_complete': result['parent_warm19_complete']}), flush=True)
            if not args.watch or result['complete_full_cells'] == 4:
                break
            time.sleep(60)
    except Exception:
        original.queue.write(STUDY/'reporter_needs_attention.json', {'traceback': traceback.format_exc()})
        raise
