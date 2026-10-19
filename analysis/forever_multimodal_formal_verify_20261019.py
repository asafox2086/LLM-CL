"""CPU-only formal FOREVER verifier; no publisher, scheduler or GPU launch.

Delegates original 19/74/21454/raw-F1/checkpoint validation to the unchanged
V2 verifier in this isolated process, then verifies all additional replay costs.
No incomplete/preflight run can be published through this entry point.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT/'exp/two_backbone_forever_20261019_v1'
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v1'
BASE_SHA = 'd7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a'
PROFILE_SHA = '99a01a7cb53b88deb4e1099e5f7618a5d9bbefcde06b66fdef24e87f4bd82d05'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def checked(path, hashes, expected=None):
    digest = sha(path)
    assert expected is None or digest == expected, str(path)
    hashes[str(path.relative_to(ROOT))] = digest
    return path


def verify(relative, host):
    own_sha = sha(Path(__file__))
    original = ROOT/'analysis/v2_verified_baseline_report.py'
    assert sha(original) == BASE_SHA
    spec = importlib.util.spec_from_file_location('isolated_forever_base_verifier', original)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    # Only this private module instance; never mutate a running publisher.
    base.FRAME = FRAME
    base.METHODS = ['FOREVER']
    result = base.verify(relative, host)
    run = (ROOT/relative).resolve()
    hashes = {}
    admission = json.loads(checked(REPORT/'FORMAL_ADMISSION.json', hashes).read_text())
    assert admission['both_backbone_GPU_and_reload_passed'] is True
    for name, expected in admission['training_source_sha256'].items():
        checked(ROOT/name, hashes, expected)
    config = json.loads(checked(run/'config.json', hashes).read_text())
    assert not config['preflight_only'] and config['primary_score'] == 'token_f1'
    assert set(config['external_helper_sha256']) == {
        'analysis/forever_multimodal_core_20261019.py',
        'analysis/forever_multimodal_replay_20261019.py'}
    for name, digest in config['external_helper_sha256'].items():
        assert name in ['analysis/forever_multimodal_core_20261019.py',
                        'analysis/forever_multimodal_replay_20261019.py']
        checked(ROOT/name, hashes, digest)
        checked(run/'runner_source'/name, hashes, digest)
    profile_path = ROOT/'summary/two_backbone_pct01_20261017/forever_preflight_20261018/EVIDENCE.json'
    profiles = json.loads(checked(profile_path, hashes, PROFILE_SHA).read_text())['profile']
    assert len(profiles) == 19
    total_QA = total_updates = positive = warmup_zero = 0
    stages = []
    for stage, profile in enumerate(profiles, 1):
        folder = run/'method_state'/f'stage_{stage:02d}'
        calibration = json.loads(checked(folder/'calibration.json', hashes).read_text())
        assert calibration['stage'] == stage and calibration['main_updates'] == profile['main_updates']
        assert calibration['dynamic_replay_triggers'] == 0
        assert calibration['calibration_steps_per_day'] == 24
        assert math.isfinite(calibration['tau']) and calibration['tau'] > 0
        memory = calibration['memory_QA']
        if stage == 1:
            assert memory == 0 and not calibration['final_replay_required']
            skipped = json.loads(checked(folder/'actual_author_replay/REPLAY_SKIPPED.json', hashes).read_text())
            assert skipped['reason'] == 'no historical TRAIN'
            continue
        selection = json.loads(checked(folder/'memory_selection.json', hashes).read_text())
        ids = selection['selected_ids']
        assert memory == len(ids) == len(set(ids)) == profile['author_memory_QA']
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == profile['selected_TRAIN_IDs_sha256']
        assert selection['selected_ids_sha256'] == profile['selected_TRAIN_IDs_sha256']
        replay = folder/'actual_author_replay'
        requested = json.loads(checked(replay/'REQUESTED_ARGUMENTS.json', hashes).read_text())
        assert not requested['CPU_control_not_formal']
        assert requested['num_train_epochs'] == 2 and requested['warmup_steps'] == 5
        assert requested['gradient_accumulation_steps'] == 16
        assert requested['per_device_train_batch_size'] == 1
        assert requested['learning_rate'] == 1e-4 and requested['lr_scheduler_type'] == 'linear'
        assert requested['beta'] == calibration['beta'] and .5 <= requested['beta'] <= 3.
        scale = json.loads(checked(replay/'SHARED_FP16_INITIAL_SCALE.json', hashes).read_text())
        assert scale['actual_before_first_backward'] == 1.
        completed = json.loads(checked(replay/'REPLAY_COMPLETE.json', hashes).read_text())
        assert completed['status'] == 'ACTUAL_REPLAY_COMPLETE'
        assert not completed['CPU_control_not_formal'] and completed['model_time_unchanged_by_replay']
        records = [json.loads(line) for line in checked(replay/'trace/updates.jsonl', hashes).read_text().splitlines()]
        assert len(records) == completed['actual_optimizer_updates'] > 1
        exposures, positive_here, zero_here, previous = [], 0, 0, None
        for index, row in enumerate(records, 1):
            assert row['step'] == index and math.isfinite(row['raw_gradient_l2']) and row['raw_gradient_l2'] > 0
            if previous is not None:
                assert row['parameter_before_sha256'] == previous
            before, after = row['parameter_before_sha256'], row['parameter_after_sha256']
            has_lr = any(x > 0 for x in row['learning_rate_used'])
            assert all(math.isfinite(x) and x >= 0 for x in row['learning_rate_used'])
            if has_lr:
                assert before != after and row['parameter_delta_l2'] > 0
                positive_here += 1
            else:
                assert index == 1 and before == after and row['parameter_delta_l2'] == 0
                zero_here += 1
            assert row['author_zero_warmup_step'] == (not has_lr)
            previous = after
            assert Path(row['raw_gradient_file']).name == row['raw_gradient_file']
            checked(replay/'trace'/row['raw_gradient_file'], hashes, row['raw_gradient_sha256'])
            for micro in row['micro_losses']:
                assert micro['TRAIN_id'] in set(ids)
                assert all(math.isfinite(micro[key]) for key in
                           ['base_loss','unscaled_regularizer','regularizer_coefficient','total_loss'])
                assert micro['unscaled_regularizer'] >= 0
                assert abs(micro['regularizer_coefficient']-.001*calibration['beta']) < 1e-15
                exposures.append(micro['TRAIN_id'])
            assert row['cumulative_QA'] == len(exposures)
        assert len(exposures) == completed['actual_QA_exposures']
        assert positive_here == completed['positive_LR_updates'] > 0
        assert zero_here == completed['author_zero_warmup_updates'] == 1
        resource = json.loads(checked(folder/'replay_resources/memory_summary.json', hashes).read_text())
        assert resource['before']['pid'] == resource['after']['pid']
        total_QA += len(exposures)
        total_updates += len(records)
        positive += positive_here
        warmup_zero += zero_here
        stages.append({'stage':stage,'memory_QA':memory,'actual_QA':len(exposures),
                       'actual_updates':len(records),'positive_LR':positive_here,
                       'author_zero_warmup':zero_here})
    assert len(stages) == 18
    assert sha(original) == BASE_SHA and sha(Path(__file__)) == own_sha
    return {'status':'FOREVER_FORMAL_COMPLETE_VERIFIED','shared_V2_proof':result,
            'extension_source_and_artifact_sha256':hashes,
            'wrapper_sha256':own_sha,'extra_replay_QA_exposures':total_QA,
            'extra_replay_optimizer_updates':total_updates,
            'extra_replay_positive_LR_updates':positive,
            'author_replay_zero_warmup_updates':warmup_zero,'replay_stages':stages,
            'formal_publisher_not_modified':True,'GPU_or_training_launched':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--host', choices=['old2080','new3090'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    proof = verify(args.run,args.host)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(proof,indent=2)+'\n')
