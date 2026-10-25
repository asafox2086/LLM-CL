"""Keep every pct10 final model on NEW; extend the existing CPU-only archive.

No training, evaluation, model deserialization, GPU call or OLD deletion.
Final weights are hard-linked from SHA-checked durable files, not duplicated.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import time

ROOT = Path('/data1/liyapeng/program/LLMCL')
BASE_SHA = '74f1fd90dd2f3de130a93aa6a4dd6e6b0b1ccead59758808fdec3e7aec991ac6'
MANIFEST_SHA = '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
METHODS = ('SeqFT_LoRA', 'FOREVER', 'CL_MoE', 'SMoLoRA', 'PASs_MoE', 'PaRSP', 'OSFT', 'SpaRTA')
BACKBONES = ('onevision', 'qwen')


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def load_base():
    path = ROOT / 'analysis/pct10_durable_archive_20261025.py'
    assert sha(path) == BASE_SHA
    spec = importlib.util.spec_from_file_location('pinned_pct10_archive', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    old = repr(base.FOLDERS)
    assert base.INVENTORY.count(old) == 1
    base.FOLDERS = [*base.FOLDERS, 'exp/two_backbone_pct10_forever_dynamic_20261025']
    base.INVENTORY = base.INVENTORY.replace(old, repr(base.FOLDERS), 1)
    return base


def pin(base, run, source, meta, origin, verified_copy):
    config = read(run / 'config.json')
    assert config['condition'] == 'pct10' and config['manifest_sha256'] == MANIFEST_SHA
    backbone, method = config['backbone'], config['method']
    assert backbone in BACKBONES and method in METHODS and meta['stage'] == 19
    assert not config['preflight_only'] and meta['manifest_sha256'] == MANIFEST_SHA
    assert source.is_file() and source.stat().st_size == meta['bytes']
    directory = base.DEST / 'final_models' / backbone / method / run.name
    receipt = directory / 'FINAL_MODEL_RETAINED.json'
    if receipt.exists():
        saved = read(receipt)
        target = base.DEST / saved['retained_checkpoint']
        assert saved['checkpoint_sha256'] == meta['sha256']
        assert target.is_file() and target.stat().st_size == meta['bytes']
        return saved
    if not verified_copy:
        assert sha(source) == meta['sha256']  # NEW-local final, once before pin.
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / source.name
    if target.exists():
        assert os.path.samefile(source, target), 'Refuse a different pre-existing final file'
    else:
        os.link(source, target)
        base.sync_directory(directory)
    assert os.path.samefile(source, target) and target.stat().st_size == meta['bytes']
    value = dict(unix=time.time(), condition='pct10', backbone=backbone, method=method,
        manifest_sha256=MANIFEST_SHA, completed_checkpoint_stage=19,
        source_host=origin, source_run=str(run), source_checkpoint=str(source),
        retained_checkpoint=str(target.relative_to(base.DEST)),
        checkpoint_sha256=meta['sha256'], checkpoint_bytes=meta['bytes'],
        config_sha256=sha(run / 'config.json'), metadata_directory=str(run),
        storage_mode='hard link to SHA-checked complete checkpoint; no second weight payload',
        SHA_basis='existing archive transfer SHA' if verified_copy else 'first NEW-local pin stream SHA',
        contains_original_saved_model_graph=True, only_adapter_or_scalar=False,
        model_quality_verification_separate=True, source_retention_unchanged=True)
    base.write(receipt, value)
    return value


def cycle(base):
    base.cycle()  # Existing transfer SHA/fsync/space handling; OLD untouched.
    for folder in base.FOLDERS:
        archived = base.DEST / folder / 'runs'
        for slot in archived.glob('*/pct10/*/training_*/durable_latest.json'):
            meta = read(slot)
            if meta['stage'] == 19:
                run = slot.parent
                source = (run / meta['path']).resolve()
                assert source.is_relative_to(run / 'checkpoints')
                pin(base, run, source, meta, 'old2080', True)
        # NEW-native BF16 runs already live on durable disk. Retain their final
        # graph too, without routing weights through OLD or changing training.
        for run in (ROOT / folder / 'runs').glob('*/pct10/*/training_*'):
            pointer = run / 'checkpoints' / ('committed_latest.json' if 'forever' in folder else 'latest.json')
            if not pointer.exists():
                continue
            meta = read(pointer)
            if meta['stage'] != 19:
                continue
            record = read(run / 'stages/stage_19.json')
            assert record['complete'] and record['weights_saved'] and record['stage'] == 19
            source = (run / meta['path']).resolve()
            assert source.is_relative_to(run / 'checkpoints')
            pin(base, run, source, meta, 'new3090', False)
    records = [read(p) for p in sorted((base.DEST / 'final_models').glob('*/*/*/FINAL_MODEL_RETAINED.json'))]
    pairs = [dict(backbone=b, method=m,
        state='FINAL_MODEL_RETAINED' if any(x['backbone'] == b and x['method'] == m for x in records) else 'PENDING_FINAL_MODEL',
        models=[x for x in records if x['backbone'] == b and x['method'] == m])
        for b in BACKBONES for m in METHODS]
    base.write(base.DEST / 'FINAL_MODEL_INDEX.json', dict(unix=time.time(),
        source_sha256=sha(Path(__file__)), base_collector_sha256=BASE_SHA,
        manifest_sha256=MANIFEST_SHA, planned_pairs=16, retained_pairs=sum(bool(x['models']) for x in pairs),
        pairs=pairs, source_checkpoints_never_deleted=True, final_pins_never_deleted=True,
        saved_model_not_just_parameter_count=True, quality_scores_not_inferred=True,
        GPU_calls=0, models_loaded=0, new_scores=0))


if __name__ == '__main__':
    assert os.getuid() == 1004
    base = load_base()
    base.DEST.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            cycle(base)
        except Exception as exc:
            base.write(base.DEST / 'FINAL_RETENTION_ERROR.json', dict(unix=time.time(), error=repr(exc)))
        time.sleep(120)
