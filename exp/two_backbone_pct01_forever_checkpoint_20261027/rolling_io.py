"""Stage-atomic retained checkpoints and per-process resource scopes.

This is runner I/O, not a change to FOREVER, model math or the data protocol.
Only a checkpoint with a separate durable commit pointer may be resumed.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from checkpoints import load as original_load
from resource_monitor import ResourceMonitor as OriginalMonitor


def inside(run, path):
    run, path = Path(run).resolve(), Path(path).resolve()
    assert path.is_relative_to(run) and path != run
    return path


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def rebuild_events(run, completed_stage, manifest):
    run = Path(run)
    output = inside(run, run/'logs/events.jsonl')
    temporary = output.with_suffix('.jsonl.partial')
    with temporary.open('w', encoding='utf-8') as stream:
        for stage in range(1, completed_stage+1):
            source = inside(run, run/'logs'/f'stage_{stage:02d}.events.jsonl')
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            expected = math.ceil(manifest['tasks'][stage-1]['files']['train']['count']/16)
            assert len(rows) == expected
            assert [row['step'] for row in rows] == list(range(1, expected+1))
            assert all(row['event'] == 'train_update' and row['stage'] == stage for row in rows)
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(output)
    sync_directory(output.parent)


def commit_stage(run, stage, manifest):
    run = Path(run)
    started = time.monotonic()
    checkpoint = run/'checkpoints'
    meta = json.loads((checkpoint/'latest.json').read_text())
    assert meta['stage'] == stage
    weight = inside(run, run/meta['path'])
    assert weight.is_file() and weight.stat().st_size == meta['bytes']
    record = json.loads((run/'stages'/f'stage_{stage:02d}.json').read_text())
    assert record['complete'] and record['weights_saved'] and record['stage'] == stage
    # The original checkpoint writer already streamed its SHA and fsynced its
    # weight file. Do not reread gigabytes solely to duplicate that same check.
    rebuild_events(run, stage, manifest)
    folders = [run/'method_state'/f'stage_{stage:02d}',
               run/'evaluations'/f'stage_{stage:02d}']
    if stage == 1:
        folders.append(run/'evaluations/stage_00')
    for folder in folders:
        if folder.exists():
            for path in folder.rglob('*'):
                if path.is_file():
                    inside(run, path)
                    with path.open('rb') as stream:
                        os.fsync(stream.fileno())
            sync_directory(folder)
    if (run/'evaluations').exists():
        sync_directory(run/'evaluations')
    sync_directory(checkpoint)
    # This pointer is the transaction boundary. Every completed stage's weight
    # file is retained; manual cleanup is separate from the training runner.
    write(checkpoint/'committed_latest.json', meta)
    write(run/'rolling_checkpoint_commits'/f'stage_{stage:02d}.json', {
        'stage': stage, 'committed_checkpoint': meta,
        'removed_older_weight_files': [],
        'all_completed_stage_weights_retained': True,
        'earlier_checkpoint_metadata_retained': True,
        'seconds': time.monotonic()-started,
    })


def pause_requested(run):
    path = Path(run)/'PAUSE_REQUEST.json'
    if not path.exists():
        return False
    request = json.loads(path.read_bytes())
    assert isinstance(request, dict)
    assert isinstance(request.get('request_id'), str) and request['request_id'].strip()
    assert isinstance(request.get('reason'), str) and request['reason'].strip()
    return True


def pause_after_commit(run, stage, manifest, config):
    """Pause only after a complete, committed stage; never truncate S19."""
    run = Path(run)
    if stage == len(manifest['tasks']) or not pause_requested(run):
        return False
    path = run/'PAUSE_REQUEST.json'
    raw = path.read_bytes(); request = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    meta = json.loads((run/'checkpoints/committed_latest.json').read_bytes())
    assert meta['stage'] == stage
    assert json.loads((run/'checkpoints/latest.json').read_bytes()) == meta
    cpmeta = run/'checkpoints'/f'stage_{stage:02d}.json'
    assert json.loads(cpmeta.read_bytes()) == meta
    record = json.loads((run/'stages'/f'stage_{stage:02d}.json').read_bytes())
    assert record['complete'] and record['weights_saved'] and record['stage'] == stage
    assert inside(run, run/meta['path']).stat().st_size == meta['bytes']
    history = run/'pause_history'; history.mkdir(exist_ok=True)
    archived = history/f'request_{digest}.json'
    if archived.exists():
        assert archived.read_bytes() == raw
    else:
        with archived.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        sync_directory(history)
    receipt = history/f'pause_S{stage:02d}_{time.time_ns()}.json'
    write(receipt, dict(status='paused_checkpoint', completed_stage=stage,
        request_id=request['request_id'], reason=request['reason'],
        request_sha256=digest, request_path=str(archived.relative_to(run)),
        manifest_sha256=config['manifest_sha256'], execution_source_sha256=config['execution_source_sha256'],
        external_helper_sha256=config['external_helper_sha256'],
        config_sha256=hashlib.sha256((run/'config.json').read_bytes()).hexdigest(),
        checkpoint=meta, checkpoint_metadata_sha256=hashlib.sha256((run/'checkpoints/latest.json').read_bytes()).hexdigest(),
        full_stage_complete=True,
        pause_boundary='complete evaluation, full checkpoint, stage proof and atomic commit', unix=time.time()))
    receipt_sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    write(run/'run_status.json', dict(status='paused_checkpoint', completed_stage=stage,
        checkpoint=meta, pause_receipt=str(receipt.relative_to(run)), pause_receipt_sha256=receipt_sha))
    # Consume only the exact request that produced the durable pause receipt.
    assert path.read_bytes() == raw
    path.unlink(); sync_directory(run)
    return True


def authorize_resume(run):
    """Explicit authorization for this exact pause and immutable runner version."""
    run = Path(run)
    status = json.loads((run/'run_status.json').read_bytes())
    assert status['status'] == 'paused_checkpoint', 'Resume requires a recorded boundary pause'
    receipt_path = inside(run, run/status['pause_receipt'])
    raw_receipt = receipt_path.read_bytes()
    digest = hashlib.sha256(raw_receipt).hexdigest()
    assert digest == status['pause_receipt_sha256']
    receipt = json.loads(raw_receipt)
    request_path = run/'RESUME_REQUEST.json'
    raw_request = request_path.read_bytes(); request = json.loads(raw_request)
    assert isinstance(request, dict) and request['pause_receipt_sha256'] == digest
    config = json.loads((run/'config.json').read_bytes())
    assert hashlib.sha256((run/'config.json').read_bytes()).hexdigest() == receipt['config_sha256']
    assert config['execution_source_sha256'] == receipt['execution_source_sha256']
    assert config['external_helper_sha256'] == receipt['external_helper_sha256']
    here = Path(__file__).resolve().parent
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in here.glob('*.py')} == config['execution_source_sha256']
    root = here.parents[1]
    for name, expected in config['external_helper_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    meta = json.loads((run/'checkpoints/committed_latest.json').read_bytes())
    assert meta == receipt['checkpoint'] == status['checkpoint']
    assert meta['stage'] == receipt['completed_stage'] == status['completed_stage']
    assert receipt['full_stage_complete'] and meta['full_architecture_and_runtime']
    assert hashlib.sha256((run/'checkpoints/latest.json').read_bytes()).hexdigest() == receipt['checkpoint_metadata_sha256']
    archived = run/'pause_history'/f'resume_{hashlib.sha256(raw_request).hexdigest()}.json'
    if archived.exists():
        assert archived.read_bytes() == raw_request
    else:
        with archived.open('xb') as stream:
            stream.write(raw_request); stream.flush(); os.fsync(stream.fileno())
        sync_directory(archived.parent)
    assert request_path.read_bytes() == raw_request
    request_path.unlink(); sync_directory(run)
    return receipt


def prepare_resume(run, manifest):
    run = Path(run)
    authorize_resume(run)
    pointer = run/'checkpoints/committed_latest.json'
    assert pointer.exists(), 'No committed stage; preserve this run and start fresh'
    meta = json.loads(pointer.read_text())
    stage = meta['stage']
    assert 0 <= stage <= len(manifest['tasks'])
    assert inside(run, run/meta['path']).is_file()
    archive = run/'interrupted_branches'/f'resume_{time.time_ns()}'
    archive.mkdir(parents=True)
    moved = []
    # Future stage artifacts belong to an uncommitted branch, including its
    # replay losses/raw gradients. Preserve all of them, not only predictions.
    for name in ['method_state','evaluations','validity','stages','logs','checkpoints']:
        folder = run/name
        if not folder.exists():
            continue
        for path in list(folder.iterdir()):
            match = re.match(r'stage_(\d+)', path.name)
            if match and int(match.group(1)) > stage:
                target = inside(run, archive/path.relative_to(run))
                target.parent.mkdir(parents=True, exist_ok=True)
                inside(run, path).rename(target)
                moved.append(str(target.relative_to(archive)))
    for path in [run/'run_status.json',run/'logs/events.jsonl',
                 run/'checkpoints/latest.json',run/'memory_summary.json']:
        if path.exists():
            target = inside(run, archive/path.relative_to(run))
            target.parent.mkdir(parents=True,exist_ok=True)
            inside(run,path).rename(target)
            moved.append(str(target.relative_to(archive)))
    write(run/'checkpoints/latest.json',meta)
    rebuild_events(run,stage,manifest)
    write(archive/'RESUME_BRANCH.json', {
        'committed_stage':stage, 'restored_checkpoint_metadata':meta,
        'archived_uncommitted_artifacts':moved,
        'extra_interrupted_compute_preserved_not_counted_as_new_learning':True,
        'old_resource_scopes_preserved_and_not_joined_across_processes':True,
    })
    return meta


def load(run, manifest):
    prepare_resume(run, manifest)
    # Original SHA checking, graph loading and RNG restoration remain intact.
    return original_load(run)


class ProcessResourceMonitor(OriginalMonitor):
    def __init__(self, run, interval=5, resume=False):
        self.formal_run = Path(run)
        self.resume_segment = resume
        directory = self.formal_run/'resources_by_process'/f'{os.getpid()}_{time.time_ns()}'
        super().__init__(directory, interval)

    def finish(self, model=None):
        try:
            summary = super().finish(model)
            summary = json.loads((self.run/'memory_summary.json').read_text())
        except (json.JSONDecodeError, OSError, StopIteration) as error:
            # Never invalidate trained QA results solely because a resource
            # sample was damaged, and never invent the missing measurements.
            summary = {'status':'incomplete','resource_error':repr(error),
                       'unit':'bytes','growth_bytes':None}
        scopes = [p for p in (self.formal_run/'resources_by_process').iterdir() if p.is_dir()]
        summary.update(resource_scope='one process measurement segment',
                       resume_segment=self.resume_segment,
                       scope_directory=str(self.run.relative_to(self.formal_run)),
                       full_run_growth_available=not self.resume_segment and len(scopes)==1,
                       no_cross_process_memory_subtraction=True)
        if self.resume_segment or len(scopes)>1:
            summary['full_run_growth_bytes'] = None
        else:
            summary['full_run_growth_bytes'] = summary.get('growth_bytes')
        write(self.run/'memory_summary.json',summary)
        write(self.formal_run/'memory_summary.json',summary)
        return summary
