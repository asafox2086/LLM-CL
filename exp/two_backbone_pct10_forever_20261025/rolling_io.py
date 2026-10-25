"""Stage-atomic rolling checkpoints and per-process resource scopes.

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
    # This pointer is the transaction boundary. Old weights are retained until
    # all stage artifacts and the new pointer are durable.
    write(checkpoint/'committed_latest.json', meta)
    removed = []
    for path in checkpoint.glob('stage_*.pt'):
        match = re.fullmatch(r'stage_(\d+)\.pt', path.name)
        assert match
        if int(match.group(1)) < stage:
            inside(run, path).unlink()
            removed.append(path.name)
    sync_directory(checkpoint)
    write(run/'rolling_checkpoint_commits'/f'stage_{stage:02d}.json', {
        'stage': stage, 'committed_checkpoint': meta,
        'removed_older_weight_files': sorted(removed),
        'earlier_checkpoint_metadata_retained': True,
        'seconds': time.monotonic()-started,
    })


def prepare_resume(run, manifest):
    run = Path(run)
    pointer = run/'checkpoints/committed_latest.json'
    assert pointer.exists(), 'No committed stage; preserve this run and start fresh'
    meta = json.loads(pointer.read_text())
    stage = meta['stage']
    assert 1 <= stage <= len(manifest['tasks'])
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
