"""Archive every closed stage of the new pct10 runners from OLD to NEW.

Run on NEW. Stdlib/SSH/rsync only; no model loading, scoring, source checkpoint
hashing, pruning or main-table edits. Complete checkpoint SHA is checked on the
NEW copy against the producer's original metadata. --once performs one cycle.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time

ROOT = Path('/data1/liyapeng/program/LLMCL')
OLD = '/dev/shm/llmcl-pct10-1084-zu7e3hb_'
FOLDERS = ('exp/two_backbone_pct10_native_checkpoint_20261027',
           'exp/two_backbone_pct10_forever_checkpoint_20261027')
RESERVE = 20*1024**3
METADATA_LIMIT = 64*1024**2
EXCLUDES = ('*.pt', '*.partial', '*.tmp', '*.lock', '*.bin', '*.safetensors')

# This runs with OLD's existing Python; it imports no model code and reads only
# small JSON/proofs. Enumerating stage_XX.json, rather than latest, closes the
# fast-stage gap. CP payloads are only stat'ed here.
INVENTORY = r'''
from pathlib import Path
import fnmatch,hashlib,json,re
root=Path(__OLD__);folders=__FOLDERS__;entries=[];metadata=[];deferred=[]
def small(p):
 raw=p.read_bytes()
 return dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
for folder in folders:
 base=root/folder
 if not base.exists():continue
 size=0
 for p in base.rglob('*'):
  if p.is_file() and not any(fnmatch.fnmatch(p.name,x) for x in __EXCLUDES__):
   n=p.stat().st_size
   if n<=__METADATA_LIMIT__:size+=n
 metadata.append(dict(path=folder,bytes=size))
 for run in sorted((base/'runs').glob('*/pct10/*/training_*')):
  try:
   config=json.loads((run/'config.json').read_bytes())
   assert config['condition']=='pct10' and not config['preflight_only']
  except (OSError,ValueError,KeyError,AssertionError):continue
  for source in sorted((run/'checkpoints').glob('stage_*.json')):
   try:
    match=re.fullmatch(r'stage_(\d{2})\.json',source.name);assert match
    stage=int(match.group(1));assert 0<=stage<=19
    meta=json.loads(source.read_bytes())
    assert meta['stage']==stage and meta['path']==f'checkpoints/stage_{stage:02d}.pt'
    assert meta['full_architecture_and_runtime'] is True
    assert meta['manifest_sha256']==config['manifest_sha256'] and meta['method']==config['method']
    assert isinstance(meta['bytes'],int) and meta['bytes']>0
    assert re.fullmatch('[0-9a-f]{64}',meta['sha256'])
    weight=run/meta['path']
    assert weight.resolve().is_relative_to((run/'checkpoints').resolve()) and weight.stat().st_size==meta['bytes']
    proofs=[small(run/'config.json'),small(source)]
    if stage:
     record_path=run/'stages'/f'stage_{stage:02d}.json'
     record=json.loads(record_path.read_bytes())
     assert record['complete'] and record['weights_saved'] and record['stage']==stage
     proofs.append(small(record_path))
    else:
     # Native S0 has no stage_00 proof; its completed pause receipt is the
     # producer's commit evidence. Keep it even after a later resume.
     found=None
     for p in sorted((run/'pause_history').glob('pause_S00_*.json')):
      receipt=json.loads(p.read_bytes())
      if receipt.get('completed_stage')==0 and receipt.get('full_stage_complete') and receipt.get('checkpoint')==meta:
       assert receipt['config_sha256']==small(run/'config.json')['sha256']
       found=p;break
     assert found is not None,'S0 has not reached its recorded pause boundary'
     proofs.append(small(found))
    if 'forever' in folder:
     committed=run/'rolling_checkpoint_commits'/f'stage_{stage:02d}.json'
     receipt=json.loads(committed.read_bytes())
     assert receipt['stage']==stage and receipt['committed_checkpoint']==meta
     proofs.append(small(committed))
    entries.append(dict(run=run.relative_to(root).as_posix(),stage=stage,meta=meta,proofs=proofs,final=stage==19))
   except (OSError,ValueError,KeyError,AssertionError) as exc:
    deferred.append(dict(path=source.relative_to(root).as_posix(),reason=repr(exc)))
print(json.dumps(dict(checkpoints=entries,metadata=metadata,deferred=deferred,source_CP_payloads_hashed=0)))
'''


def relative(value):
    p = PurePosixPath(value)
    assert not p.is_absolute() and '..' not in p.parts and p.parts
    return p.as_posix()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path); sync_directory(path.parent)


def stat_key(path):
    s = path.stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, device=s.st_dev, inode=s.st_ino)


def rsync(args, rel, target, metadata=False):
    rel = relative(rel); target.parent.mkdir(parents=True, exist_ok=True)
    command = ['rsync', '-a', '--partial', '--protect-args', '--timeout=90', '-e', shlex.join(args.ssh)]
    if metadata:
        command += ['--max-size='+str(METADATA_LIMIT)]
        command += ['--exclude='+pattern for pattern in EXCLUDES]
        target.mkdir(parents=True, exist_ok=True)
    source = args.remote+':'+args.old_root.rstrip('/')+'/'+rel+('/' if metadata else '')
    return subprocess.run(command+[source, str(target)+('/' if metadata else '')],
        capture_output=True, text=True, timeout=1200)


def proof_files_present(destination, entry):
    for proof in entry['proofs']:
        path = destination/relative(proof['path'])
        if not path.is_file() or path.stat().st_size != proof['bytes'] or sha(path) != proof['sha256']:
            return False
    return True


def cycle(args):
    started = time.monotonic(); dest = args.destination
    inventory_source = INVENTORY.replace('__OLD__', repr(args.old_root)).replace('__FOLDERS__', repr(FOLDERS))
    inventory_source = inventory_source.replace('__EXCLUDES__', repr(EXCLUDES)).replace('__METADATA_LIMIT__', str(METADATA_LIMIT))
    result = subprocess.run(args.ssh+[args.remote, shlex.join([args.old_python, '-'])], input=inventory_source,
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    inventory = json.loads(result.stdout); copied = []; existing = []; skipped = []; snapshots = []
    for entry in inventory['metadata']:
        if shutil.disk_usage(dest).free < entry['bytes']+RESERVE:
            snapshots.append(dict(path=entry['path'],copied=False,reason='NEW 20GiB reserve',estimated_bytes=entry['bytes']))
            continue
        transfer = rsync(args, entry['path'], dest/relative(entry['path']), True)
        snapshots.append(dict(path=entry['path'],returncode=transfer.returncode,stderr=transfer.stderr[-1000:]))
    # Protect available final models first; every earlier committed stage still
    # remains pending until copied. Lack of space never silently drops history.
    entries = sorted(inventory['checkpoints'], key=lambda e: (not e['final'], e['run'], -e['stage']))
    for entry in entries:
        run = relative(entry['run']); meta = entry['meta']; target = dest/run/relative(meta['path'])
        pin = dest/run/'durable_stages'/f"stage_{entry['stage']:02d}.json"
        previous = json.loads(pin.read_bytes()) if pin.exists() else None
        if previous:
            assert previous['source_meta'] == meta, 'A stage identity must never be replaced: '+run
        if previous and target.is_file() and stat_key(target) == previous['destination_stat']:
            if entry['final']:
                final_pin = dest/run/'FINAL_MODEL_RETAINED.json'
                if final_pin.exists(): assert json.loads(final_pin.read_bytes())['source_meta'] == meta
                else: write(final_pin, previous)
            existing.append(dict(run=run,stage=entry['stage'],sha256=meta['sha256'])); continue
        if not proof_files_present(dest, entry):
            skipped.append(dict(run=run,stage=entry['stage'],reason='Source proof snapshot not yet copied')); continue
        if not target.exists():
            if shutil.disk_usage(dest).free < meta['bytes']+RESERVE:
                skipped.append(dict(run=run,stage=entry['stage'],reason='NEW 20GiB reserve',bytes=meta['bytes'])); continue
            temporary = target.with_suffix('.archive_transfer')
            transfer = rsync(args, run+'/'+meta['path'], temporary)
            if transfer.returncode:
                skipped.append(dict(run=run,stage=entry['stage'],reason='Transfer incomplete; own partial retained',
                    returncode=transfer.returncode,stderr=transfer.stderr[-1200:])); continue
            assert temporary.stat().st_size == meta['bytes'] and sha(temporary) == meta['sha256'], 'Transferred CP SHA mismatch: '+str(temporary)
            with temporary.open('rb') as stream: os.fsync(stream.fileno())
            assert not target.exists(), 'Never overwrite an existing archived CP'
            temporary.replace(target); sync_directory(target.parent)
        else:
            # A prior copy may have completed immediately before its pin was
            # published. Verify once rather than replacing or deleting it.
            assert target.stat().st_size == meta['bytes'] and sha(target) == meta['sha256'], 'Existing archive collision: '+str(target)
        value = dict(run=run,stage=entry['stage'],source_meta=meta,source_proofs=entry['proofs'],
            destination=str(target),destination_stat=stat_key(target),verified_unix=time.time(),
            original_source_sha_verified_on_NEW=True,full_architecture_and_runtime=True,
            final_model=entry['final'],automatic_pruning=False)
        write(pin, value)
        if entry['final']: write(dest/run/'FINAL_MODEL_RETAINED.json', value)
        copied.append(dict(run=run,stage=entry['stage'],bytes=meta['bytes'],sha256=meta['sha256'],final=entry['final']))
    value = dict(unix=time.time(),seconds=time.monotonic()-started,copied=copied,already_verified=existing,
        skipped=skipped,source_deferred=inventory['deferred'],metadata_snapshots=snapshots,
        discovered_committed_checkpoints=len(entries),free_bytes=shutil.disk_usage(dest).free,
        reserve_bytes=RESERVE,all_discovered_checkpoints_durable=not skipped and len(copied)+len(existing)==len(entries),
        reserve_scope='Pre-transfer free-space guard; other concurrent disk writers are not controlled.',
        every_completed_stage_enumerated=True,old_files_deleted=0,archive_CP_files_deleted=0,
        source_CP_payloads_hashed=0,new_models=0,new_scores=0,
        final_retention='Copied final CPs are never automatically removed; insufficient space is explicit, not success.',
        metadata_max_file_bytes=METADATA_LIMIT,metadata_is_live_snapshot_not_scientific_result_verification=True)
    write(dest/'COLLECTOR_LATEST.json', value)
    with (dest/'COLLECTOR_HISTORY.jsonl').open('a', encoding='utf8') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--old-root', default=OLD)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--key', type=Path)
    parser.add_argument('--remote', default='liyapeng_grp@10.254.29.8')
    parser.add_argument('--old-python', default='/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python')
    parser.add_argument('--interval', type=float, default=120)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    assert os.getuid() == 1004 and args.root.is_absolute() and args.interval > 0
    assert PurePosixPath(args.old_root).is_absolute()
    args.destination = args.destination or args.root/'exp/stage_checkpoint_dispatch_20261027/private_old_stage_archive'
    assert args.destination.is_absolute() and args.destination != args.root
    args.key = args.key or args.root/'.deployment/old_pct01_id_ed25519'
    args.ssh = ['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',
        '-i',str(args.key),'-p','8666']
    args.destination.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.destination/'COLLECTOR.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                value = cycle(args)
                if args.once: print(json.dumps(value)); return
            except Exception as exc:
                write(args.destination/'COLLECTOR_ERROR.json',dict(unix=time.time(),error=repr(exc),new_models=0,new_scores=0))
                if args.once: raise
            time.sleep(args.interval)


if __name__ == '__main__': main()
