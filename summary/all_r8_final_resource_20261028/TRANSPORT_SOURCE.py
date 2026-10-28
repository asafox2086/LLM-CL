"""One-shot all-r8 asset transport ONLY; default inspect does not contact OLD.
Two scp transfers: 46 small frozen files in a tar, then the original full CP.
Never replays snapshot preparation, follows NEW asset symlinks, or runs models.
"""
import argparse,hashlib,json,os,shlex,shutil,subprocess,tarfile,time
from pathlib import Path

ROOT=Path('/data1/liyapeng/program/LLMCL')
SNAP=ROOT/'.deployment/all_r8_resource_snapshot_20261028'
OUT=ROOT/'.deployment/all_r8_resource_transport_20261028'
DEST='/dev/shm/llmcl-all-r8-resource-1084-20261028'
REMOTE='liyapeng_grp@10.254.29.8'
OLDPY='/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python'
ASSETS='/data2/liyapeng_grp/program/LLMCL'
SNAPSHA='384cedd0aee02a089ebc1bd580dbb9bfa61c68162f885238a889cd17f7472d0c'
CP='exp/osft_increment_confirmation_20261022/v1/private/checkpoints/stage_19.pt'
CPSHA='416ed177803b9bf986dacd73a6d1bf18c6c0c59a1eb65f42785d81a628d31519'
CPBYTES=1941599549
KEY=ROOT/'.deployment/old_pct01_id_ed25519'
OPTIONS=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-i',str(KEY)]


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def statkey(p):
    s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)


def write(p,value):
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def old(code):
    return subprocess.run(['ssh',*OPTIONS,'-p','8666',REMOTE,shlex.join([OLDPY,'-c',code])],
                          capture_output=True,text=True,timeout=1800,check=True).stdout


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    assert os.getuid()==1004 and ROOT.resolve()==ROOT
    assert sha(SNAP/'SNAPSHOT.json')==SNAPSHA
    receipt=json.loads((SNAP/'SNAPSHOT.json').read_bytes());files=receipt['files_sha256']
    assert len(files)==45 and receipt['original_READY_sources']==37
    assert receipt['checkpoint']['bytes']==CPBYTES and receipt['checkpoint']['sha256']==CPSHA
    members={}
    for rel,h in {**files,'SNAPSHOT.json':SNAPSHA}.items():
        q=Path(rel);p=SNAP/q
        assert not q.is_absolute() and '..' not in q.parts and p.is_file() and not p.is_symlink()
        assert p.resolve()==p and sha(p)==h
        members[rel]={'sha256':h,'bytes':p.stat().st_size}
    cp=SNAP/CP;before=statkey(cp)
    assert cp.is_file() and not cp.is_symlink() and cp.stat().st_uid==1004 and cp.stat().st_size==CPBYTES
    assert (cp.stat().st_dev,cp.stat().st_ino)==(receipt['checkpoint']['device'],receipt['checkpoint']['inode'])
    manifest={**members,CP:{'sha256':CPSHA,'bytes':CPBYTES}}
    assert len(members)==46 and len(manifest)==47
    total=sum(v['bytes'] for v in manifest.values())
    plan={'inspection_only':not a.execute,'OLD_destination':DEST,'scientific_files':47,'small_tar_members':46,
          'payload_bytes':total,'CP_bytes':CPBYTES,'minimum_OLD_pre_free_bytes':20000000000,
          'required_post_free_bytes':17700000000,'post_reserve_scope':'9.7GB remaining formal CP + at least 8GB headroom; decimal GB',
          'source_snapshot_sha256':SNAPSHA,'CP_source_payload_rehashed':False,
          'data_model_transport':False,'models':0,'scores':0,'GPU_or_allocator_calls':0}
    if not a.execute:print(json.dumps(plan));return
    assert not OUT.exists();OUT.mkdir()
    write(OUT/'STARTED.json',{'unix':time.time(),'script_sha256':sha(Path(__file__)),**plan})
    try:
        package=OUT/'sources.tar'
        with tarfile.open(package,'w',format=tarfile.GNU_FORMAT) as tar:
            for rel,entry in members.items():
                p=SNAP/rel;assert sha(p)==entry['sha256']
                info=tarfile.TarInfo(rel);info.size=entry['bytes'];info.mode=0o600
                with p.open('rb') as f:tar.addfile(info,f)
        package_sha=sha(package)
        setup="""from pathlib import Path
import os,shutil,json
D=Path(DEST);A=Path(ASSETS)
assert os.getuid()==1084 and D.parent==Path('/dev/shm') and D.resolve()==D and not D.exists()
assert all((A/n).is_dir() for n in ('data','model'))
assert shutil.disk_usage(D.parent).free>=max(20000000000,TOTAL+17700000000)
D.mkdir(mode=0o700);(D/'.incoming').mkdir(mode=0o700)
print(json.dumps({'exclusive_destination_created':str(D),'no_model_or_GPU':True}))
""".replace('DEST',repr(DEST)).replace('ASSETS',repr(ASSETS)).replace('TOTAL',str(total+package.stat().st_size))
        setup_output=old(setup);write(OUT/'OLD_SETUP.json',json.loads(setup_output))
        for local,remote_name in ((package,'sources.tar'),(cp,'checkpoint.pt')):
            subprocess.run(['scp',*OPTIONS,'-P','8666',str(local),REMOTE+':'+DEST+'/.incoming/'+remote_name],
                           capture_output=True,text=True,timeout=1800,check=True)
        assert statkey(cp)==before,'Original CP changed while transporting'
        finish="""from pathlib import Path,PurePosixPath
import hashlib,json,os,shutil,tarfile,time
D=Path(DEST);A=Path(ASSETS);expected=MANIFEST;small=SMALL
assert os.getuid()==1084 and D.is_dir() and not D.is_symlink()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
try:
 assert shutil.disk_usage(D).free>=17700000000
 package=D/'.incoming/sources.tar';assert sha(package)==PACKAGESHA
 seen=set()
 with tarfile.open(package,'r|') as tar:
  for member in tar:
   rel=member.name;q=PurePosixPath(rel)
   assert rel in small and rel not in seen and not q.is_absolute() and '..' not in q.parts
   assert member.isreg() and member.size==small[rel]['bytes']
   target=D/rel;target.parent.mkdir(parents=True,exist_ok=True)
   assert target.parent.resolve().is_relative_to(D) and not target.exists()
   h=hashlib.sha256();stream=tar.extractfile(member);assert stream is not None
   with target.open('xb') as f:
    for b in iter(lambda:stream.read(8*1024**2),b''):f.write(b);h.update(b)
    f.flush();os.fsync(f.fileno())
   assert h.hexdigest()==small[rel]['sha256'];seen.add(rel)
 assert seen==set(small)
 cp=D/'.incoming/checkpoint.pt';assert not cp.is_symlink() and cp.stat().st_size==CPBYTES and sha(cp)==CPSHA
 with cp.open('rb') as f:os.fsync(f.fileno())
 target=D/CP;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();cp.rename(target);syncdir(target.parent)
 assert target.stat().st_uid==1084
 for rel,info in expected.items():assert (D/rel).is_file() and not (D/rel).is_symlink() and (D/rel).stat().st_size==info['bytes']
 for name in ('data','model'):(D/name).symlink_to(A/name,target_is_directory=True)
 assert shutil.disk_usage(D).free>=17700000000
 result={'complete':True,'unix':time.time(),'destination':str(D),'files_sha256':expected,'files':len(expected),
         'source_snapshot_sha256':SNAPSHA,'checkpoint_full_SHA':CPSHA,'checkpoint_UID':1084,
         'local_asset_links':{n:str(A/n) for n in ('data','model')},'no_received_symlinks':True,
         'free_after':shutil.disk_usage(D).free,'original_NEW_payload_retained':True,
         'GPU_launches':0,'models':0,'scores':0,'binding_created':False,'resource_ready':False}
 with (D/'TRANSPORT_COMPLETE.json').open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
 for p in sorted((p for p in D.rglob('*') if p.is_dir() and not p.is_symlink()),key=lambda p:len(p.parts),reverse=True):syncdir(p)
 syncdir(D);syncdir(D.parent);print(json.dumps(result))
except BaseException as e:
 with (D/'TRANSPORT_FAILURE.json').open('x') as f:json.dump({'failed':True,'exception':repr(e),'partial_preserved':True,'no_retry':True},f);f.flush();os.fsync(f.fileno())
 raise
"""
        replacements={'DEST':repr(DEST),'ASSETS':repr(ASSETS),'MANIFEST':repr(manifest),'SMALL':repr(members),
                      'PACKAGESHA':repr(package_sha),'CPBYTES':str(CPBYTES),'CPSHA':repr(CPSHA),'SNAPSHA':repr(SNAPSHA),'CP':repr(CP)}
        # Replace identifier tokens through tokenize, never shell interpolation.
        import io,tokenize
        toks=[]
        for token in tokenize.generate_tokens(io.StringIO(finish).readline):
            if token.type==tokenize.NAME and token.string in replacements:token=token._replace(string=replacements[token.string])
            toks.append(token)
        finish=tokenize.untokenize(toks)
        result=json.loads(old(finish));assert result['complete'] and result['files']==47
        write(OUT/'COMPLETE.json',result);print(json.dumps(result))
    except BaseException as e:
        write(OUT/'FAILURE.json',{'unix':time.time(),'exception':repr(e),'partial_preserved':True,'no_automatic_retry':True});raise


if __name__=='__main__':main()
