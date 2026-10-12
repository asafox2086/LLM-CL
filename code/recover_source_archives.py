"""Recover our interrupted sparse checkouts from bounded, pinned author ZIPs."""
import json,requests,zipfile,hashlib,importlib.util,os,signal,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'code';TEMP=Path('/tmp/ten_task_assets_20261012/code_archives');TEMP.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('source_helpers',CODE/'collect_finetuning_sources.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def recover(folder):
 base=CODE/folder;source=base/'source';rec=json.loads((base/'source_manifest.json').read_text());assert rec['status']=='download_failed'
 source_root=source.resolve();assert source_root.is_relative_to(CODE.resolve())
 # Only terminate Git processes of these checkouts created by this task, not other jobs.
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cwd=(p/'cwd').resolve();args=(p/'cmdline').read_bytes().split(b'\0');uid=p.stat().st_uid
   if cwd.is_relative_to(source_root) and args and Path(os.fsdecode(args[0])).name.startswith('git') and uid==os.getuid():os.kill(int(p.name),signal.SIGTERM)
  except (OSError,ValueError):pass
 time.sleep(1)
 for name in ['index.lock','info/sparse-checkout.lock']:
  lock=source/'.git'/name
  if lock.exists():lock.unlink()
 entries=helper.run(['git','ls-tree','-r',rec['revision']],source).splitlines();expected={}
 for entry in entries:
  meta,name=entry.split('\t',1);mode,kind,digest=meta.split()
  if kind=='blob' and helper.is_source(name):expected[name]=digest
 repo=rec['repo_url'].split('github.com/',1)[1];url=f'https://codeload.github.com/{repo}/zip/{rec["revision"]}'
 archive=TEMP/(folder+'.zip');partial=archive.with_suffix('.partial');session=requests.Session();session.trust_env=False
 if not archive.exists():
  with session.get(url,stream=True,timeout=(20,90)) as response:
   response.raise_for_status();size=0
   with partial.open('wb') as f:
    for b in response.iter_content(1024**2):
     size+=len(b);assert size<=64*1024**2,'Bounded code snapshot exceeded 64 MiB';f.write(b)
  partial.replace(archive)
 files=[]
 with zipfile.ZipFile(archive) as z:
  prefix=z.namelist()[0].split('/')[0]+'/'
  for name,digest in expected.items():
   b=z.read(prefix+name);assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==digest,name
   dest=source/name;assert dest.resolve().is_relative_to(source_root),name
   dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.is_file() and not dest.is_symlink() and dest.read_bytes()!=b:raise RuntimeError('Refuse to overwrite changed source: '+name)
   if dest.is_symlink():raise RuntimeError('Refuse to overwrite source symlink: '+name)
   dest.write_bytes(b);files.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':digest})
 rec.update({'status':'downloaded_source_snapshot','source_files':len(files),'source_bytes':sum(x['bytes'] for x in files),'python_files':sum(x['path'].endswith('.py') for x in files),'acquisition':'pinned author ZIP; every included file matches upstream Git blob SHA-1','archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files_manifest':'files_sha256.json','license_files':[x['path'] for x in files if 'LICENSE' in Path(x['path']).name.upper()]})
 rec.pop('error',None);(base/'files_sha256.json').write_text(json.dumps(files,indent=2)+'\n');(base/'source_manifest.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
 print('RECOVERED_PINNED_SOURCE',folder,rec['python_files'],rec['source_bytes'],rec['archive_bytes'],flush=True)
if __name__=='__main__':
 import concurrent.futures
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(recover,['OSFT','FCIT','ECA']))
 print('SOURCE_RECOVERY_COMPLETE',flush=True)
