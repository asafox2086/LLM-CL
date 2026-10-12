"""Acquire DGG's two pinned custom source dependencies without installing them."""
import json,concurrent.futures,importlib.util,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'code';PARENT=CODE/'DGG/source'
spec=importlib.util.spec_from_file_location('source_helpers',CODE/'collect_finetuning_sources.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
def acquire(name):
 relative='third_party/'+name;url='https://github.com/lisongze/'+name+'.git';dest=PARENT/relative
 pin=helper.run(['git','ls-tree','HEAD',relative],PARENT).split()[2]
 if not (dest/'.git').exists():
  assert not dest.exists()
  helper.run(['git','clone','--depth','1','--filter=blob:none','--no-checkout','--no-tags',url,str(dest)],timeout=240)
 if helper.run(['git','rev-parse','HEAD'],dest)!=pin:
  helper.run(['git','fetch','--depth','1','origin',pin],dest,timeout=240)
  helper.run(['git','checkout','--detach',pin],dest,timeout=240)
 names=helper.run(['git','ls-tree','-r','--name-only',pin],dest).splitlines();selected=[n for n in names if helper.is_source(n)]
 helper.run(['git','sparse-checkout','init','--no-cone'],dest,timeout=240)
 helper.run(['git','sparse-checkout','set','--no-cone','--stdin'],dest,input='\n'.join('/'+n for n in selected)+'\n',timeout=360)
 helper.run(['git','checkout'],dest,timeout=360)
 files=[]
 for n in selected:
  p=dest/n
  if p.is_file():
   b=p.read_bytes();files.append({'path':n,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
 (CODE/'DGG'/f'{name}_files_sha256.json').write_text(json.dumps(files,indent=2)+'\n')
 result={'name':name,'source_url':url,'revision':pin,'matches_parent_gitlink':helper.run(['git','rev-parse','HEAD'],dest)==pin,'source_files':len(files),'source_bytes':sum(x['bytes'] for x in files),'python_files':sum(x['path'].endswith('.py') for x in files),'network':'direct only','installed':False}
 print('DGG_DEPENDENCY_COMPLETE',json.dumps(result),flush=True);return result
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(acquire,['DGG-transformers','DGG-DeepSpeed']))
 (CODE/'DGG/dependencies_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
 print('DGG_ALL_CUSTOM_SOURCE_DEPENDENCIES_DOWNLOADED',flush=True)
