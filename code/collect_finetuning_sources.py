"""Download reviewed author source only. Never installs dependencies or executes upstream code."""
import os,json,subprocess,concurrent.futures,hashlib,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'code'
PAPERS=json.loads((ROOT/'paper/continual_learning_manifest.json').read_text())
REPOS={
 'DGG':('DGG','https://github.com/lisongze/DGG','paper first-page URL; official README'),
 'MaDQ / UCo-VQA':('MaDQ','https://github.com/Zi-Jian-Gao/MaDQ','paper first-page author code link'),
 'FOREVER':('FOREVER','https://github.com/WoodScene/FOREVER','ACL paper footnote and official README'),
 'PASs-MoE':('PASs_MoE','https://github.com/yueluoshuangtian/PASs-MoE','ACL paper first-page URL'),
 'PaRSP':('PaRSP','https://github.com/JinhuiBot/PaRSP','paper first-page author code link'),
 'SpaRTA':('SpaRTA','https://github.com/Xnhyacinth/SpaRTA','published ACL paper first-page URL'),
 'OSFT':('OSFT','https://github.com/Red-Hat-AI-Innovation-Team/mini_trainer','paper appendix A.13 author implementation URL'),
 'SLAO / Merge before Forget':('SLAO',None,'No accessible author source found in paper/arXiv/targeted search; third-party backpropagate is not substituted'),
 'Weight Space Consolidation':('Weight_Space_Consolidation','https://github.com/umamicode/weight-space-consolidation','author repository README identifies exact ICLR paper and TRACE bridge'),
 'ALER':('ALER','https://github.com/dakfjalka/aler-distill','paper code footnote and author README'),
 'CoPE':('CoPE','https://github.com/wz7in/COPE','paper first-page author code link'),
 'ECA':('ECA','https://github.com/Snowball0823/ECA','paper code link and author README'),
 'Hyper-LLaVA':('Hyper_LLaVA','https://github.com/zhoujiahuan1991/ICML2026-Hyper-LLaVA','paper first-page URL continued on following line'),
 'CL-MoE':('CL_MoE','https://github.com/ECNU-ICALK/CL-MoE','author README identifies exact CVPR paper and authors'),
 'Any-SSR':('Any_SSR','https://github.com/ZHUANGHP/Any-SSR','paper first-page author code link'),
 'FCIT':('FCIT','https://github.com/Ghy0501/FCIT','paper first-page author code link'),
 'MVP / Instruction-Grounded Visual Projectors':('MVP',None,'Author project page Code links to https://github.com/YOUR%20REPO%20HERE (404); no verified public source'),
 'SMoLoRA':('SMoLoRA','https://github.com/Minato-Zackie/SMoLoRA','paper first-page author code link'),
 'FV-guided Training':('FV_guided_Training','https://github.com/GangweiJiang/FvForgetting','paper first-page author code link'),
 'SERS':('SERS','https://github.com/JerryWangJun/LLM_CL_SERS','paper first-page author code link'),
}
ENV={k:v for k,v in os.environ.items() if k.lower() not in ['http_proxy','https_proxy','all_proxy']}
ENV.update({'GIT_TERMINAL_PROMPT':'0','GIT_LFS_SKIP_SMUDGE':'1','GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'http.proxy','GIT_CONFIG_VALUE_0':''})
SSH_TRANSPORT=False
def run(args,cwd=None,input=None,timeout=180):
 p=subprocess.run(args,cwd=cwd,env=ENV,input=input,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr[-700:])
 return p.stdout.strip()
SUFFIX={'.py','.pyi','.sh','.bash','.yaml','.yml','.toml','.cfg','.ini','.md','.rst','.txt','.json','.c','.cc','.cpp','.h','.hpp','.cu','.cuh','.ipynb','.patch','.diff','.xml','.js','.ts','.html','.css'}
ASSET_DIRS={'data','datasets','data_superni','data_longsequence','checkpoints','weights','pretrained_models','playground','logs','results'}
def is_source(name):
 path=Path(name)
 if path.suffix in {'.py','.pyi','.c','.cc','.cpp','.h','.hpp','.cu','.cuh'}:return True
 if path.parts and path.parts[0].lower()=='third_party':
  parts={x.lower() for x in path.parts};config_dirs={'config','configs','conf','templates','scripts','shell','requirements','recipes'}
  if path.suffix=='.ipynb':return False
  if path.suffix=='.json' and not (parts&config_dirs or 'config' in path.name.lower()):return False
  if parts&ASSET_DIRS and not parts&config_dirs:return False
 if path.parts and path.parts[0].lower() in ASSET_DIRS:return False
 return path.suffix.lower() in SUFFIX or path.name in ['LICENSE','LICENSE-MIT','LICENSE-APACHE','COPYING','Dockerfile','Makefile','.gitmodules','.gitignore','.gitattributes']
def collect(p):
 folder,url,evidence=REPOS[p['method']];dest=BASE/folder;dest.mkdir(exist_ok=True)
 rec={k:p[k] for k in ['method','category','year','venue','page','file','model'] if k in p}
 rec.update({'folder':folder,'repo_url':url,'source_evidence':evidence,'verified_on':'2026-10-12','network':'direct only; no proxy fallback','training_status':'source acquisition only; Qwen2-VL/ten-task port and smoke test not yet done'})
 source=dest/'source'
 if not url:
  rec['status']='author_source_not_found'
 else:
  try:
   if not (source/'.git').exists():
    assert not source.exists(),'Refuse to replace existing non-Git files'
    clone_url=(url.replace('https://github.com/','git@github.com:') if SSH_TRANSPORT else url)+'.git'
    run(['git','clone','--depth','1','--filter=blob:none','--no-checkout','--no-tags',clone_url,str(source)],timeout=180)
   remote=run(['git','remote','get-url','origin'],source)
   canonical_remote=remote.replace('git@github.com:','https://github.com/').removesuffix('.git')
   assert canonical_remote==url.removesuffix('.git')
   rec['revision']=run(['git','rev-parse','HEAD'],source)
   names=run(['git','ls-tree','-r','--name-only','HEAD'],source).splitlines();selected=[n for n in names if is_source(n)]
   # Set the exact pattern before checkout; sparse init would fetch unrelated root assets.
   run(['git','config','core.sparseCheckout','true'],source)
   run(['git','config','core.sparseCheckoutCone','false'],source)
   (source/'.git/info/sparse-checkout').write_text('\n'.join('/'+n for n in selected)+'\n')
   run(['git','read-tree','-mu','HEAD'],source,timeout=300)
   files=[];pointers=[]
   for name in selected:
    path=source/name
    if not path.is_file():continue
    b=path.read_bytes()
    if b.startswith(b'version https://git-lfs.github.com/spec/v1'):pointers.append(name)
    files.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
   rec.update({'status':'downloaded_source_snapshot','source_files':len(files),'source_bytes':sum(f['bytes'] for f in files),'python_files':sum(f['path'].endswith('.py') for f in files),'lfs_pointer_files':pointers,'excluded_asset_files':len(names)-len(selected),'snapshot_policy':'All tracked Python/C++/CUDA source retained; source/config/docs retained; bundled datasets, model binaries, images, Git LFS assets and third-party SDK demo notebooks/benchmark annotations not downloaded','submodule_status':run(['git','submodule','status'],source) if (source/'.gitmodules').exists() else 'none','files_manifest':'files_sha256.json'})
   (dest/'files_sha256.json').write_text(json.dumps(files,indent=2)+'\n')
   rec['license_files']=[f['path'] for f in files if 'LICENSE' in Path(f['path']).name.upper() or Path(f['path']).name=='COPYING']
   if rec['python_files']==0:rec['status']='repository_downloaded_but_no_python_implementation'
  except Exception as e:
   rec.update({'status':'download_failed','error':str(e)[:800]})
 (dest/'source_manifest.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
 (dest/'README.md').write_text(f"# {p['method']}\n\n论文：[{p['venue']} {p['year']}]({p['page']})；分类：{p['category']}。\n\n作者源码：{url or '未找到可验证的公开作者源码'}\n\n核验依据：{evidence}\n\n状态：{rec['status']}。版本：{rec.get('revision','无')}。\n\n`source/` 保留上游源码，未执行安装、训练或下载权重。配置、文件 SHA-256、许可证和源码范围见本目录 manifest。十任务适配状态：尚未完成。\n",encoding='utf-8')
 print('SOURCE_RESULT',p['method'],rec['status'],rec.get('python_files'),rec.get('source_bytes'),flush=True)
 return rec

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--ssh',action='store_true',help='Use the server existing GitHub SSH access; never enable a proxy')
 args=parser.parse_args();SSH_TRANSPORT=args.ssh
 if SSH_TRANSPORT:ENV['GIT_SSH_COMMAND']='ssh -o BatchMode=yes -o ConnectTimeout=15'
 papers=[p for p in PAPERS if p['category'] in ['replay','regularization','architecture']];assert {p['method'] for p in papers}==set(REPOS)
 BASE.mkdir(exist_ok=True);results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  futures=[pool.submit(collect,p) for p in papers]
  for future in concurrent.futures.as_completed(futures):
   results.append(future.result())
   (BASE/'finetuning_source_manifest.json').write_text(json.dumps({'verified_on':'2026-10-12','expected_methods':len(papers),'completed_audits':len(results),'methods':sorted(results,key=lambda x:x['method'])},ensure_ascii=False,indent=2)+'\n')
 print('ALL_METHOD_SOURCE_AUDITS_COMPLETE',len(results),flush=True)
