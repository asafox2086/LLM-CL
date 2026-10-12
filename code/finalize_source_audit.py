"""Verify acquired code, publish status/selection metadata, and package only small metadata."""
import json,hashlib,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'code'
records=[json.loads(p.read_text()) for p in sorted(BASE.glob('*/source_manifest.json'))]
assert len(records)==20
STATUS={'downloaded_source_snapshot':'作者源码已下载；待适配','author_source_not_found':'未找到可验证作者源码','repository_downloaded_but_no_python_implementation':'仅 README，尚无实现','download_failed':'下载未完成'}
NOTES={
 'MaDQ':'公开代码主要为 BLIP；不能冒充论文 BLIP-2 大模型扩展。',
 'Weight_Space_Consolidation':'只考虑 trace/wsc_llm.py 等 LLM 桥接代码；不是完整 TRACE 训练器。',
 'CoPE':'官方仓库目前只有 README；不计入可实现的方法。',
 'MVP':'作者项目页 Code 为占位 URL /YOUR REPO HERE，404。',
 'SLAO':'未找到公开作者源码；不以第三方实现替代。',
 'DGG':'含作者定制 Transformers/DeepSpeed；官方原实现提示单卡 >=80GB，尚未做本机预检。',
 'FCIT':'原设定是联邦持续学习，不能直接当集中式方法。',
 'ECA':'以视觉对齐/Q-Former 为核心，十任务纯文本阶段需核验。'}
first=['DGG','PaRSP','SMoLoRA','PASs_MoE','Hyper_LLaVA']
verified=0;source_bytes=0
for rec in records:
 folder=BASE/rec['folder'];source=folder/'source';rec['usage_note']=NOTES.get(rec['folder'],'原作者模型和任务需适配；未运行新实验。')
 if rec['status']=='downloaded_source_snapshot':
  files=json.loads((folder/'files_sha256.json').read_text())
  expected={}
  for line in subprocess.check_output(['git','ls-tree','-r',rec['revision']],cwd=source,text=True).splitlines():
   meta,name=line.split('\t',1);mode,kind,digest=meta.split()
   if kind=='blob':expected[name]=(mode,digest)
  for entry in files:
   p=source/entry['path'];content=p.read_bytes()
   assert hashlib.sha256(content).hexdigest()==entry['sha256'],str(p)
   mode,digest=expected[entry['path']]
   if mode=='120000':
    import os
    content=os.readlink(p).encode()
   assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==digest,str(p)
  rec['integrity']='Every source file matches its recorded SHA-256 and pinned upstream Git blob SHA-1'
  verified+=len(files);source_bytes+=rec['source_bytes']
 deps=folder/'dependencies_manifest.json'
 if deps.exists():
  rec['custom_dependencies']=json.loads(deps.read_text())
  for dep in rec['custom_dependencies']:
   assert dep['matches_parent_gitlink']
   dep_files=json.loads((folder/(dep['name']+'_files_sha256.json')).read_text())
   for entry in dep_files:
    p=source/'third_party'/dep['name']/entry['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
   verified+=len(dep_files);source_bytes+=dep['source_bytes']
  rec['submodule_status']='Both author custom source dependencies downloaded at exact parent gitlinks; packages not installed'
 (folder/'source_manifest.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
 acquisition_note='`source/` 保存作者源码，未安装或执行上游代码；源文件和许可证哈希见 manifest。' if rec['status']=='downloaded_source_snapshot' else '此目录记录来源和检索状态，尚无可用的作者方法实现；README 不计作源码实现。'
 (folder/'README.md').write_text(f"# {rec['method']}\n\n论文：[{rec['venue']} {rec['year']}]({rec['page']})；分类：{rec['category']}。\n\n作者仓库：{rec['repo_url'] or '未找到'}\n\n核验依据：{rec['source_evidence']}\n\n状态：{STATUS[rec['status']]}。固定版本：`{rec.get('revision','无')}`。\n\n备注：{rec['usage_note']}\n\n{acquisition_note} 尚未完成 Qwen2-VL/十任务适配和 smoke test。\n",encoding='utf-8')

manifest={'verified_on':'2026-10-12','expected_methods':20,'completed_audits':20,'status_counts':{s:sum(r['status']==s for r in records) for s in STATUS},'verified_files_including_custom_dependencies':verified,'source_bytes_including_custom_dependencies':source_bytes,'methods':records}
(BASE/'finetuning_source_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
selection={'updated_on':'2026-10-12','paper_scope':'paper/pre_finetuning, 20 papers','base_model':'Qwen2-VL-2B-Instruct','baseline_controls':['SeqLoRA','Frozen'],'first_adaptation_batch':first,'historical_only':['MIGU-LoRA','O-LoRA','SAPT-LoRA'],'excluded_main_comparison_categories':['replay'],'retain_author_replay_in_other_methods':True,'budget_protocol':'data/TenTaskCL/small_budget_protocol.json','training_started':False,'require_before_formal_training':['Preserve author core mechanism, log upstream commit and adaptation diff','Support real text-only and image tasks without dummy-image replacement','Train/dev smoke test and memory preflight','Fixed data manifest and 11x10 evaluation matrix','Report trainable parameters, replay/auxiliary tokens and compute differences'],'rule':'A public repository or source snapshot is not evidence that the ten-task adaptation already runs.'}
for name in first:assert any(r['folder']==name and r['status']=='downloaded_source_snapshot' for r in records)
(BASE/'method_selection.json').write_text(json.dumps(selection,ensure_ascii=False,indent=2)+'\n')
lines=['# 持续微调论文作者源码','', '核验：2026-10-12。对应论文二维表的 **20 篇持续微调方法**，各方法分目录存放。源码在服务器，GitHub 保存索引、版本、哈希及获取脚本；没有安装包或启动正式训练。', '',
 f"已下载 {manifest['status_counts']['downloaded_source_snapshot']} 个方法的源码快照；{manifest['status_counts']['author_source_not_found']} 个未找到作者实现；{manifest['status_counts']['repository_downloaded_but_no_python_implementation']} 个仓库只有 README。源码下载失败剩余 {manifest['status_counts']['download_failed']} 个。",'',
 '原源码位置为 `code/<method>/source/`。源码、配置和文档按固定 Git commit 保存，捆绑数据、模型权重和二进制不重复下载；逐文件 SHA-256 与 Git blob SHA-1 已核验。作者修改版 DGG 两个依赖按父仓库 gitlink 下载。', '',
 '第一批按源码核验和适配优先级选择：**DGG、PaRSP、SMoLoRA、PASs-MoE、Hyper-LLaVA**（去掉回放类比较方法；其他方法保留作者原设置）。SeqLoRA/Frozen 作为基础对照；旧 MIGU/O-LoRA/SAPT 不作这轮主要比较方法。适配仍待完成，详见 [方法选择](method_selection.json) 与 [任务和方法协议](../rules/008_current_task_counts_and_methods.md)。','',
 '| 方法 | 分类 / 发表 | 作者仓库 | 下载与实现状态 | 复现说明 |','|---|---|---|---|---|']
for r in sorted(records,key=lambda r:(r['category'],-r['year'],r['method'])):
 link=f"[{r['repo_url'].split('github.com/',1)[1]}]({r['repo_url']})" if r['repo_url'] else '未找到作者源码'
 lines.append(f"| [{r['method']}]({r['folder']}/README.md) | {r['category']} / {r['venue']} {r['year']} | {link} | {STATUS[r['status']]} | {r['usage_note']} |")
lines+=['','主索引：[finetuning_source_manifest.json](finetuning_source_manifest.json)。每个方法目录记录完整 commit、源文件数、许可证、获取依据和 SHA-256。尚未公开源码的方法如实记录，不把 README 或第三方重写当作者实现。','',
 '复现对同一 2B 基座和十任务的适配需要保留各方法机制。不能统一删掉视觉专家、路由或重要性约束，也不能把回放法替换成普通 ER。新模型上的结果属于适配实验，不能写成原论文原模型的精确复现。','']
(BASE/'README.md').write_text('\n'.join(lines),encoding='utf-8')
metadata=['code/README.md','code/finetuning_source_manifest.json','code/method_selection.json','rules/008_current_task_counts_and_methods.md','data/TenTaskCL/task_counts_audit.json']
for folder in [BASE/r['folder'] for r in records]:
 for f in folder.glob('*.json'):metadata.append(str(f.relative_to(ROOT)))
 metadata.append(str((folder/'README.md').relative_to(ROOT)))
metadata.append('data/MedQuAD/paper_split_seed42/split_manifest.json')
with tarfile.open('/tmp/method_source_metadata_20261012.tar.gz','w:gz') as archive:
 for name in metadata:archive.add(ROOT/name,arcname=name)
print('SOURCE_AUDIT_FINAL',manifest['status_counts'],'verified_files',verified,'bytes',source_bytes,flush=True)
