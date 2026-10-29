"""Render already verified aggregates only; no QA, scoring, or model access."""
from pathlib import Path
import json, hashlib, re
R=Path(__file__).resolve().parents[2]
D=R/'summary/four_domain_scope_20261029'
readme=(R/'README.md').read_text()
orig=readme
sources={}
def load(name):
 p=R/name; b=p.read_bytes(); sources[name]=hashlib.sha256(b).hexdigest(); return json.loads(b)
indices={
'pct01':'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json',
'pct10':'summary/two_backbone_pct10_20261025/second_complete_sources/VERIFIED_INDEX.json'}
labels={'SeqFT_LoRA':'SeqFT-LoRA（q/v-r8）','CL_MoE':'CL-MoE','PASs_MoE':'PASs-MoE'}
records={}
for pct,path in indices.items():
 rows=[]
 for item in load(path)['methods']:
  p=item['proof_file']
  if not (R/p).exists() and pct=='pct10': p='summary/two_backbone_pct10_20261025/second_complete_sources/'+Path(p).name
  proof=load(p)
  if 'proof_sha256' in item: assert sources[p]==item['proof_sha256']
  rows.append((item['backbone'],item['method'],proof,p))
 if pct=='pct01':
  pub=load('summary/two_backbone_pct01_20261017/forever_verified_20261020/PUBLICATION.json')
  p=pub['FOREVER_proof']; proof=load(p)['shared_V2_proof'];assert sources[p]==pub['FOREVER_proof_sha256']
  rows.append(('onevision','FOREVER',proof,p))
 records[pct]=[]
 for bb,m,d,p in rows:
  v=d['final_native19'];assert len(v)==19 and d['final_QA']==21454
  domains=d['domains'];macro=sum(domains[k]['F1'] for k in ['natural','medical','geography','finance'])/4
  records[pct].append(dict(backbone=bb,method=m,label=labels.get(m,m),endpoint='S19',source=p,final16=v[:16],medical=domains['medical']['F1'],geography=domains['geography']['F1'],macro4=macro,AP16_at_S19=sum(v[:16])/16,aggregate19=d['aggregate']))
assert [len(records[k]) for k in ['pct01','pct10']]==[9,2]
# Historical table cells remain unchanged and serve only as published resource references.
resources={}
for pct,start,end in [('pct01','<!-- FORMAL_MAIN_TABLE_BEGIN -->','<!-- FORMAL_MAIN_TABLE_END -->'),('pct10','<!-- PCT10_MAIN_TABLE_BEGIN -->','<!-- PCT10_MAIN_TABLE_END -->')]:
 bb=None
 for l in orig.split(start)[1].split(end)[0].splitlines():
  if not l.startswith('|'):continue
  c=[x.strip() for x in l.split('|')[1:-1]]
  if c[0].startswith('**Qwen'):bb='qwen'
  elif c[0].startswith('**LLaVA'):bb='onevision'
  elif bb and len(c)>=19: resources[pct,bb,c[0]]=c
note='**已跑结果回填：带〔S19〕的行来自已核验的原19任务训练终点，本表展示其中四域／16小类，不是重新训练得到的S16结果。未来S16结果单独标记，不覆盖原记录。**'
outputs={}
for pct in ['pct01','pct10']:
 text=(D/(pct+'_TABLES.md')).read_text()
 if '〔S19〕' in text: raise RuntimeError('Already populated; inspect existing publication instead of replaying')
 by={(x['backbone'],x['label']):x for x in records[pct]}
 bb=None; lines=[]
 for line in text.splitlines():
  if '<th colspan="20">LLaVA' in line or line.startswith('| **LLaVA'):bb='onevision'
  elif '<th colspan="20">Qwen' in line or line.startswith('| **Qwen'):bb='qwen'
  match=re.match(r'<tr><th>(.*?)</th>',line)
  if match and (bb,match[1]) in by:
   x=by[bb,match[1]];v=x['final16'];vals=v[:11]+[x['medical']]+v[11:15]+[x['geography'],v[15],x['macro4']]
   assert len(vals)==19
   line='<tr><th>'+x['label']+'〔S19〕</th>'+''.join('<td>%.3f</td>'%a for a in vals)+'</tr>'
  elif line.startswith('| '):
   c=[x.strip() for x in line.split('|')[1:-1]]
   if (bb,c[0]) in by:
    x=by[bb,c[0]]; oldname=x['label'].split('（')[0]; h=resources[pct,bb,oldname]
    def clean(z):return z if z not in ['', '未测'] else '—'
    count='%.3f'%(int(h[18])/1e6) if h[18] else '—'
    agg=x['aggregate19']
    vals=[x['label']+'〔S19〕','%.3f'%x['AP16_at_S19'],'%.3f'%agg['FWT'],'%.3f'%agg['BWT'],count,'—']+[clean(h[i]) for i in [12,13,14,15,16,17]]
    assert len(vals)==12
    line='| '+' | '.join(vals)+' |'
  lines.append(line)
 text='\n'.join(lines)+'\n'
 text=text.replace('## 表1：16个子任务的最终Answer F1','## 表1：16个子任务的最终Answer F1\n\n'+note)
 a=text.index('各小类均为统一四域终点');b=text.index('\n\n## 表2',a)
 text=text[:a]+'小类指标为原始Answer F1（0–100）；〔S19〕行逐项采用已核验的R[19,t]，t=1…16。医学和遥感Avg按域内TEST题数加权，四域宏对四域等权，AP16对这16项等权。原S19的FWT19/BWT19保留原19任务口径，不伪装为16任务指标。—表示尚无相应完整核验或测量，不是0。'+text[b:]
 text=text.replace('| AP16 ↑ | FWT ↑ | BWT ↑ | 总部署参数<br>M ↓ |','| AP16@所列终点 ↑ | FWT19 ↑ | BWT19 ↑ | 完整保存系数<br>M ↓ |')
 a=text.index('† GPU峰值')
 text=text[:a]+'〔S19〕资源沿用历史主表原最终保存态、原dtype的已核验记录；完整保存系数包含必要持久状态，不等同可训练参数或纯神经参数。新增部署参数未单独核定时留空，不用权重字节差代替。† GPU为历史同卡固定五窗（含交通输入）、每题32 token的推理allocated峰；不是新四域固定窗测量。原权重、GPU、延迟及速度的测量边界与来源见README历史主表脚注。训练峰、最终checkpoint及完整证据另存。MiB=2²⁰字节，M=10⁶。\n'
 outputs[pct]=text
(D/'EXISTING_RESULTS.json').write_text(json.dumps({'scope':'Verified S19 endpoint displayed on four domains; not an S16 run','new_scoring':False,'source_sha256':sources,'resource_source':'README.md historical table blocks retained byte-for-byte','rows':records},ensure_ascii=False,indent=2)+'\n')
for pct,text in outputs.items():(D/(pct+'_TABLES.md')).write_text(text)
start='<!-- FOUR_DOMAIN_MAIN_TABLES_BEGIN -->';end='<!-- FOUR_DOMAIN_MAIN_TABLES_END -->'
section=start+'\n'+outputs['pct01'].replace('# 1%训练：四域主表','## 1%：四域／16小类结果').replace('## 表','### 表')+'\n<details>\n<summary>10%训练：性能与开销</summary>\n\n'+outputs['pct10']+'\n</details>\n'+end
readme=readme.split(start)[0]+section+readme.split(end)[1]
readme=readme.replace('两套四域主表须用统一S16终点的完整评估；下方已有五域表保留为历史结果，不直接删列充作新实验。','下方先回填已跑基线的S19终点四域明细，并逐行标注；新的纯四域训练结果使用S16终点，二者不混称。历史五域表完整保留。')
for a,b in [('<!-- FORMAL_MAIN_TABLE_BEGIN -->','<!-- FORMAL_MAIN_TABLE_END -->'),('<!-- PCT10_MAIN_TABLE_BEGIN -->','<!-- PCT10_MAIN_TABLE_END -->')]:assert orig.split(a)[1].split(b)[0]==readme.split(a)[1].split(b)[0]
(R/'README.md').write_text(readme)
print(json.dumps({'filled_pct01':len(records['pct01']),'filled_pct10':len(records['pct10']),'small_class_cells':176,'historical_tables_unchanged':True}))
