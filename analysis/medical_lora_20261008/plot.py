"""Diagnostic figures from exact audit CSVs, with project scientific style."""
import csv,json,sys,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(OUT/'plot_templates'))
from style import setup_style,polish_axes,save_png_pdf
from grouped_bar_template import plot_grouped_bar
METHODS=['seq_lora','migu_lora','olora','sapt_lora'];NAMES=['SeqLoRA','MIGU-LoRA','O-LoRA','SAPT-LoRA']
COLORS=['#4F81BD','#9BBB59','#C0504D','#8064A2'];MARKERS=['o','D','^','s'];STYLES=['-','--','-.',':']
TASKS=json.loads((ROOT/'summary_cv/process_data.json').read_text())['task_order']
LABELS=['XSum','Quoref','Relation','Dialogue','Cause','MTS','IU-Xray','VQAv2','VQA-RAD']
def rows(name):return list(csv.DictReader((OUT/name).open()))
def save(fig,name):
 save_png_pdf(fig,str(OUT/'figures'/name),dpi=300);plt.close(fig)
(OUT/'figures').mkdir(exist_ok=True)
a=rows('answer_statistics.csv');setup_style('dashboard');fig,axs=plt.subplots(1,3,figsize=(15,4.8))
for ax,task,learned,title in zip(axs,['medical_mts_dialog_note','medical_iu_xray_impression','medical_vqa_rad'],[6,7,9],['Clinical note','Findings to impression','Medical visual QA']):
 stages=[0,learned-1,learned,9]
 for i,method in enumerate(METHODS):
  y=[float(next(r for r in a if r['method']==method and r['task']==task and int(r['stage'])==s)['rougeL']) for s in stages]
  assert len(y)==4 and np.isfinite(y).all()
  ax.plot(range(4),y,label=NAMES[i],color=COLORS[i],marker=MARKERS[i],linestyle=STYLES[i])
 ax.set_xticks(range(4),[f'Base\n(s0)',f'Before\n(s{learned-1})',f'Learned\n(s{learned})','Final\n(s9)']);ax.set_title(title,fontweight='bold');ax.set_ylabel('ROUGE-L (0-100)');ax.set_ylim(0,100);polish_axes(ax)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=4,frameon=False);fig.tight_layout(rect=(0,0,1,.9));save(fig,'medical_learning_chain')
route=rows('routing_statistics.csv');arr=np.array([[float(next(r for r in route if r['stage']=='9' and r['split']=='test' and r['task']==task)[f'adapter_{j}'])*100 for j in range(1,10)] for task in TASKS]);assert arr.shape==(9,9) and np.allclose(arr.sum(1),100,atol=1e-4)
setup_style('heatmap');fig,ax=plt.subplots(figsize=(12,8));sns.heatmap(arr,ax=ax,annot=True,fmt='.3f',cmap='Blues',vmin=0,vmax=100,xticklabels=LABELS,yticklabels=LABELS,cbar_kws={'label':'Mean routing weight (%)'},annot_kws={'size':10});ax.set_title('SAPT: where test inputs are routed after stage 9',fontweight='bold');ax.set_xlabel('Adapter trained for task');ax.set_ylabel('Test task');ax.tick_params(axis='x',rotation=35);ax.tick_params(axis='y',rotation=0);fig.tight_layout();save(fig,'sapt_routing')
g=rows('gradient_conflicts.csv');arr=np.array([[float(next(r for r in g if r['stage']=='5' and r['task_a']==ta and r['task_b']==tb)['gradient_cosine']) for tb in TASKS] for ta in TASKS]);assert np.allclose(arr,arr.T,atol=1e-7)
setup_style('heatmap');fig,ax=plt.subplots(figsize=(11,8));sns.heatmap(arr,ax=ax,annot=True,fmt='.3f',cmap='RdBu',vmin=-1,vmax=1,center=0,xticklabels=LABELS,yticklabels=LABELS,cbar_kws={'label':'Cosine of mean answer-loss gradients'},annot_kws={'size':10});ax.set_title('SeqLoRA before medical training: local task gradient alignment',fontweight='bold');ax.tick_params(axis='x',rotation=35);ax.tick_params(axis='y',rotation=0);fig.tight_layout();save(fig,'gradient_conflict')
a=rows('vision_ablation_scores.csv');modes=['actual128','wrong128','no_image','actual512']
series=[{'label':name,'color':color,'values':[float(next(r for r in a if int(r['stage'])==stage and r['mode']==mode and r['group']=='all')['EM']) for mode in modes]} for stage,name,color in [(0,'Base (stage 0)','#A6A6A6'),(9,'SeqLoRA (stage 9)',COLORS[0])]]
plot_grouped_bar(['Correct image\n128 tokens','Different image\n128 tokens','No image','Correct image\n512 tokens'],series,title='Same 60 questions: input interventions at fixed weights',xlabel='Image input condition',ylabel='Exact-match accuracy (%)',out_base=str(OUT/'figures/vision_ablation'))
# Reconstruct the exact spatial resize dimensions from the saved official grid.
cases=[r for r in json.loads((OUT/'cases.json').read_text()) if r['task']=='medical_vqa_rad'];assert len(cases)==6
setup_style('dashboard');fig,axs=plt.subplots(3,4,figsize=(15,11))
for k,case in enumerate(cases):
 row=k//2;col=(k%2)*2;grid=case['grid'];w,h=grid[2]*14,grid[1]*14
 with Image.open(OUT/case['image']) as raw:
  original=raw.convert('RGB');low=original.resize((w,h),Image.Resampling.BICUBIC)
  for ax,im,title in [(axs[row,col],original,f"{case['organ']} {case['answer_type']}: original\n{original.width} x {original.height}"),(axs[row,col+1],low,f"Actual training resize\n{w} x {h}; {grid[1]*grid[2]//4} visual tokens")]:
   ax.imshow(im,interpolation='nearest');ax.set_title(title,fontsize=11,fontweight='bold');ax.set_xticks([]);ax.set_yticks([])
  axs[row,col].set_xlabel(case['instance_id'],fontsize=9)
fig.tight_layout();save(fig,'original_and_processed')
# Paired image-cluster bootstrap: question dependence is kept within each image.
rng=np.random.default_rng(42);intervals=[]
for stage in [0,9]:
 records={mode:[json.loads(line) for line in (OUT/f'vision_stage{stage}_{mode}.jsonl').read_text().splitlines()] for mode in modes}
 ids=[r['instance_id'] for r in records['actual128']];assert all([r['instance_id'] for r in rs]==ids for rs in records.values())
 images=sorted(set(r['image'] for r in records['actual128']));groups={image:[i for i,r in enumerate(records['actual128']) if r['image']==image] for image in images}
 for mode in modes[1:]:
  delta=np.array([b['scores']['exact_match']-a['scores']['exact_match'] for a,b in zip(records['actual128'],records[mode])]);boots=[]
  for _ in range(2000):
   chosen=rng.choice(images,len(images),replace=True);idx=[i for image in chosen for i in groups[image]];boots.append(delta[idx].mean())
  lo,hi=np.quantile(boots,[.025,.975]);intervals.append({'stage':stage,'comparison':mode+' - actual128','questions':60,'image_clusters':len(images),'delta_EM_points':float(delta.mean()),'cluster_bootstrap_95_low':float(lo),'cluster_bootstrap_95_high':float(hi)})
with (OUT/'vision_paired_uncertainty.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(intervals[0]),lineterminator='\n');w.writeheader();w.writerows(intervals)
manifest={p.stem:{'png':p.name,'png_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'pdf':p.with_suffix('.pdf').name,'dpi':300} for p in sorted((OUT/'figures').glob('*.png'))}
for item in manifest.values():
 item['data_sha256']={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ['answer_statistics.csv','routing_statistics.csv','gradient_conflicts.csv','vision_ablation_scores.csv','cases.json']}
(OUT/'figures/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Five diagnostic figures exported as 300 DPI PNG and vector PDF.')
