import sys,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'plot_templates'))
from style import setup_style,polish_axes,darken_color,save_png_pdf
s=json.loads((OUT/'analysis_summary.json').read_text())
setup_style('bar');fig,axes=plt.subplots(1,2,figsize=(15,5.6),dpi=300)
categories=['Anatomy Identification','Modality Recognition','Disease Diagnosis']
series=[]
for arm,condition,label,color in [('base','image','Base + image','#7F7F7F'),('SFT_r8_mean','image','VQA-RAD LoRA + image','#4F81BD'),('SFT_r8_mean','no_image','VQA-RAD LoRA, text only','#DAE8FC')]:
    rs=[next(r for r in s['primary'] if r['category']==c and r['arm']==arm and r['condition']==condition) for c in categories]
    series.append(dict(label=label,values=[r['accuracy'] for r in rs],cis=[r['image_cluster_ci'] for r in rs],color=color))
def bars(ax,labels,series):
    x=np.arange(len(labels));width=.78/len(series)
    for i,item in enumerate(series):
        v=np.asarray(item['values']);assert v.shape==x.shape and np.isfinite(v).all()
        at=x+(i-(len(series)-1)/2)*width;ci=np.asarray(item['cis']);err=np.vstack([np.maximum(0,v-ci[:,0]),np.maximum(0,ci[:,1]-v)])
        ax.bar(at,v,width,label=item['label'],color=item['color'],edgecolor=darken_color(item['color'],.65),linewidth=1.4,zorder=3,yerr=err,capsize=2)
    ax.set_xticks(x);ax.set_xticklabels(labels);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');polish_axes(ax)
bars(axes[0],['Anatomy\n(n=128)','Modality\n(n=128)','Diagnosis\n(n=512)'],series)
axes[0].set_title('A  Larger VQA transfer test',fontweight='bold',pad=14)
p=s['probe'];pseries=[]
for label,color,key in [('Base language output','#7F7F7F','base'),('VQA-RAD LoRA output','#4F81BD','SFT_r8_mean'),('Supervised linear readout','#9BBB59',None)]:
    vals=[r[key]['accuracy'] if key else r['linear_accuracy'] for r in p]
    cis=[r[key]['accuracy_ci'] if key else r['linear_ci'] for r in p]
    pseries.append(dict(label=label,color=color,values=vals,cis=cis))
if (OUT/'adaptation_summary.json').exists():
    a=json.loads((OUT/'adaptation_summary.json').read_text())
    ar=[next(x for x in a['results'] if x['source']==r['source'])['image'] for r in p]
    pseries.append(dict(label='Matched-data LoRA',color='#C0504D',values=[r['new_lora_accuracy'] for r in ar],cis=[r['new_lora_ci'] for r in ar]))
bars(axes[1],[f"{r['source']}\n(n={r['n']})" for r in p],pseries)
axes[1].set_title('B  Diagnostic labels in frozen features',fontweight='bold',pad=14)
left_handles,_=axes[0].get_legend_handles_labels();right_handles,_=axes[1].get_legend_handles_labels()
handles=left_handles+right_handles[2:]
labels=['Base + image','VQA-RAD LoRA + image','VQA-RAD LoRA, text only','Supervised linear readout']+(['Matched-data LoRA'] if len(right_handles)>3 else [])
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=3,frameon=False,fontsize=11)
caption='95% image bootstrap intervals. Linear readout and matched-data LoRA share the same new training labels.' if (OUT/'adaptation_summary.json').exists() else '95% image bootstrap intervals. Readout receives additional supervised labels; no equal-data method comparison.'
fig.text(.5,.005,caption,ha='center',fontsize=10.5)
fig.tight_layout(w_pad=2,rect=(0,.04,1,.86))
save_png_pdf(fig,str(OUT/'figures'/'results'));plt.close(fig)
(OUT/'figures'/'plot_data.json').write_text(json.dumps(dict(primary=series,probe=pseries),indent=2))
