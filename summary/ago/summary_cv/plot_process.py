"""Reproducible stage curves / matrices from exact exported scores (no smoothing)."""
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).parent/'plot_templates'))
from style import setup_style,polish_axes,save_png_pdf,darken_color
from line_chart_template import LINE_STYLES

METHODS=['seq_lora','migu_lora','olora','sapt_lora']
NAMES={'seq_lora':'SeqLoRA','migu_lora':'MIGU-LoRA','olora':'O-LoRA','sapt_lora':'SAPT-LoRA'}
COLORS={'seq_lora':'#4F81BD','migu_lora':'#9BBB59','olora':'#C0504D','sapt_lora':'#8064A2'}
SAMPLE=ROOT/'sample';PDF=ROOT/'summary_cv/figures';REGISTRY={}

def save(fig,name):
    save_png_pdf(fig,str(SAMPLE/name),dpi=300)
    (SAMPLE/(name+'.pdf')).replace(PDF/(name+'.pdf'))
    REGISTRY[name]={'png':'sample/'+name+'.png','pdf':'summary_cv/figures/'+name+'.pdf','dpi':300}
    plt.close(fig)

def curves(ax,x,values,title,ylabel,zero=False):
    for i,method in enumerate(METHODS):
        y=np.asarray([np.nan if v is None else v for v in values[method]],dtype=float)
        assert y.shape==x.shape
        assert np.isfinite(y[~np.isnan(y)]).all()
        marker,linestyle=LINE_STYLES[i]
        ax.plot(x,y,label=NAMES[method],color=COLORS[method],marker=marker,linestyle=linestyle,
                markersize=5,markeredgecolor=darken_color(COLORS[method]),markeredgewidth=1)
    if zero:ax.axhline(0,color='#777777',lw=1)
    ax.set_title(title,fontsize=16,fontweight='bold');ax.set_xlabel('Completed task stage')
    ax.set_ylabel(ylabel);ax.set_xticks(x);polish_axes(ax)

def legend(fig,ax):
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.01))

def dashboard(data,prefix):
    T=len(data['task_labels']);x=np.arange(T+1)
    setup_style('dashboard')
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    specs=[('all_fixed_mean','Fixed complete task set','Mean ROUGE-L (0-100)',False),
           ('seen_mean','Seen tasks (changing task composition)','Mean ROUGE-L (0-100)',False),
           ('old_fixed_shock','Old tasks: effect of this transition','ROUGE-L change (points)',True),
           ('future_delta_base','Not-yet-trained tasks vs. base','ROUGE-L change (points)',True)]
    for ax,(key,title,label,zero) in zip(axes.flat,specs):
        values={m:[r[key] for r in data['methods'][m]['stage_statistics']] for m in METHODS}
        curves(ax,x,values,title,label,zero)
    legend(fig,axes.flat[0]);fig.tight_layout(rect=(0,0,1,.96));save(fig,prefix+'_trajectories')

def heatmaps(data,key,prefix):
    matrices=[np.asarray(data['methods'][m][key]) for m in METHODS]
    assert all(a.shape==matrices[0].shape for a in matrices)
    limit=max(float(np.abs(a).max()) for a in matrices)
    assert np.isfinite(limit) and limit>0
    rows=matrices[0].shape[0];T=len(data['task_labels'])
    setup_style('heatmap');fig,axes=plt.subplots(2,2,figsize=(15,12.5))
    fig.subplots_adjust(right=.89,wspace=.27,hspace=.50,bottom=.12)
    color_ax=fig.add_axes([.915,.20,.018,.60])
    for index,(method,ax,array) in enumerate(zip(METHODS,axes.flat,matrices)):
        assert array.shape==(rows,T) and np.isfinite(array).all()
        sns.heatmap(array,ax=ax,cmap='RdBu',center=0,vmin=-limit,vmax=limit,annot=True,fmt='.1f',
            annot_kws={'size':9},linewidths=.4,linecolor='white',xticklabels=data['task_labels'],
            yticklabels=list(range(rows)) if key=='delta_baseline' else list(range(1,rows+1)),
            cbar=index==0,cbar_ax=color_ax if index==0 else None)
        ax.set_title(NAMES[method],fontsize=17,fontweight='bold');ax.set_xlabel('Evaluated task' if index >= 2 else '')
        ax.set_ylabel('Completed task stage');ax.tick_params(axis='x',rotation=45,labelsize=11);ax.tick_params(axis='y',rotation=0)
        for task in range(T):
            y=task+1 if key=='delta_baseline' else task
            ax.add_patch(Rectangle((task,y),1,1,fill=False,lw=1.5,edgecolor='black'))
    color_ax.set_ylabel('ROUGE-L change (points)');save(fig,prefix+'_'+key)

def domain_curves(data):
    setup_style('dashboard');fig,axes=plt.subplots(2,2,figsize=(13,9));x=np.arange(10)
    domain_names={'general_text':'General language (fixed 5 tasks)', 'medical_text':'Medical language (fixed 2 tasks)',
                  'natural_vision':'Natural vision (VQAv2)','medical_vision':'Medical vision (VQA-RAD)'}
    for ax,(domain,ids) in zip(axes.flat,data['domains'].items()):
        values={m:np.asarray(data['methods'][m]['matrix'])[:,ids].mean(1).tolist() for m in METHODS}
        curves(ax,x,values,domain_names[domain],'Mean ROUGE-L (0-100)')
        for boundary in [5.5,7.5,8.5]:ax.axvline(boundary,color='#999999',lw=.9,linestyle=':')
    legend(fig,axes.flat[0]);fig.tight_layout(rect=(0,0,1,.96));save(fig,'cv_domain_trajectories')

def task_curves(data):
    setup_style('dashboard');fig,axes=plt.subplots(3,3,figsize=(17,12));x=np.arange(10)
    for j,ax in enumerate(axes.flat):
        values={m:np.asarray(data['methods'][m]['matrix'])[:,j].tolist() for m in METHODS}
        curves(ax,x,values,data['task_labels'][j],'ROUGE-L (0-100)')
        ax.axvline(j+1,color='#777777',linestyle=':',lw=1)
        ax.set_ylim(0,100)
    legend(fig,axes.flat[0]);fig.tight_layout(rect=(0,0,1,.96));save(fig,'cv_task_trajectories')

def probes_if_ready():
    path=ROOT/'summary_cv/knowledge_probe_results.csv'
    if not path.exists():return
    import csv
    rows=list(csv.DictReader(path.open()))
    if len(rows)!=40:return
    setup_style('dashboard');fig,axes=plt.subplots(1,3,figsize=(16,4.8));x=np.arange(10)
    for ax,group,title in zip(axes,['general_accuracy','medical_accuracy','macro_accuracy'],
                             ['External general knowledge','External medical knowledge','57-subject macro accuracy']):
        values={m:[float(next(r for r in rows if r['method']==m and int(r['stage'])==s)[group]) for s in x] for m in METHODS}
        curves(ax,x,values,title,'Accuracy (%)');ax.set_ylim(0,100)
    legend(fig,axes[0]);fig.tight_layout(rect=(0,0,1,.93));save(fig,'cv_external_knowledge')

if __name__=='__main__':
    SAMPLE.mkdir(exist_ok=True);PDF.mkdir(exist_ok=True)
    cv=json.loads((ROOT/'summary_cv/process_data.json').read_text())
    t5=json.loads((ROOT/'summary/t5_process_data.json').read_text())
    dashboard(cv,'cv');domain_curves(cv);heatmaps(cv,'delta_baseline','cv');heatmaps(cv,'delta_transition','cv');task_curves(cv)
    dashboard(t5,'t5');heatmaps(t5,'delta_baseline','t5');probes_if_ready()
    for item in REGISTRY.values():
        item['sha256']=hashlib.sha256((ROOT/item['png']).read_bytes()).hexdigest()
        item['data_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'summary_cv/process_data.json',ROOT/'summary/t5_process_data.json',ROOT/'summary_cv/knowledge_probe_results.csv']}
    (SAMPLE/'figure_manifest.json').write_text(json.dumps(REGISTRY,indent=2)+'\n')
    print('Exported',len(REGISTRY),'figures at 300 DPI and vector PDFs.')
