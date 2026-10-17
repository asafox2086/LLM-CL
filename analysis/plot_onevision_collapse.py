"""Two audit panels using the project's grouped-bar style; no new inference."""
import hashlib,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis/omnimed_20261009/plot_templates'))
from style import setup_style,polish_axes,darken_color,save_png_pdf


def main():
    source=ROOT/'summary/onevision_diagnostics_20261016/ACQUISITION_COLLAPSE_AUDIT.json'
    audit=json.loads(source.read_text(encoding='utf-8'))
    cases={c['start']:c for c in audit['rural']['cases']}
    random=cases['random_A_zero_B']['endpoints'];past=cases['past_AB']['endpoints']
    strict=[r['primary_EM']['correct'] for r in random]
    auxiliary=[r['auxiliary_literal_prefix_correct'] for r in random]
    urban=[past[0]['predictions'].get('urban',0),past[1]['predictions'].get('urban',0),audit['rural']['TEST_labels']['urban']]
    rural=[past[0]['predictions'].get('rural',0),past[1]['predictions'].get('rural',0),audit['rural']['TEST_labels']['rural']]
    assert strict==[0,62] and auxiliary==[64,62] and urban==[26,100,44] and rural==[74,0,56]
    assert all(np.isfinite(v).all() for v in [strict,auxiliary,urban,rural])
    setup_style('bar');fig,axes=plt.subplots(1,2,figsize=(11.8,4.6),dpi=300)
    x=np.arange(2);width=.34
    for values,label,color,offset in [(strict,'Original exact match','#7F7F7F',-width/2),(auxiliary,'Literal-prefix auxiliary check','#4F81BD',width/2)]:
        bars=axes[0].bar(x+offset,values,width,label=label,color=color,edgecolor=darken_color(color),linewidth=1.4,zorder=3)
        axes[0].bar_label(bars,padding=4,fontsize=12)
    axes[0].set_xticks(x,['1 update','4 updates']);axes[0].set_ylabel('Correct labels / 100 test QA')
    axes[0].set_title('(a) Score gain can be format adaptation',fontsize=15,fontweight='bold')
    axes[0].legend(loc='upper left',fontsize=10,frameon=False)
    x=np.arange(3)
    bars=axes[1].bar(x,urban,.6,label='Predicted urban / gold urban',color='#4F81BD',edgecolor=darken_color('#4F81BD'),linewidth=1.4,zorder=3)
    tops=axes[1].bar(x,rural,.6,bottom=urban,label='Predicted rural / gold rural',color='#9BBB59',edgecolor=darken_color('#9BBB59'),linewidth=1.4,zorder=3)
    for i,(u,v) in enumerate(zip(urban,rural)):
        if u:axes[1].text(i,u/2,str(u),ha='center',va='center',fontsize=13,color='white',fontweight='bold')
        if v:axes[1].text(i,u+v/2,str(v),ha='center',va='center',fontsize=13,fontweight='bold')
    axes[1].set_xticks(x,['Old AB\n1 update','Old AB\n4 updates','Original\ntest labels'])
    axes[1].set_ylabel('Number of QA / 100')
    axes[1].set_title('(b) Further training collapses outputs',fontsize=15,fontweight='bold')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,1.01),fontsize=9,frameon=False)
    for ax in axes:
        ax.set_ylim(0,125);ax.set_yticks(np.arange(0,101,20));polish_axes(ax)
    fig.text(.5,.02,'T12 rural/urban: frozen 6 TRAIN / 100 TEST QA, seed42. (a) random A + zero B; auxiliary rule is post-hoc.\n(b) old AB shared-prefix endpoints; TRAIN labels: urban4 / rural2. Descriptive counts, no uncertainty bars.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.12,1,1))
    folder=source.parent/'figures';folder.mkdir(exist_ok=True)
    save_png_pdf(fig,str(folder/'rural_format_and_collapse'),dpi=300);plt.close(fig)
    data={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source':source.relative_to(ROOT).as_posix(),
          'strict_correct':strict,'literal_prefix_auxiliary_correct':auxiliary,
          'urban_counts_prediction1_prediction4_gold':urban,'rural_counts_prediction1_prediction4_gold':rural,
          'template':'analysis/omnimed_20261009/plot_templates/grouped_bar_template.py and style.py',
          'no_smoothing':True,'no_new_primary_scores':True}
    (folder/'rural_format_and_collapse_plot_data.json').write_bytes((json.dumps(data,indent=2)+'\n').encode('utf-8'))


if __name__=='__main__':main()
