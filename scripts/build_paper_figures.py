"""Build publication figures from checked-in results (no experiment execution).

Requires matplotlib. Run from the repository root: python scripts/build_paper_figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.unicode_minus': False})
BLUE, GRAY, ORANGE = '#236BB5', '#61676C', '#B74720'

def save(fig, name):
    fig.savefig(OUT / (name+'.pdf'), bbox_inches='tight', pad_inches=.08,
        metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(OUT / (name+'.svg'), bbox_inches='tight', pad_inches=.08)
    plt.close(fig)

# Original reported pooled values, retained from the submitted manuscript.
labels=['CWFM full','CWFM compiled only','Interaction g-computation',
        'Linear g-computation','Piecewise exposure response','Spline g-computation',
        'Random-forest g-computation']
# Values retained from the submitted plot; six effect cells, 30 seeds each.
mae=[.166,.166,.167,.169,.193,.218,.329]
coverage=[.906,.928,.900,.928,.911,.878,.911]
fig,axes=plt.subplots(2,1,figsize=(6.2,6.7),gridspec_kw={'height_ratios':[1,1]},layout='constrained')
y=np.arange(len(labels))
for ax,values,title,limit in [(axes[0],mae,'Pooled mean absolute error',(0,.40)),
                            (axes[1],coverage,'Interval coverage',(.84,.96))]:
    if ax is axes[0]:
        ax.barh(y,values,color=[BLUE,BLUE]+[GRAY]*5,height=.65)
        for i,v in enumerate(values):ax.text(v+.006,i,f'{v:.3f}',va='center',fontsize=10)
    else:
        ax.scatter(values,y,c=[BLUE,BLUE]+[GRAY]*5,s=35)
        ax.axvline(.9,color=GRAY,ls='--',lw=1)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(*limit)
    ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
save(fig,'rq1')

labels=['Hinge outcomes','Erdos-Renyi graphs','Multiplicative outcomes',
        'Saturating outcomes','Small-world graphs','Nonmonotone interference']
values=[.157,.163,.172,.307,.361,1.357]
fig,ax=plt.subplots(figsize=(6.2,3.5),layout='constrained')
ax.barh(np.arange(6),values,color=[BLUE]*5+[ORANGE],height=.62)
for i,v in enumerate(values):ax.text(v+.018,i,f'{v:.3f}',va='center',fontsize=10)
ax.set_yticks(np.arange(6),labels);ax.invert_yaxis();ax.set_xlim(0,1.57)
ax.set_xlabel('Mean absolute error');ax.axvline(.166,color=GRAY,ls='--',lw=1)
ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
save(fig,'rq4')

x=np.array([-2.272,-1.779,-1.286,-.793,-.300,.193,.686,1.179,1.672])
y0=[5.163,3.165,1.654,.629,.090,.037,.471,1.391,2.797]
y1=[5.165,3.175,1.672,.654,.121,.075,.514,1.438,2.849]
y2=[5.032,3.369,1.707,.736,.278,-.180,.584,1.533,3.033]
fig,ax=plt.subplots(figsize=(6.2,4),layout='constrained')
ax.plot(x,y0,color='black',label='True mechanism: one regime')
ax.plot(x,y1,color=BLUE,marker='s',ms=4,label='Quadratic model: one regime')
ax.plot(x,y2,color=ORANGE,ls='--',marker='^',ms=4,label='Linear model: four regimes')
for cut in [-.972,.272,1.576]:ax.axvline(cut,color=GRAY,ls=':',lw=1)
ax.set_xlabel('Covariate X1');ax.set_ylabel('Outcome mean')
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.34),frameon=False)
ax.grid(alpha=.15);save(fig,'qualnonlin')

summary=pd.read_csv(ROOT/'artifacts/cwfm/support_revision/summary.csv')
fig,axes=plt.subplots(3,1,figsize=(6.2,8.8),layout='constrained')
colors=['#222222','#0072B2','#009E73','#D55E00','#CC79A7']
markers=['o','s','^','D','v']
for ax,family,title in zip(axes,['linear_logistic','quadratic_logistic','network_ring'],
    ['(a) Linear logistic assignment','(b) Quadratic logistic assignment','(c) Network exposure on a degree-four ring']):
    for color,marker,n in zip(colors,markers,sorted(summary.n.unique())):
        data=summary[(summary.family==family)&(summary.n==n)].sort_values('severity')
        yy=data.refusal_rate.to_numpy()
        ax.errorbar(data.severity,yy,yerr=np.maximum(0, np.array([yy-data.refusal_ci_low,data.refusal_ci_high-yy])),
            color=color,marker=marker,ms=4,lw=1.2,capsize=2,label=f'n = {n}')
    ax.set_ylim(-.04,1.04);ax.set_ylabel('Refusal rate');ax.set_title(title,loc='left')
    ax.set_xlabel('Assignment slope' if family!='network_ring' else 'Treatment probability')
    ax.grid(alpha=.18)
    if family=='network_ring':ax.invert_xaxis()
axes[0].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.52),frameon=False)
save(fig,'support_audit')
