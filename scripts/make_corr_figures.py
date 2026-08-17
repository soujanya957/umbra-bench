#!/usr/bin/env python3
"""Correlation figures for results/ -- what each evaluation metric is really measuring.

Two questions, one heatmap each:

  corr_attributes_vs_metrics.png   target shape attributes (rows) x metrics (columns).
                                   Reveals which metrics are grading the *shape* rather
                                   than the *solve*.
  corr_metric_vs_metric.png        metric x metric. Reveals which metrics are duplicates.

Signs of lower-is-better metrics are flipped throughout, so blue always reads "this
metric reports a better score", which is what makes the two panels comparable at a
glance. Spearman rather than Pearson: several attributes are heavily skewed (elongation,
hole counts), and rank correlation is the honest summary there. The workbook's
`correlations` sheet uses Excel CORREL (Pearson) -- signs agree, magnitudes differ.

    python scripts/make_corr_figures.py        # reads results/master_table.csv

Diverging ramp: blue<->red with a neutral gray midpoint, red arm generated at the blue
arm's OKLab lightness so the two halves carry equal visual weight.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
BLUE=['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
RED=['#fad6d0', '#f4c3bb', '#f0afa4', '#ea9b8f', '#e48678', '#dd7263', '#d6594b', '#c7493c', '#b14034', '#9e352a', '#892c23', '#76231b', '#621c15']
MID='#f0efec'

INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'; GRID='#e1e0d9'; SURF='#fcfcfb'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':SURF,'axes.facecolor':SURF,
                     'savefig.facecolor':SURF,'text.color':INK,'axes.labelcolor':INK2})
CMAP=LinearSegmentedColormap.from_list('div', list(reversed(RED))+[MID]+BLUE, N=256)

import os
_HERE=os.path.dirname(os.path.abspath(__file__)); _BENCH=os.path.dirname(_HERE)
os.chdir(_BENCH)
m=pd.read_csv('results/master_table.csv')

ATTR_GROUPS=[
 ('size & shape',['area_frac','aspect_ratio','solidity','compactness','convexity_defect_depth_rel']),
 ('thinness',['median_stroke_width_rel','min_stroke_width_rel','thin_mass_frac','elongation',
              'stroke_width_ratio','skel_len_rel','neck_width_rel']),
 ('topology',['n_holes_signif','ph_n_holes_robust','ph_hole_max_size','hole_area_frac_max',
              'euler_number','n_components','ph_n_parts_robust','ph_h0_total',
              'ph_n_pockets_robust','ph_pocket_max_mouth','ph_entropy']),
 ('structure',['n_limbs','n_junctions','n_concave_extrema','sym_h','sym_v','contour_hf_energy']),
]
METRICS=[('iou','IoU'),('dice','Dice'),('boundary_iou','Boundary IoU'),('nsd','NSD'),
         ('cldice','clDice'),('chamfer','Chamfer'),('hd95','HD95'),('betti_error','Betti error'),
         ('pw_h0','PH-W H0'),('pw_h1','PH-W H1'),('limb_offset_rel','limb offset'),
         ('limbs_unmatched','limbs unmatched'),('hu_distance','Hu dist'),('fourier_distance','Fourier dist')]
FLIP={'chamfer','hd95','betti_error','pw_h0','pw_h1','limb_offset_rel','limbs_unmatched','hu_distance','fourier_distance'}

def corrmat(df):
    rows=[];labels=[];bands=[]
    for g,keys in ATTR_GROUPS:
        start=len(rows)
        for k in keys:
            c='attr_'+k
            if c not in df: continue
            r=[df[c].corr(df[mk],method='spearman') for mk,_ in METRICS]
            # flip sign of lower-is-better metrics so blue always = "the metric says BETTER"
            r=[(-v if mk in FLIP else v) for v,(mk,_) in zip(r,METRICS)]
            rows.append(r); labels.append(k)
        bands.append((g,start,len(rows)))
    return np.array(rows,float), labels, bands

def draw(df, title, sub, path, figw=13.2):
    A,labels,bands=corrmat(df)
    h=0.36*len(labels)+2.6
    fig,ax=plt.subplots(figsize=(figw,h))
    im=ax.imshow(A,cmap=CMAP,norm=TwoSlopeNorm(vmin=-0.8,vcenter=0,vmax=0.8),aspect='auto')
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([n for _,n in METRICS],rotation=42,ha='left',fontsize=9,color=INK2)
    ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels,fontsize=9,color=INK2)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-.5,len(METRICS),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(labels),1),minor=True)
    ax.grid(which='minor',color=SURF,linewidth=2); ax.tick_params(which='minor',length=0)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v=A[i,j]
            if np.isnan(v): continue
            ax.text(j,i,f'{v:+.2f}'.replace('+0.','.').replace('-0.','-.'),
                    ha='center',va='center',fontsize=7.4,
                    color='#ffffff' if abs(v)>0.45 else INK2)
    for g,s,e in bands:
        ax.text(-0.205, 1-((s+e)/2)/len(labels), g, transform=ax.transAxes, rotation=90,
                ha='center',va='center',fontsize=10.5,color=INK,fontweight='bold')
        if s>0: ax.axhline(s-0.5,color=MUTED,lw=1.1)
    fig.suptitle(title,x=0.010,y=0.988,ha='left',fontsize=15.5,fontweight='bold',color=INK)
    fig.text(0.010,0.952,sub,ha='left',fontsize=9.4,color=INK2,va='top')
    cb=fig.colorbar(im,ax=ax,fraction=0.021,pad=0.015,ticks=[-0.8,-0.4,0,0.4,0.8])
    cb.set_label('Spearman correlation  (blue = metric scores HIGHER as the attribute rises)',fontsize=8.5,color=INK2)
    cb.ax.tick_params(labelsize=8,color=MUTED,labelcolor=INK2); cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.20,right=0.945,top=0.845,bottom=0.012)
    fig.savefig(path,dpi=170); plt.close(fig); print('->',path,A.shape)

draw(m[m.sweep=='base-big'],
     'What is each evaluation metric actually measuring?',
     'Correlation of every target shape attribute (rows) against every evaluation metric (columns), over the 546 unfitted big-budget results.\n'
     'Signs of lower-is-better metrics are flipped, so BLUE always means "this metric reports a better score as the attribute increases".\n'
     'A metric that is white across the thinness block is judging the solve. One that is strongly coloured there is judging the shape.',
     'results/corr_attributes_vs_metrics.png')

draw(m[(m.sweep=='big-budget-fitted')&(m.ref=='shown')],
     'Same view, big-budget-fitted (ref = shown)',
     'The fitted sweep scored against the repositioned target it was actually asked to cast. Compare with the unfitted panel to see which\n'
     'relationships are properties of the optimizer rather than of the fit.',
     'results/corr_attributes_vs_metrics_fitted.png')

# ---- metric x metric redundancy ----
bb=m[m.sweep=='base-big']
names=[n for _,n in METRICS]; keys=[k for k,_ in METRICS]
R=np.zeros((len(keys),len(keys)))
for i,a in enumerate(keys):
    for j,b in enumerate(keys):
        v=bb[a].corr(bb[b],method='spearman')
        if (a in FLIP)!=(b in FLIP): v=-v
        R[i,j]=v
fig,ax=plt.subplots(figsize=(9.9,9.2))
im=ax.imshow(R,cmap=CMAP,norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect='auto')
ax.set_xticks(range(len(names))); ax.set_xticklabels(names,rotation=42,ha='left',fontsize=9,color=INK2)
ax.xaxis.set_ticks_position('top')
ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=9,color=INK2)
ax.set_xticks(np.arange(-.5,len(names),1),minor=True); ax.set_yticks(np.arange(-.5,len(names),1),minor=True)
ax.grid(which='minor',color=SURF,linewidth=2); ax.tick_params(length=0); ax.tick_params(which='minor',length=0)
for s in ax.spines.values(): s.set_visible(False)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j,i,f'{R[i,j]:+.2f}'.replace('+0.','.').replace('-0.','-.'),ha='center',va='center',
                fontsize=7.4,color='#ffffff' if abs(R[i,j])>0.55 else INK2)
fig.suptitle('How much do the metrics duplicate each other?',x=0.010,y=0.985,ha='left',fontsize=15.5,fontweight='bold',color=INK)
fig.text(0.010,0.947,'Metric-vs-metric agreement over the same 546 results. Signs aligned so blue = the two agree.\n'
                     'Blue blocks are redundant families: report one. White pairs measure different things: report both.',
         ha='left',fontsize=9.4,color=INK2,va='top')
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02,ticks=[-1,-0.5,0,0.5,1]); cb.outline.set_visible(False)
cb.set_label('Spearman correlation between metrics',fontsize=8.5,color=INK2); cb.ax.tick_params(labelsize=8,labelcolor=INK2)
fig.subplots_adjust(left=0.145,right=0.90,top=0.845,bottom=0.015)
fig.savefig('results/corr_metric_vs_metric.png',dpi=170); plt.close(fig)
print('-> results/corr_metric_vs_metric.png')
