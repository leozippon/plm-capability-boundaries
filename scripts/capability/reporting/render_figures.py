#!/usr/bin/env python3
"""Replot retained study results. No simulated observations and no model inference.

Run from the repository root: python scripts/capability/reporting/render_figures.py
PDF and SVG keep text and quantitative marks editable; PNG is a 350-dpi preview.
Original source estimates, supports and confidence intervals are never recomputed
from checkpoint-level summaries. Figure-specific CSVs record the plotted rows.

Each figure is drawn at the width it is inserted at and sets no type below 8 pt, so
printed text keeps that size. main.tex also caps the inserted height at
0.62\\textheight; a taller figure would be scaled down, type included, so no figure
may exceed HEIGHT_CAP. Layout is therefore written in printed points.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
REPO=Path(__file__).resolve().parents[3]
DATA=HERE/'figure_data'/'main'
OUT=REPO/'manuscript'/'figures'
DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
WIDTH=5.14        # inches: the 31-pc text width of the single-column Springer Nature layout
HEIGHT_CAP=4.74   # inches: 0.62\textheight, the height main.tex allows before downscaling
BASE=8.0          # pt: the smallest type any figure sets
LETTER=9.0        # pt: panel letters
BLUE='#347FA9'; ORANGE='#C97D51'; GREEN='#36957B'; PURPLE='#8370A6'; GREY='#818B94'
INK='#283B49'; PALE='#F0F4F6'; GRID='#E1E7EB'; LIGHTBLUE='#B9D5E5'
CMAP=LinearSegmentedColormap.from_list('signed_contrast',[ORANGE,'#FFFFFF',BLUE])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':BASE,'axes.titlesize':BASE,
 'axes.labelsize':BASE,'xtick.labelsize':BASE,'ytick.labelsize':BASE,'legend.fontsize':BASE,
 'axes.linewidth':.65,'xtick.major.width':.65,'ytick.major.width':.65,
 'xtick.major.size':2.6,'ytick.major.size':2.6,'xtick.major.pad':2,'ytick.major.pad':2,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':INK,
 'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,
 'savefig.facecolor':'white','figure.facecolor':'white','mathtext.default':'regular'})
GEN=['protgpt2','progen2-small','progen2-base','progen2-medium','progen2-large','progen2-xlarge',
 'progen3-112m','progen3-3b','protgpt3-1.3b','rita-xl','galactica-125m','galactica-1.3b',
 'galactica-6.7b','galactica-30b','llama-2-7b','prollama-stage-1','prollama','proteinglm-7b-clm','instructprotein']
TXT=['gpt2','gpt2-medium','gpt2-large','gpt2-xl','dialogpt-small','bygpt5-small-en','bygpt5-base-en',
 'bygpt5-medium-en','qwen2.5-0.5b','qwen2.5-0.5b-instruct','qwen2.5-7b','qwen2.5-32b','qwen3-8b-base','llama-3.2-3b']
ORDER=GEN+TXT
PROTEIN=set(GEN)-{'galactica-125m','galactica-1.3b','galactica-6.7b','galactica-30b','llama-2-7b'}
LABEL={'protgpt2':'ProtGPT2','protgpt3-1.3b':'ProtGPT3-1.3B','llama-2-7b':'Llama-2-7B',
 'prollama-stage-1':'ProLLaMA S1','prollama':'ProLLaMA S2','proteinglm-7b-clm':'ProteinGLM-7B',
 'instructprotein':'InstructProtein','rita-xl':'RITA-xl','dialogpt-small':'DialoGPT-small',
 'llama-3.2-3b':'Llama-3.2-3B','qwen3-8b-base':'Qwen3-8B-base','zymctrl':'ZymCTRL'}
for k,prefix in [('progen2','ProGen2'),('progen3','ProGen3'),('galactica','Galactica'),('qwen2.5','Qwen2.5'),('bygpt5','ByGPT5')]:
 for size in ['small','base','medium','large','xlarge','112m','125m','1.3b','3b','6.7b','30b','0.5b','7b','32b']:
  LABEL[f'{k}-{size}'+('-en' if k=='bygpt5' else '')]=prefix+'-'+(size.upper() if size[0].isdigit() else size)
for sz in ['','-medium','-large','-xl']:LABEL['gpt2'+sz]='GPT-2'+sz
LABEL['qwen2.5-0.5b-instruct']='Qwen2.5-0.5B-it'
PRIMARY=20260923; provenance={}; contracts=[]; checks=[]
PAGE=[WIDTH*72,HEIGHT_CAP*72]

def read(name):
 p=DATA/(name+'.csv');provenance[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 return pd.read_csv(p)
def rec(df,**kw):
 q=df
 for k,v in kw.items():q=q[q[k]==v]
 if len(q)!=1:raise ValueError((kw,len(q)))
 return q.iloc[0]
def col(arm):return BLUE if arm in PROTEIN else ORANGE
def label(a):return LABEL.get(a,a)

# Placement is in printed points measured from the bottom-left corner, because what
# has to work is the printed size of each label relative to the panel holding it.
def fig(height):
 if height>HEIGHT_CAP+1e-9:raise ValueError(f'{height} in exceeds the {HEIGHT_CAP} in insertion height cap')
 PAGE[1]=height*72
 return plt.figure(figsize=(WIDTH,height))
def fx(x):return x/PAGE[0]
def fy(y):return y/PAGE[1]
def at(x,y,w,h):return [fx(x),fy(y),fx(w),fy(h)]
def ax(f,rect,xlabel='',ylabel=''):
 a=f.add_axes(at(*rect));a.spines[['top','right']].set_visible(False);a.set_axisbelow(True)
 if xlabel:a.set_xlabel(xlabel,labelpad=2)
 if ylabel:a.set_ylabel(ylabel,labelpad=2)
 return a
def head(f,x,y,letter,title='',color=INK):
 """A panel letter and its title on one baseline, placed independently of the axes."""
 f.text(fx(x),fy(y),letter,weight='bold',fontsize=LETTER,ha='left',va='baseline')
 if title:f.text(fx(x+14),fy(y),title,ha='left',va='baseline',color=color)
def canvas(f,x,y,w,h):
 """An invisible axes whose data units are printed points, for schematics."""
 a=f.add_axes(at(x,y,w,h));a.set_axis_off();a.set_xlim(0,w);a.set_ylim(0,h);return a
def scale(f,rect,norm,ticks,orientation='vertical'):
 cax=f.add_axes(at(*rect))
 cb=plt.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),cax=cax,orientation=orientation)
 cb.set_ticks(ticks);cb.outline.set_visible(False);cax.tick_params(length=2)
 return cb
def zero(a,orientation='v'):
 (a.axvline if orientation=='v' else a.axhline)(0,color=GREY,lw=.7,ls=(0,(3,2)),zorder=0)
def interval(a,x,y,lo,hi,color=BLUE,orientation='h',filled=None,ms=3.6,alpha=1):
 if not np.isfinite([x,y,lo,hi]).all():raise ValueError('nonfinite interval')
 p=x if orientation=='h' else y
 assert lo<=p+1e-9 and p<=hi+1e-9,(lo,p,hi)
 if filled is None:filled=lo>0 or hi<0
 if orientation=='h':
  a.plot([lo,hi],[y,y],color=color,lw=.9,alpha=alpha,zorder=2)
  a.plot([lo,lo,hi,hi],[y-.045,y+.045,y-.045,y+.045],ls='',marker='|',color=color,ms=3,alpha=alpha)
  a._ranges=getattr(a,'_ranges',[])+[('x',lo,hi)]
 else:
  a.plot([x,x],[lo,hi],color=color,lw=.8,alpha=alpha,zorder=2)
  a._ranges=getattr(a,'_ranges',[])+[('y',lo,hi)]
 a.plot(x,y,'o',ms=ms,mec=color,mfc=color if filled else 'white',mew=.85,alpha=alpha,zorder=3)
def box(a,xy,w,h,title,detail='',color=BLUE):
 a.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=1.4,rounding_size=2.5',lw=.9,ec=color,fc='white'))
 a.text(xy[0]+w/2,xy[1]+h*(.72 if detail else .5),title,ha='center',va='center',weight='bold',color=color,linespacing=1.3)
 if detail:a.text(xy[0]+w/2,xy[1]+h*.27,detail,ha='center',va='center',linespacing=1.3)
def arrow(a,start,end,color=GREY):
 a.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=8,lw=.8,color=color,connectionstyle='arc3,rad=0'))
def note(f,x,y,text,color=GREY,ha='left'):
 f.text(fx(x),fy(y),text,color=color,ha=ha,va='baseline',linespacing=1.3)
def rotated(a,positions,labels):
 a.set_xticks(positions,labels,rotation=90,ha='center',va='top');a.tick_params(axis='x',length=0)
def save(f,name,rows=None):
 f.canvas.draw()
 for a in f.axes:
  for axis,lo,hi in getattr(a,'_ranges',[]):
   lim=sorted(a.get_xlim() if axis=='x' else a.get_ylim())
   assert lim[0]-1e-9<=lo<=hi<=lim[1]+1e-9,(name,axis,lo,hi,lim)
 f.savefig(OUT/f'{name}.pdf')
 for ext in ['svg','png']:f.savefig(DATA/f'{name}.{ext}',dpi=350)
 if rows is not None:rows.to_csv(DATA/f'{name}-plotted.csv',index=False)
 checks.append({'figure':name,'all_retained_intervals_inside_axes':True,'width_inches':float(f.get_figwidth()),
  'height_inches':float(f.get_figheight()),'minimum_font_points':BASE,
  'within_insertion_height_cap':float(f.get_figheight())<=HEIGHT_CAP+1e-9})
 plt.close(f)

def panel_sources(figure,panel,source,selection,unit,uncertainty):
 contracts.append(dict(figure=figure,panel=panel,source=source,selection=selection,statistical_unit=unit,uncertainty=uncertainty))

# FIGURE 1: framework -> release-family result -> distinct-source check.
local=read('local-increments-source');ext=read('external-abundance-source');family=read('lineage-robustness-source')
f=fig(4.74)
head(f,4,331,'a','A common measurement framework, three distinct model outputs')
a=canvas(f,0,232,370,96)
box(a,(4,32),96,58,'34 frozen\ncheckpoints','19 native\n14 text-string\n1 enzyme-conditioned',INK)
a.text(52,14,'Task-specific\ninterface eligibility',ha='center',va='center',color=GREY,linespacing=1.3)
for y,title,detail,c in [(64,'Native likelihood  M','Score supplied variants',BLUE),
                         (32,'Frozen states  R','Fit supervised readouts',PURPLE),
                         (0,'Generation  G','Evaluate every attempt',GREEN)]:
 box(a,(112,y),101,28,title,detail,c);arrow(a,(101,61),(110,y+14))
for y,title,detail,c in [(64,'Ranking, stability, interaction','Profile + local controls',BLUE),
                         (32,'Supervised prediction','Matched labels + held families',PURPLE),
                         (0,'Generated products','Length-matched generators',GREEN)]:
 box(a,(225,y),141,28,title,detail,c);arrow(a,(215,y+14),(223,y+14))
note(f,185,222,'Paired comparison  →  biological-group uncertainty  →  task-specific inference',INK,ha='center')
# Family intervals, predeclared alphabetical order inside groups.
fam=pd.concat([family[family.protein_pretraining=='yes'].sort_values('release_family'),
               family[family.protein_pretraining=='no'].sort_values('release_family')]).reset_index(drop=True)
head(f,4,206,'b','Release-family likelihood increments')
b=ax(f,(62,46,102,156),'Δ Spearman ρ')
b.axhspan(-.5,7.5,color='#EEF5F9',zorder=0);zero(b)
for i,r in fam.iterrows():interval(b,r.estimate,i,r.ci_low,r.ci_high,BLUE if r.protein_pretraining=='yes' else ORANGE,ms=3.2)
b.set_yticks(range(16),fam.release_family.replace({'GPT2':'GPT-2','Llama2':'Llama-2','Llama3':'Llama-3'}))
b.tick_params(axis='y',length=0)
b.set_ylim(15.7,-.7);b.set_xlim(-.0028,.036);b.set_xticks([0,.01,.02,.03]);b.grid(axis='x',color=GRID,lw=.55)
b.text(.99,.02,'Positive at 95%:\nprotein-pretrained 8/8\nother families 0/8',transform=b.transAxes,
       ha='right',va='bottom',color=BLUE,linespacing=1.3)
# Cross-task scatter: pointwise intervals on different populations, no fitted regression.
m=local[(local.endpoint=='increment_M_C_P_wall')&(local.split_seed==PRIMARY)]
e=ext[(ext.endpoint=='primary_likelihood_spearman')&(ext.split_seed==PRIMARY)]
j=m.merge(e,on='arm',suffixes=('_mutation','_abundance'))
head(f,176,206,'c','A distinct-source ranking check')
c=ax(f,(212,118,154,84),'Mutation-ranking Δρ','Abundance Δρ')
zero(c);zero(c,'h')
for _,r in j.iterrows():
 color=col(r.arm)
 c.plot([r.ci_low_mutation,r.ci_high_mutation],[r.estimate_abundance]*2,color=color,lw=.5,alpha=.25)
 c.plot([r.estimate_mutation]*2,[r.ci_low_abundance,r.ci_high_abundance],color=color,lw=.5,alpha=.25)
 c._ranges=getattr(c,'_ranges',[])+[('x',r.ci_low_mutation,r.ci_high_mutation),('y',r.ci_low_abundance,r.ci_high_abundance)]
 c.plot(r.estimate_mutation,r.estimate_abundance,'o' if r.arm in PROTEIN else 's',mfc=color,mec='white',mew=.35,ms=4,zorder=3)
c.set_xlim(-.004,.035);c.set_ylim(-.0018,.0125);c.set_xticks([0,.01,.02,.03]);c.set_yticks([0,.005,.01]);c.grid(color=GRID,lw=.45)
c.text(.03,.97,'33 checkpoints, pointwise 95%\n163 mutation / 96 abundance groups',transform=c.transAxes,
       ha='left',va='top',color=GREY,linespacing=1.3)
# Control gain is empirical, not an invented grand score.
head(f,176,86,'d','Qualifying the local baseline')
d=ax(f,(248,48,118,32),'Held-family Spearman ρ')
for i,ep in enumerate(['C_P_spearman','C_P_wall_spearman']):
 r=rec(local,arm='progen3-3b',endpoint=ep,split_seed=PRIMARY)
 interval(d,r.estimate,i,r.ci_low,r.ci_high,GREY if i==0 else BLUE,filled=True,ms=4)
 for _,q in local[(local.arm=='progen3-3b')&(local.endpoint==ep)&(local.split_seed!=PRIMARY)].iterrows():d.plot(q.estimate,i,'|',color=INK,ms=5)
d.set_yticks([0,1],['Profile + sequence','+ local windows']);d.tick_params(axis='y',length=0)
d.set_ylim(1.55,-.6);d.set_xlim(.39,.535);d.set_xticks([.4,.45,.5]);d.grid(axis='x',color=GRID,lw=.45)
note(f,4,14,'Simultaneous 95% bands in b; 163 biological groups. Release families are not')
note(f,4,4,'independent ancestries, and c compares distinct populations, not paired phenotypes.')
save(f,'Fig1_framework',pd.concat([fam.assign(panel='b'),j.assign(panel='c')],ignore_index=True))
panel_sources('Fig1','a','manuscript/main.tex; retained panel design','conceptual workflow','not a quantitative panel','none')
panel_sources('Fig1','b','lineage-robustness-source.csv','all 16 release families','163 biological groups','95% simultaneous; 10000 resamples')
panel_sources('Fig1','c','local-increments-source.csv; external-abundance-source.csv','primary split; likelihood ranking increments','163 mutation groups; 96 abundance groups','pointwise 95%; 2000 resamples; no across-model regression')
panel_sources('Fig1','d','local-increments-source.csv','shared baseline from progen3-3b record','163 biological groups','primary 95% intervals; other-split ticks')

# FIGURE 2: native score vs comparator, then term decomposition.
native=read('native-fitness-source');text=read('text-aa-fitness-source');pos=read('position-terms-replot')
f=fig(4.74)
head(f,4,331,'a','Partition a likelihood difference into own and downstream terms')
a=canvas(f,0,302,370,24)
for y,mut,tag in [(17,False,'WT'),(4,True,'Mutant')]:
 a.text(4,y,tag,weight='bold',va='center')
 for i,resi in enumerate('M A D V P Q P S K R A'.split()):
  a.text(46+i*11.5,y,('L' if mut and i==4 else resi),ha='center',va='center',
         color=ORANGE if i==4 else (BLUE if i>4 else GREY),weight='bold' if i==4 else 'normal')
a.text(176,17,'M = Δ log p(x)',color=BLUE,va='center')
a.text(242,17,'Own term',color=ORANGE,va='center');a.text(290,17,'Downstream terms',color=BLUE,va='center')
a.text(176,4,'Separate single-substitution support',color=GREY,va='center')
# Both strata share raw-score and contrast colour scales; no subtraction across cohorts.
TOP=288;ROWS=8.6
for data,order0,raw,lx,gut,fw,hx,letter,title in [
 (native,GEN,'raw_spearman',4,73,66,145,'b','Native protein-sequence interfaces'),
 (text,TXT,'raw',180,66,64,312,'c','Literal amino-acid strings')]:
 arms=[x for x in order0 if x in set(data.arm)];h=ROWS*len(arms)
 head(f,lx,292,letter,title)
 ar=ax(f,(lx-2+gut,TOP-h,fw,h),'Native-score ρ')
 zero(ar);ar.set_yticks(range(len(arms)),[label(x) for x in arms]);ar.tick_params(axis='y',length=0)
 ar.set_ylim(len(arms)-.5,-.5);ar.set_xlim(-.06,.46);ar.set_xticks([0,.2,.4]);ar.grid(axis='x',color=GRID,lw=.45)
 for i,arm in enumerate(arms):
  rs=data[(data.arm==arm)&(data.endpoint==raw)]
  if len(rs):
   r=rs.iloc[0];interval(ar,r.estimate,i,r.ci_low,r.ci_high,BLUE if data is native else ORANGE,ms=3)
  else:ar.text(.075,i,'NR',color=GREY,va='center')
 vals=np.array([[rec(data,arm=arm,endpoint=ep).estimate for ep in ['model_minus_lookup','model_minus_blosum62']] for arm in arms])
 hm=f.add_axes(at(hx,TOP-h,21,h))
 hm.imshow(vals,cmap=CMAP,norm=TwoSlopeNorm(vmin=-.40,vcenter=0,vmax=.40),aspect='auto',interpolation='none')
 hm.set_yticks([]);rotated(hm,[0,1],['Profile','BLOSUM'])
 for i,arm in enumerate(arms):
  for k,ep in enumerate(['model_minus_lookup','model_minus_blosum62']):
   r=rec(data,arm=arm,endpoint=ep)
   if r.ci_low>0 or r.ci_high<0:hm.plot(k,i,'o',color='white' if abs(r.estimate)>.23 else INK,ms=1.9)
 for spine in hm.spines.values():spine.set_visible(False)
note(f,364,292,'Paired Δρ',INK,ha='right')
scale(f,(337,TOP-ROWS*14,7,ROWS*14-8),TwoSlopeNorm(vmin=-.4,vcenter=0,vmax=.4),[-.4,0,.4])
# Every position contrast; count statistics computed from exactly these data.
w=pos[pos.control=='wall'];arms=[x for x in ORDER if x in set(w.arm)]
keys=['unique_own','unique_downstream','joint_over_full','own_minus_downstream']
v=np.array([[rec(w,arm=arm,endpoint=k).estimate for arm in arms] for k in keys])
head(f,4,123,'d','Position-term increments under the qualified window control')
d=f.add_axes(at(84,83,282,36))
d.imshow(v,cmap=CMAP,norm=TwoSlopeNorm(vmin=-.02,vcenter=0,vmax=.02),aspect='auto',interpolation='none')
d.set_yticks(range(4),['Own | downstream','Downstream | own','Both | full score','Own − downstream'])
d.tick_params(axis='y',length=0);rotated(d,range(len(arms)),[label(x) for x in arms])
for i,k in enumerate(keys):
 for jj,arm in enumerate(arms):
  r=rec(w,arm=arm,endpoint=k)
  if r.ci_low>0 or r.ci_high<0:d.plot(jj,i,'o',ms=2.6,color=INK)
for spine in d.spines.values():spine.set_visible(False)
scale(f,(10,30,7,46),TwoSlopeNorm(vmin=-.02,vcenter=0,vmax=.02),[-.02,0,.02])
note(f,4,4,'Dots mark 95% interval exclusion of zero; NR: not retained.')
save(f,'Fig2_native_and_position',pd.concat([native.assign(panel='b'),text.assign(panel='c'),w.assign(panel='d')],ignore_index=True))
panel_sources('Fig2','b/c','native-fitness-source.csv; text-aa-fitness-source.csv','all retained native and text-string records; missing raw ProteinGLM is NR','native supports vary; text 169 groups','pointwise 95%; heatmap dots only encode interval exclusion')
panel_sources('Fig2','d','position-terms-replot.csv','wall control; 32 checkpoints; four contrasts','156 families; 2991 variants','95% simultaneous over 128 contrasts; 10000 resamples')

# FIGURE 3: compare readout claims, without disguising fitted-readout selection as raw biological data.
rout=read('readout-contrasts-source');gaps=read('controlled-gap-source');depth=read('depth-profile-source');ctx=read('context-ranking-source')
f=fig(4.74)
head(f,4,331,'a','Readout advantage depends on the baseline')
a=ax(f,(44,239,120,86),ylabel='Change in Spearman ρ')
zero(a,'h');a.set_xlim(-.18,1.18);a.set_ylim(-.095,.215)
a.set_xticks([0,1],['Readout −\nnative score','Added representation\n(matched supervision)'])
a.tick_params(axis='x',length=0);a.grid(axis='y',color=GRID,lw=.5)
for _,r in rout.iterrows():
 a.plot([0,1],[r.readout_minus_likelihood,r.representation_increment],color=col(r.arm),lw=.7,alpha=.38,zorder=1)
 a.plot(0,r.readout_minus_likelihood,'o',ms=3,mfc='white',mec=col(r.arm),mew=.7)
 a.plot(1,r.representation_increment,'o',ms=3,mfc=col(r.arm),mec='white',mew=.35)
a.text(.04,.97,'33 paired checkpoints',transform=a.transAxes,color=GREY,va='top')
for arm,dy in [('proteinglm-7b-clm',.018),('progen3-3b',-.018)]:
 r=rec(rout,arm=arm)
 a.annotate(label(arm),(1,r.representation_increment),(.60,r.representation_increment+dy),
            arrowprops=dict(arrowstyle='-',color=GREY,lw=.5),ha='right',va='center')
head(f,202,331,'b','Depth: a separate exploratory axis')
b=ax(f,(230,239,136,86),'Relative transformer depth','Increment range (Δρ)')
zero(b,'h')
for axis,c,marker in [('pooled',BLUE,'o'),('position_resolved',PURPLE,'s')]:
 ds=depth[(depth.axis==axis)&(depth.interface=='native_protein')]
 for _,r in ds.iterrows():
  b.plot([r.relative_depth]*2,[r.increment_min,r.increment_max],color=c,lw=1,alpha=.8)
  b.plot(r.relative_depth,r.increment_min,marker,ms=3.1,mec=c,mfc='white',mew=.8)
b.set_xlim(-.035,1.055);b.set_ylim(-.003,.060);b.set_xticks([0,.5,1]);b.set_yticks([0,.02,.04,.06]);b.grid(color=GRID,lw=.5)
b.legend(handles=[Line2D([0],[0],marker='o',ls='-',color=BLUE,label='Pooled'),
                  Line2D([0],[0],marker='s',ls='-',color=PURPLE,label='Position-resolved')],
         frameon=False,loc='upper left',handletextpad=.4,borderpad=.1,labelspacing=.25)
note(f,366,208,'Ranges span three split estimates,',ha='right')
note(f,366,196,'not confidence intervals',ha='right')
# Split-specific gap heatmap, preserving all 99 estimates.
ars=[x for x in ORDER if x in set(gaps.arm)];seeds=sorted(gaps.split_seed.unique())
v=np.array([[rec(gaps,arm=arm,split_seed=s).estimate for arm in ars] for s in seeds])
head(f,4,184,'c','Representation minus likelihood over one composition + profile baseline')
c=f.add_axes(at(42,146,324,30))
c.imshow(v,cmap=CMAP,norm=TwoSlopeNorm(vmin=-.022,vcenter=0,vmax=.022),aspect='auto',interpolation='none')
c.set_yticks(range(3),['Split 1','Split 2','Split 3']);c.tick_params(axis='y',length=0)
rotated(c,range(len(ars)),[label(x) for x in ars])
for i,s in enumerate(seeds):
 for jj,arm in enumerate(ars):
  r=rec(gaps,arm=arm,split_seed=s)
  if r.ci_low>0 or r.ci_high<0:c.plot(jj,i,'o',color=INK,ms=2.0)
for spine in c.spines.values():spine.set_visible(False)
scale(f,(6,94,7,46),TwoSlopeNorm(vmin=-.022,vcenter=0,vmax=.022),[-.02,0,.02])
# Bottom context forest is small but interpretable and retains the distinct support.
head(f,176,68,'d','Related context: rescue versus net benefit')
d=ax(f,(244,28,116,36),'Mutation-ranking Δρ')
ctxarms=['progen3-3b','proteinglm-7b-clm','prollama-stage-1','prollama']
for i,arm in enumerate(ctxarms):
 for ep,off,clr in [('delta_spearman',-.16,BLUE),('homolog_minus_no_context_spearman',.16,GREEN)]:
  r=rec(ctx,arm=arm,endpoint=ep);interval(d,r.estimate,i+off,r.ci_low,r.ci_high,clr,ms=3)
d.set_yticks(range(4),[label(x) for x in ctxarms]);d.tick_params(axis='y',length=0)
d.set_ylim(3.6,-.6);d.set_xlim(-.17,.20);d.set_xticks([-.1,0,.1,.2]);zero(d);d.grid(axis='x',color=GRID,lw=.5)
note(f,4,56,'In d — blue: homolog − unrelated;',BLUE)
note(f,4,44,'green: homolog − no context;',GREEN)
note(f,4,32,'164 assays in 134 groups.')
note(f,4,8,'Dots in c mark pointwise 95% exclusion of zero.')
save(f,'Fig3_readouts_and_context',pd.concat([rout.assign(panel='a'),depth.assign(panel='b'),gaps.assign(panel='c'),ctx.assign(panel='d')],ignore_index=True))
panel_sources('Fig3','a','readout-contrasts-source.csv','all 33 checkpoint summaries; primary split','163 biological groups','point estimates connected by checkpoint; CI in source table')
panel_sources('Fig3','b','depth-profile-source.csv','native-protein records; both admitted summary axes','selected checkpoint-depth combinations','ranges over 3 split point estimates; not confidence intervals; post-selection')
panel_sources('Fig3','c','controlled-gap-source.csv','all 99 arm × split estimates','163 biological groups','pointwise 95%; heatmap markers, complete CI in CSV')
panel_sources('Fig3','d','context-ranking-source.csv','all 4 checkpoints × 2 contrasts','134 biological groups','pointwise 95%; 2000 resamples')

# FIGURE 4: target definition, interaction/stability evidence, endpoint qualification.
inter=read('interaction-panel-source');nested=read('nested-residual-source');resid=read('residual-full-panel-source')
bound=read('endpoint-boundaries-source');stab=read('stability-panel-source')
f=fig(4.74)
head(f,4,331,'a','The four-state interaction target')
a=canvas(f,0,260,170,68)
for xy,title in [((6,50),'WT'),((96,50),'A'),((6,18),'B'),((96,18),'AB')]:box(a,xy,44,17,title,'',BLUE)
for start,end in [((52,58),(94,58)),((52,26),(94,26)),((28,48),(28,37)),((118,48),(118,37))]:arrow(a,start,end)
a.text(73,2,'$\\epsilon=y_{AB}-y_{A}-y_{B}+y_{WT}$',ha='center',va='baseline',fontsize=11.5)
head(f,176,331,'b','Historical positives versus adjusted targets')
b=ax(f,(232,292,134,36),'MSE reduction (kcal²/mol²)')
selected=['protgpt2','progen3-3b','prollama']
for i,arm in enumerate(selected):
 r=rec(nested,arm=arm,target='historical_raw',split_seed=PRIMARY)
 interval(b,r.estimate,i-.19,r.ci_low,r.ci_high,ORANGE,ms=3.4)
 r=rec(resid,arm=arm);interval(b,r.estimate,i+.19,r.ci_low,r.ci_high,GREEN,ms=3.4)
b.set_yticks(range(3),[label(x) for x in selected]);b.tick_params(axis='y',length=0)
b.set_ylim(2.5,-.55);b.set_xlim(-.0005,.0025);b.set_xticks([0,.001],['0','0.001'])
zero(b);b.grid(axis='x',color=GRID,lw=.5)
# Full panels share checkpoint order but NOT y units/ranges.
ars=[x for x in ORDER if x in set(resid.arm)]
for data,rect,ly,letter,title,clr,limits,yticks,tlabels in [
 (resid,(30,204,336,42),249,'c','Adjusted interactions: every simultaneous interval includes zero (kcal²/mol²)',
  GREEN,(-.0018,.0055),[0,.002,.004],['0','0.002','0.004']),
 (stab,(30,150,336,42),196,'d','Single-substitution stability: two positive increments (kcal²/mol²)',
  BLUE,(-.008,.034),[0,.01,.02,.03],['0','0.01','0.02','0.03'])]:
 head(f,4,ly,letter,title)
 ar=ax(f,rect)
 for i,arm in enumerate(ars):
  r=rec(data,arm=arm);interval(ar,i,r.estimate,r.ci_low,r.ci_high,clr,orientation='v',ms=3.2)
 ar.set_xlim(-.8,len(ars)-.2);ar.set_ylim(*limits);ar.set_yticks(yticks,tlabels)
 zero(ar,'h');ar.grid(axis='y',color=GRID,lw=.5)
 rotated(ar,range(len(ars)),[label(x) for x in ars] if letter=='d' else ['']*len(ars))
# Different biological endpoints remain distinct; contact and third-order are not model failures.
head(f,4,68,'e','Endpoint agreement: shared / discordance')
e=ax(f,(72,24,108,40))
bounds=bound[bound.family=='reliability']
for i,(_,r) in enumerate(bounds.iterrows()):interval(e,r.estimate,i,r.ci_low,r.ci_high,BLUE if i<2 else GREY,ms=3)
e.set_yticks(range(4),['Stability singles','Double cycles','HIS3 order 2','HIS3 order 3']);e.tick_params(axis='y',length=0)
e.set_ylim(3.6,-.6);e.set_xlim(-.12,4.4);e.set_xticks([0,2,4]);zero(e);e.grid(axis='x',color=GRID,lw=.5)
head(f,196,68,'f','Contact − control (kcal/mol)')
ff=ax(f,(266,24,100,40))
for i,ep in enumerate(['mean_abs_epsilon','mean_abs_epsilon_adjusted']):
 r=rec(bound,family='contact_enrichment',endpoint=ep);interval(ff,r.estimate,i,r.ci_low,r.ci_high,ORANGE if i==0 else GREEN,ms=3.4)
ff.set_yticks([0,1],['Raw','Adjusted']);ff.tick_params(axis='y',length=0)
ff.set_ylim(1.7,-.7);ff.set_xlim(-.14,.46);ff.set_xticks([0,.2,.4]);zero(ff);ff.grid(axis='x',color=GRID,lw=.5)
note(f,4,4,'Measured singles are shared with the adjustment; orange pointwise, green simultaneous.')
save(f,'Fig4_interactions_and_limits',pd.concat([resid.assign(panel='c'),stab.assign(panel='d'),bound.assign(panel='e/f'),nested.assign(panel='b')],ignore_index=True))
panel_sources('Fig4','b','nested-residual-source.csv; residual-full-panel-source.csv','3 historically positive arms; raw primary vs current seed-mean','64 biological groups','raw pointwise 95%; adjusted simultaneous 95% over all 33 arms; not a paired target-change test')
panel_sources('Fig4','c/d','residual-full-panel-source.csv; stability-panel-source.csv','all 33 checkpoints; fixed order','64 interaction groups; 101 stability groups','95% simultaneous across 33 arms; 10000 resamples')
panel_sources('Fig4','e/f','endpoint-boundaries-source.csv','four ratios; raw and adjusted primary contact enrichment','units differ by endpoint, retained in CSV','pointwise 95%; 2000 resamples')

# FIGURE 5: all-attempt fractions, comparator heatmap, campaign replication and class control.
gen=read('matched-generator-source');streams=read('generation-streams-replot');conditional=read('conditional-class-yield-source')
g=gen[(gen.endpoint=='complete_domain')&(gen.cohort=='fragment')].copy()
cells=[]
for arm in GEN+['zymctrl']:
 cells+=sorted(g[g.arm==arm].cell.tolist(),key=lambda c:('requested' not in c,c))
assert len(cells)==20
celllabels={c:label(rec(g,cell=c).arm)+('†' if 'requested' in c else '') for c in cells}
f=fig(4.3)
head(f,4,299,'a','Every attempt stays in the denominator')
a=canvas(f,0,260,370,38)
for x,w,title,detail,c in [(4,80,'16,000 attempts','20 cells × 800',INK),
                           (97,116,'Any-family recognition','An assigned curated profile',BLUE),
                           (226,139,'Complete-domain recognition','≥80% coverage of one profile',GREEN)]:
 box(a,(x,4),w,30,title,detail,c)
arrow(a,(86,19),(95,19));arrow(a,(215,19),(224,19))
# Stacked fractions are mutually exclusive and exactly sum to one.
head(f,4,252,'b','Recognition across all attempts')
b=ax(f,(75,76,77,172),'Fraction of all attempts')
for i,cell in enumerate(cells):
 complete=rec(g,cell=cell);anyr=rec(gen,cell=cell,endpoint='any_family',cohort='fragment')
 assert 0<=complete.model_rate<=anyr.model_rate<=1
 b.barh(i,1,color=PALE,height=.78,zorder=0)
 b.barh(i,complete.model_rate,color=GREEN,height=.78)
 b.barh(i,anyr.model_rate-complete.model_rate,left=complete.model_rate,color=LIGHTBLUE,height=.78)
 b.plot(complete.control_rate,i,'|',color=INK,ms=6,mew=1.1)
b.set_yticks(range(20),[celllabels[c] for c in cells]);b.tick_params(axis='y',length=0)
b.set_ylim(19.6,-.7);b.set_xlim(0,1.025);b.set_xticks([0,.5,1]);b.grid(axis='x',color=GRID,lw=.4)
# Six constructed controls; unqualified order-0/order-2 controls are hatched, never used for verdicts.
cohorts=['shuffle','hydropathy','markov_0','markov_2','markov_4','fragment']
v=np.array([[rec(gen,cell=c,endpoint='complete_domain',cohort=k).difference for k in cohorts] for c in cells])
head(f,150,252,'c','Gain over each control')
c=f.add_axes(at(158,76,72,172))
c.imshow(v,cmap=CMAP,norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),aspect='auto',interpolation='none')
rotated(c,range(6),['Shuffle','Hydropathy','Order 0','Order 2','Order 4','Fragment'])
c.set_yticks([]);c.tick_params(axis='y',length=0)
for i,cell in enumerate(cells):
 for jj,k in enumerate(cohorts):
  r=rec(gen,cell=cell,endpoint='complete_domain',cohort=k)
  if not bool(r.qualified):c.add_patch(Rectangle((jj-.48,i-.48),.96,.96,fill=False,hatch='///',ec=GREY,lw=0))
  if r.ci_low>0 or r.ci_high<0:c.plot(jj,i,'o',color='white' if abs(r.difference)>.45 else INK,ms=2.1)
c.add_patch(Rectangle((4.5,-.5),1,20,fill=False,ec=INK,lw=1.2))
for s in c.spines.values():s.set_visible(False)
scale(f,(22,36,96,6),TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1),[-.8,0,.8],orientation='horizontal')
# Gate vs stream check: different interval types visibly identified.
head(f,258,252,'d','Stream replication')
d=ax(f,(266,170,94,78),'Retained-stream\ndifference','Three-stream mean')
zero(d);zero(d,'h');d.plot([-.75,.7],[-.75,.7],color=GREY,lw=.7,ls=':')
for cell in cells:
 r=rec(g,cell=cell);s=rec(streams,cell=cell,endpoint='complete_domain');clr=GREEN if s.ci_low>0 else GREY
 d.plot([r.ci_low,r.ci_high],[s.estimate]*2,color=clr,lw=.7,alpha=.55)
 d.plot([r.difference]*2,[s.ci_low,s.ci_high],color=clr,lw=.8,alpha=.8)
 d._ranges=getattr(d,'_ranges',[])+[('x',r.ci_low,r.ci_high),('y',s.ci_low,s.ci_high)]
 d.plot(r.difference,s.estimate,'o',ms=3.6,mec=clr,mfc=clr if s.ci_low>0 or s.ci_high<0 else 'white',mew=.8)
d.set_xlim(-.77,.70);d.set_ylim(-.77,.77);d.set_xticks([-.5,0,.5]);d.set_yticks([-.5,0,.5]);d.grid(color=GRID,lw=.4)
d.text(.5,.99,'Green: 3/3 positive',transform=d.transAxes,ha='center',va='top',color=GREEN)
# Conditional-class points (true class observations, no inferred DMS distributions).
head(f,258,128,'e','Condition specificity')
e=ax(f,(266,46,94,78),'Requested\ntarget-profile rate','Mismatched rate')
e.plot([0,1],[0,1],color=GREY,lw=.7,ls=':')
for arm,clr,marker in [('zymctrl',GREEN,'o'),('prollama',PURPLE,'s')]:
 ds=conditional[(conditional.arm==arm)&conditional.primary_class]
 for k in ds.class_key.unique():
  r=rec(ds,class_key=k,condition='requested');q=rec(ds,class_key=k,condition='mismatched')
  e.plot(r.target_profile_rate,q.target_profile_rate,marker,ms=4,mec=clr,mfc=clr,alpha=.75)
e.set_xlim(-.035,1.03);e.set_ylim(-.035,1.03);e.set_xticks([0,.5,1]);e.set_yticks([0,.5,1]);e.grid(color=GRID,lw=.4)
e.legend(handles=[Line2D([0],[0],marker='o',ls='',color=GREEN,label='ZymCTRL (14)'),
                  Line2D([0],[0],marker='s',ls='',color=PURPLE,label='ProLLaMA (15)')],
         frameon=False,loc='upper left',handletextpad=.3,borderpad=.1,labelspacing=.2)
note(f,4,16,'In b — green: complete domain; pale blue: other')
note(f,4,4,'profile hit; gray: unrecognized. † requested.')
save(f,'Fig5_generation',pd.concat([gen.assign(panel='b/c'),streams.assign(panel='d'),conditional.assign(panel='e')],ignore_index=True))
panel_sources('Fig5','b/c','matched-generator-source.csv','all 20 cells; both endpoints; six constructed controls; hatched order-0/order-2 controls fail qualification','frozen attempt groups; 800 attempts per cell','97.5% paired group intervals; complete differences in source; dots encode exclusion')
panel_sources('Fig5','d','matched-generator-source.csv; generation-streams-replot.csv','complete-domain model-minus-fragment; all 20 cells','attempt groups horizontally; 3 streams vertically','horizontal 97.5%; vertical 95% Student-t df=2; fixed checkpoints')
panel_sources('Fig5','e','conditional-class-yield-source.csv','eligible classes only: 14 ZymCTRL,15 ProLLaMA','200 attempts per condition per class','saved-batch descriptive rates; not class-level prevalence estimates')

pd.DataFrame(contracts).to_csv(DATA/'figure_data_contract.csv',index=False)
(DATA/'figure_validation.json').write_text(json.dumps({'checks':checks,'source_sha256':provenance,
 'insertion_width_inches':WIDTH,'insertion_height_cap_inches':HEIGHT_CAP,'minimum_font_points':BASE,
 'scope':'No inference or model training was rerun. Geometry checks do not certify scientific provenance.'},indent=2))
print('Rendered five figures with source-row exports and interval-extent checks.')
