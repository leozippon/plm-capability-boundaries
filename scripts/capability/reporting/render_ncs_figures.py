#!/usr/bin/env python3
"""Render the NCS figures from the included, full-precision source CSV files alone."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figure_data' / 'ncs'
SUBMISSION = Path(__file__).resolve().parents[3] / 'manuscript' / 'figures' / 'ncs'
SUBMISSION_NAMES = {'conditional-class-yield', 'generation-yield', 'matched-generators'}
OUT.mkdir(parents=True, exist_ok=True)
SUBMISSION.mkdir(parents=True, exist_ok=True)
# One visual grammar across every figure: BLUE is the panel's primary model-side quantity,
# ORANGE a control- or comparator-referenced quantity, GREEN a second control comparator,
# GREY a reference line, noise floor, denominator-side count, empty support or a row excluded
# from a primary aggregate. A filled marker carries a resolved interval and an open marker
# does not. Rows follow one lineage order, never a capability order, and every estimate
# carries its interval or is declared exact.
BLUE, ORANGE, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', '#808080'
PALE = '#F2F2F2'
WIDTH = 5.14  # the manuscript text width in inches; every figure is inserted at this width
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8,
                     'axes.labelsize': 8, 'axes.linewidth': .6, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none'})
DIVERGING = LinearSegmentedColormap.from_list('control_contrast', [ORANGE, '#FBEDE4', 'white', '#DCEBF5', BLUE])
# One name per checkpoint, identical to the name used in the manuscript text.
LABELS = {'llama-2-7b': 'Llama-2-7B', 'prollama-stage-1': 'ProLLaMA Stage 1', 'prollama': 'ProLLaMA Stage 2',
          'protgpt2': 'ProtGPT2', 'protgpt3-1.3b': 'ProtGPT3-1.3B', 'proteinglm-7b-clm': 'ProteinGLM-7B-CLM',
          'instructprotein': 'InstructProtein', 'rita-xl': 'RITA-xl', 'dialogpt-small': 'DialoGPT-small',
          'llama-3.2-3b': 'Llama-3.2-3B', 'qwen3-8b-base': 'Qwen3-8B-base', 'zymctrl': 'ZymCTRL',
          'qwen2.5-0.5b-instruct': 'Qwen2.5-0.5B-it'}
for key, prefix in [('progen2', 'ProGen2'), ('progen3', 'ProGen3'), ('galactica', 'Galactica'),
                    ('qwen2.5', 'Qwen2.5'), ('bygpt5', 'ByGPT5')]:
    for size in ['small', 'base', 'medium', 'large', 'xlarge', '112m', '125m', '1.3b', '3b',
                 '6.7b', '30b', '0.5b', '7b', '32b']:
        LABELS[f'{key}-{size}' + ('-en' if key == 'bygpt5' else '')] = (
            f'{prefix}-' + (size.upper() if size[0].isdigit() else size) + ('-en' if key == 'bygpt5' else ''))
for size in ['', '-medium', '-large', '-xl']:
    LABELS['gpt2' + size] = 'GPT-2' + size
GEN_ORDER = ['protgpt2', 'progen2-small', 'progen2-base', 'progen2-medium', 'progen2-large', 'progen2-xlarge',
             'progen3-112m', 'progen3-3b', 'protgpt3-1.3b', 'rita-xl', 'galactica-125m', 'galactica-1.3b',
             'galactica-6.7b', 'galactica-30b', 'llama-2-7b', 'prollama-stage-1', 'prollama',
             'proteinglm-7b-clm', 'instructprotein']
TEXT_ORDER = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'dialogpt-small', 'bygpt5-small-en',
              'bygpt5-base-en', 'bygpt5-medium-en', 'qwen2.5-0.5b', 'qwen2.5-0.5b-instruct', 'qwen2.5-7b',
              'qwen2.5-32b', 'qwen3-8b-base', 'llama-3.2-3b']
SOURCE_HASHES = {}


def read(name):
    payload = (OUT / name).read_bytes()
    SOURCE_HASHES[name] = hashlib.sha256(payload).hexdigest()
    return list(csv.DictReader(payload.decode().splitlines()))


def number(row, key):
    return None if row.get(key) in (None, '') else float(row[key])


def save(fig, name):
    # Fail rather than silently clip an effect or its uncertainty to a chosen axis.
    for axis in fig.axes:
        lower, upper = sorted(axis.get_xlim())
        for low, high in getattr(axis, '_retained_intervals', []):
            assert lower <= low <= high <= upper, (name, axis.get_title(), (low, high), (lower, upper))
    pdf_dir = SUBMISSION if name in SUBMISSION_NAMES else OUT
    fig.savefig(pdf_dir / f'{name}.pdf', facecolor='white')
    for extension in ['svg', 'png']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=300, facecolor='white')
    plt.close(fig)


def panel(fig, rect, letter, title, xlabel='', letter_dx=-.055, title_pad=6):
    ax = fig.add_axes(rect)
    if title:
        ax.set_title(title, loc='left', pad=title_pad)
    ax.text(letter_dx, 1.0, letter, transform=ax.transAxes, fontweight='bold', fontsize=9,
            ha='right', va='bottom')
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_axisbelow(True)
    return ax


def rows(ax, labels, grid=True):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - .5, -.5)
    ax.tick_params(axis='y', length=0)
    if grid:
        ax.grid(axis='x', color='#ECECEC', lw=.5)


def bar_interval(ax, y, point, low, high, color, filled=True, size=3.4, lw=.9, label=None, zorder=3):
    """One estimate with its interval, or an open marker when no interval is retained."""
    if point is None:
        return
    if low is None or high is None:
        ax.plot([point], [y], marker='D', ms=2.6, ls='', color=color, label=label, zorder=zorder)
        return
    assert low <= point <= high, (y, point, low, high)
    if not hasattr(ax, '_retained_intervals'):
        ax._retained_intervals = []
    ax._retained_intervals.append((low, high))
    ax.errorbar(point, y, xerr=[[point - low], [high - point]], fmt='o', ms=size, color=color,
                lw=lw, capsize=1.8, label=label, zorder=zorder,
                markerfacecolor=color if filled else 'white', markeredgecolor=color, markeredgewidth=.8)


# Main figures share a common arm order and retain effect sizes rather than verdict matrices.
ORDER = GEN_ORDER + TEXT_ORDER
PRIMARY = '20260923'

def axis_at(fig, rect, letter, title, xlabel='', labels=None):
    ax = panel(fig, rect, '', title, xlabel)
    fig.text(.008 if rect[0]<.4 else rect[0]-.035, rect[1]+rect[3]+.006,
             letter, fontweight='bold', fontsize=9, va='bottom')
    if labels is not None: rows(ax,labels)
    ax.axvline(0,color=GREY,lw=.7,ls='--')
    return ax

def point_row(ax,index,record,color=BLUE,offset=0,filled=None):
    lo,hi=number(record,'ci_low'),number(record,'ci_high')
    if filled is None: filled=lo is not None and (lo>0 or hi<0)
    bar_interval(ax,index+offset,number(record,'estimate'),lo,hi,color,filled=filled,size=3.1,lw=.8)

def seed_ticks(ax,index,table,arm,endpoint,color=BLUE):
    other=[number(r,'estimate') for r in table if r['arm']==arm and r['endpoint']==endpoint and r['split_seed']!=PRIMARY]
    ax.plot(other,[index]*len(other),'|',color=color,ms=4,mew=.7)

# Figure 1: local control qualification and an independent biological endpoint.
local=read('local-increments-source.csv'); external=read('external-abundance-source.csv')
order=[a for a in ORDER if any(r['arm']==a for r in local)]
fig=plt.figure(figsize=(WIDTH,5.6))
ax=fig.add_axes([.01,.867,.49,.11]);ax.set_axis_off()
ax.text(0,1,'a',weight='bold',fontsize=9,va='top')
ax.text(.08,1,'Paired prediction on held groups',va='top')
ax.text(.08,.61,'Sequence / profile / local control',va='center',color=GREY)
ax.text(.08,.29,'Same control + model score',va='center',color=BLUE)
ax.text(.08,-.03,'Compare predictions against the same labels',va='center')
b=axis_at(fig,[.65,.9,.33,.06],'b','Qualify the local control','Held-group Spearman ρ', ['C + profile','+ local window'])
for i,ep in enumerate(['C_P_spearman','C_P_wall_spearman']):
 for r in [r for r in local if r['arm']=='progen3-3b' and r['endpoint']==ep]:
  if r['split_seed']==PRIMARY:point_row(b,i,r,ORANGE if i==0 else GREEN)
  else:b.plot(number(r,'estimate'),i,'|',color=GREY,ms=4)
b.set_xlim(.40,.53);b.set_xticks([.4,.45,.5])
for letter,x,width,table,endpoint,title,xlabel in [
 ('c',.285,.325,local,'increment_M_C_P_wall','Mutation-effect ranking','Δ Spearman ρ'),
 ('d',.68,.30,external,'primary_likelihood','Domain abundance','MSE reduction (abundance²)')]:
 axis=axis_at(fig,[x,.095,width,.67],letter,title,xlabel,
              [LABELS.get(a,a) for a in order] if letter=='c' else ['']*len(order))
 for i,arm in enumerate(order):
  record=next(r for r in table if r['arm']==arm and r['endpoint']==endpoint and r['split_seed']==PRIMARY)
  point_row(axis,i,record);seed_ticks(axis,i,table,arm,endpoint)
 axis.text(0,1.055,'163 groups; 201 assays' if letter=='c' else '96 groups; 428 domains',transform=axis.transAxes)
 if letter=='c':axis.set_xlim(-.004,.034);axis.set_xticks([0,.01,.02,.03])
 else:axis.set_xlim(-.0016,.0038);axis.set_xticks([0,.002]);axis.ticklabel_format(axis='x',style='plain',useOffset=False)
save(fig,'capability-map')

# Figure 2: native-score and literal-string strata on separate biological supports.
native=read('native-fitness-source.csv');textaa=read('text-aa-fitness-source.csv')
fig=plt.figure(figsize=(WIDTH,5.55))
for table,order0,eps,y,h,letters,title in [
 (native,GEN_ORDER,['raw_spearman','model_minus_lookup','model_minus_blosum62'],.565,.34,'ab','Native protein-sequence interfaces'),
 (textaa,TEXT_ORDER,['raw','model_minus_lookup','model_minus_blosum62'],.13,.31,'cd','Literal amino-acid strings')]:
 order2=[a for a in order0 if any(r['arm']==a for r in table)]
 fig.text(.01,y+h+.052,title,weight='bold')
 left=axis_at(fig,[.285,y,.30,h],letters[0],'Native score','Spearman ρ',[LABELS.get(a,a) for a in order2])
 right=axis_at(fig,[.675,y,.305,h],letters[1],'Paired baseline contrasts','Δ Spearman ρ',['']*len(order2))
 for i,a in enumerate(order2):
  raw=[r for r in table if r['arm']==a and r['endpoint']==eps[0]]
  if raw:point_row(left,i,raw[0])
  else:left.text(.03,i,'not retained',transform=left.get_yaxis_transform(),va='center',color=GREY)
  for endpoint,offset,color in [(eps[1],-.17,ORANGE),(eps[2],.17,GREEN)]:
   point_row(right,i,next(r for r in table if r['arm']==a and r['endpoint']==endpoint),color,offset)
 left.set_xlim(-.07,.47);left.set_xticks([0,.2,.4])
 right.set_xlim(-.42,.25);right.set_xticks([-.4,-.2,0,.2])
fig.text(.285,.025,'Orange: score − profile',color=ORANGE)
fig.text(.67,.025,'Green: score − BLOSUM62',color=GREEN)
save(fig,'native-fitness')

# Figure 3: disentangle the unadjusted readout advantage and controlled predictor contrast.
readout=read('readout-contrasts-source.csv');gaps=read('controlled-gap-source.csv')
order=[a for a in ORDER if any(r['arm']==a for r in readout)]
fig=plt.figure(figsize=(WIDTH,5.6))
a=axis_at(fig,[.285,.28,.31,.66],'a','Two readout contrasts','Δ Spearman ρ',[LABELS.get(a,a) for a in order])
b=axis_at(fig,[.69,.28,.29,.66],'b','Controlled R − M','Δ Spearman ρ',['']*len(order))
for i,arm in enumerate(order):
 r=next(r for r in readout if r['arm']==arm)
 for key,off,col in [('readout_minus_likelihood',-.18,BLUE),('representation_increment',.18,ORANGE)]:
  point_row(a,i,dict(estimate=r[key],ci_low=r[key+'_ci_low'],ci_high=r[key+'_ci_high']),col,off)
 q=next(r for r in gaps if r['arm']==arm and r['split_seed']==PRIMARY)
 point_row(b,i,q,GREEN);seed_ticks(b,i,gaps,arm,'controlled_representation_minus_likelihood',GREEN)
a.set_xlim(-.09,.22);a.set_xticks([0,.1,.2]);b.set_xlim(-.12,.03);b.set_xticks([-.1,-.05,0])
fig.text(.285,.981,'201 assays; 163 held groups; primary split intervals',fontsize=8)
fig.text(.285,.195,'Readout − raw score',color=BLUE)
fig.text(.65,.195,'Added R over matched baseline',color=ORANGE)
depth=read('depth-profile-source.csv')
c=panel(fig,[.17,.078,.80,.06],'','Selected depths: ranges across three split seeds','Relative depth')
fig.text(.008,.14,'c',weight='bold',fontsize=9)
c.set_ylabel('Δ Spearman ρ');c.axhline(0,color=GREY,lw=.7,ls='--')
for name,col,marker in [('pooled',BLUE,'o'),('position_resolved',GREEN,'s')]:
 selected=[r for r in depth if r['axis']==name and r['interface']=='native_protein']
 for r in selected:
  c.plot([number(r,'relative_depth')]*2,[number(r,'increment_min'),number(r,'increment_max')],color=col,lw=.8)
 c.scatter([number(r,'relative_depth') for r in selected],[number(r,'increment_min') for r in selected],s=12,c=col,marker=marker)
c.set_xlim(-.04,1.04);c.set_xticks([0,.5,1]);c.set_yticks([0,.025,.05]);c.set_ylim(-.003,.06)
c.text(.03,.87,'Pooled',color=BLUE,transform=c.transAxes)
c.text(.26,.87,'Position-resolved',color=GREEN,transform=c.transAxes)
save(fig,'readout-contrasts')

# Figure 4: raw-target measurements and the nested adjusted-target sensitivity stay separate.
interaction=read('interaction-panel-source.csv');residual=read('residual-full-panel-source.csv')
boundaries=read('endpoint-boundaries-source.csv');contact=read('contact-sensitivity-source.csv')
fig=plt.figure(figsize=(WIDTH,6.4))
stratum={r['arm']:r['tokenisation_stratum'] for r in interaction}
for letter,x,width,arms,title in [
 ('a',.26,.25,[a for a in ORDER if stratum.get(a) in ('amino_acid','byte')],'One token per residue'),
 ('b',.765,.215,[a for a in ORDER if stratum.get(a)=='bpe'],'Multi-residue tokens')]:
 ax=axis_at(fig,[x,.645,width,.32],letter,title,'',[LABELS.get(a,a) for a in arms])
 for i,arm in enumerate(arms):
  for ep,off,col in [('first_order_likelihood',-.19,BLUE),('likelihood_interaction_over_first_order',.19,ORANGE)]:
   r=next(r for r in interaction if r['arm']==arm and r['endpoint']==ep)
   point_row(ax,i,r,col,off,filled=r['seeds_above_zero']=='3')
 ax.set_xlim(-.025,.026);ax.set_xticks([-.02,0,.02])
fig.text(.25,.601,'First-order score',color=BLUE)
fig.text(.60,.601,'Interaction beyond first order',color=ORANGE)
fig.text(.50,.573,'Group-equal MSE reduction (kcal²/mol²)',ha='center')
fig.text(.008,.535,'c',weight='bold',fontsize=9)
fig.text(.13,.535,'Adjusted target: 33 checkpoints; simultaneous 95% bands')
c=fig.add_axes([.13,.39,.85,.115])
order=[arm for arm in ORDER if any(r['arm']==arm for r in residual)]
assert len(order)==33
selected=[next(r for r in residual if r['arm']==arm) for arm in order]
for i,r in enumerate(selected):
 lo,hi=number(r,'ci_low'),number(r,'ci_high')
 c.plot([i,i],[lo,hi],color=GREEN,lw=.8)
 c.scatter([i],[number(r,'estimate')],s=12,facecolors='white',edgecolors=GREEN,linewidths=.8,zorder=3)
c.axhline(0,color=GREY,lw=.7)
c.set_xlim(-.6,32.6);c.set_ylim(-.0018,.0054)
assert min(number(r,'ci_low') for r in selected)>=c.get_ylim()[0]
assert max(number(r,'ci_high') for r in selected)<=c.get_ylim()[1]
c.set_xticks(range(33));c.set_xticklabels([LABELS.get(a,a) for a in order],rotation=90,ha='center',fontsize=8)
c.set_ylabel('Δ MSE (kcal²/mol²)');c.set_yticks([0,.002,.004])
c.spines[['top','right']].set_visible(False)
rel=[r for r in boundaries if r['family']=='reliability']
d=axis_at(fig,[.275,.075,.225,.10],'d','Endpoint repeatability','Shared / discordance',
          ['Single substitution','Double cycle','HIS3 order 2','HIS3 order 3'])
d.axvline(1,color=GREY,lw=.7,ls=':')
for i,r in enumerate(rel):point_row(d,i,r)
e=axis_at(fig,[.765,.075,.215,.10],'e','Contact sensitivity','Δ |ε| (kcal/mol)',
          ['Heavy atom','Cβ','Ensemble'])
for i,definition in enumerate(['heavy_atom','cb','ensemble_majority']):
 for version,off,col in [('historical',-.17,GREY),('corrected',.17,ORANGE)]:
  r=next(r for r in contact if r['version']==version and r['definition']==definition and r['endpoint']=='mean_abs_epsilon' and r['weighting']=='group_equal')
  point_row(e,i,r,col,off)
e.set_xlim(-.18,.5);e.set_xticks([0,.4]);d.set_xticks([0,1,2,3])
save(fig,'interaction-floors')

# Figure 5: context comparison and attempted generation yield.
context=read('context-ranking-source.csv');generators=read('matched-generator-source.csv')
fig=plt.figure(figsize=(WIDTH,4.65))
order=[a for a in GEN_ORDER if any(r['arm']==a for r in context)]
for letter,x,width,endpoint,title in [('a',.285,.30,'delta_spearman','Homologue − unrelated'),('b',.68,.30,'homolog_minus_no_context_spearman','Homologue − none')]:
 ax=axis_at(fig,[x,.745,width,.16],letter,title,'Δ Spearman ρ',[LABELS.get(a,a) for a in order] if letter=='a' else ['']*len(order))
 for i,arm in enumerate(order):
  r=next(r for r in context if r['arm']==arm and r['endpoint']==endpoint)
  point_row(ax,i,r,BLUE if letter=='a' else ORANGE)
 ax.set_xlim(-.17,.25);ax.set_xticks([-.1,0,.1,.2])
fig.text(.285,.967,'Mutation ranking: 164 assays; 134 shared groups')
# Display every generation cell: actual yields and a paired matched-fragment contrast.
cells=[r['cell'] for r in generators if r['endpoint']=='complete_domain' and r['cohort']=='fragment']
cells=sorted(cells,key=lambda c:(ORDER.index(c.split('__')[0]) if c.split('__')[0] in ORDER else 99,c))
labels=[LABELS.get(c.split('__')[0],c.split('__')[0])+('†' if c.split('__')[1]!='unconditioned' else '') for c in cells]
c=axis_at(fig,[.285,.135,.30,.45],'c','Recognized products','Count / all attempts',labels)
d=axis_at(fig,[.68,.135,.30,.45],'d','Domain − fragment','Paired rate difference',['']*len(cells))
for i,cell in enumerate(cells):
 for ep,col,off in [('any_family',BLUE,-.17),('complete_domain',GREEN,.17)]:
  r=next(r for r in generators if r['cell']==cell and r['endpoint']==ep and r['cohort']=='fragment')
  c.plot(number(r,'model_rate'),i+off,'o',color=col,ms=3)
  if ep=='complete_domain':point_row(d,i,dict(estimate=r['difference'],ci_low=r['ci_low'],ci_high=r['ci_high']),ORANGE)
c.set_xlim(-.03,1.03);c.set_xticks([0,.5,1]);d.set_xlim(-.45,.65);d.set_xticks([-.4,0,.4])
fig.text(.285,.018,'Any family',color=BLUE);fig.text(.52,.018,'Complete domain',color=GREEN)
fig.text(.82,.018,'† conditioned')
save(fig,'distance-generation')

# -------------------------------- Supplementary: every cell against every matched cohort
COHORTS = [('shuffle', 'Composition shuffle'), ('hydropathy', 'Hydropathy sampler'),
           ('markov_0', 'Corpus order 0'), ('markov_2', 'Corpus order 2'),
           ('markov_4', 'Corpus order 4'), ('fragment', 'Corpus fragment'),
           ('natural', 'Whole corpus, reference')]
cells = [row['cell'] for row in generators if row['endpoint'] == 'any_family' and row['cohort'] == 'fragment']
cell_order = sorted(cells, key=lambda cell: (GEN_ORDER.index(cell.split('__')[0])
                                             if cell.split('__')[0] in GEN_ORDER else len(GEN_ORDER),
                                             cell.split('__')[1]))


def cell_label(cell):
    arm, condition = cell.split('__')
    return LABELS.get(arm, arm) + ('' if condition == 'unconditioned' else '\u2020')


GEN_H = 3.75
fig = plt.figure(figsize=(WIDTH, GEN_H))
ceiling = {row['endpoint']: number(row, 'model_rate') for row in generators if row['cohort'] == 'ceiling'}
axa = panel(fig, [0.280, 1 - 1.22 / GEN_H, 0.640, 0.80 / GEN_H], '', '')
fig.text(0.012, 1 - 0.42 / GEN_H, 'a', fontweight='bold', fontsize=9, va='bottom')
axa.set_ylabel('Oracle rate / attempt')
axa.set_xlim(-.6, len(cell_order) - .4)
axa.set_xticks(range(len(cell_order)), [''] * len(cell_order))
axa.set_ylim(-.05, 1.05)
axa.set_yticks([0, .5, 1], ['0', '0.5', '1'])
axa.grid(axis='y', color='#ECECEC', lw=.5)
axa.tick_params(axis='x', length=0)
axa.text(1.0, 1.03, 'any curated family', color=BLUE, fontsize=8, ha='right', transform=axa.transAxes)
axa.text(.58, 1.03, 'complete domain', color=GREEN, fontsize=8, ha='right', transform=axa.transAxes)
for endpoint, color, marker in [('any_family', BLUE, 'o'), ('complete_domain', GREEN, 's')]:
    axa.axhline(ceiling[endpoint], color=color, ls=':', lw=.8)
    values = [number(next(row for row in generators if row['cell'] == cell
                          and row['endpoint'] == endpoint and row['cohort'] == 'fragment'), 'model_rate')
              for cell in cell_order]
    axa.plot(range(len(cell_order)), values, marker=marker, ms=3.0, ls='', color=color)
axb = fig.add_axes([0.280, 1 - 2.40 / GEN_H, 0.640, 0.98 / GEN_H])
fig.text(0.012, 1 - 1.42 / GEN_H, 'b', fontweight='bold', fontsize=9, va='bottom')
axb.set_title('Model minus cohort rate on the any-curated-family endpoint', loc='left', pad=5)
matrix = np.full((len(COHORTS), len(cell_order)), np.nan)
resolved = np.zeros_like(matrix, dtype=bool)
unqualified = np.zeros_like(matrix, dtype=bool)
for row_index, (cohort, _) in enumerate(COHORTS):
    for column, cell in enumerate(cell_order):
        record = next(row for row in generators if row['cell'] == cell and row['cohort'] == cohort
                      and row['endpoint'] == 'any_family')
        matrix[row_index, column] = number(record, 'difference')
        resolved[row_index, column] = number(record, 'ci_low') > 0
        unqualified[row_index, column] = record['qualified'] == 'False'
image = axb.imshow(matrix, cmap=DIVERGING, vmin=-.6, vmax=.6, aspect='auto',
                   extent=(-.5, len(cell_order) - .5, len(COHORTS) - .5, -.5))
axb.set_yticks(range(len(COHORTS)), [name for _, name in COHORTS])
axb.set_xticks(range(len(cell_order)), [cell_label(cell) for cell in cell_order],
               rotation=90, ha='center')
axb.tick_params(axis='both', length=0)
for row_index in range(len(COHORTS)):
    for column in range(len(cell_order)):
        if resolved[row_index, column]:
            axb.plot([column], [row_index], marker='o', ms=1.5, color='#111111')
        if unqualified[row_index, column]:
            axb.add_patch(Rectangle((column - .5, row_index - .5), 1, 1, facecolor='none',
                                    edgecolor='white', hatch='///', lw=0, zorder=2))
axb.axhline(5.5, color='white', lw=1.4)
bar = fig.add_axes([0.915, 1 - 2.26 / GEN_H, 0.014, 0.70 / GEN_H])
fig.colorbar(image, cax=bar, ticks=[-.5, 0, .5])
bar.tick_params(labelsize=8, length=2)
save(fig, 'matched-generators')

# ------------------------------------------------ Supplementary: unconditional attempt census
census = read('generation-yield-source.csv')
census_order = [arm for arm in GEN_ORDER if any(row['arm'] == arm for row in census)]
fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 6.10), sharey=True)
SERIES_CENSUS = [[('nonempty_yield', -.14, GREY, 'o', 'Nonempty'),
                  ('exact_unique_yield', .14, BLUE, 's', 'Exactly unique')],
                 [('profile_yield', -.14, BLUE, 'o', 'Any Pfam hit'),
                  ('profile_group_yield', .14, ORANGE, 's', 'Distinct hit groups')]]
for index, arm in enumerate(census_order):
    record = next(row for row in census if row['arm'] == arm)
    for column, series in enumerate(SERIES_CENSUS):
        for key, shift, color, marker, label in series:
            value = number(record, key)
            if value is not None:
                axs[column].scatter(value, index + shift, s=13, color=color, marker=marker,
                                    label=label if index == 0 else None)
for column, axis in enumerate(axs):
    axis.set_title(['Sequence output', 'Profile recognition'][column], loc='left', pad=16)
    axis.text(-.1, 1.025, chr(97 + column), transform=axis.transAxes, fontweight='bold', fontsize=9)
    axis.set_xlabel('Count / 800 attempts')
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='x', color='#ECECEC', lw=.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis='y', length=0)
    axis.set_xlim(-.03, 1.03)
    axis.set_xticks([0, .5, 1])
    axis.legend(frameon=False, loc='lower left', bbox_to_anchor=(-.18, -.20), fontsize=8,
                handletextpad=.3)
axs[0].set_yticks(range(len(census_order)), [LABELS.get(arm, arm) for arm in census_order])
axs[0].invert_yaxis()
fig.subplots_adjust(left=.30, right=.98, top=.93, bottom=.16, wspace=.25)
save(fig, 'generation-yield')

# ------------------------------------------------------ Supplementary: conditional class census
rows_conditional = read('conditional-class-yield-source.csv')
fig, axs = plt.subplots(2, 1, figsize=(WIDTH, 5.95))
for index, (axis, arm, title) in enumerate(zip(axs, ['zymctrl', 'prollama'],
                                               ['ZymCTRL: EC classes', 'ProLLaMA Stage 2: superfamilies'])):
    selected = sorted([row for row in rows_conditional if row['arm'] == arm
                       and row['condition'] == 'requested'], key=lambda row: row['class_key'])
    for position, row in enumerate(selected):
        mismatched = next(other for other in rows_conditional if other['arm'] == arm
                          and other['class_key'] == row['class_key'] and other['condition'] == 'mismatched')
        color = BLUE if row['primary_class'] == 'True' else GREY
        low, high = number(mismatched, 'target_profile_rate'), number(row, 'target_profile_rate')
        axis.plot([low, high], [position, position], color=color, alpha=.4)
        axis.scatter(high, position, color=color, s=16, label='Requested' if position == 0 else None)
        axis.scatter(low, position, facecolors='white', edgecolors=color, s=16,
                     label='Mismatched' if position == 0 else None)
        axis.scatter(number(row, 'distinct_target_groups_per_attempt'), position + .17,
                     color=ORANGE if row['primary_class'] == 'True' else GREY, marker='s', s=12,
                     label='Distinct target groups' if position == 0 else None)
    axis.set_yticks(range(16), [row['class_key'] + (' *' if row['primary_class'] != 'True' else '')
                                for row in selected])
    axis.invert_yaxis()
    axis.set_xlim(-.03, 1.03)
    axis.set_xticks([0, .25, .5, .75, 1])
    axis.set_title(title, loc='left', pad=16)
    axis.text(-.1, 1.025, chr(97 + index), transform=axis.transAxes, fontweight='bold', fontsize=9)
    axis.set_xlabel('Count / 200 attempts per cell')
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='x', color='#ECECEC', lw=.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis='y', length=0)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(.58, .005))
fig.subplots_adjust(left=.25, right=.98, top=.93, bottom=.16, hspace=.43)
save(fig, 'conditional-class-yield')

(OUT / 'render-provenance.json').write_text(json.dumps(
    {'csv_sha256': SOURCE_HASHES,
     'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'figure_width_inches': WIDTH, 'minimum_font_size_pt': 8.0,
     'requires_private_result_tree': False}, indent=2) + '\n')
print('Rendered five main figures and three supplementary figures from the included source tables.')
