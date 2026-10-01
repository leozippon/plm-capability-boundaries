#!/usr/bin/env python3
"""Build the editable draw.io study workflow and matching vector PDF/SVG.

Layout constants encode the frozen study design, not inferred outcomes.
Run from the repository root with the validated ct environment.
"""
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.path import Path as MplPath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=Path('manuscript/figures'))
OUT = parser.parse_args().out
W, H = 740, 790
WIDTH_MM = 130.7
SCALE = (WIDTH_MM / 25.4 * 72) / W
INK, LINE = '#17324D', '#687789'
BOXES = [
    dict(id='conditional', x=20, y=65, w=335, h=140, fill='#EDF3F8', stroke='#607B94',
         lines=[(17, 'Conditional generation', 21, True),
                (51, 'ZymCTRL: EC labels', 18, False),
                (78, 'ProLLaMA: superfamily instructions', 18, False),
                (107, '12,800 attempts; 29 eligible classes', 18, False)], dividers=[]),
    dict(id='native', x=385, y=65, w=335, h=140, fill='#F3EFF7', stroke='#89739C',
         lines=[(17, 'Native unconditional generation', 19, True),
                (51, 'ProGen3-3B: 800 attempts', 18, False),
                (78, '501 compiled outputs', 18, False),
                (107, '299 budget-censored continuations', 18, False)], dividers=[]),
    dict(id='conditional_sample', x=20, y=240, w=335, h=135, fill='#EDF3F8', stroke='#607B94',
         lines=[(17, 'Score-independent sample', 20, True),
                (52, '928 generated + 232 natural', 18, False),
                (79, 'Each paired with its own shuffle', 18, False),
                (106, 'Sampling stratified by length', 18, False)], dividers=[]),
    dict(id='native_sample', x=385, y=240, w=335, h=135, fill='#F3EFF7', stroke='#89739C',
         lines=[(17, 'Score-independent sample', 20, True),
                (52, '128 generated + 128 own shuffles', 18, False),
                (79, '89 compiled; 39 censored parents', 18, False),
                (106, 'Sampling stratified by length', 18, False)], dividers=[]),
    dict(id='pilot', x=20, y=435, w=335, h=160, fill='#EDF5EF', stroke='#759180',
         lines=[(17, 'Natural-reference calibration', 20, True),
                (51, '58 natural + 58 own shuffles', 18, False),
                (79, 'Separate from the main sample', 18, False),
                (106, 'Positive contrast required before', 18, False),
                (127, 'generated structures are evaluated', 18, False)], dividers=[]),
    dict(id='esmfold', x=385, y=435, w=335, h=160, fill='#F6F8FA', stroke='#607487',
         lines=[(17, 'Paired structure prediction', 20, True),
                (51, 'Same ESMFold settings throughout', 18, False),
                (79, 'Original versus composition shuffle', 18, False),
                (106, 'CA-pLDDT and pTM', 18, False),
                (127, 'Complete predictions retained', 18, False)], dividers=[]),
    dict(id='analysis', x=20, y=640, w=700, h=125, fill='#F0F3F6', stroke='#4B6278',
         lines=[(17, 'Linked evidence for the attempted output population', 21, True),
                (51, 'Profile recognition and nonredundant yield', 18, False),
                (77, 'Paired confidence contrasts with sampling weights and uncertainty', 18, False),
                (103, 'Reference identity and coverage, including explicit no-hit products', 18, False)], dividers=[]),
]
BY_ID = {box['id']: box for box in BOXES}
EDGES = [
    ('conditional', 'conditional_sample', [(187.5, 205), (187.5, 240)], False),
    ('native', 'native_sample', [(552.5, 205), (552.5, 240)], False),
    ('conditional_sample', 'esmfold', [(187.5, 375), (187.5, 405), (467, 405), (467, 435)], False),
    ('native_sample', 'esmfold', [(637, 375), (637, 435)], False),
    ('pilot', 'esmfold', [(355, 506), (385, 506)], False),
    ('esmfold', 'analysis', [(552.5, 595), (552.5, 640)], False),
]
FREE_TEXT = [
    ('title', 20, 12, 700, 30, 'Generation and paired evaluation', 24, True, INK),
]

mxfile = ET.Element('mxfile', host='app.diagrams.net', type='device')
diagram = ET.SubElement(mxfile, 'diagram', id='main-workflow', name='Workflow')
graph = ET.SubElement(diagram, 'mxGraphModel', dx=str(W), dy=str(H), grid='0', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1', pageWidth=str(W), pageHeight=str(H), math='0', shadow='0')
root = ET.SubElement(graph, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')


def text_cell(identifier, parent, x, y, w, h, value, size, bold, color=INK):
    style = f'text;html=0;whiteSpace=wrap;overflow=visible;strokeColor=none;fillColor=none;align=left;verticalAlign=top;spacing=0;fontFamily=Arial;fontSize={size};fontColor={color};fontStyle={1 if bold else 0};'
    cell = ET.SubElement(root, 'mxCell', id=identifier, value=value, style=style, vertex='1', parent=parent)
    ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), attrib={'as': 'geometry'})


for box in BOXES:
    style = f'rounded=0;html=0;whiteSpace=wrap;fillColor={box["fill"]};strokeColor={box["stroke"]};strokeWidth=1.6;container=1;recursiveResize=0;'
    cell = ET.SubElement(root, 'mxCell', id=box['id'], value='', style=style, vertex='1', parent='1')
    ET.SubElement(cell, 'mxGeometry', x=str(box['x']), y=str(box['y']), width=str(box['w']), height=str(box['h']), attrib={'as': 'geometry'})
    for i, (y, value, size, bold) in enumerate(box['lines']):
        text_cell(f'{box["id"]}_text_{i}', box['id'], 20, y, box['w']-40, 32, value, size, bold)
    for i, y in enumerate(box['dividers']):
        cell = ET.SubElement(root, 'mxCell', id=f'{box["id"]}_divider_{i}', value='', style='shape=line;strokeColor=#BAC5CE;strokeWidth=1;', vertex='1', parent=box['id'])
        ET.SubElement(cell, 'mxGeometry', x='20', y=str(y), width=str(box['w']-40), height='0', attrib={'as': 'geometry'})
for i, (source, target, points, dashed) in enumerate(EDGES):
    a, b = BY_ID[source], BY_ID[target]
    sx, sy = (points[0][0]-a['x'])/a['w'], (points[0][1]-a['y'])/a['h']
    tx, ty = (points[-1][0]-b['x'])/b['w'], (points[-1][1]-b['y'])/b['h']
    style = f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;endFill=1;endSize=8;strokeColor={LINE};strokeWidth=1.6;exitX={sx};exitY={sy};exitDx=0;exitDy=0;entryX={tx};entryY={ty};entryDx=0;entryDy=0;'
    if dashed:
        style += 'dashed=1;dashPattern=5 4;'
    cell = ET.SubElement(root, 'mxCell', id=f'edge_{i}', value='', style=style, edge='1', parent='1', source=source, target=target)
    geometry = ET.SubElement(cell, 'mxGeometry', relative='1', attrib={'as': 'geometry'})
    if len(points) > 2:
        array = ET.SubElement(geometry, 'Array', attrib={'as': 'points'})
        for x, y in points[1:-1]:
            ET.SubElement(array, 'mxPoint', x=str(x), y=str(y))
for spec in FREE_TEXT:
    text_cell(spec[0], '1', *spec[1:])
ET.indent(mxfile, space='  ')
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'figure_workflow.drawio').write_text(ET.tostring(mxfile, encoding='unicode')+'\n')

for font_path in font_manager.findSystemFonts():
    if Path(font_path).name.lower() in {'arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf', 'arial_bold.ttf', 'arial_italic.ttf'}:
        font_manager.fontManager.addfont(font_path)
plt.rcParams.update({'font.family': 'Arial', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(WIDTH_MM/25.4, (WIDTH_MM/25.4)*H/W), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, W), ylim=(H, 0))
ax.axis('off')
text_artists = []


def plot_text(x, y, value, size, bold, width, color=INK):
    artist = ax.text(x, y, value, fontsize=size*SCALE, fontweight='bold' if bold else 'normal', color=color, va='top', ha='left', zorder=5)
    text_artists.append((artist, width, value))


for box in BOXES:
    ax.add_patch(Rectangle((box['x'], box['y']), box['w'], box['h'], linewidth=1.6*SCALE, edgecolor=box['stroke'], facecolor=box['fill'], zorder=2))
    for y in box['dividers']:
        ax.plot([box['x']+20,box['x']+box['w']-20],[box['y']+y]*2,color='#BAC5CE',linewidth=SCALE,zorder=3)
    for y, value, size, bold in box['lines']:
        plot_text(box['x']+20, box['y']+y, value, size, bold, box['w']-40)
for source, target, points, dashed in EDGES:
    path = MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
    patch = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=8.5, linewidth=1.6*SCALE, color=LINE, linestyle=(0,(3,2.4)) if dashed else '-', zorder=1)
    ax.add_patch(patch)
for identifier, x, y, width, height, value, size, bold, color in FREE_TEXT:
    plot_text(x, y, value, size, bold, width, color)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
overflows=[]
for artist, width, value in text_artists:
    bbox=artist.get_window_extent(renderer)
    px_per_unit=(ax.transData.transform((1,0))-ax.transData.transform((0,0)))[0]
    if bbox.width/px_per_unit > width+1:
        overflows.append((value,round(bbox.width/px_per_unit,1),width))
if overflows:
    raise RuntimeError(f'Text exceeds its native editable container: {overflows}')
for extension in ('svg','pdf'):
    fig.savefig(OUT/f'figure_workflow.{extension}', metadata={'Title':'Computational biological evidence'})
fig.savefig(OUT / 'figure_workflow.png', dpi=180)
plt.close(fig)
print('Wrote native editable draw.io, text-preserving SVG, and vector PDF; all text fits its containers')
