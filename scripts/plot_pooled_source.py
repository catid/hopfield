"""Deterministic public H4ac comparison, BP AdamW first."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/research/foureyes/pooled_source_study/summary.json'
s = json.loads(SOURCE.read_text())
assert s['status'] == 'validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'h4ac-pooled-source'})
fig, ax = plt.subplots(figsize=(11.8, 6.4))
labels = ['BP AdamW\n3-of-4 vote','Canonical PL\n3-of-4 vote','Canonical PL\npooled fit','Counts\npooled observations']
values = [r['exact_graphs'] for r in s['grouped']]
ax.bar(range(4), values, color=['#738196','#bb694e','#347e72','#347e72'],width=.6)
for x, y in enumerate(values):
    ax.text(x, y+.25, f'{y}/16', ha='center',weight='bold')
ax.set(xticks=range(4),xticklabels=labels,ylim=(0,18),yticks=[0,4,8,12,16],ylabel='Exactly recovered Sudoku graphs')
ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
fig.suptitle('Pooling the same noisy observations recovers all 16 graphs',x=.08,ha='left',fontsize=17,weight='bold')
fig.text(.08,.865,'Canonical majority target FAILED: 15 exact graphs, but one true edge was lost.',fontsize=11.7,color='#a04c35')
fig.subplots_adjust(left=.08,right=.97,bottom=.34,top=.78)
fig.text(.08,.08,'All 64 archived source banks in 16 fixed quartets; 2,400 observation instances per quartet, including repeated rows.\nNine free-cell digits corrupted per board; no geometry in fitting or threshold selection. No inference or new labels.\nFinite source-bank comparison; descriptive controls use different training work. No matched-compute or fresh-generalization claim.',fontsize=10,linespacing=1.6)
out=ROOT/'assets/figures'
for ext in ['png','svg']:
    fig.savefig(out/f'pooled-source.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE,Path(__file__)]},
             output_sha256={f'pooled-source.{ext}':hashlib.sha256((out/f'pooled-source.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
(out/'pooled-source-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt['output_sha256']))
