"""Public finite-cohort cycle-cut comparison; no prospective claim."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'results/research/work/lexical-cycle-cut-summary.json'
s = json.loads(source.read_text())
assert s['status'] == 'passed'
keys = ['pl','old_binary','old_blocks','connectivity_veto','energy_ranked_cycle_cut','lexical_cycle_cut']
labels = ['BP AdamW, native','LP binary descent','Original block moves','Connectivity veto','Veto + energy-ranked cuts','Veto + lexicographic cuts']
failures = [4096 - s['joint_success'][key] for key in keys]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cycle-cut-controls-v1'})
fig, ax = plt.subplots(figsize=(12,7))
ax.barh(range(6),failures,color=['#738196','#738196','#738196','#bc8651','#347e72','#347e72'],height=.62)
ax.set(yticks=range(6),yticklabels=labels,xlim=(0,103),xlabel='Joint failures across three frozen models (out of 4,096 mazes)')
ax.invert_yaxis();ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
for y, value in enumerate(failures):ax.text(value+1.2,y,str(value),va='center',weight='bold')
fig.suptitle('Cycle cuts rescue 21 of 22 archived maze failures',x=.065,ha='left',fontsize=18,weight='bold')
fig.text(.065,.865,'Energy-ranked and lexicographic choices tie: no measured benefit from learned cut ordering.',fontsize=11.4)
fig.subplots_adjust(left=.29,right=.96,bottom=.29,top=.79)
fig.text(.065,.07,'Same 4,096 previously examined 120×120 loopy mazes, three fixed models; no new inputs.\nCycle cuts use an explicit graph prior and allow energy increases; native descent follows each cut.\nBoth cut policies solve 4,095/4,096 inputs jointly, with zero regressions versus veto. Simple paths are copied exactly.\nFinite source-cohort result. Training differs between BP and LP; lexical costs remain computed for auditing. No fresh or optimized-runtime claim.',fontsize=10,linespacing=1.6)
out=ROOT/'assets/figures'
for ext in ['png','svg']:
    fig.savefig(out/f'cycle-cut-controls.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
plt.close(fig)
receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},
             output_sha256={f'cycle-cut-controls.{ext}':hashlib.sha256((out/f'cycle-cut-controls.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
(out/'cycle-cut-controls-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt['output_sha256']))
