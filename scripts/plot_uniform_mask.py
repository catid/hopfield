"""Paired calibration-free uniform acquisition failure, with resolved categories."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source='results/research/ripper/uniform_mask.json'
s=json.loads((ROOT/source).read_text());assert s['status']=='validated'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                     'text.color':'#17243a','axes.labelcolor':'#273449','axes.edgecolor':'#b7c1cf','svg.hashsalt':'uniform-mask-v1'})
fig,ax=plt.subplots(figsize=(11.7,7.4));g=s['grouped'][2];exact=[32,g['certified_integral_exact']];nonexact=[0,g['certified_integral_nonexact']];fractional=[0,g['certified_fractional']]
ax.bar([0,1],exact,color='#347e72',width=.5,label='Certified integral exact')
ax.bar([0,1],nonexact,bottom=exact,color='#c48b4e',width=.5,label='Certified integral nonexact')
ax.bar([0,1],fractional,bottom=[a+b for a,b in zip(exact,nonexact)],color='#9b819f',width=.5,label='Certified fractional')
for x,y in enumerate(exact):ax.text(x,33,f'{y}/32 exact',ha='center',weight='bold')
ax.axhline(29,color='#6d7482',ls='--',lw=1);ax.text(.01,29.5,'Frozen uniform-mask target: at least 29 exact',transform=ax.get_yaxis_transform(),fontsize=9.5)
ax.set(ylim=(0,36),xticks=[0,1],xticklabels=['Archived adaptive masks\nCalibration and exclusion feedback','Independent uniform masks\nNo calibration or label feedback'],ylabel='Banks, out of 32',yticks=[0,8,16,24,32])
ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper left',bbox_to_anchor=(0,1.20),ncols=3,frameon=False,fontsize=10)
fig.suptitle('Uniform acquisition fails the N900 graph-recovery target',x=.08,ha='left',fontsize=17,weight='bold')
fig.text(.08,.04,
    'Same fixed source rows and mask-independent corrupted boards; unchanged smoothed-rate degree-20 recovery. No unresolved runs.\n'
    'All uniform masks saved before source labels/noise load. 32 new + 32 archived exact proofs; no rounding of the fractional optimum.\n'
    'Uniform same-cost top-810: 0/32 exact, 257 missing and 257 extra pairs. BP AdamW outside this source-only study; not run.\n'
    'Alternative deployment costs: uniform 720k queries, zero calibration; adaptive 784k including 64k calibration.\n'
    'Actual paired replay: 222,073 reused + 497,927 newly exposed answers; cumulative union including prior calibration = 1,281,927.\n'
    'Failure does not prove calibration necessary: adaptive exposure/exclusion feedback also changed. No fresh-source or latent draws.\n'
    '5.152 aggregate CPU-s / 1.272 wall-s, including independent replay and proofs; zero puzzle-solver inputs or GPU work.',fontsize=9,color='#526075',linespacing=1.5)
fig.subplots_adjust(left=.08,right=.97,bottom=.38,top=.76)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/uniform-mask.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
plt.close(fig)
(ROOT/'assets/figures/uniform-mask-provenance.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    input_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},matplotlib=matplotlib.__version__,scope=s['scope']),indent=2)+'\n')
