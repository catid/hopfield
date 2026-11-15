"""H1bs real versus empirical-marginal shuffle concentration; graph errors remain visible."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'results/research/ripper/marginal_shuffle/summary.json';s=json.loads(source.read_text());assert s['endpoint']['lower']==32
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#26334b','text.color':'#26334b','axes.edgecolor':'#b4c0d0','savefig.facecolor':'white','svg.hashsalt':'hopfield-H1bs-null-signal-v1'})
fig,axes=plt.subplots(1,2,figsize=(13.5,8));fig.subplots_adjust(left=.08,right=.97,bottom=.34,top=.76,wspace=.27)
fig.suptitle('Real curvature exceeds the exact-marginal shuffle in every bank',x=.08,y=.965,ha='left',fontsize=19,fontweight='bold')
fig.text(.08,.89,'Joint both-mask target supported: 32/32 ≥29. Sharper curvature does not certify an exact graph.',fontsize=12.5)
for ax,arm in zip(axes,['uniform','balanced']):
 rows=[r for r in s['rows'] if r['arm']==arm]
 for exact,color,label in [(True,'#328273','Real graph exact'),(False,'#bd6c69','Real graph nonexact')]:
  part=[r for r in rows if (r['real_classification']=='success')==exact]
  x=[r['null_concentration']['numerator']/r['null_concentration']['denominator'] for r in part];y=[r['real_concentration']['numerator']/r['real_concentration']['denominator'] for r in part]
  ax.scatter(x,y,s=45,color=color,label=label,alpha=.85,edgecolors='white',linewidths=.5)
 ax.plot([0,.3],[0,.3],linestyle='--',color='#8497ae',linewidth=1);ax.set(xlim=(0,.3),ylim=(0,.3),xticks=[0,.1,.2,.3],yticks=[0,.1,.2,.3],xlabel='Empirical-marginal shuffle concentration R',ylabel='Archived real concentration R')
 ax.set_title(arm.capitalize()+' masks: 32 strict contrasts',loc='left',fontsize=12);ax.legend(loc='lower right',fontsize=10,frameon=False)
fig.text(.08,.23,'R = max curvature /sum curvature over degrees 1–39; exact rational comparisons, ties fail; all 64 comparisons resolved.\n'
         'All 32 H1bq banks retained; 7 uniform and 6 balanced nonexact real graphs also exceed their shuffled control.\n'
         'All 5,184 empirical per-cell digit histograms preserved. Shuffle inferred degrees: 1–3, never 20.\n'
         '2,560 new +2,560 archived proofs; 15,360 altered proofs rejected. BP AdamW outside scope; not run.\n'
         'Zero new official labels; 829,440 separate arm assignments. Conditional contrast; no graph-confidence calibration.',
         fontsize=10.5,color='#596a83',va='top',linespacing=1.6)
out=ROOT/'assets/figures'
for ext in ['png','svg']:fig.savefig(out/f'marginal_shuffle.{ext}',dpi=170,metadata={'Date':None})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(out/'marginal_shuffle-provenance.json').write_text(json.dumps(dict(source=str(source.relative_to(ROOT)),source_sha256=sha(source),script_sha256=sha(Path(__file__)),png_sha256=sha(out/'marginal_shuffle.png'),svg_sha256=sha(out/'marginal_shuffle.svg'),status='rendered_from_frozen_summary_without_refitting'),indent=2)+'\n')
