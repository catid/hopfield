"""Render noise replication from compact public exports only."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/research/foureyes/noise_replication'
OUT=ROOT/'assets/figures'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'hopfield-noise-replication-v1'})
rows=list(csv.DictReader((DATA/'per_bank.csv').open()))
fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={'width_ratios':[1.2,1]})
fig.suptitle('Nine corrupted digits per training board: sample size matters',fontweight='bold',fontsize=16,y=.99)
ax=axes[0]
x=np.arange(6)
for n,shift,color in [(300,-.17,'#b45309'),(3000,.17,'#047857')]:
 r=[z for z in rows if int(z['n'])==n];y=np.array([100*float(z['recovery_fraction']) for z in r])
 ax.bar(x+shift,y,width=.32,color=color,label=f'{n:,} training boards')
 if n==3000:
  lower=np.array([100*float(z['lower_one_sided']) for z in r])
  ax.errorbar(x+shift,y,yerr=np.array([y-lower,np.zeros(6)]),fmt='none',color='#111827',capsize=3)
 else:
  for xx,yy in zip(x+shift,y):ax.text(xx,yy+2,f'{yy:.1f}',ha='center',fontsize=9)
ax.set(xticks=x,xticklabels=[str(i) for i in x],xlabel='Fixed source bank',ylabel='Exact graph recovery (%)',ylim=(0,112))
ax.legend(loc='center right',frameon=True,facecolor='white',framealpha=1,edgecolor='#e2e8f0')
ax.set_title('512 independent corruption draws per bank',fontsize=11)
ax=axes[1]
small=[z for z in rows if int(z['n'])==300]
exact=sum(int(z['exact_graphs']) for z in small)
separable=sum(int(z['separable_draws']) for z in small)
total=sum(int(z['draws']) for z in small)
values=[exact,separable-exact,total-separable]
labels=['Exact graph recovered','A separator exists;\ngap rule misses it','No scalar count\nseparator exists']
colors=['#047857','#d97706','#64748b']
ax.barh(np.arange(3),[100*v/total for v in values],color=colors)
for i,v in enumerate(values):ax.text(100*v/total+1,i,f'{v:,} / {total:,}',va='center',fontsize=10)
ax.set(yticks=np.arange(3),yticklabels=labels,xlim=(0,105),xlabel='Fraction of 300-board draws (%)')
ax.invert_yaxis();ax.set_title('Post-fit diagnosis of the 300-board boundary',fontsize=11)
fig.text(.5,.025,'Nine free cells changed; clues preserved. Source banks held fixed. Bars are graph recovery, not solver accuracy.\nBlack whiskers: simultaneous one-sided 95% lower bounds across six banks (N = 3,000).',ha='center',fontsize=9,color='#475569')
fig.tight_layout(rect=(0,.13,1,.94),w_pad=2)
OUT.mkdir(exist_ok=True,parents=True)
for ext in ['png','svg']:fig.savefig(OUT/f'noise-replication.{ext}',dpi=180,metadata={'Date':None} if ext=='svg' else None)
paths=[DATA/'per_bank.csv',Path(__file__)]
(OUT/'noise-replication-provenance.json').write_text(json.dumps({'public_data_only':True,'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},indent=2)+'\n')
