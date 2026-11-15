"""Exact proof coverage on all fixed H4aa strong-only Sudoku supports."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE='results/research/foureyes/strong_support_soundness/summary.json'
s=json.loads((ROOT/SOURCE).read_text());assert s['status']=='validated'and s['counts']=={'sound_proof':64,'unsound_witness':0,'unresolved':0}
direct=sum(r['missing_true']==0 for r in s['rows']);implied=64-direct
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'strong-support-soundness-v1'})
fig,axes=plt.subplots(1,2,figsize=(12.8,6.5),gridspec_kw={'width_ratios':[1.1,1]})
axes[0].barh([0],[direct],color='#527eac',label='All peer edges retained')
axes[0].barh([0],[implied],left=[direct],color='#287f77',label='Missing edges logically implied')
axes[0].text(direct/2,0,str(direct),color='white',ha='center',va='center',weight='bold',fontsize=22)
axes[0].text(direct+implied/2,0,str(implied),color='white',ha='center',va='center',weight='bold',fontsize=22)
axes[0].set(xlim=(0,64),ylim=(-1,1),yticks=[],xlabel='Fixed source-bank graphs',title='64 / 64 have complete soundness proofs')
axes[0].legend(loc='upper left',bbox_to_anchor=(-.02,-.24),frameon=False,fontsize=10)
axes[0].spines['left'].set_visible(False)
axes[1].axis('off')
axes[1].text(.02,.9,'The independent proof checks',fontsize=15,weight='bold')
axes[1].text(.02,.72,'Nine-clique → every symbol appears once',fontsize=12)
axes[1].text(.02,.56,'Cell-count identities → missing unit follows',fontsize=12)
axes[1].text(.02,.40,'All 27 units certified → valid Sudoku',fontsize=12)
axes[1].text(.02,.17,'37 identity steps checked\n31 damaged proofs rejected',fontsize=13,weight='bold',color='#287f77',linespacing=1.5)
fig.suptitle('Some missing constraints are logically redundant',x=.065,ha='left',fontsize=19,weight='bold')
fig.text(.065,.06,'All 64 H4aa reduced supports retained; 59 distinct full graphs. Exact claim for every zero-energy nine-symbol coloring.\n'
         'Geometry is supplied only to the proof audit. No new fitting, puzzle inputs or GPU inference. BP AdamW is outside this fixed-model audit.\n'
         'Soundness does not guarantee a compatible answer exists, successful search, or validity of a capped nonzero-energy return.\n'
         'Original metadata failure, corrected verifier and byte-identical complete proof replay are preserved.',fontsize=10,color='#4d5967',linespacing=1.5)
fig.subplots_adjust(left=.065,right=.98,bottom=.36,top=.82,wspace=.20)
for ext in ['png','svg']:fig.savefig(ROOT/f'assets/figures/strong-support-soundness.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
(ROOT/'assets/figures/strong-support-soundness-provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':{SOURCE:hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest()}},indent=2)+'\n')
