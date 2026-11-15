"""Public-only deterministic H2ba source stopping/budget comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2ba-terminal-cycle'
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'results/research/work/terminal-cycle';a=json.loads((D/'audit.json').read_text());labels=['BP AdamW native (archived)','LP binary (archived)','Veto (archived)','Lexical256 (source)','Terminal certification only','Terminal +16 lexical rounds'];values=[4096-a['archived_comparisons']['joint_success'][k] for k in ['pl','old_binary','connectivity_veto']]+[4096-a['joint_success'][k] for k in ['inherited_lexical256','terminal_only','terminal_then_extra16']];fig,ax=plt.subplots(figsize=(10,5.5));ax.barh(range(6),values,color=['#778899']*4+['#73b7c2','#137c8b']);ax.set_yticks(range(6),labels);ax.invert_yaxis();ax.set_xlim(0,1750);ax.set_xlabel('Joint failures among 4,096 previously seen density50 mazes')
for i,v in enumerate(values):ax.text(v+18,i,str(v),va='center',fontsize=11)
ax.spines[['top','right']].set_visible(False);ax.set_title('Terminal bookkeeping and extra search have distinct source effects',loc='left',pad=18,fontsize=13);ax.text(.97,.08,'Known terminal-only: 10 joint rescues\nNew extra search: 52 more; zero regressions\n10 inherited veto-capped model states retained',transform=ax.transAxes,ha='right',fontsize=10,bbox={'facecolor':'white','edgecolor':'#dddddd','pad':8});fig.text(.02,.02,'Source intervention only; no fresh inputs. All three models must solve each maze. Full archived prefixes charged to both new arms.',fontsize=9);fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(D/'terminal-cycle.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'terminal-cycle.svg',metadata={'Date':None});plt.close(fig)
