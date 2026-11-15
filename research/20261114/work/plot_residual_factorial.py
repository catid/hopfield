"""Public-only H2bb disjoint-family source comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bb-residual-factorial'
ROOT=Path(__file__).resolve().parents[3];D=ROOT/'results/research/work/residual-factorial';a=json.loads((D/'audit.json').read_text());bp=json.loads((ROOT/'results/research/work/density-transfer-summary.json').read_text());labels=['BP AdamW native (archived)','H2ba source','Lexical family only','Veto-cap family only','Both disjoint branches'];values=[4096-bp['joint_success']['pl']]+[4096-a['joint_success'][k] for k in ['inherited_H2ba','lexical_family_only','veto_family_only','both_families_composed']];fig,ax=plt.subplots(figsize=(10,5.2));ax.barh(range(5),values,color=['#778899','#aab8c3','#5aa6b5','#79a877','#137c8b']);ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlim(0,185);ax.set_xlabel('Joint failures among 4,096 previously seen density50 mazes')
for i,v in enumerate(values):ax.text(v+2,i,str(v),va='center',fontsize=11)
ax.spines[['top','right']].set_visible(False);ax.set_title('Two disjoint stopping-cap families tested separately',loc='left',pad=18,fontsize=13);ax.text(.97,.08,'Five inputs per family; no overlap\nJoint arm is exact source-state composition\nNo additional joint-arm trajectories or interaction claim',transform=ax.transAxes,ha='right',fontsize=10,bbox={'facecolor':'white','edgecolor':'#dddddd','pad':8});fig.text(.02,.02,'Source intervention only; no fresh inputs. At most 16 unchanged lexical256 rounds per eligible state; full source prefix charged once.',fontsize=9);fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(D/'residual-factorial.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'residual-factorial.svg',metadata={'Date':None});plt.close(fig)
