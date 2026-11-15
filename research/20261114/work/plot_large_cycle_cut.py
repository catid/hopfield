"""Deterministic H2aw source-intervention figure; inherited controls explicitly labeled."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2aw-large-cycle-cut'
ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'results/research/work/large-cycle-cut'
a=json.loads((D/'audit.json').read_text())
labels=['BP AdamW native','LP binary','Connectivity veto','Energy cycle256','Lexical cycle256','Lexical256 → lexical512','Lexical256 → energy512']
arms=['pl','old_binary','connectivity_veto','energy256','lexical256','lexical512_from_lexical256','energy512_from_lexical256']
colors=['#778899']*5+['#137c8b','#4c9f70']
fig,ax=plt.subplots(figsize=(10,5.4));failures=[4096-a['joint_success'][k] for k in arms];bars=ax.barh(range(7),failures,color=colors);ax.set_yticks(range(7),labels);ax.scatter([0,0],[5,6],c=colors[5:],s=36,zorder=3,clip_on=False);ax.invert_yaxis();ax.set_xlabel('Joint failures among 4,096 previously seen mazes (all three fixed models required)');ax.set_xlim(0,190)
for i,v in enumerate(failures):ax.text(v+2,i,str(v),va='center',fontsize=11)
ax.set_title('Larger chain deletions rescue all remaining 180×180 source failures',loc='left',fontsize=13,pad=20);ax.spines[['top','right']].set_visible(False);ax.text(.99,.03,'Gray: archived H2av results\nColor: new cap512 continuations\nBoth new arms share the lexical256 start\n28/28 rescues; zero regressions each',transform=ax.transAxes,ha='right',va='bottom',fontsize=10,bbox={'facecolor':'white','edgecolor':'#dddddd','pad':8});fig.text(.015,.015,'Source mechanism experiment only: no fresh inputs. All 84 new moves per policy lower original energy. Earlier positive cuts remain in the prefix.',fontsize=9);fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(D/'large-cycle-cut.png',dpi=160,metadata={'Software':'Matplotlib'});fig.savefig(D/'large-cycle-cut.svg',metadata={'Date':None});plt.close(fig)
