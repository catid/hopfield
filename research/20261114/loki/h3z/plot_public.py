"""Plot the measured source-constraint mechanism from tracked H3z evidence."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3z'
def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3z-public'})
    fig=plt.figure(figsize=(12.5,8));ax=fig.add_axes([.27,.31,.67,.45])
    fig.suptitle('Complete empty sources reduce ambiguity, but do not repair transfer',x=.06,y=.965,ha='left',fontsize=17.5,fontweight='bold')
    fig.text(.06,.9,'Pooled N8 + N10, all six fixed families: 4 unique exact graphs, 1 unique wrong graph, 1 ambiguous minimum.',fontsize=11.3,fontweight='bold')
    fig.text(.06,.84,'The ambiguous family has no valid N8 boards. This complete-empty information reduces its N10-only minima from 9 to 3.',fontsize=10.5)
    labels=[]
    for k,row in enumerate(s['empty_source_constraint_mechanism']):
        labels.append(' + '.join(f'({a}, {b})'for a,b in row['generators']));v=row['unrejected_complete_N8_negatives'];keep=row['survives_pooled_minimum'];color='#19876b'if keep else'#9ba9ba'
        if v:ax.barh(k,v,height=.62,color=color)
        else:ax.scatter(0,k,color=color,s=55,zorder=4,clip_on=False)
        ax.text(v+1.2,k,'0 — retained'if keep else str(v),va='center',fontsize=10,color='#146c55'if keep else'#333')
    ax.set_yticks(range(9),labels,fontsize=10.4);ax.set_ylim(8.7,-.7);ax.set_xlim(0,80);ax.set_xlabel('Complete N8 boards incorrectly admitted by each N10-only minimum',labelpad=10)
    ax.set_ylabel('Candidate pair of D4 orbit generators',labelpad=12);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor='#19876b',label='Retained: rejects every N8 negative'),Patch(facecolor='#9ba9ba',label='Eliminated by complete-empty N8 source')],
               loc='lower left',bbox_to_anchor=(.055,.20),ncol=2,frameon=False,fontsize=10)
    fig.text(.06,.15,'Three minima remain; their N11 (false, missed) attack-pair counts are (0, 0), (168, 288) and (180, 280).\nTarget truth does not choose a tie. The separate camel+zebra family still uniquely chooses the unsound camel-only rule.',fontsize=10.2,linespacing=1.5)
    fig.text(.06,.06,'Adaptive mechanistic follow-up on previously diagnosed targets, not fresh validation. Completeness and D4/minimum-description\npriors remain supplied. All six source results sealed first; 217 subsets certified, 0.474 source CPU seconds, zero new searches or censuses.',fontsize=9.7,color='#444',linespacing=1.5)
    for ext in['png','svg']:fig.savefig(PUBLIC/f'pooled-completeness.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[path,Path(__file__)]},output_sha256={f'pooled-completeness.{ext}':hashlib.sha256((PUBLIC/f'pooled-completeness.{ext}').read_bytes()).hexdigest()for ext in['png','svg']})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])
if __name__=='__main__':main()
