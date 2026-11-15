"""Display the full45-family grid including all20 insufficient source populations."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ax'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';orbits=list(itertools.combinations(range(1,6),2));matrix=np.full((10,10),np.nan);labels={}
    for row in s['family_results']:
        a,b=[orbits.index(tuple(o)) for o in row['true_orbits']]
        if row['eligible']:
            assert row['ML_exact_banks']==row['MAP_exact_banks'];count=row['MAP_exact_banks'];matrix[a,b]=1 if count==4 else 2 if count==3 else 3;labels[a,b]=f'{count}/4'
        else:matrix[a,b]=0;labels[a,b]=f"n={row['source_population_size']}"
    plt.rcParams.update({'font.size':10.7,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ax-public'})
    fig,ax=plt.subplots(figsize=(14,8.2));fig.subplots_adjust(left=.075,right=.62,bottom=.32,top=.75)
    fig.suptitle('Two-orbit truths reveal a source alias and a complexity-prior underfit',x=.065,y=.95,ha='left',fontsize=16.6,fontweight='bold')
    fig.text(.065,.88,'FALSIFIED: 23 / 25 eligible families pass; ML and MAP each recover exact graphs in 91 / 100 fresh banks',fontsize=11.7,fontweight='bold')
    fig.text(.065,.818,'All 45 radius-five true pairs retained; four 16-positive draws per eligible family; same 1,541-model candidate class',fontsize=10.8)
    cmap=ListedColormap(['#d5d5d5','#3d8c75','#d0a548','#b95748']);cmap.set_bad('white');ax.imshow(matrix,cmap=cmap,vmin=-.5,vmax=3.5,aspect='auto')
    for (a,b),label in labels.items():ax.text(b,a,label,ha='center',va='center',fontsize=9,color='white' if matrix[a,b] in (1,3) else '#222')
    ticks=[str(o).replace(' ','') for o in orbits];ax.set_xticks(range(10),ticks,rotation=45,ha='right');ax.set_yticks(range(10),ticks);ax.set_xlabel('Second true D4 orbit');ax.set_ylabel('First true D4 orbit');ax.set_title('Exact banks / 4 (identical ML and MAP counts)',loc='left',fontsize=11.5,fontweight='bold',pad=15)
    fig.legend(handles=[Patch(color='#3d8c75',label='4/4 exact'),Patch(color='#d0a548',label='3/4 exact'),Patch(color='#b95748',label='0/4 exact'),Patch(color='#d5d5d5',label='Ineligible: source n<16')],loc='lower left',bbox_to_anchor=(.065,.16),ncol=4,frameon=False,fontsize=10)
    fig.text(.67,.72,'Failure 1: exact source alias',fontsize=13,fontweight='bold',color='#a0473c')
    fig.text(.67,.68,'True {(1,2), (3,4)} and alternate\n{(1,2), (2,7)} share the identical\n120-board N12 population.\nBoth remain tied in all four banks.',fontsize=11,linespacing=1.6,va='top')
    fig.text(.67,.43,'Failure 2: a true orbit dropped',fontsize=13,fontweight='bold',color='#a0473c')
    fig.text(.67,.39,'True {(1,2), (3,5)}: MAP selects\nsingleton (1,2) in all four banks.\nML instead chooses spurious pairs.\nNeither recovers the true graph.',fontsize=11,linespacing=1.6,va='top')
    fig.text(.065,.105,'Exactness requires every selected tie to equal the true N13 graph. Graph difference does not by itself prove different valid-state sets.\nThis study measures graph identification only; no target capacity, state enumeration, BP fit or search-accuracy claim.',fontsize=10.6,linespacing=1.5)
    fig.text(.065,.04,f"Three eligible source populations contain exactly 16 boards, so their four draws have identical unordered label sets.\nStudy: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. All 154,100 candidate-score records and all ties retained.",fontsize=10.2,color='#444',linespacing=1.5)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'two-orbit-grid.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,Path(__file__))},output_sha256={f'two-orbit-grid.{ext}':hashlib.sha256((PUBLIC/f'two-orbit-grid.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
