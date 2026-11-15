"""Tracked eighteen-case structural grid and separately labeled target census."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch,Rectangle

ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3y'
def main():
    source=PUBLIC/'summary.json';census=PUBLIC/'q11-census.json';s=json.loads(source.read_text());c=json.loads(census.read_text())
    assert s['status']=='validated'and c['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3y-public'})
    fig=plt.figure(figsize=(13,9));ax=fig.add_axes([.24,.36,.71,.40])
    fig.suptitle('A unique minimum source rule can still fail to transfer',x=.06,y=.965,ha='left',fontsize=19,fontweight='bold')
    fig.text(.06,.9,'Frozen all-18 hypothesis falsified: 12 exact transfers, 4 empty sources, 1 ambiguity, 1 unique wrong graph.',fontsize=12,fontweight='bold')
    fig.text(.06,.845,'All families and sizes fixed before source enumeration · complete source banks · all 18 source outcomes sealed before target diagnosis',fontsize=10.5)
    colors={'unique_exact_transfer':'#badfce','empty_source':'#dedede','ambiguous_minimum':'#d4c3e3','unique_minimum_wrong_graph':'#edbd9e'}
    outcome={'unique_exact_transfer':'Unique; exact','empty_source':'Empty source','ambiguous_minimum':'9 minimum rules','unique_minimum_wrong_graph':'Unique; 288 missed edges'}
    for row in s['rows']:
        x=s['source_sizes'].index(row['source_n']);y=int(row['family_id'][1:]);ax.add_patch(Rectangle((x,y),1,1,facecolor=colors[row['outcome']],edgecolor='white',lw=4))
        ax.text(x+.5,y+.38,f"{row['source_patterns']:,} source boards",ha='center',va='center',fontsize=11,fontweight='bold')
        ax.text(x+.5,y+.72,outcome[row['outcome']],ha='center',va='center',fontsize=9.7)
    ax.set_xlim(0,3);ax.set_ylim(6,0);ax.set_xticks([.5,1.5,2.5],['N8 → N9','N10 → N11','N12 → N13'],fontsize=11,fontweight='bold');ax.xaxis.tick_top();ax.tick_params(length=0,pad=10)
    ax.set_yticks([i+.5 for i in range(6)],['Queens + (1, 4)','Queens + (2, 3)','Queens + (2, 5)','Queens + (3, 4)','Queens + (1, 3), (2, 3)','Queens + (1, 4), (2, 3)'],fontsize=10.4)
    for spine in ax.spines.values():spine.set_visible(False)
    fig.legend(handles=[Patch(facecolor=colors[k],label=v)for k,v in [('unique_exact_transfer','Unique exact graph'),('empty_source','Empty source; retained failure'),('ambiguous_minimum','Ambiguous source minimum'),('unique_minimum_wrong_graph','Unique minimum, wrong graph')]],
               loc='lower left',bbox_to_anchor=(.055,.29),ncol=2,frameon=False,fontsize=10)
    fig.text(.06,.23,'Unique does not guarantee transfer: the 4 complete N10 camel+zebra boards select only the camel orbit.\nThe 12 complete N12 boards from the same family identify both orbits. No target-favored tie is selected in any case.',fontsize=10.8,linespacing=1.5)
    fig.text(.06,.14,f"Descriptive follow-up, after the structural result: the selected N11 rule admits {c['selected_rule_complete_boards']} complete boards; only {c['true_rule_complete_boards']} satisfy the true rule.\nAll {c['selected_but_invalid_complete_boards']} invalid zero-conflict boards are preserved and independently verified. This disproves certificate soundness, without estimating search rates.",fontsize=10.1,linespacing=1.5)
    fig.text(.06,.055,'Added priors: supplied source completeness, D4 symmetry and minimum description. Exhaustive cardinality-layer certificates check\nall 522 lower/equal-size subsets across nonempty cases. Source CPU 4.950 s plus generation 3.701 s; zero GPU searches or optimizer fits.',fontsize=9.7,color='#444',linespacing=1.5)
    for ext in['png','svg']:fig.savefig(PUBLIC/f'multifamily-stress.{ext}',dpi=180,facecolor='white',metadata={'Date':None}if ext=='svg'else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[source,census,Path(__file__)]},output_sha256={f'multifamily-stress.{ext}':hashlib.sha256((PUBLIC/f'multifamily-stress.{ext}').read_bytes()).hexdigest()for ext in['png','svg']})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])

if __name__=='__main__':main()
