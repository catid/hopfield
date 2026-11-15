"""Show all17 source aliases separating in actual N13 valid-state populations."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3ba'


def main():
    paths=[PUBLIC/'summary.json',PUBLIC/'N12-partition.json'];s=json.loads(paths[0].read_text());p=json.loads(paths[1].read_text());assert s['status']=='validated';classes=[c for c in p['classes'] if c['eligible_M_at_least16'] and len(c['members'])>1];counts={r['candidate']:r['states'] for r in s['stage_records']['N13']}
    plt.rcParams.update({'font.size':10.8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3ba-public'})
    fig,ax=plt.subplots(figsize=(14,11));fig.subplots_adjust(left=.31,right=.94,bottom=.25,top=.77)
    fig.suptitle('All five complete-source alias classes separate at N13',x=.06,y=.95,ha='left',fontsize=17,fontweight='bold')
    fig.text(.06,.88,'LOGICALLY SUPPORTED for all 1,511 eligible models: pooled complete N12/N13 evidence uniquely identifies each model.',fontsize=11.4,fontweight='bold')
    fig.text(.06,.818,'All 17 N12 aliases evaluated; 15 new N13 populations and the two known A/B populations. Zero N14 populations measured.',fontsize=10.8)
    colors=['#26796b','#447aa1','#b17738','#845e9a','#a24e45']
    positions=[];labels=[];row=0
    for i,c in enumerate(classes):
        first=row
        for cid in c['members']:
            value=counts[cid];ax.barh(row,value,height=.72,color=colors[i],zorder=3);ax.text(value+65,row,f'{value:,}',va='center',fontsize=10,color=colors[i]);positions.append(row);orbits=p['candidates'][cid]['orbits'];labels.append(f"{cid}: "+' + '.join(f'({a},{b})' for a,b in orbits));row+=1
        ax.axhspan(first-.48,row-.52,color=colors[i],alpha=.055,zorder=0);row+=.6
    ax.set_yticks(positions,labels);ax.invert_yaxis();ax.set_xlim(0,6100);ax.set_xlabel('Complete N13 valid-state count');ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True);ax.set_title('Candidate ID and added D4 orbits; shaded groups share the complete N12 population',loc='left',fontsize=10.6,fontweight='bold',pad=15)
    fig.text(.06,.18,'All 28 within-class model pairs have different N13 valid sets: 17 differ in both directions; 11 are one-sided.',fontsize=11,fontweight='bold')
    fig.text(.06,.12,'The added long-range constraints were redundant on complete N12 states but exclude actual valid boards at N13.\nComplete-source identification is stronger supervision than positive-only samples; one-sided set differences matter for label acquisition.',fontsize=10.6,linespacing=1.5)
    fig.text(.06,.045,f"1,494 models were already N12-unique; all 17 aliases become unique at N13. The 30 M12<16 models remain ineligible.\nNo surviving ambiguity triggers N14 enumeration, so this provides no measured N14 state census. Study: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.3,color='#444',linespacing=1.5)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'all-aliases-N13.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},output_sha256={f'all-aliases-N13.{ext}':hashlib.sha256((PUBLIC/f'all-aliases-N13.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
