"""Show alias prevalence and separate logical identification from measured N14 sets."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3az'


def main():
    paths=[PUBLIC/'summary.json',PUBLIC/'N12-partition.json'];s=json.loads(paths[0].read_text());partition=json.loads(paths[1].read_text());assert s['status']=='validated';classes=[c for c in partition['classes'] if c['eligible_M_at_least16'] and len(c['members'])>1]
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3az-public'})
    fig,axes=plt.subplots(1,2,figsize=(14,7.7),gridspec_kw={'width_ratios':[1.15,1]});fig.subplots_adjust(left=.08,right=.96,bottom=.31,top=.74,wspace=.35)
    fig.suptitle('A broader source partition finds five aliases; the declared truth grid resolves early',x=.06,y=.95,ha='left',fontsize=16.2,fontweight='bold')
    fig.text(.06,.88,'LOGICALLY SUPPORTED: all 25 eligible truths become unique models by N13. Zero N14 populations were measured.',fontsize=11.4,fontweight='bold')
    fig.text(.06,.818,'Complete-source supervision in a supplied 1,541-model class; 1,511 models have at least sixteen N12 solutions',fontsize=10.8)
    axes[0].bar(range(5),[len(c['members']) for c in classes],color=['#8a9ba3']*4+['#28796f'],width=.6)
    for i,c in enumerate(classes):axes[0].text(i,len(c['members'])+.15,f"{len(c['members'])} models\n{c['source_population_size']} states",ha='center',fontsize=9.5)
    axes[0].set_xticks(range(5),['(1,2)\nbase','(1,3)\nbase','(2,4)\nbase','(3,4)\nbase','Two-pair\nalias']);axes[0].set_ylim(0,7.2);axes[0].set_ylabel('Models with identical complete N12 populations');axes[0].set_title('A   All five eligible nontrivial source classes',loc='left',fontweight='bold',fontsize=11.5,pad=15);axes[0].grid(axis='y',alpha=.15);axes[0].set_axisbelow(True)
    axes[1].barh([2,1,0],[24,1,20],color=['#28796f','#5a99a5','#c2c2c2'])
    axes[1].set_yticks([2,1,0],['N12 already unique','N13 resolves alias','Source M < 16']);axes[1].set_xlim(0,28);axes[1].set_xlabel('Families in the fixed 45-pair truth grid');axes[1].set_title('B   No surviving ambiguity reaches N14',loc='left',fontweight='bold',fontsize=11.5,pad=15)
    for y,count in [(2,24),(1,1),(0,20)]:axes[1].text(count+.5,y,str(count),va='center',fontsize=11)
    axes[1].grid(axis='x',alpha=.15);axes[1].set_axisbelow(True)
    fig.text(.06,.21,'17 / 1,511 eligible candidate models belong to five nontrivial N12 classes; the remaining 1,494 are unique.',fontsize=11,fontweight='bold')
    fig.text(.06,.145,'Only the known {(1,2),(3,4)} ↔ {(1,2),(2,7)} class intersects an eligible truth in this declared radius-five pair grid.\nIts two N13 populations differ (1,188 versus 1,252 states). The other four alias classes remain descriptive outside the target scope.',fontsize=10.6,linespacing=1.5)
    fig.text(.06,.06,f"A unique model implies identified N14 behavior under the supplied class; this is not an N14 state-enumeration result.\nAll complete-source classes and truth eligibility cases retained. Study: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s.",fontsize=10.3,color='#444',linespacing=1.5)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'alias-prevalence.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[Path(__file__)]},output_sha256={f'alias-prevalence.{ext}':hashlib.sha256((PUBLIC/f'alias-prevalence.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
