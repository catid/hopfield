"""Publication figure from tracked H3e exports only."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
DEST=ROOT/'results/research/loki/h3e'


def main():
    source=DEST/'summary.json';summary=json.loads(source.read_text());assert summary['status']=='validated'
    with (DEST/'per_seed.csv').open() as handle:seeds=list(csv.DictReader(handle))
    models=[('adamw-decay',4000),('adam-nodecay',250),('adam-nodecay',1000),('adam-nodecay',4000),('sgd-pre',4000)]
    labels=['BP AdamW\n4,000','BP Adam\n250','BP Adam\n1,000','BP Adam\n4,000','SGD + precond.\n4,000']
    colors=['#c44e52','#2878b5','#2878b5','#2878b5','#ca8a04']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,5.8));fig.subplots_adjust(left=.07,right=.98,top=.75,bottom=.29,wspace=.24)
    fig.suptitle('Fixed n=10 recipes transfer to 12-Queens',x=.07,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.07,.88,'100,000 training inputs · 8,192 disjoint evaluation inputs · 20 chains · 3 search seeds · certificate-first selection',fontsize=10.5)
    for i,(arm,updates) in enumerate(models):
        values=[r for r in seeds if r['arm']==arm and int(r['updates'])==updates];assert len(values)==3
        failures=np.array([8192-int(r['certificate_first_correct']) for r in values])
        coverage=np.array([100*int(r['coverage_numerator'])/int(r['coverage_denominator']) for r in values])
        for ax,data in zip(axes,[failures,coverage]):
            ax.bar(i,data.mean(),color=colors[i],alpha=.78,width=.67)
            ax.scatter(i+np.linspace(-.12,.12,3),data,color='black',s=18,zorder=4,clip_on=False)
        axes[0].annotate(f'{failures.mean():.2f}',(i,max(failures)),xytext=(0,7),textcoords='offset points',ha='center')
        axes[1].annotate(f'{coverage.mean():.2f}%',(i,max(coverage)),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    for ax in axes:
        ax.set_xticks(range(5),labels);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2);ax.set_xlabel('Optimizer and number of parameter updates')
    failure_top=5*np.ceil((max(8192-int(row['certificate_first_correct']) for row in seeds)+3)/5)
    axes[0].set_ylim(0,failure_top);axes[0].set_yticks(np.arange(0,failure_top+1,5));axes[0].set_ylabel('Failed inputs out of 8,192');axes[0].set_title('A   N20 failures',loc='left',fontweight='bold',pad=12)
    axes[1].set_ylim(0,86);axes[1].set_ylabel('Pooled distinct-completion coverage (%)');axes[1].set_title('B   Coverage among 20 chains',loc='left',fontweight='bold',pad=12)
    oracle=summary['posthoc_uniform_solution_reference']['expected_pooled_coverage_pct']
    axes[1].axhline(oracle,linestyle=':',color='#666',linewidth=1)
    axes[1].text(.98,.98,f'Uniform-draw oracle expectation: {oracle:.2f}%',transform=axes[1].transAxes,ha='right',va='top',fontsize=8,color='#444')
    p=summary['primary']
    fig.text(.07,.16,f"Frozen Adam250 vs SGD4000 hypothesis: {summary['decision']}. Accuracy difference {p['accuracy_difference_pp']:+.4f} pp; coverage difference {p['coverage_difference_pp']:+.4f} pp.",fontsize=10.5)
    fig.text(.07,.105,'No n=12 tuning. Adam uses no decay; AdamW uses decay 0.01. Dots are search seeds, not training replicas or confidence intervals.\nGenerated input split; all compatible completion labels retained. The uniform-draw line is an offline reference, not an upper bound.',fontsize=9,color='#444',linespacing=1.5)
    measured=summary['optimization_validation']['timings_seconds']
    full_ms=np.median(measured['expanded_upstream'])*1000;aggregate_ms=np.median(measured['aggregated_binary'])*1000
    fig.text(.07,.035,f'Training optimization: exact sufficient statistics reduced one measured forward/backward pass from {full_ms:.2f} ms to {aggregate_ms:.3f} ms.\nThis operation-level measurement is separate from model quality and does not establish an end-to-end search speedup.',fontsize=9,color='#444',linespacing=1.4)
    for extension in ['png','svg']:fig.savefig(DEST/f'n12-transfer.{extension}',dpi=180,facecolor='white')
    inputs=[source,DEST/'per_seed.csv',Path(__file__)]
    receipt=dict(input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        output_sha256={f'n12-transfer.{extension}':hashlib.sha256((DEST/f'n12-transfer.{extension}').read_bytes()).hexdigest() for extension in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
