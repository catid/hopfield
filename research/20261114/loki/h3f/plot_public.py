"""Compact H3f figure; reads only the tracked public JSON."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4]
DEST=ROOT/'results/research/loki/h3f'


def main():
    source=DEST/'summary.json';summary=json.loads(source.read_text());assert summary['status']=='validated'
    models=[('adamw-decay',4000),('adam-nodecay',250),('sgd-pre',4000),('exclusion',0),('training-bank',0)]
    labels=['BP AdamW\n4,000 updates','BP Adam\n250 updates','SGD + preconditioner\n4,000 updates',
            'Training-count exclusion\nfixed energy, no optimizer','Compatible training-bank\nretrieval baseline']
    colors=['#c44e52','#2878b5','#ca8a04','#19876b','#777777']
    rows=[next(r for r in summary['rows'] if r['arm']==arm and r['updates']==updates) for arm,updates in models]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(12,5.7));fig.subplots_adjust(left=.075,right=.98,top=.72,bottom=.30)
    fig.suptitle('12-Queens: training-count and retrieval controls',x=.075,y=.96,ha='left',fontsize=20,fontweight='bold')
    fig.text(.075,.875,'All 14,200 complete solution patterns occur in the fitting labels; evaluation holds out partial inputs.',fontsize=11,fontweight='bold')
    fig.text(.075,.815,'8,192 shared inputs · 20 chains × 3 seeds · certificate-first selection · fixed GPU search budget',fontsize=10.5)
    for i,(row,color) in enumerate(zip(rows,colors)):
        ax.bar(i,row['coverage_pct'],color=color,alpha=.85,width=.68,hatch='//' if row['arm']=='training-bank' else None)
        ax.annotate(f"{row['coverage_pct']:.2f}% coverage\n{row['certificate_first_accuracy_pct']:.3f}% N20",
                    (i,row['coverage_pct']),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,linespacing=1.4)
    ax.set_xticks(range(5),labels);ax.set_ylabel('Distinct-completion coverage (%)')
    ax.set_ylim(0,92);ax.set_yticks([0,20,40,60,80]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    p=summary['primary']
    fig.text(.075,.17,f"Fixed exclusion vs Adam250: same observed N20 accuracy, +{p['coverage_difference_pp']:.3f} pp coverage (one-sided 95% endpoints +{p['coverage_one_sided_95_lower_pp']:.3f} to +{p['coverage_one_sided_95_upper_pp']:.3f}).",fontsize=10.5)
    fig.text(.075,.105,'Exclusion support comes from training co-occurrences; rewarding allowed pairs is an added fixed energy-design prior.\nThe bank is oracle-style retrieval of compatible stored training solutions, not learned solution-pattern generalization.',fontsize=9.5,color='#444',linespacing=1.5)
    fig.text(.075,.035,'Bars average three search/sampling seeds; one fixed construction per method. Adaptive reuse of the n12 evaluation split.\nBank matching/sampling runs on CPU and has no iterative search; timings are reported separately without a cross-hardware speedup claim.',fontsize=9,color='#444',linespacing=1.4)
    for extension in ['png','svg']:fig.savefig(DEST/f'count-and-bank-controls.{extension}',dpi=180,facecolor='white')
    receipt=dict(input_sha256={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest(),
                              str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        output_sha256={f'count-and-bank-controls.{extension}':hashlib.sha256((DEST/f'count-and-bank-controls.{extension}').read_bytes()).hexdigest() for extension in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
