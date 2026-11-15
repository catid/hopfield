"""Exact structural census figure using tracked exports only."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];DEST=ROOT/'results/research/loki/h3q'
def main():
    source=DEST/'summary.json';s=json.loads(source.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3q-public'})
    fig,axes=plt.subplots(1,2,figsize=(12,6.5));fig.subplots_adjust(left=.075,right=.98,bottom=.32,top=.73,wspace=.27)
    fig.suptitle('2,784 valid partial inputs have no learned certificate',x=.075,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.075,.87,'Exact structural census over all 836,069 valid N15 inputs; this is not a search-accuracy measurement.',fontsize=10.7,fontweight='bold')
    fig.text(.075,.80,'Unchanged full-N14 hybrid and supplied oracle · 32,516 complete superqueen solutions · 1–3 given queens',fontsize=10.4)
    labels=['BP AdamW\nnot run','Full N14\nhybrid','Supplied\noracle'];axes[0].text(0,.18,'Not measured',ha='center',color='#777',fontsize=9)
    hybrid=next(r for r in s['rows'] if r['arm']=='n14-full' and r['clues']=='all')
    for k,arm,color in [(1,'n14-full','#19876b'),(2,'true-graph-oracle','#8c5eb5')]:
        row=next(r for r in s['rows'] if r['arm']==arm and r['clues']=='all');v=row['no_certificate_pct'];axes[0].bar(k,v,width=.6,color=color)
        axes[0].annotate(f'{v:.6f}%',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    axes[0].set_xticks(range(3),labels);axes[0].set_xlim(-.55,2.55);axes[0].set_title('A   Exact population fraction',loc='left',fontweight='bold',pad=12)
    values=[]
    for study in ['h3o','h3p']:
        row=next(r for r in s['cohort_crosschecks'] if r['study']==study and r['arm']=='n14-full');values.append(100*row['inputs_without_certificate']/row['inputs'])
    values.append(hybrid['no_certificate_pct'])
    for k,(v,color) in enumerate(zip(values,['#777','#2878b5','#19876b'])):
        axes[1].bar(k,v,width=.6,color=color);axes[1].annotate(f'{v:.6f}%',(k,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
    axes[1].set_xticks(range(3),['Initial\n4,096 inputs','Confirmation\n16,384 inputs','Entire finite\n836,069 inputs'],fontsize=9)
    axes[1].set_title('B   Structural fraction in each cohort',loc='left',fontweight='bold',pad=12)
    for ax in axes:ax.set_ylim(0,.42);ax.set_ylabel('Inputs without a valid certificate (%)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.text(.075,.235,'All 2,784 unsupported cases have three clues. Every valid one- or two-clue input has a certificate.',fontsize=10.6,fontweight='bold')
    fig.text(.075,.17,'The hybrid admits 17,462,500 of 18,696,700 compatible valid input–completion certificates.\nThe oracle admits every compatible completion. This census uses no GPU searches and changes no model.',fontsize=10,linespacing=1.5)
    fig.text(.075,.10,'A valid certificate may exist without finite search finding it. A fallback decoder may succeed without a certificate.\nConditional soundness of stopped boards therefore differs from certificate availability and measured search accuracy.',fontsize=9.2,color='#444',linespacing=1.5)
    fig.text(.075,.025,'Full population counts are exact for this finite universe; cohort bars show exact subset counts, not confidence intervals.\nPost hoc descriptive diagnostic, independently crosschecked against literal margins and both prior cohort audits.',fontsize=9,color='#444',linespacing=1.4)
    for ext in ['png','svg']:fig.savefig(DEST/f'certificate-census.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__)]},output_sha256={f'certificate-census.{ext}':hashlib.sha256((DEST/f'certificate-census.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (DEST/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt['output_sha256']))
if __name__=='__main__':main()
