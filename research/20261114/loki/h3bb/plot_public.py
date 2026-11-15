"""Plot realized label costs and retained random ambiguities from public evidence."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bb'


def main():
    paths=[PUBLIC/'summary.json',ROOT/'research/20261114/loki/h3bb/protocol.json'];s=json.loads(paths[0].read_text());p=json.loads(paths[1].read_text());assert s['status']=='validated';rows={(r['true_candidate'],r['method']):r for r in s['results']}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bb-public'})
    fig,ax=plt.subplots(figsize=(13.5,10));fig.subplots_adjust(left=.17,right=.94,bottom=.24,top=.75)
    fig.suptitle('Seventeen supplied aliases, at most three active queries',x=.06,y=.95,ha='left',fontsize=18,fontweight='bold')
    fig.text(.06,.88,'SUPPORTED under frozen four-query cap: active 17/17 identified; random control 0/17.',fontsize=12,fontweight='bold')
    fig.text(.06,.82,'Given complete N12 class + known candidate predictions + noiseless binary membership oracle.\nThis tests query selection on already-known N13 populations; stronger supervision than uniform positives.',fontsize=11,linespacing=1.5)
    ys=[];labels=[];row=0
    for c in p['classes']:
        first=row
        for cid in c['members']:
            a=rows[cid,'active'];b=rows[cid,'random'];ax.plot([a['labels_used'],b['labels_used']],[row,row],color='#ddd',lw=1.5,zorder=1)
            ax.scatter(a['labels_used'],row,s=75,c='#26796b',label='Active: unique model' if row==0 else None,zorder=3)
            ax.scatter(b['labels_used'],row,s=55,c='#ac5b49',marker='s',label='Random: ambiguous' if row==0 else None,zorder=3)
            ax.text(4.12,row,f"{len(b['remaining_candidates'])} remain",va='center',fontsize=10,color='#9b493a');ys.append(row);labels.append(str(cid));row+=1
        ax.axhspan(first-.48,row-.52,color='#26796b',alpha=.04,zorder=0);row+=.5
    ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_xlim(.5,5);ax.set_xticks([1,2,3,4]);ax.set_xlabel('Labels consumed before uniqueness or the frozen limit');ax.set_ylabel('True candidate ID');ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.08),ncol=2,frameon=False)
    fig.text(.06,.17,'Active: 34 labels total (mean 2.0). Random: 68 labels total; one frozen seed per truth, descriptive control.',fontsize=11,fontweight='bold')
    fig.text(.06,.11,'Every query was sealed before its answer; every compatible candidate and all 102 query/answer receipts were retained.\nAll five classes reach the binary decision-tree lower bound for worst-case depth, conditional on the supplied class.',fontsize=10.8,linespacing=1.5)
    fig.text(.06,.045,f"Initial complete-source class and candidate computation are not included in label cost. BP AdamW: not measured.\nMonitored study: {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s; 0 GPU s. No new-size or learned-optimizer claim.",fontsize=10.5,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'membership-queries.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths+[Path(__file__)]},output_sha256={f'membership-queries.{ext}':hashlib.sha256((PUBLIC/f'membership-queries.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
