"""Plot certified worst-branch depths, explicitly not minimum-depth estimates."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3bi'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated';plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3bi-public'})
    fig,axes=plt.subplots(1,2,figsize=(13,8));fig.subplots_adjust(left=.08,right=.95,bottom=.31,top=.70,wspace=.28)
    fig.suptitle('Four more queries suffice for every original unresolved checkpoint',x=.04,y=.96,ha='left',fontsize=18,fontweight='bold');fig.text(.04,.89,'SUPPORTED: 704/704 checkpoints admit a tree that identifies every remaining candidate.',fontsize=12,fontweight='bold');fig.text(.04,.82,'118 distinct candidate sets; complete signatures from all 73,712 N13 boards.\nTruth-blind compiler, exact certificates, zero new oracle labels.',fontsize=11.5,linespacing=1.5)
    for ax,key,title in zip(axes,['witness_worst_depth_group_counts','witness_worst_depth_case_counts'],['118 distinct candidate sets','704 original unresolved cases']):
        values=[s[key].get(str(d),0) for d in range(1,5)];ax.bar(range(1,5),values,color=['#91b6cb','#6c9db9','#487e9f','#275f84'],width=.65)
        for d,v in enumerate(values,1):ax.text(d,v+max(values)*.025,str(v),ha='center',fontsize=12)
        ax.set_title(title,fontsize=12);ax.set_xlabel('Worst depth of returned feasible tree');ax.set_xticks(range(1,5));ax.set_ylim(0,max(values)*1.18);ax.set_ylabel('Count');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
    fig.text(.04,.22,'These are feasible-tree depths, not minimum-depth claims. All remaining candidates, not only the true one, are covered.',fontsize=11.5,fontweight='bold');fig.text(.04,.14,'The previous unchanged eight-query heuristic left 37 of these examined cases unresolved.\nA better policy can avoid those ties within the same total label cap; new-bank robustness remains untested.',fontsize=11.3,linespacing=1.5);fig.text(.04,.055,f"1,021 memo states / 543 split attempts, including synthetic gates. Study {s['aggregate_CPU_seconds']:.3f} CPU s / {s['wall_seconds']:.3f} wall s.\nBP AdamW: not measured. Zero GPU work. Supplied finite candidate class and noiseless binary membership.",fontsize=10.7,color='#444',linespacing=1.5)
    for ext in ['png','svg']:fig.savefig(PUBLIC/f'four-query-feasibility.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [path,Path(__file__)]},output_sha256={f'four-query-feasibility.{ext}':hashlib.sha256((PUBLIC/f'four-query-feasibility.{ext}').read_bytes()).hexdigest() for ext in ['png','svg']});(PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
