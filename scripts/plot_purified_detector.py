"""Public-data rendering of the matched-recall privileged H4aw intervention."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/research/foureyes/purified_detector/summary.json'
PROTOCOL=ROOT/'research/20261114/foureyes/purified-detector.json'
PARENT=ROOT/'results/research/foureyes/conflict_mask/summary.json'


def main():
    s=json.loads(DATA.read_text());parent=json.loads(PARENT.read_text())
    assert s['status']=='validated' and all(g['unresolved']==0 for g in s['grouped'])
    purified,random=s['grouped'];old=parent['grouped'][0]
    assert purified['flagged_locations']==random['flagged_locations'] and purified['false_positive_locations']==random['false_positive_locations']==0
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.fonttype':'none','svg.hashsalt':'h4aw-purified-detector'})
    fig,(a,b)=plt.subplots(1,2,figsize=(12,6.6));fig.subplots_adjust(left=.08,right=.97,top=.73,bottom=.32,wspace=.27)
    fig.text(.04,.95,'Equal mask precision and recall, 107 more graphs recovered',fontsize=18,weight='bold')
    fig.text(.04,.88,'Removing false positives leaves recovery at 130/256. Random true locations reach 237/256.',fontsize=12.5)
    values=[old['success'],purified['success'],random['success']]
    a.bar(range(3),values,color=['#92999e','#256d8d','#be7854'],width=.65)
    for i,v in enumerate(values):a.text(i,v+5,str(v),ha='center',weight='bold')
    a.axhline(s['primary']['purified_recovery']['required'],color='#9c452a',ls='--',lw=1)
    a.text(-.4,205,'Purified target: ≥200',color='#9c452a',fontsize=10)
    a.set_xticks(range(3),['Original\ndetector','Purified\ndetector','Matched\nrandom']);a.set_ylim(0,270)
    a.set_ylabel('Exact graphs / 256 examined banks')
    x=np.arange(4);w=.34
    for offset,g,color,label in [(-w/2,purified,'#256d8d','Purified detector'),(w/2,random,'#be7854','Matched random')]:
        y=[f['exact'] for f in g['families']];b.bar(x+offset,y,w,color=color,label=label)
        for i,v in enumerate(y):b.text(i+offset,v+1,str(v),ha='center',fontsize=10)
    b.set_xticks(x,['Cycle','Square','Cube','Cliques']);b.set_ylim(0,76);b.set_ylabel('Exact graphs / 64 banks per family')
    b.legend(frameon=False,fontsize=10,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2)
    recall=100*purified['flagged_locations']/purified['actual_corrupt_locations']
    fig.text(.04,.18,f'Both new arms: 100% precision, {recall:.2f}% recall, identical true-positive count in every row.\nBoth registered targets fail: purified recovery ≥200; absolute count difference ≤16 (observed 107).',fontsize=11,linespacing=1.5)
    fig.text(.04,.06,'Actual corruption locations supplied to both arms; this is a privileged mechanism test on examined data.\nAll 512 fits retained. No new puzzle or generalization claim. BP AdamW outside this source intervention.',fontsize=10.5,color='#444',linespacing=1.5)
    outputs=[]
    for ext in ['png','svg']:
        path=ROOT/f'assets/figures/purified-detector.{ext}';fig.savefig(path,dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None);outputs.append(path)
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [DATA,PROTOCOL,PARENT,Path(__file__)]},
                 output_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in outputs})
    (ROOT/'assets/figures/purified-detector-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
