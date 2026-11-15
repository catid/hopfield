"""Public H4av exact recovery and observable-mask quality."""
from pathlib import Path
import hashlib, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'results/research/foureyes/conflict_mask/summary.json'
PROTOCOL = ROOT/'research/20261114/foureyes/conflict-mask.json'
PARENT = ROOT/'results/research/foureyes/corruption_censor/summary.json'


def main():
    s = json.loads(DATA.read_text()); parent = json.loads(PARENT.read_text())
    assert s['status'] == 'validated' and all(g['unresolved'] == 0 for g in s['grouped'])
    detector, random = s['grouped']
    baseline = sum(r['baseline_exact'] for r in parent['rows'])
    full = sum(r['classification'] == 'success' for r in parent['rows'])
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none', 'svg.hashsalt':'h4av-conflict-mask'})
    fig, (a,b) = plt.subplots(1,2,figsize=(12,6.4),gridspec_kw={'width_ratios':[1.7,1]})
    fig.subplots_adjust(left=.08,right=.97,top=.73,bottom=.33,wspace=.29)
    fig.text(.04,.95,'Observable conflicts beat random masks, barely improve recovery',fontsize=17,weight='bold')
    fig.text(.04,.88,'130/256 graphs: absolute target fails; the +59 advantage over random passes.',fontsize=12.5)
    labels=['Original\ncounts','Conflict\ndetector','Random\n9-cell mask','Known\nlocations']
    values=[baseline,detector['success'],random['success'],full]
    bars=a.bar(range(4),values,color=['#8a9298','#256d8d','#b85838','#b8cbb9'],width=.65)
    for bar,v in zip(bars,values):a.text(bar.get_x()+bar.get_width()/2,v+5,str(v),ha='center',weight='bold')
    a.axhline(s['primary']['detector_exact']['required'],ls='--',color='#9c452a',lw=1)
    a.text(-.4,205,'Detector target: 200',fontsize=10,color='#9c452a')
    a.set_xticks(range(4),labels);a.set_ylim(0,285);a.set_ylabel('Exact graphs / 256 examined banks')
    precision=[100*g['true_positive_locations']/g['flagged_locations'] for g in [detector,random]]
    b.bar([0,1],precision,color=['#256d8d','#b85838'],width=.6)
    for i,v in enumerate(precision):b.text(i,v+2,f'{v:.2f}%',ha='center',weight='bold')
    b.set_xticks([0,1],['Detector','Random']);b.set_ylim(0,100);b.set_ylabel('Precision = recall (%)')
    fig.text(.04,.16,'Detector: 1 gain and 0 losses versus the original graphs. Random: 3 gains and 61 losses.\nKnown-location reference is privileged; original and known-location fits are inherited controls.',fontsize=11,linespacing=1.5)
    fig.text(.04,.065,'Same noisy rows and fitted graphs; supplied count of 9 corruptions per row. All 512 fits sealed before truth diagnostics.\nNo fresh generalization or puzzle-solving claim. BP AdamW outside this source intervention.',fontsize=10.4,color='#444',linespacing=1.5)
    outputs=[]
    for ext in ['png','svg']:
        path=ROOT/f'assets/figures/conflict-mask.{ext}'
        fig.savefig(path,dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
        outputs.append(path)
    sources=[DATA,PROTOCOL,PARENT,Path(__file__)]
    receipt=dict(input_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in sources},
                 output_sha256={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in outputs})
    (ROOT/'assets/figures/conflict-mask-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt['output_sha256']))


if __name__=='__main__':main()
