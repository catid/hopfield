"""Plot exact-derived failure probabilities rather than rounded 100% success."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4];PUBLIC=ROOT/'results/research/loki/h3as'


def main():
    path=PUBLIC/'summary.json';s=json.loads(path.read_text());assert s['status']=='validated'
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'cram-h3as-public'})
    fig,ax=plt.subplots(figsize=(12,7.6));fig.subplots_adjust(left=.1,right=.66,bottom=.28,top=.75)
    fig.suptitle('16-label recovery is highly reliable under this supplied vocabulary',x=.065,y=.95,ha='left',fontsize=16.7,fontweight='bold')
    fig.text(.065,.88,'SUPPORTED: exact recovery probability exceeds 99% for both fixed finite populations',fontsize=11.8,fontweight='bold')
    fig.text(.065,.818,'Uniform distinct positive subsets; ordinary queens + 0 / 1 / 2 orbits from ten supplied D4 leaper rules',fontsize=10.8)
    for family,label,color in [(0,'Zebra (472 source boards)','#226f68'),(1,'(2,5) leaper (1,304 boards)','#b06432')]:
        rows=[r for r in s['rows'] if r['family']==family];failures=[(int(r['total_subsets'])-int(r['successful_subsets']))/int(r['total_subsets']) for r in rows]
        ax.plot([r['dose'] for r in rows],failures,'o-',color=color,label=label,lw=2,markersize=6)
    ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks([4,8,16,32,64],[4,8,16,32,64]);ax.set_ylim(1e-61,.04);ax.set_yticks([1e-2,1e-10,1e-20,1e-30,1e-40,1e-50,1e-60]);ax.set_xlabel('Distinct raw positive labels');ax.set_ylabel('Probability of failing to uniquely recover the true graph')
    ax.axhline(.01,color='#99413d',ls='--',lw=1.2);ax.axvline(16,color='#777',ls=':',lw=1.1);ax.text(17,1e-6,'Primary: 16 labels',fontsize=9.3,color='#444');ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,loc='lower left',fontsize=9.8)
    fig.text(.71,.70,'At 16 labels',fontsize=13,fontweight='bold')
    fig.text(.71,.62,'Zebra failure probability',fontsize=10.8,color='#226f68');fig.text(.71,.57,'6.938 × 10⁻¹³',fontsize=20,fontweight='bold',color='#226f68')
    fig.text(.71,.48,'(2,5) leaper failure probability',fontsize=10.8,color='#b06432');fig.text(.71,.43,'8.905 × 10⁻¹⁰',fontsize=20,fontweight='bold',color='#b06432')
    fig.text(.71,.32,'Exact fractions are retained.\nNo probability shown is zero.',fontsize=10.4,linespacing=1.6)
    fig.text(.065,.18,'The consistency envelope has the identical recovery event and probability; no likelihood-normalization advantage.',fontsize=10.8,fontweight='bold')
    fig.text(.065,.12,'All 5,120 joint union masses agree between integer Möbius inversion and grouped sampling DP.\nA public audit independently uses literal source enumeration and direct inclusion–exclusion. No BP fits or target scores.',fontsize=10.5,linespacing=1.5)
    fig.text(.065,.05,f"Conditional on two fixed source populations and the supplied 56-model prior; no generalization to broader rule vocabularies.\nStudy: {s['aggregate_CPU_seconds']:.6f} CPU s / {s['wall_seconds']:.6f} wall s; 0 GPU s. Doses 4 / 8 / 32 / 64 are descriptive.",fontsize=10.2,color='#444',linespacing=1.4)
    for ext in ('png','svg'):fig.savefig(PUBLIC/f'exact-recovery-reliability.{ext}',dpi=180,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    receipt=dict(input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,Path(__file__))},output_sha256={f'exact-recovery-reliability.{ext}':hashlib.sha256((PUBLIC/f'exact-recovery-reliability.{ext}').read_bytes()).hexdigest() for ext in ('png','svg')})
    (PUBLIC/'plot-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['output_sha256'])


if __name__=='__main__':main()
