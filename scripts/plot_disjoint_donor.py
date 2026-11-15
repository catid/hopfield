"""Public H4ay/H4az repaired evaluation; all fixed comparisons and losses retained."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'assets/figures'
matplotlib.rcParams['svg.hashsalt']='h4az-disjoint-donor'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    paths=[ROOT/'results/research/foureyes/disjoint_donor_repair/summary.json',ROOT/'results/research/foureyes/crossfit_conflicts/summary.json']
    s,old=[json.loads(p.read_text()) for p in paths];g=s['grouped'][0];assert s['new_LPs']==0 and s['inherited_sealed_LPs']==512
    assert old['grouped'][0]['success']==38
    vals=[sum(r['baseline_exact'] for r in s['rows']),sum(r['parent_detector_exact'] for r in s['rows']),38,g['success']]
    fig,axs=plt.subplots(1,2,figsize=(11.6,4.8),gridspec_kw={'width_ratios':[1.6,1]})
    axs[0].bar(range(4),vals,color=['#7a8793','#167b91','#9d9690','#c85c37'],width=.66)
    axs[0].set_xticks(range(4),['Original\ncounts','Target-trained\ndetector','Opposite\n300-row half','Disjoint\n600-row donor'])
    for i,v in enumerate(vals):axs[0].text(i,v+4,str(v),ha='center',fontsize=12)
    axs[0].axhline(162,color='#6d526f',linestyle='--',linewidth=1.3);axs[0].text(.02,165,'Fixed target: at least 162',fontsize=10)
    axs[0].set_ylim(0,192);axs[0].set_ylabel('Exact graphs / 256 examined banks');axs[0].set_title('Recovery target fails',loc='left')
    changes=[g['gains_vs_detector'],g['losses_vs_detector']];axs[1].bar([0,1],changes,color=['#167b91','#c85c37'],width=.6)
    axs[1].set_xticks([0,1],['Gained successes','Lost successes'])
    for i,v in enumerate(changes):axs[1].text(i,v+1,str(v),ha='center',fontsize=12)
    axs[1].axhline(8,color='#6d526f',linestyle='--',linewidth=1.3);axs[1].text(-.3,10,'Loss limit: 8',fontsize=10)
    axs[1].set_ylim(0,62);axs[1].set_ylabel('Banks versus target-trained detector');axs[1].set_title('Regression target fails',loc='left')
    for ax in axs:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('A disjoint 600-row donor recovers 122 graphs, with 41 gains and 49 losses',x=.02,ha='left',fontsize=14)
    fig.text(.02,.08,'Repaired post-run evaluation: original audit failed on uint8 storage; all 512 sealed fits unchanged. Zero new fits.',fontsize=10)
    fig.text(.02,.025,'Same examined banks; inferred degree and nine flags per row supplied. Opposite-half result is prior-audited context. BP AdamW not run.',fontsize=9)
    fig.tight_layout(rect=[0,.13,1,.93]);OUT.mkdir(exist_ok=True)
    for ext in ['png','svg']:fig.savefig(OUT/f'disjoint-donor.{ext}',dpi=160,metadata={'Date':None} if ext=='svg' else {'Software':'Matplotlib'})
    plt.close(fig)
    (OUT/'disjoint-donor-provenance.json').write_text(json.dumps(dict(renderer_sha256=sha(Path(__file__)),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},scope='Repaired post-run evaluation of immutable H4ay fits; no fresh confirmation; prior-audited H4ax reference.'),indent=2)+'\n')
if __name__=='__main__':main()
