"""Descriptive comparison of all measured H2bo/H2bp complete-cost pairs."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.hashsalt']='h2bo-h2bp-complete-cost-comparison'
ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'results/research/work/endpoint-cost-comparison'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def render():
    DEST.mkdir(exist_ok=True)
    cohorts=[];inputs={}
    for name,label in [('bp-first-fresh-confirmation','Corner endpoints (H2bo)'),('bp-first-endpoint-shift','Uniform open-cell endpoints (H2bp)')]:
        d=ROOT/'results/research/work'/name;a=json.loads((d/'audit.json').read_text())
        assert a['status']=='passed' and a['fresh_independent_inputs']==4096
        for filename in ['audit.json','protocol.json']:
            path=d/filename;inputs[str(path.relative_to(ROOT))]=sha(path)
        pairs=[]
        for pair in range(4):
            times={}
            for arm in ['LP96','BP_geometry_LP96']:
                path=d/'runs'/f'pair{pair}-{arm}'/'result.json';c=json.loads(path.read_text());inputs[str(path.relative_to(ROOT))]=sha(path)
                assert c['status']=='complete' and c['model_inputs']==12288
                total=c['model_readiness_seconds']+sum(r['timing']['answer_seconds'] for r in c['rows'])
                assert abs(total-c['complete_answer_seconds'])<1e-8;times[arm]=total
            saving=1-Fraction(times['BP_geometry_LP96'])/Fraction(times['LP96'])
            pairs.append({'pair':pair,'LP96_seconds':times['LP96'],'cascade_seconds':times['BP_geometry_LP96'],'exact_saving':str(saving),'saving':float(saving)})
        values=sorted(Fraction(p['exact_saving']) for p in pairs);median=(values[1]+values[2])/2
        assert str(median)==a['cost_endpoint']['exact_median_fraction']
        cohorts.append({'cohort':name,'label':label,'pairs':pairs,'exact_median_saving':str(median),'median_saving':float(median),'endpoint_decision':a['cost_endpoint']['decision']})
    fig,axes=plt.subplots(1,2,figsize=(11,5.4),sharey=True)
    for ax,c in zip(axes,cohorts):
        values=[100*p['saving'] for p in c['pairs']];median=100*c['median_saving'];color='#087f8c' if median>=0 else '#be4d25'
        ax.axhline(0,color='#444444',linewidth=1)
        ax.bar(range(4),values,width=.62,color=color,alpha=.85)
        ax.axhline(median,color=color,linestyle='--',linewidth=1.6)
        for i,v in enumerate(values):ax.text(i,v+1.2 if v>=0 else -20,f'{v:+.2f}%',ha='center',va='bottom' if v>=0 else 'top',fontsize=10)
        ax.text(.02,.96,f'Median {median:+.2f}% saving' if median>=0 else f'Median {-median:.2f}% higher cost',transform=ax.transAxes,ha='left',va='top',fontsize=11,color=color)
        ax.set_title(c['label'],loc='left',fontsize=11,pad=12)
        ax.set_xticks(range(4),['GPU0\npair0','GPU0\npair1','GPU1\npair2','GPU1\npair3'])
        ax.set_ylim(-25,47);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Complete answer-cost saving versus standalone LP96 (%)')
    fig.suptitle('The BP-first cascade has opposite cost outcomes on two fresh cohorts',x=.02,ha='left',fontsize=13)
    fig.text(.02,.09,'Each cohort: 4,096 fresh mazes × 3 fixed models; all four paired timings retained. Positive = saving; negative = added cost.',fontsize=9)
    fig.text(.02,.055,'Complete costs charge model readiness, transfer, hidden/solver, geometry, certificates and scatter. BP is an executed prefix.',fontsize=9)
    fig.text(.02,.02,'Distinct endpoint distributions; no pooling, causal distance attribution or timing confidence interval. Archives/barriers excluded.',fontsize=9)
    fig.tight_layout(rect=[0,.15,1,.92])
    fig.savefig(DEST/'endpoint-cost-comparison.png',dpi=160,metadata={'Software':'Matplotlib'})
    fig.savefig(DEST/'endpoint-cost-comparison.svg',metadata={'Date':None});plt.close(fig)
    (DEST/'provenance.json').write_text(json.dumps({'status':'complete','renderer_sha256':sha(Path(__file__)),'source_sha256':inputs,'cohorts':cohorts,'scope':'Descriptive two-cohort comparison, all fixed paired costs and exact medians. No pooled accuracy or causal distance claim. No additional experiment or scientific observations.'},indent=2)+'\n')


if __name__=='__main__':render()
