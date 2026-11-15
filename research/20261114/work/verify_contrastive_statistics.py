"""Independent SciPy beta-tail checks of separate frozen endpoints."""
import json
from scipy.stats import beta
import study as S


def bounds(k,n,alpha):return (0. if k==0 else float(beta.ppf(alpha,k,n-k+1)),1. if k==n else float(beta.ppf(1-alpha,k+1,n-k)))


def main():
    out=S.ROOT/'results/research/work';a=json.loads((out/'contrastive-training-summary.json').read_text());rows=[]
    for endpoint,key in [('ranking','ranking_primary'),('accuracy','accuracy_primary')]:
        p=a[key];g,l=p['gains'],p['losses'];gl,gu=bounds(g,4096,.025);ll,lu=bounds(l,4096,.025);lower,upper=100*(gl-lu),100*(gu-ll);error=max(abs(lower-p['separately_one_sided95_net_lower_pp']),abs(upper-p['separately_one_sided95_net_upper_pp']));assert error<1e-8
        gl2,gu2=bounds(g,4096,.0125);ll2,lu2=bounds(l,4096,.0125);row={'endpoint':endpoint,'gains':g,'losses':l,'one_sided95_net_lower_pp':lower,'one_sided95_net_upper_pp':upper,'maximum_primary_error_pp':error,'conservative_two_sided95_net_pp':[100*(gl2-lu2),100*(gu2-ll2)]}
        if endpoint=='accuracy':
            low,high=bounds(l,4096,.05);assert abs(100*low-p['loss_one_sided95_lower_pp'])<1e-8 and abs(100*high-p['loss_one_sided95_upper_pp'])<1e-8;row.update(loss_one_sided95_lower_pp=100*low,loss_one_sided95_upper_pp=100*high)
        rows.append(row)
    S.save(out/'contrastive-training-statistics-verification.json',{'status':'passed','rows':rows,'method':'IndependentSciPybetaquantiles. Primarynet endpoints are separately one-sided95bounds using marginalalpha.025. Descriptive two-sided95usesalpha.0125across4tails. NIuses one-sided95lossprobabilityalpha.05. Endpoints are separate and no simultaneous95coverageclaim is made.','source_sha256':S.sha(__file__),'audit_sha256':S.sha(S.OUT/'contrastive_training/audit.json')});S.emit('contrastive_statistics_verified',rows=rows)


if __name__=='__main__':main()
