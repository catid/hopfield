"""Public-only independent SciPy endpoint intervals forH2ae."""
import json
from scipy.stats import beta
import argparse
import hashlib
from pathlib import Path

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def bounds(k,n,alpha):return (0. if k==0 else float(beta.ppf(alpha,k,n-k+1)),1. if k==n else float(beta.ppf(1-alpha,k+1,n-k)))


def main(out):
    path=out/'second-negative-transfer-summary.json';a=json.loads(path.read_text());rows=[]
    for endpoint,key in [('gain','gain_primary'),('noninferiority','noninferiority_primary')]:
        p=a[key];g,l=p['gains'],p['losses'];gl,gu=bounds(g,4096,.025);ll,lu=bounds(l,4096,.025);lower,upper=100*(gl-lu),100*(gu-ll);error=max(abs(lower-p['separately_one_sided95_net_lower_pp']),abs(upper-p['separately_one_sided95_net_upper_pp']));assert error<1e-8;gl2,gu2=bounds(g,4096,.0125);ll2,lu2=bounds(l,4096,.0125);row={'endpoint':endpoint,'gains':g,'losses':l,'one_sided95_net_lower_pp':lower,'one_sided95_net_upper_pp':upper,'maximum_primary_error_pp':error,'conservative_two_sided95_net_pp':[100*(gl2-lu2),100*(gu2-ll2)]}
        if endpoint=='noninferiority':
            low,high=bounds(l,4096,.05);assert abs(100*low-p['loss_one_sided95_lower_pp'])<1e-8 and abs(100*high-p['loss_one_sided95_upper_pp'])<1e-8;row.update(loss_one_sided95_lower_pp=100*low,loss_one_sided95_upper_pp=100*high)
        rows.append(row)
    save(out/'second-negative-transfer-statistics-verification.json',{'status':'passed','rows':rows,'method':'IndependentSciPybetaquantiles. Primarynet intervalsseparatelyone-sided95viaalpha.025 marginaltails; descriptivetwo-sided95viaalpha.0125 across4tails. NIone-sided95lossboundsalpha.05. No simultaneous95claim.','source_sha256':sha(Path(__file__)),'public_summary_sha256':sha(path)});print(json.dumps(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);main(p.parse_args().directory)
