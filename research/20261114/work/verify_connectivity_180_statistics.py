"""Independent public SciPy beta-quantile verification of H2am endpoints."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bounds(k,n,a):return (0. if k==0 else float(beta.ppf(a,k,n-k+1)),1. if k==n else float(beta.ppf(1-a,k+1,n-k)))


def main(directory):
    summary=directory/'connectivity-transfer-180-summary.json';s=json.loads(summary.read_text());path=directory/'connectivity-transfer-180-per-input.csv';rows=list(csv.DictReader(path.open()));assert len(rows)==4096 and [int(r['input_index']) for r in rows]==list(range(4096));result=[]
    for name,p in s['primary_endpoints'].items():
        a=np.array([int(r['joint_'+p['candidate']+'_success']) for r in rows],bool);b=np.array([int(r['joint_'+p['control']+'_success']) for r in rows],bool);g=int((a&~b).sum());l=int((~a&b).sum());assert (g,l)==(p['gains'],p['losses']);gl,gu=bounds(g,4096,.025);ll,lu=bounds(l,4096,.025);lower,upper=100*(gl-lu),100*(gu-ll);error=max(abs(lower-p['separately_one_sided95_net_lower_pp']),abs(upper-p['separately_one_sided95_net_upper_pp']));assert error<1e-8
        if 'target_gain_pp' in p:decision='supported' if lower>=1 else 'falsified' if upper<1 else 'inconclusive'
        else:
            low,high=bounds(l,4096,.05);assert abs(100*low-p['loss_one_sided95_lower_pp'])<1e-8 and abs(100*high-p['loss_one_sided95_upper_pp'])<1e-8;decision='supported' if high*100<=.1 else 'falsified' if low*100>.1 else 'inconclusive'
        assert decision==p['decision'];result.append({'endpoint':name,'gains':g,'losses':l,'one_sided95_net_lower_pp':lower,'one_sided95_net_upper_pp':upper,'maximum_error_pp':error,'decision':decision})
    value={'status':'passed','rows':result,'method':'Independent SciPy betaquantiles, all4096 pairedrows. Two separately decided primary endpoints; no joint95claim.','source_sha256':sha(Path(__file__)),'summary_sha256':sha(summary),'per_input_sha256':sha(path)};(directory/'connectivity-transfer-180-statistics-verification.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);main(p.parse_args().directory)
