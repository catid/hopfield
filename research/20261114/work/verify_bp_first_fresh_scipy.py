"""Independent SciPy reconstruction of frozen maze-level CP decisions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from scipy.stats import beta


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds(k,n,alpha):return (0. if k==0 else float(beta.ppf(alpha,k,n-k+1)),1. if k==n else float(beta.ppf(1-alpha,k+1,n-k)))


def verify(directory):
    started=time.perf_counter();a=json.loads((directory/'audit.json').read_text());assert a['status']=='passed'
    assert sha(directory/'per-input.csv')==a['per_input_rows_sha256']
    with (directory/'per-input.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==4096 and [int(r['input_index']) for r in rows]==list(range(4096))
    assert all(r[k] in ['0','1'] for r in rows for k in ['BP_AdamW','LP96','cascade'])
    successes=sum(r['cascade']=='1' for r in rows);gains=sum(r['cascade']=='1' and r['LP96']=='0' for r in rows);losses=sum(r['cascade']=='0' and r['LP96']=='1' for r in rows)
    lo,hi=bounds(successes,4096,.05);gl,gu=bounds(gains,4096,.025);ll,lu=bounds(losses,4096,.025)
    absolute=a['primary_endpoints']['absolute_joint_success_999'];ni=a['primary_endpoints']['paired_LP96_noninferiority']
    assert absolute['successes']==successes and ni['gains']==gains and ni['losses']==losses and absolute['n']==ni['n']==4096
    discrepancies=[abs(lo-absolute['one_sided95_success_lower_pp']/100),abs(hi-absolute['one_sided95_success_upper_pp']/100),abs(gl-lu-ni['separately_one_sided95_net_lower_pp']/100),abs(gu-ll-ni['separately_one_sided95_net_upper_pp']/100)]
    assert max(discrepancies)<1e-10
    absolute_decision='supported' if lo>=.999 else 'falsified' if hi<.999 else 'inconclusive'
    paired_decision='supported' if gl-lu>=-.001 else 'falsified' if gu-ll<-.001 else 'inconclusive'
    assert absolute_decision==absolute['decision'] and paired_decision==ni['decision']
    result={'status':'passed','n':4096,'successes':successes,'gains':gains,'losses':losses,'absolute_bounds':[lo,hi],
        'paired_net_bounds':[gl-lu,gu-ll],'absolute_decision':absolute_decision,'paired_decision':paired_decision,
        'maximum_probability_error':max(discrepancies),'seconds':time.perf_counter()-started,
        'checker_sha256':sha(Path(__file__)),'audit_sha256':sha(directory/'audit.json'),'per_input_sha256':sha(directory/'per-input.csv'),
        'scope':'Independent SciPy beta-quantile reconstruction from every maze-level row; separate95percent claims, no simultaneous coverage claim.'}
    (directory/'scipy-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);verify(parser.parse_args().directory)
