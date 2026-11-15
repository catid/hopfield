"""Independent SciPy reconstruction of H2bk complete public outcome tables."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from scipy.stats import beta


def verify(directory):
    started=time.perf_counter();a=json.loads((directory/'audit.json').read_text());n=4096
    with (directory/'per-input.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==n and [int(r['input_index']) for r in rows]==list(range(n))
    names=['pl','unified96','hybrid','dual_certificate'];joint={name:np.array([int(r[name]) for r in rows],bool) for name in names}
    with (directory/'per-model-input.csv').open() as stream:mr=list(csv.DictReader(stream))
    assert len(mr)==49152;model={name:np.zeros((3,n),bool) for name in names};seen=set()
    for row in mr:
        key=(row['arm'],int(row['model_seed']),int(row['input_index']));assert key not in seen;seen.add(key)
        assert row['success'] in ['0','1'];model[key[0]][key[1],key[2]]=bool(int(row['success']))
    assert all(np.array_equal(x.all(0),joint[name]) for name,x in model.items())
    assert {name:int(x.sum()) for name,x in joint.items()}==a['joint_success']
    assert {name:int(x.sum()) for name,x in model.items()}==a['model_success']
    def bounds(k,alpha):return (float(beta.ppf(alpha,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha,k+1,n-k)) if k<n else 1.)
    checked={};max_error=0.
    for key,name in [('hybrid_absolute','hybrid'),('strict_absolute','dual_certificate')]:
        k=int(joint[name].sum());lo,hi=bounds(k,.05);r=a['primary_endpoints'][key]
        assert r['successes']==k and r['n']==n and r['minimum_success_pp']==99.9
        error=max(abs(100*lo-r['one_sided95_success_lower_pp']),abs(100*hi-r['one_sided95_success_upper_pp']));assert error<1e-8;max_error=max(max_error,error)
        decision='supported' if lo>=.999 else 'falsified' if hi<.999 else 'inconclusive';assert r['decision']==decision
        checked[key]={'successes':k,'lower_pp':100*lo,'upper_pp':100*hi,'decision':decision}
    g=int((joint['hybrid']&~joint['pl']).sum());l=int((~joint['hybrid']&joint['pl']).sum());gl,gu=bounds(g,.025);ll,lu=bounds(l,.025);lower=100*(gl-lu);upper=100*(gu-ll);r=a['primary_endpoints']['bp_noninferiority']
    assert r['gains']==g and r['losses']==l and r['n']==n and r['minimum_net_gain_pp']==-.1
    error=max(abs(lower-r['separately_one_sided95_net_lower_pp']),abs(upper-r['separately_one_sided95_net_upper_pp']));assert error<1e-8;max_error=max(max_error,error)
    decision='supported' if lower>=-.1 else 'falsified' if upper<-.1 else 'inconclusive';assert decision==r['decision'];checked['bp_noninferiority']={'gains':g,'losses':l,'lower_pp':lower,'upper_pp':upper,'decision':decision}
    c=a['cost_endpoint'];ratio=100*(c['main_hybrid_charged_seconds_including_bp_load']-c['lp96_charged_seconds'])/c['lp96_charged_seconds'];assert ratio==c['main_overhead_percent'] and c['decision']==('supported' if ratio<=10 else 'falsified')
    return {'status':'passed','joint_inputs':n,'model_maze_arm_rows':len(mr),'endpoints':checked,'maximum_error_pp':max_error,'seconds':time.perf_counter()-started,
            'scope':'Independent SciPy beta quantiles and every published model/input join. Four separately labelled endpoints; no familywise claim. Cost arithmetic is checked, raw timing receipts audited separately.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);print(json.dumps(verify(parser.parse_args().directory),indent=2))
