"""Independent SciPy checks of regression probability and conditional retention."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bounds(k,n):return (0. if k==0 else float(beta.ppf(.05,k,n-k+1)),1. if k==n else float(beta.ppf(.95,k+1,n-k)))


def main(directory):
    sp=directory/'safe-block-transfer-summary.json';pp=directory/'safe-block-transfer-per-input.csv';s=json.loads(sp.read_text());rows=list(csv.DictReader(pp.open()));assert len(rows)==4096 and [int(r['input_index']) for r in rows]==list(range(4096));get=lambda arm:np.array([int(r['joint_'+arm+'_success']) for r in rows],bool);base=get('old_binary');old=get('old_blocks');veto=get('connectivity_veto');loss=int((base&~veto).sum());lo,hi=bounds(loss,4096);r=s['primary_endpoints']['veto_regression_probability'];assert loss==r['regressions'] and abs(lo-r['one_sided95_lower'])<1e-10 and abs(hi-r['one_sided95_upper'])<1e-10;decision='supported' if hi<=.001 else 'falsified' if lo>.001 else 'inconclusive';assert decision==r['decision'];reg={'regressions':loss,'n':4096,'one_sided95_lower':lo,'one_sided95_upper':hi,'decision':decision}
    opportunity=old&~base;n=int(opportunity.sum());k=int((opportunity&veto).sum());r=s['primary_endpoints']['retain_original_gains'];assert (n,k)==(r['opportunities'],r['retained'])
    if n:
        lo,hi=bounds(k,n);assert abs(lo-r['one_sided95_lower'])<1e-10 and abs(hi-r['one_sided95_upper'])<1e-10;decision='supported' if lo>=.9 else 'falsified' if hi<.9 else 'inconclusive'
    else:lo=hi=None;decision='inconclusive';assert r['one_sided95_lower'] is r['one_sided95_upper'] is None
    assert decision==r['decision'];ret={'opportunities':n,'retained':k,'one_sided95_lower':lo,'one_sided95_upper':hi,'decision':decision};result={'status':'passed','regression':reg,'conditional_retention':ret,'summary_sha256':sha(sp),'per_input_sha256':sha(pp),'source_sha256':sha(Path(__file__)),'method':'Independent SciPy betaquantiles/all4096pairedrows. Separatelyone-sided95CPalpha.05; retentionconditions onoriginalblockgain event. No simultaneous95claim; zeroopportunitiesinconclusive.'};(directory/'safe-block-transfer-statistics-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);main(p.parse_args().directory)
