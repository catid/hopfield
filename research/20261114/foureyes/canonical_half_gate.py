"""Prospective canonical-PL lambda/2 and positive-support source gate."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/research/20261114/foureyes/canonical_half_gate'
PROTOCOL=HERE/'canonical_half_gate.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists()
    p=json.loads((HERE/'canonical_pl_gate.json').read_text())
    p.update(id='canonical-PL-half-penalty-positive-support-gate',frozen_utc=datetime.now(timezone.utc).isoformat(),
        raw_lambda=1/6000,
        adaptive_reason='Afterlambda/10canonicalfits passednumericalchecks butlargestgap retained1/3edges in2fixtures, plan directpositiveL1support. Lambda/2 is motivated by prior exactorigin/momentumfixed-point analysis; it is not equivalent to the paperfiniteoptimizer.',
        next_scientific_design='Ifsourcegatespass, freeze64freshbanks: canonicallambda/2 positivesupport primary, canonicallambda/10 positivesupport secondary, conventionalBPAdamW32+fixedtransform, restrictedcount control. Primary2ppNIandzerowrongstops; nofreshdataaccessatgate.')
    p['source_sha256'][str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    p['source_sha256'][str((HERE/'canonical_pl_gate.py').relative_to(ROOT))]=sha(HERE/'canonical_pl_gate.py')
    p['conditions']+=['CPU64/GPU64 exactpositive-support identity andminimumpositivecoefficient>4*coefficientdifference']
    PROTOCOL.write_text(json.dumps(p,indent=2)+'\n');OUT.mkdir(parents=True)
    print(json.dumps(dict(protocol_sha256=sha(PROTOCOL))))


def gate(bank):
    import canonical_pl_gate as old
    old.OUT=OUT;old.PROTOCOL=PROTOCOL
    old.gate(bank)
    import numpy as np
    data=np.load(OUT/f'bank{bank}-coefficients.npz',allow_pickle=False)
    a,b=data['cpu64'],data['gpu64'];error=float(np.abs(a-b).max())
    same=np.array_equal(a>0,b>0);minimum=min(float(a[a>0].min()),float(b[b>0].min()))
    passed=same and minimum>4*error
    result=dict(status='passed' if passed else 'failed',bank=bank,positive_support_equal=same,
        min_positive=minimum,coefficient_error=error,positive_edges=int((a>0).sum()),protocol_sha256=sha(PROTOCOL))
    (OUT/f'positive-bank{bank}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert passed


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--bank',type=int)
    args=parser.parse_args()
    if args.freeze:freeze()
    else:gate(args.bank)
