"""Bind prior sources/noise and independently check the supplied exact null."""
from decimal import Decimal,localcontext
from fractions import Fraction
import json
import numpy as np
from run import ROOT,HERE,sha,save,now
from adaptive_null_common import paths,load_protocol

def main():
    out,pp=paths();p=load_protocol(require_registration=False);assert not out.exists();out.mkdir()
    sources=json.loads((ROOT/p['inputs']).read_text());assert len(sources['inputs'])==32
    parent=json.loads((ROOT/'artifacts/research/20261114/ripper/permuted_noise/audit.json').read_text())
    for e in sources['inputs']:
        assert sha(ROOT/e['training_path'])==e['training_sha256']
        d=ROOT/'artifacts/research/20261114/ripper/permuted_noise'/f's{e["seed"]}'
        assert sha(d/'receipt.json')==parent['artifact_sha256'][str((d/'receipt.json').relative_to(ROOT))]
        r=json.loads((d/'receipt.json').read_text());assert r['training_source_sha256']==e['training_sha256']and r['artifact_sha256']['potential_noise.npz']==sha(d/'potential_noise.npz')
    threshold=np.load(ROOT/p['prior_thresholds'])['exact_law'];q0=Fraction(9,1600);q1=Fraction(1,9);a=q1/q0;b=(1-q1)/(1-q0)
    with localcontext()as ctx:
        ctx.prec=80;la=(Decimal(a.numerator)/a.denominator).ln();lb=(Decimal(b.numerator)/b.denominator).ln();bound=Decimal(648000).ln()
        for m,c in enumerate(threshold):
            c=int(c)
            if c<=m:assert c*la+(m-c)*lb>=bound
            if c>0:assert(c-1)*la+(m-c+1)*lb<bound
    assert q0*a+(1-q0)*b==1
    np.savez_compressed(out/'thresholds.npz',threshold=threshold)
    save(out/'input_registration.json',dict(status='registered_before_queries',utc=now(),protocol_sha256=sha(pp),preparer_sha256=sha(__file__),source_manifest_sha256=sha(ROOT/p['inputs']),threshold_sha256=sha(out/'thresholds.npz'),decimal_threshold_checks=1111,new_source_boards=0,new_calibrations=0,new_solver_inputs=0))
    print(json.dumps(dict(status='registered',banks=32)),flush=True)
if __name__=='__main__':main()
