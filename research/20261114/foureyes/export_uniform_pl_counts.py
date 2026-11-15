"""Independently reconstruct sufficient statistics for the public origin proof."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RAW=ROOT/'artifacts/research/20261114/foureyes/noisy_pl_long'
PUB=ROOT/'results/research/foureyes/uniform_pl_origin'


def main():
    start=time.monotonic();summary=json.loads((PUB/'summary.json').read_text());manifest=json.loads((RAW/'inputs.json').read_text())
    exposure=[];clean=[];noisy=[];totals=[];pairs=list(zip(*np.triu_indices(81,1)))
    for entry in manifest['banks']:
        path=ROOT/entry['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['input_sha256'][entry['path']]
        z=np.load(path);free=z['free'];A=[];Cs=[]
        for i,j in pairs:A.append(np.count_nonzero(free[:,i])+np.count_nonzero(free[:,j]))
        for source in ['solutions','noisy']:
            s=z[source];C=[]
            for i,j in pairs:
                equal=s[:,i]==s[:,j]
                C.append(np.count_nonzero(equal&free[:,i])+np.count_nonzero(equal&free[:,j]))
            Cs.append(C)
        for kind,C in zip(['clean','nine_corruptions'],Cs):
            old=np.load(RAW/f'origin-{entry["bank"]:02d}-{kind}.npz')
            assert np.array_equal(old['pair_exposure'],A) and np.array_equal(old['free_weighted_equal'],C)
        exposure.append(A);clean.append(Cs[0]);noisy.append(Cs[1]);totals.append(np.count_nonzero(free))
    path=PUB/'sufficient-counts.npz'
    np.savez_compressed(path,pair_exposure=np.asarray(exposure,dtype=np.int16),clean_equal=np.asarray(clean,dtype=np.int16),
                        noisy_equal=np.asarray(noisy,dtype=np.int16),free_totals=np.asarray(totals,dtype=np.int32))
    receipt=dict(status='independently_reconstructed',banks=len(exposure),pair_statistics_checked=len(exposure)*3240*3,
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest(),producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope='Only aggregate integer pair exposures/equalities and free-cell totals; no source boards or labels.',wall_s=time.monotonic()-start)
    (PUB/'count-export.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
