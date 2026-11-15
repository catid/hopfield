"""Public-only independent full-grid comparison for all512 fixed actual deletions."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,LD
from local_factor_delta import deltas


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();m=json.loads((directory/'manifest.json').read_text());assert m['candidate_count']==512;count=0;error=stored_error=0.
    for r in m['groups']:
        path=directory/r['path'];assert sha(path)==r['sha256']
        with np.load(path) as z:d={k:z[k] for k in z.files}
        t=d['tokens'];b=d['boards'];a=d['readout'].astype(LD);theta=d['theta'].astype(LD);mu=d['mu_offsets'];ao=d['aoffs'];items=[(int(owner),tuple(map(int,d['cells'][lo:hi]))) for owner,lo,hi in zip(d['owners'],d['cell_offsets'][:-1],d['cell_offsets'][1:])];local=deltas(t,b,a,theta,mu,ao,items);literal=[]
        for lo in range(0,len(items),32):
            part=items[lo:lo+32];owners=np.array([owner for owner,_ in part]);before=b[owners];after=before.copy()
            for j,(_,cells) in enumerate(part):after[j].flat[list(cells)]=False
            literal.extend(energy(t[owners],after,a[owners],theta,mu,ao)-energy(t[owners],before,a[owners],theta,mu,ao))
        literal=np.array(literal);error=max(error,float(abs(literal-local).max()));stored_error=max(stored_error,float(abs(literal-d['full_grid_costs']).max()));count+=len(items)
    assert count==512 and error<1e-9 and stored_error<1e-9
    return {'status':'passed','candidate_deletions':count,'maximum_local_vs_full_grid_error':error,'maximum_published_full_grid_error':stored_error,'manifest_sha256':sha(directory/'manifest.json'),'verifier_sha256':sha(Path(__file__)),'local_helper_sha256':sha(Path(__file__).with_name('local_factor_delta.py')),'seconds':time.perf_counter()-started,'scope':'Published fixtures only; independent full-grid original-factor evaluation versus local-factor cancellation. No production flip or quadratic scorer imports; no new inference or scientific samples.'}


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);print(json.dumps(verify(a.parse_args().directory),indent=2))
