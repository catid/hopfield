"""Public-only independent NumPy original energies, margins and graph states."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_hard_negative_feasibility import valid_shortest


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(directory):
    started=time.perf_counter();m=json.loads((directory/'manifest.json').read_text());assert len(m['arms'])==6;rows=[];max_energy=max_margin=0.;labels=0
    for r in m['fixtures']:
        path=directory/r['file'];assert sha(path)==r['sha256']
        with np.load(path) as z:t=z['tokens'];a=z['couplings'].astype(LD);pred=z['predictions'];theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs'];margin=z['minimum_flip_margin64'];e=z['energy64'];y=z['labels']
        assert valid_shortest(t[0],y[0]);labels+=1;out=[]
        for v in range(6):
            b=pred[v];assert np.array_equal(b[t!=1],(t>=2)[t!=1]);delta=flip(t,b,a,theta[v],mu,ao);actual=np.where(t==1,delta,np.inf).reshape(1,-1).min(1);err=float(abs(actual-margin[v]).max());assert err<1e-9;max_margin=max(max_margin,err);entropy=LD(float(np.log(np.float64(1)+np.float64(1e-12))))*(t==1).sum((1,2));actual_e=energy(t,b,a,theta[v],mu,ao)+entropy;err=float(abs(actual_e-e[v]).max());assert err<1e-6;max_energy=max(max_energy,err);out.append(valid_shortest(t[0],b[0]))
        rows.append({'file':r['file'],'graph_valid_shortest_by_arm':out})
    return {'status':'passed','fixtures':len(rows),'final_states':len(rows)*6,'valid_shortest_labels':labels,'maximum_energy_error':max_energy,'maximum_margin_error':max_margin,'rows':rows,'scope':'Deterministic publicsubset, nooutcomeselection. Full73728 final states and all candidate geometry/costs/transitions audited separately; both512 arms share lexical256 starts. NumPy only; noTorch/decoder.','manifest_sha256':sha(directory/'manifest.json'),'verifier_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-started}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();print(json.dumps(verify(a.directory),indent=2))
