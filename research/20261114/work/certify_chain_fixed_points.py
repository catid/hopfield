"""Independent all-source native/chain-veto fixed-point control for H2ao."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from verify_training_feasibility import energy,flip,LD
from verify_graph_block_source import geometry,connected


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def one(args):
    root,parent,p,job=args;source=parent/'runs'/job['id']/'outputs.npz';assert sha(source)==p['source_outputs'][job['id']]['sha256'];ds=p['datasets'][job['shard']]
    assert sha(root/ds['path'])==ds['sha256']
    with np.load(root/ds['path']) as z:t=z['tokens']
    with np.load(source) as z:b=z['predictions'][3];a=z['couplings'].astype(LD);assert z['stopped'][3].all() and z['finite'][3].all() and not z['block_cap'][1].any() and not z['block_native_cap'][1].any();reported=z['minimum_flip_margin64'][3]
    mp=parent/f'model-seed{job["seed"]}-old_binary.npz';assert sha(mp)==p['model_arrays'][str(job['seed'])]['sha256']
    with np.load(mp) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    delta=flip(t,b,a,theta,mu,ao);margin=np.where(t==1,delta,np.inf).reshape(256,-1).min(1);err=float(abs(margin-reported).max());assert err<1e-9 and margin.min()>=-1e-8;items=[];rejected=0
    for i,board in enumerate(b):
        raw,_=geometry(t[i],board)
        for cells in raw:
            changed=board.copy();changed.flat[list(cells)]=False
            if connected(t[i],changed):items.append((i,cells))
            else:rejected+=1
    smallest=None
    for lo in range(0,len(items),32):
        part=items[lo:lo+32];owners=np.array([i for i,_ in part]);old=b[owners];new=old.copy()
        for j,(_,cells) in enumerate(part):new[j].flat[list(cells)]=False
        actual=energy(t[owners],new,a[owners],theta,mu,ao)-energy(t[owners],old,a[owners],theta,mu,ao);assert np.isfinite(actual).all() and actual.min()>=-1e-8;v=float(actual.min());smallest=v if smallest is None else min(smallest,v)
    return {'job':job['id'],'status':'fixed_point_verified','inputs':256,'minimum_native_flip':float(margin.min()),'maximum_reported_margin_error':err,'admitted_chain_candidates':len(items),'connectivity_rejected_chain_candidates':rejected,'minimum_admitted_literal_delta':smallest,'source_outputs_sha256':sha(source),'caps':0,'nonfinite':0}


def main(directory,parent,root):
    started=time.perf_counter();p=json.loads((directory/'protocol.json').read_text());rows=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for r in pool.map(one,[(root,parent,p,j) for j in p['jobs']]):rows.append(r);print(json.dumps({'event':'chain_fixed_point_verified','job':r['job']}),flush=True)
    minima=[r['minimum_admitted_literal_delta'] for r in rows if r['minimum_admitted_literal_delta'] is not None];result={'status':'passed','protocol_sha256':sha(directory/'protocol.json'),'rows':rows,'source_model_inputs':12288,'admitted_chain_candidates':sum(r['admitted_chain_candidates'] for r in rows),'connectivity_rejected_chain_candidates':sum(r['connectivity_rejected_chain_candidates'] for r in rows),'minimum_native_flip':min(r['minimum_native_flip'] for r in rows),'minimum_admitted_literal_delta':min(minima) if minima else None,'maximum_reported_margin_error':max(r['maximum_reported_margin_error'] for r in rows),'new_searches':0,'seconds':time.perf_counter()-started,'source_sha256':sha(Path(__file__)),'scope':'Every retained native margin and every terminal chain-veto candidate independently checked; no cap-terminal or nonfinite case admitted asfixedpoint. The unchanged oldpolicy hasnoallowedfirstmove, henceextrabudget alone cannotchange its output. Zero new search control.'};(directory/'old-chain-fixed-point-control.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('directory',type=Path);a.add_argument('parent',type=Path);a.add_argument('root',type=Path);v=a.parse_args();main(v.directory,v.parent,v.root)
