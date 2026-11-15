"""H2az direct-factor/full-grid equivalence gate and immutable benchmark fixtures."""
import itertools
import json
import time
import numpy as np
import study as S
import fast_npz as Z
from verify_training_feasibility import energy,LD
from local_factor_delta import deletion_delta,deltas
DEST=S.OUT/'local_factor_preflight'


def main():
    DEST.mkdir(exist_ok=True);fixtures=DEST/'fixtures';fixtures.mkdir(exist_ok=True);started=time.perf_counter();rng=np.random.default_rng(2026111524);source=S.OUT/'large_cycle_cut';p=json.loads((source/'protocol.json').read_text());model=S.ROOT/p['model_arrays']['0']['path']
    with np.load(model) as z:mu=z['mu_offsets'];ao=z['aoffs']
    theta=rng.normal(size=18+len(ao)).astype(LD);a=rng.normal(size=(len(ao),3,3)).astype(LD);tok=np.ones((3,3),np.uint8);small_count=empty_count=0;max_small=0.
    for code in itertools.product(range(3),repeat=9):
        state=np.array(code).reshape(3,3);b=state>0;cells=np.flatnonzero(state==2);changed=b.copy();changed.flat[cells]=False;old=energy(tok[None],b[None],a[None],theta,mu,ao)[0];new=energy(tok[None],changed[None],a[None],theta,mu,ao)[0];v=deletion_delta(tok,b,a,theta,mu,ao,cells);err=float(abs(v-(new-old)));assert err<1e-10;max_small=max(max_small,err);small_count+=1;empty_count+=not len(cells)
        if small_count%1000==0:assert time.perf_counter()-started<120
    for shape in [(3,4),(4,3),(4,4)]:
        for rep in range(32):
            tok=rng.integers(0,2,size=shape,dtype=np.uint8);tok.flat[0]=2;tok.flat[-1]=3;b=np.where(tok==1,rng.integers(0,2,size=shape),tok>=2).astype(bool);cells=np.flatnonzero((tok==1)&b&(rng.random(shape)<.5));changed=b.copy();changed.flat[cells]=False;aa=rng.normal(size=(len(ao),*shape)).astype(LD);v=deletion_delta(tok,b,aa,theta,mu,ao,cells);literal=energy(tok[None],changed[None],aa[None],theta,mu,ao)[0]-energy(tok[None],b[None],aa[None],theta,mu,ao)[0];err=float(abs(v-literal));assert err<1e-10;max_small=max(max_small,err)
    groups=[];counts={'H2aw':0,'H2ay':0};max_actual=0.;max_published=0.
    def check_save(tag,t,b,a,theta,mu,ao,items,values,refs):
        nonlocal max_actual,max_published
        if not items:return
        local=deltas(t,b,a,theta,mu,ao,items);actual=[]
        for lo in range(0,len(items),32):
            part=items[lo:lo+32];owners=np.array([owner for owner,_ in part]);before=b[owners];after=before.copy()
            for j,(_,cells) in enumerate(part):after[j].flat[list(cells)]=False
            actual.extend(energy(t[owners],after,a[owners],theta,mu,ao)-energy(t[owners],before,a[owners],theta,mu,ao))
        error=float(abs(local-np.array(actual)).max());published=float(abs(local-np.array(values)).max());assert error<1e-9 and published<1e-8;max_actual=max(max_actual,error);max_published=max(max_published,published);counts[tag]+=len(items);path=fixtures/f'group{len(groups):03d}.npz';lengths=np.array([len(c) for _,c in items]);Z.save(path,tokens=t,boards=b,readout=a.astype(np.float32),theta=theta.astype(np.float64),mu_offsets=mu,aoffs=ao,owners=np.array([x[0] for x in items]),cell_offsets=np.r_[0,lengths.cumsum()],cells=np.concatenate([np.array(c) for _,c in items]),full_grid_costs=np.array(actual,dtype=np.float64));groups.append({'path':path.name,'sha256':S.sha(path),'campaign':tag,'candidate_count':len(items),'source_hashes':refs});assert time.perf_counter()-started<120
    for job in p['jobs']:
        ds=S.ROOT/p['datasets'][job['shard']]['path'];parent=S.ROOT/p['source_outputs'][job['id']]['path'];run=source/'lexical512/runs'/job['id'];receipt=json.loads((run/'result.json').read_text());mp=S.ROOT/p['model_arrays'][str(job['seed'])]['path']
        with np.load(ds) as z:t=z['tokens']
        with np.load(parent) as z:a=z['couplings'].astype(LD)
        with np.load(mp) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
        for batch in receipt['batches']:
            path=run/batch['path'];assert S.sha(path)==batch['sha256'];lo=batch['offset']
            with np.load(path) as z:ix=z['intervention_indices'];history=z['states_before'];trace=json.loads(z['trace_utf8'].tobytes())['trace']
            for record,b in zip(trace,history):
                items=[(v['local_input'],tuple(v['cells'])) for v in record['candidates']];values=[v['delta'] for v in record['candidates']];check_save('H2aw',t[lo:lo+32][ix],b,a[lo:lo+32][ix],theta,mu,ao,items,values,{str(x.relative_to(S.ROOT)):S.sha(x) for x in [ds,parent,mp,path]})
    assert counts['H2aw']==168
    dense=S.OUT/'density_transfer';run=dense/'runs/shard00-s0';receipt=json.loads((run/'result.json').read_text());ds=S.ROOT/json.loads((dense/'datasets.json').read_text())['datasets'][0]['path'];op=run/'outputs.npz';mp=dense/'model-seed0-old_binary.npz'
    with np.load(ds) as z:t=z['tokens']
    with np.load(op) as z:a=z['couplings'].astype(LD)
    with np.load(mp) as z:theta=z['theta'].astype(LD);mu=z['mu_offsets'];ao=z['aoffs']
    for tr in receipt['trace_archives']:
        if counts['H2ay']==344:break
        path=run/tr['path'];assert S.sha(path)==tr['sha256'];lo=tr['offset']
        with np.load(path) as z:ix=z['valid_prefix_indices'];history=z['states_before'];trace=json.loads(z['trace_utf8'].tobytes())['trace']
        for record,b in zip(trace,history):
            take=min(len(record['candidates']),344-counts['H2ay']);part=record['candidates'][:take];items=[(v['local_input'],tuple(v['cells'])) for v in part];values=[v['delta'] for v in part];check_save('H2ay',t[lo:lo+32][ix],b,a[lo:lo+32][ix],theta,mu,ao,items,values,{str(x.relative_to(S.ROOT)):S.sha(x) for x in [ds,op,mp,path]})
            if counts['H2ay']==344:break
    assert counts=={'H2aw':168,'H2ay':344};manifest={'selection':'All168 H2aw lexical512 candidate deletions plus first344 H2ay source shard00-s0 trace candidates in receipt/round order. First256 density candidates satisfy stated gate; additional88 complete fixed512 benchmark. No outcome-dependent selection.','groups':groups,'candidate_count':512};S.save(fixtures/'manifest.json',manifest);r={'status':'passed','small_ternary_states':small_count,'small_empty_changes':empty_count,'boundary_wall_endpoint_cases':96,'actual_candidates':counts,'maximum_small_error':max_small,'maximum_actual_full_grid_error':max_actual,'maximum_published_cost_error':max_published,'fixture_manifest_sha256':S.sha(fixtures/'manifest.json'),'local_factor_sha256':S.sha(S.HERE/'local_factor_delta.py'),'script_sha256':S.sha(__file__),'seconds':time.perf_counter()-started,'new_scientific_inputs':0,'new_inference_trajectories':0};S.save(DEST/'numerical-gate.json',r);S.emit('local_factor_preflight_passed',**r)


if __name__=='__main__':main()
