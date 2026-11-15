"""H2aj frozen, old-cohort120 gate before fresh generation."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import graph_block_moves as B
import factorial_batch as Q
import fast_npz as Z
from factorial_transfer import PP,DEST,protocol,load_models,hidden


def literal_check(model,t,a,boards,items,values):
    torch,_,_=S.imports();error=0.
    for lo in range(0,len(items),32):
        part=items[lo:lo+32];ix=torch.tensor([x[0] for x in part],device=t.device);old=boards[ix];new=old.clone()
        for j,(_,cells) in enumerate(part):new[j].flatten()[list(cells)]=0
        literal=(model.energy(t[ix],new,a[ix])-model.energy(t[ix],old,a[ix])).cpu().numpy();error=max(error,float(abs(literal-np.asarray(values[lo:lo+len(part)])).max()))
    assert error<1e-8,error
    return error


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();p=protocol();DEST.mkdir(exist_ok=True);started=time.perf_counter();rows=[]
    def watch():
        if time.perf_counter()-started>180:raise TimeoutError('Frozen180s sourcegate')
    with np.load(S.OUT/'second_negative_transfer/shard00.npz') as z:tokens=z['tokens'][:64]
    t=torch.tensor(tokens.astype(np.int64),device='cuda');max_error=0.;checks=0
    with torch.no_grad():
        for seed in range(3):
            models,doubles=load_models(p,seed);a,hs,good=hidden(models[0],t)
            with np.load(S.OUT/f'second_negative_transfer/runs/shard00-s{seed}/outputs.npz') as z:retained_a=z['couplings'][:64];retained_pred=z['predictions'][[0,2],:64];retained_iters=z['iterations'][[0,2],:64]
            assert np.array_equal(a.cpu().numpy(),retained_a) and good.all();r=Q.solve(models,doubles,t,a,rotation=seed,watchdog=watch)
            for v,oldv in [(0,0),(1,1)]:
                assert np.array_equal(r['outputs'][v]['prediction'].cpu().numpy(),retained_pred[oldv]);assert np.array_equal(r['outputs'][v]['iterations'].cpu().numpy(),retained_iters[oldv])
            for v in range(5):
                board=r['outputs'][v]['prediction'];assert torch.equal(board[t!=1],(t>=2).to(board.dtype)[t!=1]);margin=doubles[Q.MODEL_INDEX[v]].flip_delta(t,board.double(),a.double()).masked_fill(t!=1,float('inf')).flatten(1).amin(1);assert (margin[r['outputs'][v]['stopped']]>=-1e-6).all()
            for v in [3,4]:
                raw=r['blocks'][v]['raw'];ix=r['blocks'][v]['valid_prefix_indices'];assert len(ix)==64;assert np.array_equal(raw['states_before'][0],r['prefixes'][v-2]['prediction'].cpu().numpy())
                for record,boards in zip(raw['trace'],raw['states_before']):
                    assert np.array_equal(boards[tokens!=1],(tokens>=2)[tokens!=1]);items=[(z['local_input'],z['cells']) for z in record['candidates']];values=[z['delta'] for z in record['candidates']]
                    if items:max_error=max(max_error,literal_check(doubles[v-2],t,a.double(),torch.tensor(boards,device='cuda',dtype=torch.float64),items,values));checks+=len(items)
            rng=np.random.default_rng(2026111516+seed);random=(rng.random(tokens[:4].shape)<.6)&(tokens[:4]==1);random|=tokens[:4]>=2;random=torch.tensor(random,device='cuda',dtype=torch.float64)
            for v in [1,2]:
                cache=C.BinaryQuadratic(doubles[v],t[:4],a[:4].double());items=[]
                for i,board in enumerate(random.cpu().numpy().astype(bool)):items.extend((i,cells) for cells in B.candidates(tokens[i],board)[0])
                values=B.scores(cache,random,items).cpu().numpy();max_error=max(max_error,literal_check(doubles[v],t[:4],a[:4].double(),random,items,values));checks+=len(items)
            rows.append({'seed':seed,'reused_seen_inputs':64,'exact_hidden_and_old_bp_binary_prefix_parity':True,'all_five_arm_clamps_and_final_original_certificates':True,'prefix_reuse_exact':True,'timing':r['timing'],'block_moves':[int(r['blocks'][v]['block_moves'].sum()) for v in [3,4]]});watch()
    source=S.OUT/'hard_negative_transfer/runs/shard00-s0/outputs.npz';archive=DEST/'archive-roundtrip.npz'
    with np.load(source) as z:original={k:z[k] for k in z.files}
    Z.save(archive,**original)
    with np.load(archive) as z:
        assert z.files==list(original)
        for k,v in original.items():assert z[k].dtype==v.dtype and z[k].shape==v.shape and z[k].tobytes()==v.tobytes()
    S.save(DEST/'numerical-gate.json',{'status':'passed','utc':S.utc(),'protocol_sha256':S.sha(PP),'inventory':inventory,'rows':rows,'candidate_literal_checks':checks,'maximum_literal_full_energy_delta_error':max_error,'archive_exact_dtype_shape_bytes_order':True,'roundtrip_source_sha256':S.sha(source),'roundtrip_archive_sha256':S.sha(archive),'new_independent_inputs':0,'seconds':time.perf_counter()-started,'script_sha256':S.sha(__file__)});S.emit('factorial_source_gate_passed',checks=checks,maximum_error=max_error,seconds=time.perf_counter()-started)


if __name__=='__main__':main()
