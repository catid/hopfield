"""H2av complete actual180 objective/policy preflight before science freeze."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import cycle_transfer_batch as Q
from connectivity_transfer_180 import PP,protocol,load_models,hidden
DEST=S.OUT/'cycle_transfer_180_preflight'
from gate_factorial_transfer import literal_check
from audit_connectivity_screen import path_valid
from verify_graph_block_source import connected
from verify_training_feasibility import energy,flip,LD,offsets


def shifted(a,dr,dc,fill):
    out=np.full_like(a,fill);l,r=offsets(a.shape[-2:],dr,dc);out[...,l[0],l[1]]=a[...,r[0],r[1]];return out


def readout_reference(model,t,psi):
    assert model.coupling=='attention';nb=np.stack([shifted(psi,dr,dc,np.inf) for dr,dc in model.aoffs],1);inside=np.stack([shifted(np.ones_like(t,bool),dr,dc,False) for dr,dc in model.aoffs],1);live=inside&np.isfinite(nb);anyl=live.any(1,keepdims=True);lat=np.where(live,-np.where(live,nb,0),-np.inf);lat=np.where(anyl,lat,np.float32(0));scores=model.beta.cpu().numpy()[None,:,None,None]+lat;scores=np.where(inside,scores,-np.inf);e=np.exp((scores-scores.max(1,keepdims=True)).astype(np.float64));return e/e.sum(1,keepdims=True)


def continuous_gradient(t,board,a,theta,mu,ao):
    p=np.where(t==1,board,t>=2);g=np.full(p.shape,theta[16]-theta[17],np.float64);table=theta[:16].reshape(4,2,2)
    for k,(dr,dc) in enumerate(mu):
        l,r=offsets(t.shape[-2:],dr,dc);left=p[:,l[0],l[1]];right=p[:,r[0],r[1]];T=table[k];g[:,l[0],l[1]]+=(T[1,0]-T[0,0])*(1-right)+(T[1,1]-T[0,1])*right;g[:,r[0],r[1]]+=(T[0,1]-T[0,0])*(1-left)+(T[1,1]-T[1,0])*left
    for k,(dr,dc) in enumerate(ao):
        if dr==dc==0:g-=2*theta[18+k]*a[:,k]*p;continue
        l,r=offsets(t.shape[-2:],dr,dc);edge=theta[18+k]*a[:,k,l[0],l[1]];g[:,l[0],l[1]]-=edge*p[:,r[0],r[1]];g[:,r[0],r[1]]-=edge*p[:,l[0],l[1]]
    return g


def main():
    lock,inventory=F.acquire(0);torch,_,M=S.imports();protocol();DEST.mkdir(exist_ok=True);started=time.perf_counter();rows=[];max_delta=max_energy=max_flip=max_gradient=max_readout=0.;checks=0;coverage=np.zeros(4,np.int64)
    with np.load(S.OUT/'connectivity_transfer_180/shard00.npz') as z:tokens=z['tokens'][:64]
    coverage=np.array([(tokens[:,0,:]==1).sum(),(tokens[:,-1,:]==1).sum(),(tokens[:,:,0]==1).sum(),(tokens[:,:,-1]==1).sum()]);assert (coverage>0).all()
    for seed in range(3):
        models,doubles=load_models(json.loads(PP.read_text()),seed);native_checks=0;binary_checks=0
        for lo in [0,32]:
            tok=tokens[lo:lo+32];t=torch.tensor(tok.astype(np.int64),device='cuda')
            with torch.no_grad():
                a,hs,good=hidden(models[0],t);assert good.all();psi,_=models[0].settle(t,tol=1e-6,max_steps=4000,check=8);reference=readout_reference(models[0],tok,psi.cpu().numpy());err=float(abs(reference-a.cpu().numpy()).max());assert err<1e-6;max_readout=max(max_readout,err);r=Q.solve(models,doubles,t,a,rotation=(seed+lo//32)%5);native=M.descend(models[0],t,a,steps=8192,eta=.5,margin=3.,certify_every=16)
                for k,key in enumerate(['prediction','iterations','stopped']):assert torch.equal(native[k],r['outputs'][0][key]);native_checks+=len(tok) if k==0 else 0
                cache=C.BinaryQuadratic(doubles[1],t,a.double())
                class Direct:
                    free=cache.free;colors=cache.colors
                    def initial(self):return cache.initial()
                    def delta(self,b):return doubles[1].flip_delta(t,b,a.double())
                direct=C.solve(Direct())
                for key in ['prediction','sweeps','stopped','finite','flip_count']:assert torch.equal(direct[key],r['prefixes'][1][key]);binary_checks+=len(tok) if key=='prediction' else 0
                with np.load(S.OUT/f'connectivity_transfer_180/runs/shard00-s{seed}/outputs.npz') as z:
                    assert np.array_equal(z['couplings'][lo:lo+32],a.cpu().numpy())
                    for v,oldv in [(0,0),(1,1),(2,3)]:
                        for key in ['predictions','iterations','stopped','finite']:assert np.array_equal(r['outputs'][v]['prediction' if key=='predictions' else key].cpu().numpy(),z[key][oldv,lo:lo+32])
                simple=np.array([path_valid(x,b) for x,b in zip(tok,r['outputs'][2]['prediction'].cpu().numpy().astype(bool))])
                for v in [3,4]:
                    assert np.array_equal(simple,r['blocks'][v]['initial_simple_path']);assert torch.equal(r['outputs'][v]['prediction'][simple],r['outputs'][2]['prediction'][simple])
                for v in range(5):
                    m=doubles[Q.MODEL_INDEX[v]];b=r['outputs'][v]['prediction'];assert torch.equal(b[t!=1],(t>=2).to(b.dtype)[t!=1]);theta=np.concatenate([m.T.cpu().numpy().ravel(),m.b.cpu().numpy(),m.kappa.cpu().numpy()]);mu=np.array(m.mu_offsets);ao=np.array(m.aoffs);board=b.cpu().numpy();aa=a.cpu().numpy().astype(LD);ff=flip(tok,board,aa,theta.astype(LD),mu,ao);actual=m.flip_delta(t,b.double(),a.double()).cpu().numpy();err=float(abs((ff-actual)[tok==1]).max());assert err<1e-9;max_flip=max(max_flip,err);e=energy(tok,board,aa,theta.astype(LD),mu,ao)+LD(float(np.log(1+1e-12)))*(tok==1).sum((1,2));err=float(abs(e-m.energy(t,b.double(),a.double()).cpu().numpy()).max());assert err<1e-6;max_energy=max(max_energy,err)
                for v in [2,3,4]:
                    raw=r['blocks'][v]['raw'];ix=r['blocks'][v]['valid_prefix_indices']
                    if raw is None:assert len(ix)==0;continue
                    assert np.array_equal(raw['states_before'][0],r['outputs'][1 if v==2 else 2]['prediction'].cpu().numpy()[ix])
                    for record,states in zip(raw['trace'],raw['states_before']):
                        items=[(x['local_input'],x['cells']) for x in record['candidates']];values=[x['delta'] for x in record['candidates']]
                        if items:max_delta=max(max_delta,literal_check(doubles[1],t[ix],a[ix].double(),torch.tensor(states,device='cuda',dtype=torch.float64),items,values));checks+=len(items)
                        if v>=2:
                            for i,cells in items:z=states[i].copy();z.flat[cells]=False;assert connected(tok[ix[i]],z)
            # A second mathematical formulation checks the full continuous gradient.
            for m in doubles:
                theta=np.concatenate([m.T.cpu().numpy().ravel(),m.b.cpu().numpy(),m.kappa.cpu().numpy()]);mu=np.array(m.mu_offsets);ao=np.array(m.aoffs);rng=np.random.default_rng(2026111517+seed+lo);board=.2+.6*rng.random((4,180,180));tt=t[:4];bb=torch.tensor(board,device='cuda',dtype=torch.float64,requires_grad=True);aa=a[:4].double();gradient=torch.autograd.grad(m.energy(tt,bb,aa).sum(),bb)[0].detach().cpu().numpy();g=continuous_gradient(tok[:4],board,aa.cpu().numpy(),theta,mu,ao);eps=1e-12;entropy=np.log(board+eps)-np.log(1-board+eps)+board/(board+eps)-(1-board)/(1-board+eps);err=float(abs((gradient-g-entropy)[tok[:4]==1]).max());assert err<1e-10;max_gradient=max(max_gradient,err)
            assert time.perf_counter()-started<120
        rows.append({'seed':seed,'seen_transformed_inputs':64,'upstream_native_exact_boards_steps_flags':native_checks,'uncached_binary_exact_boards_sweeps_flags_flips':binary_checks,'all_five_arm_clamps_and_prefix_reuse':True,'cycle_simple_identity_and_candidate_connectivity':True})
    result={'status':'passed','kind':'preflight_not_scientific_study','source_protocol_sha256':S.sha(PP),'batch_helper_sha256':S.sha(S.HERE/'cycle_transfer_batch.py'),'inventory':inventory,'rows':rows,'free_cells_on_actual_top_bottom_left_right_edges':coverage.tolist(),'candidate_literal_checks':checks,'maximum_literal_delta_error':max_delta,'maximum_independent_energy_error':max_energy,'maximum_independent_flip_error':max_flip,'maximum_independent_gradient_error':max_gradient,'maximum_independent_readout_error':max_readout,'seconds':time.perf_counter()-started,'new_independent_inputs':0,'scope':'Actual180seeninputs, sameobjective, allfivepolicies andretainedBP/binary/vetoexactparity; allfourouteredges/internalwalls. No paddinginvarianceclaim. No shortestlabels/outcome endpoints inspected; no newindependentinputs.','script_sha256':S.sha(__file__)};S.save(DEST/'numerical-gate.json',result);S.emit('cycle_transfer_180_preflight_passed',seconds=result['seconds'],checks=checks)


if __name__=='__main__':main()
