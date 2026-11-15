"""Independent boundary-wall interaction diagnosis of the failed H2al gate."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
from connectivity_transfer import PP,DEST,protocol,load_models,hidden
from verify_training_feasibility import energy,flip,LD


def correction(theta,offsets,h=120,w=120,H=180,W=180):
    table=theta[:16].reshape(4,2,2);value=np.zeros((h,w),LD)
    for k,(dr,dc) in enumerate(offsets):
        for r in range(h):
            for c in range(w):
                rr,cc=r+dr,c+dc
                if 0<=rr<H and 0<=cc<W and not (0<=rr<h and 0<=cc<w):value[r,c]+=table[k,1,0]-table[k,0,0]
                rr,cc=r-dr,c-dc
                if 0<=rr<H and 0<=cc<W and not (0<=rr<h and 0<=cc<w):value[r,c]+=table[k,0,1]-table[k,0,0]
    return value


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();protocol();started=time.perf_counter();rows=[]
    with np.load(S.OUT/'factorial_transfer/shard00.npz') as z:oldtok=z['tokens'][:64]
    newtok=np.pad(oldtok,((0,0),(0,60),(0,60)));old_t=torch.tensor(oldtok.astype(np.int64),device='cuda');new_t=torch.tensor(newtok.astype(np.int64),device='cuda')
    for seed in range(3):
        models,doubles=load_models(json.loads(PP.read_text()),seed);newa,_,_=hidden(models[0],new_t)
        with np.load(S.OUT/f'factorial_transfer/runs/shard00-s{seed}/outputs.npz') as z:olda=z['couplings'][:64];oldpred=z['predictions'][:2,:64]
        mask=np.repeat((oldtok>0)[:,None],5,1);assert np.array_equal(newa.cpu().numpy()[:,:,:120,:120][mask],olda[mask]);assert (oldtok==1).sum()==(newtok==1).sum()
        for v in [0,1]:
            m=doubles[v];theta=np.concatenate([m.T.cpu().numpy().ravel(),m.b.cpu().numpy(),m.kappa.cpu().numpy()]).astype(LD);mu=np.array(m.mu_offsets);ao=np.array(m.aoffs);delta=correction(theta,mu);free=oldtok==1;aa=newa.cpu().numpy().astype(LD);aold=olda.astype(LD);expected=np.broadcast_to(delta,oldtok.shape);assert np.all(expected[:,:118,:118]==0);zero_old=oldtok>=2;zero_new=newtok>=2;base_old=energy(oldtok,zero_old,aold,theta,mu,ao);base_new=energy(newtok,zero_new,aa,theta,mu,ao);max_e=max_d=0.
            for board in [oldpred[v],oldtok>0,((np.indices(oldtok.shape)[1]+np.indices(oldtok.shape)[2])%2==0)&free|(oldtok>=2)]:
                padded=np.pad(board,((0,0),(0,60),(0,60)));observed=energy(newtok,padded,aa,theta,mu,ao)-base_new-energy(oldtok,board,aold,theta,mu,ao)+base_old;predicted=(expected*board*free).sum((1,2));err=float(abs(observed-predicted).max());assert err<1e-8;max_e=max(max_e,err);fd=flip(newtok,padded,aa,theta,mu,ao)[:,:120,:120]-flip(oldtok,board,aold,theta,mu,ao);err=float(abs((fd-(1-2*board.astype(int))*expected)[free]).max());assert err<1e-10;max_d=max(max_d,err)
            oldp=torch.full(old_t.shape,.5,device='cuda',dtype=torch.float64,requires_grad=True);newp=torch.full(new_t.shape,.5,device='cuda',dtype=torch.float64,requires_grad=True);adold=torch.tensor(olda,device='cuda',dtype=torch.float64);adnew=newa.double();oldgrad=torch.autograd.grad(m.energy(old_t,oldp,adold).sum(),oldp)[0];newgrad=torch.autograd.grad(m.energy(new_t,newp,adnew).sum(),newp)[0];gd=(newgrad[:,:120,:120]-oldgrad).detach().cpu().numpy();gradient_error=float(abs((gd-expected)[free]).max());assert gradient_error<1e-10
            with torch.no_grad():field_diff=(m.field(new_t,newp,adnew)[:,:120,:120]-m.field(old_t,oldp,adold)).cpu().numpy()
            field_error=float(abs((field_diff+expected)[free]).max());assert field_error<1e-10;selected=expected[free];rows.append({'seed':seed,'arm':['pl','old_binary'][v],'seen_inputs':64,'free_readout_exact':True,'free_count_unchanged':True,'boundary_changed_free_cells':int((selected!=0).sum()),'boundary_linear_correction_range':[float(selected.min()),float(selected.max())],'maximum_independent_binary_relative_energy_error':max_e,'maximum_independent_flip_difference_error':max_d,'maximum_autograd_gradient_correction_error':gradient_error,'maximum_field_correction_error':field_error})
    result={'status':'cause_verified','protocol_sha256':S.sha(PP),'rows':rows,'inventory':inventory,'seconds':time.perf_counter()-started,'source_sha256':S.sha(__file__),'cause':'Convolution pads the binary-state channels outside the originalgrid with(0,0), whereas an addedclamped wall is state(1,0). Added right/bottom wallneighbors contribute genuine pair-table unarycosts at oldboundaryfreecells. The freegraph, readouts andfreecount are unchanged, buttheanswerobjective is not. Relativebinaryenergies, flipdeltas, autogradgradients andfields match the exact addedpairterms. No globalnormalization explanation needed.','fresh_inputs':0};S.save(DEST/'padding-boundary-diagnosis.json',result);print(json.dumps(result))


if __name__=='__main__':main()
