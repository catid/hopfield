"""Complete source replay gate for H2ax six-arm composition, before fresh draws."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import cap_transfer_batch as Q
from cycle_transfer_180 import load_models,hidden
from gate_factorial_transfer import literal_check
from audit_connectivity_screen import path_valid
from verify_training_feasibility import energy,flip,LD
PARENT=S.OUT/'cycle_transfer_180'
SOURCE=S.OUT/'large_cycle_cut'
DEST=S.OUT/'cap_transfer_preflight'


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();DEST.mkdir(exist_ok=True);started=time.perf_counter();p=json.loads((PARENT/'protocol.json').read_text());source=json.loads((SOURCE/'protocol.json').read_text());audit=json.loads((SOURCE/'audit.json').read_text());assert audit['status']=='passed';rows=[];checks=cases=nonsimple=0;max_delta=max_energy=max_flip=0.;coverage=np.zeros(4,np.int64)
    gates=[PARENT/'numerical-gate.json',SOURCE/'numerical-gate.json',SOURCE/'old256-fixed-point-control.json']
    for path in gates:assert json.loads(path.read_text())['status']=='passed'
    for path,h in source['source_sha256'].items():assert S.sha(S.ROOT/path)==h
    with torch.no_grad():
        for seed in range(3):
            models,doubles=load_models(p,seed)
            for shard in range(16):
                jid=f'shard{shard:02d}-s{seed}';ds=source['datasets'][shard];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
                with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
                with np.load(PARENT/'runs'/jid/'outputs.npz') as z:old={k:z[k] for k in ['predictions','iterations','stopped','finite','couplings','block_moves','post_block_sweeps','block_native_cap','block_cap','initial_simple_path']}
                new={}
                for policy in ['lexical512','energy512']:
                    run=SOURCE/policy/'runs'/jid;receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
                    with np.load(run/'outputs.npz') as z:new[policy]={k:z[k] for k in z.files}
                ix=np.array(sorted({0,1,*np.flatnonzero(~new['lexical512']['initial_simple_path']).tolist()}));nonsimple+=int((~new['lexical512']['initial_simple_path'][ix]).sum());cases+=len(ix);tok=tokens[ix];coverage+=np.array([(tok[:,0,:]==1).sum(),(tok[:,-1,:]==1).sum(),(tok[:,:,0]==1).sum(),(tok[:,:,-1]==1).sum()]);t=torch.tensor(tok.astype(np.int64),device='cuda');a,hs,good=hidden(models[0],t);assert good.all() and np.array_equal(a.cpu().numpy(),old['couplings'][ix]);r=Q.solve(models,doubles,t,a,rotation=(seed+shard)%6)
                for v in range(6):
                    for key in ['prediction','iterations','stopped','finite']:
                        expected=old['predictions' if key=='prediction' else key][[0,1,2,4][v],ix] if v<4 else new['lexical512' if v==4 else 'energy512'][key][ix]
                        assert np.array_equal(r['outputs'][v][key].cpu().numpy(),expected),(jid,v,key)
                    m=doubles[Q.MODEL_INDEX[v]];b=r['outputs'][v]['prediction'];assert torch.equal(b[t!=1],(t>=2).to(b.dtype)[t!=1]);theta=np.concatenate([m.T.cpu().numpy().ravel(),m.b.cpu().numpy(),m.kappa.cpu().numpy()]).astype(LD);mu=np.array(m.mu_offsets);ao=np.array(m.aoffs);board=b.cpu().numpy();aa=a.cpu().numpy().astype(LD);ff=flip(tok,board,aa,theta,mu,ao);actual=m.flip_delta(t,b.double(),a.double()).cpu().numpy();err=float(abs((ff-actual)[tok==1]).max());assert err<1e-9;max_flip=max(max_flip,err);e=energy(tok,board,aa,theta,mu,ao)+LD(float(np.log(1+1e-12)))*(tok==1).sum((1,2));err=float(abs(e-m.energy(t,b.double(),a.double()).cpu().numpy()).max());assert err<1e-6;max_energy=max(max_energy,err)
                for v in [2,3,4,5]:
                    block=r['blocks'][v];parent=1 if v==2 else 2 if v==3 else 3
                    for key,newkey in [('block_moves','block_moves'),('post_block_sweeps','native_sweeps'),('native_cap','native_cap'),('block_cap','block_cap'),('initial_simple_path','initial_simple_path')]:
                        if v<4:
                            oldkey='block_native_cap' if key=='native_cap' else key;expected=old[oldkey][0 if v==2 else 2,ix]
                        else:expected=new['lexical512' if v==4 else 'energy512'][newkey][ix]
                        assert np.array_equal(block[key],expected),(jid,v,key)
                    raw=block['raw'];indices=block['valid_prefix_indices']
                    if raw is None:assert len(indices)==0;continue
                    assert np.array_equal(raw['states_before'][0],r['outputs'][parent]['prediction'].cpu().numpy()[indices])
                    for record,states in zip(raw['trace'],raw['states_before']):
                        items=[(x['local_input'],x['cells']) for x in record['candidates']];values=[x['delta'] for x in record['candidates']]
                        if items:max_delta=max(max_delta,literal_check(doubles[1],t[indices],a[indices].double(),torch.tensor(states,device='cuda',dtype=torch.float64),items,values));checks+=len(items)
                charged=np.array(r['timing']['charged_answer_seconds']);unique=np.array(r['timing']['unique_operation_seconds']);expected=unique.copy();expected[2:]+=unique[1];expected[3:]+=unique[2];expected[4:]+=unique[3];assert np.array_equal(charged,expected);rows.append({'job':jid,'source_indices':ix.tolist(),'all_six_boards_steps_flags_exact':True,'all_four_block_traces_flags_exact':True});assert time.perf_counter()-started<120
    assert nonsimple==84 and (coverage>0).all()
    result={'status':'passed','kind':'complete_source_preflight_before_science_freeze','inventory':inventory,'source_protocol_sha256':S.sha(SOURCE/'protocol.json'),'source_audit_sha256':S.sha(SOURCE/'audit.json'),'batch_helper_sha256':S.sha(S.HERE/'cap_transfer_batch.py'),'source_gate_hashes':{str(x.relative_to(S.ROOT)):S.sha(x) for x in gates},'new_independent_inputs':0,'source_cases':cases,'all_frozen_nonsimple_starts':nonsimple,'candidate_literal_checks':checks,'maximum_literal_error':max_delta,'maximum_independent_energy_error':max_energy,'maximum_independent_flip_error':max_flip,'actual_outer_boundary_coverage':coverage.tolist(),'rows':rows,'seconds':time.perf_counter()-started,'script_sha256':S.sha(__file__),'scope':'First two inputs per shard/model plus all84 H2aw nonsimple starts; all six predictions/iterations/flags/readouts and block flags exactly match archived results. All six original objectives/flips independently checked. Existing actual180 gradient/readout/upstream/uncached and positive/restoration/selector/size-boundary gates hash-bound above. No fresh data or science outcomes.'};S.save(DEST/'numerical-gate.json',result);S.emit('cap_transfer_preflight_passed',cases=cases,checks=checks,seconds=result['seconds'])


if __name__=='__main__':main()
