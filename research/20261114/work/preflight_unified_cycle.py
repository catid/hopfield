"""H2bc complete six-arm source replay and original-factor gate before fresh draws."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import unified_cycle_batch as Q
from density_transfer import load_models,hidden
from verify_training_feasibility import energy,flip,LD
from local_factor_delta import deltas
from verify_large_cycle_geometry import geometry

ORIGIN=S.OUT/'density_transfer';SOURCE=S.OUT/'residual_factorial';DEST=S.OUT/'unified_cycle_preflight'


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();DEST.mkdir(exist_ok=True);started=time.perf_counter();p=json.loads((ORIGIN/'protocol.json').read_text());sp=json.loads((SOURCE/'protocol.json').read_text());audit=json.loads((SOURCE/'audit.json').read_text());assert audit['status']=='passed' and audit['joint_success']['both_families_composed']==4096;manifest=json.loads((ORIGIN/'datasets.json').read_text());rows=[];checks=cases=caps=0;max_delta=max_energy=max_flip=0.
    with torch.no_grad():
        for seed in range(3):
            models,doubles=load_models(p,seed)
            for shard in range(16):
                jid=f'shard{shard:02d}-s{seed}';ds=manifest['datasets'][shard];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
                with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
                run=ORIGIN/'runs'/jid;receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
                with np.load(run/'outputs.npz') as z:old={k:z[k] for k in ['predictions','iterations','stopped','finite','couplings','block_cap','block_native_cap']}
                new={}
                for family in ['lexical_family','veto_family']:
                    run=SOURCE/family/'runs'/jid;r=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==r['outputs_sha256']
                    with np.load(run/'outputs.npz') as z:new[family]={k:z[k] for k in ['predictions','iterations','stopped','finite','block_cap','native_cap']}
                ix=np.array(sorted({0,*np.flatnonzero(old['block_cap'][1]).tolist()}));caps+=int(old['block_cap'][1,ix].sum());cases+=len(ix);tok=tokens[ix];t=torch.tensor(tok.astype(np.int64),device='cuda');a,hs,good=hidden(models[0],t);assert good.all() and np.array_equal(a.cpu().numpy(),old['couplings'][ix]);r=Q.solve(models,doubles,t,a,rotation=(seed+shard)%6)
                family_mask=old['block_cap'][0,ix]
                for v in range(6):
                    for key in ['prediction','iterations','stopped','finite']:
                        key0='predictions' if key=='prediction' else key
                        if v<4:expected=old[key0][v,ix]
                        elif v==5:
                            expected=new['lexical_family'][key0][ix].copy();expected[family_mask]=new['veto_family'][key0][ix][family_mask]
                        else:continue
                        assert np.array_equal(r['outputs'][v][key].cpu().numpy(),expected),(jid,v,key)
                    m=doubles[Q.MODEL_INDEX[v]];b=r['outputs'][v]['prediction'];assert torch.equal(b[t!=1],(t>=2).to(b.dtype)[t!=1]);theta=np.concatenate([m.T.cpu().numpy().ravel(),m.b.cpu().numpy(),m.kappa.cpu().numpy()]).astype(LD);mu=np.array(m.mu_offsets);ao=np.array(m.aoffs);board=b.cpu().numpy();aa=a.cpu().numpy().astype(LD);ff=flip(tok,board,aa,theta,mu,ao);err=float(abs((ff-m.flip_delta(t,b.double(),a.double()).cpu().numpy())[tok==1]).max());assert err<1e-9;max_flip=max(max_flip,err);e=energy(tok,board,aa,theta,mu,ao)+LD(float(np.log(1+1e-12)))*(tok==1).sum((1,2));err=float(abs(e-m.energy(t,b.double(),a.double()).cpu().numpy()).max());assert err<1e-6;max_energy=max(max_energy,err)
                for key,oldkey in [('native_cap','native_cap'),('block_cap','block_cap')]:
                    expected=new['lexical_family'][oldkey][ix].copy();expected[family_mask]=new['veto_family'][oldkey][ix][family_mask];assert np.array_equal(r['blocks'][5][key],expected),(jid,key)
                m=doubles[1];theta=np.concatenate([m.T.cpu().numpy().ravel(),m.b.cpu().numpy(),m.kappa.cpu().numpy()]).astype(LD);mu=np.array(m.mu_offsets);ao=np.array(m.aoffs)
                for v in [2,3,4,5]:
                    block=r['blocks'][v];raw=block['raw'];ids=block['valid_prefix_indices']
                    if raw is None:continue
                    for record,states in zip(raw['trace'],raw['states_before']):
                        items=[(x['local_input'],x['cells']) for x in record['candidates']];values=np.array([x['delta'] for x in record['candidates']])
                        if items:
                            literal=deltas(tok[ids],states,a.cpu().numpy()[ids].astype(LD),theta,mu,ao,items);err=float(abs(literal-values).max());assert err<1e-8;max_delta=max(max_delta,err);checks+=len(items)
                        if v>=3:
                            for g in record['geometry']:
                                cells,geo=geometry(tok[ids][g['local_input']],states[g['local_input']],256);assert g=={'local_input':g['local_input'],'candidate_count':len(cells),**geo}
                    for g in raw.get('terminal_check_records',[]):
                        cells,geo=geometry(tok[ids][g['local_input']],raw['prediction'].cpu().numpy().astype(bool)[g['local_input']],256);assert g=={'local_input':g['local_input'],'candidate_count':len(cells),'candidates':[list(c) for c in cells],'geometry':geo}
                total_moves=sum(r['blocks'][v]['block_moves'] for v in [3,4,5]);assert (total_moves<=48).all();assert np.allclose(r['timing']['charged_answer_seconds'][1:],np.cumsum(r['timing']['unique_operation_seconds'][1:]),rtol=0,atol=1e-12);rows.append({'job':jid,'source_indices':ix.tolist(),'original_four_arms_exact':True,'unified48_composition_boards_iterations_flags_exact':True});assert time.perf_counter()-started<300;S.emit('unified_source_gate_job',job=jid,cases=len(ix),seconds=time.perf_counter()-started)
    assert caps==204
    prior=S.OUT/'terminal_cycle_preflight/numerical-gate.json';result={'status':'passed','new_independent_inputs':0,'source_cases':cases,'all_original_lexical_cap_states':caps,'candidate_original_factor_checks':checks,'maximum_cost_error':max_delta,'maximum_energy_error':max_energy,'maximum_flip_error':max_flip,'source_protocol_sha256':S.sha(SOURCE/'protocol.json'),'source_audit_sha256':S.sha(SOURCE/'audit.json'),'terminal_synthetic_gate_sha256':S.sha(prior),'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['unified_cycle.py','unified_cycle_batch.py','bounded_lexical_cycle_moves.py','preflight_unified_cycle.py','terminal_cycle_moves.py']},'rows':rows,'seconds':time.perf_counter()-started,'inventory':inventory,'scope':'All204 original cap states plus first input per shard/model. Original four arms match archived outputs exactly; unified48 matches H2bb disjoint composition including boards/iterations/native and block flags. Six objectives/margins, every literal original-factor candidate cost, cycle geometry and final terminal check independently verified. Source replay only; zero fresh scientific evidence.'};S.save(DEST/'numerical-gate.json',result)


if __name__=='__main__':main()
