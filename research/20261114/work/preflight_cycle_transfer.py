"""Complete H2au integration preflight on already seen source states only."""
import json
import time
import numpy as np
import study as S
import selective_fallback as F
import cycle_transfer_batch as Q
from safe_block_transfer import load_models
from factorial_transfer import hidden
from gate_factorial_transfer import literal_check
PARENT=S.OUT/'safe_block_transfer'
ENERGY=S.OUT/'cycle_cut_rescue_checked'
LEXICAL=S.OUT/'lexical_cycle_cut'
DEST=S.OUT/'cycle_transfer_preflight'


def load(seed,indices,p):
    tokens=[];readout=[];expected={k:[] for k in ['predictions','iterations','stopped','finite']};mapping=[]
    for shard in sorted(set(i//256 for i in indices)):
        wanted=[i%256 for i in indices if i//256==shard];mapping.extend([i for i in indices if i//256==shard]);job=f'shard{shard:02d}-s{seed}'
        with np.load(S.ROOT/p['datasets'][shard]['path']) as z:tokens.append(z['tokens'][wanted])
        with np.load(PARENT/'runs'/job/'outputs.npz') as z:
            readout.append(z['couplings'][wanted]);arrays={k:[z[k][v,wanted] for v in [0,1,3]] for k in expected}
        for directory in [ENERGY,LEXICAL]:
            with np.load(directory/'runs'/job/'outputs.npz') as z:
                for key in expected:arrays[key].append(z['prediction' if key=='predictions' else key][wanted])
        for key in expected:expected[key].append(np.stack(arrays[key]))
    return np.concatenate(tokens),np.concatenate(readout),{k:np.concatenate(v,axis=1) for k,v in expected.items()},mapping


def main():
    lock,inventory=F.acquire(0);torch,_,_=S.imports();started=time.perf_counter();p=json.loads((S.HERE/'pendant-rescue-v2-protocol.json').read_text());models_p=json.loads((S.HERE/'safe-block-transfer-protocol.json').read_text());audit=json.loads((ENERGY/'audit.json').read_text());assert audit['status']=='passed';rows=[];checks=0;maxerr=0.;source_hashes={};synthetic=[]
    for path in [S.OUT/'cycle_cut_preflight/numerical-gate.json',S.OUT/'lexical_cycle_cut_preflight/numerical-gate.json']:
        r=json.loads(path.read_text());assert r['status']=='passed';synthetic.append({'path':str(path.relative_to(S.ROOT)),'sha256':S.sha(path)})
    def watch():
        if time.perf_counter()-started>120:raise TimeoutError('Integrationpreflight120s')
    with torch.no_grad():
        for seed in range(3):
            models,doubles=load_models(models_p,seed);bad=[]
            for r in audit['rows']:
                if r['job']['seed']==seed:bad.extend(r['job']['shard']*256+i for i,v in enumerate(r['success'][3]) if not v)
            augmented=sorted(set(bad));augmented.extend(i for i in range(64) if i not in augmented);augmented=augmented[:64]
            for batch_index,indices in enumerate([list(range(64)),augmented]):
                tok,a0,expected,mapping=load(seed,indices,p);assert len(tok)==64;tt=torch.tensor(tok.astype(np.int64),device='cuda');aa,hs,good=hidden(models[0],tt);assert good.all() and np.array_equal(aa.cpu().numpy(),a0);r=Q.solve(models,doubles,tt,aa,rotation=(seed+batch_index)%5,watchdog=watch)
                for v in range(5):
                    for key in ['predictions','iterations','stopped','finite']:assert np.array_equal(r['outputs'][v]['prediction' if key=='predictions' else key].cpu().numpy(),expected[key][v]),(seed,batch_index,v,key)
                    board=r['outputs'][v]['prediction'];assert torch.equal(board[tt!=1],(tt>=2).to(board.dtype)[tt!=1]);marg=doubles[Q.MODEL_INDEX[v]].flip_delta(tt,board.double(),aa.double()).masked_fill(tt!=1,float('inf')).flatten(1).amin(1);assert (marg[r['outputs'][v]['stopped']]>=-1e-6).all()
                for v in [2,3,4]:
                    block=r['blocks'][v];ix=block['valid_prefix_indices'];raw=block['raw']
                    if raw:
                        parent=block['parent_arm'];assert np.array_equal(raw['states_before'][0],r['outputs'][parent]['prediction'][ix].cpu().numpy().astype(bool))
                        for trace,before in zip(raw['trace'],raw['states_before']):
                            items=[(c['local_input'],c['cells']) for c in trace['candidates']];values=[c['delta'] for c in trace['candidates']]
                            if items:maxerr=max(maxerr,literal_check(doubles[1],tt[ix],aa[ix].double(),torch.tensor(before,device='cuda',dtype=torch.float64),items,values));checks+=len(items)
                rows.append({'seed':seed,'batch':batch_index,'source_indices':mapping,'all_five_boards_iterations_flags_exact':True,'hidden_readout_exact':True,'nonsimple_parent_cases_included':len(bad) if batch_index else None,'answer_timings':r['timing']});watch()
            for shard in sorted(set(i//256 for i in augmented)|{0}):
                for d in [PARENT,ENERGY,LEXICAL]:path=d/'runs'/f'shard{shard:02d}-s{seed}'/'outputs.npz';source_hashes[str(path.relative_to(S.ROOT))]=S.sha(path)
    DEST.mkdir(exist_ok=True);out={'status':'passed','kind':'preflight_not_scientific_study','inventory':inventory,'rows':rows,'source_output_sha256':source_hashes,'synthetic_policy_gates':synthetic,'all63_nonsimple_source_model_cases_included':True,'literal_candidate_checks':checks,'maximum_literal_error':maxerr,'seconds':time.perf_counter()-started,'source_sha256':S.sha(__file__),'batch_helper_sha256':S.sha(S.HERE/'cycle_transfer_batch.py'),'scope':'Six64casebatchesofseenH2ansource,includingall63nonsimplecases; exactBP/binary/veto/energy/lexicalparity beforefreshgeneration. Completebothsyntheticpolicygates reusedbyhash. Noindependentinputs.'};S.save(DEST/'numerical-gate.json',out);S.emit('cycle_transfer_preflight_passed',seconds=out['seconds'],literal_checks=checks)


if __name__=='__main__':main()
