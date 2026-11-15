"""H2af original-training hard-negative mining with frozen H2ac coefficients and unchanged H2aa decoder."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as G
import binary_coordinate as C
from second_negative_transfer import PP as DECODER,protocol as decoder_protocol
from prepare_lp_transfer import TRAIN

DEST=S.OUT/'leaf_removability'
PRIOR=S.OUT/'second_hard_negative_feasibility'
PP=S.HERE/'leaf-removability-protocol.json'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=decoder_protocol();p=json.loads((S.HERE/'second-hard-negative-feasibility-protocol.json').read_text());DEST.mkdir(exist_ok=True)
    p.update(id='original-training-false-positive-leaf-removability',hypothesis_id='H2af',frozen_utc=S.utc(),hypothesis='All3 fixed H2ae models yield at least1 wrong source board, every wrong board has at least1 eligible false-positive leaf, and the answer parametrization admits all old constraints plus unit-negative flip costs for EVERY eligible leaf.')
    p['models']={str(seed):parent['models'][str(seed)]['second_negative_binary'] for seed in range(3)};p['decoder_protocol_sha256']=S.sha(DECODER);p['previous_feasibility_protocol_sha256']=S.sha(S.HERE/'second-hard-negative-feasibility-protocol.json')
    p['previous_witnesses']={str(seed):{'sha256':S.sha(PRIOR/f'seed{seed}/candidate-theta.npy'),'negative_archive_sha256':S.sha(PRIOR/f'all-negatives-seed{seed}.npz')} for seed in range(3)}
    p['coefficient_arrays']={str(seed):{'path':str((S.OUT/f'second_negative_transfer/second-negative-scaled-seed{seed}.npy').relative_to(S.ROOT)),'sha256':S.sha(S.OUT/f'second_negative_transfer/second-negative-scaled-seed{seed}.npy')} for seed in range(3)}
    p['constraints']['old']='Every H2w stability/ranking row plus all869 H2ab/H2ad negative rows unchanged, no deduplication.'
    p['constraints']['new']='For each current wrong source board, every selected free cell that is absent from the canonical training label and has exactly ONE selected north/east/south/west neighbor: E(board with this bit removed)-E(board)<=-1. Equivalently negative flip-feature row has unit lower margin. All eligible cells retained; no chosen subset.'
    p['informativeness']={'selection':'False positive means free token1, predicted1, canonical label0. Selected neighbor count includes selected endpoints and free cells, four-neighbor only; walls/outside count0.','empty':'Zero wrong boards or any wrong board without an eligible leaf makes that model uninformative; record every such board, do not choose a subset to avoid infeasibility.','prior_margins':'Record original cast-model flip cost at every chosen leaf and exact row hashes. New-constraint novelty vs old rows certified by their distinct scalar products under the previous cast model when separation holds; within-new duplicates reported and retained.'}
    p['solver']['conditional_infeasibility']='If primal reports infeasible and budget remains, one HiGHS dual-simplex feasibility call for w>=0, A^T w=0, sumw=1. Select all positive returned support weights; reconstruct exact rows, solve that support once by rational RREF, verify nonnegative exact weights and exact zero sum. No further support search/retry. Numerical status alone is inconclusive.'
    p['decision']='Support only if all3 informative systems have independently verified feasible primals. Any checkable exact Farkas witness falsifies joint feasibility. Empty/no-leaf cases uninformative; numerical failure/timeout not falsification. No fresh inference.'
    p['source_sha256'].update({str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ['mine_leaf_negatives.py','solve_leaf_removability.py','verify_second_novelty.py']})
    S.save(PP,p);S.emit('leaf_removability_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['decoder_protocol_sha256']==S.sha(DECODER);assert p['previous_feasibility_protocol_sha256']==S.sha(S.HERE/'second-hard-negative-feasibility-protocol.json')
    for seed in range(3):assert S.sha(PRIOR/f'seed{seed}/candidate-theta.npy')==p['previous_witnesses'][str(seed)]['sha256'] and S.sha(PRIOR/f'all-negatives-seed{seed}.npz')==p['previous_witnesses'][str(seed)]['negative_archive_sha256']
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    assert S.sha(TRAIN/'training-arrays.npz')==p['training_manifest']['arrays_sha256']
    for seed in range(3):assert S.sha(TRAIN/f'train-readout-seed{seed}.npz')==p['readouts'][str(seed)]['readout_sha256'] and S.sha(S.ROOT/p['models'][str(seed)]['path'])==p['models'][str(seed)]['sha256']
    return p


def mine(gpu):
    lock,inventory=G.acquire(gpu);torch,_,_=S.imports();p=protocol();started=time.perf_counter();records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens']
    for seed in range(3):
        if seed%2!=gpu:continue
        path=DEST/f'mined-seed{seed}.npz'
        if path.exists():raise FileExistsError(path)
        model=S.model_for(seed,30,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True))
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
        values={k:[] for k in ['prediction','sweeps','stopped','finite','flip_count','minimum_flip_margin']};seconds=0.
        for lo in range(0,1000,64):
            torch.cuda.synchronize();before=time.perf_counter();t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');a=torch.tensor(attention[lo:lo+64],device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,t,a);out=C.solve(cache);torch.cuda.synchronize();seconds+=time.perf_counter()-before
            for key in values:values[key].append(out[key].cpu().numpy())
            S.emit('source_hard_negative_batch',gpu=gpu,seed=seed,inputs=min(lo+64,1000))
        arrays={k:np.concatenate(v) for k,v in values.items()};arrays['prediction']=arrays['prediction'].astype(bool);np.savez_compressed(path,**arrays)
        receipt={'status':'complete','seed':seed,'inputs':1000,'inventory':inventory,'protocol_sha256':S.sha(PP),'outputs_sha256':S.sha(path),'decoder_seconds_including_cache_and_input':seconds,'certified':int(arrays['stopped'].sum()),'finite':int(arrays['finite'].sum()),'maximum_sweeps':int(arrays['sweeps'].max())};S.save(DEST/f'mined-seed{seed}.json',receipt);records.append(receipt)
    S.save(DEST/f'mining-gpu{gpu}.json',{'status':'complete','inventory':inventory,'records':records,'seconds':time.perf_counter()-started,'new_independent_test_inputs':0,'source_model_input_trajectories':sum(r['inputs'] for r in records),'scope':'Originaltrainingonly. Labels/graphoutcomes notread byGPUworker; postminingCPUcheck separate.'})


def audit():
    from audit import independent
    p=protocol();started=time.perf_counter();records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    for t,y in zip(tokens,labels):assert independent(t,y)[0]
    for seed in range(3):
        source=DEST/f'mined-seed{seed}.npz';receipt=json.loads((DEST/f'mined-seed{seed}.json').read_text());assert S.sha(source)==receipt['outputs_sha256']
        with np.load(source) as z:pred=z['prediction'];finite=z['finite'];stopped=z['stopped'];margins=z['minimum_flip_margin']
        assert pred.shape==(1000,30,30) and finite.all();correct=np.array([independent(t,b)[0] for t,b in zip(tokens,pred)]);indices=np.flatnonzero(~correct);ham=np.count_nonzero((pred[indices]!=labels[indices])&(tokens[indices]==1),axis=(1,2));assert (ham>0).all()
        path=DEST/f'negatives-seed{seed}.npz';np.savez_compressed(path,indices=indices,boards=pred[indices],hamming=ham)
        mapping=[{'negative_row':i,'training_index':int(index),'free_hamming':int(ham[i]),'certified':bool(stopped[index] and margins[index]>=-1e-8),'board_sha256':S.sha_bytes(pred[index].tobytes()) if hasattr(S,'sha_bytes') else __import__('hashlib').sha256(pred[index].tobytes()).hexdigest()} for i,index in enumerate(indices)]
        record={'seed':seed,'negative_count':len(indices),'correct_valid_shortest':int(correct.sum()),'certified_negatives':sum(r['certified'] for r in mapping),'negative_archive_sha256':S.sha(path),'source_outputs_sha256':S.sha(source),'mapping':mapping,'deduplicated_rows':0};S.save(DEST/f'negative-map-seed{seed}.json',record);records.append(record)
    S.save(DEST/'mining-audit.json',{'status':'passed','utc':S.utc(),'records':records,'all_original_training_labels_valid_shortest':1000,'source_outputs_checked':3000,'new_independent_test_inputs':0,'seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'script_sha256':S.sha(__file__)})


def select():
    import verify_training_feasibility as V
    from verify_second_novelty import feature
    import hashlib
    p=protocol();started=time.perf_counter();records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens'];labels=z['labels']
    for seed in range(3):
        with np.load(DEST/f'negatives-seed{seed}.npz') as z:indices=z['indices'];boards=z['boards'];hamming=z['hamming']
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
        with np.load(S.ROOT/p['initial_models'][seed]['path']) as z:mu=z['mu_offsets'];ao=z['aoffs']
        cp=p['coefficient_arrays'][str(seed)];assert S.sha(S.ROOT/cp['path'])==cp['sha256'];theta=np.load(S.ROOT/cp['path']).astype(V.LD)
        rows=[];no_leaf=[];seen=set();minimum_old=json.loads((S.OUT/'second_negative_transfer/cast-gate.json').read_text())['records'][seed]['minimum_certified_positive_margin']
        for j,(index,b) in enumerate(zip(indices,boards)):
            t=tokens[index];y=labels[index];degree=np.zeros_like(t,np.int8);degree[1:]+=b[:-1];degree[:-1]+=b[1:];degree[:,1:]+=b[:,:-1];degree[:,:-1]+=b[:,1:];mask=(t==1)&b&~y&(degree==1);flat=np.flatnonzero(mask)
            if not len(flat):no_leaf.append({'negative_row':j,'training_index':int(index)})
            delta=V.flip(t[None],b[None],attention[index:index+1].astype(V.LD),theta,mu,ao)[0];base=feature(t,b,attention[index],mu,ao)
            for f in flat:
                z=b.copy();z.flat[f]=False;new=feature(t,z,attention[index],mu,ao);row=tuple(v-w for v,w in zip(base,new));digest=hashlib.sha256(json.dumps([[v.numerator,v.denominator] for v in row],separators=(',',':')).encode()).hexdigest();exact=-sum(v*__import__('fractions').Fraction.from_float(float(w)) for v,w in zip(row,theta));allowance=1e-6
                assert abs(float(exact)-float(delta.flat[f]))<allowance
                rows.append({'leaf_row':len(rows),'negative_row':j,'training_index':int(index),'flat':int(f),'selected_neighbor_count':int(degree.flat[f]),'previous_cast_flip_cost':float(delta.flat[f]),'previous_exact_flip_numerator':str(exact.numerator),'previous_exact_flip_denominator':str(exact.denominator),'new_exact_row_within_round':digest not in seen,'exact_row_sha256':digest,'provably_distinct_from_all_prior_rows':bool(-float(exact)+allowance<minimum_old)});seen.add(digest)
        records.append({'seed':seed,'wrong_boards':len(indices),'eligible_leaves':len(rows),'wrong_boards_without_leaf':no_leaf,'unique_new_leaf_rows':len(seen),'new_vs_prior_by_scalar_product_separation':sum(r['provably_distinct_from_all_prior_rows'] for r in rows),'minimum_previous_certified_old_margin':minimum_old,'mapping':rows,'informative':bool(len(indices)>0 and not no_leaf)})
    S.save(DEST/'leaf-selection.json',{'status':'complete','utc':S.utc(),'records':records,'seconds':time.perf_counter()-started,'protocol_sha256':S.sha(PP),'source_sha256':S.sha(__file__),'new_independent_test_inputs':0,'scope':'Every eligible false-positive four-neighbor leaf retained. Exact scalar products under frozen previous cast coefficients. Novelty follows when new-row scalar products are separated from the certified positive lower bound for every prior row; exact within-round row identity also checked. No subset selection.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','mine','audit','select']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='mine':mine(args.gpu)
    else:globals()[args.command]()
