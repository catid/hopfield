"""H2ad original-training hard-negative mining with frozen H2ac coefficients and unchanged H2aa decoder."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as G
import binary_coordinate as C
from hard_negative_transfer import PP as DECODER,protocol as decoder_protocol
from prepare_lp_transfer import TRAIN

DEST=S.OUT/'second_hard_negative_feasibility'
PRIOR=S.OUT/'hard_negative_feasibility'
PP=S.HERE/'second-hard-negative-feasibility-protocol.json'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=decoder_protocol();p=json.loads((S.HERE/'hard-negative-feasibility-protocol.json').read_text());DEST.mkdir(exist_ok=True)
    p.update(id='second-original-training-hard-negative-feasibility',hypothesis_id='H2ad',frozen_utc=S.utc(),hypothesis='Every model yields at least one genuinely new source negative board and admits a real23-parameter vector satisfying every H2w/H2ab constraint plus every second-round Hamming-normalized unit margin.')
    p['models']={str(seed):parent['models'][str(seed)]['hard_negative_binary'] for seed in range(3)};p['decoder_protocol_sha256']=S.sha(DECODER);p['previous_feasibility_protocol_sha256']=S.sha(S.HERE/'hard-negative-feasibility-protocol.json')
    p['previous_witnesses']={str(seed):{'sha256':S.sha(PRIOR/f'seed{seed}/candidate-theta.npy'),'negative_archive_sha256':S.sha(PRIOR/f'negatives-seed{seed}.npz')} for seed in range(3)}
    p['constraints']['old']='Every H2w original source constraint and all855 H2ab mined-negative rows, unchanged, no deduplication.'
    p['informativeness']={'board_novelty':'Training-index plus full-board identity versus H2ab and earlier rows; require>=1 genuinely new negative board/model for feasibility support.','constraint_novelty':'Exact rational normalized23-vector identity versus every prior H2w and H2ab row and earlier second-round rows. Rounded first18 coordinates only filter candidates; exact rational comparison decides. All rows retained even if duplicates.','violation':'Count exact normalized margin<1 under previous UNSCALED H2ab feasibility witness, with direct extended-precision energy cross-check and conservative allowance reported separately.','interpretation':'Novelty and violations reported separately. If zero violated rows, explicitly report absence of new ranking violations and do not attribute future changes to repairing them.'}
    p['decision']='Support only if all3 have >=1 genuinely new negative board and independently verified feasible primal for all old/new rows. Exact infeasibility certificate required to falsify real feasibility; numerical failure/timeout inconclusive. Empty or duplicate-only mining uninformative. No fresh test authorized.'
    p['source_sha256'].update({str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ['mine_second_hard_negatives.py','solve_second_hard_negatives.py','second_negative_novelty.py']})
    S.save(PP,p);S.emit('second_hard_negative_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['decoder_protocol_sha256']==S.sha(DECODER);assert p['previous_feasibility_protocol_sha256']==S.sha(S.HERE/'hard-negative-feasibility-protocol.json')
    for seed in range(3):assert S.sha(PRIOR/f'seed{seed}/candidate-theta.npy')==p['previous_witnesses'][str(seed)]['sha256'] and S.sha(PRIOR/f'negatives-seed{seed}.npz')==p['previous_witnesses'][str(seed)]['negative_archive_sha256']
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


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','mine','audit']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    if args.command=='mine':mine(args.gpu)
    else:globals()[args.command]()
