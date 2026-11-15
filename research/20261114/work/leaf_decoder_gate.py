"""H2ah source-only fixed decoder gate after strict leaf-margin feasibility."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as G
import binary_coordinate as C
from prepare_leaf_decoder_gate import DEST,TRAIN,NORM

PP=S.HERE/'leaf-decoder-gate-protocol.json'
SOURCE=S.OUT/'leaf_removability'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=json.loads((S.HERE/'leaf-block-feasibility-protocol.json').read_text());proof=json.loads((NORM/'independent-verification.json').read_text());assert proof['status']=='passed';p={k:parent[k] for k in ['training_manifest','readouts','initial_models']}
    p.update(id='source-leaf-fitted-unchanged-binary-decoder-gate',hypothesis_id='H2ah',status='frozen_before_source_gate_inference',frozen_utc=S.utc(),hypothesis='Descriptive source target: at least50percent fewer wrong model-input outputs than the retained H2af580 wrong outputs, with no exclusions.',parent_protocol_sha256=S.sha(S.HERE/'leaf-block-feasibility-protocol.json'),parent_verification_sha256=S.sha(NORM/'independent-verification.json'))
    p['models']={str(seed):{'path':str((DEST/f'leaf-scaled-seed{seed}.pt').relative_to(S.ROOT)),'sha256':S.sha(DEST/f'leaf-scaled-seed{seed}.pt')} for seed in range(3)}
    p['coefficient_arrays']={str(seed):{'path':str((DEST/f'leaf-scaled-seed{seed}.npy').relative_to(S.ROOT)),'sha256':S.sha(DEST/f'leaf-scaled-seed{seed}.npy')} for seed in range(3)}
    p['gates']={}
    for name in ['cast-gate.json','numerical-gate.json']:
        path=DEST/name;assert json.loads(path.read_text())['status']=='passed';p['gates'][str(path.relative_to(S.ROOT))]=S.sha(path)
    p['source_sha256']={str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ['leaf_decoder_gate.py','prepare_leaf_decoder_gate.py','gate_leaf_decoder.py','binary_coordinate.py','study.py','audit.py','verify_training_feasibility.py']}
    p['retained_baseline']={'directory':str(SOURCE.relative_to(S.ROOT)),'mining_audit_sha256':S.sha(SOURCE/'mining-audit.json'),'independent_verification_sha256':S.sha(SOURCE/'independent-source-verification.json'),'wrong_by_model':[233,114,233],'wrong_total':580}
    p['scope']='All original1000 training inputs/model, fixed hidden readouts and original norm match plus one float32 cast. Unchanged H2ae/H2aa binary inference: allopen, float64 promotion, fourcolors, threshold-1e-8, cap512. Source labels only after inference. No fresh target input, subset exclusion, coefficient tuning or graph repair.'
    p['decision']='Report source wrong-count reduction descriptively; target met iff totalwrong<=290. No independent test/generalization or confidence claim; preserve all cases including previous leafless boards.'
    p['budget']={'total_gpu_minutes':5,'overall_wall_seconds':180,'per_worker_wall_seconds':150,'watchdog':'Check every source batch, retain partial and stop; independent source/state audit separate.'}
    S.save(PP,p);S.emit('leaf_decoder_gate_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['parent_protocol_sha256']==S.sha(S.HERE/'leaf-block-feasibility-protocol.json') and p['parent_verification_sha256']==S.sha(NORM/'independent-verification.json')
    for path,h in {**p['source_sha256'],**p['gates']}.items():assert S.sha(S.ROOT/path)==h,path
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
            if time.perf_counter()-started>150:
                np.savez_compressed(DEST/f'partial-seed{seed}.npz',**{k:np.concatenate(v) for k,v in values.items()});raise TimeoutError('Frozen150sworkerbudget')
            S.emit('source_leaf_decoder_batch',gpu=gpu,seed=seed,inputs=min(lo+64,1000))
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
