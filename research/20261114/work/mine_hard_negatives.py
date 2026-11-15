"""H2ab original-training hard-negative mining with frozen H2aa decoder."""
import argparse
import json
import time
import numpy as np
import study as S
import selective_fallback as G
import binary_coordinate as C
from binary_coordinate_study import PP as DECODER,protocol as decoder_protocol
from prepare_lp_transfer import TRAIN

DEST=S.OUT/'hard_negative_feasibility'
PP=S.HERE/'hard-negative-feasibility-protocol.json'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=decoder_protocol();old=json.loads((S.HERE/'training-feasibility-protocol.json').read_text());DEST.mkdir(exist_ok=True)
    p={'id':'original-training-mined-negative-linear-feasibility','hypothesis_id':'H2ab','status':'frozen_before_mining_and_LP','frozen_utc':S.utc(),'hypothesis':'Each of3fixedreadouts yieldsatleast1wrongsourcebinaryoutput and admits a real23parameter vector satisfyingallpriorunitmarginconstraints plusallminednegative Hamming-normalized unitmargins.','training_manifest':old['training_manifest'],'readouts':old['readouts'],'initial_models':old['initial_models'],'models':{str(seed):parent['models'][str(seed)]['lp_binary'] for seed in range(3)},'decoder_protocol_sha256':S.sha(DECODER),'old_constraints_protocol_sha256':S.sha(S.HERE/'training-feasibility-protocol.json'),
      'mining':{'source':'Allactualoriginal1000trainingarrays permodel, samefrozenfloat32hiddenreadouts; no generatedtrainingdata or targetdata.','decoder':'UnchangedH2aa: allopen, float64promotion, colors(0,0),(0,1),(1,0),(1,1), delta<-1e-8, max512sweeps.','batch':64,'negative_rule':'Retain everyfinitebinaryoutput thatfails independentvalidshortest checking onits suppliedgraph. Preservealloutputs/flags includinguncertifiedones; labelwrongstable status separately. Do not treat alternatevalidshortestboards asnegatives. Verify canonicaltraininglabels independently.','deduplication':'None. Retainallnegative rows, trainingindex/boardhash and identityrowmapping, evenifduplicates.','empty':'Amodelwithzero minedwrongboards is uninformative; no inventednegative or extra mininground.','timing':'Generation/loading,GPUmining andCPUgraphclassification reportedseparately and excludedfromLP120sbudget. Traininginputs reused, not newindependenttestevidence.'},
      'constraints':{'old':'ExactlyH2w: everyfreecanonical-label flipmargin>=1, normalizedendpoint andallopenrankingmargins>=1.','new':'For eachretainedwrongboardz, E(z)-E(y)>=free-cellHamming(z,y)>0; dividebythatpositiveinteger forunitnormalizedrow.','parameters':'Unrestrictedreal23coordinates T[4,2,2],b[2],kappa[5]; hiddenmu/beta/readouts fixed. No artificialbound/gauge/signrestriction.'},
      'solver':{'method':'OneSciPyHiGHSdualsimplex call/model minimizingcoordinateLinf via free23theta andt>=0, with±theta<=t.','options':{'presolve':True,'threads':1,'small_matrix_value':1e-12,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9},'candidate':'Ifsolverreturns success, exactly2x returnedtheta once; no retuning/retries. Independently verifyallold/newunitmargins afterconservativefloat64allowance. Numericalminimum-normobjective doesnot constitute anoptimalitycertificate.'},
      'decision':'Support onlyifall3have>=1minednegative andindependentlyverifiedfeasibleprimals. Falsificationrequires independentlyverifiedexactinfeasibilitycertificate; numericalinfeasible/status/timeout alone inconclusive. Emptymining separatelyuninformative. No freshH2ac underthisprotocol.',
      'budget':{'aggregate_cpu_wall_seconds':120,'includes':'Solverrunentry imports, hashes/dataloading, all3coefficientextractions/LPs/primalchecks/receipts.','watchdog':'SIGALRM120 plusremainingHiGHStime_limit; no extension, alternate solver, additionaloptimization oroutcome-selectedconstraints. Read-onlyindependentcertificateaudits separate.'},
      'primal_verification':'IndependentdirectNumPy extendedprecision pairtable/directedlink energy andflip arithmetic. Sameconservativefloat64allowance asH2w: m=256*900+1000,u=2^-53,gamma=m*u/(1-m*u),allowance=4gamma(10000max(1,||theta||inf)+1); requireallnormalizedmargins>1+allowance.',
      'source_sha256':{str(path.relative_to(S.ROOT)):S.sha(path) for path in [S.HERE/'mine_hard_negatives.py',S.HERE/'solve_hard_negatives.py',S.HERE/'binary_coordinate.py',S.HERE/'feasibility_features.py',S.HERE/'verify_training_feasibility.py',S.HERE/'study.py',S.HERE/'audit.py']}}
    S.save(PP,p);S.emit('hard_negative_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['decoder_protocol_sha256']==S.sha(DECODER)
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
