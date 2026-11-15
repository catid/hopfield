"""H2ai frozen source-only graph-chain block intervention; baseline outputs reused."""
import argparse
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import graph_block_moves as B
from mine_leaf_negatives import PP as BASE_PROTOCOL,DEST as BASE,TRAIN,protocol as baseline_protocol

DEST=S.OUT/'graph_block_source'
PP=S.HERE/'graph-block-source-protocol.json'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=baseline_protocol();DEST.mkdir(exist_ok=True);p={k:parent[k] for k in ['training_manifest','readouts','initial_models','models','coefficient_arrays']}
    p.update(id='source-only-label-free-selected-chain-block-descent',hypothesis_id='H2ai',status='frozen_before_gate_and_source_intervention',frozen_utc=S.utc(),baseline_protocol_sha256=S.sha(BASE_PROTOCOL),baseline_wrong_by_model=[233,114,233],baseline_wrong_total=580)
    p['source_outputs']={str(seed):{'path':str((BASE/f'mined-seed{seed}.npz').relative_to(S.ROOT)),'sha256':S.sha(BASE/f'mined-seed{seed}.npz')} for seed in range(3)}
    p['candidate_rule']='Compute selected four-neighbor degree. Eligible vertices are selected free cells with global selected degree1or2. Every maximal four-connected component of these eligible vertices is a free-leaf/degree2 chain or a closed free cycle. Retain each component with2..256cells. All clamps protected, including walls and endpoints. Protected junctions/clamps split chains; no labels, shortest distance, graph validity oracle, targets or search-based repair in candidate selection.'
    p['energy_rule']='For deleting selected setM, delta=sum original single-flip costs overM plus every internal quadratic interaction once. Include all original pair/link interactions, including diagonals. Exact algebra checked against literal before/after full energy. Float64 promotion of frozenfloat32 coefficients/readout.'
    p['policy']={'threshold':B.THRESHOLD,'max_cells':B.MAX_CELLS,'max_rounds':B.MAX_ROUNDS,'choice':'Most negative full-original-energy delta below-1e-8; exact score ties resolved by sorted flattened cell tuple. One move/input/round.','after_move':'Run unchanged four-color binary solver from modified board, cap512sweeps; stop that input on nativecap/nonfinite.','stop':'No improving allowed block. Inputs still active after16rounds are block-cap failures, no postcapcredit.','baseline':'Reuse every retained H2af original one-bit output; no extra original-prefix trajectories. Coefficients fixed H2ae, no H2ag fit.'}
    p['decision']='Descriptive fixed-source target met iff complete, finite, locally certified valid-shortest wrong model-input count<=290 (at least50percent reduction vs580). All1000 source inputs/model retained, including regressions, endpointdisconnections, native/blockcap failures. No independent test or confidence claim.'
    p['budget']={'gate_cpu_seconds':120,'whole_study_gpu_minutes':5,'whole_study_wall_seconds':300,'per_worker_wall_seconds':150,'watchdog':'Per round and batch; preserve partial states on expiry. CPU graph work, GPU scoring and native descent separately timed; CPU bottleneck explicit.'}
    p['source_sha256']={str((S.HERE/name).relative_to(S.ROOT)):S.sha(S.HERE/name) for name in ['graph_block_study.py','graph_block_moves.py','gate_graph_block_moves.py','binary_coordinate.py','study.py','audit.py']};S.save(PP,p);S.emit('graph_block_protocol_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['baseline_protocol_sha256']==S.sha(BASE_PROTOCOL)
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for records in [p['models'],p['coefficient_arrays'],p['source_outputs']]:
        for item in records.values():assert S.sha(S.ROOT/item['path'])==item['sha256']
    assert S.sha(TRAIN/'training-arrays.npz')==p['training_manifest']['arrays_sha256']
    return p


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();gate=json.loads((DEST/'numerical-gate.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==S.sha(PP);started=time.perf_counter();records=[]
    with np.load(TRAIN/'training-arrays.npz') as z:tokens=z['tokens']
    state={'status':'running','gpu':gpu,'pid':os.getpid(),'inventory':inventory,'started_utc':S.utc(),'completed_model_inputs':0}
    def watchdog():
        if time.perf_counter()-started>150:raise TimeoutError('Frozen150sworker wall budget')
    for seed in range(3):
        if seed%2!=gpu:continue
        folder=DEST/f'seed{seed}';folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():raise FileExistsError(folder)
        model=S.model_for(seed,30,'original','cuda',torch.float64);model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True))
        with np.load(TRAIN/f'train-readout-seed{seed}.npz') as z:attention=z['couplings']
        with np.load(S.ROOT/p['source_outputs'][str(seed)]['path']) as z:initial=z['prediction'];assert z['stopped'].all() and z['finite'].all()
        saved={k:[] for k in ['prediction','stopped','finite','block_moves','native_sweeps','native_cap','block_cap','minimum_flip_margin','upstream_minimum_flip_margin','energy']};receipts=[];timings={k:0. for k in ['cpu_graph_seconds','score_seconds','descent_seconds']};seed_start=time.perf_counter()
        with torch.no_grad():
            for lo in range(0,1000,64):
                t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');a=torch.tensor(attention[lo:lo+64],device='cuda',dtype=torch.float64);board=torch.tensor(initial[lo:lo+64],device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(model,t,a)
                out=B.solve(cache,board,watchdog);path=folder/f'batch{lo//64:02d}.npz';pred=out['prediction'];margin=model.flip_delta(t,pred,a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy();energy=model.energy(t,pred,a).cpu().numpy();assert np.max(abs(margin-out['minimum_flip_margin']))<1e-10
                arrays={k:out[k] for k in saved if k not in ['prediction','upstream_minimum_flip_margin','energy']};arrays.update(prediction=pred.cpu().numpy().astype(bool),upstream_minimum_flip_margin=margin,energy=energy,states_before=out['states_before']);np.savez_compressed(path,**arrays);trace=folder/f'batch{lo//64:02d}-trace.json';S.save(trace,{'offset':lo,'trace':out['trace'],'protocol_sha256':S.sha(PP)})
                for key in saved:saved[key].append(arrays[key])
                for key in timings:timings[key]+=out[key]
                receipts.append({'offset':lo,'n':len(t),'states_sha256':S.sha(path),'trace_sha256':S.sha(trace)});state.update(current_seed=seed,completed_model_inputs=state['completed_model_inputs']+len(t),worker_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state);S.emit('graph_block_source_batch',gpu=gpu,seed=seed,inputs=min(lo+64,1000),elapsed=state['worker_seconds']);watchdog()
        np.savez_compressed(folder/'outputs.npz',**{k:np.concatenate(v) for k,v in saved.items()});record={'status':'complete','seed':seed,'source_inputs':1000,'protocol_sha256':S.sha(PP),'outputs_sha256':S.sha(folder/'outputs.npz'),'baseline_output_sha256':p['source_outputs'][str(seed)]['sha256'],'batches':receipts,**timings,'seconds':time.perf_counter()-seed_start};S.save(folder/'result.json',record);records.append(record)
    state.update(status='complete',seconds=time.perf_counter()-started,records=records);S.save(DEST/f'worker{gpu}.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','evaluate']);p.add_argument('--gpu',type=int,choices=[0,1]);a=p.parse_args()
    if a.command=='freeze':freeze()
    else:evaluate(a.gpu)
