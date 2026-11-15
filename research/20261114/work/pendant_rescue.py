"""H2ao finite post-hoc candidate-family rescue from archived H2an veto outputs."""
import argparse
import hashlib
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import pendant_block_moves as B
import fast_npz as Z

PP=S.HERE/'pendant-rescue-protocol.json'
DEST=S.OUT/'pendant_rescue'
PARENT=S.OUT/'safe_block_transfer'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=json.loads((S.HERE/'safe-block-transfer-protocol.json').read_text());a=json.loads((PARENT/'audit.json').read_text());assert a['status']=='validated' and a['joint_success']['connectivity_veto']==4074;original=(S.HERE/'graph_block_moves.py').read_text().split('def scores(',1)[1];current=(S.HERE/'pendant_block_moves.py').read_text().split('def scores(',1)[1];assert original==current
    p={'id':'seen-cohort-articulation-pendant-candidate-family-rescue','hypothesis_id':'H2ao','status':'frozen_before_control_certificate_gate_and_newtrajectory','frozen_utc':S.utc(),'parent_protocol_sha256':S.sha(S.HERE/'safe-block-transfer-protocol.json'),'parent_audit_sha256':S.sha(PARENT/'audit.json'),'models':{s:x['old_binary'] for s,x in parent['models'].items()},'model_arrays':{str(s):{'path':str((PARENT/f'model-seed{s}-old_binary.npz').relative_to(S.ROOT)),'sha256':S.sha(PARENT/f'model-seed{s}-old_binary.npz')} for s in range(3)},'jobs':parent['jobs'],'datasets':json.loads((PARENT/'datasets.json').read_text())['datasets'],'source_outputs':{r['job']['id']:{'path':str((PARENT/'runs'/r['job']['id']/'outputs.npz').relative_to(S.ROOT)),'sha256':r['outputs_sha256'],'archived_answer_prefix_seconds':r['answer_seconds_by_arm'][3],'archived_shared_hidden_seconds':r['hidden_seconds_shared']} for r in a['records']},'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['pendant_rescue.py','pendant_block_moves.py','gate_pendant_rescue.py','verify_pendant_geometry.py','certify_chain_fixed_points.py','connected_block_moves.py','graph_block_moves.py','binary_coordinate.py','fast_npz.py','study.py','verify_training_feasibility.py','verify_graph_block_source.py']},'unchanged_score_solve_suffix_sha256':hashlib.sha256(original.encode()).hexdigest(),'candidate_rule':'Within eachselected4neighbor connectedcomponent, remove eacharticulationvertex inturn and consider everyremainingcomponent containingneitherendpoint; candidate excludes thearticulation. Require2..256cells,allselectedfree,then selectedendpointconnectivity afterdeletion. Unrelatedconnectedcomponents not behind thecut are not candidates. Deduplicate exactsorted flattenedcelltuples, retainallcutvertex witnesses. No labels/shortestdistance.','determinism':'Vertices increasingflattenedindex; componentroots increasingindex; iterativeDFS neighbors increasingindex; Tarjandiscovery/low/subtree intervals. Enumerate articulationvertices increasingindex; sort/dedup candidatecelltuples. Choose mostnegative fulloriginalenergy delta<−1e−8, exactscoreties lexicographiccelltuple.','policy':'Initialsimple endpointpaths return byte-identicalanswer/flags; no search. Allother eligible frozenvetooutputs start pendant-only search. Sameoriginalenergy, fixedoldLP, max16rounds and512sweep nativebinarydescent afterone deletion/input/round. Allcaps/nonfinite/disconnections retained. No oldchainmoves added to the newfamily.','control':'Independently check everyoldnative margin and everyoldchain-veto terminal candidate againstliteraloriginalenergy beforeusing zero-search unchangedcontinuation. Cap-terminal/nonfinite outputs cannot befixedpoints. No allowedfirstmove implies additionaloldpolicybudget alone cannotchange output.','primary':{'initial_joint_failures':22,'minimum_joint_rescues':11,'maximum_joint_regressions':0,'decision':'Report>=11of22rescues andzero pairedregressions separately. No confidence/freshgeneralizationclaim. Conditionalidentity oncorrectsimplepaths is mathematical, rescuecount is measured.'},'scope':'All4096alreadyseenH2aninputs/all3fixedoldLPmodels, no caseexclusion; retainBPcomparison. Explicit articulation/reachability graphprior.','budget':{'new_trajectory_gpu_minutes':10,'per_worker_wall_seconds':300,'source_numerical_gate_wall_seconds':120,'fixedpoint_control_wall_seconds':120,'fixedpoint_control_processes':16,'independent_audit_wall_seconds':300,'independent_audit_processes':16,'watchdog':'Perbatch/blockround, preservecompletedarchives, noextensions.'},'charging':'Each answer charged archivedfullveto answerprefix plusnew selector/cache/search. Archivedsharedhidden andnewdiagnostics/archive separated; exact reusedprefix no rerun.','accounting':{'fresh_inputs':0,'included_model_inputs':12288,'reused_prefixes':12288,'new_searches':'Only nonsimple eligibleinitialoutputs, actualcountreported.','oldpolicy_control_new_searches':0,'new_optimizer_updates':0}}
    S.save(PP,p);S.emit('pendant_rescue_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['parent_protocol_sha256']==S.sha(S.HERE/'safe-block-transfer-protocol.json') and p['parent_audit_sha256']==S.sha(PARENT/'audit.json')
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for group in [p['models'],p['model_arrays']]:
        for x in group.values():assert S.sha(S.ROOT/x['path'])==x['sha256']
    return p


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol()
    for name in ['numerical-gate.json','old-chain-fixed-point-control.json']:
        r=json.loads((DEST/name).read_text());assert r['status']=='passed' and r['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();models={};state={'status':'running','gpu':gpu,'pid':os.getpid(),'inventory':inventory,'started_utc':S.utc(),'jobs_complete':[]}
    def watch():
        if time.perf_counter()-started>300:raise TimeoutError('Frozen300sworkerbudget; completebatcharchivesretained')
    for seed in range(3):
        m=S.model_for(seed,120,'original','cuda',torch.float64);m.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True));models[seed]=m
    for job in p['jobs']:
        if job['gpu']!=gpu:continue
        folder=DEST/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=True);source=p['source_outputs'][job['id']];assert S.sha(S.ROOT/source['path'])==source['sha256'];ds=p['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256'];before_job=time.perf_counter()
        with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
        with np.load(S.ROOT/source['path']) as z:initial=z['predictions'][3];readout=z['couplings'];initial_stopped=z['stopped'][3];initial_finite=z['finite'][3];prefix_iterations=z['iterations'][3];initial_margin=z['minimum_flip_margin64'][3];initial_energy=z['energy64'][3]
        before=time.perf_counter();simple=np.array([B.simple_path(t,b) for t,b in zip(tokens,initial)]);selector_seconds=time.perf_counter()-before;eligible=initial_stopped&initial_finite&~simple;saved={k:[] for k in ['prediction','stopped','finite','block_moves','native_sweeps','native_cap','block_cap','minimum_flip_margin','energy','initial_simple_path']};batches=[];timings={k:0. for k in ['cpu_graph_seconds','score_seconds','descent_seconds']};search_seconds=diagnostic_seconds=archive_seconds=0.
        with torch.no_grad():
            for lo in range(0,256,64):
                watch();ix=np.flatnonzero(eligible[lo:lo+64]);arrays={'prediction':initial[lo:lo+64].copy(),'stopped':initial_stopped[lo:lo+64].copy(),'finite':initial_finite[lo:lo+64].copy(),'block_moves':np.zeros(64,np.int64),'native_sweeps':np.zeros(64,np.int64),'native_cap':~initial_stopped[lo:lo+64]&initial_finite[lo:lo+64],'block_cap':np.zeros(64,bool),'minimum_flip_margin':initial_margin[lo:lo+64].copy(),'energy':initial_energy[lo:lo+64].copy(),'initial_simple_path':simple[lo:lo+64].copy()};raw=None
                if len(ix):
                    torch.cuda.synchronize();before=time.perf_counter();t=torch.tensor(tokens[lo:lo+64][ix].astype(np.int64),device='cuda');a=torch.tensor(readout[lo:lo+64][ix],device='cuda',dtype=torch.float64);board=torch.tensor(initial[lo:lo+64][ix],device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(models[job['seed']],t,a);raw=B.solve(cache,board,watch);torch.cuda.synchronize();search_seconds+=time.perf_counter()-before
                    before=time.perf_counter()
                    for key in ['prediction','stopped','finite','block_moves','native_sweeps','native_cap','block_cap']:arrays[key][ix]=raw[key].cpu().numpy().astype(bool) if key=='prediction' else raw[key]
                    arrays['minimum_flip_margin'][ix]=models[job['seed']].flip_delta(t,raw['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy();arrays['energy'][ix]=models[job['seed']].energy(t,raw['prediction'],a).cpu().numpy();diagnostic_seconds+=time.perf_counter()-before
                    for k in timings:timings[k]+=raw[k]
                before=time.perf_counter();path=folder/f'batch{lo//64:02d}.npz';payload={'protocol_sha256':S.sha(PP),'offset':lo,'trace':raw['trace'] if raw else []};Z.save(path,**arrays,intervention_indices=ix,states_before=raw['states_before'] if raw else np.zeros((0,0,120,120),bool),trace_utf8=np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8));batches.append({'offset':lo,'path':path.name,'sha256':S.sha(path)});archive_seconds+=time.perf_counter()-before
                for k in saved:saved[k].append(arrays[k])
                state.update(current_job=job,completed_inputs=lo+64,wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state);watch()
        arrays={k:np.concatenate(v) for k,v in saved.items()};arrays['iterations']=prefix_iterations+arrays['native_sweeps'];before=time.perf_counter();Z.save(folder/'outputs.npz',**arrays);archive_seconds+=time.perf_counter()-before;increment=selector_seconds+search_seconds;r={'status':'complete','job':job,'protocol_sha256':S.sha(PP),'source_outputs_sha256':source['sha256'],'outputs_sha256':S.sha(folder/'outputs.npz'),'batches':batches,'initial_simple_paths_skipped':int(simple.sum()),'new_search_inputs':int(eligible.sum()),'archived_answer_prefix_seconds':source['archived_answer_prefix_seconds'],'archived_shared_hidden_seconds':source['archived_shared_hidden_seconds'],'new_incremental_answer_seconds':increment,'charged_answer_seconds':source['archived_answer_prefix_seconds']+increment,'selector_seconds':selector_seconds,'search_seconds':search_seconds,'diagnostic_seconds':diagnostic_seconds,'archive_seconds':archive_seconds,**timings,'seconds':time.perf_counter()-before_job};S.save(folder/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('pendant_rescue_job_complete',job=job['id'],gpu=gpu,new_search_inputs=int(eligible.sum()),elapsed=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args()
    if v.command=='evaluate':evaluate(v.gpu)
    else:freeze()
