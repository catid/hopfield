"""H2aw cap512 source intervention from the same archived H2av lexical256 outputs."""
import argparse
import hashlib
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import large_cycle_cut_moves as B
import fast_npz as Z

PP=S.HERE/'large-cycle-cut-protocol.json'
DEST=S.OUT/'large_cycle_cut'
PARENT=S.OUT/'cycle_transfer_180'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    parent=json.loads((PARENT/'protocol.json').read_text());audit=json.loads((PARENT/'audit.json').read_text());gate_path=S.OUT/'large_cycle_cut_preflight/numerical-gate.json';control_path=S.OUT/'large_cycle_cut_preflight/old256-fixed-point-control.json';gate=json.loads(gate_path.read_text());control=json.loads(control_path.read_text());assert audit['status']=='passed' and audit['joint_success']['lexical_cycle_cut']==4068;assert gate['status']==control['status']=='passed' and control['model_inputs']==12288 and control['nonsimple_terminal_cases']==84;assert gate['helper_sha256']==S.sha(S.HERE/'large_cycle_cut_moves.py') and gate['source_sha256']==S.sha(S.HERE/'preflight_large_cycle_cut.py');assert control['source_sha256']==S.sha(S.HERE/'certify_cycle256_fixed_points.py');assert gate['parent_protocol_sha256']==control['parent_protocol_sha256']==S.sha(PARENT/'protocol.json')
    models={str(seed):parent['models'][str(seed)]['old_binary'] for seed in range(3)};ex=json.loads((PARENT/'model-extraction.json').read_text());arrays={str(r['seed']):{'path':str((PARENT/r['path']).relative_to(S.ROOT)),'sha256':r['sha256']} for r in ex['records'] if r['arm']=='old_binary'};hashes={r['job']:r['source_outputs_sha256'] for r in control['rows']};costs={r['job']:r['answer_seconds'] for r in audit['run_rows'] if r['arm']=='lexical_cycle_cut'};hidden={r['job']['id']:r['hidden_seconds'] for r in audit['rows']};jobs=[{k:v for k,v in j.items() if k!='gpu'} for j in parent['jobs']]
    names=['large_cycle_cut.py','large_cycle_cut_moves.py','verify_large_cycle_geometry.py','preflight_large_cycle_cut.py','certify_cycle256_fixed_points.py','cycle_cut_moves.py','connected_block_moves.py','graph_block_moves.py','binary_coordinate.py','fast_npz.py','study.py','verify_training_feasibility.py']
    p={'id':'seen180-cap512-shared-lexical256-prefix','hypothesis_id':'H2aw','status':'frozen_after_complete_control_and_numerical_preflight_before_interventions','frozen_utc':S.utc(),'parent_protocol_sha256':S.sha(PARENT/'protocol.json'),'parent_audit_sha256':S.sha(PARENT/'audit.json'),'residual_diagnosis_sha256':S.sha(PARENT/'residual-size-diagnosis.json'),'arms':['inherited_lexical256','lexical512_from_lexical256','energy512_from_lexical256'],'new_policies':{'0':'lexical512','1':'energy512'},'models':models,'model_arrays':arrays,'jobs':jobs,'datasets':json.loads((PARENT/'datasets.json').read_text())['datasets'],'source_outputs':{j['id']:{'path':str((PARENT/'runs'/j['id']/'outputs.npz').relative_to(S.ROOT)),'sha256':hashes[j['id']],'archived_answer_prefix_seconds':costs[j['id']],'archived_shared_hidden_seconds':hidden[j['id']]} for j in jobs},'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names},'preflight':{'numerical':{'path':str(gate_path.relative_to(S.ROOT)),'sha256':S.sha(gate_path)},'old256_fixed_point_control':{'path':str(control_path.relative_to(S.ROOT)),'sha256':S.sha(control_path)}},'candidate_rule':'Exactly H2av selectedfree globaldegree1or2 maximalchain family withminimum2cells, butmaximum512 instead of256; unchangedendpointreachability andstrictselected4neighborcycle-rank decrease. No other geometricchange.','policy':'All4096H2avinputs/all3models; BOTH newpolicies start from identicalarchivedH2avLEXICAL256 output arm4, includingpreviouslexicalcuts. This isnot continuationofarchivedenergyranked256 policy. Initialsimplepaths copiedbyte-identically withflags. Nonsimple finite/stopped prefixes searched; lexicalminimumcelltuple orminimumfulloriginalenergy/lexictie. Bothacceptpositivecost; sameoriginalenergy/readout/nativefourcolor512sweeps and16blockrounds;caps retained asfailures. No label/shortestdistance selection.','primary':{'candidate':'lexical512_from_lexical256','control':'inherited_lexical256','remaining_joint_failures':28,'minimum_joint_rescues':14,'maximum_joint_regressions':0,'decision':'Finite source target, report>=14rescuesandzero regressions separately. No freshgeneralization/confidenceclaim.'},'secondary':'Energy512fromsamelexical256prefix descriptivepairedrankingcontrol; retain archivedBP/binary/veto/energy256/lexical256 comparisons withcompositionlabels. Old256control hasnoallowedfirstmove, certifiedindependently, soextrabudgetalone cannotmoveoldanswers.','budget':{'per_worker_wall_seconds':300,'aggregate_new_trajectory_gpu_minutes':10,'source_gate_wall_seconds':120,'fixedpoint_control_wall_seconds':120,'independent_audit_wall_seconds':300,'independent_audit_processes':16,'watchdog':'Perbatch/blockround; externalaudit timeout; preservepartials,noextensions.'},'charging':'Fullarchivedlexical256answerprefix plusnewselector/cache/searchcharged toeachnewarm. Hiddenarchivedseparate;nohidden/BP/binary/vetoprefix rerun. OneGPUpernewpolicy,all48jobs each. No optimizedruntimeclaim.','accounting':{'fresh_inputs':0,'included_model_inputs':12288,'reused_archived_prefixes_per_new_policy':12288,'new_search_inputs':'All eligible nonsimple starts perpolicy, actualcountreported','new_optimizer_updates':0}}
    S.save(PP,p);DEST.mkdir(exist_ok=True);S.save(DEST/'protocol.json',p)
    for name,ref in [('numerical-gate.json',p['preflight']['numerical']),('old256-fixed-point-control.json',p['preflight']['old256_fixed_point_control'])]:r=json.loads((S.ROOT/ref['path']).read_text());r.update(protocol_sha256=S.sha(PP),source_preflight=ref,new_gate_runs=0);S.save(DEST/name,r)
    S.emit('large_cycle_cut_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['parent_protocol_sha256']==S.sha(PARENT/'protocol.json') and p['parent_audit_sha256']==S.sha(PARENT/'audit.json') and p['residual_diagnosis_sha256']==S.sha(PARENT/'residual-size-diagnosis.json')
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for group in [p['models'],p['model_arrays']]:
        for x in group.values():assert S.sha(S.ROOT/x['path'])==x['sha256']
    for ref in p['preflight'].values():assert S.sha(S.ROOT/ref['path'])==ref['sha256']
    return p


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol()
    for name in ['numerical-gate.json','old256-fixed-point-control.json']:
        r=json.loads((DEST/name).read_text());assert r['status']=='passed' and r['protocol_sha256']==S.sha(PP)
    started=time.perf_counter();models={};policy=p['new_policies'][str(gpu)];state={'status':'running','gpu':gpu,'policy':policy,'pid':os.getpid(),'inventory':inventory,'started_utc':S.utc(),'jobs_complete':[]}
    def watch():
        if time.perf_counter()-started>300:raise TimeoutError('Frozen300sworkerbudget; completebatcharchivesretained')
    for seed in range(3):
        m=S.model_for(seed,180,'original','cuda',torch.float64);m.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True));models[seed]=m
    for job in p['jobs']:
        folder=DEST/policy/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=True);source=p['source_outputs'][job['id']];assert S.sha(S.ROOT/source['path'])==source['sha256'];ds=p['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256'];before_job=time.perf_counter()
        with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
        with np.load(S.ROOT/source['path']) as z:initial=z['predictions'][4];readout=z['couplings'];initial_stopped=z['stopped'][4];initial_finite=z['finite'][4];prefix_iterations=z['iterations'][4];initial_margin=z['minimum_flip_margin64'][4];initial_energy=z['energy64'][4]
        before=time.perf_counter();simple=np.array([B.simple_path(t,b) for t,b in zip(tokens,initial)]);selector_seconds=time.perf_counter()-before;eligible=initial_stopped&initial_finite&~simple;saved={k:[] for k in ['prediction','stopped','finite','block_moves','native_sweeps','native_cap','block_cap','minimum_flip_margin','energy','initial_simple_path']};batches=[];timings={k:0. for k in ['cpu_graph_seconds','score_seconds','descent_seconds']};search_seconds=diagnostic_seconds=archive_seconds=0.
        with torch.no_grad():
            for lo in range(0,256,32):
                watch();ix=np.flatnonzero(eligible[lo:lo+32]);arrays={'prediction':initial[lo:lo+32].copy(),'stopped':initial_stopped[lo:lo+32].copy(),'finite':initial_finite[lo:lo+32].copy(),'block_moves':np.zeros(32,np.int64),'native_sweeps':np.zeros(32,np.int64),'native_cap':~initial_stopped[lo:lo+32]&initial_finite[lo:lo+32],'block_cap':np.zeros(32,bool),'minimum_flip_margin':initial_margin[lo:lo+32].copy(),'energy':initial_energy[lo:lo+32].copy(),'initial_simple_path':simple[lo:lo+32].copy()};raw=None
                if len(ix):
                    torch.cuda.synchronize();before=time.perf_counter();t=torch.tensor(tokens[lo:lo+32][ix].astype(np.int64),device='cuda');a=torch.tensor(readout[lo:lo+32][ix],device='cuda',dtype=torch.float64);board=torch.tensor(initial[lo:lo+32][ix],device='cuda',dtype=torch.float64);cache=C.BinaryQuadratic(models[job['seed']],t,a);raw=B.solve(cache,board,watch,selector="lexical" if gpu==0 else "energy");torch.cuda.synchronize();search_seconds+=time.perf_counter()-before
                    before=time.perf_counter()
                    for key in ['prediction','stopped','finite','block_moves','native_sweeps','native_cap','block_cap']:arrays[key][ix]=raw[key].cpu().numpy().astype(bool) if key=='prediction' else raw[key]
                    arrays['minimum_flip_margin'][ix]=models[job['seed']].flip_delta(t,raw['prediction'],a).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy();arrays['energy'][ix]=models[job['seed']].energy(t,raw['prediction'],a).cpu().numpy();diagnostic_seconds+=time.perf_counter()-before
                    for k in timings:timings[k]+=raw[k]
                before=time.perf_counter();path=folder/f'batch{lo//32:02d}.npz';payload={'protocol_sha256':S.sha(PP),'offset':lo,'trace':raw['trace'] if raw else []};Z.save(path,**arrays,intervention_indices=ix,states_before=raw['states_before'] if raw else np.zeros((0,0,180,180),bool),trace_utf8=np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8));batches.append({'offset':lo,'path':path.name,'sha256':S.sha(path)});archive_seconds+=time.perf_counter()-before
                for k in saved:saved[k].append(arrays[k])
                state.update(current_job=job,completed_inputs=lo+32,wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state);watch()
        arrays={k:np.concatenate(v) for k,v in saved.items()};arrays['iterations']=prefix_iterations+arrays['native_sweeps'];before=time.perf_counter();Z.save(folder/'outputs.npz',**arrays);archive_seconds+=time.perf_counter()-before;increment=selector_seconds+search_seconds;r={'status':'complete','job':job,'policy':policy,'execution_gpu':gpu,'protocol_sha256':S.sha(PP),'source_outputs_sha256':source['sha256'],'outputs_sha256':S.sha(folder/'outputs.npz'),'batches':batches,'initial_simple_paths_skipped':int(simple.sum()),'new_search_inputs':int(eligible.sum()),'archived_answer_prefix_seconds':source['archived_answer_prefix_seconds'],'archived_shared_hidden_seconds':source['archived_shared_hidden_seconds'],'new_incremental_answer_seconds':increment,'charged_answer_seconds':source['archived_answer_prefix_seconds']+increment,'selector_seconds':selector_seconds,'search_seconds':search_seconds,'diagnostic_seconds':diagnostic_seconds,'archive_seconds':archive_seconds,**timings,'seconds':time.perf_counter()-before_job};S.save(folder/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('large_cycle_cut_job_complete',job=job['id'],gpu=gpu,new_search_inputs=int(eligible.sum()),elapsed=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args()
    if v.command=='evaluate':evaluate(v.gpu)
    else:freeze()
