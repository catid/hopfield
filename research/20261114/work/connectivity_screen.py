"""H2ak post-hoc archived skip-path control and connectivity-veto intervention."""
import argparse
import hashlib
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import connected_block_moves as B
import fast_npz as Z

PP=S.HERE/'connectivity-screen-protocol.json'
DEST=S.OUT/'connectivity_screen'
PARENT=S.OUT/'factorial_transfer'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    old=json.loads((S.HERE/'factorial-transfer-protocol.json').read_text());a=json.loads((PARENT/'audit.json').read_text());assert a['status']=='validated';orig=(S.HERE/'graph_block_moves.py').read_text().split('def scores(',1)[1];current=(S.HERE/'connected_block_moves.py').read_text().split('def scores(',1)[1];assert orig==current
    p={'id':'seen-cohort-skip-path-and-predeletion-connectivity-screen','hypothesis_id':'H2ak','frozen_utc':S.utc(),'status':'posthoc_rules_frozen_before_control_mapping_gate_and_intervention','parent_protocol_sha256':S.sha(S.HERE/'factorial-transfer-protocol.json'),'parent_audit_sha256':S.sha(PARENT/'audit.json'),'models':{s:x['old_binary'] for s,x in old['models'].items()},'jobs':old['jobs'],'datasets':json.loads((PARENT/'datasets.json').read_text())['datasets'],'parent_results':{r['job']['id']:{'outputs_sha256':r['outputs_sha256'],'checkpoint_sha256':r['checkpoint_sha256_by_arm'][1]} for r in a['records']},'policies':{'skip_initial_simple_path':'If initialbinary selected cells form one simple start-to-goalpath on suppliedgraph, select exact archived binary output andflags; otherwise select exact archived old-block output andflags. No shortestdistance orlabel oracle. Zero new searches. All disconnected/capped/nonfinitecases retained.','connectivity_veto':'Same oldLP binaryprefix, same candidate degree1/2components,size2..256. Before scoring/application, reject a deletion iff startcannotreachgoal in remaining selected4neighborgraph. Same energyminimum/lexicographicties,threshold-1e-8,max16rounds,native512sweepdescent. Native descent is unchanged and may disconnect; record anysuchtransition. This is explicitly a stronger symbolic reachability-filtereddecoder.'},'targets':{'zero_joint_regressions':'For eachpolicy separately,0 jointsuccess→failure inputs versus archived oldbinary. Skip-path identity on anycorrectsimplepath is mathematical, not statistical evidence.','retain_prior_gains':'For eachpolicy separately retain at least90percent of the63 jointgaininputs originally observed foroldblocks (integerthreshold57/63). Descriptive reuseddata measurement, not freshconfidence.'},'scope':'All4096alreadyseenH2ajinputs×3fixedoldLPmodels. Posthocdiagnostic, nofreshgeneralizationclaim. No labels/shortestdistance for selection; labels only independentoutcomeaudit.','budget':{'maximum_new_trajectory_gpu_minutes':10,'per_worker_wall_seconds':300,'source_gate_wall_seconds':120,'watchdog':'Perbatch/blockround; preserve completedbatcharchives, noextension. No baseline reruns.'},'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['connectivity_screen.py','connected_block_moves.py','gate_connectivity_screen.py','graph_block_moves.py','binary_coordinate.py','fast_npz.py','study.py']},'unchanged_score_and_solve_suffix_sha256':hashlib.sha256(orig.encode()).hexdigest(),'accounting':{'fresh_inputs':0,'new_block_interventions':12288,'reused_binary_prefixes':12288,'archived_skip_control_mappings':12288,'additional_baseline_trajectories':0}}
    S.save(PP,p);S.emit('connectivity_screen_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert p['parent_protocol_sha256']==S.sha(S.HERE/'factorial-transfer-protocol.json') and p['parent_audit_sha256']==S.sha(PARENT/'audit.json')
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h,path
    for m in p['models'].values():assert S.sha(S.ROOT/m['path'])==m['sha256']
    return p


def skip_control():
    p=protocol();started=time.perf_counter();records=[]
    for job in p['jobs']:
        folder=DEST/'skip'/job['id'];folder.mkdir(parents=True,exist_ok=True);source=PARENT/'runs'/job['id']/'outputs.npz';assert S.sha(source)==p['parent_results'][job['id']]['outputs_sha256'];ds=p['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as z:t=z['tokens']
        with np.load(source) as z:pred=z['predictions'];before=time.perf_counter();simple=np.array([B.simple_path(tok,b) for tok,b in zip(t,pred[1])]);selection_seconds=time.perf_counter()-before;selected=np.where(simple,1,3);arrays={k:z[k][selected,np.arange(256)] for k in ['predictions','iterations','stopped','finite','minimum_flip_margin64','energy64']}
        arrays.update(selected_parent_arm=selected,initial_simple_path=simple);path=folder/'outputs.npz';Z.save(path,**arrays);record={'job':job,'outputs_sha256':S.sha(path),'source_outputs_sha256':S.sha(source),'selected_binary':int(simple.sum()),'selected_archived_blocks':int((~simple).sum()),'selector_seconds':selection_seconds,'new_searches':0};S.save(folder/'result.json',record);records.append(record)
    S.save(DEST/'skip-control.json',{'status':'complete','protocol_sha256':S.sha(PP),'records':records,'seconds':time.perf_counter()-started,'new_searches':0,'scope':'Exact archived output/flag mapping; selector timing only, not deploytime blockruntime estimate.'});S.emit('skip_control_mapped',model_inputs=12288,seconds=time.perf_counter()-started)


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();gate=json.loads((DEST/'numerical-gate.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==S.sha(PP);started=time.perf_counter();models={};state={'status':'running','gpu':gpu,'pid':os.getpid(),'inventory':inventory,'started_utc':S.utc(),'jobs_complete':[]}
    def watch():
        if time.perf_counter()-started>300:raise TimeoutError('Frozen300sworker budget')
    for seed in range(3):
        m=S.model_for(seed,120,'original','cuda',torch.float64);m.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True));models[seed]=m
    for job in p['jobs']:
        if job['gpu']!=gpu:continue
        folder=DEST/'veto'/job['id'];folder.mkdir(parents=True,exist_ok=True);source=PARENT/'runs'/job['id']/'outputs.npz';assert S.sha(source)==p['parent_results'][job['id']]['outputs_sha256'];ds=p['datasets'][job['shard']]
        assert S.sha(S.ROOT/ds['path'])==ds['sha256']
        with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
        with np.load(source) as z:initial=z['predictions'][1];readout=z['couplings'];prefix_good=z['stopped'][1]&z['finite'][1];initial_finite=z['finite'][1];prefix_iterations=z['iterations'][1]
        saved={k:[] for k in ['prediction','stopped','finite','block_moves','native_sweeps','native_cap','block_cap','minimum_flip_margin','energy']};batches=[];timings={k:0. for k in ['cpu_graph_seconds','score_seconds','descent_seconds']};before_job=time.perf_counter()
        with torch.no_grad():
            for lo in range(0,256,64):
                watch();t=torch.tensor(tokens[lo:lo+64].astype(np.int64),device='cuda');aa=torch.tensor(readout[lo:lo+64],device='cuda',dtype=torch.float64);board=torch.tensor(initial[lo:lo+64],device='cuda',dtype=torch.float64);ix=np.flatnonzero(prefix_good[lo:lo+64]);arrays={'prediction':initial[lo:lo+64].copy(),'stopped':np.zeros(64,bool),'finite':initial_finite[lo:lo+64].copy(),'block_moves':np.zeros(64,np.int64),'native_sweeps':np.zeros(64,np.int64),'native_cap':~prefix_good[lo:lo+64]&initial_finite[lo:lo+64],'block_cap':np.zeros(64,bool)};raw=None
                if len(ix):
                    ids=torch.tensor(ix,device='cuda');cache=C.BinaryQuadratic(models[job['seed']],t[ids],aa[ids]);raw=B.solve(cache,board[ids],watch)
                    for key in arrays:arrays[key][ix]=raw[key].cpu().numpy().astype(bool) if key=='prediction' else raw[key]
                    for k in timings:timings[k]+=raw[k]
                final=torch.tensor(arrays['prediction'],device='cuda',dtype=torch.float64);arrays['minimum_flip_margin']=models[job['seed']].flip_delta(t,final,aa).masked_fill(t!=1,float('inf')).flatten(1).amin(1).cpu().numpy();arrays['energy']=models[job['seed']].energy(t,final,aa).cpu().numpy();payload={'protocol_sha256':S.sha(PP),'offset':lo,'trace':raw['trace'] if raw else []};archive=folder/f'batch{lo//64:02d}.npz';Z.save(archive,**arrays,valid_prefix_indices=ix,states_before=raw['states_before'] if raw else np.zeros((0,0,120,120),bool),trace_utf8=np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),dtype=np.uint8));batches.append({'offset':lo,'path':archive.name,'sha256':S.sha(archive)})
                for k in saved:saved[k].append(arrays[k])
                state.update(current_job=job,completed_inputs=lo+64,wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state);watch()
        arrays={k:np.concatenate(v) for k,v in saved.items()};arrays['iterations']=prefix_iterations+arrays['native_sweeps'];Z.save(folder/'outputs.npz',**arrays);r={'status':'complete','job':job,'protocol_sha256':S.sha(PP),'source_outputs_sha256':S.sha(source),'outputs_sha256':S.sha(folder/'outputs.npz'),'batches':batches,'seconds':time.perf_counter()-before_job,**timings};S.save(folder/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('connectivity_veto_job_complete',job=job['id'],gpu=gpu,elapsed=time.perf_counter()-started)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','skip_control','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args()
    if v.command=='evaluate':evaluate(v.gpu)
    else:globals()[v.command]()
