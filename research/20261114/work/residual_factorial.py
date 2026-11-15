"""H2bb disjoint source cap-family continuations; joint arm is exact composition."""
import argparse
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import binary_coordinate as C
import terminal_cycle_moves as T
import fast_npz as Z
PP=S.HERE/'residual-factorial-protocol.json';DEST=S.OUT/'residual_factorial';PARENT=S.OUT/'terminal_cycle';ORIGIN=S.OUT/'density_transfer'


def freeze():
    if PP.exists():raise FileExistsError(PP)
    p0=json.loads((PARENT/'protocol.json').read_text());audit=json.loads((PARENT/'audit.json').read_text());gp=S.OUT/'residual_factorial_preflight/numerical-gate.json';gate=json.loads(gp.read_text());assert audit['status']=='passed' and audit['joint_success']['terminal_then_extra16']==4086;assert gate['status']=='passed' and gate['source_caps_by_family']=={'veto':10,'lexical':15} and gate['families_disjoint_by_input'] and gate['script_sha256']==S.sha(S.HERE/'preflight_residual_factorial.py') and gate['terminal_helper_sha256']==S.sha(S.HERE/'terminal_cycle_moves.py');sources={};origins={}
    for job in p0['jobs']:
        run=PARENT/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==r['outputs_sha256'];sources[job['id']]={'path':str((run/'outputs.npz').relative_to(S.ROOT)),'sha256':r['outputs_sha256'],'archived_answer_prefix_seconds':r['charged_answer_seconds'][1],'archived_shared_hidden_seconds':r['archived_shared_hidden_seconds']};origins[job['id']]=p0['source_outputs'][job['id']]
    names=['residual_factorial.py','preflight_residual_factorial.py','terminal_cycle_moves.py','lexical_cycle_cut_moves.py','connected_block_moves.py','graph_block_moves.py','binary_coordinate.py','study.py','fast_npz.py'];p={'id':'density180-disjoint-veto-and-lexical-cap-rescue','hypothesis_id':'H2bb','status':'frozen_after_complete_native_geometry_cost_preflight','frozen_utc':S.utc(),'arms':['inherited_H2ba','lexical_family_only','veto_family_only','both_families_composed'],'new_policies':{'0':'lexical_family','1':'veto_family'},'jobs':[{k:v for k,v in j.items() if k!='gpu'} for j in p0['jobs']],'models':p0['models'],'datasets':p0['datasets'],'source_outputs':sources,'origin_outputs':origins,'parent_protocol_sha256':S.sha(PARENT/'protocol.json'),'parent_audit_sha256':S.sha(PARENT/'audit.json'),'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names},'gate':{'path':str(gp.relative_to(S.ROOT)),'sha256':S.sha(gp)},'source_families':gate['joint_input_ids_by_family'],'policy':'All4096 previously seen inputs/all3models retained. Start from identical H2ba extra16 final state. Family determined solely by original H2ay veto-cap lineage. Finite current block-cap states with no nativecap and margin>=-1e-8 eligible. Lexical family15states/5inputs; inherited-veto family10states/5disjointinputs. One knownterminal veto state certified descriptively; all24nonterminal cases get atmost16unchangedlexical256 rounds and final terminalcheck. No addedveto moves, no512sizechange, no labels/distance used. Native512sweeps,16rounds unchanged percontinuation.','primary':{'lexical_family':{'minimum_joint_rescues':3,'family_failure_inputs':5,'maximum_joint_regressions':0},'veto_family':{'minimum_joint_rescues':3,'family_failure_inputs':5,'maximum_joint_regressions':0},'both_families_composed':{'minimum_joint_rescues':6,'family_failure_inputs':10,'maximum_joint_regressions':0},'decision':'Each rescue target and zero-regression requirement reported separately on finite source. Combined is exact union of disjoint interventions, not interaction evidence or extra independent trial. No fresh generalization/confidence claim.'},'budget':{'per_worker_wall_seconds':300,'aggregate_new_trajectory_gpu_minutes':10,'preflight_wall_seconds':120,'independent_audit_wall_seconds':300,'independent_audit_processes':16},'charging':'Eachfamily fullH2ba answerprefix plus own selector/cache/search/finalcheck. Joint arm derived perstate from disjoint branch outputs, fullsourceprefix charged once plus both incremental costs. No duplicateinference/hidden/baseline rerun. EachGPU executesall48jobs for onefamily.','accounting':{'fresh_inputs':0,'source_model_inputs':12288,'eligible_model_inputs':25,'known_terminal_model_inputs':1,'new_search_model_inputs':24,'joint_arm_new_trajectories':0,'new_optimizer_updates':0}};S.save(PP,p);DEST.mkdir(exist_ok=True);S.save(DEST/'protocol.json',p);gate.update(protocol_sha256=S.sha(PP));S.save(DEST/'numerical-gate.json',gate);S.emit('residual_factorial_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert S.sha(PARENT/'protocol.json')==p['parent_protocol_sha256'] and S.sha(PARENT/'audit.json')==p['parent_audit_sha256']
    for path,h in p['source_sha256'].items():assert S.sha(S.ROOT/path)==h
    assert S.sha(S.ROOT/p['gate']['path'])==p['gate']['sha256'];return p


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();policy=p['new_policies'][str(gpu)];started=time.perf_counter();models={};state={'status':'running','gpu':gpu,'policy':policy,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'jobs_complete':[]}
    def watch():
        if time.perf_counter()-started>300:raise TimeoutError('Frozen300s worker budget')
    for seed in range(3):
        model=S.model_for(seed,180,'original','cuda',torch.float64);assert S.sha(S.ROOT/p['models'][str(seed)]['path'])==p['models'][str(seed)]['sha256'];model.load_state_dict(torch.load(S.ROOT/p['models'][str(seed)]['path'],map_location='cuda',weights_only=True));models[seed]=model
    for job in p['jobs']:
        run=DEST/policy/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True);source=p['source_outputs'][job['id']];origin=p['origin_outputs'][job['id']];ds=p['datasets'][job['shard']]
        for ref in [source,origin,ds]:assert S.sha(S.ROOT/ref['path'])==ref['sha256']
        with np.load(S.ROOT/source['path']) as z:old={k:z[k][1] for k in ['predictions','iterations','stopped','finite','block_cap','native_cap','minimum_flip_margin64','energy64']}
        with np.load(S.ROOT/origin['path']) as z:readout=z['couplings'];veto=z['block_cap'][0]
        with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
        eligible=old['block_cap']&old['finite']&~old['native_cap']&(old['minimum_flip_margin64']>=-1e-8)&(veto if gpu==1 else ~veto);saved={k:[] for k in ['predictions','iterations','stopped','finite','block_cap','native_cap','minimum_flip_margin64','energy64','new_block_moves','new_native_sweeps','terminal_certified','eligible','initial_terminal']};batches=[];selector_seconds=search_seconds=diagnostic_seconds=archive_seconds=0.;search_count=terminal_count=0;job_started=time.perf_counter()
        with torch.no_grad():
            for lo in range(0,256,32):
                watch();tok=tokens[lo:lo+32];initial=old['predictions'][lo:lo+32];before=time.perf_counter();term,initial_records=T.terminal_checks(tok,initial,eligible[lo:lo+32]);selector_seconds+=time.perf_counter()-before;ix=np.flatnonzero(eligible[lo:lo+32]&~term);search_count+=len(ix);terminal_count+=int(term.sum());arrays={k:old[k][lo:lo+32].copy() for k in old};arrays.update(new_block_moves=np.zeros(32,np.int64),new_native_sweeps=np.zeros(32,np.int64),terminal_certified=term.copy(),eligible=eligible[lo:lo+32],initial_terminal=term);arrays['stopped'][term]=True;arrays['block_cap'][term]=False;raw=None
                if len(ix):
                    torch.cuda.synchronize();before=time.perf_counter();tt=torch.tensor(tok[ix].astype(np.int64),device='cuda');aa=torch.tensor(readout[lo:lo+32][ix],device='cuda',dtype=torch.float64);bb=torch.tensor(initial[ix],device='cuda',dtype=torch.float64);raw=T.solve(C.BinaryQuadratic(models[job['seed']],tt,aa),bb,watch);torch.cuda.synchronize();search_seconds+=time.perf_counter()-before;arrays['predictions'][ix]=raw['prediction'].cpu().numpy().astype(bool)
                    for k in ['stopped','finite','block_cap','native_cap','terminal_certified']:arrays[k][ix]=raw[k]
                    arrays['new_block_moves'][ix]=raw['block_moves'];arrays['new_native_sweeps'][ix]=raw['native_sweeps'];arrays['iterations'][ix]+=raw['native_sweeps'];before=time.perf_counter();arrays['minimum_flip_margin64'][ix]=models[job['seed']].flip_delta(tt,raw['prediction'],aa).masked_fill(tt!=1,float('inf')).flatten(1).amin(1).cpu().numpy();arrays['energy64'][ix]=models[job['seed']].energy(tt,raw['prediction'],aa).cpu().numpy();diagnostic_seconds+=time.perf_counter()-before
                before=time.perf_counter();path=run/f'batch{lo//32:02d}.npz';payload={'protocol_sha256':S.sha(PP),'offset':lo,'initial_terminal_checks':initial_records,'trace':raw['trace'] if raw else [],'final_terminal_checks':raw['terminal_check_records'] if raw else []};Z.save(path,**arrays,search_indices=ix,states_before=raw['states_before'] if raw else np.zeros((0,0,180,180),bool),pre_terminal_stopped=raw['pre_terminal_stopped'] if raw else np.zeros(0,bool),pre_terminal_block_cap=raw['pre_terminal_block_cap'] if raw else np.zeros(0,bool),trace_utf8=np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),np.uint8));batches.append({'offset':lo,'path':path.name,'sha256':S.sha(path)});archive_seconds+=time.perf_counter()-before
                for k in saved:saved[k].append(arrays[k])
                state.update(current_job=job,completed_inputs=lo+32,worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state);watch()
        out={k:np.concatenate(v) for k,v in saved.items()};before=time.perf_counter();Z.save(run/'outputs.npz',**out);archive_seconds+=time.perf_counter()-before;r={'status':'complete','job':job,'policy':policy,'protocol_sha256':S.sha(PP),'source_outputs_sha256':source['sha256'],'origin_outputs_sha256':origin['sha256'],'outputs_sha256':S.sha(run/'outputs.npz'),'batches':batches,'eligible_caps':int(eligible.sum()),'terminal_only_cases':terminal_count,'new_search_inputs':search_count,'archived_answer_prefix_seconds':source['archived_answer_prefix_seconds'],'archived_shared_hidden_seconds':source['archived_shared_hidden_seconds'],'selector_seconds':selector_seconds,'search_seconds':search_seconds,'diagnostic_seconds':diagnostic_seconds,'archive_seconds':archive_seconds,'charged_answer_seconds':source['archived_answer_prefix_seconds']+selector_seconds+search_seconds,'seconds':time.perf_counter()-job_started};S.save(run/'result.json',r);state['jobs_complete'].append(job['id']);S.emit('residual_factorial_job_complete',job=job['id'],gpu=gpu,elapsed=time.perf_counter()-started,new_search_inputs=search_count)
    state.update(status='complete',wall_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','evaluate']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args();evaluate(v.gpu) if v.command=='evaluate' else freeze()
