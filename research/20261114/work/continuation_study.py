"""H2bg: fixed archived-cap continuation; no new independent inputs."""
import argparse
import copy
import json
import os
import time
import numpy as np
import study as S
import selective_fallback as F
import fast_npz as Z
import binary_coordinate as C
import continuation_moves as M

SOURCE=S.OUT/'loop80_transfer'
DEST=S.OUT/'loop80_continuation'
PP=S.HERE/'continuation-protocol.json'
ARMS=['total64_limit256','total96_limit256','total96_limit512']


def protocol():
    p=json.loads(PP.read_text())
    for path,h in {**p['source_sha256'],**p['source_evidence_sha256']}.items():assert S.sha(S.ROOT/path)==h,path
    return p


def prepare():
    p=protocol();started=time.perf_counter();source=json.loads((SOURCE/'audit.json').read_text());byjob={r['job']['id']:r for r in source['rows']};ds=json.loads((SOURCE/'datasets.json').read_text());rows=[];arrays={s:[] for s in range(3)}
    for shard in range(16):
        d=ds['datasets'][shard];assert S.sha(S.ROOT/d['path'])==d['sha256']
        with np.load(S.ROOT/d['path']) as z:tokens=z['tokens'];distance=z['shortest_edges']
        for seed in range(3):
            jid=f'shard{shard:02d}-s{seed}';r=json.loads((SOURCE/'runs'/jid/'result.json').read_text());path=SOURCE/'runs'/jid/'outputs.npz';assert S.sha(path)==r['outputs_sha256']
            with np.load(path) as z:
                cap=z['block_cap'][3];assert np.array_equal(cap,~np.array(byjob[jid]['success'][5],bool));ix=np.flatnonzero(cap)
                assert z['finite'][5,ix].all() and not z['block_native_cap'][3,ix].any() and (z['minimum_flip_margin64'][5,ix]>=-1e-8).all() and (z['block_moves'][1:,ix].sum(0)==48).all()
                for i in ix:arrays[seed].append({'tokens':tokens[i],'distance':distance[i],'index':shard*256+i,'board':z['predictions'][5,i],'couplings':z['couplings'][i],'iterations':z['iterations'][5,i],'hidden_good':z['hidden_good'][i]})
            rows.append({'job':jid,'source_outputs_sha256':r['outputs_sha256'],'source_receipt_sha256':S.sha(SOURCE/'runs'/jid/'result.json'),'indices':ix.tolist(),'source_prefix_answer_seconds':r['answer_seconds_by_arm'][5]})
    jobs=[]
    for seed in range(3):
        for gpu,part in enumerate(np.array_split(np.arange(len(arrays[seed])),2)):
            job={'id':f's{seed}-part{gpu}','seed':seed,'gpu':gpu,'n':len(part)};values=[arrays[seed][i] for i in part];path=DEST/'inputs'/f'{job["id"]}.npz';Z.save(path,**{key:np.stack([v[key] for v in values]) for key in values[0]});job.update(path=str(path.relative_to(S.ROOT)),sha256=S.sha(path));jobs.append(job)
    assert sum(j['n'] for j in jobs)==148
    S.save(DEST/'inputs.json',{'status':'passed','protocol_sha256':S.sha(PP),'jobs':jobs,'source_rows':rows,'source_all_inputs':4096,'source_all_model_inputs':12288,'eligible_caps':148,'seconds':time.perf_counter()-started,'new_trajectories':0})


def save_raw(run,name,raw,indices):
    path=run/f'{name}-trace.npz';payload={'trace':raw['trace'],'terminal_checks':raw['terminal_check_records']};Z.save(path,indices=indices,states_before=raw['states_before'],pre_terminal_stopped=raw['pre_terminal_stopped'],pre_terminal_block_cap=raw['pre_terminal_block_cap'],trace_utf8=np.frombuffer(json.dumps(payload,separators=(',',':')).encode(),np.uint8));return {'stage':name,'path':path.name,'sha256':S.sha(path)}


def evaluate(gpu):
    lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();inp=json.loads((DEST/'inputs.json').read_text());assert inp['protocol_sha256']==S.sha(PP)
    gates=[json.loads((DEST/f'source-gate-gpu{g}.json').read_text()) for g in [0,1]];assert all(g['status']=='passed' and g['protocol_sha256']==S.sha(PP) for g in gates)
    started=time.perf_counter();state={'status':'running','gpu':gpu,'inventory':inventory,'pid':os.getpid(),'started_utc':S.utc(),'completed_jobs':[]};S.save(DEST/f'worker{gpu}.json',state)
    def watch():
        state.update(elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'worker{gpu}.json',state)
        if state['elapsed_seconds']>120:state['status']='budget_exhausted';S.save(DEST/f'worker{gpu}.json',state);raise TimeoutError('Frozen120s worker budget; no extension')
    with torch.no_grad():
        for job in inp['jobs']:
            if job['gpu']!=gpu:continue
            watch();state['current_job']=job['id'];run=DEST/'runs'/job['id'];run.mkdir(parents=True,exist_ok=True);assert not (run/'result.json').exists();assert S.sha(S.ROOT/job['path'])==job['sha256'];before=time.perf_counter()
            model=S.model_for(job['seed'],180,'original','cuda');ref=p['models'][str(job['seed'])];assert S.sha(S.ROOT/ref['path'])==ref['sha256'];model.load_state_dict(torch.load(S.ROOT/ref['path'],map_location='cuda',weights_only=True));model=model.double();load_seconds=time.perf_counter()-before
            with np.load(S.ROOT/job['path']) as z:data={k:z[k] for k in z.files}
            torch.cuda.synchronize();before=time.perf_counter();t=torch.tensor(data['tokens'].astype(np.int64),device='cuda');a=torch.tensor(data['couplings'],device='cuda').double();initial=torch.tensor(data['board'],device='cuda').double();cache=C.BinaryQuadratic(model,t,a);torch.cuda.synchronize();setup_seconds=time.perf_counter()-before;traces=[];times=[];outputs=[];routing=np.zeros(3);n=len(t)
            def stage(name,start,indices,rounds,limit):
                torch.cuda.synchronize();before=time.perf_counter();sub=cache if len(indices)==n else C.BinaryQuadratic(model,t[indices],a[indices]);raw=M.certify_final(data['tokens'][indices],M.solve(sub,start,watch,max_rounds=rounds,max_cells=limit),limit);torch.cuda.synchronize();seconds=time.perf_counter()-before;traces.append(save_raw(run,name,raw,indices));times.append({'stage':name,'seconds':seconds,'geometry_seconds':raw['cpu_graph_seconds'],'score_seconds':raw['score_seconds'],'native_seconds':raw['descent_seconds'],'inputs':len(indices)});return raw
            first=stage('extra16_limit256',initial,np.arange(n),16,256)
            def result(raw,iterations):return {'prediction':raw['prediction'].cpu().numpy().astype(bool),'stopped':raw['stopped'].copy(),'finite':raw['finite'].copy(),'native_cap':raw['native_cap'].copy(),'block_cap':raw['block_cap'].copy(),'extra_moves':raw['block_moves'].copy(),'extra_sweeps':raw['native_sweeps'].copy(),'iterations':iterations+raw['native_sweeps'],'margin':raw['minimum_flip_margin'].copy()}
            before=time.perf_counter();outputs.append(result(first,data['iterations']));second=copy.deepcopy(outputs[0]);eligible=first['block_cap']&first['finite']&~first['native_cap']&(first['minimum_flip_margin']>=-1e-8);ix=np.flatnonzero(eligible);routing[:2]+=time.perf_counter()-before
            if len(ix):
                raw=stage('extra32_limit256',first['prediction'][ix],ix,32,256);before=time.perf_counter();r=result(raw,outputs[0]['iterations'][ix])
                for key in second:second[key][ix]=r[key]
                second['extra_moves'][ix]+=outputs[0]['extra_moves'][ix];second['extra_sweeps'][ix]+=outputs[0]['extra_sweeps'][ix];routing[1]+=time.perf_counter()-before
            outputs.append(second);large=stage('extra48_limit512',initial,np.arange(n),48,512);before=time.perf_counter();outputs.append(result(large,data['iterations']));routing[2]+=time.perf_counter()-before;saved={key:np.stack([r[key] for r in outputs]) for key in outputs[0]};saved['energy']=np.stack([model.energy(t,torch.tensor(r['prediction'],device='cuda').double(),a).cpu().numpy() for r in outputs]);Z.save(run/'outputs.npz',**saved)
            receipt={'status':'complete','job':job,'protocol_sha256':S.sha(PP),'input_sha256':job['sha256'],'outputs_sha256':S.sha(run/'outputs.npz'),'trace_archives':traces,'stage_timings':times,'model_load_seconds':load_seconds,'setup_seconds':setup_seconds,'routing_seconds':routing.tolist(),'charging':'Full H2be source answer prefix for every source row plus all measured eligibility/setup/stage work. Stages16+32 shared for256;512 independently starts at48. Source prefix was archived; this is not a fresh full-pipeline runtime measurement.'};S.save(run/'result.json',receipt);state['completed_jobs'].append(job['id']);watch();S.emit('continuation_job_complete',job=job['id'],gpu=gpu,elapsed=time.perf_counter()-started)
    state.update(status='complete',seconds=time.perf_counter()-started);S.save(DEST/f'worker{gpu}.json',state)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','evaluate']);ap.add_argument('--gpu',type=int,choices=[0,1]);args=ap.parse_args();evaluate(args.gpu) if args.command=='evaluate' else prepare()
