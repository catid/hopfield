"""H2bd freeze and separate bounded geometry/full-policy speed endpoints."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import numpy as np
import study as S
import fast_npz as Z

PP=S.HERE/'geometry-speed-protocol.json';DEST=S.OUT/'geometry_speed';GATE=S.OUT/'geometry_speed_preflight';SOURCE=S.OUT/'unified_cycle';ORDER=[['legacy','fast'],['fast','legacy'],['fast','legacy'],['legacy','fast']]


def freeze():
    if PP.exists():raise FileExistsError(PP)
    names=['synthetic-gate.json','archive-gate.json','policy-gate-gpu0.json','policy-gate-gpu1.json'];gates={}
    for name in names:
        path=GATE/name;r=json.loads(path.read_text());assert r['status']=='passed'
        for file,h in r['source_sha256'].items():assert S.sha(S.ROOT/file)==h,file
        gates[str(path.relative_to(S.ROOT))]=S.sha(path)
    assert sum(sum(r['inputs'] for r in json.loads((GATE/f'policy-gate-gpu{gpu}.json').read_text())['rows']) for gpu in [0,1])==768
    fixture=GATE/'geometry-fixtures/manifest.json';f=json.loads(fixture.read_text());assert f['n']==1024
    for r in f['fixtures']:assert S.sha(fixture.parent/r['file'])==r['sha256']
    assert 0 in os.sched_getaffinity(0);hardware={'host':platform.node(),'platform':platform.platform(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'cpu_model':[s.strip() for s in open('/proc/cpuinfo') if s.startswith('model name')][0],'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True),'process_snapshot':subprocess.check_output(['ps','-e','-o','pid,psr,pcpu,comm','--sort=-pcpu'],text=True).splitlines()[:25]};before_cpu=[int(v) for line in open('/proc/stat') if line.startswith('cpu0 ') for v in line.split()[1:]];time.sleep(1);after_cpu=[int(v) for line in open('/proc/stat') if line.startswith('cpu0 ') for v in line.split()[1:]];ticks=[b-a for a,b in zip(before_cpu,after_cpu)];hardware['cpu0_one_second_busy_fraction']=1-(ticks[3]+ticks[4])/sum(ticks);reference_sources=json.loads((SOURCE/'protocol.json').read_text())['source_sha256'];assert all(S.sha(S.ROOT/k)==h for k,h in reference_sources.items());proposal=json.loads((S.HERE/'geometry-speed-proposal.json').read_text());sources=['geometry_speed_study.py','geometry_policy_batch.py','bridge_cycle_geometry.py','bridge_lexical_moves.py','bridge_terminal_moves.py','preflight_bridge_geometry.py','preflight_bridge_archives.py','preflight_bridge_policy.py'];p={'hypothesis_id':'H2bd','status':'frozen_after_complete_parity_gates_before_speed_measurements','frozen_utc':S.utc(),'proposal':proposal,'proposal_sha256':S.sha(S.HERE/'geometry-speed-proposal.json'),'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in sources},'reference_source_sha256':reference_sources,'frozen_reference_protocol_sha256':S.sha(SOURCE/'protocol.json'),'gate_files':gates,'geometry_fixture_manifest':{'path':str(fixture.relative_to(S.ROOT)),'sha256':S.sha(fixture)},'full_policy_cohort':json.loads((GATE/'policy-gate-gpu0.json').read_text())['dataset'],'hardware':hardware,'paired_order':ORDER,'science_accounting':{'fresh_inputs':0,'new_optimizer_updates':0,'timing_replays_are_not_new_scientific_evaluations':True},'decision_rule':'Geometry and full-policy targets decided separately from medians of four paired ratios. Exact parity prerequisite. A measured median below target falsifies that finite benchmark target; no statistical timing CI.'};DEST.mkdir(exist_ok=True);S.save(PP,p);S.save(DEST/'protocol.json',p);S.emit('geometry_speed_frozen',sha256=S.sha(PP))


def protocol():
    p=json.loads(PP.read_text());assert S.sha(SOURCE/'protocol.json')==p['frozen_reference_protocol_sha256']
    for path,h in {**p['source_sha256'],**p['reference_source_sha256'],**p['gate_files']}.items():assert S.sha(S.ROOT/path)==h,path
    return p


def geometry():
    import lexical_cycle_cut_moves as OLD
    import bridge_cycle_geometry as NEW
    p=protocol();started=time.perf_counter();os.sched_setaffinity(0,{0});assert os.sched_getaffinity(0)=={0};ref=p['geometry_fixture_manifest'];manifest=S.ROOT/ref['path'];assert S.sha(manifest)==ref['sha256'];states=[]
    for row in json.loads(manifest.read_text())['fixtures']:
        path=manifest.parent/row['file'];assert S.sha(path)==row['sha256']
        with np.load(path) as z:states.extend(zip(z['tokens'],z['boards']))
    assert len(states)==1024;functions={'legacy':OLD.candidates,'fast':NEW.candidates};before=time.perf_counter()
    for name in ['legacy','fast']:
        for t,b in states[:32]:functions[name](t,b)
    warmup=time.perf_counter()-before;rows=[]
    for rep,order in enumerate(ORDER):
        outputs={};timing={}
        for name in order:
            process=time.process_time();wall=time.perf_counter();outputs[name]=[functions[name](t,b) for t,b in states];timing[name]={'process_seconds':time.process_time()-process,'wall_seconds':time.perf_counter()-wall};assert time.perf_counter()-started<120
        assert outputs['legacy']==outputs['fast'];row={'repetition':rep,'order':order,'timing':timing,'paired_process_speedup':timing['legacy']['process_seconds']/timing['fast']['process_seconds'],'all_1024_outputs_exact':True};rows.append(row);S.save(DEST/'geometry-partial.json',{'rows':rows,'warmup_seconds':warmup});S.emit('geometry_pair_complete',repetition=rep,ratio=row['paired_process_speedup'])
    median=float(np.median([r['paired_process_speedup'] for r in rows]));S.save(DEST/'geometry-benchmark.json',{'status':'passed','protocol_sha256':S.sha(PP),'rows':rows,'median_paired_process_speedup':median,'target':2.,'decision':'supported' if median>=2 else 'falsified','warmup_seconds':warmup,'cpu_affinity':[0],'seconds':time.perf_counter()-started,'fixture_manifest_sha256':ref['sha256'],'scope':'Four balanced paired complete geometry sweeps on1024 fixed states; purePython process timing with input loading/warmup separate. No full-policy inference speed claim.'})


def policy(gpu):
    import selective_fallback as F
    import geometry_policy_batch as Q
    from unified_cycle import load_models
    from factorial_transfer import hidden
    lock,inventory=F.acquire(gpu);p=protocol();started=time.perf_counter();torch,_,_=S.imports();source=json.loads((SOURCE/'protocol.json').read_text());ds=p['full_policy_cohort'];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
    models={};doubles={};expected={};traces={};before=time.perf_counter()
    for seed in range(3):
        models[seed],doubles[seed]=load_models(source,seed);run=SOURCE/'runs'/f'shard00-s{seed}';receipt=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==receipt['outputs_sha256']
        with np.load(run/'outputs.npz') as z:
            expected[seed]={k:z[k][5] for k in ['predictions','iterations','stopped','finite']};expected[seed].update(native_cap=z['block_native_cap'][3],block_cap=z['block_cap'][3])
        for item in receipt['trace_archives']:
            if item['arm']=='connectivity_veto' or item['arm'] in Q.ARMS[3:]:
                path=run/item['path'];assert S.sha(path)==item['sha256']
                with np.load(path) as z:traces[(seed,item['offset'],Q.ARMS.index(item['arm']))]={'payload':json.loads(z['trace_utf8'].tobytes()),'states':z['states_before'],'indices':z['valid_prefix_indices'],'initial_terminal':z['initial_terminal'],'pre_terminal_stopped':z['pre_terminal_stopped'],'pre_terminal_block_cap':z['pre_terminal_block_cap']}
    load_seconds=time.perf_counter()-before;state={'status':'running','gpu':gpu,'pid':os.getpid(),'inventory':inventory,'pairs_complete':[],'started_utc':S.utc()};S.save(DEST/f'policy-worker{gpu}.json',state)
    def watch():
        if time.perf_counter()-started>240:raise TimeoutError('Frozen240s full-policy worker budget')
    def run(fast,n=256):
        saved={k:[] for k in ['predictions','iterations','stopped','finite','native_cap','block_cap']};records=[];torch.cuda.synchronize();wall=time.perf_counter()
        for seed in range(3):
            arrays={k:[] for k in saved}
            for lo in range(0,n,32):
                t=torch.tensor(tokens[lo:lo+32].astype(np.int64),device='cuda');a,hs,good=hidden(models[seed][0],t);assert good.all();r=Q.solve(models[seed],doubles[seed],t,a,fast=fast,watchdog=watch)
                for key in arrays:
                    arrays[key].append(r['blocks'][5][key].copy() if key in ['native_cap','block_cap'] else r['outputs'][5]['prediction' if key=='predictions' else key].cpu().numpy())
                for v in range(2,6):
                    b=r['blocks'][v];raw=b['raw'];records.append({'seed':seed,'lo':lo,'arm':v,'initial_terminal_checks':b['initial_terminal_records'],'final_terminal_checks':raw.get('terminal_check_records',[]) if raw else [],'trace':raw['trace'] if raw else [],'states':raw['states_before'] if raw else np.zeros((0,0,180,180),bool),'indices':b['valid_prefix_indices'],'initial_terminal':b['initial_terminal'],'pre_terminal_stopped':raw.get('pre_terminal_stopped',np.zeros(0,bool)) if raw else np.zeros(0,bool),'pre_terminal_block_cap':raw.get('pre_terminal_block_cap',np.zeros(0,bool)) if raw else np.zeros(0,bool)})
                watch()
            for key in saved:saved[key].append(np.concatenate(arrays[key]))
        torch.cuda.synchronize();seconds=time.perf_counter()-wall;return {k:np.stack(v) for k,v in saved.items()},records,seconds
    def check(arrays,records,n):
        for seed in range(3):
            for key in arrays:assert np.array_equal(arrays[key][seed],expected[seed][key][:n]),(seed,key)
        hashes=[]
        for r in records:
            old=traces[(r['seed'],r['lo'],r['arm'])];assert np.array_equal(r['states'],old['states']) and np.array_equal(r['indices'],old['indices'])
            for key in ['trace','initial_terminal_checks','final_terminal_checks']:assert r[key]==old['payload'][key]
            for key in ['initial_terminal','pre_terminal_stopped','pre_terminal_block_cap']:assert np.array_equal(r[key],old[key])
            digest=hashlib.sha256(r['states'].tobytes()+r['indices'].tobytes()+json.dumps({k:r[k] for k in ['trace','initial_terminal_checks','final_terminal_checks']},sort_keys=True,separators=(',',':')).encode()).hexdigest();hashes.append({'seed':r['seed'],'offset':r['lo'],'arm':Q.ARMS[r['arm']],'sha256':digest})
        return hashes
    with torch.no_grad():
        warmup=[]
        for name in ['legacy','fast']:
            arrays,records,seconds=run(name=='fast',32);check(arrays,records,32);warmup.append({'implementation':name,'seconds':seconds})
        pairs=[]
        for rep in ([0,1] if gpu==0 else [2,3]):
            outputs={};seconds={};audit_seconds=archive_seconds=0.;receipts={}
            for name in ORDER[rep]:
                arrays,records,seconds[name]=run(name=='fast');before=time.perf_counter();hashes=check(arrays,records,256);audit_seconds+=time.perf_counter()-before;before=time.perf_counter();path=DEST/f'gpu{gpu}-rep{rep}-{name}.npz';Z.save(path,**arrays);receipts[name]={'outputs_file':path.name,'outputs_sha256':S.sha(path),'trace_hashes':hashes};archive_seconds+=time.perf_counter()-before;outputs[name]=arrays;watch()
            for key in outputs['legacy']:assert np.array_equal(outputs['legacy'][key],outputs['fast'][key])
            assert receipts['legacy']['trace_hashes']==receipts['fast']['trace_hashes'];row={'repetition':rep,'gpu':gpu,'order':ORDER[rep],'wall_seconds':seconds,'paired_wall_speedup':seconds['legacy']/seconds['fast'],'source_and_pair_exact':True,'audit_seconds':audit_seconds,'archive_seconds':archive_seconds,'receipts':receipts};pairs.append(row);state['pairs_complete'].append(rep);state.update(worker_elapsed_seconds=time.perf_counter()-started,updated_utc=S.utc());S.save(DEST/f'policy-worker{gpu}.json',state);S.save(DEST/f'policy-pairs-gpu{gpu}.json',{'protocol_sha256':S.sha(PP),'rows':pairs});S.emit('policy_pair_complete',gpu=gpu,repetition=rep,ratio=row['paired_wall_speedup']);watch()
    state.update(status='complete',wall_seconds=time.perf_counter()-started,load_seconds=load_seconds,warmup=warmup,updated_utc=S.utc());S.save(DEST/f'policy-worker{gpu}.json',state)


def collect():
    p=protocol();rows=[];workers=[]
    for gpu in [0,1]:
        r=json.loads((DEST/f'policy-pairs-gpu{gpu}.json').read_text());assert r['protocol_sha256']==S.sha(PP);rows.extend(r['rows']);w=json.loads((DEST/f'policy-worker{gpu}.json').read_text());assert w['status']=='complete' and w['wall_seconds']<=240;workers.append(w)
    rows.sort(key=lambda r:r['repetition']);assert [r['repetition'] for r in rows]==[0,1,2,3]
    for r in rows:
        for receipt in r['receipts'].values():assert S.sha(DEST/receipt['outputs_file'])==receipt['outputs_sha256']
    median=float(np.median([r['paired_wall_speedup'] for r in rows]));micro=json.loads((DEST/'geometry-benchmark.json').read_text());S.save(DEST/'audit.json',{'status':'passed','hypothesis':'H2bd','protocol_sha256':S.sha(PP),'geometry':micro,'policy':{'rows':rows,'workers':workers,'median_paired_wall_speedup':median,'target':1.1,'decision':'supported' if median>=1.1 else 'falsified'},'fresh_inputs':0,'primary_scientific_evaluations_added':0,'scope':'Separate finite geometry and full-policy timing targets, four balanced pairs each; complete board/iteration/flag/choice/trace parity against archived source and between implementations. No timing CI or new accuracy samples.'})


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['freeze','geometry','policy','collect']);a.add_argument('--gpu',type=int,choices=[0,1]);v=a.parse_args();policy(v.gpu) if v.command=='policy' else globals()[v.command]()
