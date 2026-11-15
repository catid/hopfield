"""H2bf every archived lexical geometry plus fixed hash-selected timing fixtures."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
import time
import numpy as np
import study as S
import connected_block_moves as OLD
import bridge_veto_geometry as NEW
import fast_npz as Z

DEST=S.OUT/'veto_speed_preflight';SOURCES=('density_transfer','unified_cycle','loop80_transfer')


def state_hash(t,b):return hashlib.sha256(b'H2bf-veto-v1'+np.ascontiguousarray(t,dtype=np.uint8).tobytes()+np.ascontiguousarray(b,dtype=np.uint8).tobytes()).hexdigest()


def job_check(args):
    source,job=args;started=time.perf_counter();d=S.OUT/source;run=d/'runs'/job['id'];p=json.loads((d/'protocol.json').read_text());receipt=json.loads((run/'result.json').read_text());assert receipt['outputs_sha256']==S.sha(run/'outputs.npz');manifest=json.loads((d/'datasets.json').read_text());ds=manifest['datasets'][job['shard']];assert S.sha(S.ROOT/ds['path'])==ds['sha256']
    with np.load(S.ROOT/ds['path']) as z:tokens=z['tokens']
    with np.load(run/'outputs.npz') as z:pred=z['predictions']
    refs={};checks=candidate_count=0
    def check(tok,board,cells,geometry,reference):
        nonlocal checks,candidate_count
        expected=(cells,geometry);old=OLD.candidates(tok,board);new=NEW.candidates(tok,board);assert old==new==expected,(source,job['id'],reference);checks+=1;candidate_count+=len(cells);digest=state_hash(tok,board);refs.setdefault(digest,{'source':source,'job':job['id'],'dataset':ds['path'],**reference})
    for item in receipt['trace_archives']:
        arm=p['arms'].index(item['arm'])
        if arm!=2:continue
        lo=item['offset'];path=run/item['path'];assert S.sha(path)==item['sha256']
        with np.load(path) as z:ix=z['valid_prefix_indices'];history=z['states_before'];trace=json.loads(z['trace_utf8'].tobytes())
        for ri,record in enumerate(trace['trace']):
            for g in record['geometry']:
                i=g['local_input'];cells=[tuple(r['cells']) for r in record['candidates'] if r['local_input']==i];stats={k:v for k,v in g.items() if k not in ['local_input','candidate_count']};assert g['candidate_count']==len(cells);check(tokens[lo+ix[i]],history[ri,i],cells,stats,{'kind':'round','trace_file':item['path'],'trace_sha256':item['sha256'],'round':ri,'local_input':i,'input_index':int(lo+ix[i]),'arm':arm})
        for kind,key in [('initial_terminal','initial_terminal_checks'),('final_terminal','final_terminal_checks')]:
            for record in trace.get(key,[]):
                i=record['local_input'];index=lo+i if kind=='initial_terminal' else lo+int(ix[i]);board=pred[arm-1 if kind=='initial_terminal' else arm,index];cells=[tuple(c) for c in record['candidates']];assert len(cells)==record['candidate_count'];check(tokens[index],board,cells,record['geometry'],{'kind':kind,'trace_file':item['path'],'trace_sha256':item['sha256'],'input_index':int(index),'arm':arm})
    r={'status':'passed','source':source,'job':job['id'],'source_protocol_sha256':S.sha(d/'protocol.json'),'source_outputs_sha256':S.sha(run/'outputs.npz'),'states_checked':checks,'candidate_tuples':candidate_count,'unique_state_references':refs,'seconds':time.perf_counter()-started};path=DEST/'archive-rows'/f'{source}-{job["id"]}.json';temp=path.with_name(path.name+f'.tmp-{os.getpid()}');temp.write_text(json.dumps(r,indent=2)+'\n');temp.replace(path);return r


def main():
    started=time.perf_counter();(DEST/'archive-rows').mkdir(exist_ok=True);jobs=[]
    for source in SOURCES:
        p=json.loads((S.OUT/source/'protocol.json').read_text());jobs.extend((source,j) for j in p['jobs'])
    rows=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for future in as_completed([pool.submit(job_check,j) for j in jobs]):
            rows.append(future.result());S.emit('bridge_archive_job_checked',completed=len(rows),total=len(jobs));assert time.perf_counter()-started<600
    rows.sort(key=lambda r:(SOURCES.index(r['source']),r['job']));refs={}
    for row in rows:
        for key,value in row['unique_state_references'].items():refs.setdefault(key,value)
    selected={k:refs[k] for k in sorted(refs)[:1024]};assert len(selected)==1024;groups={}
    for digest,ref in selected.items():groups.setdefault((ref['source'],ref['job']),[]).append((digest,ref))
    arrays={};
    for (source,jid),group in groups.items():
        run=S.OUT/source/'runs'/jid;dataset=group[0][1]['dataset']
        with np.load(S.ROOT/dataset) as z:tokens=z['tokens']
        with np.load(run/'outputs.npz') as z:pred=z['predictions']
        trace_cache={}
        for digest,ref in group:
            if ref['kind']=='round':
                file=ref['trace_file']
                if file not in trace_cache:
                    with np.load(run/file) as z:trace_cache[file]=z['states_before']
                board=trace_cache[file][ref['round'],ref['local_input']]
            else:board=pred[ref['arm']-1 if ref['kind']=='initial_terminal' else ref['arm'],ref['input_index']]
            tok=tokens[ref['input_index']];assert state_hash(tok,board)==digest;arrays[digest]=(tok.copy(),board.copy())
    target=DEST/'geometry-fixtures';target.mkdir(exist_ok=True);fixture_rows=[];ordered=sorted(selected)
    for start in range(0,1024,64):
        keys=ordered[start:start+64];path=target/f'states{start//64:02d}.npz';Z.save(path,tokens=np.stack([arrays[k][0] for k in keys]),boards=np.stack([arrays[k][1] for k in keys]));fixture_rows.append({'file':path.name,'sha256':S.sha(path),'state_hashes':keys,'source_references':[selected[k] for k in keys]})
    S.save(target/'manifest.json',{'status':'fixed_before_speed_measurements','n':1024,'unique_available':len(refs),'selection':'Smallest1024 SHA256 over domain H2bf-veto-v1 and canonical contiguous uint8 tokens then boards; deduplicate exact states. No timing values used.','fixtures':fixture_rows});result={'status':'passed','proposal_sha256':S.sha(S.HERE/'veto-speed-proposal.json'),'states_checked':sum(r['states_checked'] for r in rows),'candidate_tuples':sum(r['candidate_tuples'] for r in rows),'unique_states':len(refs),'selected_states':1024,'selected_manifest_sha256':S.sha(target/'manifest.json'),'rows':[{k:v for k,v in r.items() if k!='unique_state_references'} for r in rows],'processes':16,'seconds':time.perf_counter()-started,'source_sha256':{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in ['bridge_veto_geometry.py','preflight_veto_archives.py']},'scope':'All archived active veto and terminal geometry checked against immutable published traces and legacy implementation; zero fresh inputs or benchmark timings.'};assert result['seconds']<600;S.save(DEST/'archive-gate.json',result)


if __name__=='__main__':main()
