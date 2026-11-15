"""H2bl margin measurement and complete seal, without graph outcome access."""
import argparse
import copy
import json
import os
import resource
import time
import numpy as np
import study as S
import selective_fallback as F
import fast_npz as Z
from cross_certificate_screen_policy import screen

PP=S.HERE/'cross-certificate-screen-protocol.json';DEST=S.OUT/'cross_certificate_screen';SOURCE=S.OUT/'loop80_confirmation'


def protocol():
    p=json.loads(PP.read_text())
    for key in ['source_sha256','source_evidence_sha256']:
        for name,h in p[key].items():assert S.sha(S.ROOT/name)==h,name
    return p


def freeze():
    assert not PP.exists();proposal=S.HERE/'cross-certificate-screen-proposal.json';p=copy.deepcopy(json.loads(proposal.read_text()))
    sp=json.loads((SOURCE/'protocol.json').read_text());a=json.loads((SOURCE/'audit.json').read_text());assert a['status']=='passed'
    gate=DEST/'synthetic-gate.json';g=json.loads(gate.read_text());assert g['status']=='passed' and g['synthetic_cases']==1296
    for n,h in g['source_sha256'].items():assert S.sha(S.HERE/n)==h
    jobs=[]
    for job in sp['jobs']:
        run=SOURCE/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert S.sha(run/'outputs.npz')==r['outputs_sha256']
        jobs.append({**job,'gpu':job['shard']%2,'outputs_sha256':r['outputs_sha256'],'receipt_sha256':S.sha(run/'result.json')})
    names=['cross_certificate_screen_policy.py','preflight_cross_certificate_screen.py','cross_certificate_screen_study.py','audit_cross_certificate_screen.py']
    p.update(status='frozen_before_source_gates_and_all_case_margins',frozen_utc=S.utc(),jobs=jobs,models=sp['models'],
             datasets=json.loads((SOURCE/'datasets.json').read_text())['datasets'])
    p['source_sha256']={**sp['source_sha256'],**{str((S.HERE/n).relative_to(S.ROOT)):S.sha(S.HERE/n) for n in names}}
    evidence=[proposal,gate,SOURCE/'protocol.json',SOURCE/'audit.json',SOURCE/'datasets.json',SOURCE/'model-extraction.json',
              S.OUT/'certificate_fallback/protocol.json',S.OUT/'certificate_fallback/audit.json']
    evidence.extend(SOURCE/f'model-seed{s}-{arm}.npz' for s in range(3) for arm in ['pl','old_binary'])
    evidence.extend(S.OUT/'certificate_fallback'/folder/f'i4078-s{s}.npz' for s in range(3) for folder in ['fallback','diagnostics'])
    p['source_evidence_sha256']={str(q.relative_to(S.ROOT)):S.sha(q) for q in evidence};S.save(PP,p);S.save(DEST/'protocol.json',p);S.emit('cross_certificate_screen_frozen',sha256=S.sha(PP))


def measure(model,torch,tokens,boards,couplings):
    torch.cuda.synchronize();before=time.perf_counter();t=torch.tensor(tokens.astype(np.int64),device='cuda');b=torch.tensor(boards.astype(np.float64),device='cuda');a=torch.tensor(couplings.astype(np.float64),device='cuda');torch.cuda.synchronize();transfer=time.perf_counter()-before
    before=time.perf_counter()
    with torch.no_grad():
        delta=model.flip_delta(t,b,a).masked_fill(t!=1,float('inf')).flatten(1);margin,witness=delta.min(1);m=margin.cpu().numpy();w=witness.cpu().numpy()
    torch.cuda.synchronize();seconds=time.perf_counter()-before
    return m,w,transfer,seconds


def evaluate(gpu):
    started=time.perf_counter();lock,inventory=F.acquire(gpu);torch,_,_=S.imports();p=protocol();models={}
    state={'status':'running','gpu':gpu,'pid':os.getpid(),'inventory':inventory,'started_utc':S.utc(),'jobs_complete':[],'gate_seeds_complete':[],'watchdog_checks':0}
    def watch():
        state['watchdog_checks']+=1;state.update(seconds=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,updated_utc=S.utc())
        S.save(DEST/f'worker{gpu}.json',state)
        if state['seconds']>120 or state['peak_rss_kib']>8*1024**2:raise RuntimeError('Frozen120s/8GiB envelope; no extension')
    watch()
    for seed in range(3):
        record=p['models'][str(seed)]['old_binary'];assert S.sha(S.ROOT/record['path'])==record['sha256'];m=S.model_for(seed,180,'original','cuda');m.load_state_dict(torch.load(S.ROOT/record['path'],map_location='cuda',weights_only=True));models[seed]=m.double()
    gates=[]
    for seed in ([0,2] if gpu==0 else [1]):
        with np.load(S.OUT/'certificate_fallback/fallback'/f'i4078-s{seed}.npz') as z:boards=z['prediction']
        with np.load(S.OUT/'certificate_fallback/diagnostics'/f'i4078-s{seed}.npz') as z:tokens=z['tokens'];couplings=z['couplings'];expected=z['margins'][1]
        margin,witness,transfer,seconds=measure(models[seed],torch,tokens,boards,couplings);assert np.array_equal(margin,expected)
        gates.append({'seed':seed,'margin':margin.tolist(),'witness_flat':witness.tolist(),'transfer_seconds':transfer,'margin_seconds':seconds});state['gate_seeds_complete'].append(seed);watch()
    S.save(DEST/f'gate-gpu{gpu}.json',{'status':'passed','protocol_sha256':S.sha(PP),'rows':gates})
    while not all((DEST/f'gate-gpu{g}.json').exists() for g in [0,1]):
        if (DEST/f'failure-evaluate-{1-gpu}.json').exists():raise RuntimeError('Peer source gate failed')
        time.sleep(.1);watch()
    for g in [0,1]:
        gate=json.loads((DEST/f'gate-gpu{g}.json').read_text());assert gate['status']=='passed' and gate['protocol_sha256']==S.sha(PP)
    token_cache={}
    for job in p['jobs']:
        if job['gpu']!=gpu:continue
        before=time.perf_counter();ds=p['datasets'][job['shard']]
        if job['shard'] not in token_cache:
            assert S.sha(S.ROOT/ds['path'])==ds['sha256']
            with np.load(S.ROOT/ds['path']) as z:token_cache[job['shard']]=z['tokens']
        tokens=token_cache[job['shard']];source=SOURCE/'runs'/job['id']/'outputs.npz';assert S.sha(source)==job['outputs_sha256']
        # Labels, shortest distances and graph outcomes are never loaded in Stage A.
        with np.load(source) as z:boards=z['predictions'][0];couplings=z['couplings'];hidden=z['hidden_good'];finite=z['finite'][0];stopped=z['stopped'][0];own_margin=z['minimum_flip_margin64'][0]
        io=time.perf_counter()-before;margin=[];witness=[];transfer=compute=0.;state['current_job']=job['id']
        for lo in range(0,256,32):
            m,w,x,y=measure(models[job['seed']],torch,tokens[lo:lo+32],boards[lo:lo+32],couplings[lo:lo+32]);margin.append(m);witness.append(w);transfer+=x;compute+=y;watch()
        lp_margin=np.concatenate(margin);witness=np.concatenate(witness);clamps=((tokens==1)|(boards==(tokens>=2))).all((1,2));own,cross,dual=screen(hidden,finite,stopped,clamps,own_margin,lp_margin)
        run=DEST/'runs'/job['id'];run.mkdir(parents=True);path=run/'margins.npz';assert not path.exists();before=time.perf_counter()
        Z.save(path,hidden_good=hidden,finite=finite,stopped=stopped,clamps_ok=clamps,bp_margin=own_margin,lp_margin=lp_margin,witness_flat=witness,bp_own_accept=own,lp_cross_accept=cross,dual_accept=dual)
        receipt={'status':'complete','job':job,'n':256,'protocol_sha256':S.sha(PP),'outputs_sha256':S.sha(path),'source_outputs_sha256':job['outputs_sha256'],
                 'source_io_seconds':io,'transfer_seconds':transfer,'margin_seconds':compute,'sealing_seconds':time.perf_counter()-before,
                 'labels_distances_outcomes_loaded':False};S.save(run/'result.json',receipt);state['jobs_complete'].append(job['id']);watch()
    state.update(status='complete',seconds=time.perf_counter()-started);S.save(DEST/f'worker{gpu}.json',state)


def seal():
    p=protocol();workers=[json.loads((DEST/f'worker{g}.json').read_text()) for g in [0,1]]
    assert all(w['status']=='complete' and w['seconds']<=120 for w in workers);rows=[]
    for job in p['jobs']:
        run=DEST/'runs'/job['id'];r=json.loads((run/'result.json').read_text());assert r['job']==job and r['n']==256 and r['protocol_sha256']==S.sha(PP) and not r['labels_distances_outcomes_loaded']
        assert S.sha(run/'margins.npz')==r['outputs_sha256'];rows.append({'job':job,'margin_sha256':r['outputs_sha256'],'receipt_sha256':S.sha(run/'result.json')})
    assert len(rows)==48;assert not (DEST/'sealed-margins.json').exists()
    S.save(DEST/'sealed-margins.json',{'status':'complete','protocol_sha256':S.sha(PP),'sealed_utc':S.utc(),'model_inputs':12288,'rows':rows,'worker_receipts':workers,'labels_joined':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','evaluate','seal']);parser.add_argument('--gpu',type=int,choices=[0,1]);args=parser.parse_args()
    try:evaluate(args.gpu) if args.command=='evaluate' else globals()[args.command]()
    except BaseException as error:
        import traceback
        S.save(DEST/f'failure-{args.command}-{args.gpu}.json',{'status':'incomplete','error':repr(error),'traceback':traceback.format_exc(),'partial_evidence_retained':True});raise
