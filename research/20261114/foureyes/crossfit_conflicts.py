"""H4ax cross-fitting: seal folds before masks and final outputs before diagnostics."""
from pathlib import Path
import argparse,datetime,json,os,resource,shutil,signal,subprocess,time
WALL_ORIGIN=time.monotonic()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
from corruption_censor import ROOT,HERE,PYTHON,FAMILIES,sha,read,save
OBS=ROOT/'results/research/foureyes/corruption_censor'
BASE=ROOT/'results/research/foureyes/prefix_budget/degree'
PARENT_PROTOCOL=HERE/'conflict-mask.json'
DETECTOR=ROOT/'results/research/foureyes/conflict_mask'
OUT=ROOT/'artifacts/research/20261114/foureyes/crossfit_conflicts'
PUBLIC=ROOT/'results/research/foureyes/crossfit_conflicts'
PROTOCOL=HERE/'crossfit-conflicts.json'
ARMS=['fold0','fold1','final']


def verify(p,sources_only=False):
    for rel,digest in (p['source_sha256'] if sources_only else p['source_sha256']|p['input_sha256']).items():
        assert sha(ROOT/rel)==digest,rel


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=read(HERE/'crossfit-conflicts-preflight.json');seeds=read(HERE/'crossfit-conflicts-seed-audit.json')
    assert gate['status']=='passed' and not gate['study_masks_scores_or_LPs_read'] and seeds['status']=='passed'
    assert gate['seed_audit_sha256']==sha(HERE/'crossfit-conflicts-seed-audit.json')
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest
    for rel,digest in seeds['frozen_protocol_sha256'].items():assert sha(ROOT/rel)==digest
    parent=read(PARENT_PROTOCOL);design=read(HERE/'crossfit-conflicts-proposal.json')
    inputs={rel:h for rel,h in parent['input_sha256'].items() if not rel.startswith('artifacts/')}
    for path in [PARENT_PROTOCOL,HERE/'crossfit-conflicts-proposal.json',HERE/'crossfit-conflicts-preflight.json',HERE/'crossfit-conflicts-seed-audit.json',*sorted(DETECTOR.rglob('*.npz')),*sorted(DETECTOR.rglob('*.json'))]:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    jobs=[dict(index=j['index'],inferred_degree=j['inferred_degree'],parent_prefix_index=j['parent_prefix_index']) for j in parent['jobs']]
    p=dict(design,status='frozen_before_fold_fits_crossfit_masks_and_final_LPs',
           frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=jobs,
           source_sha256=gate['source_sha256'],input_sha256=inputs,
           solver=dict(parent['solver'],random_seed_base=design['solver_seed_base']))
    verify(p);save(PROTOCOL,p);print('FROZEN',sha(PROTOCOL),flush=True)


def phase_files(phase):
    arms=['fold0','fold1'] if phase=='fold' else ARMS
    return {f'b{i:03d}/{arm}.{ext}' for i in range(256) for arm in arms for ext in ['npz','json']}


def check_phase_seal(folder,phase):
    assert phase in ['fold','fit']
    prior=check_phase_seal(folder,'fold') if phase=='fit' else None
    seal=read(folder/f'{phase}-seal.json')
    assert seal['status']==f'all_{phase}_outputs_sealed' and seal['protocol_sha256']==sha(PROTOCOL)
    assert set(seal['files'])==phase_files(phase)
    for rel,h in seal['files'].items():assert sha(folder/rel)==h,rel
    latest=0
    for index in range(256):
        for ai,arm in enumerate(['fold0','fold1'] if phase=='fold' else ARMS):
            rec=read(folder/f'b{index:03d}/{arm}.json')
            assert rec['index']==index and rec['arm']==arm and rec['protocol_sha256']==sha(PROTOCOL)
            assert rec['models_sha256']==sha(folder/f'b{index:03d}/{arm}.npz')
            assert rec['fit_started_monotonic_ns']<=rec['fit_completed_monotonic_ns']
            latest=max(latest,rec['fit_completed_monotonic_ns'])
            if arm=='final':
                assert prior['sealed_monotonic_ns']<rec['fit_started_monotonic_ns']
                assert rec['fold_seal_sha256']==sha(folder/'fold-seal.json')
    assert latest==seal['latest_fit_monotonic_ns']<seal['sealed_monotonic_ns']
    return seal


def seal_phase(phase):
    assert phase in ['fold','fit'] and not (OUT/f'{phase}-seal.json').exists()
    assert not list(OUT.glob('b*/audit.json'))
    if phase=='fold':assert not list(OUT.glob('b*/final.json'))
    else:check_phase_seal(OUT,'fold')
    latest=max(read(OUT/rel)['fit_completed_monotonic_ns'] for rel in phase_files(phase) if rel.endswith('.json'))
    save(OUT/f'{phase}-seal.json',dict(status=f'all_{phase}_outputs_sealed',protocol_sha256=sha(PROTOCOL),
        latest_fit_monotonic_ns=latest,sealed_monotonic_ns=time.monotonic_ns(),
        files={rel:sha(OUT/rel) for rel in sorted(phase_files(phase))}))
    check_phase_seal(OUT,phase)


def check_seal(folder):return check_phase_seal(folder,'fit')


def fit_allowed(folds):
    assert len(folds)==2
    return all(r['certificate']['certified'] for r in folds)


def fit_one(index,arm,cost,extra,p,folder,started,fold_seal=None):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    degree=p['jobs'][index]['inferred_degree'];ii,jj=np.triu_indices(81,1);phase=ARMS.index(arm)
    log=folder/f'{arm}.log'
    x,y,meta=solve(81,degree,ii,jj,cost,p['solver_seed_base']-2026111502+3*index+phase,log)
    if meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']:
        cert,proof=attempt_certificate(81,degree,ii,jj,cost,x,y)
    else:
        cert=dict(certified=False,reason='solver_status_or_validity')
        proof={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
    np.savez_compressed(folder/f'{arm}.npz',cost=cost,**extra,primal_values=x,node_dual_values=y,**proof)
    save(folder/f'{arm}.json',dict(index=index,arm=arm,degree=degree,executed=True,protocol_sha256=sha(PROTOCOL),
        models_sha256=sha(folder/f'{arm}.npz'),log_sha256=sha(log),solver=meta,certificate=cert,
        fit_started_monotonic_ns=started,fit_completed_monotonic_ns=time.monotonic_ns(),fold_seal_sha256=fold_seal))


def fold_worker(worker_id):
    from crossfit_conflict_math import fold_costs
    p=read(PROTOCOL);verify(p,sources_only=True)
    for index in range(worker_id,256,8):
        folder=OUT/f'b{index:03d}';folder.mkdir()
        with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:noisy=z['noisy']
        assert noisy.shape==(600,81)
        costs=fold_costs(noisy)
        for phase,cost in enumerate(costs):
            fit_one(index,f'fold{phase}',cost,{},p,folder,time.monotonic_ns())
            save(folder/'progress.json',dict(completed_LPs=phase+1,pid=os.getpid()))


def final_bank(index,p):
    from crossfit_conflict_math import crossfit_mask
    from corruption_censor_math import score
    folder=OUT/f'b{index:03d}';started=time.monotonic_ns()
    seal=read(OUT/'fold-seal.json');assert started>seal['sealed_monotonic_ns']
    folds=[read(folder/f'fold{k}.json') for k in [0,1]]
    if not fit_allowed(folds):
        np.savez_compressed(folder/'final.npz',unexecuted=np.array(True))
        save(folder/'final.json',dict(index=index,arm='final',degree=p['jobs'][index]['inferred_degree'],
            executed=False,reason='required_fold_uncertified',unresolved_folds=[k for k,r in enumerate(folds) if not r['certificate']['certified']],
            certificate=dict(certified=False),models_sha256=sha(folder/'final.npz'),protocol_sha256=sha(PROTOCOL),
            fold_seal_sha256=sha(OUT/'fold-seal.json'),fit_started_monotonic_ns=started,fit_completed_monotonic_ns=time.monotonic_ns()))
        return
    # Decode only noisy observations and the opposite-fold literal primal vectors.
    with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:noisy=z['noisy']
    primals=[]
    for k in [0,1]:
        with np.load(folder/f'fold{k}.npz',allow_pickle=False) as z:primals.append(z['primal_twice'])
    scores,mask=crossfit_mask(noisy,*primals);result=score(noisy,mask)
    fit_one(index,'final',result['cost'],dict(scores=scores,mask=mask,exposure=result['exposure']),p,folder,started,sha(OUT/'fold-seal.json'))
    save(folder/'progress.json',dict(completed_LPs=3,pid=os.getpid()))


def final_worker(worker_id):
    p=read(PROTOCOL);verify(p,sources_only=True);check_phase_seal(OUT,'fold')
    for index in range(worker_id,256,8):final_bank(index,p)


def audit_worker(worker_id):
    from audit_crossfit_conflicts import audit_bank
    p=read(PROTOCOL);verify(p,sources_only=True);seal=check_seal(OUT)
    for index in range(worker_id,256,8):
        started=time.monotonic_ns();assert started>seal['sealed_monotonic_ns']
        audit=audit_bank(index,p)
        save(OUT/f'b{index:03d}/audit.json',audit)
        save(OUT/f'b{index:03d}/audit-order.json',dict(audit_started_monotonic_ns=started,seal_sha256=sha(OUT/'fit-seal.json')))


def export(p):
    from audit_crossfit_conflicts import aggregate
    PUBLIC.mkdir();rows=[];files={}
    for index in range(256):
        source=OUT/f'b{index:03d}';audit=read(source/'audit.json')
        assert audit['protocol_sha256']==sha(PROTOCOL) and audit['auditor_sha256']==sha(HERE/'audit_crossfit_conflicts.py')
        for name,h in audit['artifact_sha256'].items():assert sha(source/name)==h
        rows.extend(audit['rows']);target=PUBLIC/source.name;target.mkdir()
        for name in list(audit['artifact_sha256'])+['audit.json','audit-order.json']:
            shutil.copyfile(source/name,target/name);files[str((target/name).relative_to(PUBLIC))]=sha(target/name)
    for name in ['fold-seal.json','fit-seal.json']:
        shutil.copyfile(OUT/name,PUBLIC/name);files[name]=sha(PUBLIC/name)
    save(PUBLIC/'summary.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),rows=rows,**aggregate(rows),artifact_sha256=files,
         new_LPs=sum(r['attempted_LPs'] for r in rows),new_source_draws=0,new_labels=0,inherited_certificates_checked=1024,inherited_mutations_rejected=3072,
         scope='Fixed opposite-half conflict detector with inherited full-data degree and supplied corruption count9; examined256banks. Truth read only after all final outputs sealed.'))


def launch():
    p=read(PROTOCOL);assert not OUT.exists();OUT.mkdir(parents=True)
    children=[];logs=[];samples=[];status='incomplete';error=None;ticks=os.sysconf('SC_CLK_TCK')
    def cpu():
        total=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])
        for c in children:
            if c.poll() is None:
                try:
                    f=Path(f'/proc/{c.pid}/stat').read_text().split();total+=(int(f[13])+int(f[14]))/ticks
                except FileNotFoundError:pass
        return total
    def monitor(phase):
        sample=dict(phase=phase,wall_seconds=time.monotonic()-WALL_ORIGIN,aggregate_cpu_seconds=cpu(),
                    active_pids=[c.pid for c in children if c.poll() is None],
                    completed_LPs=sum(read(q)['completed_LPs'] for q in OUT.glob('b*/progress.json')),
                    audited_banks=len(list(OUT.glob('b*/audit.json'))))
        samples.append(sample);save(OUT/'progress.json',sample)
        assert sample['wall_seconds']<p['budget']['wall_seconds'] and sample['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds'],'frozen_resource_ceiling'
    def phase(action):
        for i in range(8):
            stream=(OUT/f'{action}{i}.log').open('x');logs.append(stream)
            children.append(subprocess.Popen([str(PYTHON),str(Path(__file__)),action,'--worker',str(i)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children):
            monitor(action);assert all(c.poll() in [None,0] for c in children),'worker_failure';time.sleep(.2)
        assert all(c.returncode==0 for c in children);monitor(action)
    try:
        verify(p);monitor('hashes');phase('fold_worker');seal_phase('fold');monitor('all_folds_sealed');phase('final_worker');seal_phase('fit');monitor('all_fits_sealed');phase('audit_worker');export(p);monitor('export');status='complete'
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid,signal.SIGTERM)
                try:c.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
        for stream in logs:stream.close()
        runtime=dict(status=status,error=error,protocol_sha256=sha(PROTOCOL),wall_seconds=time.monotonic()-WALL_ORIGIN,
                     aggregate_cpu_seconds=cpu(),returncodes=[c.returncode for c in children],samples=samples)
        save(OUT/'runtime.json',runtime)
        if PUBLIC.exists():save(PUBLIC/'runtime.json',runtime)
        print(json.dumps({k:v for k,v in runtime.items() if k!='samples'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','fold_worker','final_worker','audit_worker','launch']);parser.add_argument('--worker',type=int);args=parser.parse_args()
    globals()[args.action](args.worker) if args.action.endswith('_worker') else globals()[args.action]()
