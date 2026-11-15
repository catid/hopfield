"""H4ba disjoint1200-row donor fits, globally sealed before target masking."""
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
OUT=ROOT/'artifacts/research/20261114/foureyes/larger_donor'
PUBLIC=ROOT/'results/research/foureyes/larger_donor'
PROTOCOL=HERE/'larger-donor.json'
ARMS=['donor','final']


def verify(p,sources_only=False,public=False):
    for rel,digest in (p['source_sha256'] if sources_only else p['source_sha256']|p['input_sha256']).items():
        if public and rel.startswith('artifacts/') and not (ROOT/rel).exists():continue
        assert sha(ROOT/rel)==digest,rel


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=read(HERE/'larger-donor-preflight.json');seeds=read(HERE/'larger-donor-seed-audit.json')
    assert gate['status']=='passed' and not gate['study_masks_scores_or_LPs_read'] and seeds['status']=='passed'
    assert gate['seed_audit_sha256']==sha(HERE/'larger-donor-seed-audit.json')
    assert gate['proposal_sha256']==sha(HERE/'larger-donor-proposal.json')
    assert gate['compact_dtype_gate_sha256']==sha(HERE/'disjoint-donor-repair-preflight.json')
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest
    for rel,digest in seeds['frozen_protocol_sha256'].items():assert sha(ROOT/rel)==digest
    parent=read(PARENT_PROTOCOL);design=read(HERE/'larger-donor-proposal.json')
    inputs={rel:h for rel,h in parent['input_sha256'].items() if not rel.startswith('artifacts/')}
    for path in [PARENT_PROTOCOL,HERE/'larger-donor-proposal.json',HERE/'larger-donor-preflight.json',HERE/'larger-donor-seed-audit.json',HERE/'disjoint-donor-repair-preflight.json',*sorted(DETECTOR.rglob('*.npz')),*sorted(DETECTOR.rglob('*.json'))]:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    jobs=[]
    for j in read(HERE/'corruption-censor.json')['jobs']:
        jobs.append(dict(index=j['index'],inferred_degree=j['inferred_degree'],parent_prefix_index=j['parent_prefix_index'],archive=j['archive']))
        path=ROOT/j['archive']['source_path'];assert sha(path)==j['archive']['source_sha256'];inputs[str(path.relative_to(ROOT))]=sha(path)
    for name in ['varying_degree','cycle_power']:
        for path in [HERE/(name+'.json'),ROOT/f'results/research/foureyes/{name}/controls/source-audit.json']:
            inputs[str(path.relative_to(ROOT))]=sha(path)
    p=dict(design,status='frozen_before_larger_donor_costs_fits_and_masks',
           frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=jobs,
           source_sha256=gate['source_sha256'],input_sha256=inputs,
           solver=dict(parent['solver'],random_seed_base=design['solver_seed_base']))
    verify(p);save(PROTOCOL,p);print('FROZEN',sha(PROTOCOL),flush=True)


def phase_files(phase):
    arms=['donor'] if phase=='donor' else ARMS
    return {f'b{i:03d}/{arm}.{ext}' for i in range(256) for arm in arms for ext in ['npz','json']}


def check_phase_seal(folder,phase):
    assert phase in ['donor','fit']
    prior=check_phase_seal(folder,'donor') if phase=='fit' else None
    seal=read(folder/f'{phase}-seal.json')
    assert seal['status']==f'all_{phase}_outputs_sealed' and seal['protocol_sha256']==sha(PROTOCOL)
    assert set(seal['files'])==phase_files(phase)
    for rel,h in seal['files'].items():assert sha(folder/rel)==h,rel
    latest=0
    for index in range(256):
        for ai,arm in enumerate(['donor'] if phase=='donor' else ARMS):
            rec=read(folder/f'b{index:03d}/{arm}.json')
            assert rec['index']==index and rec['arm']==arm and rec['protocol_sha256']==sha(PROTOCOL)
            assert rec['models_sha256']==sha(folder/f'b{index:03d}/{arm}.npz')
            assert rec['fit_started_monotonic_ns']<=rec['fit_completed_monotonic_ns']
            latest=max(latest,rec['fit_completed_monotonic_ns'])
            if arm=='final':
                assert prior['sealed_monotonic_ns']<rec['fit_started_monotonic_ns']
                assert rec['donor_seal_sha256']==sha(folder/'donor-seal.json')
    assert latest==seal['latest_fit_monotonic_ns']<seal['sealed_monotonic_ns']
    return seal


def seal_phase(phase):
    assert phase in ['donor','fit'] and not (OUT/f'{phase}-seal.json').exists()
    assert not list(OUT.glob('b*/audit.json'))
    if phase=='donor':assert not list(OUT.glob('b*/final.json'))
    else:check_phase_seal(OUT,'donor')
    latest=max(read(OUT/rel)['fit_completed_monotonic_ns'] for rel in phase_files(phase) if rel.endswith('.json'))
    save(OUT/f'{phase}-seal.json',dict(status=f'all_{phase}_outputs_sealed',protocol_sha256=sha(PROTOCOL),
        latest_fit_monotonic_ns=latest,sealed_monotonic_ns=time.monotonic_ns(),
        files={rel:sha(OUT/rel) for rel in sorted(phase_files(phase))}))
    check_phase_seal(OUT,phase)


def check_seal(folder):return check_phase_seal(folder,'fit')


def fit_allowed(donor):
    return donor['certificate']['certified'] is True


def fit_one(index,arm,cost,extra,p,folder,started,donor_seal=None):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    degree=p['jobs'][index]['inferred_degree'];ii,jj=np.triu_indices(81,1);phase=ARMS.index(arm)
    log=folder/f'{arm}.log'
    x,y,meta=solve(81,degree,ii,jj,cost,p['solver_seed_base']-2026111502+2*index+phase,log)
    if meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']:
        cert,proof=attempt_certificate(81,degree,ii,jj,cost,x,y)
    else:
        cert=dict(certified=False,reason='solver_status_or_validity')
        proof={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
    np.savez_compressed(folder/f'{arm}.npz',cost=cost,**extra,primal_values=x,node_dual_values=y,**proof)
    save(folder/f'{arm}.json',dict(index=index,arm=arm,degree=degree,executed=True,protocol_sha256=sha(PROTOCOL),
        models_sha256=sha(folder/f'{arm}.npz'),log_sha256=sha(log),solver=meta,certificate=cert,
        fit_started_monotonic_ns=started,fit_completed_monotonic_ns=time.monotonic_ns(),donor_seal_sha256=donor_seal))


def donor_worker(worker_id):
    from larger_donor_math import counts
    p=read(PROTOCOL);verify(p,sources_only=True)
    for index in range(worker_id,256,8):
        folder=OUT/f'b{index:03d}';folder.mkdir();job=p['jobs'][index]
        raw=ROOT/job['archive']['source_path'];assert sha(raw)==job['archive']['source_sha256']
        with np.load(raw,allow_pickle=False) as z:noisy=z['noisy']
        assert noisy.shape==(2400,81)
        with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:target=z['noisy']
        assert np.array_equal(noisy[:600],target)
        donor=noisy[600:1800].copy()
        fit_one(index,'donor',counts(donor),dict(noisy=donor,source_rows=np.arange(600,1800)),p,folder,time.monotonic_ns())
        save(folder/'progress.json',dict(completed_LPs=1,pid=os.getpid()))


def final_bank(index,p):
    from conflict_mask_math import detector
    from corruption_censor_math import score
    folder=OUT/f'b{index:03d}';started=time.monotonic_ns()
    seal=read(OUT/'donor-seal.json');assert started>seal['sealed_monotonic_ns']
    donor=read(folder/'donor.json')
    if not fit_allowed(donor):
        np.savez_compressed(folder/'final.npz',unexecuted=np.array(True))
        save(folder/'final.json',dict(index=index,arm='final',degree=p['jobs'][index]['inferred_degree'],
            executed=False,reason='required_donor_uncertified',certificate=dict(certified=False),
            models_sha256=sha(folder/'final.npz'),protocol_sha256=sha(PROTOCOL),
            donor_seal_sha256=sha(OUT/'donor-seal.json'),fit_started_monotonic_ns=started,fit_completed_monotonic_ns=time.monotonic_ns()))
        return
    with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:noisy=z['noisy']
    with np.load(folder/'donor.npz',allow_pickle=False) as z:primal=z['primal_twice']
    scores,mask=detector(noisy,primal);result=score(noisy,mask)
    fit_one(index,'final',result['cost'],dict(scores=scores,mask=mask,exposure=result['exposure']),p,folder,started,sha(OUT/'donor-seal.json'))
    save(folder/'progress.json',dict(completed_LPs=2,pid=os.getpid()))


def final_worker(worker_id):
    p=read(PROTOCOL);verify(p,sources_only=True);check_phase_seal(OUT,'donor')
    for index in range(worker_id,256,8):final_bank(index,p)


def audit_worker(worker_id):
    from audit_larger_donor import audit_bank
    p=read(PROTOCOL);verify(p,sources_only=True);seal=check_seal(OUT)
    for index in range(worker_id,256,8):
        started=time.monotonic_ns();assert started>seal['sealed_monotonic_ns']
        audit=audit_bank(index,p)
        save(OUT/f'b{index:03d}/audit.json',audit)
        save(OUT/f'b{index:03d}/audit-order.json',dict(audit_started_monotonic_ns=started,seal_sha256=sha(OUT/'fit-seal.json')))


def export(p):
    from audit_larger_donor import aggregate
    PUBLIC.mkdir();rows=[];files={}
    for index in range(256):
        source=OUT/f'b{index:03d}';audit=read(source/'audit.json')
        assert audit['protocol_sha256']==sha(PROTOCOL) and audit['auditor_sha256']==sha(HERE/'audit_larger_donor.py')
        for name,h in audit['artifact_sha256'].items():assert sha(source/name)==h
        rows.extend(audit['rows']);target=PUBLIC/source.name;target.mkdir()
        for name in list(audit['artifact_sha256'])+['audit.json','audit-order.json']:
            shutil.copyfile(source/name,target/name);files[str((target/name).relative_to(PUBLIC))]=sha(target/name)
    for name in ['donor-seal.json','fit-seal.json']:
        shutil.copyfile(OUT/name,PUBLIC/name);files[name]=sha(PUBLIC/name)
    save(PUBLIC/'summary.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),rows=rows,**aggregate(rows),artifact_sha256=files,
         new_LPs=sum(r['attempted_LPs'] for r in rows),new_source_draws=0,new_labels=0,inherited_certificates_checked=1024,inherited_mutations_rejected=3072,
         scope='Disjoint1200-row donor detector on examined nested banks, inherited target600-row degree and supplied9flags perrow. Full original noise replay occurs after all target outputs seal; raw source archives required for original-provenance recheck.'))


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
        verify(p);monitor('hashes');phase('donor_worker');seal_phase('donor');monitor('all_donors_sealed');phase('final_worker');seal_phase('fit');monitor('all_fits_sealed');phase('audit_worker');export(p);monitor('export');status='complete'
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
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','donor_worker','final_worker','audit_worker','launch']);parser.add_argument('--worker',type=int);args=parser.parse_args()
    globals()[args.action](args.worker) if args.action.endswith('_worker') else globals()[args.action]()
