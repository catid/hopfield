"""H4av two-phase execution: seal all fits before any truth/mask diagnostics."""
from pathlib import Path
import argparse,datetime,json,os,resource,shutil,signal,subprocess,time
WALL_ORIGIN=time.monotonic()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
from corruption_censor import ROOT,HERE,PYTHON,FAMILIES,sha,read,save
OBS=ROOT/'results/research/foureyes/corruption_censor'
BASE=ROOT/'results/research/foureyes/prefix_budget/degree'
PARENT_PROTOCOL=HERE/'corruption-censor.json'
OUT=ROOT/'artifacts/research/20261114/foureyes/conflict_mask'
PUBLIC=ROOT/'results/research/foureyes/conflict_mask'
PROTOCOL=HERE/'conflict-mask.json'
ARMS=['detector','random']


def verify(p,sources_only=False):
    for rel,digest in (p['source_sha256'] if sources_only else p['source_sha256']|p['input_sha256']).items():
        assert sha(ROOT/rel)==digest,rel


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=read(HERE/'conflict-mask-preflight.json');seeds=read(HERE/'conflict-mask-seed-audit.json')
    assert gate['status']=='passed' and not gate['study_masks_scores_or_LPs_read'] and seeds['status']=='passed'
    assert gate['seed_audit_sha256']==sha(HERE/'conflict-mask-seed-audit.json')
    for rel,digest in gate['source_sha256'].items():assert sha(ROOT/rel)==digest
    for rel,digest in seeds['frozen_protocol_sha256'].items():assert sha(ROOT/rel)==digest
    parent=read(PARENT_PROTOCOL);design=read(HERE/'conflict-mask-proposal.json')
    inputs={rel:h for rel,h in parent['input_sha256'].items() if not rel.startswith('artifacts/')}
    for path in [PARENT_PROTOCOL,HERE/'conflict-mask-proposal.json',HERE/'conflict-mask-preflight.json',HERE/'conflict-mask-seed-audit.json',*sorted(OBS.rglob('*.npz')),*sorted(OBS.rglob('*.json'))]:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    jobs=[dict(index=j['index'],inferred_degree=j['inferred_degree'],parent_prefix_index=j['parent_prefix_index']) for j in parent['jobs']]
    p=dict(design,status='frozen_before_cohort_detector_random_masks_scores_and_LPs',
           frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jobs=jobs,
           source_sha256=gate['source_sha256'],input_sha256=inputs,
           solver=dict(parent['solver'],random_seed_base=design['solver_seed_base']))
    verify(p);save(PROTOCOL,p);print('FROZEN',sha(PROTOCOL),flush=True)


def seal_fits(p):
    assert not (OUT/'fit-seal.json').exists() and not list(OUT.glob('b*/audit.json'))
    files={};latest=0
    for index in range(256):
        for arm in ARMS:
            receipt=read(OUT/f'b{index:03d}/{arm}.json')
            assert receipt['index']==index and receipt['arm']==arm and receipt['protocol_sha256']==sha(PROTOCOL)
            latest=max(latest,receipt['fit_completed_monotonic_ns'])
            assert receipt['models_sha256']==sha(OUT/f'b{index:03d}/{arm}.npz')
        for name in ['detector-scores.npz']+[f'{arm}.{ext}' for arm in ARMS for ext in ['npz','json']]:
            rel=f'b{index:03d}/{name}';files[rel]=sha(OUT/rel)
    sealed=time.monotonic_ns();assert sealed>latest
    save(OUT/'fit-seal.json',dict(status='all512_fits_sealed_before_truth_or_corruption_diagnostics',
         protocol_sha256=sha(PROTOCOL),fits=512,latest_fit_monotonic_ns=latest,sealed_monotonic_ns=sealed,files=files))


def check_seal(folder):
    seal=read(folder/'fit-seal.json')
    assert seal['protocol_sha256']==sha(PROTOCOL) and seal['fits']==512
    assert seal['status']=='all512_fits_sealed_before_truth_or_corruption_diagnostics'
    latest=0
    for index in range(256):
        for arm in ARMS:
            rec=read(folder/f'b{index:03d}/{arm}.json')
            assert rec['index']==index and rec['arm']==arm and rec['protocol_sha256']==sha(PROTOCOL)
            latest=max(latest,rec['fit_completed_monotonic_ns'])
    expected={f'b{i:03d}/{name}' for i in range(256) for name in ['detector-scores.npz']+[f'{a}.{e}' for a in ARMS for e in ['npz','json']]}
    assert set(seal['files'])==expected and latest==seal['latest_fit_monotonic_ns']<seal['sealed_monotonic_ns']
    for rel,h in seal['files'].items():assert sha(folder/rel)==h
    return seal


def fit_worker(worker_id):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    from conflict_mask_math import detector,random_mask
    from corruption_censor_math import score
    p=read(PROTOCOL);verify(p,sources_only=True);ii,jj=np.triu_indices(81,1)
    for index in range(worker_id,256,8):
        job=p['jobs'][index];degree=job['inferred_degree'];folder=OUT/f'b{index:03d}';folder.mkdir()
        # NPZ members are lazy: no cells, full mask, truth, clean digits or offsets are decoded.
        with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:noisy=z['noisy']
        with np.load(BASE/f"j{job['parent_prefix_index']:03d}/models.npz",allow_pickle=False) as z:primal=z[f'd{degree:02d}_primal_twice']
        conflicts,detected=detector(noisy,primal)
        np.savez_compressed(folder/'detector-scores.npz',scores=conflicts)
        masks=[detected,random_mask(600,81,p['random_seed_base']+index)]
        for ai,(arm,mask) in enumerate(zip(ARMS,masks)):
            result=score(noisy,mask);log=folder/f'{arm}.log'
            x,y,meta=solve(81,degree,ii,jj,result['cost'],p['solver_seed_base']-2026111502+2*index+ai,log)
            if meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']:
                cert,proof=attempt_certificate(81,degree,ii,jj,result['cost'],x,y)
            else:
                cert=dict(certified=False,reason='solver_status_or_validity')
                proof={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
            np.savez_compressed(folder/f'{arm}.npz',mask=mask,**result,primal_values=x,node_dual_values=y,**proof)
            save(folder/f'{arm}.json',dict(index=index,arm=arm,degree=degree,protocol_sha256=sha(PROTOCOL),
                 models_sha256=sha(folder/f'{arm}.npz'),log_sha256=sha(log),solver=meta,certificate=cert,
                 fit_completed_monotonic_ns=time.monotonic_ns()))
            save(folder/'progress.json',dict(completed_LPs=ai+1,pid=os.getpid()))


def audit_worker(worker_id):
    from audit_conflict_mask import audit_bank
    p=read(PROTOCOL);verify(p,sources_only=True);seal=check_seal(OUT)
    for index in range(worker_id,256,8):
        started=time.monotonic_ns();assert started>seal['sealed_monotonic_ns']
        audit=audit_bank(index,p)
        save(OUT/f'b{index:03d}/audit.json',audit)
        save(OUT/f'b{index:03d}/audit-order.json',dict(audit_started_monotonic_ns=started,seal_sha256=sha(OUT/'fit-seal.json')))


def export(p):
    from audit_conflict_mask import aggregate
    PUBLIC.mkdir();rows=[];files={}
    for index in range(256):
        source=OUT/f'b{index:03d}';audit=read(source/'audit.json')
        assert audit['protocol_sha256']==sha(PROTOCOL) and audit['auditor_sha256']==sha(HERE/'audit_conflict_mask.py')
        for name,h in audit['artifact_sha256'].items():assert sha(source/name)==h
        rows.extend(audit['rows']);target=PUBLIC/source.name;target.mkdir()
        for name in list(audit['artifact_sha256'])+['audit.json','audit-order.json']:
            shutil.copyfile(source/name,target/name);files[str((target/name).relative_to(PUBLIC))]=sha(target/name)
    shutil.copyfile(OUT/'fit-seal.json',PUBLIC/'fit-seal.json');files['fit-seal.json']=sha(PUBLIC/'fit-seal.json')
    save(PUBLIC/'summary.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),rows=rows,**aggregate(rows),artifact_sha256=files,
         new_LPs=512,new_source_draws=0,new_labels=0,inherited_certificates_checked=512,inherited_mutations_rejected=1536,
         scope='Observable same-data conflict detector with supplied per-row corruption count9; matched uniform9mask control. Actual locations used only in post-seal audits.'))


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
        verify(p);monitor('hashes');phase('fit_worker');seal_fits(p);monitor('all_fits_sealed');phase('audit_worker');export(p);monitor('export');status='complete'
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
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','fit_worker','audit_worker','launch']);parser.add_argument('--worker',type=int);args=parser.parse_args()
    globals()[args.action](args.worker) if args.action.endswith('_worker') else globals()[args.action]()
