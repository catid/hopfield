"""H4bb: one raw-count LP on the exact1800-row H4ba observation union."""
from pathlib import Path
import argparse,datetime,json,os,resource,shutil,signal,subprocess,time
WALL_ORIGIN=time.monotonic()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from larger_donor import ROOT,HERE,PYTHON,OBS,FAMILIES,sha,read,save,PROTOCOL as PARENT_PROTOCOL,PUBLIC as PARENT,verify as verify_parent,check_seal as check_parent_seal
from larger_donor_math import counts,validate_labels
OUT=ROOT/'artifacts/research/20261114/foureyes/pooled_donor'
PUBLIC=ROOT/'results/research/foureyes/pooled_donor'
PROTOCOL=HERE/'pooled-donor.json'


def observe(index):
    with np.load(OBS/f'b{index:03d}/models.npz',allow_pickle=False) as z:target=z['noisy']
    with np.load(PARENT/f'b{index:03d}/donor.npz',allow_pickle=False) as z:donor=z['noisy'];ids=z['source_rows']
    validate_labels(target,(600,81));validate_labels(donor,(1200,81))
    assert np.array_equal(ids,np.arange(600,1800)) and ids.dtype==np.int64
    rows=np.concatenate([target,donor],axis=0);validate_labels(rows,(1800,81));return rows


def verify(p,public=False,sources_only=False):
    for rel,h in (p['source_sha256'] if sources_only else p['source_sha256']|p['input_sha256']).items():
        if public and rel.startswith('artifacts/') and not (ROOT/rel).exists():continue
        assert sha(ROOT/rel)==h,rel


def freeze():
    assert not PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=read(HERE/'pooled-donor-preflight.json');seeds=read(HERE/'pooled-donor-seed-audit.json')
    assert gate['status']==seeds['status']=='passed' and not gate['study_pooled_costs_or_fits_read']
    assert gate['proposal_sha256']==sha(HERE/'pooled-donor-proposal.json') and gate['seed_audit_sha256']==sha(HERE/'pooled-donor-seed-audit.json')
    for rel,h in gate['source_sha256'].items():assert sha(ROOT/rel)==h
    for rel,h in seeds['frozen_protocol_sha256'].items():assert sha(ROOT/rel)==h
    old=read(PARENT_PROTOCOL);verify_parent(old);check_parent_seal(PARENT)
    inputs=old['input_sha256'].copy()
    for path in [PARENT_PROTOCOL,HERE/'pooled-donor-proposal.json',HERE/'pooled-donor-preflight.json',HERE/'pooled-donor-seed-audit.json',*sorted(PARENT.rglob('*.npz')),*sorted(PARENT.rglob('*.json'))]:inputs[str(path.relative_to(ROOT))]=sha(path)
    design=read(HERE/'pooled-donor-proposal.json');p=dict(design,status='frozen_before_pooled1800_costs_or_fits',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        jobs=old['jobs'],solver=dict(old['solver'],random_seed_base=design['solver_seed_base']),source_sha256=gate['source_sha256'],input_sha256=inputs)
    verify(p);save(PROTOCOL,p);print('FROZEN',sha(PROTOCOL),flush=True)


def phase_files():return {f'b{i:03d}/pooled.{ext}' for i in range(256) for ext in ['npz','json']}


def check_seal(folder):
    seal=read(folder/'fit-seal.json');assert seal['status']=='all_fits_sealed' and seal['protocol_sha256']==sha(PROTOCOL)
    assert set(seal['files'])==phase_files();latest=0
    for rel,h in seal['files'].items():assert sha(folder/rel)==h,rel
    for i in range(256):
        rec=read(folder/f'b{i:03d}/pooled.json')
        assert rec['index']==i and rec['arm']=='pooled' and rec['protocol_sha256']==sha(PROTOCOL)
        assert rec['models_sha256']==sha(folder/f'b{i:03d}/pooled.npz')
        assert rec['fit_started_monotonic_ns']<=rec['fit_completed_monotonic_ns'];latest=max(latest,rec['fit_completed_monotonic_ns'])
    assert latest==seal['latest_fit_monotonic_ns']<seal['sealed_monotonic_ns'];return seal


def seal_phase():
    assert not (OUT/'fit-seal.json').exists() and not list(OUT.glob('b*/audit.json'))
    latest=max(read(OUT/rel)['fit_completed_monotonic_ns'] for rel in phase_files() if rel.endswith('.json'))
    save(OUT/'fit-seal.json',dict(status='all_fits_sealed',protocol_sha256=sha(PROTOCOL),latest_fit_monotonic_ns=latest,sealed_monotonic_ns=time.monotonic_ns(),files={rel:sha(OUT/rel) for rel in sorted(phase_files())}))
    check_seal(OUT)


def fit_worker(worker_id):
    from degree_prior_solver import solve
    from degree_prior_math import attempt_certificate
    p=read(PROTOCOL);verify(p,sources_only=True);ii,jj=np.triu_indices(81,1)
    for index in range(worker_id,256,8):
        folder=OUT/f'b{index:03d}';folder.mkdir();started=time.monotonic_ns();rows=observe(index);cost=counts(rows);degree=p['jobs'][index]['inferred_degree']
        log=folder/'pooled.log';x,y,meta=solve(81,degree,ii,jj,cost,p['solver_seed_base']-2026111502+index,log)
        if meta['solver_optimal'] and meta['value_valid'] and meta['dual_valid']:cert,proof=attempt_certificate(81,degree,ii,jj,cost,x,y)
        else:cert=dict(certified=False,reason='solver_status_or_validity');proof={k:np.empty(0,np.int64) for k in ['primal_twice','node_dual_twice','upper_dual_twice']}
        np.savez_compressed(folder/'pooled.npz',cost=cost,source_rows=np.arange(1800),primal_values=x,node_dual_values=y,**proof)
        save(folder/'pooled.json',dict(index=index,arm='pooled',degree=degree,protocol_sha256=sha(PROTOCOL),models_sha256=sha(folder/'pooled.npz'),log_sha256=sha(log),solver=meta,certificate=cert,
            fit_started_monotonic_ns=started,fit_completed_monotonic_ns=time.monotonic_ns()))
        save(folder/'progress.json',dict(completed_LPs=1,pid=os.getpid()))


def audit_worker(worker_id):
    from audit_pooled_donor import audit_bank
    p=read(PROTOCOL);verify(p,sources_only=True);seal=check_seal(OUT)
    for index in range(worker_id,256,8):
        started=time.monotonic_ns();assert started>seal['sealed_monotonic_ns'];audit=audit_bank(index,p)
        save(OUT/f'b{index:03d}/audit.json',audit);save(OUT/f'b{index:03d}/audit-order.json',dict(audit_started_monotonic_ns=started,seal_sha256=sha(OUT/'fit-seal.json')))


def export():
    from audit_pooled_donor import aggregate
    PUBLIC.mkdir();rows=[];files={}
    for i in range(256):
        folder=OUT/f'b{i:03d}';audit=read(folder/'audit.json');assert audit['protocol_sha256']==sha(PROTOCOL) and audit['auditor_sha256']==sha(HERE/'audit_pooled_donor.py')
        for name,h in audit['artifact_sha256'].items():assert sha(folder/name)==h
        rows.extend(audit['rows']);dest=PUBLIC/folder.name;dest.mkdir()
        for name in list(audit['artifact_sha256'])+['audit.json','audit-order.json']:shutil.copyfile(folder/name,dest/name);files[str((dest/name).relative_to(PUBLIC))]=sha(dest/name)
    shutil.copyfile(OUT/'fit-seal.json',PUBLIC/'fit-seal.json');files['fit-seal.json']=sha(PUBLIC/'fit-seal.json')
    save(PUBLIC/'summary.json',dict(status='validated',protocol_sha256=sha(PROTOCOL),rows=rows,**aggregate(rows),artifact_sha256=files,new_LPs=256,
        inherited_certificates_checked=1536,inherited_mutations_rejected=4608,new_source_draws=0,new_labels=0,
        scope='One pooled1800-row fit on exactly the H4ba observation union, with unchanged inherited target-degree; examined-data control.'))


def launch():
    p=read(PROTOCOL);assert not OUT.exists();OUT.mkdir(parents=True);children=[];logs=[];samples=[];status='incomplete';error=None;ticks=os.sysconf('SC_CLK_TCK')
    def cpu():
        total=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])
        for c in children:
            if c.poll() is None:
                try:f=Path(f'/proc/{c.pid}/stat').read_text().split();total+=(int(f[13])+int(f[14]))/ticks
                except FileNotFoundError:pass
        return total
    def monitor(phase):
        row=dict(phase=phase,wall_seconds=time.monotonic()-WALL_ORIGIN,aggregate_cpu_seconds=cpu(),active_pids=[c.pid for c in children if c.poll() is None],
            completed_LPs=sum(read(q)['completed_LPs'] for q in OUT.glob('b*/progress.json')),audited_banks=len(list(OUT.glob('b*/audit.json'))))
        samples.append(row);save(OUT/'progress.json',row)
        assert row['wall_seconds']<p['budget']['wall_seconds'] and row['aggregate_cpu_seconds']<p['budget']['aggregate_cpu_seconds'],'frozen_resource_ceiling'
    def phase(action):
        for i in range(8):
            stream=(OUT/f'{action}{i}.log').open('x');logs.append(stream)
            children.append(subprocess.Popen([str(PYTHON),str(Path(__file__)),action,'--worker',str(i)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children):monitor(action);assert all(c.poll() in [None,0] for c in children),'worker_failure';time.sleep(.2)
        assert all(c.returncode==0 for c in children);monitor(action)
    try:
        verify(p);check_parent_seal(PARENT);monitor('hashes');phase('fit_worker');seal_phase();monitor('all_fits_sealed');phase('audit_worker');export();monitor('export');status='complete'
    except BaseException as exc:error=repr(exc);raise
    finally:
        for c in children:
            if c.poll() is None:
                os.killpg(c.pid,signal.SIGTERM)
                try:c.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
        for stream in logs:stream.close()
        runtime=dict(status=status,error=error,protocol_sha256=sha(PROTOCOL),wall_seconds=time.monotonic()-WALL_ORIGIN,aggregate_cpu_seconds=cpu(),returncodes=[c.returncode for c in children],samples=samples)
        save(OUT/'runtime.json',runtime)
        if PUBLIC.exists():save(PUBLIC/'runtime.json',runtime)
        print(json.dumps({k:v for k,v in runtime.items() if k!='samples'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','fit_worker','audit_worker','launch']);parser.add_argument('--worker',type=int);args=parser.parse_args()
    globals()[args.action](args.worker) if args.action.endswith('_worker') else globals()[args.action]()
