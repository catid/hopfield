"""H4az bounded audit-only repair; original H4ay fits and failure stay immutable."""
from pathlib import Path
import argparse,datetime,json,os,resource,shutil,signal,subprocess,time
WALL_ORIGIN=time.monotonic()
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from disjoint_donor import ROOT,HERE,PYTHON,PROTOCOL,OUT as ORIGINAL,sha,read,save,verify,check_seal
OUT=ROOT/'artifacts/research/20261114/foureyes/disjoint_donor_repair'
PUBLIC=ROOT/'results/research/foureyes/disjoint_donor_repair'
REPAIR_PROTOCOL=HERE/'disjoint-donor-repair.json'
EVIDENCE=ROOT/'artifacts/research/20261114/foureyes/disjoint_donor_audit_repair'


def verify_repair(public=False):
    p=read(REPAIR_PROTOCOL)
    assert p['new_LPs']==p['new_data']==0 and p['original_protocol_sha256']==sha(PROTOCOL)
    for rel,h in (p['source_sha256']|p['input_sha256']).items():
        if public and rel.startswith('artifacts/') and not (ROOT/rel).exists():continue
        assert sha(ROOT/rel)==h,rel
    return p


def verify_original_failure(base):
    p=read(REPAIR_PROTOCOL);origin=base/'original';rt=read(origin/'runtime.json')
    assert rt['status']=='incomplete' and rt['protocol_sha256']==sha(PROTOCOL)
    assert rt['returncodes']==[0]*16+[1]*8 and rt['error']=='AssertionError()'
    for rel,h in p['historical_evidence_sha256'].items():assert sha(origin/rel)==h,rel
    for rel,h in p['sealed_file_sha256'].items():assert sha(base/rel)==h,rel
    for i in range(8):
        error=(origin/f'audit_worker{i}.log').read_text()
        assert 'donor.dtype==np.int64' in error and 'AssertionError' in error
    assert all(s['audited_banks']==0 for s in rt['samples'])
    assert rt['samples'][-1]['completed_LPs']==512
    check_seal(base)


def freeze():
    assert not REPAIR_PROTOCOL.exists() and not OUT.exists() and not PUBLIC.exists()
    gate=read(HERE/'disjoint-donor-repair-preflight.json');assert gate['status']=='passed' and not gate['study_graph_outcomes_read']
    assert gate['proposal_sha256']==sha(HERE/'disjoint-donor-repair-proposal.json')
    for rel,h in gate['source_sha256'].items():assert sha(ROOT/rel)==h
    verify(read(PROTOCOL));check_seal(ORIGINAL)
    assert not list(ORIGINAL.glob('b*/audit.json')) and not list(ORIGINAL.glob('b*/source-provenance.json'))
    historical={f'audit_worker{i}.log':sha(ORIGINAL/f'audit_worker{i}.log') for i in range(8)}
    historical['runtime.json']=sha(ORIGINAL/'runtime.json')
    for name in ['source-format-headers.json','donor-format-headers.json']:historical[name]=sha(EVIDENCE/name)
    sealed=dict(read(ORIGINAL/'fit-seal.json')['files'])
    for name in ['donor-seal.json','fit-seal.json']:sealed[name]=sha(ORIGINAL/name)
    inputs={str(PROTOCOL.relative_to(ROOT)):sha(PROTOCOL)}
    for name in ['disjoint-donor-repair-proposal.json','disjoint-donor-repair-preflight.json']:
        path=HERE/name;inputs[str(path.relative_to(ROOT))]=sha(path)
    for path in sorted(ORIGINAL.rglob('*')):
        if path.is_file():inputs[str(path.relative_to(ROOT))]=sha(path)
    for path in sorted(EVIDENCE.glob('*')):
        if path.is_file():inputs[str(path.relative_to(ROOT))]=sha(path)
    proposal=read(HERE/'disjoint-donor-repair-proposal.json')
    p=dict(proposal,status='frozen_before_repaired_graph_outcome_audit',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           original_protocol_sha256=sha(PROTOCOL),source_sha256=gate['source_sha256'],input_sha256=inputs,
           historical_evidence_sha256=historical,sealed_file_sha256=sealed)
    save(REPAIR_PROTOCOL,p);verify_repair();print('FROZEN_REPAIR',sha(REPAIR_PROTOCOL),flush=True)


def worker(worker_id):
    from audit_disjoint_donor_repair import audit_bank
    rp=read(REPAIR_PROTOCOL)
    for rel,h in rp['source_sha256'].items():assert sha(ROOT/rel)==h
    p=read(PROTOCOL);seal=check_seal(OUT)
    for index in range(worker_id,256,8):
        started=time.monotonic_ns();assert started>seal['sealed_monotonic_ns']
        audited=audit_bank(index,p);save(OUT/f'b{index:03d}/audit.json',audited)
        save(OUT/f'b{index:03d}/audit-order.json',dict(audit_started_monotonic_ns=started,seal_sha256=sha(OUT/'fit-seal.json'),repair_protocol_sha256=sha(REPAIR_PROTOCOL)))


def copy_original():
    rp=read(REPAIR_PROTOCOL);OUT.mkdir(parents=True)
    for rel,h in rp['sealed_file_sha256'].items():
        dest=OUT/rel;dest.parent.mkdir(exist_ok=True);shutil.copyfile(ORIGINAL/rel,dest);assert sha(dest)==h
    for i in range(256):
        for arm in ['donor','final']:shutil.copyfile(ORIGINAL/f'b{i:03d}/{arm}.log',OUT/f'b{i:03d}/{arm}.log')
    origin=OUT/'original';origin.mkdir()
    for rel,h in rp['historical_evidence_sha256'].items():
        source=(EVIDENCE if rel.endswith('headers.json') else ORIGINAL)/rel
        shutil.copyfile(source,origin/rel);assert sha(origin/rel)==h
    verify_original_failure(OUT)


def export():
    from audit_disjoint_donor_repair import aggregate
    assert not PUBLIC.exists();PUBLIC.mkdir();rows=[];files={}
    for i in range(256):
        folder=OUT/f'b{i:03d}';audit=read(folder/'audit.json')
        assert audit['protocol_sha256']==sha(PROTOCOL) and audit['repair_protocol_sha256']==sha(REPAIR_PROTOCOL)
        assert audit['auditor_sha256']==sha(HERE/'audit_disjoint_donor_repair.py')
        for name,h in audit['artifact_sha256'].items():assert sha(folder/name)==h
        rows.extend(audit['rows']);dest=PUBLIC/folder.name;dest.mkdir()
        for name in list(audit['artifact_sha256'])+['audit.json','audit-order.json']:
            shutil.copyfile(folder/name,dest/name);files[str((dest/name).relative_to(PUBLIC))]=sha(dest/name)
    for name in ['donor-seal.json','fit-seal.json']:
        shutil.copyfile(OUT/name,PUBLIC/name);files[name]=sha(PUBLIC/name)
    shutil.copytree(OUT/'original',PUBLIC/'original')
    for path in sorted((PUBLIC/'original').iterdir()):files[str(path.relative_to(PUBLIC))]=sha(path)
    save(PUBLIC/'summary.json',dict(status='validated_repaired_postrun_evaluation',protocol_sha256=sha(PROTOCOL),repair_protocol_sha256=sha(REPAIR_PROTOCOL),rows=rows,**aggregate(rows),
        artifact_sha256=files,new_LPs=0,inherited_sealed_LPs=sum(r['attempted_LPs'] for r in rows),new_source_draws=0,new_labels=0,
        inherited_certificates_checked=1024,inherited_mutations_rejected=3072,
        scope='Verification-only repair of immutable H4ay fits; original run remains technically incomplete. Original256-bank endpoints retained. Raw source replay was completed after the original seal under this separate repair budget.'))


def launch():
    assert not OUT.exists() and not PUBLIC.exists();rp=read(REPAIR_PROTOCOL);children=[];streams=[];samples=[];status='incomplete';error=None
    ticks=os.sysconf('SC_CLK_TCK')
    def cpu():
        total=sum(r.ru_utime+r.ru_stime for r in [resource.getrusage(resource.RUSAGE_SELF),resource.getrusage(resource.RUSAGE_CHILDREN)])
        for child in children:
            if child.poll() is None:
                try:
                    f=Path(f'/proc/{child.pid}/stat').read_text().split();total+=(int(f[13])+int(f[14]))/ticks
                except FileNotFoundError:pass
        return total
    def monitor(phase):
        row=dict(phase=phase,wall_seconds=time.monotonic()-WALL_ORIGIN,aggregate_cpu_seconds=cpu(),active_pids=[c.pid for c in children if c.poll() is None],
                 new_LPs=0,audited_banks=len(list(OUT.glob('b*/audit.json'))))
        samples.append(row)
        if OUT.exists():save(OUT/'progress.json',row)
        assert row['wall_seconds']<rp['budget']['wall_seconds'] and row['aggregate_cpu_seconds']<rp['budget']['aggregate_cpu_seconds'],'frozen_repair_resource_ceiling'
    try:
        verify_repair();verify(read(PROTOCOL));check_seal(ORIGINAL);monitor('hashes');copy_original();monitor('copied_and_seals_verified')
        for i in range(8):
            stream=(OUT/f'worker{i}.log').open('x');streams.append(stream)
            children.append(subprocess.Popen([str(PYTHON),str(Path(__file__)),'worker','--worker',str(i)],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True))
        while any(c.poll() is None for c in children):
            monitor('audit');assert all(c.poll() in [None,0] for c in children),'repair_worker_failure';time.sleep(.2)
        assert all(c.returncode==0 for c in children);monitor('audited');verify_original_failure(OUT);export();monitor('export');status='complete'
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in streams:stream.close()
        runtime=dict(status=status,error=error,repair_protocol_sha256=sha(REPAIR_PROTOCOL),original_protocol_sha256=sha(PROTOCOL),new_LPs=0,
                     wall_seconds=time.monotonic()-WALL_ORIGIN,aggregate_cpu_seconds=cpu(),returncodes=[c.returncode for c in children],samples=samples)
        if OUT.exists():save(OUT/'runtime.json',runtime)
        if PUBLIC.exists():save(PUBLIC/'runtime.json',runtime)
        print(json.dumps({k:v for k,v in runtime.items() if k!='samples'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','launch','worker']);parser.add_argument('--worker',type=int);args=parser.parse_args()
    worker(args.worker) if args.action=='worker' else globals()[args.action]()
