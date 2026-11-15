"""H4bb synthetic union, phase-ordering, endpoint, seed and certificate gates."""
from pathlib import Path
import tempfile,time
import numpy as np
import pooled_donor as S
from larger_donor_math import counts,independent_counts,validate_labels
from audit_pooled_donor import aggregate,audit_fit
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def rejected(fn,*args):
    try:fn(*args)
    except (AssertionError,FileNotFoundError):return
    raise AssertionError('Invalid synthetic case accepted')


def seed_audit(design):
    values,files=set(),{}
    def scan(v,context=False):
        if isinstance(v,dict):
            for k,x in v.items():scan(x,context or 'seed' in k.lower())
        elif isinstance(v,list):
            for x in v:scan(x,context)
        elif context and type(v) is int:values.add(v)
    for path in sorted((S.ROOT/'research').rglob('*.json')):
        try:p=S.read(path)
        except (ValueError,UnicodeDecodeError):continue
        if isinstance(p,dict) and p.get('frozen_utc'):
            files[str(path.relative_to(S.ROOT))]=S.sha(path);scan(p)
    expanded=set();ranges={}
    for name,count in [('corruption-censor',256),('partial-mask-recall',1280),('conflict-mask',512),('purified-detector',512),('crossfit-conflicts',768),('disjoint-donor',512),('larger-donor',512)]:
        p=S.read(S.HERE/f'{name}.json');base=p['solver']['random_seed_base']
        expanded.update(range(base,base+count));ranges[name+'-solver']=[base,base+count-1]
        if 'random_seed_base' in p:
            base=p['random_seed_base'];expanded.update(range(base,base+256));ranges[name+'-mask']=[base,base+255]
    thinning=S.read(S.HERE/'partial-mask-recall.json')['thinning']['seed_base']
    thinning_sequences=[[thinning,i] for i in range(256)]
    solver=set(range(design['solver_seed_base'],design['solver_seed_base']+256))
    assert len(solver)==256 and not solver&(values|expanded) and 0<=min(solver)<max(solver)<2**31
    S.save(S.HERE/'pooled-donor-seed-audit.json',dict(status='passed',frozen_protocol_sha256=files,
        historical_explicit_seed_count=len(values),expanded_ancestor_ranges=ranges,H4au_actual_thinning_seed_sequences=thinning_sequences,
        solver_range=[min(solver),max(solver)],collisions=[],
        scope='Explicit frozen seed fields and expanded H4at/H4au/H4av/H4aw/H4ax/H4ay/H4ba solver and declared random-mask seeds. Other historical formulas not exhaustive; no proof of PRNG independence.'))


def endpoints(changes):
    rows=[]
    for i in range(256):
        kind=changes.get(i,'success');rows.append(dict(index=i,parent_exact=i<229,donor_exact=i<229,classification=kind,certified=kind!='unresolved',family=S.FAMILIES[i%4]))
    return aggregate(rows)['primary']


def lifecycle():
    oldout,oldproto=S.OUT,S.PROTOCOL
    with tempfile.TemporaryDirectory(prefix='h4bb-seal-') as tmp:
        try:
            S.OUT=Path(tmp)/'out';S.OUT.mkdir();S.PROTOCOL=Path(tmp)/'protocol.json';S.save(S.PROTOCOL,dict(source_sha256={}))
            rejected(S.audit_worker,0);rejected(S.seal_phase);ph=S.sha(S.PROTOCOL)
            for i in range(256):
                folder=S.OUT/f'b{i:03d}';folder.mkdir();np.savez(folder/'pooled.npz',fixture=np.zeros(1))
                S.save(folder/'pooled.json',dict(index=i,arm='pooled',protocol_sha256=ph,models_sha256=S.sha(folder/'pooled.npz'),fit_started_monotonic_ns=time.monotonic_ns(),fit_completed_monotonic_ns=time.monotonic_ns()))
            S.seal_phase();S.check_seal(S.OUT);rejected(S.seal_phase)
            path=S.OUT/'b000/pooled.npz';original=path.read_bytes();path.write_bytes(original+b'bad');rejected(S.check_seal,S.OUT);path.write_bytes(original)
            path=S.OUT/'fit-seal.json';original=path.read_text();rec=S.read(path);rec['sealed_monotonic_ns']=0;S.save(path,rec);rejected(S.check_seal,S.OUT);path.write_text(original)
        finally:S.OUT,S.PROTOCOL=oldout,oldproto
    return dict(missing_fit_or_seal_rejected=True,early_audit_rejected=True,repeat_seal_rejected=True,mutated_fit_rejected=True,seal_order_tamper_rejected=True)


def union():
    oldobs,oldparent=S.OBS,S.PARENT;rng=np.random.default_rng(982110)
    target=rng.integers(0,9,(600,81),dtype=np.uint8);donor=rng.integers(0,9,(1200,81),dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix='h4bb-union-') as tmp:
        try:
            S.OBS=Path(tmp)/'target';S.PARENT=Path(tmp)/'donor'
            (S.OBS/'b000').mkdir(parents=True);(S.PARENT/'b000').mkdir(parents=True)
            # Unrelated ground truth is deliberately absent from both synthetic inputs.
            np.savez(S.OBS/'b000/models.npz',noisy=target)
            dest=S.PARENT/'b000/donor.npz';np.savez(dest,noisy=donor,source_rows=np.arange(600,1800))
            joined=S.observe(0);expected=np.array([target[r] if r<600 else donor[r-600] for r in range(1800)])
            assert np.array_equal(joined,expected) and joined.dtype==np.uint8
            assert np.array_equal(counts(joined),independent_counts(expected))
            assert np.array_equal(counts(joined),counts(target)+counts(donor))
            for row in [0,599,600,1799]:assert np.array_equal(joined[row],expected[row])
            np.savez(dest,noisy=donor,source_rows=np.arange(1200));rejected(S.observe,0)
            np.savez(dest,noisy=donor[:600],source_rows=np.arange(600,1800));rejected(S.observe,0)
            wrong=donor.copy();wrong[-1,-1]=9;np.savez(dest,noisy=wrong,source_rows=np.arange(600,1800));rejected(S.observe,0)
        finally:S.OBS,S.PARENT=oldobs,oldparent
    return dict(exact1800_rows=True,first_last_and_boundary_identity=True,counts_additive_and_literal_parity=True,source_identity_corruptions_rejected=3,truth_or_masks_unavailable_to_producer=True)


def main():
    path=S.HERE/'pooled-donor-preflight.json';assert not path.exists() and not S.PROTOCOL.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    start=time.monotonic();seed_audit(S.read(S.HERE/'pooled-donor-proposal.json'));u=union();life=lifecycle()
    bad,unk='resolved_failure_nonexact','unresolved';cases=[]
    for changes,want in [({},('supported','supported')),({i:bad for i in range(245,256)},('supported','supported')),
        ({i:bad for i in range(244,256)},('falsified','supported')),({**{i:bad for i in range(245,256)},244:unk},('inconclusive','supported')),
        ({i:bad for i in range(4)},('supported','supported')),({i:bad for i in range(5)},('supported','falsified')),
        ({**{i:bad for i in range(4)},4:unk},('supported','inconclusive')),({i:unk for i in range(256)},('inconclusive','inconclusive'))]:
        result=endpoints(changes);assert tuple(v['decision'] for v in result.values())==want;cases.append(result)
    folder=S.ROOT/'artifacts/research/20261114/foureyes/pooled_donor_preflight';folder.mkdir(parents=True,exist_ok=True)
    ii,jj=np.triu_indices(4,1);cost=np.ones(6,np.int64);x,y,meta=solve(4,1,ii,jj,cost,982111-2026111502,folder/'toy.log');cert,w=attempt_certificate(4,1,ii,jj,cost,x,y)
    proof=[w[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']];assert cert['certified'] and check_certificate(4,1,ii,jj,cost,*proof)
    for k in range(3):
        badproof=[v.copy() for v in proof];badproof[k][0]+=1;assert not check_certificate(4,1,ii,jj,cost,*badproof)
    inherited=S.read(S.HERE/'larger-donor-preflight.json');assert inherited['status']=='passed'
    for rel,h in inherited['source_sha256'].items():assert S.sha(S.ROOT/rel)==h
    sources=inherited['source_sha256'].copy()
    for name in ['pooled_donor.py','audit_pooled_donor.py','preflight_pooled_donor.py']:
        p=S.HERE/name;sources[str(p.relative_to(S.ROOT))]=S.sha(p)
    S.save(path,dict(status='passed',study_pooled_costs_or_fits_read=False,union=u,lifecycle=life,endpoint_cases=cases,certificate_mutations_rejected=3,
        proposal_sha256=S.sha(S.HERE/'pooled-donor-proposal.json'),seed_audit_sha256=S.sha(S.HERE/'pooled-donor-seed-audit.json'),
        inherited_full2400_noise_and_compact_gate_sha256=S.sha(S.HERE/'larger-donor-preflight.json'),source_sha256=sources,gate_wall_seconds=time.monotonic()-start))
    print('POOLED_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':main()
