"""H4aw synthetic mask, seal, exact-certificate and endpoint tests only."""
from pathlib import Path
import tempfile
import time
import numpy as np
import purified_detector as S
from purified_detector_math import masks, independent_masks
from corruption_censor_math import score, independent_score, literal_score
from audit_purified_detector import aggregate
from degree_prior_solver import solve
from degree_prior_math import attempt_certificate
from audit_degree_prior import check_certificate


def rejected(fn, *args):
    try:fn(*args)
    except (AssertionError, FileNotFoundError):return
    raise AssertionError('Invalid input/lifecycle accepted')


def seeds(design):
    values, files = set(), {}
    def scan(v, context=False):
        if isinstance(v, dict):
            for key, item in v.items():
                scan(item, context or 'seed' in key.lower())
        elif isinstance(v, list):
            for item in v:
                scan(item, context)
        elif context and type(v) is int:
            values.add(v)
    for path in sorted((S.ROOT / 'research').rglob('*.json')):
        try:
            data = S.read(path)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and data.get('frozen_utc'):
            files[str(path.relative_to(S.ROOT))] = S.sha(path)
            scan(data)
    expanded, ranges = set(), {}
    for filename, count in [('corruption-censor.json', 256), ('partial-mask-recall.json', 1280), ('conflict-mask.json', 512)]:
        p = S.read(S.HERE / filename)
        base = p['solver']['random_seed_base']
        expanded.update(range(base, base + count))
        ranges[filename] = [base, base + count - 1]
        files[str((S.HERE / filename).relative_to(S.ROOT))] = S.sha(S.HERE / filename)
    av=S.read(S.HERE/'conflict-mask.json')
    expanded.update(range(av['random_seed_base'],av['random_seed_base']+256))
    ranges['conflict-mask-random']=[av['random_seed_base'],av['random_seed_base']+255]
    random = set(range(design['random_seed_base'], design['random_seed_base'] + 256))
    solver = set(range(design['solver_seed_base'], design['solver_seed_base'] + 512))
    assert len(random | solver) == 768 and not (random | solver) & (values | expanded)
    assert min(random | solver) >= 0 and max(random | solver) < 2**31
    S.save(S.HERE / 'purified-detector-seed-audit.json', dict(status='passed',
        frozen_protocol_sha256=files, historical_explicit_seed_values=len(values),
        expanded_ancestor_ranges=ranges, random_range=[min(random), max(random)],
        solver_range=[min(solver), max(solver)], collisions=[],
        scope='Explicit integer seed fields in frozen_utc protocols plus all actual H4at/H4au/H4av solver and H4av mask seeds. Other historical seed formulas not exhaustively expanded; no proof of stream independence.'))


def endpoint_fixture(p, pu, r, ru):
    rows=[]
    for index in range(256):
        for arm,exact,unknown in [('purified',p,pu),('matched_random',r,ru)]:
            kind='success' if index<exact else 'unresolved' if index<exact+unknown else 'resolved_failure_nonexact'
            rows.append(dict(index=index,arm=arm,classification=kind,certified=kind!='unresolved',
                baseline_exact=index<129,parent_detector_exact=index<130,family=S.FAMILIES[index%4],
                true_positive_locations=600,false_positive_locations=0,false_negative_locations=4800))
    return aggregate(rows)


def lifecycle():
    old_out,old_protocol=S.OUT,S.PROTOCOL
    with tempfile.TemporaryDirectory(prefix='h4aw-synthetic-seal-') as tmp:
        try:
            S.OUT=Path(tmp)/'data';S.OUT.mkdir();S.PROTOCOL=Path(tmp)/'protocol.json'
            S.save(S.PROTOCOL,dict(source_sha256={}))
            rejected(S.seal_masks,{})
            rejected(S.fit_worker,0)
            rejected(S.audit_worker,0)
            ph=S.sha(S.PROTOCOL)
            for index in range(256):
                folder=S.OUT/f'b{index:03d}';folder.mkdir()
                np.savez(folder/'prepared-masks.npz',**{a:np.zeros((1,18),bool) for a in S.ARMS})
                S.save(folder/'prepared.json',dict(index=index,protocol_sha256=ph,
                    models_sha256=S.sha(folder/'prepared-masks.npz'),prepared_monotonic_ns=time.monotonic_ns()))
            S.seal_masks({});maskseal=S.check_mask_seal(S.OUT)
            assert maskseal['masks']==512 and len(maskseal['files'])==512
            rejected(S.seal_masks,{})
            rejected(S.seal_fits,{})
            target=S.OUT/'b000/prepared-masks.npz';original=target.read_bytes()
            target.write_bytes(original+b'bad');rejected(S.check_mask_seal,S.OUT);target.write_bytes(original)
            S.check_mask_seal(S.OUT)
            for index in range(256):
                folder=S.OUT/f'b{index:03d}'
                for arm in S.ARMS:
                    np.savez(folder/f'{arm}.npz',fixture=np.zeros(1))
                    S.save(folder/f'{arm}.json',dict(index=index,arm=arm,protocol_sha256=ph,
                        models_sha256=S.sha(folder/f'{arm}.npz'),mask_seal_sha256=S.sha(S.OUT/'mask-seal.json'),
                        fit_started_monotonic_ns=time.monotonic_ns(),fit_completed_monotonic_ns=time.monotonic_ns()))
            file=S.OUT/'b000/purified.json';rec=S.read(file)
            S.save(file,dict(rec,fit_started_monotonic_ns=0));rejected(S.seal_fits,{})
            S.save(file,rec);S.seal_fits({});seal=S.check_seal(S.OUT)
            assert seal['fits']==512 and len(seal['files'])==1536
            rejected(S.seal_fits,{})
            target=S.OUT/'b000/purified.npz';target.write_bytes(target.read_bytes()+b'bad')
            rejected(S.check_seal,S.OUT)
        finally:S.OUT,S.PROTOCOL=old_out,old_protocol
    return dict(missing_masks=True,early_fits=True,early_graph_audit=True,
        repeated_masks=True,missing_fits=True,mutated_masks=True,
        fit_before_mask_seal=True,repeated_fit_seal=True,mutated_fit=True,
        complete_mask_and_fit_seals=True)


def main():
    assert not S.PROTOCOL.exists() and not S.OUT.exists() and not S.PUBLIC.exists()
    target=S.HERE/'purified-detector-preflight.json';assert not target.exists()
    started=time.perf_counter();seeds(S.read(S.HERE/'purified-detector-proposal.json'))
    true=np.zeros((20,18),bool);true[:,:9]=True
    detected=np.zeros_like(true)
    for row in range(20):
        k=row%10;detected[row,:k]=True;detected[row,9:18-k]=True
    rng=np.random.default_rng(982070);noisy=rng.integers(0,9,true.shape,dtype=np.int8)
    a,b=masks(detected,true,982071);aa,bb=independent_masks(detected,true,982071)
    assert np.array_equal(a,aa) and np.array_equal(b,bb) and np.array_equal(a,detected&true)
    assert np.array_equal(a.sum(1),np.arange(20)%10) and np.array_equal(a.sum(1),b.sum(1))
    assert not (a&~true).any() and not (b&~true).any()
    assert not a[0].any() and not b[0].any() and np.array_equal(a[9],true[9]) and np.array_equal(b[9],true[9])
    permutation=rng.permutation(9)
    for mask in [a,b,np.zeros_like(true),np.ones_like(true)]:
        expected=literal_score(noisy,mask)
        for key,value in score(noisy,mask).items():
            assert np.array_equal(value,expected[key]) and np.array_equal(value,independent_score(noisy,mask)[key])
            assert np.array_equal(value,score(permutation[noisy],mask)[key])
        if mask.all():assert not expected['cost'].any() and not expected['exposure'].any()
    for args in [(detected.astype(int),true,1),(detected,true.astype(int),1),
                 (detected[:-1],true,1),(np.zeros_like(detected),true,1),
                 (detected,np.zeros_like(true),1),(detected,true,-1),(detected,true,1.)]:
        rejected(masks,*args)
    folder=S.ROOT/'artifacts/research/20261114/foureyes/purified_detector_preflight';folder.mkdir(parents=True,exist_ok=True)
    ii,jj=np.triu_indices(4,1);cost=np.ones(6,np.int64)
    x,y,meta=solve(4,1,ii,jj,cost,982072-2026111502,folder/'positive.log')
    cert,proof=attempt_certificate(4,1,ii,jj,cost,x,y)
    w=[proof[k] for k in ['primal_twice','node_dual_twice','upper_dual_twice']]
    assert cert['certified'] and meta['solver_optimal'] and check_certificate(4,1,ii,jj,cost,*w)
    for k in range(3):
        bad=[v.copy() for v in w];bad[k][0]+=1
        assert not check_certificate(4,1,ii,jj,cost,*bad)
    endpoints=[]
    for p,pu,r,ru,absolute,location in [
        (200,0,184,0,'supported','supported'),(200,0,183,0,'supported','falsified'),
        (199,1,184,0,'inconclusive','supported'),(199,0,199,0,'falsified','supported'),
        (200,0,216,0,'supported','supported'),(200,0,217,0,'supported','falsified'),
        (200,1,184,0,'supported','inconclusive'),(200,0,216,1,'supported','inconclusive'),
        (0,256,0,256,'inconclusive','inconclusive')]:
        primary=endpoint_fixture(p,pu,r,ru)['primary']
        assert primary['purified_recovery']['decision']==absolute
        assert primary['location_sensitivity']['decision']==location
        endpoints.append(primary)
    life=lifecycle();true=np.zeros((600,81),bool);detected=np.zeros_like(true)
    for row in range(600):
        true[row,rng.choice(81,9,replace=False)]=True
        detected[row,rng.choice(81,9,replace=False)]=True
    timings={}
    expected=masks(detected,true,982073)
    for fn in [masks,independent_masks]:
        samples=[]
        for _ in range(5):
            before=time.perf_counter();out=fn(detected,true,982073);samples.append(time.perf_counter()-before)
            assert all(np.array_equal(x,y) for x,y in zip(out,expected))
        timings[fn.__name__]=float(np.median(samples))
    sources=S.read(S.PARENT_PROTOCOL)['source_sha256'].copy()
    for name in ['purified_detector.py','purified_detector_math.py','audit_purified_detector.py','preflight_purified_detector.py']:
        path=S.HERE/name;sources[str(path.relative_to(S.ROOT))]=S.sha(path)
    S.save(target,dict(status='passed',study_masks_scores_or_LPs_read=False,
        all_k0_to9_and_fixed_choice_consumption=True,independent_masks_and_scores=True,
        no_false_positives_and_rowwise_matched_recall=True,digit_score_invariance=True,
        invalid_inputs_rejected=7,certificate_mutations_rejected=3,
        positive_objective_certificate=cert,primary_boundaries=endpoints,lifecycle=life,
        mask_timings_seconds=timings,benchmark_scope='Unrelated600x81 masks only; no complete-study speed claim.',
        source_sha256=sources,seed_audit_sha256=S.sha(S.HERE/'purified-detector-seed-audit.json'),
        gate_wall_seconds=time.perf_counter()-started))
    print('PREFLIGHT_PASSED',timings,flush=True)


if __name__=='__main__':main()
